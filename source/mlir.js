
// Experimental
// contributor @tucan9389

const mlir = {};

mlir.ModelFactory = class {

    async match(context) {
        const stream = context.stream;
        if (stream && stream.length > 4) {
            const buffer = stream.peek(4);
            const signature = String.fromCharCode.apply(null, buffer);
            if (signature === 'ML\xEFR') {
                return context.set('mlir.binary');
            }
        }
        try {
            const reader = await context.read('text', 0x10000);
            for (let line = reader.read('\n'); line !== undefined; line = reader.read('\n')) {
                if (/module\s+(@\w+|\w+|attributes)/.test(line) ||
                    /tensor<[\w\d]+>/.test(line) ||
                    /func[.\s]*@\w+/.test(line) ||
                    /%\w+\s*=\s*"[\w.]+/.test(line) ||
                    /%\w+\s*=\s*\w+\./.test(line) ||
                    /#\w+\s*=\s*loc\s*\(/.test(line) ||
                    /\w+\.\w+\s+@\w+/.test(line) ||
                    /:\s*![\w.]+/.test(line) ||
                    /(%\w+|\w{2,}|[)])\s*:\s*(\[|tensor<)/.test(line) ||
                    /->\s*(![\w.]+|\(|tensor<)/.test(line)) {
                    return context.set('mlir.text');
                }
            }
        } catch {
            // continue regardless of error
        }
        return null;
    }

    async open(context) {
        const metadata = await mlir.Metadata.open(context);
        switch (context.type) {
            case 'mlir.text': {
                const decoder = await context.read('text.decoder');
                const parser = new mlir.Parser(decoder, metadata);
                const obj = await parser.read();
                return new mlir.Model(metadata, obj);
            }
            case 'mlir.binary': {
                const reader = await context.read('binary');
                const parser = new mlir.BytecodeReader(reader);
                parser.read();
                throw new mlir.Error('File contains unsupported MLIR bytecode data.');
            }
            default: {
                throw new mlir.Error(`Unsupported MLIR format '${context.type}'.`);
            }
        }
    }
};

mlir.Model = class {

    constructor(metadata, obj) {
        this.format = 'MLIR';
        this.modules = [];
        this.metadata = [];
        for (const op of obj.operations) {
            if (op.name.endsWith('.func')) {
                const graph = new mlir.Graph(metadata, op);
                this.modules.push(graph);
            }
            if (op.name.endsWith('.module')) {
                for (const region of op.regions) {
                    for (const block of region.blocks) {
                        for (const op of block.operations) {
                            if (op.name.endsWith('.func')) {
                                const graph = new mlir.Graph(metadata, op);
                                this.modules.push(graph);
                            }
                        }
                    }
                }
            }
        }
        if (obj.definitions) {
            for (const attribute of obj.definitions) {
                const value = typeof attribute.value === 'string' ? attribute.value : JSON.stringify(attribute.value);
                const metadata = new mlir.Argument(attribute.name, value, 'attribute');
                this.metadata.push(metadata);
            }
        }
    }
};

mlir.Graph = class {

    constructor(metadata, func) {
        const attr = Object.fromEntries(func.attributes.map((attr) => [attr.name, attr.value]));
        this.name = attr.sym_name || '';
        this.type = func.name === 'func' || func.name.endsWith('.func') ? 'function' : '';
        this.description = func.name;
        this.inputs = [];
        this.outputs = [];
        this.nodes = [];
        const tensors = new Map();
        const tensor = (arg) => {
            if (!tensors.has(arg.name)) {
                tensors.set(arg.name, new mlir.Value(arg.name, arg.type, null, arg.value));
            }
            return tensors.get(arg.name);
        };
        // inputs of function
        const function_type = attr.function_type;
        for (let i = 0; i < function_type.inputs.length; i++) {
            const input = function_type.inputs[i];
            const name = input.name || i.toString();
            const type = mlir.Utility.valueType(input.type);
            const value = new mlir.Value(input.value, type, '', null);
            const argument = new mlir.Argument(name, [value]);
            this.inputs.push(argument);
        }
        // outputs of function
        for (let i = 0; i < function_type.results.length; i++) {
            const output = function_type.results[i];
            const name = output.name || i.toString();
            const type = mlir.Utility.valueType(output.type);
            const value = new mlir.Value(output.value, type, '', null);
            const argument = new mlir.Argument(name, [value]);
            this.outputs.push(argument);
        }
        // operations
        // args is map of edges. args will be converted to mlir.Arguemnts.
        const values = new Map();
        values.map = (name) => {
            if (!values.has(name)) {
                values.set(name, { name, to: [], from: [] });
            }
            return values.get(name);
        };
        // operations - setup arguments
        const operations = [];
        for (const region of func.regions) {
            for (const block of region.blocks) {
                for (const op of block.operations) {
                    const operation = {
                        type: op.kind || op.name,
                        identifier: op.name,
                        attributes: op.attributes,
                        operands: [],
                        results: [],
                        delete: false,
                    };
                    // Get metadata for this operation to populate proper input/output names
                    const opMetadata = metadata.type(op.name);
                    const operands = op.operands || [];
                    for (let i = 0; i < operands.length; i++) {
                        const input = op.operands[i];
                        // Use metadata input name if available and input.name is not set
                        const inputName = input.name || (opMetadata && opMetadata.inputs && opMetadata.inputs[i] ? opMetadata.inputs[i].name : null) || i.toString();
                        if (input.value instanceof Uint8Array) {
                            operation.operands.push({
                                name: inputName,
                                value: input.value,
                                type: input.type
                            });
                        } else if (Number.isInteger(input.value)) {
                            operation.operands.push({
                                name: inputName,
                                value: input.value,
                                type: 'int64'
                            });
                        } else if (typeof input.value === 'boolean') {
                            operation.operands.push({
                                name: inputName,
                                value: input.value,
                                type: 'boolean'
                            });
                        } else if (Array.isArray(input.value)) {
                            operation.operands.push({
                                name: inputName,
                                value: input.value
                            });
                        } else {
                            const value = values.map(input);
                            value.to.push(operation);
                            const args = [{ name: input.value, type: input.type }];
                            operation.operands.push({
                                name: inputName,
                                value: args
                            });
                        }
                    }
                    const results = op.results || [];
                    for (let i = 0; i < results.length; i++) {
                        const output = results[i];
                        const value = values.map(output.value);
                        value.type = mlir.Utility.valueType(output.type);
                        value.from.push(operation);
                        // Use metadata output name if available and output.name is not set
                        const outputName = output.name || (opMetadata && opMetadata.outputs && opMetadata.outputs[i] ? opMetadata.outputs[i].name : null) || i.toString();
                        operation.results.push({
                            name: outputName,
                            value: [value]
                        });
                    }
                    operations.push(operation);
                }
            }
        }
        // // operations - constant ops
        // for (const op of operations) {
        //     if (op.type === 'const' && op.inputs.length === 0 &&
        //         op.outputs.length === 1 && op.outputs[0].value.length === 1) {
        //         const argument = op.outputs[0].value[0];
        //         if (op.attributes && op.attributes.val) {
        //             const type = argument.type;
        //             const data = op.attributes.val;
        //             if (data instanceof Uint8Array && data.length === 2 &&
        //                 type.dataType === 'float16' && type.shape.dimensions.length === 0) {
        //                 const view = new DataView(data.buffer, data.byteOffset, data.byteLength);
        //                 argument.value = view.getFloat16(0, true);
        //             } else {
        //                 argument.value = data;
        //             }
        //             argument.const = true;
        //             op.delete = true;
        //         }
        //     }
        // }

        // //
        // for (const op of operations) {
        //     for (const input of op.inputs) {
        //         if (input.value.length > 1 && input.value.some((argument) => argument.const)) {
        //             if (input.value.every((argument) => argument.value instanceof mlir.Tensor)) {
        //                 continue;
        //             }
        //             for (const argument of input.value) {
        //                 for (const from of argument.from) {
        //                     from.delete = false;
        //                 }
        //                 delete argument.value;
        //             }
        //         }
        //     }
        // }

        // for (const op of operations) {
        //     if (op.delete) {
        //         continue;
        //     }
        //     op.inputs = op.inputs.filter((input) => {
        //         if (input.value.every((argument) => argument.value === undefined || argument.value instanceof coreml.Tensor)) {
        //             return true;
        //         }
        //         if (input.value.length === 1) {
        //             const argument = input.value[0];
        //             op.attributes[input.name] = argument.value;
        //             return false;
        //         }
        //         op.attributes[input.name] = input.value.map((argument) => argument.value[0]);
        //         return false;
        //     });
        // }
        // Add function inputs to tensors map first
        for (const input of this.inputs) {
            for (const arg of input.value) {
                if (!tensors.has(arg.name)) {
                    tensors.set(arg.name, arg);
                }
            }
        }
        // Add function outputs to tensors map, reusing existing values if they have the same name
        for (const output of this.outputs) {
            for (let i = 0; i < output.value.length; i++) {
                const arg = output.value[i];
                if (tensors.has(arg.name)) {
                    // Reuse the existing value object instead of creating a duplicate
                    output.value[i] = tensors.get(arg.name);
                } else {
                    tensors.set(arg.name, arg);
                }
            }
        }
        for (const op of operations) {
            if (op.delete) {
                continue;
            }
            op.operands = op.operands.map((input) => {
                if (input.type && input.type.startsWith('tensor<')) {
                    const type = mlir.Utility.valueType(input.type);
                    const tensor = new mlir.Tensor(type, input.value);
                    return new mlir.Argument(input.name, tensor, 'tensor');
                }
                if (input.type) {
                    return new mlir.Argument(input.name, input.value, input.type);
                }
                if (Array.isArray(input.value) && !input.value.every((value) => typeof value.name === 'string' && value.name.startsWith('%'))) {
                    return new mlir.Argument(input.name, input.value, input.type || 'attribute');
                }
                return new mlir.Argument(input.name, input.value.map((argument) => tensor(argument)));
            });
            op.results = op.results.map((output) => {
                return new mlir.Argument(output.name, output.value.map((argument) => tensor(argument)));
            });
        }
        for (const op of operations.filter((op) => !op.delete)) {
            const node = new mlir.Node(metadata, op);
            this.nodes.push(node);
        }
    }
};

mlir.Argument = class {

    constructor(name, value, type) {
        this.name = name;
        this.value = value;
        this.type = type || null;
        // Normalize common type aliases and accept extended MLIR types
        if (this.type) {
            switch (this.type) {
                case 'i64': case 'si64': this.type = 'int64'; break;
                case 'i32': case 'si32': this.type = 'int32'; break;
                case 'i16': case 'si16': this.type = 'int16'; break;
                case 'i8': case 'si8': this.type = 'int8'; break;
                case 'i1': this.type = 'boolean'; break;
                case 'f32': case 'float32': this.type = 'float32'; break;
                case 'f64': case 'float64': this.type = 'float64'; break;
                case 'f16': this.type = 'float16'; break;
                case null:
                case 'attribute':
                case 'boolean':
                case 'string':
                case 'int64':
                case 'int32':
                case 'int16':
                case 'int8':
                case 'float16':
                case 'tensor':
                case 'type':
                    break;
                default:
                    // Accept other MLIR types without normalization
                    if (/^[usi]i?[0-9]+$/.test(this.type) || /^f[0-9]+$/.test(this.type) ||
                        this.type === 'bf16' || this.type === 'index' || this.type === 'none' ||
                        this.type.startsWith('!') || this.type.startsWith('tensor<') ||
                        this.type.startsWith('memref<') || this.type.startsWith('vector<')) {
                        break;
                    }
                    throw new mlir.Error(`Unsupported argument type '${this.type}'.`);
            }
        }
    }
};

mlir.Value = class {

    constructor(name, type, description, initializer) {
        if (typeof name !== 'string') {
            throw new mlir.Error(`Invalid value identifier '${JSON.stringify(name)}'.`);
        }
        this.name = name;
        this.type = !type && initializer ? initializer.type : type;
        this.description = description || null;
        this.initializer = initializer || null;
    }
};

mlir.Node = class {

    constructor(metadata, op) {
        if (!op.type) {
            throw new mlir.Error('Undefined node type.');
        }
        this.type = { ...metadata.type(op.identifier || '') };
        this.type.name = op.type || '';
        this.type.identifier = op.identifier || '';
        this.name = op.name || '';
        this.inputs = op.operands || [];
        this.outputs = op.results || [];
        this.attributes = [];
        if (op.attributes) {
            for (let i = 0; i < op.attributes.length; i++) {
                const attr = op.attributes[i];
                const name = attr.name || i.toString();
                let type = attr.type;
                let value = attr.value;
                if (type && type.startsWith('tensor<')) {
                    value = new mlir.Tensor(mlir.Utility.valueType(type), value);
                    type = 'tensor';
                }
                const attribute = new mlir.Argument(name, value, type || 'attribute');
                this.attributes.push(attribute);
            }
        }
    }
};

mlir.Tensor = class {

    constructor(type, data) {
        this.type = type;
        this.values = data;
        this.encoding = data instanceof Uint8Array ? '<' : '|';
    }
};

mlir.TensorType = class {

    constructor(dataType, shape) {
        this.dataType = mlir.Utility.dataType(dataType); // string
        this.shape = shape || new mlir.TensorShape([]);  // mlir.TensorShape
    }

    toString() {
        return this.dataType + this.shape.toString();
    }
};

mlir.TensorShape = class {

    constructor(dimensions) {
        this.dimensions = dimensions;
    }

    toString() {
        if (!this.dimensions || this.dimensions.length === 0) {
            return '';
        }
        return `[${this.dimensions.map((dimension) => dimension.toString()).join(',')}]`;
    }
};

mlir.Token = class {

    constructor(kind, value) {
        this.kind = kind;
        this.value = value;
    }
};

mlir.Tokenizer = class {

    constructor(decoder) {
        this._decoder = decoder;
        this._currentPosition = this._decoder.position;
        this._current = this._decoder.decode();
        this._nextPosition = this._decoder.position;
        this._next = this._decoder.decode();
    }

    read() {
        this._position = this._currentPosition;
        while (this._current) {
            switch (this._current) {
                case ' ':
                case '\t':
                case '\n':
                case '\r':
                case '\f':
                    this._skipWhitespace();
                    this._position = this._currentPosition;
                    continue;
                case '/':
                    this._skipComment();
                    this._position = this._currentPosition;
                    continue;
                case '.':
                    if (/[0-9]/.test(this._peek())) {
                        return this._number();
                    }
                    throw new mlir.Error(`Unexpected character '${this._current}' ${this.location()}`);
                case '-':
                    if (/[0-9]/.test(this._peek())) {
                        return this._number();
                    } else if (this._peek() === '>') {
                        this._read();
                        this._read();
                        return new mlir.Token('->', '->');
                    }
                    this._read();
                    return new mlir.Token('keyword', '-');
                case '+':
                    if (/[0-9]/.test(this._peek())) {
                        return this._number();
                    }
                    this._read();
                    return new mlir.Token('keyword', '+');
                case '"':
                    return this._stringLiteral();
                case '@':
                    return this._symbolRefId();
                case '%':
                    return this._valueId();
                case '#':
                    return this._attributeAlias();
                case '!': { // type alias
                    return this._typeAlias();
                }
                case '^':
                    return this._caretId();
                case '=':
                    if (this._peek() === '=') {
                        this._read();
                        this._read();
                        return new mlir.Token('==', '==');
                    }
                    this._read();
                    return new mlir.Token('=', '=');
                case ':':
                    if (this._peek() === ':') {
                        this._read();
                        this._read();
                        return new mlir.Token('::', '::');
                    }
                    this._read();
                    return new mlir.Token(':', ':');
                case ',':
                case '(':
                case ')':
                case '{':
                case '}':
                case '[':
                case ']':
                case '<':
                case '?':
                case '*': {
                    const value = this._read();
                    return new mlir.Token(value, value);
                }
                case '>':
                    if (this._peek() === '=') {
                        this._read();
                        this._read();
                        return new mlir.Token('>=', '>=');
                    }
                    this._read();
                    return new mlir.Token('>', '>');
                default:
                    if (/[a-zA-Z_$]/.test(this._current) || /[-.]/.test(this._current)) {
                        const token = this._identifier();
                        if (token.value === 'tensor' || token.value === 'vector') {
                            let v = '';
                            let c = '';
                            let level = 0;
                            do {
                                c = this._read();
                                if (c === '<') {
                                    level++;
                                } else if (c === '>') {
                                    level--;
                                }
                                v += c;
                            } while (level > 0 || c !== '>');
                            return new mlir.Token(token.value, `${token.value}${v}`);
                        }
                        return token;
                    }
                    if (/[0-9]/.test(this._current)) {
                        return this._number();
                    }
                    throw new mlir.Error(`Unexpected character '${this._current}' ${this.location()}`);
            }
        }
        return new mlir.Token('eof', null);
    }

    location() {
        let line = 1;
        let column = 1;
        const position = this._decoder.position;
        this._decoder.position = 0;
        let c = '';
        do {
            if (this._decoder.position === this._position) {
                this._decoder.position = position;
                return `at ${line}:${column}.`;
            }
            c = this._decoder.decode();
            if (c === '\n') {
                line++;
                column = 1;
            } else {
                column++;
            }
        }
        while (c !== undefined);
        this._decoder.position = position;
        return `at ${line}:${column}.`;
    }

    _read() {
        const current = this._current;
        this._current = this._next;
        this._currentPosition = this._nextPosition;
        this._nextPosition = this._decoder.position;
        this._next = this._decoder.decode();
        return current;
    }

    _peek() {
        return this._next;
    }

    _skipWhitespace() {
        while (this._current !== undefined && (this._current === ' ' || this._current === '\t' || this._current === '\n' || this._current === '\r' || this._current === '\f')) {
            this._read();
        }
    }

    _eat(value) {
        if (this._current === value) {
            this._read();
            return true;
        }
        return false;
    }

    _skipComment() {
        this._read('/');
        if (this._current === '/') {
            while (this._current && this._current !== '\n') {
                this._read();
            }
            this._skipWhitespace();
            if (this._current === '/') {
                this._skipComment();
            }
            return;
        }
        if (this._current === '*') {
            while (this._current) {
                this._read();
                if (this._eat('*') && this._eat('/')) {
                    break;
                }
            }
            this._skipWhitespace();
            if (this._current === '/') {
                this._skipComment();
            }
            return;
        }
        throw new mlir.Error('Invalid comment.');
    }

    _number() {
        let v = '';
        let type = 'int';
        if (this._current === '-') {
            v += this._read();
        }
        while (this._current && /[0-9]/.test(this._current)) {
            v += this._read();
        }
        if (v === '0' && this._current === 'x') {
            v += this._read();
            while (this._current && /[0-9a-fA-F]/.test(this._current)) {
                v += this._read();
            }
            return new mlir.Token(type, parseInt(v, 16));
        }
        if (this._current === '.') {
            v += this._read();
            type = 'float';
            while (this._current && /[0-9]/.test(this._current)) {
                v += this._read();
            }
            if (this._current === 'e' || this._current === 'E') {
                v += this._read();
                if (this._current === '+' || this._current === '-') {
                    v += this._read();
                }
                while (this._current && /[0-9]/.test(this._current)) {
                    v += this._read();
                }
                if (type === 'hex' && !/[x]/.test(this._current)) {
                    return new mlir.Token(type, parseInt(v, 16));
                }
            }
            return new mlir.Token(type, parseFloat(v));
        }
        return new mlir.Token(type, parseInt(v, 10));
    }

    _stringLiteral() {
        let result = '';
        this._read();
        while (this._current && this._current !== '"') {
            if (this._eat('\\')) {
                switch (this._current) {
                    case 'n':
                        result += '\n';
                        break;
                    case 'r':
                        result += '\r';
                        break;
                    case 't':
                        result += '\t';
                        break;
                    default:
                        result += this._current;
                        break;
                }
            } else {
                result += this._current;
            }
            this._read();
        }
        if (this._eat('"')) {
            return new mlir.Token('string', result);
        }
        throw new mlir.Error('Unterminated string literal');
    }

    _identifier() {
        let result = '';
        while (this._current && (/[a-zA-Z_$\-.*]/.test(this._current) || /[0-9]/.test(this._current))) {
            result += this._read();
        }
        switch (result) {
            case 'loc':
                return new mlir.Token('keyword', result);
            case 'true':
            case 'false':
                return new mlir.Token('boolean', result === 'true');
            case 'unknown':
                return new mlir.Token('id', result);
            default:
                return new mlir.Token('id', result);
        }
    }

    _attributeAlias() {
        let value = '#';
        this._read();
        if (this._current === '"') {
            value += this._stringLiteral().value;
        } else {
            while (this._current && (/[a-zA-Z_$]/.test(this._current) || /[0-9]/.test(this._current) || /[-.]/.test(this._current))) {
                value += this._read();
            }
            if (this._current === ':' && this._peek() === ':') {
                value += this._read();
                value += this._read();
                value += this._symbolRefId().value;
            }
        }
        return new mlir.Token('#', value);
    }

    _symbolRefId() {
        let result = '@';
        this._read();
        if (this._current === '"') {
            result += this._stringLiteral().value;
        } else {
            while (this._current && (/[a-zA-Z_$]/.test(this._current) || /[0-9]/.test(this._current) || /[-.]/.test(this._current))) {
                result += this._read();
            }
            if (this._current === ':' && this._peek() === ':') {
                result += this._read();
                result += this._read();
                result += this._symbolRefId().value;
            }
        }
        return new mlir.Token('@', result);
    }

    _typeAlias() {
        this._read();
        const id = this._identifier();
        if (!id) {
            throw new mlir.Error('Invalid type alias.');
        }
        return new mlir.Token('!', `!${id.value}`);
    }

    _valueId() {
        let result = '';
        if (this._current === '%') {
            result = '%';
        } else if (this._current === '$') {
            result = '$';
        }
        this._read();
        while (this._current) {
            if (/[a-zA-Z_$]/.test(this._current) || /[0-9]/.test(this._current) || /[-.#]/.test(this._current)) {
                result += this._read();
            } else if (/[:]/.test(this._current) && /[0-9]/.test(this._next)) { // %myid:3 case
                result += this._read();
            } else {
                break;
            }
        }
        return new mlir.Token('%', result);
    }

    _caretId() {
        let result = '^';
        this._read();
        if (this._current === ':' && this._peek() !== ':') {
            result += this._read();
            return new mlir.Token('^', result);
        }
        while (this._current && (/[a-zA-Z_$]/.test(this._current) || /[0-9]/.test(this._current) || /[-.]/.test(this._current))) {
            result += this._read();
        }
        if (this._current === ':' && this._peek() === ':') {
            result += this._read();
            result += this._read();
            result += this._caretId().value;
        }
        return new mlir.Token('^', result);
    }
};

mlir.Parser = class {

    constructor(decoder, metadata) {
        this._tokenizer = new mlir.Tokenizer(decoder);
        this._token = this._tokenizer.read();
        this._state = {
            defaultDialectStack: ['builtin']
        };
        this._dialects = new Map();
        const operations = Array.from(metadata.types.values());
        this._dialects.set('stablehlo', new mlir.StableHLODialect(operations));
        this._dialects.set('affine', new mlir.AffineDialect(operations));
        this._dialects.set('func', new mlir.FuncDialect(operations));
        this._dialects.set('memref', new mlir.MemRefDialect(operations));
        this._dialects.set('vector', new mlir.VectorDialect(operations));
        this._dialects.set('onnx', new mlir.ONNXDialect(operations));
        this._dialects.set('torch', new mlir.TorchDialect(operations));
        this._dialects.set('hal', new mlir.HALDialect(operations));
        this._dialects.set('mhlo', new mlir.MHLODialect(operations));
        this._dialects.set('flow', new mlir.FlowDialect(operations));
        this._dialects.set('linalg', new mlir.LinalgDialect(operations));
        this._dialects.set('quant', new mlir.QuantDialect(operations));
        this._dialects.set('tf', new mlir.TFDialect(operations));
        this._dialects.set('tfl', new mlir.TFLDialect(operations));
        this._dialects.set('irdl', new mlir.IRDLDialect(operations));
        this._dialects.set('spv', new mlir.SPIRVDialect(operations));
    }

    async read() {
        return this.parse();
    }

    _getDialectForOp(opName) {
        if (!opName) {
            return null;
        }
        const cleanName = opName.replace(/^"|"$/g, '');
        const [dialect] = cleanName.split('.');
        return this._dialects.get(dialect);
    }

    parse() {
        // https://mlir.llvm.org/docs/LangRef/#top-level-productions
        const block = {
            operations: [],
            definitions: []
        };

        while (true) {
            if (this._match('eof')) {
                break;
            }
            if (this._match('#')) { // attribute-alias-def
                const name = this._read();
                this._read('=');
                const value = this._parseAttributeValue();
                block.definitions.push({ name, value });
                continue;
            }
            if (this._match('!')) { // type-alias-def
                const name = this._read();
                this._read('=');
                const type = this._parseType();
                block.definitions.push({ name, type });
                continue;
            }
            if (this._match('{-#')) { // file metadata
                throw new mlir.Error('File metadata is not implemented.');
            }
            const op = this.parseOperation();
            block.operations.push(op);
        }
        return block;
    }

    parseFunctionArgumentList() {
        const inputs = [];
        if (this._eat('(')) {
            while (!this._eat(')')) {
                if (this._match('%')) {
                    // Argument with name: %arg0: type
                    const input = {};
                    input.value = this._token.value;
                    this._read('%');
                    this._read(':');
                    input.type = this._parseType();
                    if (this._match('{')) {
                        input.attributes = [];
                        this.parseAttributeDict(input.attributes);
                    }
                    input.loc = this._parseLocation();
                    inputs.push(input);
                    this._eat(',');
                } else {
                    const input = {};
                    input.value = `%arg${inputs.length}`;  // Generate a name
                    input.type = this._parseType();
                    inputs.push(input);
                    this._eat(',');
                }
            }
        }
        return inputs;
    }

    parseFunctionResultList() {
        const outputs = [];
        if (this._eat('(')) {
            while (!this._eat(')')) {
                const output = {};
                output.value = `%result${outputs.length}`;  // Generate a name
                output.type = this._parseType();
                if (this._match('{')) {
                    output.attributes = [];
                    this.parseAttributeDict(output.attributes);
                }
                outputs.push(output);
                this._eat(',');
            }
        } else {
            const output = {};
            output.value = `%result0`;  // Generate a name
            output.type = this._parseType();
            outputs.push(output);
        }
        return outputs;
    }

    _skipSymbolBetween(open, close) {
        let value = '';
        if (this._match(open)) {
            this._tokenizer.location();
            value += this._read();
            let count = 1;
            while (count > 0) {
                if (this._match(open)) {
                    count++;
                } else if (this._match(close)) {
                    count--;
                }
                value += this._read();
            }
        }
        return value;
    }

    parseOperation() {
        const results = [];
        if (this._match('%')) {
            do {
                if (!this._match('%')) {
                    break;
                }
                const value = this._read();
                const index = value.indexOf(':');
                if (index === -1) {
                    results.push({ value });
                } else {
                    const id = value.substring(0, index);
                    const length = value.substring(index + 1);
                    for (let i = 0; i < length; i++) {
                        const value = `${id}#${i}`;
                        results.push({ value });
                    }
                }
            } while (this._eat(','));
            this._read('=');
        }
        let op = null;
        if (this._match('id')) {
            op = this.parseCustomOperation(results);
        } else if (this._match('string')) {
            op = this.parseGenericOperation();
        } else {
            throw new mlir.Error(`Unexpected operation name '${this._token.value}' ${this._tokenizer.location()}`);
        }
        op.attributes = [];
        op.operands = [];
        op.results = results;
        op.regions = [];
        op.kind = op.name.split('.').pop();
        if (op.name.startsWith('torch.')) {
            const parts = op.name.split('.');
            if (parts[1] === 'aten' || parts[1] === 'prim') {
                [, , op.kind] = parts;
            } else {
                [, op.kind] = parts;
            }
        }
        const dialect = this._getDialectForOp(op.name);
        if (dialect && dialect.parseOperation(this, op.name, op)) {
            op.loc = this._parseLocation();
            return op;
        }
        if (op.name.endsWith('.call') || op.name.endsWith('.generic_call')) {
            this.parseSymbolName('callee', op.attributes);
        }
        if (op.name === 'arith.cmpi' || op.name.endsWith('.contract')) {
            if (this._match('id')) {
                const list = [];
                do {
                    list.push(this._read());
                } while (this._eat(',') && this._match('id'));
                op.attributes.push({ name: 'predicate', value: list });
            }
        }
        if (op.name.endsWith('.func')) {
            this.parseOptionalVisibilityKeyword(op.attributes);
            this.parseSymbolName('sym_name', op.attributes);
            const type = {};
            type.inputs = this.parseFunctionArgumentList();
            this.parseOptionalAttrDictWithKeyword(op.attributes);
            type.results = [];
            if (this._eat('->')) {
                for (const result of this.parseFunctionResultList()) {
                    type.results.push(result);
                }
            }
            op.attributes.push({ name: 'function_type', value: type });
            this.parseOptionalAttrDictWithKeyword(op.attributes);
            if (this._match('{')) {
                const region = {};
                this.parseRegion(region);
                op.regions.push(region);
            }
            if (op.regions.length > 0) {
                const region = op.regions[op.regions.length - 1];
                if (region.blocks.length > 0) {
                    const block = region.blocks[region.blocks.length - 1];
                    if (block.operations.length > 0) {
                        const lastOp = block.operations[block.operations.length - 1];
                        type.results = lastOp.operands;
                        block.operations.pop();
                    }
                }
            }
            return op;
        }
        if (op.name.endsWith('.module') || op.name.endsWith('.state')) {
            op.sym_name = this.parseOptionalSymbolName();
            this.parseOptionalAttrDictWithKeyword(op.attributes);
            const region = {};
            this.parseRegion(region);
            op.regions.push(region);
            return op;
        }
        // IRDL operations: irdl.dialect, irdl.operation, irdl.type, irdl.attribute
        if (op.name === 'irdl.dialect' || op.name === 'irdl.operation' ||
            op.name === 'irdl.type' || op.name === 'irdl.attribute') {
            // Parse @symbol_name
            op.sym_name = this.parseOptionalSymbolName();
            this.parseOptionalAttrDictWithKeyword(op.attributes);
            const region = {};
            this.parseRegion(region);
            op.regions.push(region);
            return op;
        }
        if (this._match('}')) {
            return op;
        }
        if (op.name === 'torch.constant.none') {
            return op;
        }
        // (%a, %b)
        // condition: start with `(%`, `%`, or `()`
        op.operands = this._parseArguments();
        if (op.name.endsWith('.for')) {
            this._read('=');
            // Lower bound can be: number, %var, or #map(...)
            if (this._match('#')) {
                this._parseValue();
            } else {
                this._read();  // Read number or %var
            }
            this._read('id', 'to');
            // Upper bound can be: number, %var, #map(...), or id #map(...)
            if (this._match('#')) {
                // Just #map(...) without a preceding id
                this._parseValue();
            } else if (this._match('id') && !this._match('id', 'step') && !this._match('id', 'iter_args')) {
                this._read('id');
                // Check if it's a function application like min #map(%arg)
                if (this._match('#')) {
                    // Read the map reference and its arguments
                    this._parseValue();
                }
            } else if (this._match('%') || this._match('int') || this._match('float')) {
                this._read();
            }
            if (this._eat('id', 'step')) {
                this._read();
            }
            if (this._eat('id', 'iter_args')) {
                this._skipSymbolBetween('(', ')');
            }
            if (this._eat('->') || this._eat('id', 'to')) {
                if (op.results.length > 0) {
                    this._parseArgumentTypes(op.results);
                } else {
                    op.results = this._parseArguments();
                }
            }
            const region = {};
            this.parseRegion(region);
            op.regions.push(region);
            return op;
        }
        // Parse successor blocks (for branch operations)
        // Successors are block labels starting with ^
        if (this._match('^')) {
            op.successors = [];
            while (this._match('^')) {
                const successor = {};
                successor.label = this._read('^');
                // Parse successor arguments: ^bb1(%arg0, %arg1 : type1, type2)
                if (this._eat('(')) {
                    successor.arguments = [];
                    while (!this._eat(')')) {
                        if (this._match('%')) {
                            const value = this._read('%');
                            successor.arguments.push({ value });
                        }
                        this._eat(',');
                    }
                    // Parse types if present
                    if (this._eat(':')) {
                        let idx = 0;
                        while (idx < successor.arguments.length && !this._match(',') && !this._match('[') && !this._match('{') && !this._match('^')) {
                            const type = this._parseType();
                            if (successor.arguments[idx]) {
                                successor.arguments[idx].type = type;
                            }
                            idx++;
                            this._eat(',');
                        }
                    }
                }
                op.successors.push(successor);
                // Multiple successors separated by comma
                if (!this._eat(',')) {
                    break;
                }
            }
        }

        // successor-list (indices)?
        // condition: start with `[`, end with `]`
        // Some operations like tensor.extract_slice have multiple consecutive bracket groups
        while (this._match('[')) {
            this._skipSymbolBetween('[', ']');
        }
        // dictionary-properties?
        // condition: start with `<`, end with `>`
        this._skipSymbolBetween('<', '>');
        // region-list?
        // condition: start with `(` followed by `{` or `id`, parse regions until `)`
        let parsedRegionList = false;
        if (this._match('(')) {
            const savedToken = this._token;
            this._eat('(');
            if (this._match('{') || this._match('id')) {
                parsedRegionList = true;
            } else {
                // Put the '(' back - it's not a region-list
                this._token = savedToken;
            }
        }
        if (parsedRegionList) {
            while (!this._match(')')) {
                // Handle region entry point labels like: reducer(%arg1: type, %arg2: type) {
                let entryLabel = null;
                const entryArgs = [];
                if (this._match('id') && !this._match('{')) {
                    entryLabel = this._read('id');
                    if (this._eat('(')) {
                        while (!this._eat(')')) {
                            const value = this._read('%');
                            this._read(':');
                            const type = this._parseType();
                            entryArgs.push({ value, type });
                            this._eat(',');
                        }
                    }
                }
                if (!this._match('{')) {
                    throw new mlir.Error(`Expected '{' for region in region-list, but got '${this._token.value}' ${this._tokenizer.location()}`);
                }
                const region = {};
                this.parseRegion(region);
                if (entryLabel) {
                    region.entryLabel = entryLabel;
                    region.entryArgs = entryArgs;
                }
                op.regions.push(region);
                if (!this._eat(',') && !this._match(')')) {
                    throw new mlir.Error(`Expected ',' or ')' after region in region-list, but got '${this._token.value}' ${this._tokenizer.location()}`);
                }
            }
            this._read(')');  // Consume closing )
        }
        // dictionary-attribute?
        // condition: start with `{`, end with `}`
        if (this._match('{')) {
            // After a region-list, attributes should be parsed even if regions exist
            if (parsedRegionList || op.attributes.length === 0 || (op.attributes.length === 1 && op.attributes[0].name === 'predicate')) {
                this.parseAttributeDict(op.attributes);
            } else {
                const region = {};
                this.parseRegion(region);
                op.regions.push(region);
            }
        }
        // : (f32, tensor<1xf32>)
        if (this._eat(':')) {
            this._parseArgumentTypes(op.operands);
        }
        // -> f32
        if (this._eat('->') || this._eat('id', 'to')) {
            if (op.results.length > 0) {
                this._parseArgumentTypes(op.results);
            } else {
                op.results = this._parseArguments();
            }
        }
        if (this._match('{')) {
            const region = {};
            this.parseRegion(region);
            op.regions.push(region);
            if (op.name.endsWith('.if') && this._match('id', 'else')) {
                this._read('id', 'else');
                const region = {};
                this.parseRegion(region);
                op.regions.push(region);
            }
        }
        op.loc = this._parseLocation(); // trailing-location
        return op;
    }

    parseCustomOperation(/* results */) {
        const opNameInfo = this.parseCustomOperationName();
        const op = {};
        op.name = opNameInfo;
        return op;
    }

    parseCustomOperationName() {
        let opName = this._read('id');
        if (opName.indexOf('.') === -1) {
            const dialect = this._state.defaultDialectStack[this._state.defaultDialectStack.length - 1];
            opName = `${dialect}.${opName}`;
        }
        return opName;
    }

    parseGenericOperation() {
        const op = {};
        op.name = this._read('string');
        return op;
    }

    parseOptionalVisibilityKeyword(attributes) {
        if (this._match('id', 'private') || this._match('id', 'public') || this._match('id', 'nested')) {
            const value = this._read();
            attributes.push({ name: 'sym_visibility', value });
        }
    }

    parseSymbolName(name, attributes) {
        const value = this._read('@');
        attributes.push({ name, value });
    }

    parseOptionalSymbolName() {
        if (this._match('@')) {
            return this._read('@');
        }
        return null;
    }

    parseOptionalAttrDictWithKeyword(attributes) {
        if (this._eat('id', 'attributes')) {
            this.parseAttributeDict(attributes);
        }
    }

    parseAttributeDict(attributes) {
        if (this._eat('{')) {
            while (!this._eat('}')) {
                let name = null;
                if (this._match('id') || this._match('string') || this._match('keyword')) {
                    name = this._read();
                } else if (!this._match('=') && !this._match(':') && !this._match('}')) {
                    throw new mlir.Error(`Expected attribute name or '}', but got '${this._token.value}' ${this._tokenizer.location()}`);
                }
                let attribute = {};
                if (this._eat('=') || this._eat(':')) {
                    attribute = this._parseValue();
                    if (this._eat(':')) {
                        attribute.type = this._parseType();
                    }
                } else if (name) {
                    attribute = { name };
                    attributes.push(attribute);
                    this._eat(',');
                    continue;
                } else {
                    // No name and no =, this is an error we should have caught above
                    break;
                }

                attribute.name = name;
                attributes.push(attribute);
                if (!this._eat(',') && !this._match('}')) {
                    throw new mlir.Error(`Expected ',' or '}' after attribute, but got '${this._token.value}' ${this._tokenizer.location()}`);
                }
            }
        }
    }

    parseRegion(region) {
        region.blocks = Array.isArray(region.blocks) ? region.blocks : [];
        const block = {};
        this.parseBlock(block);
        region.blocks.push(block);

        // Handle additional blocks in the region
        // After parseBlock breaks on encountering ^label, we're still inside the region
        // Continue parsing blocks until we hit the closing }
        while (this._match('^') && !this._match('}')) {
            const nextBlock = {};
            nextBlock.operations = [];
            nextBlock.arguments = [];
            // Parse block label
            nextBlock.name = this._read('^');
            if (this._eat('(')) {
                while (!this._eat(')')) {
                    const value = this._read('%');
                    this._read(':');
                    const type = this._parseType();
                    nextBlock.arguments.push({ value, type });
                    this._eat(',');
                }
            }
            this._read(':');
            while (!this._match('^') && !this._match('}')) {
                const op = this.parseOperation();
                nextBlock.operations.push(op);
            }
            region.blocks.push(nextBlock);
        }
        return region;
    }

    parseBlock(block) {
        block.operations = Array.isArray(block.operations) ? block.operations : [];
        block.arguments = Array.isArray(block.arguments) ? block.arguments : [];
        this._read('{');
        if (this._match('^')) {
            block.name = this._read('^');
            if (this._eat('(')) {
                while (!this._eat(')') && !this._match('^')) {
                    const value = this._read('%');
                    this._read(':');
                    const type = this._parseType();
                    block.arguments.push({ value, type });
                    this._eat(',');
                }
            }
            this._read(':');
        }
        while (!this._eat('}')) {
            // Check if this is a new block label (for regions with multiple blocks)
            if (this._match('^')) {
                // This is a subsequent block in the same region
                // We need to signal to parseRegion to handle this
                // For now, we'll break and let parseRegion handle it
                break;
            }
            const op = this.parseOperation();
            block.operations.push(op);
        }
        block.loc = this._parseLocation();
        return block;
    }

    _parseLocation() {
        if (this._eat('keyword', 'loc')) {
            const location = {};
            this._read('(');
            if (this._match('string')) {
                const str = this._read('string');
                if (this._match('(')) {
                    location.name = str;
                    this._read('(');
                    location.child = this._parseLocationContent();
                    this._read(')');
                } else {
                    location.file = str;
                    if (this._eat(':')) {
                        location.line = this._read('int');
                        if (this._eat(':')) {
                            location.col = this._read('int');
                        }
                    }
                }
            } else if (this._match('#')) {
                location.alias = this._read();
            } else if (this._match('id', 'unknown')) {
                this._read();
                location.unknown = true;
            } else if (this._match('id', 'callsite')) {
                this._read('id', 'callsite');
                this._read('(');
                location.type = 'callsite';
                location.callee = this._parseLocationContent();
                this._read('id', 'at');
                location.caller = this._parseLocationContent();
                this._read(')');
            } else if (this._match('id', 'fused')) {
                this._read('id', 'fused');
                location.type = 'fused';
                if (this._eat('<')) {
                    location.metadata = this._parseValue();
                    this._read('>');
                }
                this._read('[');
                location.locations = [];
                do {
                    location.locations.push(this._parseLocationContent());
                } while (this._eat(','));
                this._read(']');
            } else {
                throw new mlir.Error(`Unexpected location '${this._token.value}' ${this._tokenizer.location()}`);
            }
            this._read(')');
            return location;
        }
        return null;
    }

    _parseLocationContent() {
        if (this._match('#')) {
            return { alias: this._read() };
        }
        if (this._match('keyword', 'loc')) {
            return this._parseLocation();
        }
        throw new mlir.Error(`Expected location content, got '${this._token.value}' ${this._tokenizer.location()}`);
    }

    _parseOperationName() {
        switch (this._token.kind) {
            case 'string':
                return this._read();
            case 'id':
                return this._read('id');
            default:
                throw new mlir.Error(`Unexpected operation '${this._token.value}' ${this._tokenizer.location()}`);
        }
    }

    _parseArguments() {
        const inputs = [];
        if (this._match('{')) {
            return inputs;
        }
        const open = this._eat('(');
        // eslint-disable-next-line no-unmodified-loop-condition
        while (!this._match(')') && !this._match('->') && !this._match('{') && !this._match('}') && !this._match('[') && !this._match('=') && !this._match('^') && !(this._match(':') && !open)) {
            const input = {};
            if (this._token.kind === 'id' && this._token.value !== 'dense' && this._token.value !== 'dense_resource') {
                const identifier = this._read('id');
                if (this._eat('(')) {
                    const args = this._parseArguments();
                    for (let i = 0; i < args.length; i++) {
                        const arg = args[i];
                        arg.name = `${identifier}.${i}`;
                        inputs.push(arg);
                    }
                    if (this._eat(':')) {
                        this._parseArgumentTypes(inputs);
                    }
                    this._read(')');
                    continue;
                } else if (this._match('=')) {
                    input.name = identifier;
                    this._read('=');
                } else if (this._match(':')) {
                    // Named argument syntax: identifier: value (e.g., init: %init)
                    input.name = identifier;
                    this._read(':');
                } else {
                    // Identifier is the value itself
                    input.value = identifier;
                    inputs.push(input);
                    if (!this._eat(',')) {
                        break;
                    }
                    continue;
                }
            }
            if (this._match('%')) {
                input.value = this._read();
                if (open && this._eat(':')) {
                    input.type = this._parseType();
                }
            } else if (this._match('keyword', 'loc')) {
                // Location keyword - stop parsing arguments
                break;
            } else {
                const value = this._parseValue();
                input.type = value.type;
                input.value = value.value;
                if (open && this._eat(':')) {
                    input.type = this._parseType();
                }
            }
            inputs.push(input);
            // Commas are optional - arguments can be space-separated
            this._eat(',');
        }
        if (open) {
            this._read(')');
        }
        return inputs;
    }

    _parseType() {
        if (this._token.kind === 'id') {
            const value = this._token.value;
            // Check for standard types and integer/float types with various bit widths
            if (value === 'none' || value === 'index' ||
                // Integer types: i1, i2, i4, i8, i16, i32, i64
                /^i[0-9]+$/.test(value) ||
                // Signed integer types: si2, si4, si8, si16, si32, si64
                /^si[0-9]+$/.test(value) ||
                // Unsigned integer types: ui2, ui4, ui8, ui16, ui32, ui64
                /^ui[0-9]+$/.test(value) ||
                // Float types: f8, f16, f32, f64, f128
                /^f[0-9]+$/.test(value) ||
                // BFloat16
                value === 'bf16') {
                return this._read('id');
            }
        }
        if (this._match('tensor') || this._match('vector')) {
            return this._read();
        }
        if (this._match('id', 'memref')) {
            let value = this._read('id');
            value += this._skipSymbolBetween('<', '>');
            // Handle dynamic dimensions: memref<?xf32>{%c123}
            // We need to check if { starts a dimension spec or a region
            // Dimension specs start with % (value reference), regions start with operations
            if (this._match('{')) {
                // Save state to be able to look ahead
                const savedToken = this._token;
                this._read('{');  // consume {
                const isValueRef = this._match('%');
                // Put back the token we read
                this._token = savedToken;
                if (isValueRef) {
                    // This is a dimension spec, consume it
                    value += this._skipSymbolBetween('{', '}');
                }
                // Otherwise, leave the { for the caller to handle as a region
            }
            return value;
        }
        if (this._match('!')) {
            if (this._match('!', '!torch.vtensor') ||
                this._match('!', '!torch.list')) {
                let value = this._read();
                value += this._skipSymbolBetween('<', '>');
                return value;
            }
            if (this._match('!', '!torch.int') ||
                this._match('!', '!torch.bool') ||
                this._match('!', '!torch.none')) {
                return this._read();
            }
            let value = this._read();
            if (this._match('<')) {
                value += this._skipSymbolBetween('<', '>');
            }
            return value;
        }
        throw new mlir.Error(`Invalid type ${this._tokenizer.location()}`);
    }

    _parseArgumentTypes(args) {
        let index = 0;
        const open = this._eat('(');
        if (open) {
            while (this._token.kind !== ')') {
                const type = this._parseType();
                if (!type) {
                    break;
                }
                if (index < args.length) {
                    args[index].type = type;
                } else {
                    const arg = {};
                    arg.type = type;
                    args.push(arg);
                }
                index++;
                if (!this._eat(',')) {
                    break;
                }
            }
            this._read(')');
        } else {
            while ((index === 0 || index < args.length) &&
                !this._eat(')') &&
                this._token.kind !== '->' &&
                this._token.value !== 'loc' &&
                this._token.value !== 'return' && this._token.value !== 'func.return' &&
                this._token.kind !== '}' &&
                this._token.kind !== '%') {
                const type = this._parseType();
                if (!type) {
                    break;
                }
                if (index < args.length) {
                    args[index].type = type;
                } else {
                    const input = {};
                    input.type = type;
                    args.push(input);
                }
                index++;
                if (!this._eat(',')) {
                    break;
                }
            }
        }
    }

    _parseOutputArguments() {
        const outputs = [];
        const outputTypes = [];
        this._read('(');
        while (!this._eat(')')) {
            const value = this._eat('%');
            if (value) {
                outputs.push(value.value);
            }
            if (this._eat(':')) {
                const type = this._parseType();
                outputTypes.push(type.value);
            }
            this._eat(',');
        }
        return { outputs, outputTypes };
    }

    _parseOutputTypes() {
        const outputTypes = [];
        if (this._eat('(')) {
            while (!this._eat(')')) {
                const type = this._parseType();
                outputTypes.push(type);
                this._eat(',');
            }
        } else {
            const type = this._parseType();
            outputTypes.push(type);
        }
        return outputTypes;
    }

    _parseValue() {
        const value = {};
        if (this._match('string')) {
            value.value = this._read();
            value.type = 'string';
            return value;
        }
        if (this._match('int')) {
            value.value = this._read();
            value.type = 'int64';
            return value;
        }
        if (this._match('float')) {
            value.value = this._read();
            value.type = 'float32';
            return value;
        }
        if (this._match('boolean')) {
            value.value = this._read();
            value.type = 'boolean';
            return value;
        }
        if (this._match('@')) {
            value.value = this._read();
            return value;
        }
        if (this._match('id', 'DEFAULT')) {
            value.value = this._read();
            return value;
        }
        // Handle inline affine_map<...> and affine_set<...>
        if (this._match('id', 'affine_map') || this._match('id', 'affine_set')) {
            const name = this._read();
            const args = this._skipSymbolBetween('<', '>');
            // Check if there are dim arguments: affine_map<...>()[%arg0, %arg1]
            // The () part is for dimension arguments, [] part is for symbol arguments
            if (this._match('(')) {
                const dimArgs = this._skipSymbolBetween('(', ')');
                if (this._match('[')) {
                    const symbolArgs = this._skipSymbolBetween('[', ']');
                    return { name, args, dimArgs, symbolArgs };
                }
                return { name, args, dimArgs };
            }
            return { name, args };
        }
        // Handle array<type: values> syntax for StableHLO
        if (this._match('id', 'array')) {
            this._read('id', 'array');
            this._read('<');
            const arrayType = this._parseType();
            const arrayValues = [];
            if (this._eat(':')) {
                while (!this._match('>')) {
                    const val = this._parseValue();
                    arrayValues.push(val.value === undefined ? val : val.value);
                    this._eat(',');
                }
            }
            this._read('>');
            return { value: arrayValues, type: arrayType };
        }
        if (this._eat('[')) {
            const list = [];
            while (!this._eat(']')) {
                const item = this._parseValue();
                // Check for type annotation: value : type
                if (this._eat(':')) {
                    this._parseType(); // Skip the type
                }
                list.push(item.value);
                this._eat(',');
            }
            if (this._eat('id', 'x')) {
                list[0] = Array.from(list);
                const second = [];
                this._read('[');
                while (!this._eat(']')) {
                    const item = this._parseValue();
                    if (this._eat(':')) {
                        this._parseType(); // Skip the type
                    }
                    second.push(item.value);
                    this._eat(',');
                }
                list.push(second);
            }
            return { value: list };
        }
        if (this._match('{')) {
            const attributes = [];
            this.parseAttributeDict(attributes);
            const obj = {};
            for (const attribute of attributes) {
                obj[attribute.name] = attribute.value;
            }
            return { value: obj };
        }
        if (this._match('#')) {
            value.value = this._read('#');
            // Handle attribute with angle brackets: #attr<...>
            if (this._match('<')) {
                value.value += this._skipSymbolBetween('<', '>');
            }
            // Handle map applications like #map(%arg3) for affine operations
            if (this._match('(')) {
                value.value += this._skipSymbolBetween('(', ')');
            }
            return value;
        }
        if (this._match('tensor')) {
            value.value = this._parseType();
            value.type = 'type';
            return value;
        }
        if (this._eat('id', 'dense_resource')) {
            value.value = null;
            value.type = 'dense';
            this._read('<');
            if (!this._match('>')) {
                value.value = this._read();
            }
            this._read('>');
            return value;
        }
        if (this._eat('id', 'dense')) {
            value.value = null;
            value.type = 'dense';
            this._read('<');
            if (!this._match('>')) {
                value.value = this._parseValue().value;
                if (typeof value.value === 'string' && value.value.startsWith('0x')) {
                    const data = new Uint8Array((value.value.length >> 1) - 1);
                    for (let i = 0; i < data.length; i++) {
                        const index = (i << 1) + 2;
                        data[i] = parseInt(value.value.substring(index, index + 2), 16);
                    }
                    value.value = data;
                }
            }
            this._read('>');
            return value;
        }
        throw new mlir.Error(`Unexpected value '${this._token.value}' ${this._tokenizer.location()}`);
    }

    _parseAttributeValue() {
        if (this._match('keyword', 'loc')) {
            return this._parseLocation();
        }
        if (this._match('id', 'affine_map') || this._match('id', 'affine_set')) {
            const name = this._read();
            const args = this._skipSymbolBetween('<', '>');
            return { name, args };
        }
        if (this._match('#')) {
            const name = this._read();
            if (this._eat('<')) {
                while (!this._eat('>')) {
                    this._token = this._tokenizer.read();
                }
            }
            return { name };
        }
        // Handle inline affine map definitions: (d0) -> (d0)
        if (this._match('(')) {
            const parts = [];
            // Parse the entire affine map expression
            let depth = 0;
            let seenArrow = false;
            while (true) {
                if (this._match('(')) {
                    depth++;
                    parts.push(this._read());
                } else if (this._match(')')) {
                    parts.push(this._read());
                    depth--;
                    // After closing the first (), check if there's a ->
                    // After closing the second () (after ->), we're done
                    if (depth === 0) {
                        if (seenArrow) {
                            break;
                        } else if (!this._match('->')) {
                            // No arrow after first (), this might not be an affine map
                            // Put it back by returning a simpler representation
                            break;
                        }
                    }
                } else if (this._match('->')) {
                    seenArrow = true;
                    parts.push(this._read());
                } else {
                    parts.push(this._read());
                }
            }
            return { affine_map: parts.join(' ') };
        }
        throw new mlir.Error(`Unexpected attribute value ${this._tokenizer.location()}`);
    }

    _match(kind, value) {
        return (this._token.kind === kind && (!value || this._token.value === value));
    }

    _read(kind, value) {
        if (kind && this._token.kind !== kind) {
            throw new mlir.Error(`Expected token of type '${kind}', but got '${this._token.value}' ${this._tokenizer.location()}`);
        }
        if (value && this._token.value !== value) {
            throw new mlir.Error(`Expected token with value '${value}', but got '${this._token.value}' ${this._tokenizer.location()}`);
        }
        const token = this._token;
        this._token = this._tokenizer.read();
        return token.value;
    }

    _eat(kind, value) {
        if (this._match(kind, value)) {
            return this._read();
        }
        return null;
    }
};

mlir.BytecodeReader = class {

    constructor(reader) {
        this._reader = new mlir.BinaryReader(reader);
    }

    read() {
        const reader = this._reader;
        reader.read(4); // signature 'ML\xEFR'
        this.version = reader.varint().toNumber();
        this.producer = reader.string();
        this.sections = new Map();
        while (reader.position < reader.length) {
            // https://mlir.llvm.org/docs/BytecodeFormat/
            // https://github.com/llvm/llvm-project/blob/main/mlir/lib/Bytecode/Reader/BytecodeReader.cpp
            const sectionIDAndHasAlignment = reader.byte();
            const sectionID = sectionIDAndHasAlignment & 0x7F;
            const length = reader.varint().toNumber();
            const hasAlignment = sectionIDAndHasAlignment & 0x80;
            if (sectionID >= 9) {
                throw new mlir.Error(`Unsupported section identifier '${sectionID}'.`);
            }
            if (hasAlignment) {
                const alignment = reader.varint();
                reader.skip(alignment);
            }
            const offset = reader.position;
            reader.skip(length);
            this.sections.set(sectionID, { start: offset, end: reader.position });
        }
        if (!this.sections.has(0) || !this.sections.has(1) ||
            !this.sections.has(2) || !this.sections.has(3) ||
            !this.sections.has(4) || (this.version >= 5 && !this.sections.has(8))) {
            throw new mlir.Error('Missing required section.');
        }
        this._parseStringSection();
        if (this.sections.has(8)) {
            this._parsePropertiesSection();
        }
        this._parseDialectSection();
        this._parseResourceSection();
        this._parseAttrTypeSection();
    }

    _parseStringSection() {
        const section = this.sections.get(0);
        const reader = this._reader;
        reader.seek(section.start);
        const lengths = new Array(reader.varint().toNumber());
        for (let i = 0; i < lengths.length; i++) {
            lengths[i] = reader.varint().toNumber();
        }
        const decoder = new TextDecoder('utf-8');
        this.strings = new Array(lengths.length);
        for (let i = 0; i < this.strings.length; i++) {
            const size = lengths[lengths.length - 1 - i];
            const buffer = reader.read(size);
            this.strings[i] = decoder.decode(buffer);
        }
        if (reader.position !== section.end) {
            throw new mlir.Error(`Invalid string section size.`);
        }
    }

    _parseDialectSection() {
        const section = this.sections.get(1);
        const reader = this._reader;
        reader.seek(section.start);
        const numDialects = reader.varint().toNumber();
        this.dialects = new Array(numDialects);
        for (let i = 0; i < this.dialects.length; i++) {
            this.dialects[i] = {};
            if (this.version < 1) { // kDialectVersioning
                const entryIdx = reader.varint().toNumber();
                this.dialects[i].name = this.strings[entryIdx];
                continue;
            }
            const nameAndIsVersioned = reader.varint();
            const dialectNameIdx = (nameAndIsVersioned >> 1n).toNumber();
            this.dialects[i].name = this.strings[dialectNameIdx];
            if (nameAndIsVersioned & 1n) {
                const size = reader.varint().toNumber();
                this.dialects[i].version = reader.read(size);
            }
        }
        let numOps = -1;
        this.opNames = [];
        if (this.version > 4) { // kElideUnknownBlockArgLocation
            numOps = reader.varint().toNumber();
            this.opNames = new Array(numOps);
        }
        let i = 0;
        while (reader.position < section.end) {
            const dialect = this.dialects[reader.varint().toNumber()];
            const numEntries = reader.varint().toNumber();
            for (let j = 0; j < numEntries; j++) {
                const opName = {};
                if (this.version < 5) { // kNativePropertiesEncoding
                    opName.name = this.strings[reader.varint().toNumber()];
                    opName.dialect = dialect;
                } else {
                    const nameAndIsRegistered = reader.varint();
                    opName.name = this.strings[(nameAndIsRegistered >> 1n).toNumber()];
                    opName.dialect = dialect;
                    opName.isRegistered = (nameAndIsRegistered & 1n) === 1n;
                }
                if (numOps < 0) {
                    this.opNames.push(opName);
                } else {
                    this.opNames[i++] = opName;
                }
            }
        }
        if (reader.position !== section.end) {
            throw new mlir.Error(`Invalid dialect section size.`);
        }
    }

    _parseResourceSection() {
        const section = this.sections.get(6);
        const reader = this._reader;
        reader.seek(section.start);
        const numExternalResourceGroups = reader.varint().toNumber();
        if (numExternalResourceGroups > 0) {
            throw new mlir.Error(`Unsupported resource section.`);
        }
        /*
        for (let i = 0; i < numExternalResourceGroups; i++) {
            const numResources = reader.varint().toNumber();
            for (let j = 0; j < numResources; j++) {
                const resource = {};
                resource.key = this.strings[reader.varint().toNumber()];
                resource.offset = reader.varint().toNumber();
                resource.kind = reader.byte();
            }
        }
        */
        if (reader.position !== section.end) {
            throw new mlir.Error(`Invalid dialect section size.`);
        }
    }

    _parseAttrTypeSection() {
        const section = this.sections.get(3);
        const reader = this._reader;
        reader.seek(section.start);
        this.attributes = new Array(reader.varint().toNumber());
        this.types = new Array(reader.varint().toNumber());
        let offset = 0;
        const parseEntries = (range) => {
            for (let i = 0; i < range.length;) {
                const dialect = this.dialects[reader.varint().toNumber()];
                const numEntries = reader.varint().toNumber();
                for (let j = 0; j < numEntries; j++) {
                    const entry = {};
                    const entrySizeWithFlag = reader.varint();
                    entry.hasCustomEncoding = (entrySizeWithFlag & 1n) === 1n;
                    entry.size = (entrySizeWithFlag >> 1n).toNumber();
                    entry.offset = offset;
                    entry.dialect = dialect;
                    offset += entry.size;
                    range[i++] = entry;
                }
            }
        };
        parseEntries(this.attributes);
        parseEntries(this.types);
        if (reader.position !== section.end) {
            throw new mlir.Error(`Invalid dialect section size.`);
        }
        offset = this.sections.get(2).start;
        const parseCustomEntry = (entry, reader, entryType) => {
            // throw new mlir.Error(`Unsupported custom encoding.`);
            if (entryType === 'type') {
                // debugger;
            } else {
                // debugger;
            }
        };
        const parseAsmEntry = (entry, reader, entryType) => {
            if (entryType === 'type') {
                // debugger;
            } else {
                // debugger;
            }
        };
        const resolveEntries = (range, entryType) => {
            for (const entry of this.attributes) {
                reader.seek(offset + entry.offset);
                if (entry.hasCustomEncoding) {
                    parseCustomEntry(entry, reader);
                } else {
                    parseAsmEntry(entry, reader, entryType);
                }
                // if (reader.position !== (offset + entry.offset + entry.size)) {
                //     throw new mlir.Error(`Invalid '${entryType}' section size.`);
                // }
                // delete entry.offset;
                // delete entry.size;
            }
        };
        resolveEntries(this.attributes, 'attribute');
        resolveEntries(this.types, 'type');
    }

    _parsePropertiesSection() {
        const section = this.sections.get(8);
        const reader = this._reader;
        reader.seek(section.start);
        const count = reader.varint().toNumber();
        const offsetTable = new Array(count);
        for (let i = 0; i < offsetTable.length; i++) {
            const offset = reader.position;
            const size = reader.varint().toNumber();
            const data = reader.read(size);
            offsetTable[i] = { offset, data };
        }
        if (reader.position !== section.end) {
            throw new mlir.Error(`Invalid properties section size.`);
        }
    }
};

mlir.BinaryReader = class {

    constructor(reader) {
        this._reader = reader;
    }

    get length() {
        return this._reader.length;
    }

    get position() {
        return this._reader.position;
    }

    skip(length) {
        this._reader.skip(length);
    }

    seek(offset) {
        this._reader.seek(offset);
    }

    read(length) {
        return this._reader.read(length);
    }

    stream(length) {
        return this._reader.stream(length);
    }

    byte() {
        return this._reader.byte();
    }

    varint() {
        let result = this._reader.byte();
        if (result & 1) {
            return BigInt(result >> 1);
        }
        if (result === 0) {
            return this._reader.uint64();
        }
        result = BigInt(result);
        let mask = 1n;
        let numBytes = 0n;
        let shift = 8n;
        while (result > 0n && (result & mask) === 0n) {
            result |= (BigInt(this._reader.byte()) << shift);
            mask <<= 1n;
            shift += 8n;
            numBytes++;
        }
        result >>= BigInt(numBytes + 1n);
        return result;
    }

    string() {
        const reader = this._reader;
        let result = '';
        let value = -1;
        for (;;) {
            value = reader.byte();
            if (value === 0x00) {
                break;
            }
            result += String.fromCharCode(value);
        }
        return result;
    }
};

mlir.Utility = class {

    static dataType(value) {
        switch (value) {
            case 'index': return 'int64';
            case 'f16': return 'float16';
            case 'f32': return 'float32';
            case 'f64': return 'float64';
            case 'bf16': return 'bfloat16';
            case 'fp8': return 'float8';
            case 'fp8e4m3': return 'float8e4m3';
            case 'fp8_e4m3': return 'float8e4m3';
            case 'fp8e4m3fn': return 'float8e4m3fn';
            case 'fp8e5m2': return 'float8e5m2';
            case 'fp8_e5m2': return 'float8e5m2';
            case 'float8': return 'float8';
            case 'i1': return 'boolean';
            case 'i8': return 'int8';
            case 'i16': return 'int16';
            case 'i32': return 'int32';
            case 'i64': return 'int64';
            case 'si8': return 'int8';
            case 'si16': return 'int16';
            case 'si32': return 'int32';
            case 'si64': return 'int64';
            case 'ui1': return 'uint1';
            case 'ui8': return 'uint8';
            case 'ui16': return 'uint16';
            case 'ui32': return 'uint32';
            case 'ui64': return 'uint64';
            case 'b8': return 'int8';
            case 'boolean': return 'boolean';
            default:
                if (value && value.startsWith('!')) {
                    return value;
                }
                // Empty string can occur for operations with no inputs: () -> type
                // Just return empty string - it will be handled appropriately by caller
                if (value === '') {
                    return '';
                }
                throw new mlir.Error(`Unknown data type '${value}'.`);
        }
    }

    static valueType(type) {
        if (type === undefined) {
            return null;
        }
        // Handle dialect-specific types like !tosa.shape<3>, !quant.uniform<...>, etc.
        // These should be returned as-is without trying to decompose them
        if (type.startsWith('!') && !type.startsWith('!torch.vtensor<')) {
            return type;
        }
        // eg. tensor<?x3x2x2xf32>
        if (type.startsWith('tensor<') && type.endsWith('>')) {
            const spec = type.substring(7, type.length - 1).trim();
            if (spec.startsWith('!')) {
                return mlir.Utility.valueType(spec);
            }
            const index = spec.lastIndexOf('x');
            let dataType = spec;
            let shape = [];
            if (index > -1) {
                dataType = spec.substring(index + 1);
                if (!Number.isInteger(parseInt(dataType, 10))) {
                    shape = spec.substring(0, index).split('x').map((dim) => parseInt(dim, 10)).map((dim) => isNaN(dim) ? '?' : dim);
                    return new mlir.TensorType(dataType, new mlir.TensorShape(shape));
                }
            }
            return new mlir.TensorType(dataType, new mlir.TensorShape(shape));
        }
        if (type.startsWith('!torch.vtensor<') && type.endsWith('>')) {
            const spec = type.substring(15, type.length - 1);
            const index = spec.lastIndexOf(',');
            const shape = JSON.parse(spec.substring(0, index));
            const dataType = spec.substring(index + 1);
            return new mlir.TensorType(dataType, new mlir.TensorShape(shape));
        }
        return type;
    }
};

// Dialect Plugin System

mlir.AssemblyFormatParser = class {

    constructor(format) {
        this._format = format || '';
        this._pos = 0;
    }

    parse() {
        const directives = [];
        this._skipWhitespace();

        while (this._pos < this._format.length) {
            const directive = this._parseDirective();
            if (directive) {
                directives.push(directive);
            }
            this._skipWhitespace();
        }

        return directives;
    }

    _parseDirective() {
        const ch = this._format[this._pos];
        // Literal: `keyword`
        if (ch === '`') {
            this._pos++;
            const value = this._parseUntil('`');
            this._pos++; // skip closing `
            return { type: 'literal', value };
        }
        // Variable reference: $name
        if (ch === '$') {
            this._pos++;
            const name = this._parseIdentifier();
            return { type: 'operand_ref', name };
        }
        // Check for keywords
        const remaining = this._format.substring(this._pos);
        if (remaining.startsWith('operands')) {
            this._pos += 'operands'.length;
            return { type: 'operands' };
        }
        if (remaining.startsWith('attr-dict')) {
            this._pos += 'attr-dict'.length;
            return { type: 'attr_dict' };
        }
        if (remaining.startsWith('functional-type')) {
            this._pos += 'functional-type'.length;
            const args = this._parseParenList();
            return { type: 'functional_type', args };
        }
        if (remaining.startsWith('custom<')) {
            this._pos += 'custom<'.length;
            const parser = this._parseUntil('>');
            this._pos++; // consume '>'
            const args = this._parseParenList();
            return { type: 'custom', parser, args };
        }
        // Unknown - skip character
        this._pos++;
        return null;
    }

    _parseIdentifier() {
        let name = '';
        while (this._pos < this._format.length) {
            const ch = this._format[this._pos];
            if (/[a-zA-Z0-9_]/.test(ch)) {
                name += ch;
                this._pos++;
            } else {
                break;
            }
        }
        return name;
    }

    _parseUntil(terminator) {
        let value = '';
        while (this._pos < this._format.length && this._format[this._pos] !== terminator) {
            value += this._format[this._pos];
            this._pos++;
        }
        return value;
    }

    _parseParenList() {
        if (this._format[this._pos] !== '(') {
            return [];
        }
        this._pos++; // consume '('
        const items = [];
        while (this._pos < this._format.length && this._format[this._pos] !== ')') {
            this._skipWhitespace();
            const item = this._parseIdentifier();
            if (item) {
                items.push(item);
            }
            this._skipWhitespace();
            if (this._format[this._pos] === ',') {
                this._pos++;
            }
        }
        if (this._format[this._pos] === ')') {
            this._pos++;
        }
        return items;
    }

    _skipWhitespace() {
        while (this._pos < this._format.length && /\s/.test(this._format[this._pos])) {
            this._pos++;
        }
    }
};

mlir.Dialect = class {

    constructor(name, operations) {
        this._name = name;
        this._operations = new Map();
        for (const op of operations) {
            if (op.assemblyFormat) {
                const parser = new mlir.AssemblyFormatParser(op.assemblyFormat);
                const directives = parser.parse();
                this._operations.set(op.name, { metadata: op, directives });
            }
        }
    }

    parseOperation(parser, opName, op) {
        const name = opName.replace(/^"|"$/g, '');
        const opInfo = this._operations.get(name);
        if (!opInfo) {
            return false;
        }
        const { directives } = opInfo;
        for (const directive of directives) {
            switch (directive.type) {
                case 'literal':
                    if (parser._match('id') && parser._token.value === directive.value) {
                        parser._read('id');
                    }
                    break;
                case 'operands':
                    op.operands = parser._parseArguments();
                    break;
                case 'attr_dict':
                    parser.parseAttributeDict(op.attributes);
                    break;
                case 'functional_type':
                    if (parser._eat(':')) {
                        parser._parseArgumentTypes(op.operands);
                    }
                    if (parser._eat('->') || parser._eat('id', 'to')) {
                        if (op.results.length > 0) {
                            parser._parseArgumentTypes(op.results);
                        } else {
                            op.results = parser._parseArguments();
                        }
                    }
                    break;
                case 'custom':
                    this._skipCustomParser(parser, directive.parser);
                    break;
                default:
                    throw new mlir.Error(`Unknown assembly format directive type '${directive.type}'.`);
            }
        }
        return true;
    }

    _skipCustomParser(parser, parserName) {
        if (parserName === 'ConvolutionDimensions') {
            // dim_numbers = [...]x[...]->[...]
            if (parser._match('[')) {
                parser._skipSymbolBetween('[', ']');
            }
            if (parser._match('id') && parser._token.value === 'x') {
                parser._read('id');
                if (parser._match('[')) {
                    parser._skipSymbolBetween('[', ']');
                }
            }
            if (parser._eat('->')) {
                if (parser._match('[')) {
                    parser._skipSymbolBetween('[', ']');
                }
            }
        } else if (parserName === 'WindowAttributes') {
            // Handled by normal attr-dict parsing
        } else if (parser._match('[')) {
            parser._skipSymbolBetween('[', ']');
        } else if (parser._match('{')) {
            parser._skipSymbolBetween('{', '}');
        }
    }
};

mlir.StableHLODialect = class extends mlir.Dialect {

    constructor(operations) {
        operations = operations.filter((op) => op.name && op.name.startsWith('stablehlo.'));
        super('stablehlo', operations);
    }

    parseOperation(parser, opName, op) {
        const name = opName.replace(/^"|"$/g, '');
        // For reduce/scan operations with custom form, try custom parsing
        // Custom form: stablehlo.reduce(%input init: %init) across dimensions = [...]
        // Generic form: "stablehlo.reduce"(...) ({ ... }) {...} - let default handle this
        if ((name === 'stablehlo.reduce' || name === 'stablehlo.scan') && parser._match('(')) {
            return this._parseReduceLikeOp(parser, op);
        }
        // Special handling for convolution - has custom attributes that assemblyFormat parser doesn't handle well
        if (name === 'stablehlo.convolution' && parser._match('(')) {
            return this._parseConvolutionOp(parser, op);
        }
        // Try assemblyFormat-based parsing
        return super.parseOperation(parser, opName, op);
    }

    _parseConvolutionOp(parser, op) {
        // Parse operands: (%input, %filter)
        op.operands = parser._parseArguments();
        // Skip dim_numbers = [...]
        if (parser._match('id', 'dim_numbers')) {
            parser._read('id');
            parser._read('=');
            // Skip the dim_numbers value - it's a complex expression like [b,0,1,f]x[0,1,i,o]->[b,0,1,f]
            // It ends when we see a comma at depth 0, then check if next is 'window' or '{'
            let depth = 0;
            while (parser._token.kind !== 'eof') {
                if (parser._token.value === '[') {
                    depth++;
                    parser._read();
                } else if (parser._token.value === ']') {
                    depth--;
                    parser._read();
                } else if (depth === 0 && parser._token.value === ',') {
                    parser._read(',');
                    // Check if we've reached the end of dim_numbers
                    if (parser._match('id', 'window') || parser._match('{')) {
                        break;
                    }
                    // Otherwise, continue - this comma is part of the dim_numbers expression
                } else {
                    parser._read();
                }
            }
        }
        // Skip window = {...}
        if (parser._match('id', 'window')) {
            parser._read('id');
            parser._read('=');
            parser._skipSymbolBetween('{', '}');
            parser._eat(',');
        }
        // Parse attribute dict {batch_group_count = ..., feature_group_count = ...}
        if (parser._match('{')) {
            parser.parseAttributeDict(op.attributes);
        }
        // Parse type signature : (...) -> (...)
        if (parser._eat(':')) {
            parser._parseArgumentTypes(op.operands);
        }
        if (parser._eat('->') || parser._eat('id', 'to')) {
            if (op.results.length > 0) {
                parser._parseArgumentTypes(op.results);
            } else {
                op.results = parser._parseArguments();
            }
        }
        return true;
    }

    _parseReduceLikeOp(parser, op) {
        // This handles the custom form: stablehlo.reduce(%input init: %init) across dimensions = [...]
        // For the generic string form: "stablehlo.reduce"(...) ({...}) {...}, return false
        // so the default parsing handles the region-list and attributes
        op.operands = parser._parseArguments();
        // Check if this is the generic form with parenthesized region-list
        // Generic form: arguments followed by `(` for region-list
        // Custom form: arguments followed by `across` or `:` or identifier for labeled region
        if (parser._match('(')) {
            // Generic form with parenthesized region-list: ({ ... })
            if (parser._eat('(') && parser._match('{')) {
                let regionCount = 0;
                while (!parser._match(')')) {
                    if (regionCount++ > 10) {
                        throw new Error(`Too many regions in region-list (>10) - possible infinite loop at ${parser._tokenizer.location()}, current token: '${parser._token.value}'`);
                    }
                    if (!parser._match('{')) {
                        throw new Error(`Expected '{' for region in region-list, got '${parser._token.value}' at ${parser._tokenizer.location()}`);
                    }
                    const region = {};
                    parser.parseRegion(region);
                    op.regions.push(region);
                    if (!parser._eat(',') && !parser._match(')')) {
                        throw new Error(`Expected ',' or ')' after region, got '${parser._token.value}' at ${parser._tokenizer.location()}`);
                    }
                }
                parser._read(')');
            }

            // Parse attributes dictionary { dimensions = ... }
            if (parser._match('{')) {
                parser.parseAttributeDict(op.attributes);
            }

            // Parse type signature : (...) -> (...)
            if (parser._eat(':')) {
                parser._parseArgumentTypes(op.operands);
            }

            if (parser._eat('->') || parser._eat('id', 'to')) {
                if (op.results.length > 0) {
                    parser._parseArgumentTypes(op.results);
                } else {
                    op.results = parser._parseArguments();
                }
            }

            return true;
        }

        // Handle "across dimensions = [...]"
        if (parser._eat('id', 'across')) {
            if (parser._eat('id', 'dimensions')) {
                parser._read('=');
                parser._skipSymbolBetween('[', ']');
            }
        }

        // Type signature
        if (parser._eat(':')) {
            parser._parseArgumentTypes(op.operands);
        }

        if (parser._eat('->') || parser._eat('id', 'to')) {
            if (op.results.length > 0) {
                parser._parseArgumentTypes(op.results);
            } else {
                op.results = parser._parseArguments();
            }
        }

        // Handle regions
        if (parser._match('id') && !parser._match('keyword', 'loc')) {
            // Labeled region: reducer(...) { ... }
            const label = parser._read('id');
            const region = { blocks: [] };
            const block = { operations: [], arguments: [], name: label };

            if (parser._eat('(')) {
                while (!parser._eat(')')) {
                    const value = parser._read('%');
                    parser._read(':');
                    const type = parser._parseType();
                    block.arguments.push({ value, type });
                    parser._eat(',');
                }
            }

            parser._read('{');
            while (!parser._eat('}')) {
                const innerOp = parser.parseOperation();
                block.operations.push(innerOp);
            }

            block.loc = parser._parseLocation();
            region.blocks.push(block);
            op.regions.push(region);
        } else if (parser._eat('(') && parser._match('{')) {
            // Parenthesized region list: ({ ... })
            const region = {};
            parser.parseRegion(region);
            op.regions.push(region);
            parser._read(')');
        } else if (parser._match('{')) {
            // Simple region: { ... }
            const region = {};
            parser.parseRegion(region);
            op.regions.push(region);
        }

        return true;
    }
};

mlir.AffineDialect = class extends mlir.Dialect {

    constructor(operations) {
        operations = operations.filter((op) => op.name && op.name.startsWith('affine.'));
        super('affine', operations);
    }

    parseOperation(parser, opName, op) {
        const name = opName.replace(/^"|"$/g, '');
        // Special handling for affine.if - has condition before region
        if (name === 'affine.if') {
            // affine.if #set(...) { region }
            if (parser._match('#')) {
                const condition = parser._parseValue();
                op.attributes.push({ name: 'condition', value: condition });
            }
            const region = {};
            parser.parseRegion(region);
            op.regions.push(region);
            if (parser._match('id', 'else')) {
                parser._read('id', 'else');
                const elseRegion = {};
                parser.parseRegion(elseRegion);
                op.regions.push(elseRegion);
            }
            return true;
        }
        // Special handling for affine.apply and affine.min
        if (name === 'affine.apply' || name === 'affine.min') {
            // affine.apply and affine.min have special syntax:
            // affine.apply #map(args) or affine.apply affine_map<...>()[args]
            // No type annotations, just the map application
            if (parser._match('#') || parser._match('id', 'affine_map') || parser._match('id', 'affine_set')) {
                const value = parser._parseValue();
                op.attributes.push({ name: 'map', value });
            }
            return true;
        }
        if (name === 'affine.store') {
            return this._parseStoreOp(parser, op);
        }
        if (name === 'affine.load') {
            return this._parseLoadOp(parser, op);
        }
        return super.parseOperation(parser, name, op);
    }

    _parseStoreOp(parser, op) {
        if (parser._match('%')) {
            const value = parser._read('%');
            op.operands.push({ value });
        } else {
            const value = parser._parseValue();
            op.operands.push(value);
        }
        if (!parser._eat('id', 'to')) {
            parser._eat(',');
        }
        const address = parser._read('%');
        op.operands.push({ value: address });
        parser._skipSymbolBetween('[', ']');
        if (parser._eat(':')) {
            const type = parser._parseType();
            op.operands[1].type = type;
        }
        return true;
    }

    _parseLoadOp(parser, op) {
        const address = parser._read('%');
        op.operands.push({ value: address });
        parser._skipSymbolBetween('[', ']');
        if (parser._eat(':')) {
            const type = parser._parseType();
            op.operands[0].type = type;
        }
        return true;
    }
};

mlir.FuncDialect = class extends mlir.Dialect {

    constructor(operations) {
        operations = operations.filter((op) => op.name && op.name.startsWith('func.'));
        super('func', operations);
    }
};

mlir.MemRefDialect = class extends mlir.Dialect {

    constructor(operations) {
        operations = operations.filter((op) => op.name && op.name.startsWith('memref.'));
        super('memref', operations);
    }

    parseOperation(parser, opName, op) {
        const name = opName.replace(/^"|"$/g, '');
        if (name === 'memref.store') {
            return this._parseStoreOp(parser, op);
        }
        if (name === 'memref.load') {
            return this._parseLoadOp(parser, op);
        }
        return super.parseOperation(parser, opName, op);
    }

    _parseStoreOp(parser, op) {
        if (parser._match('%')) {
            const value = parser._read('%');
            op.operands.push({ value });
        } else {
            const value = parser._parseValue();
            op.operands.push(value);
        }
        if (!parser._eat('id', 'to')) {
            parser._eat(',');
        }
        const address = parser._read('%');
        op.operands.push({ value: address });
        parser._skipSymbolBetween('[', ']');
        if (parser._eat(':')) {
            const type = parser._parseType();
            op.operands[1].type = type;
        }
        return true;
    }

    _parseLoadOp(parser, op) {
        const address = parser._read('%');
        op.operands.push({ value: address });
        parser._skipSymbolBetween('[', ']');
        if (parser._eat(':')) {
            const type = parser._parseType();
            op.operands[0].type = type;
        }
        return true;
    }
};

mlir.VectorDialect = class extends mlir.Dialect {

    constructor(operations) {
        operations = operations.filter((op) => op.name && op.name.startsWith('vector.'));
        super('vector', operations);
    }

    parseOperation(parser, opName, op) {
        const name = opName.replace(/^"|"$/g, '');
        if (name === 'vector.load') {
            return this._parseLoadOp(parser, op);
        }
        if (name === 'vector.store') {
            return this._parseStoreOp(parser, op);
        }
        if (name === 'vector.transfer_read' || name === 'vector.transfer_write') {
            return this._parseTransferOp(parser, op);
        }
        if (name === 'vector.insert' || name === 'vector.extract') {
            return this._parseInsertExtractOp(parser, op);
        }
        return super.parseOperation(parser, opName, op);
    }

    _parseLoadOp(parser, op) {
        const address = parser._read('%');
        op.operands.push({ value: address });
        parser._skipSymbolBetween('[', ']');
        if (parser._eat(':')) {
            const operandType = parser._parseType();
            op.operands[0].type = operandType;
            parser._eat(',');
            const resultType = parser._parseType();
            if (op.results.length > 0) {
                op.results[0].type = resultType;
            }
        }
        return true;
    }

    _parseStoreOp(parser, op) {
        if (parser._match('%')) {
            const value = parser._read('%');
            op.operands.push({ value });
        } else {
            const value = parser._parseValue();
            op.operands.push(value);
        }
        parser._eat(',');
        const address = parser._read('%');
        op.operands.push({ value: address });
        parser._skipSymbolBetween('[', ']');
        if (parser._eat(':')) {
            const memrefType = parser._parseType();
            op.operands[1].type = memrefType;
            parser._eat(',');
            const vectorType = parser._parseType();
            op.operands[0].type = vectorType;
        }
        return true;
    }

    _parseTransferOp(parser, op) {
        // Parse: vector.transfer_read %source[%i, %j, ...], %padding {attrs} : memref_type, vector_type
        //    or: vector.transfer_write %value, %dest[%i, %j, ...] {attrs} : vector_type, memref_type

        // First operand: source/value
        const first = parser._read('%');
        op.operands.push({ value: first });

        // Check if indices follow first operand or second operand
        const hasIndicesAfterFirst = parser._match('[');
        if (hasIndicesAfterFirst) {
            parser._skipSymbolBetween('[', ']');
        }

        // Comma
        parser._eat(',');

        // Second operand: padding value or destination
        const second = parser._read('%');
        op.operands.push({ value: second });

        // If indices didn't follow first operand, they follow second operand
        if (!hasIndicesAfterFirst && parser._match('[')) {
            parser._skipSymbolBetween('[', ']');
        }

        // Optional attribute dictionary
        if (parser._match('{')) {
            parser.parseAttributeDict(op.attributes);
        }

        // Type signature: : memref_type, vector_type
        if (parser._eat(':')) {
            const type1 = parser._parseType();
            op.operands[0].type = type1;
            parser._eat(',');
            const type2 = parser._parseType();
            // For transfer_read, type2 is the result type
            // For transfer_write, type2 is just the vector type
            if (op.results.length > 0) {
                op.results[0].type = type2;
            }
            op.operands[1].type = type2;
        }

        return true;
    }

    _parseInsertExtractOp(parser, op) {
        // Parse: vector.insert %source, %dest [position] : source_type into dest_type
        //    or: vector.extract %source [position] : source_type

        // Parse operands (%source, %dest or just %source)
        op.operands = parser._parseArguments();

        // Parse position [...]
        if (parser._match('[')) {
            parser._skipSymbolBetween('[', ']');
        }

        // Parse type signature
        if (parser._eat(':')) {
            parser._parseArgumentTypes(op.operands);
        }

        // For insert, handle 'into' keyword followed by destination type
        if (parser._eat('id', 'into')) {
            const destType = parser._parseType();
            // Result type for insert is the destination type
            if (op.results.length > 0) {
                op.results[0].type = destType;
            }
        } else if (parser._eat('->') || parser._eat('id', 'to')) {
            // For extract, normal result type
            if (op.results.length > 0) {
                parser._parseArgumentTypes(op.results);
            } else {
                const resultType = parser._parseType();
                op.results.push({ type: resultType });
            }
        }

        return true;
    }
};

mlir.ONNXDialect = class extends mlir.Dialect {

    constructor(operations) {
        operations = operations.filter((op) => op.name && op.name.startsWith('onnx.'));
        super('onnx', operations);
    }
};

mlir.TorchDialect = class extends mlir.Dialect {

    constructor(operations) {
        operations = operations.filter((op) => op.name && op.name.startsWith('torch.'));
        super('torch', operations);
    }

    parseOperation(parser, opName, op) {
        const name = opName.replace(/^"|"$/g, '');
        if (name.startsWith('torch.constant.')) {
            return this._parseConstantOp(parser, name, op);
        }
        return super.parseOperation(parser, opName, op);
    }

    _parseConstantOp(parser, opName, op) {
        if (opName === 'torch.constant.int') {
            if (parser._match('int')) {
                const value = parser._read('int');
                op.attributes.push({ name: 'value', value });
            }
        } else if (opName === 'torch.constant.bool') {
            if (parser._match('boolean')) {
                const value = parser._read('boolean');
                op.attributes.push({ name: 'value', value });
            }
        } else if (opName === 'torch.constant.none') {
            // torch.constant.none has no operands or attributes
        } else if (opName === 'torch.constant.str') {
            if (parser._match('string')) {
                const value = parser._read('string');
                op.attributes.push({ name: 'value', value });
            }
        } else if (opName === 'torch.constant.float') {
            if (parser._match('float')) {
                const value = parser._read('float');
                op.attributes.push({ name: 'value', value });
            }
        }
        return true;
    }
};

mlir.HALDialect = class extends mlir.Dialect {

    constructor(operations) {
        operations = operations.filter((op) => op.name && op.name.startsWith('hal.'));
        super('hal', operations);
    }

    parseOperation(parser, opName, op) {
        const name = opName.replace(/^"|"$/g, '');
        // Handle operations with visibility + symbol (similar to flow dialect)
        if (name === 'hal.executable' || name === 'hal.interface') {
            if (parser._match('id', 'private') || parser._match('id', 'public') || parser._match('id', 'nested')) {
                parser._read('id');
            }
            if (parser._match('@')) {
                const symbol = parser._read('@');
                op.attributes.push({ name: 'sym_name', value: symbol });
            }
            if (parser._match('id', 'attributes')) {
                parser._read('id', 'attributes');
                parser.parseAttributeDict(op.attributes);
            }
            if (parser._match('{')) {
                const region = {};
                parser.parseRegion(region);
                op.regions.push(region);
            }
            return true;
        }
        // Handle operations with named parameters: hal.interface.binding, hal.executable.variant, etc.
        if (name === 'hal.interface.binding' || name === 'hal.executable.variant' || name === 'hal.executable.entry_point') {
            if (parser._match('id', 'private') || parser._match('id', 'public') || parser._match('id', 'nested')) {
                parser._read('id');
            }
            if (parser._match('@')) {
                const symbol = parser._read('@');
                op.attributes.push({ name: 'sym_name', value: symbol });
            }
            // Parse comma-separated named parameters
            while (parser._eat(',')) {
                if (parser._match('id')) {
                    parser._read('id');
                    if (parser._eat('=')) {
                        // Skip the value - could be complex attribute
                        if (parser._match('#')) {
                            parser._parseValue();
                        } else {
                            parser._read();
                        }
                    }
                }
            }
            // Parse attributes dict if present
            if (parser._match('id', 'attributes')) {
                parser._read('id', 'attributes');
                parser.parseAttributeDict(op.attributes);
            }
            // Parse region if present
            if (parser._match('{')) {
                const region = {};
                parser.parseRegion(region);
                op.regions.push(region);
            }
            return true;
        }
        return super.parseOperation(parser, opName, op);
    }
};

mlir.MHLODialect = class extends mlir.Dialect {

    constructor(operations) {
        operations = operations.filter((op) => op.name && op.name.startsWith('mhlo.'));
        super('mhlo', operations);
    }

    parseOperation(parser, opName, op) {
        const name = opName.replace(/^"|"$/g, '');
        // Special handling for convolution - has custom attributes that assemblyFormat parser doesn't handle well
        if (name === 'mhlo.convolution' && parser._match('(')) {
            return this._parseConvolutionOp(parser, op);
        }
        // Try assemblyFormat-based parsing
        return super.parseOperation(parser, opName, op);
    }

    _parseConvolutionOp(parser, op) {
        // Parse operands: (%input, %filter)
        op.operands = parser._parseArguments();
        // Skip dim_numbers = [...]
        if (parser._match('id', 'dim_numbers')) {
            parser._read('id');
            parser._read('=');
            // Skip the dim_numbers value - it's a complex expression like [b,0,1,f]x[0,1,i,o]->[b,0,1,f]
            // It ends when we see a comma at depth 0, then check if next is 'window' or '{'
            let depth = 0;
            while (parser._token.kind !== 'eof') {
                if (parser._token.value === '[') {
                    depth++;
                    parser._read();
                } else if (parser._token.value === ']') {
                    depth--;
                    parser._read();
                } else if (depth === 0 && parser._token.value === ',') {
                    parser._read(',');
                    // Check if we've reached the end of dim_numbers
                    if (parser._match('id', 'window') || parser._match('{')) {
                        break;
                    }
                    // Otherwise, continue - this comma is part of the dim_numbers expression
                } else {
                    parser._read();
                }
            }
        }
        // Skip window = {...}
        if (parser._match('id', 'window')) {
            parser._read('id');
            parser._read('=');
            parser._skipSymbolBetween('{', '}');
            parser._eat(',');
        }
        // Parse attribute dict {batch_group_count = ..., feature_group_count = ...}
        if (parser._match('{')) {
            parser.parseAttributeDict(op.attributes);
        }
        // Parse type signature : (...) -> (...)
        if (parser._eat(':')) {
            parser._parseArgumentTypes(op.operands);
        }
        if (parser._eat('->') || parser._eat('id', 'to')) {
            if (op.results.length > 0) {
                parser._parseArgumentTypes(op.results);
            } else {
                op.results = parser._parseArguments();
            }
        }
        return true;
    }
};

mlir.FlowDialect = class extends mlir.Dialect {

    constructor(operations) {
        operations = operations.filter((op) => op.name && op.name.startsWith('flow.'));
        super('flow', operations);
    }

    parseOperation(parser, opName, op) {
        const name = opName.replace(/^"|"$/g, '');
        // Handle operations with custom syntax not in schema or using complex custom parsers
        if ((name === 'flow.dispatch.workgroups' && parser._match('[')) || name === 'flow.ex.stream.fragment') {
            return this._parseDispatchWorkgroupsOp(parser, op);
        }
        if (name === 'flow.dispatch.tensor.load' || name === 'flow.dispatch.tensor.store') {
            return this._parseTensorLoadStoreOp(parser, op);
        }
        // flow.dispatch has complex symbol references not handled by default parser
        if (name === 'flow.dispatch') {
            return this._parseDispatchOp(parser, op);
        }
        // Handle operations with visibility + symbol that aren't in schema or need manual parsing
        if (name === 'flow.executable' || name === 'flow.dispatch.entry') {
            if (parser._match('id', 'private') || parser._match('id', 'public') || parser._match('id', 'nested')) {
                parser._read('id');
            }
            if (parser._match('@')) {
                const symbol = parser._read('@');
                op.attributes.push({ name: 'sym_name', value: symbol });
            }
            if (parser._match('id', 'attributes')) {
                parser._read('id', 'attributes');
                parser.parseAttributeDict(op.attributes);
            }
            if (parser._match('{')) {
                const region = {};
                parser.parseRegion(region);
                op.regions.push(region);
            }
            return true;
        }
        return super.parseOperation(parser, opName, op);
    }

    _parseDispatchWorkgroupsOp(parser, op) {
        // Parse subscript values: [%c32, %c112, %c112]
        if (parser._eat('[')) {
            while (!parser._eat(']')) {
                parser._read(); // read subscript value
                parser._eat(',');
            }
        }
        // Parse operands: (%0, %1)
        op.operands = parser._parseArguments();
        // Parse type signature : (...) -> (...)
        if (parser._eat(':')) {
            parser._parseArgumentTypes(op.operands);
        }
        if (parser._eat('->') || parser._eat('id', 'to')) {
            if (op.results.length > 0) {
                parser._parseArgumentTypes(op.results);
            } else {
                op.results = parser._parseArguments();
            }
        }
        // Parse region with arguments: = (%arg2: type, %arg3: type) { ... }
        if (parser._eat('=')) {
            const region = {};
            region.blocks = [];
            const block = {};
            block.operations = [];
            block.arguments = [];
            // Parse region arguments
            if (parser._eat('(')) {
                while (!parser._eat(')')) {
                    const value = parser._read('%');
                    parser._read(':');
                    const type = parser._parseType();
                    block.arguments.push({ value, type });
                    parser._eat(',');
                }
            }
            // Some operations like flow.ex.stream.fragment have -> type after region args
            if (parser._eat('->') || parser._eat('id', 'to')) {
                parser._parseType();
            }
            // Parse region body
            parser.parseBlock(block);
            region.blocks.push(block);
            op.regions.push(region);
        }
        return true;
    }

    _parseDispatchOp(parser, op) {
        // flow.dispatch @symbol::@entry[subscripts](operands) : types -> type
        if (parser._match('@')) {
            const symbol = parser._read('@');
            op.attributes.push({ name: 'entry_point', value: symbol });
            // Handle :: nested symbol
            if (parser._eat('id', '::') || (parser._match(':') && parser._eat(':') && parser._eat(':'))) {
                if (parser._match('@')) {
                    const nested = parser._read('@');
                    op.attributes[op.attributes.length - 1].value += `::${nested}`;
                }
            }
        }
        // Parse subscripts [...]
        if (parser._eat('[')) {
            while (!parser._eat(']')) {
                parser._read();
                parser._eat(',');
            }
        }
        // Parse operands
        op.operands = parser._parseArguments();
        // Parse optional attribute dictionary before type signature
        if (parser._match('{')) {
            parser.parseAttributeDict(op.attributes);
        }
        // Parse type signature
        if (parser._eat(':')) {
            parser._parseArgumentTypes(op.operands);
        }
        if (parser._eat('->') || parser._eat('id', 'to')) {
            if (op.results.length > 0) {
                parser._parseArgumentTypes(op.results);
            } else {
                op.results = parser._parseArguments();
            }
        }
        return true;
    }

    _parseTensorLoadStoreOp(parser, op) {
        // Parse: load %arg2, offsets = [...] : type -> type
        //    or: store %26, %arg4, offsets = [...] : type -> type
        // Parse operands: one or more % values separated by commas
        while (parser._match('%')) {
            const value = parser._read('%');
            op.operands.push({ value });
            // If next is not a comma, break
            if (!parser._eat(',')) {
                break;
            }
            // If after comma, next is not %, break (we've hit named parameters)
            if (!parser._match('%')) {
                // We have a comma followed by non-%, so continue to named parameters
                break;
            }
        }
        // At this point, if we broke because of named params, we've already consumed the comma
        // Parse comma-separated named parameters: offsets = [...], sizes = [...], strides = [...]
        // Note: first parameter might not need comma-eating if we just broke from operand loop
        let needComma = !parser._match('id'); // If we're not at 'id', we need to eat commas
        while (needComma ? parser._eat(',') : true) {
            needComma = true; // After first iteration, always need comma
            if (parser._match('id')) {
                const paramName = parser._read('id');
                if (parser._eat('=')) {
                    // Skip the parameter value (usually an array)
                    if (parser._match('[')) {
                        parser._skipSymbolBetween('[', ']');
                    } else {
                        parser._read(); // Read single value
                    }
                    op.attributes.push({ name: paramName, value: paramName });
                }
            } else {
                break;
            }
        }
        // Parse type signature : type -> type
        if (parser._eat(':')) {
            parser._parseArgumentTypes(op.operands);
        }
        // For tensor.load, there's a -> result type
        // For tensor.store, the -> is followed by the output tensor type (not a result)
        if (parser._eat('->') || parser._eat('id', 'to')) {
            // Just skip the type - we don't need to parse it as results
            parser._parseType();
        }
        return true;
    }
};

mlir.LinalgDialect = class extends mlir.Dialect {

    constructor(operations) {
        operations = operations.filter((op) => op.name && op.name.startsWith('linalg.'));
        super('linalg', operations);
    }

    parseOperation(parser, opName, op) {
        const name = opName.replace(/^"|"$/g, '');
        // Many linalg operations (especially named structured ops) follow the pattern:
        // linalg.<op_name> {attributes} ins(...) outs(...) [-> type]
        // These named structured ops are generated at LLVM build time from YAML
        // and won't be in our schema, so we need custom parsing for them.

        // First try default parsing (for operations in schema)
        const hasSchemaEntry = this._operations.has(name);
        if (hasSchemaEntry) {
            return super.parseOperation(parser, opName, op);
        }

        // Not in schema - try to parse as ins/outs operation
        // These are named structured ops generated from YAML at LLVM build time
        if (parser._match('{') || parser._match('id', 'ins')) {
            return this._parseInsOutsOp(parser, op);
        }

        // Can't parse this operation
        return false;
    }

    _parseInsOutsOp(parser, op) {
        // Parse: linalg.op {attrs} ins(%0, %1 : type, type) outs(%2 : type) [-> type]

        // Parse optional attribute dictionary
        if (parser._match('{')) {
            parser.parseAttributeDict(op.attributes);
        }

        // Parse 'ins' section
        if (parser._eat('id', 'ins')) {
            if (!parser._eat('(')) {
                return false;
            }
            // Parse operands: %0, %1
            while (parser._match('%')) {
                const value = parser._read('%');
                op.operands.push({ value });
                if (!parser._eat(',')) {
                    break;
                }
            }
            // Parse types: : type1, type2
            if (parser._eat(':')) {
                let idx = 0;
                const startIdx = 0;
                while (!parser._match(')')) {
                    const type = parser._parseType();
                    if (startIdx + idx < op.operands.length) {
                        op.operands[startIdx + idx].type = type;
                    }
                    idx++;
                    if (!parser._eat(',')) {
                        break;
                    }
                }
            }
            if (!parser._eat(')')) {
                return false;
            }
        }

        // Parse 'outs' section
        if (parser._eat('id', 'outs')) {
            if (!parser._eat('(')) {
                return false;
            }
            const outsStart = op.operands.length;
            // Parse operands: %2, %3
            while (parser._match('%')) {
                const value = parser._read('%');
                op.operands.push({ value });
                if (!parser._eat(',')) {
                    break;
                }
            }
            // Parse types: : type1, type2
            if (parser._eat(':')) {
                let idx = 0;
                while (!parser._match(')')) {
                    const type = parser._parseType();
                    if (outsStart + idx < op.operands.length) {
                        op.operands[outsStart + idx].type = type;
                    }
                    idx++;
                    if (!parser._eat(',')) {
                        break;
                    }
                }
            }
            if (!parser._eat(')')) {
                return false;
            }
        }

        // Some linalg operations may have a region after the signature
        if (parser._match('{')) {
            const region = {};
            parser.parseRegion(region);
            op.regions.push(region);
        }

        // Parse optional return type (can come after region for linalg.generic)
        if (parser._eat('->') || parser._eat('id', 'to')) {
            if (op.results.length > 0) {
                parser._parseArgumentTypes(op.results);
            } else {
                const type = parser._parseType();
                op.results.push({ type });
            }
        }

        return true;
    }
};

mlir.QuantDialect = class extends mlir.Dialect {

    constructor(operations) {
        operations = operations.filter((op) => op.name && op.name.startsWith('quant.'));
        super('quant', operations);
    }
};

mlir.TensorDialect = class extends mlir.Dialect {

    constructor(operations) {
        operations = operations.filter((op) => op.name && op.name.startsWith('tensor.'));
        super('tensor', operations);
    }
};

mlir.TFDialect = class extends mlir.Dialect {

    constructor(operations) {
        operations = operations.filter((op) => op.name && op.name.startsWith('tf.'));
        super('tf', operations);
    }
};

mlir.TFLDialect = class extends mlir.Dialect {

    constructor(operations) {
        operations = operations.filter((op) => op.name && op.name.startsWith('tfl.'));
        super('tfl', operations);
    }
};

mlir.IRDLDialect = class extends mlir.Dialect {

    constructor(operations) {
        operations = operations.filter((op) => op.name && op.name.startsWith('irdl.'));
        super('irdl', operations);
    }

    parseOperation(parser, opName, op) {
        const name = opName.replace(/^"|"$/g, '');
        if (name === 'irdl.operands' || name === 'irdl.results' ||
            name === 'irdl.parameters' || name === 'irdl.attributes' ||
            name === 'irdl.regions') {
            if (parser._match('(')) {
                parser._read('(');
                while (!parser._eat(')')) {
                    if (parser._match('id') || parser._match('string')) {
                        const paramName = parser._read();
                        parser._read(':');
                        const paramValue = parser._read(); // Read the SSA value like %tensor
                        op.attributes.push({ name: paramName, value: paramValue });
                    }
                    parser._eat(',');
                }
            }
            op.loc = parser._parseLocation();
            return true;
        }

        // Use default parsing for other IRDL operations
        return super.parseOperation(parser, opName, op);
    }
};

mlir.SPIRVDialect = class extends mlir.Dialect {

    constructor(operations) {
        // Filter for spirv.* operations (official MLIR uses 'spirv' dialect name)
        operations = operations.filter((op) => op.name && op.name.startsWith('spirv.'));
        // Note: This dialect handles both 'spirv.' and 'spv.' prefixes for compatibility
        super('spirv', operations);
    }

    parseOperation(parser, opName, op) {
        const name = opName.replace(/^"|"$/g, '');
        // spirv.module / spv.module has addressing model and memory model before the region
        if (name === 'spirv.module' || name === 'spv.module') {
            // Parse: spv.module Logical GLSL450 { ... }
            // Read addressing model (Logical, Physical32, Physical64, etc.)
            if (parser._match('id')) {
                const addressingModel = parser._read('id');
                op.attributes.push({ name: 'addressing_model', value: addressingModel });
            }
            // Read memory model (GLSL450, Vulkan, OpenCL, etc.)
            if (parser._match('id')) {
                const memoryModel = parser._read('id');
                op.attributes.push({ name: 'memory_model', value: memoryModel });
            }
            // Parse region
            if (parser._match('{')) {
                const region = {};
                parser.parseRegion(region);
                op.regions.push(region);
            }
            return true;
        }
        // spirv.func / spv.func has symbol, function type, and optional control string
        if (name === 'spirv.func' || name === 'spv.func') {
            // Parse: spirv.func @symbol() "None" attributes {...} { ... }
            //    or: spirv.func @symbol(%arg: type) -> type "None" { ... }

            // Parse symbol (@name)
            if (parser._match('@')) {
                const symbol = parser._read('@');
                op.attributes.push({ name: 'sym_name', value: symbol });
            }

            // Create function_type structure
            const function_type = {
                inputs: [],
                results: []
            };

            // Parse function signature (arguments and return type)
            if (parser._match('(')) {
                function_type.inputs = parser.parseFunctionArgumentList();
            }

            // Parse return type if present
            if (parser._eat('->')) {
                function_type.results = parser.parseFunctionResultList();
            }

            op.attributes.push({ name: 'function_type', value: function_type });

            // Parse control string ("None", "Inline", "DontInline", etc.)
            if (parser._match('string')) {
                const control = parser._read('string');
                op.attributes.push({ name: 'function_control', value: control });
            }

            // Parse optional attributes
            if (parser._match('id', 'attributes')) {
                parser._read('id', 'attributes');
                parser.parseAttributeDict(op.attributes);
            }

            // Parse region
            if (parser._match('{')) {
                const region = {};
                parser.parseRegion(region);
                op.regions.push(region);
            }
            return true;
        }
        // spirv.mlir.loop / spv.mlir.loop - region with multi-block control flow
        // spirv.mlir.selection / spv.mlir.selection - region with multi-block control flow
        if (name === 'spirv.mlir.loop' || name === 'spv.mlir.loop' ||
            name === 'spirv.mlir.selection' || name === 'spv.mlir.selection') {
            // These operations have regions that are parsed by the generic parser
            // Add sentinel attribute to manipulate the heuristic at line 1226:
            // Setting op.attributes.length > 0 forces the else branch (region parsing)
            op.attributes.push({ name: '_has_region', value: true });
            // Return false to let the generic parser handle the region
            return false;
        }
        // spirv.Branch / spv.Branch and other branch operations with successors
        if (name === 'spirv.Branch' || name === 'spv.Branch' ||
            name === 'spirv.BranchConditional' || name === 'spv.BranchConditional' ||
            name.includes('Branch')) {
            // Parse operands if any (for conditional branches)
            op.operands = parser._parseArguments();

            // Parse successors
            if (parser._match('^')) {
                op.successors = [];
                while (parser._match('^')) {
                    const successor = {};
                    successor.label = parser._read('^');
                    // Parse successor arguments with types
                    // Format: ^label(%val1, %val2, ... : type1, type2, ...)
                    if (parser._eat('(')) {
                        successor.arguments = [];
                        // Parse all values first
                        while (!parser._match(':') && !parser._match(')')) {
                            if (parser._match('%')) {
                                const arg = {};
                                arg.value = parser._read('%');
                                successor.arguments.push(arg);
                                parser._eat(',');
                            } else {
                                break;
                            }
                        }
                        // Parse types if present
                        if (parser._eat(':')) {
                            let idx = 0;
                            while (idx < successor.arguments.length && !parser._match(')')) {
                                const type = parser._parseType();
                                successor.arguments[idx].type = type;
                                idx++;
                                parser._eat(',');
                            }
                        }
                        parser._eat(')');
                    }
                    op.successors.push(successor);
                    if (!parser._eat(',')) {
                        break;
                    }
                }
            }

            return true;
        }
        return super.parseOperation(parser, opName, op);
    }
};

mlir.Metadata = class {

    static async open(context) {
        if (!mlir.Metadata._metadata) {
            const data = await context.request('mlir-metadata.json');
            mlir.Metadata._metadata = new mlir.Metadata(data);
        }
        return mlir.Metadata._metadata;
    }

    constructor(data) {
        this.types = new Map();
        if (data) {
            const operations = JSON.parse(data);
            for (const op of operations) {
                const metadata = { name: op.name };
                if (op.category) {
                    metadata.category = op.category;
                }
                if (op.summary) {
                    metadata.summary = op.summary;
                }
                if (op.description) {
                    metadata.description = op.description;
                }
                if (op.inputs) {
                    metadata.inputs = op.inputs;
                }
                if (op.outputs) {
                    metadata.outputs = op.outputs;
                }
                if (op.attributes) {
                    metadata.attributes = op.attributes;
                }
                this.types.set(op.name, metadata);
            }
        }
        this.register('asuka.split', 'Tensor');
        this.register('asuka.softmax', 'Activation');
        this.register('toy.transpose', 'Transform');
    }

    register(name, category) {
        if (!this.types.has(name)) {
            this.types.set(name, { name, category });
        }
    }

    type(name) {
        return this.types.get(name) || { name };
    }
};

mlir.Error = class extends Error {

    constructor(message) {
        super(message);
        this.name = 'Error loading MLIR model.';
    }
};

export const ModelFactory = mlir.ModelFactory;
export const Parser = mlir.Parser;
export const MlirError = mlir.Error;
