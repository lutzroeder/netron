
// Experimental

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
                if (/module\s+(@\w+|\w+|attributes|\{)/.test(line) ||
                    /tensor<[\w\d]+>/.test(line) ||
                    /func[.\s]*@\w+/.test(line) ||
                    /%\w+\s*=\s*"[\w.]+/.test(line) ||
                    /%\w+\s*=\s*\w+\./.test(line) ||
                    /!\w+\s*=\s*![\w.]+</.test(line) ||
                    /#\w+\s*=\s*#[\w.]+</.test(line) ||
                    /#\w+\s*=\s*loc\s*\(/.test(line) ||
                    /\w+\.\w+(?:\s+\w+)*\s+@\w+/.test(line) ||
                    /\w+\.\w+\.\w+\s*\{/.test(line) ||
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
        const function_type = attr.function_type;
        for (let i = 0; i < function_type.inputs.length; i++) {
            const input = function_type.inputs[i];
            const name = input.value || i.toString();
            const type = mlir.Utility.valueType(input.type);
            const valueName = input.value || input.name || `%arg${i}`;
            const value = new mlir.Value(valueName, type, '', null);
            const argument = new mlir.Argument(name, [value]);
            this.inputs.push(argument);
        }
        for (let i = 0; i < function_type.results.length; i++) {
            const output = function_type.results[i];
            const name = output.value || i.toString();
            const type = mlir.Utility.valueType(output.type);
            const valueName = output.value || output.name || `%result${i}`;
            const value = new mlir.Value(valueName, type, '', null);
            const argument = new mlir.Argument(name, [value]);
            this.outputs.push(argument);
        }
        const values = new Map();
        values.map = (name) => {
            if (!values.has(name)) {
                values.set(name, { name, to: [], from: [] });
            }
            return values.get(name);
        };
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
                    const opMetadata = metadata.type(op.name);
                    const operands = op.operands || [];
                    for (let i = 0; i < operands.length; i++) {
                        const input = op.operands[i];
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
                        } else if (typeof input.value === 'string' && input.value) {
                            const value = values.map(input);
                            value.to.push(operation);
                            const args = [{ name: input.value, type: input.type }];
                            operation.operands.push({
                                name: inputName,
                                value: args
                            });
                        } else {
                            operation.operands.push({
                                name: inputName,
                                value: input
                            });
                        }
                    }
                    const results = op.results || [];
                    for (let i = 0; i < results.length; i++) {
                        const output = results[i];
                        if (!output.value) {
                            // Skip results without value identifiers
                            continue;
                        }
                        const value = values.map(output.value);
                        value.type = mlir.Utility.valueType(output.type);
                        value.from.push(operation);
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
        for (const input of this.inputs) {
            for (const arg of input.value) {
                if (!tensors.has(arg.name)) {
                    tensors.set(arg.name, arg);
                }
            }
        }
        for (const output of this.outputs) {
            for (let i = 0; i < output.value.length; i++) {
                const arg = output.value[i];
                if (tensors.has(arg.name)) {
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
                if (!Array.isArray(input.value)) {
                    // Handle non-array values (e.g., affine maps, complex expressions) as attributes
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
                case 'i48': case 'si48': this.type = 'int48'; break;
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
                case 'dense':
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

    constructor(kind, value, text) {
        this.kind = kind;
        this.value = value;
        this.text = text;
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
                    this._read();
                    return new mlir.Token('keyword', '+');
                case '"':
                    return this._stringLiteral();
                case '@':
                    return this._symbolRefId();
                case '%':
                    return this._valueId();
                case '#':
                    if (this._peek() === '-') {
                        const position = this._decoder.position;
                        const next = this._decoder.decode();
                        this._decoder.position = position;
                        if (next === '}') {
                            this._read();
                            this._read();
                            this._read();
                            return new mlir.Token('#-}', '#-}');
                        }
                    }
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
                case '{': {
                    if (this._peek() === '-') {
                        const position = this._decoder.position;
                        const next = this._decoder.decode();
                        this._decoder.position = position;
                        if (next === '#') {
                            this._read();
                            this._read();
                            this._read();
                            return new mlir.Token('{-#', '{-#');
                        }
                    }
                    const value = this._read();
                    return new mlir.Token(value, value);
                }
                case '}':
                case '[':
                case ']':
                case '<':
                case '?':
                case '*':
                case '|': {
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
                        return this._identifier();
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
            return;
        }
        if (this._current === '*') {
            this._read();
            while (this._current) {
                if (this._current === '*') {
                    this._read();
                    if (this._current === '/') {
                        this._read();
                        return;
                    }
                } else {
                    this._read();
                }
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
        if (v === '0' && this._current === 'x' && /[0-9a-fA-F]/.test(this._peek())) {
            v += this._read();
            while (this._current && /[0-9a-fA-F]/.test(this._current)) {
                v += this._read();
            }
            return new mlir.Token(type, parseInt(v, 16), v);
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
            }
            return new mlir.Token(type, parseFloat(v), v);
        }
        return new mlir.Token(type, parseInt(v, 10), v);
    }

    _stringLiteral() {
        let result = '';
        this._read();
        while (this._current && this._current !== '"') {
            if (this._eat('\\')) {
                const hexDigit = /[0-9a-fA-F]/;
                if (hexDigit.test(this._current) && this._next && hexDigit.test(this._next)) {
                    const hex = this._current + this._next;
                    result += String.fromCharCode(parseInt(hex, 16));
                    this._read();
                    this._read();
                    continue;
                }
                switch (this._current) {
                    case 'n':
                        result += '\n';
                        this._read();
                        break;
                    case 'r':
                        result += '\r';
                        this._read();
                        break;
                    case 't':
                        result += '\t';
                        this._read();
                        break;
                    case '"':
                    case '\\':
                        result += this._current;
                        this._read();
                        break;
                    default:
                        throw new mlir.Error(`Unknown escape sequence '\\${this._current}' in string literal`);
                }
            } else {
                result += this._current;
                this._read();
            }
        }
        if (this._eat('"')) {
            return new mlir.Token('string', result);
        }
        throw new mlir.Error('Unterminated string literal');
    }

    _identifier() {
        let result = '';
        while (this._current && (/[a-zA-Z_$\-.]/.test(this._current) || /[0-9]/.test(this._current))) {
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
        let result = `!${id.value}`;
        if (this._current === '<') {
            let depth = 0;
            while (this._current) {
                if (this._current === '<') {
                    depth++;
                    result += this._read();
                } else if (this._current === '>') {
                    result += this._read();
                    depth--;
                    if (depth === 0) {
                        break;
                    }
                } else if (this._current === '"') {
                    result += this._read();
                    while (this._current && this._current !== '"') {
                        if (this._current === '\\') {
                            result += this._read();
                        }
                        if (this._current) {
                            result += this._read();
                        }
                    }
                    if (this._current === '"') {
                        result += this._read();
                    }
                } else {
                    result += this._read();
                }
            }
        }
        return new mlir.Token('!', result);
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
        const operations = metadata.operations;
        this._dialects.set('builtin', new mlir.BuiltinDialect(operations));
        this._dialects.set('bufferization', new mlir.BufferizationDialect(operations));
        this._dialects.set('stablehlo', new mlir.StableHLODialect(operations));
        this._dialects.set('vhlo', new mlir.VhloDialect(operations));
        this._dialects.set('interpreter', new mlir.InterpreterDialect(operations));
        this._dialects.set('affine', new mlir.AffineDialect(operations));
        this._dialects.set('asuka', new mlir.AsukaDialect(operations));
        this._dialects.set('arith', new mlir.ArithDialect(operations));
        this._dialects.set('async', new mlir.AsyncDialect(operations));
        this._dialects.set('cf', new mlir.CFDialect(operations));
        this._dialects.set('emitc', new mlir.EmitCDialect(operations));
        this._dialects.set('complex', new mlir.Dialect('complex', operations));
        this._dialects.set('index', new mlir.Dialect('index', operations));
        this._dialects.set('pdl', new mlir.Dialect('pdl', operations));
        this._dialects.set('ptr', new mlir.Dialect('ptr', operations));
        this._dialects.set('ub', new mlir.Dialect('ub', operations));
        this._dialects.set('amdgpu', new mlir.Dialect('amdgpu', operations));
        this._dialects.set('nvgpu', new mlir.Dialect('nvgpu', operations));
        this._dialects.set('nvvm', new mlir.NVVMDialect(operations));
        this._dialects.set('rocdl', new mlir.Dialect('rocdl', operations));
        this._dialects.set('nvws', new mlir.Dialect('nvws', operations));
        this._dialects.set('tti', new mlir.Dialect('tti', operations));
        this._dialects.set('omp', new mlir.OpenMPDialect(operations));
        this._dialects.set('proton', new mlir.ProtonDialect(operations));
        this._dialects.set('proton_gpu', new mlir.Dialect('proton_gpu', operations));
        this._dialects.set('arm_sme', new mlir.ArmSMEDialect(operations));
        this._dialects.set('arm_neon', new mlir.ArmNeonDialect(operations));
        this._dialects.set('arm_sve', new mlir.ArmSVEDialect(operations));
        this._dialects.set('shard', new mlir.Dialect('shard', operations));
        this._dialects.set('amx', new mlir.Dialect('amx', operations));
        this._dialects.set('smt', new mlir.Dialect('smt', operations));
        this._dialects.set('lagrad', new mlir.Dialect('lagrad', operations));
        this._dialects.set('iree_codegen', new mlir.Dialect('iree_codegen', operations));
        this._dialects.set('iree_encoding', new mlir.Dialect('iree_encoding', operations));
        this._dialects.set('test', new mlir.TestDialect(operations));
        this._dialects.set('scf', new mlir.SCFDialect(operations));
        this._dialects.set('shape', new mlir.ShapeDialect(operations));
        this._dialects.set('sparse_tensor', new mlir.SparseTensorDialect(operations));
        this._dialects.set('func', new mlir.FuncDialect(operations));
        this._dialects.set('gpu', new mlir.GpuDialect(operations));
        this._dialects.set('llvm', new mlir.LLVMDialect(operations));
        this._dialects.set('xegpu', new mlir.Dialect('xegpu', operations));
        this._dialects.set('memref', new mlir.MemRefDialect(operations));
        this._dialects.set('vector', new mlir.VectorDialect(operations));
        this._dialects.set('x86vector', new mlir.Dialect('x86vector', operations));
        this._dialects.set('onnx', new mlir.ONNXDialect(operations));
        this._dialects.set('krnl', new mlir.Dialect('krnl', operations));
        this._dialects.set('torch', new mlir.TorchDialect(operations));
        this._dialects.set('torch_c', new mlir.Dialect('torch_c', operations));
        this._dialects.set('hal', new mlir.HALDialect(operations));
        this._dialects.set('hal_loader', new mlir.Dialect('hal_loader', operations));
        this._dialects.set('hal_inline', new mlir.Dialect('hal_inline', operations));
        this._dialects.set('util', new mlir.UtilDialect(operations));
        this._dialects.set('mhlo', new mlir.MhloDialect(operations));
        this._dialects.set('chlo', new mlir.Dialect('chlo', operations));
        this._dialects.set('flow', new mlir.FlowDialect(operations));
        this._dialects.set('stream', new mlir.StreamDialect(operations));
        this._dialects.set('iree_vector_ext', new mlir.Dialect('iree_vector_ext', operations));
        this._dialects.set('iree_tensor_ext', new mlir.IREETensorExtDialect(operations));
        this._dialects.set('linalg', new mlir.LinalgDialect(operations));
        this._dialects.set('iree_linalg_ext', new mlir.Dialect('iree_linalg_ext', operations));
        this._dialects.set('quant', new mlir.QuantDialect(operations));
        this._dialects.set('tensor', new mlir.Dialect('tensor', operations));
        this._dialects.set('tosa', new mlir.TosaDialect(operations));
        this._dialects.set('tf', new mlir.TFDialect(operations));
        this._dialects.set('tf_saved_model', new mlir.Dialect('tf_saved_model', operations));
        this._dialects.set('tf_type', new mlir.TFTypeDialect(operations));
        this._dialects.set('tf_device', new mlir.TFDeviceDialect(operations));
        this._dialects.set('tf_executor', new mlir.TFExecutorDialect(operations));
        this._dialects.set('tf_framework', new mlir.TFFrameworkDialect(operations));
        this._dialects.set('tfr', new mlir.TFRDialect(operations));
        this._dialects.set('tfrt', new mlir.TFRTDialect(operations));
        this._dialects.set('tfrt_fallback', new mlir.Dialect('tfrt_fallback', operations));
        this._dialects.set('tfl', new mlir.TFLDialect(operations));
        this._dialects.set('stdx', new mlir.StdxDialect(operations));
        this._dialects.set('vm', new mlir.VMDialect(operations));
        this._dialects.set('math', new mlir.MathDialect(operations));
        this._dialects.set('tm_tensor', new mlir.TMTensorDialect(operations));
        this._dialects.set('ml_program', new mlir.MLProgramDialect(operations));
        this._dialects.set('iree_gpu', new mlir.IREEGPUDialect(operations));
        this._dialects.set('tile', new mlir.TileDialect(operations));
        this._dialects.set('irdl', new mlir.IRDLDialect(operations));
        this._dialects.set('transform', new mlir.TransformDialect(operations));
        this._dialects.set('wasmssa', new mlir.Dialect('wasmssa', operations));
        this._dialects.set('spirv', new mlir.SPIRVDialect(operations));
        this._dialects.set('spv', this._dialects.get('spirv'));
        this._dialects.set('toy', new mlir.ToyDialect(operations));
        this._dialects.set('top', new mlir.Dialect('top', operations));
        this._dialects.set('tpu', new mlir.Dialect('tpu', operations));
        this._dialects.set('sdfg', new mlir.SdfgDialect(operations));
        this._dialects.set('sdir', this._dialects.get('sdfg'));
        this._dialects.set('check', new mlir.CheckDialect(operations));
        this._dialects.set('tt', new mlir.TritonDialect(operations));
        this._dialects.set('ttg', new mlir.TritonGPUDialect(operations));
        this._dialects.set('triton_gpu', this._dialects.get('ttg'));
        this._dialects.set('gluon', new mlir.GluonDialect(operations));
        this._dialects.set('ttng', new mlir.TritonNvidiaGPUDialect(operations));
        this._dialects.set('nvidia_gpu', this._dialects.get('ttng'));
        this._dialects.set('michelson', new mlir.MichelsonDialect(operations));
        this._redirect = new Map([
            ['builtin.constant', 'arith.constant'],
            ['builtin.return', 'func.return'],
            ['builtin.select', 'arith.select'],
            ['scf.select', 'arith.select'],
            ['scf.call', 'func.call'],
            ['linalg.select', 'arith.select'],
            ['builtin.view', 'memref.view'],
            ['builtin.dealloc', 'memref.dealloc'],
            ['builtin.addi', 'arith.addi'],
            ['builtin.subi', 'arith.subi'],
            ['builtin.muli', 'arith.muli'],
            ['builtin.divi_signed', 'arith.divsi'],
            ['builtin.divi_unsigned', 'arith.divui'],
            ['builtin.divsi', 'arith.divsi'],
            ['builtin.divui', 'arith.divui'],
            ['builtin.andi', 'arith.andi'],
            ['builtin.ori', 'arith.ori'],
            ['builtin.xori', 'arith.xori'],
            ['builtin.shli', 'arith.shli'],
            ['builtin.shrsi', 'arith.shrsi'],
            ['builtin.shrui', 'arith.shrui'],
            ['builtin.addf', 'arith.addf'],
            ['builtin.subf', 'arith.subf'],
            ['builtin.mulf', 'arith.mulf'],
            ['builtin.divf', 'arith.divf'],
            ['builtin.splat', 'vector.splat'],
            ['scf.splat', 'vector.splat'],
            ['flow.constant', 'flow.tensor.constant']
        ]);
    }

    async read() {
        return this.parse();
    }

    parse() {
        // https://mlir.llvm.org/docs/LangRef/#top-level-productions
        const block = {
            operations: [],
            definitions: []
        };

        while (true) {
            if (this.match('eof')) {
                break;
            }
            if (this.match('#')) { // attribute-alias-def
                const name = this.expect();
                this.expect('=');
                const value = this.parseAttributeValue();
                block.definitions.push({ name, value });
                continue;
            }
            if (this.match('!')) { // type-alias-def
                const name = this.expect();
                this.expect('=');
                const type = this.parseType();
                block.definitions.push({ name, type });
                continue;
            }
            if (this.match('{-#')) { // file metadata
                this.expect('{-#');
                while (!this.match('#-}') && !this.match('eof')) {
                    this._token = this._tokenizer.read();
                }
                this.expect('#-}');
                continue;
            }
            const op = this.parseOperation();
            block.operations.push(op);
        }
        return block;
    }

    parseFunctionArgumentList() {
        const inputs = [];
        if (this.accept('(')) {
            while (!this.accept(')')) {
                if (this.match(')')) {
                    break;
                }
                if (this.match('%')) {
                    const input = {};
                    input.value = this._token.value;
                    this.expect('%');
                    this.expect(':');
                    input.type = this.parseType();
                    if (this.match('{')) {
                        input.attributes = [];
                        this.parseAttributeDict(input.attributes);
                    }
                    input.loc = this.parseLocation();
                    inputs.push(input);
                } else {
                    const input = {};
                    input.value = `%arg${inputs.length}`;
                    input.type = this.parseType();
                    if (this.match('{')) {
                        input.attributes = [];
                        this.parseAttributeDict(input.attributes);
                    }
                    inputs.push(input);
                }
                if (!this.match(')')) {
                    if (!this.accept(',')) {
                        break;
                    }
                    if (this.match(')')) {
                        break;
                    }
                }
            }
        }
        return inputs;
    }

    parseFunctionResultList() {
        const outputs = [];
        if (this.accept('(')) {
            while (!this.accept(')')) {
                const output = {};
                output.value = `%result${outputs.length}`;  // Generate a name
                output.type = this.parseType();
                if (this.match('{')) {
                    output.attributes = [];
                    this.parseAttributeDict(output.attributes);
                }
                outputs.push(output);
                this.accept(',');
            }
        } else {
            const output = {};
            output.value = `%result0`;  // Generate a name
            output.type = this.parseType();
            outputs.push(output);
        }
        return outputs;
    }

    parseTypeList() {
        const types = [];
        while (!this.match(')') && !this.match('eof')) {
            const type = this.parseType();
            if (type) {
                types.push(type);
            }
            if (!this.accept(',')) {
                break;
            }
        }
        return types;
    }

    skip(open, close) {
        let value = '';
        if (this.match(open)) {
            value += this.expect();
            let count = 1;
            while (count > 0) {
                if (this.match('eof')) {
                    throw new mlir.Error(`Unexpected end of file while looking for '${close}' ${this.location()}`);
                }
                if (this.match(open)) {
                    count++;
                } else if (this.match(close)) {
                    count--;
                }
                value += this.expect();
            }
        }
        return value;
    }

    parseOperation() {
        const results = [];
        if (this.match('%')) {
            do {
                if (!this.match('%')) {
                    break;
                }
                const value = this.expect();
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
            } while (this.accept(','));
            this.expect('=');
        }
        let op = null;
        if (this.match('id')) {
            op = this.parseCustomOperation(results);
        } else if (this.match('string')) {
            op = this.parseGenericOperation();
        } else {
            throw new mlir.Error(`Unexpected operation name '${this._token.value}' ${this.location()}`);
        }
        if (!op) {
            throw new mlir.Error(`Failed to parse operation ${this.location()}`);
        }
        op.results = results;
        return op;
    }

    parseGenericOperationAfterOpName(op) {
        op.attributes = op.attributes || [];
        op.operands = op.operands || [];
        op.regions = op.regions || [];
        op.results = op.results || [];
        if (this.match('}')) {
            op.loc = this.parseLocation();
            return op;
        }
        if (!op.operands.length) {
            op.operands = this.parseArguments();
        }
        // Parse successor blocks (for branch operations like cf.br)
        if (this.match('^')) {
            op.successors = [];
            while (this.match('^')) {
                const successor = {};
                successor.label = this.expect('^');
                if (this.accept('(')) {
                    successor.arguments = [];
                    while (!this.accept(')')) {
                        if (this.match('%')) {
                            const value = this.expect('%');
                            successor.arguments.push({ value });
                            if (!this.accept(',')) {
                                break;
                            }
                        } else {
                            break;
                        }
                    }
                    const hasTypes = this.accept(':');
                    if (!hasTypes) {
                        this.accept(')');
                    }
                    if (hasTypes) {
                        let idx = 0;
                        while (idx < successor.arguments.length && !this.match(',') && !this.match('[') && !this.match('{') && !this.match('^')) {
                            const type = this.parseType();
                            if (successor.arguments[idx]) {
                                successor.arguments[idx].type = type;
                            }
                            idx++;
                            this.accept(',');
                        }
                        this.accept(')');
                    }
                }
                op.successors.push(successor);
                if (!this.accept(',')) {
                    break;
                }
            }
        }
        while (this.match('[')) {
            this.skip('[', ']');
        }
        this.skip('<', '>');
        let parsedRegionList = false;
        if (this.match('(')) {
            const savedToken = this._token;
            this.accept('(');
            if (this.match('{') || this.match('id') || this.match('^')) {
                parsedRegionList = true;
            } else {
                this._token = savedToken;
            }
        }
        if (parsedRegionList) {
            while (!this.match(')')) {
                let entryLabel = null;
                const entryArgs = [];
                if (this.match('id') && !this.match('{')) {
                    entryLabel = this.expect('id');
                    if (this.accept('(')) {
                        while (!this.accept(')')) {
                            const value = this.expect('%');
                            this.expect(':');
                            const type = this.parseType();
                            entryArgs.push({ value, type });
                            this.accept(',');
                        }
                    }
                }
                if (!this.match('{')) {
                    throw new mlir.Error(`Expected '{' for region in region-list, but got '${this._token.value}' ${this.location()}`);
                }
                const region = {};
                this.parseRegion(region);
                if (entryLabel) {
                    region.entryLabel = entryLabel;
                    region.entryArgs = entryArgs;
                }
                op.regions.push(region);
                if (!this.accept(',') && !this.match(')')) {
                    throw new mlir.Error(`Expected ',' or ')' after region in region-list, but got '${this._token.value}' ${this.location()}`);
                }
            }
            this.expect(')');
        }
        if (this.match('{')) {
            if (parsedRegionList || op.attributes.length === 0 || (op.attributes.length === 1 && op.attributes[0].name === 'predicate')) {
                this.parseAttributeDict(op.attributes);
            } else {
                const region = {};
                this.parseRegion(region);
                op.regions.push(region);
            }
        }
        if (this.accept(':')) {
            this.parseArgumentTypes(op.operands);
        }
        if (this.accept('->') || this.accept('id', 'to')) {
            this.parseArgumentTypes(op.results);
        }
        if (this.accept('id', 'attributes')) {
            if (this.match('{')) {
                this.parseAttributeDict(op.attributes);
            }
        }
        if (this.match('{')) {
            const region = {};
            this.parseRegion(region);
            op.regions.push(region);
            while (this.accept(',') && this.match('{')) {
                const region = {};
                this.parseRegion(region);
                op.regions.push(region);
            }
            if (op.name.endsWith('.if') && this.match('id', 'else')) {
                this.expect('id', 'else');
                const region = {};
                this.parseRegion(region);
                op.regions.push(region);
            }
            if (this.match('{')) {
                this.parseAttributeDict(op.attributes);
            }
        }
        op.loc = this.parseLocation();
        return op;
    }

    parseCustomOperation(results) {
        const opNameInfo = this.parseCustomOperationName();
        const op = { name: opNameInfo, results, attributes: [], operands: [], regions: [] };
        if (op.name.startsWith('triton_gpu.')) {
            op.name = op.name.replace('triton_gpu.', 'ttg.');
        }
        if (op.name.startsWith('nvidia_gpu.')) {
            op.name = op.name.replace('nvidia_gpu.', 'ttng.');
        }
        if (this._redirect.has(op.name)) {
            op.name = this._redirect.get(op.name);
        }
        const index = op.name.indexOf('.');
        if (index === -1) {
            throw new mlir.Error(`No dialect found '${op.name}' ${this.location()}`);
        }
        const dialectName = op.name.substring(0, index);
        if (!this._dialects.has(dialectName)) {
            throw new mlir.Error(`Unsupported dialect '${dialectName}' ${this.location()}`);
        }
        const dialect = this._dialects.get(dialectName);
        this._state.defaultDialectStack.push(dialectName);
        if (dialect.parseOperation(this, op.name, op)) {
            if (!dialect.hasParser(op.name) && dialect.hasAssemblyFormat(op.name) && dialect.hasParseOperation(op.name) !== false) {
                throw new mlir.Error(`Operation '${op.name}' has assembly format but was handled by custom dialect code.`);
            }
            if (this.match('{')) {
                this.parseAttributeDict(op.attributes);
            }
            op.loc = this.parseLocation() || {};
            this._state.defaultDialectStack.pop();
            return op;
        }
        this._state.defaultDialectStack.pop();
        throw new mlir.Error(`Unsupported custom operation '${op.name}' ${this.location()}`);
    }

    parseCustomOperationName() {
        let opName = this.expect('id');
        if (opName.indexOf('.') === -1) {
            const dialect = this._state.defaultDialectStack[this._state.defaultDialectStack.length - 1];
            opName = `${dialect}.${opName}`;
        }
        return opName;
    }

    parseGenericOperation() {
        const op = { name: this.expect('string'), attributes: [], operands: [], regions: [], results: [] };
        return this.parseGenericOperationAfterOpName(op);
    }

    parseOptionalVisibilityKeyword(attributes) {
        if (this.match('id', 'private') || this.match('id', 'public') || this.match('id', 'nested')) {
            const value = this.expect();
            attributes.push({ name: 'sym_visibility', value });
        }
    }

    parseSymbolName(name, attributes) {
        const value = this.expect('@');
        attributes.push({ name, value });
    }

    parseOptionalSymbolName() {
        return this.accept('@');
    }

    parseOptionalAttrDictWithKeyword(attributes) {
        if (this.accept('id', 'attributes')) {
            this.parseAttributeDict(attributes);
        }
    }

    parseOptionalAttrDict(attributes) {
        if (this.match('{')) {
            this.parseAttributeDict(attributes);
        }
    }

    parseAttributeDict(attributes) {
        if (this.accept('{')) {
            while (!this.accept('}')) {
                let name = null;
                if (this.match('id') || this.match('string') || this.match('keyword')) {
                    name = this.expect();
                } else if (this.match('[')) {
                    const arrayValue = this.parseValue();
                    attributes.push({ name: 'array', value: arrayValue.value });
                    this.accept(',');
                    continue;
                } else if (!this.match('=') && !this.match(':') && !this.match('}')) {
                    throw new mlir.Error(`Expected attribute name or '}', but got '${this._token.value}' ${this.location()}`);
                }
                let attribute = {};
                if (this.accept('=') || this.accept(':')) {
                    attribute = this.parseValue();
                    if (this.accept(':')) {
                        attribute.type = this.parseType();
                    }
                } else if (name) {
                    attribute = { name };
                    attributes.push(attribute);
                    this.accept(',');
                    continue;
                } else {
                    break;
                }

                attribute.name = name;
                attributes.push(attribute);
                if (!this.accept(',') && !this.match('}')) {
                    throw new mlir.Error(`Expected ',' or '}' after attribute, but got '${this._token.value}' ${this.location()}`);
                }
            }
        }
    }

    parseRegion(region) {
        region.blocks = Array.isArray(region.blocks) ? region.blocks : [];
        const block = {};
        this.parseBlock(block);
        region.blocks.push(block);

        let hasMultipleBlocks = false;
        while ((this._token.kind === '^' || (this._token.kind === 'id' && this._token.value && this._token.value.startsWith('^'))) && !this.match('}')) {
            hasMultipleBlocks = true;
            const nextBlock = {};
            nextBlock.operations = [];
            nextBlock.arguments = [];
            if (this._token.kind === '^') {
                nextBlock.name = this.expect('^');
            } else {
                nextBlock.name = this.expect('id');
            }
            if (this.accept('(')) {
                while (!this.accept(')')) {
                    const value = this.expect('%');
                    this.expect(':');
                    const type = this.parseType();
                    const arg = { value, type };
                    const loc = this.parseLocation();
                    if (loc) {
                        arg.loc = loc;
                    }
                    nextBlock.arguments.push(arg);
                    this.accept(',');
                }
            }
            if (nextBlock.name && nextBlock.name.endsWith(':')) {
                nextBlock.name = nextBlock.name.slice(0, -1);
            } else {
                this.expect(':');
            }
            while (!(this._token.kind === '^' || (this._token.kind === 'id' && this._token.value && this._token.value.startsWith('^'))) && !this.match('}')) {
                const op = this.parseOperation();
                nextBlock.operations.push(op);
            }
            region.blocks.push(nextBlock);
        }
        if (hasMultipleBlocks && this.match('}')) {
            this.expect('}');
        }
        return region;
    }

    parseBlock(block) {
        block.operations = Array.isArray(block.operations) ? block.operations : [];
        block.arguments = Array.isArray(block.arguments) ? block.arguments : [];
        this.expect('{');
        if (this._token.kind === '^' || (this._token.kind === 'id' && this._token.value && this._token.value.startsWith('^'))) {
            if (this._token.kind === '^') {
                block.name = this.expect('^');
            } else {
                block.name = this.expect('id');
            }
            if (this.accept('(')) {
                while (!this.accept(')') && !this.match('^')) {
                    const value = this.expect('%');
                    this.expect(':');
                    const type = this.parseType();
                    const arg = { value, type };
                    const loc = this.parseLocation();
                    if (loc) {
                        arg.loc = loc;
                    }
                    block.arguments.push(arg);
                    this.accept(',');
                }
            }
            if (block.name && block.name.endsWith(':')) {
                block.name = block.name.slice(0, -1);
            } else {
                this.expect(':');
            }
        }
        while (!this.accept('}')) {
            if (this._token.kind === '^' || (this._token.kind === 'id' && this._token.value && this._token.value.startsWith('^'))) {
                break;
            }
            const op = this.parseOperation();
            block.operations.push(op);
        }
        block.loc = this.parseLocation();
        return block;
    }

    parseLocation() {
        if (this.accept('keyword', 'loc')) {
            const location = {};
            this.expect('(');
            if (this.match('string')) {
                const str = this.expect('string');
                if (this.match('(')) {
                    location.name = str;
                    this.expect('(');
                    location.child = this.parseLocationContent();
                    this.expect(')');
                } else {
                    location.file = str;
                    if (this.accept(':')) {
                        location.line = this.expect('int');
                        if (this.accept(':')) {
                            location.col = this.expect('int');
                        }
                    }
                }
            } else if (this.match('#')) {
                location.alias = this.expect();
            } else if (this.accept('id', 'unknown')) {
                location.unknown = true;
            } else if (this.accept('id', 'callsite')) {
                this.expect('(');
                location.type = 'callsite';
                location.callee = this.parseLocationContent();
                this.expect('id', 'at');
                location.caller = this.parseLocationContent();
                this.expect(')');
            } else if (this.accept('id', 'fused')) {
                location.type = 'fused';
                if (this.accept('<')) {
                    location.metadata = this.parseValue();
                    this.expect('>');
                }
                this.expect('[');
                location.locations = [];
                do {
                    location.locations.push(this.parseLocationContent());
                } while (this.accept(','));
                this.expect(']');
            } else {
                throw new mlir.Error(`Unexpected location '${this._token.value}' ${this.location()}`);
            }
            this.expect(')');
            return location;
        }
        return null;
    }

    parseLocationContent() {
        if (this.match('#')) {
            return { alias: this.expect() };
        }
        if (this.match('keyword', 'loc')) {
            return this.parseLocation();
        }
        throw new mlir.Error(`Expected location content, got '${this._token.value}' ${this.location()}`);
    }

    parseOperationName() {
        switch (this._token.kind) {
            case 'string':
                return this.expect();
            case 'id':
                return this.expect('id');
            default:
                throw new mlir.Error(`Unexpected operation name '${this._token.value}' ${this.location()}`);
        }
    }

    parseArguments() {
        const inputs = [];
        if (this.match('{')) {
            return inputs;
        }
        const open = this.accept('(');
        // eslint-disable-next-line no-unmodified-loop-condition
        while (!this.match(')') && !this.match('->') && !this.match('{') && !this.match('}') && !this.match('=') && !this.match('^') && !(this.match(':') && !open)) {
            const input = {};
            if (this.match('[')) {
                this.expect('[');
                const array = [];
                while (!this.match(']')) {
                    if (this.match('%')) {
                        array.push(this.expect());
                    } else if (this.match('int')) {
                        array.push(parseInt(this.expect('int'), 10));
                    } else if (this.match('-')) {
                        this.expect('-');
                        if (this.match('int')) {
                            array.push(-parseInt(this.expect('int'), 10));
                        } else {
                            throw new mlir.Error(`Expected integer after '-' in array literal ${this.location()}`);
                        }
                    } else {
                        break;
                    }
                    if (!this.accept(',')) {
                        break;
                    }
                }
                this.expect(']');
                input.value = array;
                inputs.push(input);
                if (!this.accept(',')) {
                    break;
                }
                continue;
            }
            if (this._token.kind === 'id' && this._token.value !== 'dense' && this._token.value !== 'dense_resource') {
                const identifier = this.expect('id');
                if (this.accept('(')) {
                    const args = this.parseArguments();
                    for (let i = 0; i < args.length; i++) {
                        const arg = args[i];
                        arg.name = `${identifier}.${i}`;
                        inputs.push(arg);
                    }
                    if (this.accept(':')) {
                        this.parseArgumentTypes(inputs);
                    }
                    this.expect(')');
                    continue;
                } else if (this.match('=')) {
                    input.name = identifier;
                    this.expect('=');
                } else if (this.match(':')) {
                    // Named argument syntax: identifier: value (e.g., init: %init)
                    input.name = identifier;
                    this.expect(':');
                } else {
                    input.value = identifier;
                    inputs.push(input);
                    if (!this.accept(',')) {
                        break;
                    }
                    continue;
                }
            }
            if (this.match('%')) {
                input.value = this.expect();
                if (open && this.accept(':')) {
                    input.type = this.parseType();
                }
            } else if (this.match('keyword', 'loc')) {
                break;
            } else {
                const value = this.parseValue();
                input.type = value.type;
                input.value = value.value;
                if (open && this.accept(':')) {
                    input.type = this.parseType();
                }
            }
            inputs.push(input);
            if (!this.accept(',') && !this.match('id')) {
                break;
            }
        }
        if (open) {
            this.expect(')');
        }
        return inputs;
    }

    parseElementTypeFromPrefix(prefix, dimensions) {
        if (/^[0-9?]/.test(prefix)) {
            let i = 0;
            while (i < prefix.length) {
                if (prefix[i] === '?') {
                    dimensions.push('?');
                    i++;
                } else if (/[0-9]/.test(prefix[i])) {
                    let numStr = '';
                    while (i < prefix.length && /[0-9]/.test(prefix[i])) {
                        numStr += prefix[i];
                        i++;
                    }
                    dimensions.push(parseInt(numStr, 10));
                } else {
                    break;
                }

                if (i < prefix.length && prefix[i] === 'x') {
                    i++;
                } else {
                    break;
                }
            }

            prefix = prefix.substring(i);
        }

        // Handle nested types like memref<4xvector<16xf32>> or tensor<20x20xcomplex<f32>>
        if (prefix === 'complex') {
            if (this.accept('<')) {
                const elementType = this.parseType();
                this.expect('>');
                return `complex<${elementType}>`;
            }
        } else if (prefix === 'tensor' || prefix === 'vector' || prefix === 'memref') {
            if (this.accept('<')) {
                const nestedDimInfo = this.parseDimensionListRanked();

                let nestedElementType = null;
                if (nestedDimInfo.elementTypePrefix) {
                    nestedElementType = this.parseElementTypeFromPrefix(nestedDimInfo.elementTypePrefix, nestedDimInfo.dimensions);
                    if (!nestedElementType) {
                        if (this.match('?') || this.match('int')) {
                            const moreDims = this.parseDimensionListRanked();
                            nestedDimInfo.dimensions.push(...moreDims.dimensions);
                            if (moreDims.elementTypePrefix) {
                                nestedElementType = this.parseElementTypeFromPrefix(moreDims.elementTypePrefix, nestedDimInfo.dimensions);
                            } else {
                                nestedElementType = this.parseType();
                            }
                        } else {
                            nestedElementType = this.parseType();
                        }
                    }
                } else {
                    nestedElementType = this.parseType();
                }

                this.expect('>');

                let nestedTypeStr = `${prefix}<`;
                if (nestedDimInfo.unranked) {
                    nestedTypeStr += '*x';
                } else if (nestedDimInfo.dimensions.length > 0) {
                    nestedTypeStr += `${nestedDimInfo.dimensions.join('x')}x`;
                }
                nestedTypeStr += `${nestedElementType}>`;

                return nestedTypeStr;
            }
        }
        if (prefix.startsWith('!')) {
            this._token = new mlir.Token('!', prefix, prefix);
            return this.parseType();
        }
        return prefix;
    }

    parseDimensionListRanked() {
        const dimensions = [];

        if (this.accept('*')) {
            if (this.match('id')) {
                const token = this._token.value;
                if (token === 'x' || token.startsWith('x')) {
                    this.expect('id');
                    return { unranked: true, dimensions: [], elementTypePrefix: token === 'x' ? null : token.substring(1) };
                }
            }
            return { unranked: true, dimensions: [], elementTypePrefix: null };
        }

        while (true) {
            if (this.accept('[')) {
                if (this.match('int')) {
                    dimensions.push(`[${this.expect('int')}]`);
                } else if (this.match('?')) {
                    dimensions.push('[?]');
                    this.expect('?');
                }
                this.expect(']');
            } else if (this.match('?')) {
                dimensions.push('?');
                this.expect('?');
            } else if (this.match('int')) {
                const tokenText = this._token.text;
                if (tokenText && tokenText.length > 2 && tokenText.startsWith('0x') && /[fiub]/.test(tokenText[2])) {
                    dimensions.push(0);
                    this._token = new mlir.Token('id', tokenText.substring(1), tokenText.substring(1));
                } else {
                    dimensions.push(parseInt(this.expect('int'), 10));
                }
            } else {
                break;
            }

            if (this.match('id')) {
                const token = this._token.value;
                if (token === 'x') {
                    this.expect('id', 'x');
                } else if (token.startsWith('x')) {
                    this.expect('id');
                    let rest = token.substring(1);
                    while (rest.length > 0) {
                        if (/^[0-9]/.test(rest)) {
                            let i = 0;
                            while (i < rest.length && /[0-9]/.test(rest[i])) {
                                i++;
                            }
                            const numPart = rest.substring(0, i);
                            dimensions.push(parseInt(numPart, 10));
                            rest = rest.substring(i);
                            if (rest.startsWith('x')) {
                                rest = rest.substring(1);
                                continue;
                            }
                            if (rest.length > 0) {
                                return { unranked: false, dimensions, elementTypePrefix: rest };
                            }
                            break;
                        } else if (rest === '?') {
                            dimensions.push('?');
                            break;
                        } else {
                            return { unranked: false, dimensions, elementTypePrefix: rest };
                        }
                    }
                } else {
                    break;
                }
            } else {
                break;
            }
        }

        return { unranked: false, dimensions, elementTypePrefix: null };
    }

    parseTensorType() {
        this.expect('id', 'tensor');
        this.expect('<');
        const dimInfo = this.parseDimensionListRanked();
        let elementType = null;
        if (dimInfo.elementTypePrefix) {
            elementType = this.parseElementTypeFromPrefix(dimInfo.elementTypePrefix, dimInfo.dimensions);
            if (!elementType) {
                if (this.match('?') || this.match('int')) {
                    const moreDims = this.parseDimensionListRanked();
                    dimInfo.dimensions.push(...moreDims.dimensions);
                    if (moreDims.elementTypePrefix) {
                        elementType = this.parseElementTypeFromPrefix(moreDims.elementTypePrefix, dimInfo.dimensions);
                    } else {
                        elementType = this.parseType();
                    }
                } else {
                    elementType = this.parseType();
                }
            }
        } else {
            elementType = this.parseType();
        }
        let encoding = null;
        if (this.accept(',')) {
            encoding = this.parseAttributeValue();
        }
        this.expect('>');
        let typeStr = 'tensor<';
        if (dimInfo.unranked) {
            typeStr += '*x';
        } else if (dimInfo.dimensions.length > 0) {
            typeStr += `${dimInfo.dimensions.join('x')}x`;
        }
        typeStr += elementType;
        if (encoding) {
            const enc = typeof encoding === 'object' ? JSON.stringify(encoding) : encoding;
            typeStr += `, ${enc}`;
        }
        typeStr += '>';
        return typeStr;
    }

    parseMemRefType() {
        this.expect('id', 'memref');
        this.expect('<');
        const dimInfo = this.parseDimensionListRanked();
        let elementType = null;
        if (dimInfo.elementTypePrefix) {
            elementType = this.parseElementTypeFromPrefix(dimInfo.elementTypePrefix, dimInfo.dimensions);
            if (!elementType) {
                if (this.match('?') || this.match('int')) {
                    const moreDims = this.parseDimensionListRanked();
                    dimInfo.dimensions.push(...moreDims.dimensions);
                    if (moreDims.elementTypePrefix) {
                        elementType = this.parseElementTypeFromPrefix(moreDims.elementTypePrefix, dimInfo.dimensions);
                    } else {
                        elementType = this.parseType();
                    }
                } else {
                    elementType = this.parseType();
                }
            }
        } else {
            elementType = this.parseType();
        }

        const extras = [];
        while (this.accept(',')) {
            const extra = this.parseAttributeValue();
            extras.push(extra);
        }

        this.expect('>');

        let memorySpaceBraces = '';
        if (this.match('{')) {
            memorySpaceBraces = this.skip('{', '}');
        }

        let typeStr = 'memref<';
        if (dimInfo.unranked) {
            typeStr += '*x';
        } else if (dimInfo.dimensions.length > 0) {
            typeStr += `${dimInfo.dimensions.join('x')}x`;
        }
        typeStr += elementType;
        if (extras.length > 0) {
            const content = extras.map((e) => typeof e === 'object' ? JSON.stringify(e) : e).join(', ');
            typeStr += `, ${content}`;
        }
        typeStr += '>';
        typeStr += memorySpaceBraces;

        return typeStr;
    }

    parseVectorType() {
        this.expect('id', 'vector');
        this.expect('<');
        const dimInfo = this.parseDimensionListRanked();
        let elementType = null;
        if (dimInfo.elementTypePrefix) {
            elementType = this.parseElementTypeFromPrefix(dimInfo.elementTypePrefix, dimInfo.dimensions);
            if (!elementType) {
                if (this.match('?') || this.match('int')) {
                    const moreDims = this.parseDimensionListRanked();
                    dimInfo.dimensions.push(...moreDims.dimensions);
                    if (moreDims.elementTypePrefix) {
                        elementType = this.parseElementTypeFromPrefix(moreDims.elementTypePrefix, dimInfo.dimensions);
                    } else {
                        elementType = this.parseType();
                    }
                } else {
                    elementType = this.parseType();
                }
            }
        } else {
            elementType = this.parseType();
        }

        this.expect('>');

        let typeStr = 'vector<';
        if (dimInfo.dimensions.length > 0) {
            typeStr += `${dimInfo.dimensions.join('x')}x`;
        }
        typeStr += elementType;
        typeStr += '>';

        return typeStr;
    }

    parseComplexType() {
        this.expect('id', 'complex');
        this.expect('<');
        const elementType = this.parseType();
        this.expect('>');
        return `complex<${elementType}>`;
    }

    parseTupleType() {
        this.expect('id', 'tuple');
        this.expect('<');
        const types = [];
        while (!this.match('>')) {
            types.push(this.parseType());
            this.accept(',');
        }
        this.expect('>');
        return `tuple<${types.join(', ')}>`;
    }

    parseType() {
        if (this._token.kind === 'id') {
            const value = this._token.value;
            if (value === 'none' || value === 'index' || /^[su]?i[0-9]+$/.test(value) ||
                /^f[0-9]+$/.test(value) || value === 'bf16' || value === 'tf32' || value.startsWith('f8') || value.startsWith('f6') || value.startsWith('f4')) {
                return this.expect('id');
            }
            if (value === 'tensor') {
                return this.parseTensorType();
            }
            if (value === 'vector') {
                return this.parseVectorType();
            }
            if (value === 'memref') {
                return this.parseMemRefType();
            }
            if (value === 'complex') {
                return this.parseComplexType();
            }
            if (value === 'tuple') {
                return this.parseTupleType();
            }
        }
        if (this.match('!')) {
            const value = this.expect();  // Read !dialect.typename
            if (value && value.startsWith('!')) {
                const match = value.match(/^!([^.<]+)/);
                if (match) {
                    const [,dialectName] = match;
                    if (!this._dialects.has(dialectName)) {
                        throw new mlir.Error(`Unsupported dialect '${dialectName}' ${this.location()}`);
                    }
                    const dialect = this._dialects.get(dialectName);
                    const parsedType = dialect.parseType(this, value);
                    if (!parsedType) {
                        throw new mlir.Error(`Failed to parse type '${value}' ${this.location()}`);
                    }
                    return parsedType;
                }
            }
            throw new mlir.Error(`Invalid type '${value}' ${this.location()}`);
        }
        if (this.match('(')) {
            let value = this.skip('(', ')');
            if (this.match('->')) {
                value += this.expect();
                if (this.match('(')) {
                    value += this.skip('(', ')');
                } else {
                    const resultType = this.parseType();
                    if (resultType) {
                        value += resultType;
                    }
                }
            }
            return value;
        }
        if (this.match('<')) {
            return this.skip('<', '>');
        }
        throw new mlir.Error(`Invalid type '${this._token.value}' ${this.location()}`);
    }

    parseArgumentTypes(args) {
        let index = 0;
        const open = this.accept('(');
        if (open) {
            while (this._token.kind !== ')') {
                const type = this.parseType();
                if (!type) {
                    break;
                }
                if (index < args.length) {
                    args[index].type = type;
                } else {
                    const arg = {};
                    arg.type = type;
                    arg.value = `%${index}`;
                    args.push(arg);
                }
                index++;
                if (!this.accept(',')) {
                    break;
                }
            }
            this.expect(')');
        } else {
            while (!this.accept(')') &&
                this._token.kind !== '->' &&
                this._token.value !== 'loc' &&
                this._token.value !== 'return' && this._token.value !== 'func.return' &&
                this._token.value !== 'to' &&
                this._token.value !== 'into' &&
                this._token.kind !== '}' &&
                this._token.kind !== '%' &&
                this._token.kind !== '^') {
                const type = this.parseType();
                if (!type) {
                    break;
                }
                if (index < args.length) {
                    args[index].type = type;
                } else {
                    const input = {};
                    input.type = type;
                    input.value = `%${index}`;
                    args.push(input);
                }
                index++;
                if (!this.accept(',')) {
                    break;
                }
            }
        }
    }

    parseValue() {
        const value = {};
        if (this.match('string')) {
            value.value = this.expect();
            value.type = 'string';
            return value;
        }
        if (this.match('int')) {
            value.value = this.expect();
            value.type = 'int64';
            return value;
        }
        if (this.match('float')) {
            value.value = this.expect();
            value.type = 'float32';
            return value;
        }
        if (this.match('boolean')) {
            value.value = this.expect();
            value.type = 'boolean';
            return value;
        }
        if (this.match('%')) {
            value.value = this.expect();
            return value;
        }
        if (this.match('@')) {
            value.value = this.expect();
            return value;
        }
        if (this.match('id', 'DEFAULT')) {
            value.value = this.expect();
            return value;
        }
        if (this.match('<')) {
            value.value = this.skip('<', '>');
            return value;
        }
        if (this.match('id', 'affine_map') || this.match('id', 'affine_set')) {
            const name = this.expect();
            const args = this.skip('<', '>');
            if (this.match('(')) {
                const dimArgs = this.skip('(', ')');
                if (this.match('[')) {
                    const symbolArgs = this.skip('[', ']');
                    return { name, args, dimArgs, symbolArgs };
                }
                return { name, args, dimArgs };
            }
            return { name, args };
        }
        if (this.accept('id', 'array')) {
            this.expect('<');
            const arrayType = this.parseType();
            const arrayValues = [];
            if (this.accept(':')) {
                while (!this.match('>')) {
                    const val = this.parseValue();
                    arrayValues.push(val.value === undefined ? val : val.value);
                    this.accept(',');
                }
            }
            this.expect('>');
            return { value: arrayValues, type: arrayType };
        }
        if (this.accept('[')) {
            const list = [];
            while (!this.accept(']')) {
                const item = this.parseValue();
                if (this.accept(':')) {
                    this.parseType();
                }
                list.push(item.value);
                this.accept(',');
            }
            if (this.accept('id', 'x')) {
                list[0] = Array.from(list);
                const second = [];
                this.expect('[');
                while (!this.accept(']')) {
                    const item = this.parseValue();
                    if (this.accept(':')) {
                        this.parseType();
                    }
                    second.push(item.value);
                    this.accept(',');
                }
                list.push(second);
            }
            return { value: list };
        }
        if (this.match('{')) {
            const attributes = [];
            this.parseAttributeDict(attributes);
            const obj = {};
            for (const attribute of attributes) {
                obj[attribute.name] = attribute.value;
            }
            return { value: obj };
        }
        if (this.match('#')) {
            value.value = this.expect('#');
            if (this.match('<')) {
                value.value += this.skip('<', '>');
            }
            if (this.match('(')) {
                value.value += this.skip('(', ')');
            }
            return value;
        }
        if (this.match('tensor')) {
            value.value = this.parseType();
            value.type = 'type';
            return value;
        }
        if (this.accept('id', 'dense_resource')) {
            this.expect('<');
            const resourceHandle = this.expect();
            this.expect('>');
            return { value: resourceHandle, type: 'dense' };
        }
        if (this.accept('id', 'sparse')) {
            this.expect('<');
            value.type = 'sparse';
            const indices = this.parseValue();
            this.accept(',');
            const values = this.parseValue();
            this.expect('>');
            value.value = { indices: indices.value, values: values.value };
            return value;
        }
        if (this.accept('id', 'dense')) {
            this.expect('<');
            value.type = 'dense';
            if (this.accept('>')) {
                value.value = null;
                return value;
            }
            if (this.match('string')) {
                const hexStr = this.expect();
                if (hexStr.startsWith('"0x') || hexStr.startsWith('0x')) {
                    const cleanHex = hexStr.replace(/"/g, '').substring(2);
                    const data = new Uint8Array(cleanHex.length >> 1);
                    for (let i = 0; i < data.length; i++) {
                        const index = i << 1;
                        data[i] = parseInt(cleanHex.substring(index, index + 2), 16);
                    }
                    value.value = data;
                } else {
                    value.value = hexStr;
                }
            } else if (this.match('[')) {
                const arrayValue = this.parseValue();
                value.value = arrayValue.value;
            } else if (this.accept('(')) {
                const real = this.parseValue();
                this.accept(',');
                const imag = this.parseValue();
                this.expect(')');
                value.value = { real: real.value, imag: imag.value };
            } else {
                const scalarValue = this.parseValue();
                value.value = scalarValue.value;
            }
            this.expect('>');
            return value;
        }
        if (this._token.kind === 'id') {
            const tokenValue = this._token.value;
            if (tokenValue === 'tensor' || tokenValue === 'vector' || tokenValue === 'memref' ||
                tokenValue === 'none' || tokenValue === 'index' || /^[su]?i[0-9]+$/.test(tokenValue) ||
                /^f[0-9]+$/.test(tokenValue) || tokenValue === 'bf16' || tokenValue === 'tf32' ||
                tokenValue.startsWith('f8')) {
                const type = this.parseType();
                return { value: type, type: 'type' };
            }
        }
        if (this.match('!')) {
            const type = this.parseType();
            return { value: type, type: 'type' };
        }
        if (this.match('id')) {
            value.value = this.expect('id');
            if (this.match('<')) {
                value.value += this.skip('<', '>');
            }
            return value;
        }
        if (this.accept('(')) {
            const real = this.parseValue();
            this.accept(',');
            const imag = this.parseValue();
            this.expect(')');
            return { value: { real: real.value, imag: imag.value }, type: 'complex' };
        }
        throw new mlir.Error(`Unexpected value '${this._token.value}' ${this.location()}`);
    }

    parseAttributeValue() {
        if (this.match('keyword', 'loc')) {
            return this.parseLocation();
        }
        if (this.match('id', 'affine_map') || this.match('id', 'affine_set')) {
            const name = this.expect();
            const args = this.skip('<', '>');
            return { name, args };
        }
        if (this.match('#')) {
            const name = this.expect();
            if (this.accept('<')) {
                const bracketStack = ['<'];
                while (bracketStack.length > 0) {
                    if (this.match('eof')) {
                        throw new mlir.Error(`Unexpected end of file while parsing attribute ${this.location()}`);
                    }
                    const token = this._token.kind;
                    if (token === '<' || token === '{' || token === '[' || token === '(') {
                        bracketStack.push(token);
                        this._token = this._tokenizer.read();
                    } else if (token === '>') {
                        if (bracketStack[bracketStack.length - 1] === '<') {
                            bracketStack.pop();
                            if (bracketStack.length === 0) {
                                this.expect('>');
                            } else {
                                this._token = this._tokenizer.read();
                            }
                        } else {
                            this._token = this._tokenizer.read();
                        }
                    } else if (token === '}') {
                        if (bracketStack[bracketStack.length - 1] === '{') {
                            bracketStack.pop();
                        }
                        this._token = this._tokenizer.read();
                    } else if (token === ']') {
                        if (bracketStack[bracketStack.length - 1] === '[') {
                            bracketStack.pop();
                        }
                        this._token = this._tokenizer.read();
                    } else if (token === ')') {
                        if (bracketStack[bracketStack.length - 1] === '(') {
                            bracketStack.pop();
                        }
                        this._token = this._tokenizer.read();
                    } else {
                        this._token = this._tokenizer.read();
                    }
                }
            }
            return { name };
        }
        if (this.accept('<')) {
            const dict = {};
            while (!this.match('>')) {
                if (this.match('id')) {
                    const key = this.expect('id');
                    if (this.accept('=')) {
                        const value = this.parseAttributeValue();
                        dict[key] = value;
                    }
                }
                this.accept(',');
            }
            this.expect('>');
            return dict;
        }
        if (this.match('(')) {
            const parts = [];
            let depth = 0;
            let seenArrow = false;
            while (true) {
                if (this.match('(')) {
                    depth++;
                    parts.push(this.expect());
                } else if (this.match(')')) {
                    parts.push(this.expect());
                    depth--;
                    if (depth === 0) {
                        if (seenArrow) {
                            break;
                        } else if (!this.match('->')) {
                            break;
                        }
                    }
                } else if (this.match('->')) {
                    seenArrow = true;
                    parts.push(this.expect());
                } else {
                    parts.push(this.expect());
                }
            }
            return { affine_map: parts.join(' ') };
        }
        if (this._token.kind === 'id') {
            const value = this._token.value;
            if (value === 'tensor' || value === 'vector' || value === 'memref' ||
                value === 'none' || value === 'index' || /^[su]?i[0-9]+$/.test(value) ||
                /^f[0-9]+$/.test(value) || value === 'bf16') {
                return this.parseType();
            }
        }
        if (this.match('!')) {
            return this.parseType();
        }
        return this.parseValue();
    }

    match(kind, value) {
        return (this._token.kind === kind && (!value || this._token.value === value));
    }

    expect(kind, value) {
        if (kind && this._token.kind !== kind) {
            throw new mlir.Error(`Expected token of type '${kind}', but got '${this._token.value}' ${this.location()}`);
        }
        if (value && this._token.value !== value) {
            throw new mlir.Error(`Expected token with value '${value}', but got '${this._token.value}' ${this.location()}`);
        }
        const token = this._token;
        this._token = this._tokenizer.read();
        return token.value;
    }

    accept(kind, value) {
        if (this.match(kind, value)) {
            return this.expect();
        }
        return null;
    }

    get token() {
        return this._token;
    }

    location() {
        return this._tokenizer.location();
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
        const parseCustomEntry = (/* entry, reader, entryType */) => {
        };
        const parseAsmEntry = (/* entry, reader, entryType */) => {
        };
        const resolveEntries = (range, entryType) => {
            for (const entry of this.attributes) {
                reader.seek(offset + entry.offset);
                if (entry.hasCustomEncoding) {
                    parseCustomEntry(entry, reader);
                } else {
                    parseAsmEntry(entry, reader, entryType);
                }
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
            case 'f4E2M1FN': return 'float4e2m1fn';
            case 'f6E2M3FN': return 'float6e2m3fn';
            case 'f6E3M2FN': return 'float6e3m2fn';
            case 'f8E3M4': return 'float8e3m4';
            case 'f8E4M3': return 'float8e4m3';
            case 'f8E4M3B11FNUZ': return 'float8e4m3b11fnuz';
            case 'f8E4M3FN': return 'float8e4m3fn';
            case 'f8E4M3FNUZ': return 'float8e4m3fnuz';
            case 'f8E5M2': return 'float8e5m2';
            case 'f8E5M2FNUZ': return 'float8e5m2fnuz';
            case 'f8E8M0FNU': return 'float8e8m0fnu';
            case 'float8': return 'float8';
            case 'tf32': return 'tensorfloat32';
            case 'i1': return 'boolean';
            case 'i2': return 'int2';
            case 'i4': return 'int4';
            case 'i8': return 'int8';
            case 'i16': return 'int16';
            case 'i32': return 'int32';
            case 'i48': return 'int48';
            case 'i64': return 'int64';
            case 'si8': return 'int8';
            case 'si16': return 'int16';
            case 'si32': return 'int32';
            case 'si64': return 'int64';
            case 'ui1': return 'uint1';
            case 'ui2': return 'uint2';
            case 'ui4': return 'uint4';
            case 'ui8': return 'uint8';
            case 'ui16': return 'uint16';
            case 'ui32': return 'uint32';
            case 'ui64': return 'uint64';
            case 'complex<f32>': return 'complex64';
            case 'complex<f64>': return 'complex128';
            case 'b8': return 'int8';
            case 'boolean': return 'boolean';
            default:
                if (value && value.startsWith('!')) {
                    // Workaround for !quant.uniform
                    return value;
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
        // eg. tensor<?x3x2x2xf32> or tensor<20x20xcomplex<f64>>
        if (type.startsWith('tensor<') && type.endsWith('>')) {
            const spec = type.substring(7, type.length - 1).trim();
            if (spec.startsWith('!')) {
                return mlir.Utility.valueType(spec);
            }
            // Parse dimensions from left to right until we hit the element type
            // Dimensions are numbers, '?', or '*' separated by 'x'
            let i = 0;
            const shape = [];
            while (i < spec.length) {
                if (spec[i] === '?' || spec[i] === '*') {
                    shape.push('?');
                    i++;
                } else if (/[0-9]/.test(spec[i])) {
                    let numStr = '';
                    while (i < spec.length && /[0-9]/.test(spec[i])) {
                        numStr += spec[i];
                        i++;
                    }
                    const dim = parseInt(numStr, 10);
                    if (isNaN(dim)) {
                        // This shouldn't happen, but handle it gracefully
                        shape.push('?');
                    } else {
                        shape.push(dim);
                    }
                } else {
                    // Not a dimension, rest is the element type
                    break;
                }
                // Skip 'x' separator
                if (i < spec.length && spec[i] === 'x') {
                    i++;
                } else {
                    break;
                }
            }
            let dataType = spec.substring(i);
            // Strip encoding if present (e.g., "f32, #stablehlo.type_extensions<...>")
            const encodingIndex = dataType.indexOf(',');
            if (encodingIndex !== -1) {
                dataType = dataType.substring(0, encodingIndex).trim();
            }
            return new mlir.TensorType(dataType, new mlir.TensorShape(shape));
        }
        if (type.startsWith('!torch.vtensor<') && type.endsWith('>')) {
            const spec = type.substring(15, type.length - 1);
            const index = spec.lastIndexOf(',');
            const shapeStr = spec.substring(0, index).replace(/\?/g, '"?"');
            const shape = JSON.parse(shapeStr);
            const dataType = spec.substring(index + 1);
            return new mlir.TensorType(dataType, new mlir.TensorShape(shape));
        }
        // Handle tuple types: tuple<tensor<f32>, tensor<i32>>
        if (type.startsWith('tuple<') && type.endsWith('>')) {
            // const spec = type.substring(6, type.length - 1).trim();
            // For now, return as-is since we don't have a TupleType class
            // In the future, we could parse the comma-separated types
            return type;
        }
        return type;
    }
};

// Dialect Plugin System

mlir.AssemblyFormatParser = class {

    constructor(metadata) {
        this._metadata = metadata;
        this._format = metadata.assemblyFormat || '';
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
        // Optional group: (...)?
        // Check if this is the start of an optional group by looking ahead for `?` after matching `)`
        if (ch === '(') {
            // Look ahead to see if this is an optional group
            const savedPos = this._pos;
            this._pos++; // skip '('
            let depth = 1;
            let tempPos = this._pos;
            // Find matching ')'
            while (tempPos < this._format.length && depth > 0) {
                if (this._format[tempPos] === '(') {
                    depth++;
                } else if (this._format[tempPos] === ')') {
                    depth--;
                }
                tempPos++;
            }
            // Check if there's a '?' after the ')'
            tempPos = this._skipWhitespaceAt(tempPos);
            const isOptional = tempPos < this._format.length && this._format[tempPos] === '?';

            if (isOptional) {
                // This is an optional group - parse it as a single directive
                const elements = [];
                let anchorElement = null;
                this._skipWhitespace();
                let loopCount = 0;
                const maxLoops = 100;
                while (this._pos < this._format.length && this._format[this._pos] !== ')') {
                    if (loopCount++ > maxLoops) {
                        throw new Error(`Infinite loop detected in optional group parsing at position ${this._pos}`);
                    }
                    const startPos = this._pos;
                    const elem = this._parseDirective();
                    if (elem) {
                        // Check if this element has an anchor marker
                        if (elem.anchor) {
                            anchorElement = elem.name || elem.type;
                        }
                        elements.push(elem);
                    }
                    // Safety check: ensure we're making progress
                    if (this._pos === startPos) {
                        // If we didn't advance, skip one character to prevent infinite loop
                        this._pos++;
                    }
                    this._skipWhitespace();
                }
                this._pos++; // skip ')'
                this._skipWhitespace();
                this._pos++; // skip '?'
                return { type: 'optional_group', elements, anchor: anchorElement };
            }
            // Not an optional group, restore position and treat '(' as literal
            this._pos = savedPos;
        }
        // Literal: `keyword`
        if (ch === '`') {
            this._pos++;
            const value = this._parseUntil('`');
            this._pos++; // skip closing `
            // MLIR reference: Empty literals (`` or ` `) are whitespace, not literals
            if (value.length === 0 || value === ' ' || value === '\\n') {
                return null; // Skip whitespace elements
            }
            return { type: 'literal', value };
        }
        // Variable reference: $name with optional anchor ^
        if (ch === '$') {
            this._pos++;
            const name = this._parseIdentifier();
            // Check for anchor marker ^
            let hasAnchor = false;
            if (this._pos < this._format.length && this._format[this._pos] === '^') {
                hasAnchor = true;
                this._pos++; // consume '^'
            }
            // Determine if name is an attribute, input, region, or successor
            const metadata = this._metadata;
            if ((metadata.attributes && metadata.attributes.some((a) => a.name === name)) ||
                (metadata.inputs && metadata.inputs.some((a) => a.name === name)) ||
                (metadata.regions && metadata.regions.some((a) => a.name === name)) ||
                (metadata.successors && metadata.successors.some((a) => a.name === name))) {
                return { type: 'operand_ref', name, anchor: hasAnchor };
            }
            if (name === 'operands') {
                return { type: 'operands', anchor: hasAnchor };
            }
            if (name === 'results') {
                return { type: 'results', anchor: hasAnchor };
            }
            if (name === 'regions') {
                return { type: 'regions', anchor: hasAnchor };
            }
            if (name === 'successors') {
                return { type: 'successors', anchor: hasAnchor };
            }
            // Otherwise, it's a named operand/attribute reference
            return { type: 'operand_ref', name, anchor: hasAnchor };
        }
        // Check for keywords
        const remaining = this._format.substring(this._pos);
        if (remaining.startsWith('type(')) {
            this._pos += 'type'.length;
            const args = this._parseParenList();
            return { type: 'type', args };
        }
        if (remaining.startsWith('qualified(')) {
            this._pos += 'qualified'.length;
            const args = this._parseParenList();
            return { type: 'qualified', args };
        }
        if (remaining.startsWith('attr-dict-with-keyword')) {
            this._pos += 'attr-dict-with-keyword'.length;
            return { type: 'attr_dict_with_keyword' };
        }
        if (remaining.startsWith('attr-dict')) {
            this._pos += 'attr-dict'.length;
            return { type: 'attr_dict' };
        }
        if (remaining.startsWith('operands')) {
            this._pos += 'operands'.length;
            return { type: 'operands' };
        }
        if (remaining.startsWith('results')) {
            this._pos += 'results'.length;
            return { type: 'results' };
        }
        if (remaining.startsWith('regions')) {
            this._pos += 'regions'.length;
            return { type: 'regions' };
        }
        if (remaining.startsWith('successors')) {
            this._pos += 'successors'.length;
            return { type: 'successors' };
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
        if (remaining.startsWith('oilist(')) {
            this._pos += 'oilist('.length;
            let content = '';
            let depth = 1;
            while (this._pos < this._format.length && depth > 0) {
                const ch = this._format[this._pos];
                if (ch === '(') {
                    depth++;
                    content += ch;
                    this._pos++;
                } else if (ch === ')') {
                    depth--;
                    if (depth > 0) {
                        content += ch;
                    }
                    this._pos++;
                } else {
                    content += ch;
                    this._pos++;
                }
            }
            return { type: 'oilist', content };
        }
        if (/^[:()[\]{}<>,=|]/.test(ch)) {
            this._pos++;
            return { type: 'literal', value: ch };
        }
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
        this._pos++;
        const items = [];
        let parenDepth = 1;
        while (this._pos < this._format.length && parenDepth > 0) {
            this._skipWhitespace();
            const startPos = this._pos;
            if (this._format[this._pos] === '$') {
                this._pos++;
                const item = this._parseIdentifier();
                if (item) {
                    items.push(`$${item}`);
                }
            } else if (this._format[this._pos] === '(') {
                parenDepth++;
                this._pos++;
            } else if (this._format[this._pos] === ')') {
                parenDepth--;
                if (parenDepth > 0) {
                    this._pos++;
                }
            } else if (this._format[this._pos] === ',') {
                this._pos++;
            } else {
                const item = this._parseIdentifier();
                if (item) {
                    items.push(item);
                }
                if (this._pos === startPos) {
                    this._pos++;
                }
            }
        }
        if (this._pos < this._format.length && this._format[this._pos] === ')') {
            this._pos++;
        }
        return items;
    }

    _skipWhitespace() {
        while (this._pos < this._format.length && /\s/.test(this._format[this._pos])) {
            this._pos++;
        }
    }

    _skipWhitespaceAt(pos) {
        while (pos < this._format.length && /\s/.test(this._format[pos])) {
            pos++;
        }
        return pos;
    }
};

mlir.Dialect = class {

    constructor(name, operations) {
        this._name = name;
        this._operations = new Map();
        this._customParsers = new Map();
        this.registerCustomParser('SameOperandsAndResultType', this._parseSameOperandsAndResultType.bind(this));
        this.registerCustomParser('VariadicSameOperandsAndResultType', this._parseVariadicSameOperandsAndResultType.bind(this));
        this.registerCustomParser('ComplexOpType', this._parseComplexOpType.bind(this));
        this.registerCustomParser('SelectOpType', this._parseSelectOpType.bind(this));
        this.registerCustomParser('TupleOpType', this._parseTupleOpType.bind(this));
        this.registerCustomParser('PairwiseOpType', this._parsePairwiseOpType.bind(this));
        this.registerCustomParser('ConvolutionDimensions', this._parseConvolutionDimensions.bind(this));
        this.registerCustomParser('DotDimensionNumbers', this._parseDotDimensionNumbers.bind(this));
        this.registerCustomParser('PrecisionConfig', this._parsePrecisionConfig.bind(this));
        this.registerCustomParser('PrecisionConfigAndAlgorithm', this._parsePrecisionConfigAndAlgorithm.bind(this));
        this.registerCustomParser('WindowAttributes', this._parseWindowAttributes.bind(this));
        this.registerCustomParser('SliceRanges', this._parseSliceRanges.bind(this));
        this.registerCustomParser('CustomCallTarget', this._parseCustomCallTarget.bind(this));
        this.registerCustomParser('VariadicOperandWithAttribute', this._parseVariadicOperandWithAttribute.bind(this));
        this.registerCustomParser('DynamicIndexList', this._parseDynamicIndexList.bind(this));
        this.registerCustomParser('Offsets', this._parseOffsets.bind(this));
        this.registerCustomParser('SymbolVisibility', this._parseSymbolVisibility.bind(this));
        this.registerCustomParser('SymbolAlias', this._parseSymbolAlias.bind(this));
        this.registerCustomParser('WorkgroupCountRegion', this._parseWorkgroupCountRegion.bind(this));
        this.registerCustomParser('OptionalUnitAttr', this._parseOptionalUnitAttr.bind(this));
        for (const metadata of operations.get(name) || []) {
            const op = { metadata };
            if (metadata.assemblyFormat) {
                const parser = new mlir.AssemblyFormatParser(metadata);
                op.directives = parser.parse();
            }
            this._operations.set(metadata.name, op);
        }
    }

    registerCustomParser(name, parserFn) {
        this._customParsers.set(name, parserFn);
    }

    parseType(/* parser, dialectType */) {
        // Default implementation: return null to indicate no custom parsing
        // Subclasses should override this to provide dialect-specific type parsing
        return null;
    }

    hasParser(opName) {
        if (this._operations.has(opName)) {
            const op = this._operations.get(opName);
            return op.metadata.hasCustomAssemblyFormat || op.metadata.parser;
        }
        return null;
    }

    hasAssemblyFormat(opName) {
        if (this._operations.has(opName)) {
            const op = this._operations.get(opName);
            return op.metadata.assemblyFormat;
        }
        return false;
    }
    hasParseOperation(opName) {
        if (this._operations.has(opName)) {
            const op = this._operations.get(opName);
            return op.hasParseOperation;
        }
        return false;
    }

    parseOperation(parser, opName, op) {
        const opInfo = this._operations.get(opName);
        if (!opInfo) {
            return false;
        }
        if (!this.hasAssemblyFormat(opName) && this.hasParser(opName) && this.hasParseOperation(opName)) {
            throw new mlir.Error(`Operation '${opName}' parser is not implemented.`);
        }
        const directives = opInfo.directives || [];
        for (let i = 0; i < directives.length; i++) {
            const directive = directives[i];
            opInfo.hasParseOperation = false;
            switch (directive.type) {
                case 'literal':
                    parser.expect(null, directive.value);
                    break;
                case 'operand_ref': {
                    const refName = directive.name;
                    let isAttribute = false;
                    let isVariadic = false;
                    let isSuccessor = false;
                    if (opInfo.metadata && opInfo.metadata.successors) {
                        const successorInfo = opInfo.metadata.successors.find((succ) => succ.name === refName);
                        if (successorInfo) {
                            isSuccessor = true;
                        }
                    }
                    if (!isSuccessor && parser.match('^')) {
                        isSuccessor = true;
                    }
                    if (!isSuccessor && opInfo.metadata && opInfo.metadata.attributes) {
                        const attrInfo = opInfo.metadata.attributes.find((attr) => attr.name === refName);
                        if (attrInfo) {
                            isAttribute = true;
                        }
                    }
                    if (!isAttribute && !isSuccessor && opInfo.metadata && opInfo.metadata.inputs) {
                        const inputInfo = opInfo.metadata.inputs.find((inp) => inp.name === refName);
                        if (inputInfo && inputInfo.type === 'Variadic') {
                            isVariadic = true;
                        }
                    }
                    if (isSuccessor) {
                        if (parser.match('^')) {
                            if (!op.successors) {
                                op.successors = [];
                            }
                            const successor = {};
                            successor.label = parser.expect('^');
                            if (parser.accept('(')) {
                                successor.arguments = [];
                                while (!parser.match(':') && !parser.match(')')) {
                                    if (parser.match('%')) {
                                        const arg = {};
                                        arg.value = parser.expect('%');
                                        successor.arguments.push(arg);
                                        parser.accept(',');
                                    } else {
                                        break;
                                    }
                                }
                                if (parser.accept(':')) {
                                    let idx = 0;
                                    while (idx < successor.arguments.length && !parser.match(')')) {
                                        const type = parser.parseType();
                                        successor.arguments[idx].type = type;
                                        idx++;
                                        parser.accept(',');
                                    }
                                }
                                parser.accept(')');
                            }
                            op.successors.push(successor);
                        }
                    } else if (isAttribute) {
                        const attrValue = parser.match('#') ? parser.parseAttributeValue() : parser.parseValue();
                        if (attrValue) {
                            let nextIsColonLiteral = false;
                            for (let j = i + 1; j < opInfo.directives.length; j++) {
                                const nextDir = directives[j];
                                if (nextDir.type !== 'attr_dict') {
                                    if (nextDir.type === 'literal' && nextDir.value === ':') {
                                        nextIsColonLiteral = true;
                                    }
                                    break; // Only check the next non-attr-dict directive
                                }
                            }
                            if (!nextIsColonLiteral && (attrValue.type === 'int64' || attrValue.type === 'float32' || attrValue.type === 'boolean' || attrValue.type === 'dense' || attrValue.type === 'sparse' || attrValue.name) && parser.accept(':')) {
                                attrValue.attrType = parser.parseType();
                            }
                            op.attributes.push({ name: refName, value: attrValue.value });
                        }
                    } else if (isVariadic) {
                        while (!parser.match(')') && !parser.match(']') && !parser.match('}') && !parser.match(':')) {
                            if (parser.match('%')) {
                                const input = {};
                                input.value = parser.expect();
                                op.operands.push(input);
                            } else {
                                break;
                            }
                            parser.accept(',');
                        }
                    } else if (parser.match('%')) {
                        const input = {};
                        input.value = parser.expect();
                        op.operands.push(input);
                    } else if ((refName === 'region' || refName === 'regions' || refName.endsWith('Region') || refName.endsWith('Regions')) && parser.match('{')) {
                        const region = {};
                        parser.parseRegion(region);
                        op.regions.push(region);
                    } else if (parser.match('{')) {
                        const region = {};
                        parser.parseRegion(region);
                        op.regions.push(region);
                    } else if (parser.match('@')) {
                        const value = parser.expect('@');
                        if (directive.name) {
                            op.attributes.push({ name: directive.name, value });
                        } else {
                            op.attributes.push({ name: 'callee', value });
                        }
                    } else if (parser.match('id') && !(refName === 'region' || refName === 'regions' || refName.endsWith('Region') || refName.endsWith('Regions'))) {
                        const input = {};
                        input.value = parser.expect('id');
                        op.operands.push(input);
                    } else if (parser.match('int')) {
                        const input = {};
                        input.value = parser.expect('int');
                        op.operands.push(input);
                    } else if (!parser.match(':') && !parser.match(')') && !parser.match(']')) {
                        // Try to parse as a general value, but not if we're at a delimiter
                        const input = parser.parseValue();
                        if (input) {
                            op.operands.push(input);
                        }
                    }
                    break;
                }
                case 'operands':
                    op.operands = parser.parseArguments();
                    break;
                case 'results':
                    op.results = parser.parseArguments();
                    break;
                case 'type':
                case 'qualified':
                    if (directive.args && directive.args.length > 0) {
                        const arg = directive.args[0] === 'type' && directive.args.length > 1 ? directive.args[1] : directive.args[0];
                        if (arg === 'results' || arg === '$results') {
                            const opMetadata = opInfo.metadata;
                            const hasVariadicResult = opMetadata && opMetadata.outputs && (opMetadata.outputs.length > 1 || (opMetadata.outputs.length === 1 && (opMetadata.outputs[0].type === 'Variadic' || opMetadata.outputs[0].isVariadic)));
                            if (hasVariadicResult) {
                                parser.parseArgumentTypes(op.results);
                            } else {
                                const type = parser.parseType();
                                if (op.results.length === 0) {
                                    op.results.push({ type });
                                } else {
                                    op.results[0].type = type;
                                }
                            }
                        } else if (arg === 'operands' || arg === '$operands') {
                            parser.parseArgumentTypes(op.operands);
                        } else {
                            const opMetadata = opInfo.metadata;
                            let isResult = false;
                            let isVariadic = false;
                            if (opMetadata && opMetadata.outputs) {
                                for (const output of opMetadata.outputs) {
                                    if (output.name === arg || `$${output.name}` === arg) {
                                        isResult = true;
                                        isVariadic = output.type === 'Variadic' || output.isVariadic || false;
                                        break;
                                    }
                                }
                            }
                            if (!isResult && opMetadata && opMetadata.inputs) {
                                for (const input of opMetadata.inputs) {
                                    if (input.name === arg || `$${input.name}` === arg) {
                                        isVariadic = input.type === 'Variadic' || input.isVariadic || false;
                                        break;
                                    }
                                }
                            }
                            if (isResult) {
                                if (isVariadic) {
                                    parser.parseArgumentTypes(op.results);
                                } else {
                                    const type = parser.parseType();
                                    if (op.results.length === 0) {
                                        op.results.push({ type });
                                    } else {
                                        op.results[0].type = type;
                                    }
                                }
                            } else if (isVariadic) {
                                parser.parseArgumentTypes(op.operands);
                            } else if (op.operands.length > 0) {
                                op.operands[op.operands.length - 1].type = parser.parseType();
                            } else {
                                parser.parseType();
                            }
                        }
                    } else {
                        parser.parseArgumentTypes(op.operands);
                    }
                    break;
                case 'attr_dict_with_keyword':
                    if (parser.match('id') && parser._token.value === 'attributes') {
                        parser.expect('id');
                        parser.parseAttributeDict(op.attributes);
                    }
                    break;
                case 'attr_dict':
                    parser.parseAttributeDict(op.attributes);
                    break;
                case 'regions':
                    while (parser.match('{')) {
                        const region = {};
                        parser.parseRegion(region);
                        op.regions.push(region);
                        if (!parser.accept(',')) {
                            break;
                        }
                    }
                    break;
                case 'successors':
                    if (parser.match('[')) {
                        parser.skip('[', ']');
                    }
                    break;
                case 'functional_type': {
                    parser.parseArgumentTypes(op.operands);
                    if (parser.accept('->') || parser.accept('id', 'to')) {
                        if (op.results.length > 0) {
                            parser.parseArgumentTypes(op.results);
                        } else {
                            op.results = parser.parseArguments();
                        }
                    }
                    break;
                }
                case 'custom': {
                    const fn = this._customParsers.get(directive.parser);
                    if (!fn) {
                        break;
                    }
                    const result = fn(parser, directive.args);
                    if (result) {
                        if (result.kind === 'SymbolVisibility' && result.visibility && directive.args && directive.args.length > 0) {
                            const attrName = directive.args[0].replace(/^\$/, '');
                            op.attributes.push({ name: attrName, value: result.visibility });
                        } else if (result.kind === 'SymbolAlias' && result.symbolName) {
                            if (result.symNameArg) {
                                op.attributes.push({ name: result.symNameArg, value: result.symbolName });
                            }
                            if (result.functionRefArg) {
                                op.attributes.push({ name: result.functionRefArg, value: result.symbolName });
                            }
                        } else if (result.kind === 'WorkgroupCountRegion' && result.region) {
                            op.regions.push(result.region);
                        } else if (result.kind === 'SameOperandsAndResultType' && result.type) {
                            for (const operand of op.operands) {
                                if (!operand.type) {
                                    operand.type = result.type;
                                }
                            }
                            for (const res of op.results) {
                                if (!res.type) {
                                    res.type = result.type;
                                }
                            }
                        } else if (result.kind === 'PairwiseOpType') {
                            if (result.operandTypes && op.operands.length > 0) {
                                for (let i = 0; i < Math.min(result.operandTypes.length, op.operands.length); i++) {
                                    if (!op.operands[i].type) {
                                        op.operands[i].type = result.operandTypes[i];
                                    }
                                }
                            }
                            if (result.resultTypes && op.results.length > 0) {
                                for (let i = 0; i < Math.min(result.resultTypes.length, op.results.length); i++) {
                                    if (!op.results[i].type) {
                                        op.results[i].type = result.resultTypes[i];
                                    }
                                }
                            }
                        } else if (result.kind === 'PrecisionConfig' && result.precision && result.precision.length > 0) {
                            op.attributes.push({ name: 'precision_config', value: result.precision });
                        } else if (result.kind === 'PrecisionConfigAndAlgorithm') {
                            if (result.precision && result.precision.length > 0) {
                                op.attributes.push({ name: 'precision_config', value: result.precision });
                            }
                            if (result.algorithm) {
                                op.attributes.push({ name: 'algorithm', value: result.algorithm });
                            }
                        } else if (result.kind === 'DispatchEntryPoints' && result.entryPoints) {
                            const attrName = result.attrName || 'entry_points';
                            // Store as array if multiple, or single value if one
                            const value = result.entryPoints.length === 1 ? result.entryPoints[0] : result.entryPoints;
                            op.attributes.push({ name: attrName, value });
                        } else if (result.kind === 'ShapedFunctionType') {
                            // Apply operand types
                            if (result.operandTypes && result.operandTypes.length > 0) {
                                // Find the actual argument operands (skip workload operands)
                                // We need to figure out which operands are arguments vs workload
                                // For now, apply types to the last N operands where N = operandTypes.length
                                const startIdx = Math.max(0, op.operands.length - result.operandTypes.length);
                                for (let i = 0; i < result.operandTypes.length; i++) {
                                    if (startIdx + i < op.operands.length && !op.operands[startIdx + i].type) {
                                        op.operands[startIdx + i].type = result.operandTypes[i];
                                    }
                                }
                            }
                            // Apply result types
                            if (result.resultTypes && result.resultTypes.length > 0) {
                                for (let i = 0; i < result.resultTypes.length; i++) {
                                    if (i < op.results.length) {
                                        op.results[i].type = result.resultTypes[i];
                                    } else {
                                        op.results.push({ type: result.resultTypes[i] });
                                    }
                                }
                            }
                        } else if (result.kind === 'VariadicOperandWithAttribute' && result.operands) {
                            for (const operand of result.operands) {
                                op.operands.push(operand);
                            }
                        }
                    }
                    break;
                }
                case 'oilist': {
                    break;
                }
                case 'optional_group': {
                    let shouldParse = false;
                    const [firstElem] = directive.elements;
                    if (firstElem) {
                        if (firstElem.type === 'literal') {
                            if (firstElem.value.length === 1 && /[(){}[\],:<>=]/.test(firstElem.value)) {
                                shouldParse = parser.match(firstElem.value);
                            } else if (firstElem.value === '->') {
                                shouldParse = parser.match('->');
                            } else {
                                shouldParse = parser.match('id', firstElem.value) || parser.match('keyword', firstElem.value);
                            }
                        } else if (firstElem.type === 'operand_ref') {
                            if (firstElem.name === 'overflowFlags') {
                                shouldParse = parser.match('id', 'overflow');
                            } else {
                                // Check if this is an attribute or an operand
                                let isFirstAttribute = false;
                                if (opInfo.metadata && opInfo.metadata.attributes) {
                                    const attrInfo = opInfo.metadata.attributes.find((attr) => attr.name === firstElem.name);
                                    if (attrInfo) {
                                        isFirstAttribute = true;
                                    }
                                }
                                if (isFirstAttribute) {
                                    // For symbol attributes like sym_name, check for '@'
                                    // For other attributes, check if there's a value present
                                    if (firstElem.name === 'sym_name') {
                                        shouldParse = parser.match('@');
                                    } else if (firstElem.name === 'sym_visibility') {
                                        shouldParse = parser.match('@') || parser.match('string') || parser.match('id', 'private') || parser.match('id', 'public') || parser.match('id', 'nested');
                                    } else {
                                        shouldParse = parser.match('id') || parser.match('int') || parser.match('float') || parser.match('[') || parser.match('@');
                                    }
                                } else {
                                    // For operands, check for %
                                    shouldParse = parser.match('%');
                                }
                            }
                        } else if (firstElem.type === 'operands') {
                            shouldParse = parser.match('(') || parser.match('%');
                        }
                    }
                    if (shouldParse) {
                        for (const elem of directive.elements) {
                            switch (elem.type) {
                                case 'literal':
                                    parser.expect(null, elem.value);
                                    break;
                                case 'operand_ref': {
                                    const refName = elem.name;
                                    if (refName === 'overflowFlags') {
                                        parser.expect('id', 'overflow');
                                        parser.expect('<');
                                        const flags = [];
                                        while (!parser.match('>')) {
                                            flags.push(parser.expect('id'));
                                            parser.accept(',');
                                        }
                                        parser.expect('>');
                                        op.attributes.push({ name: 'overflowFlags', value: flags.join(', ') });
                                        break;
                                    }
                                    let isAttribute = false;
                                    let isVariadic = false;
                                    if (opInfo.metadata && opInfo.metadata.attributes) {
                                        const attrInfo = opInfo.metadata.attributes.find((attr) => attr.name === refName);
                                        if (attrInfo) {
                                            isAttribute = true;
                                        }
                                    }
                                    if (!isAttribute && opInfo.metadata && opInfo.metadata.inputs) {
                                        const inputInfo = opInfo.metadata.inputs.find((inp) => inp.name === refName);
                                        if (inputInfo && inputInfo.type === 'Variadic') {
                                            isVariadic = true;
                                        }
                                    }
                                    if (isAttribute) {
                                        let attrValue = null;
                                        if (elem.anchor && parser.match('<')) {
                                            parser.expect('<');
                                            const flags = [];
                                            while (!parser.match('>')) {
                                                if (parser.match('id')) {
                                                    flags.push(parser.expect('id'));
                                                } else if (parser.match('int')) {
                                                    flags.push(parser.expect('int'));
                                                } else if (parser.match('string')) {
                                                    flags.push(parser.expect('string'));
                                                } else {
                                                    break;
                                                }
                                                parser.accept(',');
                                            }
                                            parser.expect('>');
                                            attrValue = flags.length === 1 ? flags[0] : flags.join(',');
                                        } else {
                                            const value = parser.parseValue();
                                            attrValue = value.value === undefined ? value : value.value;
                                        }
                                        op.attributes.push({ name: refName, value: attrValue });
                                    } else if (isVariadic) {
                                        while (parser.match('%')) {
                                            const operand = parser.parseValue();
                                            op.operands.push(operand);
                                            if (!parser.accept(',')) {
                                                break;
                                            }
                                        }
                                    } else if (parser.match('%') || parser.match('@') || parser.match('#') || parser.match('[')) {
                                        const operand = parser.parseValue();
                                        op.operands.push(operand);
                                        while (parser.accept(',') && parser.match('%')) {
                                            const nextOperand = parser.parseValue();
                                            op.operands.push(nextOperand);
                                        }
                                    }
                                    break;
                                }
                                case 'operands': {
                                    op.operands = parser.parseArguments();
                                    break;
                                }
                                case 'type': {
                                    if (elem.args && elem.args.length > 0) {
                                        const [arg] = elem.args;
                                        if (arg.startsWith('$')) {
                                            const varName = arg.substring(1);
                                            if (varName === 'result' || varName === 'results') {
                                                let isInput = false;
                                                if (opInfo.metadata && opInfo.metadata.inputs) {
                                                    isInput = opInfo.metadata.inputs.some((inp) => inp.name === varName);
                                                }
                                                if (isInput) {
                                                    parser.parseArgumentTypes(op.operands);
                                                } else {
                                                    parser.parseArgumentTypes(op.results);
                                                }
                                            } else if (varName === 'operands') {
                                                parser.parseArgumentTypes(op.operands);
                                            } else {
                                                let isVariadic = false;
                                                if (opInfo.metadata && opInfo.metadata.inputs) {
                                                    const input = opInfo.metadata.inputs.find((inp) => inp.name === varName);
                                                    if (input && input.type === 'Variadic') {
                                                        isVariadic = true;
                                                    }
                                                }
                                                if (isVariadic) {
                                                    parser.parseArgumentTypes(op.operands);
                                                } else {
                                                    const type = parser.parseType();
                                                    if (op.operands.length > 0) {
                                                        op.operands[op.operands.length - 1].type = type;
                                                    }
                                                }
                                            }
                                        }
                                    }
                                    break;
                                }
                                case 'results': {
                                    while (parser.match('%')) {
                                        parser.expect();
                                        parser.accept(',');
                                    }
                                    break;
                                }
                                case 'custom': {
                                    const fn = this._customParsers.get(elem.parser);
                                    if (!fn) {
                                        break;
                                    }
                                    const result = fn(parser, elem.args);
                                    if (result && result.kind === 'SameOperandsAndResultType' && result.type) {
                                        for (const operand of op.operands) {
                                            if (!operand.type) {
                                                operand.type = result.type;
                                            }
                                        }
                                        for (const res of op.results) {
                                            if (!res.type) {
                                                res.type = result.type;
                                            }
                                        }
                                    } else if (result && result.kind === 'PairwiseOpType') {
                                        if (result.operandTypes && op.operands.length > 0) {
                                            for (let i = 0; i < Math.min(result.operandTypes.length, op.operands.length); i++) {
                                                if (!op.operands[i].type) {
                                                    op.operands[i].type = result.operandTypes[i];
                                                }
                                            }
                                        }
                                        if (result.resultTypes && op.results.length > 0) {
                                            for (let i = 0; i < Math.min(result.resultTypes.length, op.results.length); i++) {
                                                if (!op.results[i].type) {
                                                    op.results[i].type = result.resultTypes[i];
                                                }
                                            }
                                        }
                                    }
                                    break;
                                }
                                default: {
                                    throw new mlir.Error(`Unsupported directive type '${elem.type}' ${parser.location()}.`);
                                }
                            }
                        }
                    }
                    break;
                }
                default: {
                    throw new mlir.Error(`Unsupported directive type '${directive.type}' ${parser.location()}.`);
                }
            }
        }
        return true;
    }

    _parseSameOperandsAndResultType(parser, args) {
        const type = parser.parseType();
        return { kind: 'SameOperandsAndResultType', type, args };
    }

    _parseVariadicSameOperandsAndResultType(parser, /*, args */) {
        const type = parser.parseType();
        return { kind: 'VariadicSameOperandsAndResultType', type };
    }

    _parseComplexOpType(parser, /*, args */) {
        const type = parser.parseType();
        return { kind: 'ComplexOpType', type };
    }

    _parseSelectOpType(parser, /*, args */) {
        const firstType = parser.parseType();
        if (parser.accept(',')) {
            const secondType = parser.parseType();
            return { kind: 'SelectOpType', predType: firstType, resultType: secondType };
        }
        return { kind: 'SelectOpType', functionType: firstType };
    }

    _parseTupleOpType(parser, /*, args */) {
        const type = parser.parseType();
        return { kind: 'TupleOpType', type };
    }

    _parsePairwiseOpType(parser /*, args */) {
        const operandTypes = [];
        const resultTypes = [];
        while (true) {
            const type = parser.parseType();
            if (!type) {
                break;
            }
            operandTypes.push(type);
            resultTypes.push(type);
            if (!parser.accept(',')) {
                break;
            }
        }

        return { kind: 'PairwiseOpType', operandTypes, resultTypes };
    }

    _parseConvolutionDimensions(parser, /*, args */) {
        const result = {
            kind: 'ConvolutionDimensions',
            input: [],
            kernel: [],
            output: []
        };

        if (parser.accept('[')) {
            while (!parser.match(']')) {
                if (parser.match('int') || parser.match('number')) {
                    result.input.push(parseInt(parser.expect(), 10));
                } else if (parser.match('id')) {
                    result.input.push(parser.expect('id'));
                } else {
                    break;
                }
                parser.accept(',');
            }
            parser.accept(']');
        }

        if (parser.accept('id', 'x')) {
            if (parser.accept('[')) {
                while (!parser.match(']')) {
                    if (parser.match('int') || parser.match('number')) {
                        result.kernel.push(parseInt(parser.expect(), 10));
                    } else if (parser.match('id')) {
                        result.kernel.push(parser.expect('id'));
                    } else {
                        break;
                    }
                    parser.accept(',');
                }
                parser.accept(']');
            }
        }

        if (parser.accept('->')) {
            if (parser.accept('[')) {
                while (!parser.match(']')) {
                    if (parser.match('int') || parser.match('number')) {
                        result.output.push(parseInt(parser.expect(), 10));
                    } else if (parser.match('id')) {
                        result.output.push(parser.expect('id'));
                    } else {
                        break;
                    }
                    parser.accept(',');
                }
                parser.accept(']');
            }
        }

        return result;
    }

    _parseDotDimensionNumbers(parser /*, args */) {
        const result = {
            kind: 'DotDimensionNumbers',
            lhs_batching_dimensions: [],
            rhs_batching_dimensions: [],
            lhs_contracting_dimensions: [],
            rhs_contracting_dimensions: []
        };

        const parsePair = () => {
            const first = [];
            const second = [];
            if (parser.accept('[')) {
                while (!parser.match(']')) {
                    if (parser.match('int')) {
                        first.push(parseInt(parser.expect('int'), 10));
                        parser.accept(',');
                    } else {
                        parser.expect();
                        parser.accept(',');
                    }
                }
                parser.accept(']');
            }
            if (parser.accept('id', 'x')) {
                if (parser.accept('[')) {
                    while (!parser.match(']')) {
                        if (parser.match('int')) {
                            second.push(parseInt(parser.expect('int'), 10));
                            parser.accept(',');
                        } else {
                            parser.expect();
                            parser.accept(',');
                        }
                    }
                    parser.accept(']');
                }
            }
            return { first, second };
        };

        if (parser.match('id', 'batching_dims') || parser.match('id', 'batch_dims')) {
            parser.expect('id');
            parser.accept('=');
            const pair = parsePair();
            result.lhs_batching_dimensions = pair.first;
            result.rhs_batching_dimensions = pair.second;
            parser.accept(',');
        }

        if (parser.accept('id', 'contracting_dims')) {
            parser.accept('=');
            const pair = parsePair();
            result.lhs_contracting_dimensions = pair.first;
            result.rhs_contracting_dimensions = pair.second;
        }

        return result;
    }

    _parsePrecisionConfig(parser /*, args */) {
        const result = {
            kind: 'PrecisionConfig',
            precision: []
        };

        if (!parser.match('id', 'precision')) {
            return result;
        }

        parser.expect('id', 'precision');
        parser.expect('=');
        parser.expect('[');
        while (!parser.match(']')) {
            if (parser.match('id')) {
                result.precision.push(parser.expect('id'));
                parser.accept(',');
            } else {
                parser.expect();
                parser.accept(',');
            }
        }
        parser.expect(']');

        return result;
    }

    _parsePrecisionConfigAndAlgorithm(parser /*, args */) {
        const result = {
            kind: 'PrecisionConfigAndAlgorithm',
            precision: [],
            algorithm: null
        };

        if (!parser.accept(',')) {
            return result;
        }

        if (parser.accept('id', 'algorithm')) {
            parser.accept('=');
            result.algorithm = parser.parseAttributeValue();
            return result;
        }

        if (parser.accept('id', 'precision')) {
            parser.accept('=');
            if (parser.accept('[')) {
                while (!parser.match(']')) {
                    if (parser.match('id')) {
                        result.precision.push(parser.expect('id'));
                        parser.accept(',');
                    } else {
                        parser.expect();
                        parser.accept(',');
                    }
                }
                parser.accept(']');
            }

            if (parser.accept(',')) {
                if (parser.accept('id', 'algorithm')) {
                    parser.accept('=');
                    result.algorithm = parser.parseAttributeValue();
                }
            }
        }

        return result;
    }

    _parseWindowAttributes(parser, /*, args */) {
        const result = {
            kind: 'WindowAttributes',
            stride: [],
            pad: [],
            lhs_dilate: [],
            rhs_dilate: [],
            window_reversal: []
        };

        while (!parser.match('}')) {
            if (parser.match('id')) {
                const key = parser.expect('id');
                if (parser.accept('=')) {
                    const parseArray = () => {
                        const arr = [];
                        if (parser.accept('[')) {
                            while (!parser.match(']')) {
                                if (parser.match('[')) {
                                    arr.push(parseArray());
                                } else if (parser.match('int') || parser.match('number')) {
                                    arr.push(parseInt(parser.expect(), 10));
                                } else if (parser.match('id')) {
                                    arr.push(parser.expect('id'));
                                } else {
                                    break;
                                }
                                parser.accept(',');
                            }
                            parser.accept(']');
                        }
                        return arr;
                    };

                    result[key] = parseArray();
                }
                parser.accept(',');
            } else {
                break;
            }
        }

        return result;
    }

    _parseSliceRanges(parser, /*, args */) {
        const result = {
            kind: 'SliceRanges',
            start_indices: [],
            limit_indices: [],
            strides: []
        };

        if (parser.accept('[')) {
            while (!parser.match(']')) {
                if (parser.match('int')) {
                    result.start_indices.push(parseInt(parser.expect('int'), 10));
                }
                parser.accept(':');
                if (parser.match('int')) {
                    result.limit_indices.push(parseInt(parser.expect('int'), 10));
                }
                if (parser.accept(':')) {
                    if (parser.match('int')) {
                        result.strides.push(parseInt(parser.expect('int'), 10));
                    }
                } else {
                    result.strides.push(1);
                }
                parser.accept(',');
            }
            parser.accept(']');
        }

        return result;
    }

    _parseCustomCallTarget(parser, /*, args */) {
        if (parser.match('@')) {
            const target = parser.expect('@');
            return { kind: 'CustomCallTarget', target };
        } else if (parser.match('string')) {
            const target = parser.expect('string');
            return { kind: 'CustomCallTarget', target };
        }
        throw new mlir.Error(`Expected '@' or string for CustomCallTarget at ${parser.location()}`);
    }

    _parseDynamicIndexList(parser, /*, args */) {
        const result = {
            kind: 'DynamicIndexList',
            indices: []
        };

        if (parser.accept('[')) {
            while (!parser.match(']')) {
                if (parser.match('int') || parser.match('number')) {
                    result.indices.push(parseInt(parser.expect(), 10));
                } else if (parser.match('%')) {
                    result.indices.push(parser.expect('%'));
                } else {
                    break;
                }
                parser.accept(',');
            }
            parser.accept(']');
        }

        return result;
    }

    _parseOffsets(parser, args) {
        const values = [];
        while (parser.match('int') || parser.match('-')) {
            if (parser.accept('-')) {
                if (parser.match('int')) {
                    values.push(-parseInt(parser.expect('int'), 10));
                } else {
                    throw new mlir.Error(`Expected integer after '-' in offsets ${parser.location()}`);
                }
            } else {
                values.push(parseInt(parser.expect('int'), 10));
            }
            if (!parser.accept(',')) {
                break;
            }
        }
        if (args && args.length > 0) {
            const [attrName] = args;
            return { name: attrName, value: values };
        }
        return values;
    }

    _parseVariadicOperandWithAttribute(parser, /*, args */) {
        const result = {
            kind: 'VariadicOperandWithAttribute',
            operands: []
        };
        while (parser.match('%')) {
            const operand = {
                value: parser.expect('%'),
                attributes: []
            };
            // Check for inline attributes
            if (parser.match('{')) {
                parser.parseAttributeDict(operand.attributes);
            }
            result.operands.push(operand);
            if (!parser.accept(',')) {
                break;
            }
        }
        return result;
    }

    _parseSymbolVisibility(parser /*, args */) {
        if (parser.match('id', 'private') || parser.match('id', 'public') || parser.match('id', 'nested')) {
            const visibility = parser.expect('id');
            return { kind: 'SymbolVisibility', visibility };
        }
        if (parser.match('string')) {
            const visibility = parser.expect('string');
            return { kind: 'SymbolVisibility', visibility };
        }
        return { kind: 'SymbolVisibility', visibility: null };
    }

    _parseSymbolAlias(parser, args) {
        if (!parser.match('@')) {
            return { kind: 'SymbolAlias' };
        }
        const symbolName = parser.expect('@');
        const result = { kind: 'SymbolAlias', symbolName };
        if (args && args.length >= 1) {
            result.symNameArg = args[0].replace(/^\$/, '');
        }
        if (args && args.length >= 2) {
            result.functionRefArg = args[1].replace(/^\$/, '');
        }
        return result;
    }

    _parseWorkgroupCountRegion(parser /*, args */) {
        if (parser.match('{')) {
            const region = {};
            parser.parseRegion(region);
            return { kind: 'WorkgroupCountRegion', region };
        }
        return { kind: 'WorkgroupCountRegion', region: null };
    }

    _parseOptionalUnitAttr(parser /*, args */) {
        if (parser.match('id') || parser.match('%') || parser.match('(')) {
            return { kind: 'OptionalUnitAttr', present: true };
        }
        return { kind: 'OptionalUnitAttr', present: false };
    }
};

mlir.StableHLODialect = class extends mlir.Dialect {

    constructor(operations) {
        super('stablehlo', operations);
    }

    parseType(parser, dialectType) {
        if (dialectType === '!stablehlo.token') {
            return dialectType;
        }
        return null;
    }

    parseOperation(parser, opName, op) {
        if (opName === 'stablehlo.constant') {
            if (parser.accept('(') && parser.accept(')')) {
                if (parser.match('{')) {
                    parser.parseAttributeDict(op.attributes);
                }
                if (parser.accept(':')) {
                    parser.parseArgumentTypes(op.operands);
                }
                if (parser.accept('->')) {
                    parser.parseArgumentTypes(op.results);
                }
            } else {
                if (parser.match('{')) {
                    parser.parseAttributeDict(op.attributes);
                }
                const value = parser.parseValue();
                if (value) {
                    op.attributes.push({ name: 'value', value: value.value });
                }
                // Parse result type: : tensor<...>
                if (parser.accept(':')) {
                    parser.parseArgumentTypes(op.results);
                }
            }
            return true;
        }
        if (opName === 'stablehlo.while' && parser.match('(')) {
            parser.accept('(');
            while (!parser.match(')')) {
                const arg = {};
                arg.value = parser.expect('%');
                if (parser.accept('=')) {
                    arg.name = arg.value;
                    arg.value = parser.expect('%');
                }
                op.operands.push(arg);
                parser.accept(',');
            }
            parser.expect(')');
            if (parser.accept(':')) {
                let index = 0;
                while (!parser.match('id', 'cond') && !parser.match('id', 'attributes') && index < op.operands.length * 2) {
                    const type = parser.parseType();
                    if (index < op.operands.length) {
                        op.operands[index].type = type;
                    }
                    if (index < op.results.length) {
                        op.results[index].type = type;
                    }
                    index++;
                    if (!parser.accept(',')) {
                        break;
                    }
                }
            }
            if (parser.accept('id', 'attributes')) {
                if (parser.match('{')) {
                    parser.parseAttributeDict(op.attributes);
                }
            }
            if (parser.accept('id', 'cond')) {
                const condRegion = {};
                parser.parseRegion(condRegion);
                op.regions.push(condRegion);
            }
            if (parser.accept('id', 'do')) {
                const bodyRegion = {};
                parser.parseRegion(bodyRegion);
                op.regions.push(bodyRegion);
            }
            return true;
        }
        if ((opName === 'stablehlo.reduce' || opName === 'stablehlo.scan') && parser.match('(')) {
            return this._parseReduceLikeOp(parser, op);
        }
        return super.parseOperation(parser, opName, op);
    }

    _parseReduceLikeOp(parser, op) {
        op.operands = parser.parseArguments();

        while (parser.accept(',')) {
            const moreOperands = parser.parseArguments();
            op.operands.push(...moreOperands);
        }

        if (parser.accept('id', 'applies')) {
            const innerOpName = parser.expect('id');
            op.attributes.push({ name: 'body_op', value: innerOpName });

            if (parser.accept('id', 'across')) {
                if (parser.accept('id', 'dimensions')) {
                    parser.accept('=');
                    const dims = parser.parseAttributeValue();
                    op.attributes.push({ name: 'dimensions', value: dims });
                }
            }

            if (parser.match('{')) {
                parser.parseAttributeDict(op.attributes);
            }

            if (parser.accept(':')) {
                parser.parseArgumentTypes(op.operands);
            }

            if (parser.accept('->')) {
                parser.parseArgumentTypes(op.results);
            }

            return true;
        }

        if (parser.match('(')) {
            // Generic form with parenthesized region-list: ({ ... })
            if (parser.accept('(') && parser.match('{')) {
                let regionCount = 0;
                while (!parser.match(')')) {
                    if (regionCount++ > 10) {
                        throw new mlir.Error(`Too many regions in region-list (>10) - possible infinite loop at ${parser.location()}, current token: '${parser.token.value}'`);
                    }
                    if (!parser.match('{')) {
                        throw new mlir.Error(`Expected '{' for region in region-list, got '${parser.token.value}' at ${parser.location()}`);
                    }
                    const region = {};
                    parser.parseRegion(region);
                    op.regions.push(region);
                    if (!parser.accept(',') && !parser.match(')')) {
                        throw new mlir.Error(`Expected ',' or ')' after region, got '${parser.token.value}' at ${parser.location()}`);
                    }
                }
                parser.expect(')');
            }

            // Parse attributes dictionary { dimensions = ... }
            if (parser.match('{')) {
                parser.parseAttributeDict(op.attributes);
            }

            // Parse type signature : (...) -> (...)
            if (parser.accept(':')) {
                parser.parseArgumentTypes(op.operands);
            }

            if (parser.accept('->') || parser.accept('id', 'to')) {
                if (op.results.length > 0) {
                    parser.parseArgumentTypes(op.results);
                } else {
                    op.results = parser.parseArguments();
                }
            }

            return true;
        }

        // Handle "across dimensions = [...]"
        if (parser.accept('id', 'across')) {
            if (parser.accept('id', 'dimensions')) {
                parser.expect('=');
                parser.skip('[', ']');
            }
        }

        // Type signature
        if (parser.accept(':')) {
            parser.parseArgumentTypes(op.operands);
        }

        if (parser.accept('->') || parser.accept('id', 'to')) {
            if (op.results.length > 0) {
                parser.parseArgumentTypes(op.results);
            } else {
                op.results = parser.parseArguments();
            }
        }

        // Handle regions
        if (parser.match('id') && !parser.match('keyword', 'loc')) {
            // Labeled region: reducer(...) { ... } or reducer(...) (...) { ... }
            const label = parser.expect('id');
            const region = { blocks: [] };
            const block = { operations: [], arguments: [], name: label };

            while (parser.accept('(')) {
                while (!parser.accept(')')) {
                    const value = parser.expect('%');
                    parser.expect(':');
                    const type = parser.parseType();
                    block.arguments.push({ value, type });
                    parser.accept(',');
                }
            }

            parser.expect('{');
            while (!parser.accept('}')) {
                const innerOp = parser.parseOperation();
                block.operations.push(innerOp);
            }

            block.loc = parser.parseLocation();
            region.blocks.push(block);
            op.regions.push(region);
        } else if (parser.accept('(') && parser.match('{')) {
            // Parenthesized region list: ({ ... })
            const region = {};
            parser.parseRegion(region);
            op.regions.push(region);
            parser.expect(')');
        } else if (parser.match('{')) {
            // Simple region: { ... }
            const region = {};
            parser.parseRegion(region);
            op.regions.push(region);
        }

        return true;
    }
};

mlir.VhloDialect = class extends mlir.Dialect {

    constructor(operations) {
        super('vhlo', operations);
    }

    parseOperation(parser, opName, op) {

        if (opName === 'vhlo.constant_v1') {
            if (parser.match('{')) {
                parser.parseAttributeDict(op.attributes);
            }
            const value = parser.parseValue();
            if (value) {
                op.attributes.push({ name: 'value', value: value.value });
            }
            // Parse result type: : tensor<...>
            if (parser.accept(':')) {
                parser.parseArgumentTypes(op.results);
            }
            return true;
        }

        if (opName === 'vhlo.return_v1') {
            op.operands = parser.parseArguments();
            if (parser.accept(':')) {
                parser.parseArgumentTypes(op.operands);
            }
            return true;
        }

        if (opName === 'vhlo.func_v1') {
            parser.parseOptionalVisibilityKeyword(op.attributes);
            parser.parseSymbolName(opName, op.attributes);
            if (parser.accept('(')) {
                op.attributes.push({ name: 'inputs', value: parser.parseTypeList() });
                parser.accept(')');
            }
            if (parser.accept('->')) {
                const outputType = parser.parseType();
                if (outputType) {
                    const outputs = outputType.value === 'tuple' ? outputType.params : [outputType];
                    op.attributes.push({ name: 'outputs', value: outputs });
                }
            }
            if (parser.match('{')) {
                parser.parseAttributeDict(op.attributes);
            }
            op.regions = [parser.parseRegion()];
            return true;
        }

        return false;
    }
};

mlir.InterpreterDialect = class extends mlir.Dialect {

    constructor(operations) {
        super('interpreter', operations);
    }
};

mlir.AffineDialect = class extends mlir.Dialect {

    constructor(operations) {
        super('affine', operations);
    }

    parseOperation(parser, opName, op) {
        // Special handling for affine.for - similar to scf.for but with affine expressions
        if (opName === 'affine.for') {
            return this._parseForOp(parser, op);
        }
        // Special handling for affine.if - has condition before region
        if (opName === 'affine.if') {
            // affine.if #set(...) { region }
            if (parser.match('#')) {
                const condition = parser.parseValue();
                op.attributes.push({ name: 'condition', value: condition });
            }
            const region = {};
            parser.parseRegion(region);
            op.regions.push(region);
            if (parser.accept('id', 'else')) {
                const elseRegion = {};
                parser.parseRegion(elseRegion);
                op.regions.push(elseRegion);
            }
            return true;
        }
        // Special handling for affine.apply, affine.min, and affine.max
        if (opName === 'affine.apply' || opName === 'affine.min' || opName === 'affine.max') {
            // Reference: parseAffineMinMaxOp in AffineOps.cpp
            // Syntax: affine.min #map(%dims)[%symbols]
            // 1. Parse affine map attribute
            if (parser.match('#') || parser.match('id', 'affine_map') || parser.match('id', 'affine_set')) {
                const value = parser.parseValue();
                op.attributes.push({ name: 'map', value });
            }
            // 2. Parse dimension operands in (...)
            if (parser.match('(')) {
                const dimOperands = parser.parseArguments();
                op.operands.push(...dimOperands);
            }
            // 3. Parse symbol operands in [...]
            if (parser.accept('[')) {
                while (!parser.accept(']')) {
                    if (parser.match('%')) {
                        const value = parser.expect('%');
                        op.operands.push({ value });
                    } else {
                        break;
                    }
                    parser.accept(',');
                }
            }
            // 4. Parse optional attr-dict
            if (parser.match('{')) {
                parser.parseAttributeDict(op.attributes);
            }
            return true;
        }
        if (opName === 'affine.store') {
            return this._parseStoreOp(parser, op);
        }
        if (opName === 'affine.load') {
            return this._parseLoadOp(parser, op);
        }
        return super.parseOperation(parser, opName, op);
    }

    _parseForOp(parser, op) {
        if (!parser.match('%')) {
            return false;
        }
        const inductionVar = parser.expect('%');
        if (!parser.accept('=')) {
            return false;
        }
        this._parseAffineBound(parser, op, 'lowerBound');
        if (!parser.accept('id', 'to')) {
            return false;
        }
        this._parseAffineBound(parser, op, 'upperBound');
        if (parser.accept('id', 'step')) {
            if (parser.match('int')) {
                const step = parser.expect('int');
                op.attributes.push({ name: 'step', value: step });
            }
        }
        if (parser.match('{')) {
            const region = {};
            parser.parseRegion(region);
            if (region.blocks && region.blocks.length > 0) {
                if (!region.blocks[0].arguments) {
                    region.blocks[0].arguments = [];
                }
                if (region.blocks[0].arguments.length > 0) {
                    region.blocks[0].arguments[0] = { value: inductionVar };
                } else {
                    region.blocks[0].arguments.push({ value: inductionVar });
                }
            }
            op.regions.push(region);
        }
        return true;
    }

    _parseAffineBound(parser, op, boundName) {
        // Parse affine bound: can be int, SSA value, or affine expression
        // Examples: 0, %N, min #map(%arg), max #map(), #map(%args)
        if (parser.match('int')) {
            const value = parser.expect('int');
            op.attributes.push({ name: boundName, value });
        } else if (parser.accept('id', 'min') || parser.accept('id', 'max')) {
            // min/max affine expression
            const expr = parser.parseValue();
            if (expr) {
                op.attributes.push({ name: boundName, value: expr });
            }
        } else if (parser.match('#')) {
            // Affine map reference
            const expr = parser.parseValue();
            if (expr) {
                op.attributes.push({ name: boundName, value: expr });
            }
        } else if (parser.match('%')) {
            op.operands.push({ value: parser.expect('%') });
        }
    }

    _parseStoreOp(parser, op) {
        if (parser.match('%')) {
            const value = parser.expect('%');
            op.operands.push({ value });
        } else {
            const value = parser.parseValue();
            op.operands.push(value);
        }
        if (!parser.accept('id', 'to')) {
            parser.accept(',');
        }
        const address = parser.expect('%');
        op.operands.push({ value: address });
        parser.skip('[', ']');
        if (parser.accept(':')) {
            const type = parser.parseType();
            op.operands[1].type = type;
        }
        return true;
    }

    _parseLoadOp(parser, op) {
        const address = parser.expect('%');
        op.operands.push({ value: address });
        parser.skip('[', ']');
        if (parser.accept(':')) {
            const type = parser.parseType();
            op.operands[0].type = type;
        }
        return true;
    }
};

mlir.MemRefDialect = class extends mlir.Dialect {

    constructor(operations) {
        super('memref', operations);
    }

    parseOperation(parser, opName, op) {
        if (opName === 'memref.tensor_load') {
            op.operands = parser.parseArguments();
            if (parser.accept(':')) {
                parser.parseArgumentTypes(op.operands);
            }
            return true;
        }
        if (opName === 'memref.store') {
            // Backwards compatibility
            this._operations.get(opName).hasParseOperation = false;
            return this._parseStoreOp(parser, op);
        }
        return super.parseOperation(parser, opName, op);
    }

    _parseStoreOp(parser, op) {
        if (parser.match('%')) {
            const value = parser.expect('%');
            op.operands.push({ value });
        } else {
            const value = parser.parseValue();
            op.operands.push(value);
        }
        if (!parser.accept('id', 'to')) {
            parser.accept(',');
        }
        const address = parser.expect('%');
        op.operands.push({ value: address });
        parser.skip('[', ']');
        if (parser.accept(':')) {
            const type = parser.parseType();
            op.operands[1].type = type;
        }
        return true;
    }
};

mlir.VectorDialect = class extends mlir.Dialect {

    constructor(operations) {
        super('vector', operations);
    }

    parseOperation(parser, opName, op) {
        if (opName === 'vector.splat') {
            op.operands = parser.parseArguments();
            if (parser.accept(':')) {
                parser.parseArgumentTypes(op.results);
            }
            return true;
        }
        if (opName === 'vector.contract') {
            if (parser.match('{')) {
                parser.skip('{', '}');
            } else if (parser.match('#')) {
                parser.expect('#');
            }
            op.operands = parser.parseArguments();
            parser.parseOptionalAttrDict(op.attributes);
            if (parser.accept(':')) {
                parser.parseArgumentTypes(op.operands);
                if (parser.accept('id', 'into')) {
                    parser.parseArgumentTypes(op.results);
                }
            }
            return true;
        }
        if (opName === 'vector.mask') {
            if (parser.match('%')) {
                const mask = parser.expect('%');
                op.operands.push({ value: mask, name: 'mask' });
            }
            if (parser.accept(',')) {
                const passthru = parser.expect('%');
                op.operands.push({ value: passthru, name: 'passthru' });
            }
            if (parser.match('{')) {
                const region = {};
                parser.parseRegion(region);
                op.regions.push(region);
            }
            parser.parseOptionalAttrDict(op.attributes);
            if (parser.accept(':')) {
                parser.parseArgumentTypes(op.operands);
                if (parser.accept('->')) {
                    parser.parseArgumentTypes(op.results);
                }
            }
            return true;
        }
        if (opName === 'vector.transfer_read' || opName === 'vector.transfer_write') {
            return this._parseTransferOp(parser, op);
        }
        if (opName === 'vector.extract' && !op.isGeneric) {
            // Backwards compatibility
            this._operations.get(opName).hasParseOperation = false;
            return this._parseExtractOp(parser, op);
        }
        return super.parseOperation(parser, opName, op);
    }

    _parseExtractOp(parser, op) {
        // Parse: %source [ indices ] : result_type [from source_type]
        // The 'from' keyword indicates new syntax (current MLIR)
        // Old syntax (pre-2023): %r = vector.extract %v[0] : vector<4xf32>
        // New syntax: %r = vector.extract %v[0] : f32 from vector<4xf32>

        // Parse source operand
        const source = parser.expect('%');
        op.operands.push({ value: source });

        // Parse indices: [0, 1, ...]
        if (parser.accept('[')) {
            while (!parser.match(']')) {
                if (parser.match('int') || parser.match('number')) {
                    parser.expect(); // Consume index but don't store (indices are in static_position attribute)
                } else if (parser.match('%')) {
                    const dynIndex = parser.expect('%');
                    op.operands.push({ value: dynIndex }); // Dynamic indices are operands
                } else {
                    break;
                }
                parser.accept(',');
            }
            parser.accept(']');
        }

        // Parse optional attributes
        if (parser.match('{')) {
            parser.parseAttributeDict(op.attributes);
        }

        // Parse type signature: : result_type [from source_type]
        if (parser.accept(':')) {
            const resultType = parser.parseType();

            // Check for 'from' keyword (new syntax)
            if (parser.accept('id', 'from')) {
                const sourceType = parser.parseType();
                op.operands[0].type = sourceType;
                op.results.push({ type: resultType });
            } else {
                // Old syntax: the type after ':' is the source type
                // Result type is extracted element type (scalar or sub-vector)
                op.operands[0].type = resultType;
                // We don't set result type - let the default inference handle it
            }
        }

        return true;
    }

    _parseTransferOp(parser, op) {
        // Parse: vector.transfer_read %source[%i, %j, ...], %padding {attrs} : memref_type, vector_type
        //    or: vector.transfer_read %source[%i, %j, ...], %padding, %mask {attrs} : memref_type, vector_type
        //    or: vector.transfer_write %value, %dest[%i, %j, ...] {attrs} : vector_type, memref_type
        //    or: vector.transfer_write %value, %dest[%i, %j, ...], %mask {attrs} : vector_type, memref_type

        // First operand: source/value
        const first = parser.expect('%');
        op.operands.push({ value: first });

        // Check if indices follow first operand or second operand
        const hasIndicesAfterFirst = parser.match('[');
        if (hasIndicesAfterFirst) {
            parser.skip('[', ']');
        }

        // Comma
        parser.accept(',');

        // Second operand: padding value or destination
        const second = parser.expect('%');
        op.operands.push({ value: second });

        // If indices didn't follow first operand, they follow second operand
        if (!hasIndicesAfterFirst && parser.match('[')) {
            parser.skip('[', ']');
        }

        // Optional mask parameter (third operand)
        if (parser.accept(',')) {
            const mask = parser.expect('%');
            op.operands.push({ value: mask });
        }

        // Optional attribute dictionary
        if (parser.match('{')) {
            parser.parseAttributeDict(op.attributes);
        }

        // Type signature: : memref_type, vector_type
        if (parser.accept(':')) {
            const type1 = parser.parseType();
            op.operands[0].type = type1;
            parser.accept(',');
            const type2 = parser.parseType();
            // For transfer_read, type2 is the result type
            // For transfer_write, type2 is just the vector type
            if (op.results.length > 0) {
                op.results[0].type = type2;
            }
            op.operands[1].type = type2;
        }

        return true;
    }
};

mlir.TorchDialect = class extends mlir.Dialect {

    constructor(operations) {
        super('torch', operations);
    }

    parseType(parser, dialectType) {
        const simpleTypes = [
            'int', 'float', 'bool', 'str', 'none', 'Device', 'Generator',
            'qint8', 'quint8', 'qint16', 'qint32', 'quint4x2', 'quint2x4',
            'LinearParams', 'number', 'any'
        ];
        for (const simpleType of simpleTypes) {
            if (dialectType === `!torch.${simpleType}`) {
                return dialectType;
            }
        }
        if (dialectType.startsWith('!torch.nn.Module')) {
            if (parser.match('<')) {
                const content = parser.skip('<', '>');
                return dialectType + content;
            }
            return dialectType;
        }
        if (dialectType.startsWith('!torch.vtensor') || dialectType.startsWith('!torch.tensor') ||
            dialectType.startsWith('!torch.list') || dialectType.startsWith('!torch.tuple') ||
            dialectType.startsWith('!torch.union') || dialectType.startsWith('!torch.optional') ||
            dialectType.startsWith('!torch.dict')) {
            if (parser.match('<')) {
                const content = parser.skip('<', '>');
                return dialectType + content;
            }
            return dialectType;
        }
        return null;
    }

    parseOperation(parser, opName, op) {
        if (opName === 'torch.constant.int') {
            if (parser.match('int')) {
                const value = parser.expect('int');
                op.attributes.push({ name: 'value', value });
            }
            return true;
        }
        if (opName.startsWith('torch.aten.') || (opName.startsWith('torch.prim.'))) {
            const operator = this._operations.get(opName);
            if (!operator.metadata.assemblyFormat) {
                parser.parseGenericOperationAfterOpName(op);
                return true;
            }
        }
        return super.parseOperation(parser, opName, op);
    }
};

mlir.HALDialect = class extends mlir.Dialect {

    constructor(operations) {
        super('hal', operations);
    }

    parseType(parser, dialectType) {
        const simpleTypes = [
            'allocator', 'buffer', 'buffer_view', 'channel', 'command_buffer',
            'device', 'event', 'executable', 'fence', 'file', 'semaphore'
        ];
        for (const simpleType of simpleTypes) {
            if (dialectType === `!hal.${simpleType}`) {
                return dialectType;
            }
        }
        return null;
    }

    parseOperation(parser, opName, op) {
        if (opName === 'hal.tensor.cast') {
            op.operands = parser.parseArguments();
            if (parser.accept(':')) {
                const type = parser.parseType();
                if (op.operands.length > 0) {
                    op.operands[0].type = type;
                }
            }
            if (parser.accept('->')) {
                parser.parseArgumentTypes(op.results);
            }
            return true;
        }
        if (opName === 'hal.constant') {
            parser.parseOptionalAttrDictWithKeyword(op.attributes);
            const value = parser.parseValue();
            op.attributes.push({ name: 'value', value: value.value === undefined ? value : value.value });
            if (parser.accept(':')) {
                parser.parseArgumentTypes(op.results);
            }
            return true;
        }
        if (opName === 'hal.device.switch') {
            if (parser.accept('<')) {
                while (!parser.accept('>')) {
                    const operand = parser.expect('%');
                    op.operands.push({ value: operand });
                    if (parser.accept(':')) {
                        const type = parser.parseType();
                        if (op.operands.length > 0) {
                            op.operands[op.operands.length - 1].type = type;
                        }
                    }
                    parser.accept(',');
                }
            }
            if (parser.accept('->') || parser.accept(':')) {
                const resultType = parser.parseType();
                op.results = [{ type: resultType }];
            }
            while (parser.match('#')) {
                const region = {};
                const caseAttr = parser.parseAttributeValue();
                region.caseAttribute = caseAttr;
                if (parser.match('{')) {
                    parser.parseRegion(region);
                }
                op.regions.push(region);
                parser.accept(',');
            }
            return true;
        }
        // Handle hal.executable.create with both old (layouts) and new (affinity) syntax
        if (opName === 'hal.executable.create') {
            // Parse named parameters: device(...), target(...), and either layouts(...) or affinity(...)
            while (parser.match('id') && !parser.match(':') && !parser.match('loc')) {
                const paramName = parser.expect('id');
                if (parser.accept('(')) {
                    let parenDepth = 1;
                    let paramValue = '';
                    while (parenDepth > 0 && !parser.match('eof')) {
                        if (parser.match('(')) {
                            parenDepth++;
                            paramValue += parser.expect();
                        } else if (parser.match(')')) {
                            parenDepth--;
                            if (parenDepth > 0) {
                                paramValue += parser.expect();
                            } else {
                                parser.expect(')');
                            }
                        } else {
                            paramValue += parser.expect();
                        }
                    }
                    // Normalize old 'layouts' parameter to 'affinity' for consistency
                    const normalizedName = paramName === 'layouts' ? 'affinity' : paramName;
                    op.attributes.push({ name: normalizedName, value: paramValue });
                } else {
                    break;
                }
            }
            // Parse result type
            if (parser.accept(':')) {
                parser.parseArgumentTypes(op.results);
            }
            return true;
        }
        // Handle operations with <%operand : type> syntax and/or named parameters
        // e.g., hal.allocator.compute_size<%allocator : !hal.allocator> shape([...]) type(...) encoding(...) : index
        // or hal.executable_layout.lookup device(%device : !hal.device) layouts([[...]]) : !hal.executable_layout
        // Exclude hal.executable, hal.interface, and hal.device.switch which have special handling
        if ((opName.startsWith('hal.allocator.') || opName.startsWith('hal.buffer') ||
             opName.startsWith('hal.command_buffer') || opName.startsWith('hal.executable_layout') ||
             opName.startsWith('hal.executable.') || opName.startsWith('hal.descriptor_set_layout') ||
             opName.startsWith('hal.device')) &&
            opName !== 'hal.executable' && opName !== 'hal.interface' && opName !== 'hal.device.switch' &&
            opName !== 'hal.executable.variant' && opName !== 'hal.executable.entry_point' && opName !== 'hal.interface.binding' &&
            opName !== 'hal.executable.create' && opName !== 'hal.executable.export') {
            // Parse <%operand : type> if present
            if (parser.accept('<')) {
                while (!parser.accept('>')) {
                    const operand = parser.expect('%');
                    op.operands.push({ value: operand });
                    if (parser.accept(':')) {
                        const type = parser.parseType();
                        if (op.operands.length > 0) {
                            op.operands[op.operands.length - 1].type = type;
                        }
                    }
                    parser.accept(',');
                }
            }
            // Parse named parameters like shape([...]) type(...) encoding(...)
            // Also handle bracket expressions between parameters like layout(...)[%c0]
            // Stop when we hit a colon (result type) or something that doesn't look like a parameter
            // Named parameters don't have dots, so if we see an id with a dot, it's likely the next operation
            while (parser.match('[') ||
                   (parser.match('id') && !parser.match('id', 'attributes') &&
                    !parser.match(':') && !parser.match('loc') &&
                    parser.token.value && parser.token.value.indexOf('.') === -1)) {
                // Handle bracket expressions (e.g., [%c0])
                if (parser.match('[')) {
                    parser.skip('[', ']');
                    continue;
                }
                // Try to parse a named parameter (id followed by '(')
                const paramName = parser.expect('id');
                if (parser.accept('(')) {
                    // This is a named parameter, parse the value
                    // Track depth separately for () and []
                    let parenDepth = 1;  // We've already consumed the opening (
                    let paramValue = '';
                    while (parenDepth > 0 && !parser.match('eof')) {
                        if (parser.match('(')) {
                            parenDepth++;
                            paramValue += parser.expect();
                        } else if (parser.match(')')) {
                            parenDepth--;
                            if (parenDepth > 0) {
                                paramValue += parser.expect();
                            } else {
                                // This is the closing ), consume it
                                parser.expect(')');
                            }
                        } else {
                            paramValue += parser.expect();
                        }
                    }
                    op.attributes.push({ name: paramName, value: paramValue });
                } else {
                    // Not a named parameter - we've consumed an id token that doesn't belong to us
                    // This shouldn't happen with proper MLIR, but break gracefully
                    break;
                }
            }
            // Parse result types if present (: type1, type2, ...)
            if (parser.accept(':')) {
                parser.parseArgumentTypes(op.results);
            }
            // Parse optional = <value> (default value or attribute)
            if (parser.accept('=')) {
                const value = parser.parseValue();
                op.attributes.push({ name: 'default', value: value.value });
            }
            return true;
        }
        // Handle operations with visibility + symbol (similar to flow dialect)
        if (opName === 'hal.executable' || opName === 'hal.interface') {
            if (parser.match('id', 'private') || parser.match('id', 'public') || parser.match('id', 'nested')) {
                parser.expect('id');
            }
            if (parser.match('@')) {
                const symbol = parser.expect('@');
                op.attributes.push({ name: 'sym_name', value: symbol });
            }
            if (parser.accept('id', 'attributes')) {
                parser.parseAttributeDict(op.attributes);
            }
            if (parser.match('{')) {
                const region = {};
                parser.parseRegion(region);
                op.regions.push(region);
            }
            return true;
        }
        // Handle hal.interface.binding.subspan with old syntax (symbol reference)
        // Old syntax: hal.interface.binding.subspan @io::@binding[operand] : type
        // New syntax: hal.interface.binding.subspan layout(...) binding(...) : type
        if (opName === 'hal.interface.binding.subspan' && parser.match('@')) {
            // Old syntax - parse symbol reference and bracket expression
            const symbolRef = parser.expect('@');
            op.attributes.push({ name: 'layout', value: symbolRef });
            // Parse optional bracket expression [operand]
            if (parser.accept('[')) {
                while (!parser.accept(']')) {
                    if (parser.match('%')) {
                        const operand = parser.expect('%');
                        op.operands.push({ value: operand });
                    } else {
                        parser.expect();
                    }
                    parser.accept(',');
                }
            }
            // Parse result type
            if (parser.accept(':')) {
                parser.parseArgumentTypes(op.results);
            }
            return true;
        }
        // Handle operations with named parameters: hal.interface.binding, hal.executable.variant, etc.
        if (opName === 'hal.interface.binding' || opName === 'hal.executable.variant' || opName === 'hal.executable.entry_point' || opName === 'hal.executable.export') {
            if (parser.match('id', 'private') || parser.match('id', 'public') || parser.match('id', 'nested')) {
                parser.expect('id');
            }
            if (parser.match('@')) {
                const symbol = parser.expect('@');
                op.attributes.push({ name: 'sym_name', value: symbol });
                parser.accept(',');
            }
            while (parser.match('id') && !parser.match('id', 'attributes') && !parser.match('{') && !parser.match('loc')) {
                const tokenValue = parser.token.value;
                if (tokenValue && tokenValue.includes('.')) {
                    break;
                }
                const paramName = parser.expect('id');
                if (parser.accept('(')) {
                    let parenDepth = 1;
                    let paramValue = '';
                    while (parenDepth > 0 && !parser.match('eof')) {
                        if (parser.match('(')) {
                            parenDepth++;
                            paramValue += parser.expect();
                        } else if (parser.match(')')) {
                            parenDepth--;
                            if (parenDepth > 0) {
                                paramValue += parser.expect();
                            } else {
                                parser.expect(')');
                            }
                        } else {
                            paramValue += parser.expect();
                        }
                    }
                    op.attributes.push({ name: paramName, value: paramValue });
                    parser.accept(',');
                } else if (parser.accept('=')) {
                    if (parser.match('#')) {
                        const value = parser.parseValue();
                        op.attributes.push({ name: paramName, value: value.value });
                    } else if (parser.match('string')) {
                        const value = parser.expect('string');
                        op.attributes.push({ name: paramName, value });
                    } else {
                        const value = parser.expect();
                        op.attributes.push({ name: paramName, value });
                    }
                    if (!parser.accept(',')) {
                        break;
                    }
                } else {
                    break;
                }
            }
            if (parser.accept('->')) {
                parser.parseFunctionResultList();
            }
            // Parse attributes dict if present
            if (parser.accept('id', 'attributes')) {
                parser.parseAttributeDict(op.attributes);
            }
            // Parse region if present
            if (parser.match('{')) {
                const region = {};
                parser.parseRegion(region);
                op.regions.push(region);
            }
            return true;
        }
        return super.parseOperation(parser, opName, op);
    }
};

mlir.UtilDialect = class extends mlir.Dialect {

    constructor(operations) {
        super('util', operations);
        this.registerCustomParser('ShapedTiedResult', this._parseShapedTiedResult.bind(this));
        this.registerCustomParser('OperandTypeList', this._parseOperandTypeList.bind(this));
        this.registerCustomParser('TiedFunctionResultList', this._parseTiedFunctionResultList.bind(this));
    }

    parseType(parser, dialectType) {
        const simpleTypes = ['buffer', 'list', 'object'];
        for (const simpleType of simpleTypes) {
            if (dialectType === `!util.${simpleType}`) {
                if (parser.match('<')) {
                    const content = parser.skip('<', '>');
                    return dialectType + content;
                }
                return dialectType;
            }
            if (dialectType.startsWith(`!util.${simpleType}<`)) {
                return dialectType;
            }
        }
        if (dialectType.startsWith('!util.ptr')) {
            if (parser.match('<')) {
                const content = parser.skip('<', '>');
                return dialectType + content;
            }
            return dialectType;
        }
        return null;
    }

    _parseShapedTiedResult(parser) {
        // Parse: %arg0 as tensor<?x?xf32>{%d0, %d1}
        //    or: tensor<?x?xf32>{%d0, %d1}
        let tiedOperand = null;
        if (parser.match('%')) {
            tiedOperand = parser.expect('%');
            if (!parser.accept('id', 'as')) {
                throw new mlir.Error(`Expected 'as' after tied operand ${parser.location()}`);
            }
        }
        const resultType = parser.parseType();
        const dims = [];
        if (parser.accept('{')) {
            while (!parser.match('}')) {
                if (parser.match('%')) {
                    const dim = parser.expect('%');
                    dims.push(dim);
                    parser.accept(',');
                } else {
                    break;
                }
            }
            parser.expect('}');
        }
        return { kind: 'ShapedTiedResult', tiedOperand, resultType, dims };
    }

    _parseOperandTypeList(parser) {
        // Parse: (type, type, ...)  or ()
        const types = [];
        if (!parser.accept('(')) {
            throw new mlir.Error(`Expected '(' in OperandTypeList ${parser.location()}`);
        }
        if (!parser.match(')')) {
            do {
                const type = parser.parseType();
                types.push(type);
            } while (parser.accept(','));
        }
        if (!parser.accept(')')) {
            throw new mlir.Error(`Expected ')' in OperandTypeList ${parser.location()}`);
        }
        return { kind: 'OperandTypeList', types };
    }

    _parseTiedFunctionResultList(parser) {
        // Parse: (type, type, ...)  or type
        // This is similar to OperandTypeList but for results
        const types = [];
        if (parser.accept('(')) {
            if (!parser.match(')')) {
                do {
                    const type = parser.parseType();
                    types.push(type);
                } while (parser.accept(','));
            }
            if (!parser.accept(')')) {
                throw new mlir.Error(`Expected ')' in TiedFunctionResultList ${parser.location()}`);
            }
        } else {
            const type = parser.parseType();
            types.push(type);
        }
        return { kind: 'TiedFunctionResultList', types };
    }

    parseOperation(parser, opName, op) {
        // Handle util.global with visibility and symbol
        if (opName === 'util.global') {
            if (parser.match('id', 'private') || parser.match('id', 'public') || parser.match('id', 'nested')) {
                parser.expect('id');
            }
            if (parser.match('id', 'mutable')) {
                const mutable = parser.expect('id');
                op.attributes.push({ name: 'is_mutable', value: mutable });
            }
            if (parser.match('@')) {
                const symbol = parser.expect('@');
                op.attributes.push({ name: 'sym_name', value: symbol });
            }
            if (parser.match('{')) {
                parser.parseAttributeDict(op.attributes);
            }
            if (parser.accept('=')) {
                const initialValue = parser.parseAttributeValue();
                op.attributes.push({ name: 'initial_value', value: initialValue });
            }
            if (parser.accept(':')) {
                const type = parser.parseType();
                op.attributes.push({ name: 'type', value: type });
            }
            return true;
        }
        // Handle util.initializer with region
        if (opName === 'util.initializer') {
            if (parser.match('{')) {
                const region = {};
                parser.parseRegion(region);
                op.regions.push(region);
            }
            return true;
        }
        // Handle util.unreachable with optional message
        // assemblyFormat: ($message^)? attr-dict
        if (opName === 'util.unreachable') {
            // Parse optional message string
            if (parser.match('string')) {
                const message = parser.expect('string');
                op.attributes.push({ name: 'message', value: message });
            }
            // Parse attr-dict if present
            if (parser.match('{')) {
                parser.parseAttributeDict(op.attributes);
            }
            return true;
        }
        if (opName === 'util.func') {
            return this._parseFuncOp(parser, op);
        }
        return super.parseOperation(parser, opName, op);
    }

    _parseFuncOp(parser, op) {
        parser.parseOptionalVisibilityKeyword(op.attributes);
        parser.parseSymbolName('sym_name', op.attributes);
        const type = {};
        type.inputs = parser.parseFunctionArgumentList();
        parser.parseOptionalAttrDictWithKeyword(op.attributes);
        type.results = [];
        if (parser.accept('->')) {
            for (const result of parser.parseFunctionResultList()) {
                type.results.push(result);
            }
        }
        op.attributes.push({ name: 'function_type', value: type });
        parser.parseOptionalAttrDictWithKeyword(op.attributes);
        if (parser.match('{')) {
            const region = {};
            parser.parseRegion(region);
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
        return true;
    }
};

mlir.FlowDialect = class extends mlir.Dialect {

    constructor(operations) {
        super('flow', operations);
        this.registerCustomParser('DispatchEntryPoints', this._parseDispatchEntryPoints.bind(this));
        this.registerCustomParser('ShapedFunctionType', this._parseShapedFunctionType.bind(this));
        this.registerCustomParser('ShapedTiedResult', this._parseShapedTiedResult.bind(this));
    }

    parseType(parser, dialectType) {
        if (dialectType === '!flow.channel') {
            return dialectType;
        }
        if (dialectType.startsWith('!flow.dispatch.tensor')) {
            if (parser.match('<')) {
                const content = parser.skip('<', '>');
                return dialectType + content;
            }
            return dialectType;
        }
        return null;
    }

    parseOperation(parser, opName, op) {
        // Handle operations with custom syntax not in schema or using complex custom parsers
        if ((opName === 'flow.dispatch.workgroups' && parser.match('[')) || opName === 'flow.ex.stream.fragment') {
            return this._parseDispatchWorkgroupsOp(parser, op);
        }
        if (opName === 'flow.dispatch.tensor.load' || opName === 'flow.dispatch.tensor.store') {
            return this._parseTensorLoadStoreOp(parser, op);
        }
        // Handle operations with visibility + symbol that aren't in schema or need manual parsing
        if (opName === 'flow.executable' || opName === 'flow.dispatch.entry') {
            if (parser.match('id', 'private') || parser.match('id', 'public') || parser.match('id', 'nested')) {
                parser.expect('id');
            }
            if (parser.match('@')) {
                const symbol = parser.expect('@');
                op.attributes.push({ name: 'sym_name', value: symbol });
            }
            if (parser.accept('id', 'attributes')) {
                parser.parseAttributeDict(op.attributes);
            }
            if (parser.match('{')) {
                const region = {};
                parser.parseRegion(region);
                op.regions.push(region);
            }
            return true;
        }
        if (opName === 'flow.func') {
            return this._parseFuncOp(parser, op);
        }
        return super.parseOperation(parser, opName, op);
    }

    _parseFuncOp(parser, op) {
        parser.parseOptionalVisibilityKeyword(op.attributes);
        parser.parseSymbolName('sym_name', op.attributes);
        const type = {};
        type.inputs = parser.parseFunctionArgumentList();
        parser.parseOptionalAttrDictWithKeyword(op.attributes);
        type.results = [];
        if (parser.accept('->')) {
            for (const result of parser.parseFunctionResultList()) {
                type.results.push(result);
            }
        }
        op.attributes.push({ name: 'function_type', value: type });
        parser.parseOptionalAttrDictWithKeyword(op.attributes);
        if (parser.match('{')) {
            const region = {};
            parser.parseRegion(region);
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
        return true;
    }

    _parseDispatchWorkgroupsOp(parser, op) {
        // Parse subscript values: [%c32, %c112, %c112]
        if (parser.accept('[')) {
            while (!parser.accept(']')) {
                parser.expect(); // read subscript value
                parser.accept(',');
            }
        }
        // Parse operands: (%0, %1)
        op.operands = parser.parseArguments();
        // Parse type signature : (...) -> (...)
        if (parser.accept(':')) {
            parser.parseArgumentTypes(op.operands);
        }
        if (parser.accept('->')) {
            if (op.results.length > 0) {
                parser.parseArgumentTypes(op.results);
            } else {
                op.results = parser.parseArguments();
            }
        }
        // Parse region with arguments: = (%arg2: type, %arg3: type) { ... }
        if (parser.accept('=')) {
            const region = {};
            region.blocks = [];
            const block = {};
            block.operations = [];
            block.arguments = [];
            // Parse region arguments
            if (parser.accept('(')) {
                while (!parser.accept(')')) {
                    const value = parser.expect('%');
                    parser.expect(':');
                    const type = parser.parseType();
                    block.arguments.push({ value, type });
                    parser.accept(',');
                }
            }
            // Some operations like flow.ex.stream.fragment have -> type after region args
            if (parser.accept('->') || parser.accept('id', 'to')) {
                parser.parseType();
            }
            // Parse region body
            parser.parseBlock(block);
            region.blocks.push(block);
            op.regions.push(region);
        }
        return true;
    }

    _parseDispatchEntryPoints(parser, args) {
        // Parse either:
        // - Single: @symbol or @symbol::@nested
        // - Multiple: {@symbol1, @symbol2::@nested2}
        const entryPoints = [];

        if (parser.accept('{')) {
            // Parse multiple entry points
            do {
                if (parser.match('@')) {
                    let symbol = parser.expect('@');
                    // Handle :: nested symbol reference
                    if (parser.accept('id', '::') || (parser.match(':') && parser.accept(':') && parser.accept(':'))) {
                        if (parser.match('@')) {
                            const nested = parser.expect('@');
                            symbol += `::${nested}`;
                        }
                    }
                    entryPoints.push(symbol);
                }
            } while (parser.accept(','));
            parser.expect('}');
        } else if (parser.match('@')) {
            // Parse single entry point
            let symbol = parser.expect('@');
            // Handle :: nested symbol reference
            if (parser.accept('id', '::') || (parser.match(':') && parser.accept(':') && parser.accept(':'))) {
                if (parser.match('@')) {
                    const nested = parser.expect('@');
                    symbol += `::${nested}`;
                }
            }
            entryPoints.push(symbol);
        }

        // Return the result with the attribute name from args
        if (args && args.length > 0) {
            const attrName = args[0].replace(/^\$/, '');
            return { kind: 'DispatchEntryPoints', attrName, entryPoints };
        }
        return { kind: 'DispatchEntryPoints', entryPoints };
    }

    _parseShapedFunctionType(parser /*, args */) {
        // Parse: (operand_types) -> result_types
        // This handles shaped function types with dynamic dimensions
        // Example: (tensor<?x?xf32>, tensor<4xf32>) -> tensor<?xf32>
        //
        // Args from assemblyFormat:
        // - ref($arguments): reference to operands
        // - type($arguments): extract types for arguments
        // - $argument_dims: dynamic dimension operands
        // - type($results): extract types for results
        // - $result_dims: dynamic dimension operands for results
        // - $tied_operands: tied operand indices

        // Parse the function type: (input_types) -> output_types
        const operandTypes = [];
        const resultTypes = [];

        // Parse input types
        if (parser.accept('(')) {
            if (!parser.match(')')) {
                do {
                    const type = parser.parseType();
                    if (type) {
                        operandTypes.push(type);
                    }
                } while (parser.accept(','));
            }
            parser.expect(')');
        }

        // Parse arrow
        if (!parser.accept('->')) {
            return { kind: 'ShapedFunctionType', operandTypes, resultTypes };
        }

        // Parse result types
        const hasParens = parser.accept('(');
        if (!parser.match(')') && !parser.match('{') && !parser.match('loc')) {
            do {
                const type = parser.parseType();
                if (type) {
                    resultTypes.push(type);
                }
                if (!hasParens) {
                    break; // Single result without parens
                }
            } while (parser.accept(','));
        }
        if (hasParens) {
            parser.expect(')');
        }

        return { kind: 'ShapedFunctionType', operandTypes, resultTypes };
    }

    _parseShapedTiedResult(parser) {
        // Parse: %arg0 as tensor<?x?xf32>{%d0, %d1}
        //    or: tensor<?x?xf32>{%d0, %d1}
        // Based on IREE's parseShapedTiedResult in UtilOps.cpp

        let tiedOperand = null;

        // Optionally parse a tied operand
        if (parser.match('%')) {
            tiedOperand = parser.expect('%');
            // Expect "as" keyword
            if (!parser.accept('id', 'as')) {
                throw new mlir.Error(`Expected 'as' after tied operand ${parser.location()}`);
            }
        }

        // Parse the result type (first arg should be type($result))
        const resultType = parser.parseType();

        // Parse dynamic dimensions in braces if present (second arg should be $target_dims)
        const dims = [];
        if (parser.accept('{')) {
            while (!parser.match('}')) {
                if (parser.match('%')) {
                    const dim = parser.expect('%');
                    dims.push(dim);
                    parser.accept(',');
                } else {
                    break;
                }
            }
            parser.expect('}');
        }

        return { kind: 'ShapedTiedResult', tiedOperand, resultType, dims };
    }

    _parseTensorLoadStoreOp(parser, op) {
        // Parse: load %arg2, offsets = [...] : type -> type
        //    or: store %26, %arg4, offsets = [...] : type -> type
        while (parser.match('%')) {
            const value = parser.expect('%');
            op.operands.push({ value });
            if (!parser.accept(',')) {
                break;
            }
            if (!parser.match('%')) {
                break;
            }
        }
        // At this point, if we broke because of named params, we've already consumed the comma
        // Parse comma-separated named parameters: offsets = [...], sizes = [...], strides = [...]
        // Note: first parameter might not need comma-eating if we just broke from operand loop
        let needComma = !parser.match('id'); // If we're not at 'id', we need to eat commas
        while (needComma ? parser.accept(',') : true) {
            needComma = true; // After first iteration, always need comma
            if (parser.match('id')) {
                const paramName = parser.expect('id');
                if (parser.accept('=')) {
                    // Skip the parameter value (usually an array)
                    if (parser.match('[')) {
                        parser.skip('[', ']');
                    } else {
                        parser.expect(); // Read single value
                    }
                    op.attributes.push({ name: paramName, value: paramName });
                }
            } else {
                break;
            }
        }
        // Parse type signature : type -> type
        if (parser.accept(':')) {
            parser.parseArgumentTypes(op.operands);
        }
        // For tensor.load, there's a -> result type
        // For tensor.store, the -> is followed by the output tensor type (not a result)
        if (parser.accept('->') || parser.accept('id', 'to')) {
            // Just skip the type - we don't need to parse it as results
            parser.parseType();
        }
        return true;
    }
};

mlir.StreamDialect = class extends mlir.Dialect {

    constructor(operations) {
        super('stream', operations);
        this.registerCustomParser('ShapedTiedResult', this._parseShapedTiedResult.bind(this));
    }

    parseType(parser, dialectType) {
        const simpleTypes = ['binding', 'timepoint', 'file'];
        for (const simpleType of simpleTypes) {
            if (dialectType === `!stream.${simpleType}`) {
                return dialectType;
            }
        }
        if (dialectType.startsWith('!stream.resource')) {
            if (parser.match('<')) {
                const content = parser.skip('<', '>');
                return dialectType + content;
            }
            return dialectType;
        }
        return null;
    }

    _parseShapedTiedResult(parser) {
        // Parse: %arg0 as tensor<?x?xf32>{%d0, %d1}
        //    or: tensor<?x?xf32>{%d0, %d1}
        let tiedOperand = null;
        if (parser.match('%')) {
            tiedOperand = parser.expect('%');
            if (!parser.accept('id', 'as')) {
                throw new mlir.Error(`Expected 'as' after tied operand ${parser.location()}`);
            }
        }
        const resultType = parser.parseType();
        const dims = [];
        if (parser.accept('{')) {
            while (!parser.match('}')) {
                if (parser.match('%')) {
                    const dim = parser.expect('%');
                    dims.push(dim);
                    parser.accept(',');
                } else {
                    break;
                }
            }
            parser.expect('}');
        }
        return { kind: 'ShapedTiedResult', tiedOperand, resultType, dims };
    }

    parseOperation(parser, opName, op) {
        if (opName === 'stream.cmd.execute') {
            if (parser.accept('id', 'with')) {
                if (parser.accept('(')) {
                    while (!parser.match(')') && !parser.match('eof')) {
                        if (parser.match('%')) {
                            const captureVar = parser.expect('%');
                            op.operands.push({ value: captureVar });
                            if (parser.accept('id', 'as')) {
                                parser.accept('%');
                            }
                            if (parser.accept(':')) {
                                const type = parser.parseType();
                                if (op.operands.length > 0) {
                                    op.operands[op.operands.length - 1].type = type;
                                }
                                if (parser.match('{')) {
                                    parser.skip('{', '}');
                                }
                            }
                            if (!parser.accept(',')) {
                                break;
                            }
                        } else {
                            break;
                        }
                    }
                    parser.accept(')');
                }
            }
            if (parser.accept('->')) {
                const resultType = parser.parseType();
                op.results.push({ type: resultType });
            }
            if (parser.match('{')) {
                const region = {};
                parser.parseRegion(region);
                op.regions.push(region);
            }
            return true;
        }
        return super.parseOperation(parser, opName, op);
    }
};

mlir.IREETensorExtDialect = class extends mlir.Dialect {

    constructor(operations) {
        super('iree_tensor_ext', operations);
    }

    parseType(parser, dialectType) {
        if (dialectType.startsWith('!iree_tensor_ext.dispatch.tensor')) {
            if (parser.match('<')) {
                const content = parser.skip('<', '>');
                return dialectType + content;
            }
            return dialectType;
        }
        return null;
    }
};

mlir.LinalgDialect = class extends mlir.Dialect {

    constructor(operations) {
        super('linalg', operations);
    }

    parseOperation(parser, opName, op) {

        if (opName === 'linalg.generic') {
            return this._parseGenericOp(parser, op);
        }

        if (opName === 'linalg.init_tensor') {
            if (parser.accept('[')) {
                const dims = [];
                while (!parser.match(']')) {
                    if (parser.match('%')) {
                        dims.push(parser.expect('%'));
                    } else if (parser.match('int')) {
                        dims.push(parser.expect('int'));
                    }
                    parser.accept(',');
                }
                parser.expect(']');
                op.attributes.push({ name: 'static_sizes', value: dims });
            }
            if (parser.accept(':')) {
                parser.parseArgumentTypes(op.results);
            }
            return true;
        }

        if (opName === 'linalg.fill') {
            if (parser.match('id', 'ins') || parser.match('{')) {
                return this._parseInsOutsOp(parser, op);
            }
            if (parser.accept('(')) {
                op.operands = parser.parseArguments();
                parser.expect(')');
            }
            if (parser.match('{')) {
                parser.parseAttributeDict(op.attributes);
            }
            if (parser.accept(':')) {
                parser.parseArgumentTypes(op.operands);
            }
            if (parser.accept('->')) {
                parser.parseArgumentTypes(op.results);
            }
            return true;
        }

        if (opName === 'linalg.conv') {
            if (parser.accept('(')) {
                op.operands = parser.parseArguments();
                parser.expect(')');
            }
            if (parser.match('{')) {
                parser.parseAttributeDict(op.attributes);
            }
            if (parser.accept(':')) {
                parser.parseArgumentTypes(op.operands);
            }
            return true;
        }

        if (opName === 'linalg.yield') {
            op.operands = parser.parseArguments();
            if (parser.accept(':')) {
                parser.parseArgumentTypes(op.operands);
            }
            return true;
        }

        // Many linalg operations (especially named structured ops) follow the pattern:
        // linalg.<op_name> {attributes} ins(...) outs(...) [-> type]
        // These named structured ops are generated at LLVM build time from YAML
        // and won't be in our schema, so we need custom parsing for them.

        // First try default parsing (for operations in schema)
        const hasSchemaEntry = this._operations.has(opName);
        if (hasSchemaEntry) {
            return super.parseOperation(parser, opName, op);
        }

        // Not in schema - try to parse as ins/outs operation
        // These are named structured ops generated from LLVM build time from YAML
        if (parser.match('{') || parser.match('id', 'ins')) {
            return this._parseInsOutsOp(parser, op);
        }

        return false;
    }

    _parseInsOutsOp(parser, op) {
        // Parse: linalg.op {attrs} ins(%0, %1 : type, type) outs(%2 : type) [-> type]

        // Parse optional attribute dictionary
        if (parser.match('{')) {
            parser.parseAttributeDict(op.attributes);
        }

        // Parse 'ins' section
        if (parser.accept('id', 'ins')) {
            if (!parser.accept('(')) {
                return false;
            }
            // Parse operands: %0, %1
            while (parser.match('%')) {
                const value = parser.expect('%');
                op.operands.push({ value });
                if (!parser.accept(',')) {
                    break;
                }
            }
            // Parse types: : type1, type2
            if (parser.accept(':')) {
                let idx = 0;
                const startIdx = 0;
                while (!parser.match(')')) {
                    const type = parser.parseType();
                    if (startIdx + idx < op.operands.length) {
                        op.operands[startIdx + idx].type = type;
                    }
                    idx++;
                    if (!parser.accept(',')) {
                        break;
                    }
                }
            }
            if (!parser.accept(')')) {
                return false;
            }
        }

        // Parse 'outs' section
        if (parser.accept('id', 'outs')) {
            if (!parser.accept('(')) {
                return false;
            }
            const outsStart = op.operands.length;
            // Parse operands: %2, %3
            while (parser.match('%')) {
                const value = parser.expect('%');
                op.operands.push({ value });
                if (!parser.accept(',')) {
                    break;
                }
            }
            // Parse types: : type1, type2
            if (parser.accept(':')) {
                let idx = 0;
                while (!parser.match(')')) {
                    const type = parser.parseType();
                    if (outsStart + idx < op.operands.length) {
                        op.operands[outsStart + idx].type = type;
                    }
                    idx++;
                    if (!parser.accept(',')) {
                        break;
                    }
                }
            }
            if (!parser.accept(')')) {
                return false;
            }
        }

        // Note: Some linalg operations like linalg.broadcast have required attributes
        // (e.g., dimensions = [0]) that appear after outs. These operations have
        // custom parsers in the reference MLIR implementation and should be handled
        // by operation-specific code in the schema, not by this generic parser.

        // Some linalg operations may have a region after the signature
        if (parser.match('{')) {
            const region = {};
            parser.parseRegion(region);
            op.regions.push(region);
        }

        // Parse optional return type (can come after region for linalg.generic)
        if (parser.accept('->') || parser.accept('id', 'to')) {
            if (op.results.length > 0) {
                parser.parseArgumentTypes(op.results);
            } else {
                const type = parser.parseType();
                op.results.push({ type });
            }
        }

        return true;
    }

    _parseGenericOp(parser, op) {
        if (parser.match('{') || parser.match('#')) {
            if (parser.match('#')) {
                const attrRef = parser.expect('#');
                op.attributes.push({ name: 'trait', value: attrRef });
            } else {
                parser.parseAttributeDict(op.attributes);
            }
        }
        if (parser.accept('id', 'ins')) {
            parser.expect('(');
            while (parser.match('%')) {
                const value = parser.expect('%');
                op.operands.push({ value });
                if (!parser.accept(',')) {
                    break;
                }
            }
            if (parser.accept(':')) {
                let idx = 0;
                while (!parser.match(')')) {
                    const type = parser.parseType();
                    if (idx < op.operands.length) {
                        op.operands[idx].type = type;
                    }
                    idx++;
                    if (!parser.accept(',')) {
                        break;
                    }
                }
            }
            parser.expect(')');
        }
        if (parser.accept('id', 'outs')) {
            parser.expect('(');
            const outsStart = op.operands.length;
            while (parser.match('%')) {
                const value = parser.expect('%');
                op.operands.push({ value });
                if (!parser.accept(',')) {
                    break;
                }
            }
            if (parser.accept(':')) {
                let idx = 0;
                while (!parser.match(')')) {
                    const type = parser.parseType();
                    if (outsStart + idx < op.operands.length) {
                        op.operands[outsStart + idx].type = type;
                    }
                    idx++;
                    if (!parser.accept(',')) {
                        break;
                    }
                }
            }
            parser.expect(')');
        }
        if (parser.accept('id', 'attrs')) {
            parser.expect('=');
            parser.parseAttributeDict(op.attributes);
        }
        if (parser.match('{')) {
            const region = {};
            parser.parseRegion(region);
            op.regions.push(region);
        }
        if (parser.accept('->')) {
            parser.parseArgumentTypes(op.results);
        }
        return true;
    }
};

mlir.ONNXDialect = class extends mlir.Dialect {

    constructor(operations) {
        super('onnx', operations);
    }

    parseOperation(parser, opName, op) {

        // onnx.Constant has custom assembly format: dense<...> : type
        // Similar to stablehlo.constant
        if (opName === 'onnx.Constant') {
            // Parse attribute (e.g., dense<"0x...">, dense<[1, 2, 3]>, etc.)
            const value = parser.parseValue();
            if (value) {
                op.attributes.push({ name: 'value', value: value.value });
            }
            // Parse optional : type
            if (parser.accept(':')) {
                parser.parseArgumentTypes(op.results);
            }
            return true;
        }

        return super.parseOperation(parser, opName, op);
    }
};

mlir.MhloDialect = class extends mlir.Dialect {

    constructor(operations) {
        super('mhlo', operations);
        this._binaryOps = new Set([
            'mhlo.add', 'mhlo.subtract', 'mhlo.multiply', 'mhlo.divide', 'mhlo.remainder',
            'mhlo.maximum', 'mhlo.minimum', 'mhlo.and', 'mhlo.or', 'mhlo.xor',
            'mhlo.shift_left', 'mhlo.shift_right_arithmetic', 'mhlo.shift_right_logical',
            'mhlo.atan2', 'mhlo.power', 'mhlo.complex'
        ]);
        this._unaryOps = new Set([
            'mhlo.abs', 'mhlo.acos', 'mhlo.acosh', 'mhlo.asin', 'mhlo.asinh', 'mhlo.atanh',
            'mhlo.cbrt', 'mhlo.ceil', 'mhlo.cos', 'mhlo.cosine', 'mhlo.cosh', 'mhlo.count_leading_zeros',
            'mhlo.erf', 'mhlo.exp', 'mhlo.exponential', 'mhlo.exponential_minus_one', 'mhlo.expm1',
            'mhlo.floor', 'mhlo.imag', 'mhlo.is_finite',
            'mhlo.log', 'mhlo.log_plus_one', 'mhlo.logistic', 'mhlo.negate', 'mhlo.not',
            'mhlo.popcnt', 'mhlo.real', 'mhlo.round_nearest_afz', 'mhlo.round_nearest_even',
            'mhlo.rsqrt', 'mhlo.sign', 'mhlo.sine', 'mhlo.sinh', 'mhlo.sqrt',
            'mhlo.tan', 'mhlo.tanh', 'mhlo.copy'
        ]);
        this._typeConversionOps = new Set([
            'mhlo.bitcast_convert', 'mhlo.convert', 'mhlo.reshape', 'mhlo.dynamic_reshape',
            'mhlo.bitcast', 'mhlo.broadcast_in_dim', 'mhlo.broadcast'
        ]);
        this._ternaryOps = new Set([
            'mhlo.clamp', 'mhlo.select'
        ]);
        this._terminatorOps = new Set([
            'mhlo.return'
        ]);
    }

    parseOperation(parser, opName, op) {
        if (this._binaryOps.has(opName)) {
            op.operands = parser.parseArguments();
            parser.parseOptionalAttrDictWithKeyword(op.attributes);
            if (parser.accept(':')) {
                const type = parser.parseType();
                for (const operand of op.operands) {
                    operand.type = type;
                }
                if (op.results.length > 0) {
                    op.results[0].type = type;
                }
            }
            return true;
        }
        if (this._unaryOps.has(opName)) {
            op.operands = parser.parseArguments();
            parser.parseOptionalAttrDictWithKeyword(op.attributes);
            if (parser.accept(':')) {
                const type = parser.parseType();
                if (op.operands.length > 0) {
                    op.operands[0].type = type;
                }
                if (op.results.length > 0) {
                    op.results[0].type = type;
                }
            }
            return true;
        }
        if (this._typeConversionOps.has(opName)) {
            op.operands = parser.parseArguments();
            parser.parseOptionalAttrDictWithKeyword(op.attributes);
            if (parser.accept(':')) {
                if (parser.accept('(')) {
                    const inputType = parser.parseType();
                    if (op.operands.length > 0) {
                        op.operands[0].type = inputType;
                    }
                    parser.expect(')');
                    parser.expect('->');
                    const outputType = parser.parseType();
                    if (op.results.length > 0) {
                        op.results[0].type = outputType;
                    }
                } else {
                    const type = parser.parseType();
                    if (op.operands.length > 0) {
                        op.operands[0].type = type;
                    }
                    if (op.results.length > 0) {
                        op.results[0].type = type;
                    }
                }
            }
            return true;
        }
        if (this._ternaryOps.has(opName)) {
            op.operands = parser.parseArguments();
            parser.parseOptionalAttrDictWithKeyword(op.attributes);
            if (parser.accept(':')) {
                if (opName === 'mhlo.select') {
                    const condType = parser.parseType();
                    if (op.operands.length > 0) {
                        op.operands[0].type = condType;
                    }
                    if (parser.accept(',')) {
                        const valueType = parser.parseType();
                        if (op.operands.length > 1) {
                            op.operands[1].type = valueType;
                        }
                        if (op.operands.length > 2) {
                            op.operands[2].type = valueType;
                        }
                        if (op.results.length > 0) {
                            op.results[0].type = valueType;
                        }
                    }
                } else {
                    const type = parser.parseType();
                    for (const operand of op.operands) {
                        operand.type = type;
                    }
                    if (op.results.length > 0) {
                        op.results[0].type = type;
                    }
                }
            }
            return true;
        }
        if (this._terminatorOps.has(opName)) {
            op.operands = parser.parseArguments();
            parser.parseOptionalAttrDictWithKeyword(op.attributes);
            if (parser.accept(':')) {
                const types = [];
                do {
                    types.push(parser.parseType());
                } while (parser.accept(','));
                for (let i = 0; i < op.operands.length && i < types.length; i++) {
                    op.operands[i].type = types[i];
                }
            }
            return true;
        }
        return super.parseOperation(parser, opName, op);
    }
};

mlir.QuantDialect = class extends mlir.Dialect {

    constructor(operations) {
        super('quant', operations);
    }

    parseType(parser, dialectType) {
        if (dialectType.startsWith('!quant.uniform') || dialectType.startsWith('!quant.calibrated')) {
            if (parser.match('<')) {
                const content = parser.skip('<', '>');
                return dialectType + content;
            }
            return dialectType;
        }
        return null;
    }
};

mlir.TosaDialect = class extends mlir.Dialect {

    constructor(operations) {
        super('tosa', operations);
        this._binaryOps = new Set(['tosa.maximum', 'tosa.minimum']);
        this._unaryOps = new Set([]);
        this._reduceOps = new Set(['tosa.reduce_min', 'tosa.reduce_max', 'tosa.reduce_sum', 'tosa.reduce_prod', 'tosa.reduce_any']);
    }

    parseType(parser, dialectType) {
        if (dialectType.startsWith('!tosa.shape')) {
            if (parser.match('<')) {
                const content = parser.skip('<', '>');
                return dialectType + content;
            }
            return dialectType;
        }
        return null;
    }

    parseOperation(parser, opName, op) {
        if (opName === 'tosa.variable') {
            parser.parseSymbolName('sym_name', op.attributes);
            if (parser.accept('=')) {
                const initialValue = parser.parseAttributeValue();
                op.attributes.push({ name: 'initial_value', value: initialValue });
            }
            if (parser.accept(':')) {
                const type = parser.parseType();
                op.attributes.push({ name: 'type', value: type });
            }
            return true;
        }
        if (this._binaryOps.has(opName) || this._unaryOps.has(opName) || this._reduceOps.has(opName) || opName === 'tosa.argmax' || opName === 'tosa.rescale') {
            op.operands = parser.parseArguments();
            if (parser.match('{')) {
                parser.parseAttributeDict(op.attributes);
            }
            if (parser.accept(':')) {
                if (parser.accept('(')) {
                    parser.parseArgumentTypes(op.operands);
                    parser.accept(')');
                }
                if (parser.accept('->')) {
                    parser.parseArgumentTypes(op.results);
                }
            }
            return true;
        }
        return super.parseOperation(parser, opName, op);
    }
};

mlir.IRDLDialect = class extends mlir.Dialect {

    constructor(operations) {
        super('irdl', operations);
    }

    parseOperation(parser, opName, op) {
        if (opName === 'irdl.operands' || opName === 'irdl.results' ||
            opName === 'irdl.parameters' || opName === 'irdl.attributes' ||
            opName === 'irdl.regions') {
            if (parser.accept('(')) {
                while (!parser.accept(')')) {
                    if (parser.match('id') || parser.match('string')) {
                        const paramName = parser.expect();
                        parser.expect(':');
                        const paramValue = parser.expect(); // Read the SSA value like %tensor
                        op.attributes.push({ name: paramName, value: paramValue });
                    }
                    parser.accept(',');
                }
            }
            op.loc = parser.parseLocation();
            return true;
        }
        return super.parseOperation(parser, opName, op);
    }
};

mlir.SPIRVDialect = class extends mlir.Dialect {

    constructor(operations) {
        super('spirv', operations);
    }

    parseType(parser, dialectType) {
        const simpleTypes = ['sampler', 'sampled_image', 'matrix', 'image', 'rtarray'];
        for (const simpleType of simpleTypes) {
            if (dialectType === `!spirv.${simpleType}`) {
                if (parser.match('<')) {
                    const content = parser.skip('<', '>');
                    return dialectType + content;
                }
                return dialectType;
            }
        }
        if (dialectType.startsWith('!spirv.ptr') || dialectType.startsWith('!spirv.array') ||
            dialectType.startsWith('!spirv.struct') || dialectType.startsWith('!spirv.coopmatrix')) {
            if (parser.match('<')) {
                const content = parser.skip('<', '>');
                return dialectType + content;
            }
            return dialectType;
        }
        return null;
    }

    parseOperation(parser, opName, op) {
        if (opName === 'spirv.SpecConstant') {
            parser.parseSymbolName('sym_name', op.attributes);
            if (parser.match('id', 'spec_id')) {
                parser.expect('id', 'spec_id');
                parser.expect('(');
                const specId = parser.parseValue();
                op.attributes.push({ name: 'spec_id', value: specId });
                parser.expect(')');
            }
            if (parser.accept('=')) {
                const defaultValue = parser.parseAttributeValue();
                op.attributes.push({ name: 'default_value', value: defaultValue });
            }
            if (parser.accept(':')) {
                const type = parser.parseType();
                op.attributes.push({ name: 'type', value: type });
            }
            return true;
        }
        // spirv.module / spv.module has addressing model and memory model before the region
        if (opName === 'spirv.module' || opName === 'spv.module') {
            // Parse: spv.module Logical GLSL450 [requires #spv.vce<...>] { ... }
            // Read addressing model (Logical, Physical32, Physical64, etc.)
            if (parser.match('id')) {
                const addressingModel = parser.expect('id');
                op.attributes.push({ name: 'addressing_model', value: addressingModel });
            }
            // Read memory model (GLSL450, Vulkan, OpenCL, etc.)
            if (parser.match('id')) {
                const memoryModel = parser.expect('id');
                op.attributes.push({ name: 'memory_model', value: memoryModel });
            }
            // Parse optional 'requires' clause
            if (parser.accept('id', 'requires')) {
                const vce = parser.parseAttributeValue();
                op.attributes.push({ name: 'vce_triple', value: vce });
            }
            // Parse region
            if (parser.match('{')) {
                const region = {};
                parser.parseRegion(region);
                op.regions.push(region);
            }
            return true;
        }
        // spirv.func / spv.func has symbol, function type, and optional control string
        if (opName === 'spirv.func' || opName === 'spv.func') {
            // Parse: spirv.func @symbol() "None" attributes {...} { ... }
            //    or: spirv.func @symbol(%arg: type) -> type "None" { ... }
            // Parse symbol (@name)
            if (parser.match('@')) {
                const symbol = parser.expect('@');
                op.attributes.push({ name: 'sym_name', value: symbol });
            }
            // Create function_type structure
            const function_type = {
                inputs: [],
                results: []
            };
            // Parse function signature (arguments and return type)
            if (parser.match('(')) {
                function_type.inputs = parser.parseFunctionArgumentList();
            }
            // Parse return type if present
            if (parser.accept('->')) {
                function_type.results = parser.parseFunctionResultList();
            }
            op.attributes.push({ name: 'function_type', value: function_type });
            // Parse control string ("None", "Inline", "DontInline", etc.)
            if (parser.match('string')) {
                const control = parser.expect('string');
                op.attributes.push({ name: 'function_control', value: control });
            }
            // Parse optional attributes
            if (parser.accept('id', 'attributes')) {
                parser.parseAttributeDict(op.attributes);
            }
            // Parse region
            if (parser.match('{')) {
                const region = {};
                parser.parseRegion(region);
                op.regions.push(region);
            }
            return true;
        }
        // spirv.GlobalVariable / spv.GlobalVariable - declares a global variable
        // Format: spv.GlobalVariable @symbol [built_in("...")] [bind(n, m)] : type
        if (opName === 'spirv.GlobalVariable' || opName === 'spv.GlobalVariable') {
            // Parse symbol reference
            if (parser.match('@')) {
                const symbol = parser.expect('@');
                op.attributes.push({ name: 'sym_name', value: symbol });
            }
            // Parse optional built_in attribute
            if (parser.accept('id', 'built_in')) {
                parser.expect('(');
                const builtIn = parser.expect('string');
                parser.expect(')');
                op.attributes.push({ name: 'built_in', value: builtIn });
            }
            // Parse optional bind attribute
            if (parser.accept('id', 'bind')) {
                parser.expect('(');
                const binding = parser.expect();
                parser.accept(',');
                const set = parser.expect();
                parser.expect(')');
                op.attributes.push({ name: 'descriptor_set', value: set });
                op.attributes.push({ name: 'binding', value: binding });
            }
            // Parse type after colon
            if (parser.accept(':')) {
                const type = parser.parseType();
                op.results = [{ type }];
            }
            return true;
        }
        // spirv.EntryPoint / spv.EntryPoint - defines entry point with execution model and interface variables
        // Format: spv.EntryPoint "GLCompute" @func_name, @var1, @var2, ...
        if (opName === 'spirv.EntryPoint' || opName === 'spv.EntryPoint') {
            // Parse execution model string ("GLCompute", "Vertex", "Fragment", etc.)
            if (parser.match('string')) {
                const executionModel = parser.expect('string');
                op.attributes.push({ name: 'execution_model', value: executionModel });
            }
            op.operands = [];
            while (parser.match('@')) {
                const symbol = parser.expect('@');
                op.operands.push({ value: symbol });
                parser.accept(',');
            }
            return true;
        }
        // spirv.ExecutionMode / spv.ExecutionMode - specifies execution mode for entry point
        // Format: spv.ExecutionMode @func_name "LocalSize", 8, 2, 1
        if (opName === 'spirv.ExecutionMode' || opName === 'spv.ExecutionMode') {
            // Parse function symbol
            if (parser.match('@')) {
                const symbol = parser.expect('@');
                op.operands.push({ value: symbol });
            }
            // Parse execution mode string
            if (parser.match('string')) {
                const mode = parser.expect('string');
                op.attributes.push({ name: 'execution_mode', value: mode });
            }
            // Parse mode parameters (comma-separated integers)
            while (parser.accept(',')) {
                if (parser.match('int') || parser.match('number') || parser.match('id')) {
                    const param = parser.expect();
                    op.operands.push({ value: param });
                } else {
                    break;
                }
            }
            return true;
        }
        // spirv.mlir.loop / spv.mlir.loop - region with multi-block control flow
        // spirv.mlir.selection / spv.mlir.selection - region with multi-block control flow
        if (opName === 'spirv.mlir.loop' || opName === 'spv.mlir.loop' ||
            opName === 'spirv.mlir.selection' || opName === 'spv.mlir.selection') {
            // These operations have regions that are parsed by the generic parser
            // Add sentinel attribute to manipulate the heuristic at line 1226:
            // Setting op.attributes.length > 0 forces the else branch (region parsing)
            op.attributes.push({ name: '_has_region', value: true });
            // Return false to let the generic parser handle the region
            return false;
        }
        // spirv.Branch / spv.Branch and other branch operations with successors
        if (opName === 'spirv.Branch' || opName === 'spv.Branch' ||
            opName === 'spirv.BranchConditional' || opName === 'spv.BranchConditional' ||
            opName.includes('Branch')) {
            // Parse operands if any (for conditional branches)
            op.operands = parser.parseArguments();

            // Parse successors
            if (parser.match('^')) {
                op.successors = [];
                while (parser.match('^')) {
                    const successor = {};
                    successor.label = parser.expect('^');
                    // Parse successor arguments with types
                    // Format: ^label(%val1, %val2, ... : type1, type2, ...)
                    if (parser.accept('(')) {
                        successor.arguments = [];
                        // Parse all values first
                        while (!parser.match(':') && !parser.match(')')) {
                            if (parser.match('%')) {
                                const arg = {};
                                arg.value = parser.expect('%');
                                successor.arguments.push(arg);
                                parser.accept(',');
                            } else {
                                break;
                            }
                        }
                        // Parse types if present
                        if (parser.accept(':')) {
                            let idx = 0;
                            while (idx < successor.arguments.length && !parser.match(')')) {
                                const type = parser.parseType();
                                successor.arguments[idx].type = type;
                                idx++;
                                parser.accept(',');
                            }
                        }
                        parser.accept(')');
                    }
                    op.successors.push(successor);
                    if (!parser.accept(',')) {
                        break;
                    }
                }
            }
            return true;
        }
        // spirv.CompositeInsert with 'into' keyword
        // Format: spirv.CompositeInsert %object, %composite[indices] : object-type into composite-type
        if (opName === 'spirv.CompositeInsert' || opName === 'spv.CompositeInsert') {
            // Parse operands (object and composite)
            op.operands = parser.parseArguments();
            // Parse indices as attributes
            if (parser.accept('[')) {
                const indices = [];
                while (!parser.accept(']')) {
                    const index = parser.expect();
                    if (parser.accept(':')) {
                        parser.expect(); // Skip type (e.g., i32)
                    }
                    indices.push(index);
                    parser.accept(',');
                }
                op.attributes.push({ name: 'indices', value: indices });
            }
            // Parse operand types after ':'
            if (parser.accept(':')) {
                parser.parseArgumentTypes(op.operands);
            }
            // Parse result type after 'into'
            if (parser.accept('id', 'into')) {
                const resultType = parser.parseType();
                op.results = [{ type: resultType }];
            }
            return true;
        }
        return super.parseOperation(parser, opName, op);
    }
};

mlir.CFDialect = class extends mlir.Dialect {

    constructor(operations) {
        super('cf', operations);
    }
};

mlir.EmitCDialect = class extends mlir.Dialect {

    constructor(operations) {
        super('emitc', operations);
    }

    parseOperation(parser, opName, op) {
        if (opName === 'emitc.global' || opName === 'emitc.field') {
            if (parser.match('id', 'extern')) {
                const externAttr = parser.expect('id');
                op.attributes.push({ name: 'extern', value: externAttr });
            }
            if (parser.match('id', 'static')) {
                const staticAttr = parser.expect('id');
                op.attributes.push({ name: 'static', value: staticAttr });
            }
            if (parser.match('id', 'const')) {
                const constAttr = parser.expect('id');
                op.attributes.push({ name: 'const', value: constAttr });
            }
            parser.parseSymbolName('sym_name', op.attributes);
            if (parser.accept(':')) {
                const type = parser.parseType();
                op.attributes.push({ name: 'type', value: type });
            }
            if (parser.accept('=')) {
                const initialValue = parser.parseAttributeValue();
                op.attributes.push({ name: 'initial_value', value: initialValue });
            }
            return true;
        }
        return super.parseOperation(parser, opName, op);
    }
};

mlir.AsukaDialect = class extends mlir.Dialect {

    constructor(operations) {
        super('asuka', operations);
        // https://github.com/monellz/FlashTensor/blob/main/bench/ea.mlir
        // uses batch_dims and reduce_dims not valid given the assemblyFormat spec.
        // Custom parsing preserves compatibility with this file.
        this._customParse = new Set(['asuka.dot', 'asuka.add', 'asuka.split', 'asuka.softmax']);
    }

    parseOperation(parser, opName, op) {
        if (this._customParse.has(opName)) {
            this._operations.get(opName).hasParseOperation = false;
            op.operands = parser.parseArguments();
            while (parser.match('id') && !parser.match(':') && !parser.match('{')) {
                const attrName = parser.expect('id');
                if (parser.accept('=')) {
                    let attrValue = null;
                    if (parser.match('[')) {
                        attrValue = parser.parseValue();
                        if (parser.match('id') && parser.token.value === 'x') {
                            parser.expect('id'); // consume 'x'
                            const secondValue = parser.parseValue();
                            attrValue = { kind: 'pair', first: attrValue, second: secondValue };
                        }
                    } else {
                        attrValue = parser.parseValue();
                    }
                    op.attributes.push({ name: attrName, value: attrValue });
                    parser.accept(',');
                }
            }
            if (parser.accept(':')) {
                parser.parseArgumentTypes(op.operands);
            }
            if (parser.accept('->')) {
                if (op.results.length > 0) {
                    parser.parseArgumentTypes(op.results);
                } else {
                    op.results = parser.parseArguments();
                }
            }
            return true;
        }
        return super.parseOperation(parser, opName, op);
    }
};

mlir.AsyncDialect = class extends mlir.Dialect {

    constructor(operations) {
        super('async', operations);
    }

    parseType(parser, dialectType) {
        const simpleTypes = ['token', 'group', 'coro.id', 'coro.handle', 'coro.state'];
        for (const simpleType of simpleTypes) {
            if (dialectType === `!async.${simpleType}`) {
                return dialectType;
            }
        }
        if (dialectType === '!async.value') {
            if (parser.match('<')) {
                const content = parser.skip('<', '>');
                return dialectType + content;
            }
            return dialectType;
        }
        return null;
    }

    parseOperation(parser, opName, op) {
        if (opName === 'async.execute') {
            return this._parseExecuteOp(parser, op);
        }
        if (opName === 'async.func') {
            return this._parseFuncOp(parser, op);
        }
        if (opName === 'async.await') {
            return this._parseAwaitOp(parser, op);
        }
        return super.parseOperation(parser, opName, op);
    }

    _parseExecuteOp(parser, op) {
        if (parser.accept('[')) {
            while (!parser.match(']')) {
                op.operands.push({ value: parser.expect('%') });
                if (!parser.accept(',')) {
                    break;
                }
            }
            parser.expect(']');
        }
        if (parser.accept('(')) {
            while (!parser.match(')') && !parser.match(':')) {
                op.operands.push({ value: parser.expect('%') });
                if (parser.accept('id', 'as')) {
                    parser.expect('%');
                }
                if (parser.match(':')) {
                    parser.parseType();
                }
                if (!parser.accept(',')) {
                    break;
                }
            }
            parser.expect(')');
        }
        if (parser.accept('->')) {
            if (parser.accept('(')) {
                while (!parser.match(')')) {
                    const resultType = parser.parseType();
                    if (op.results.length < 1) {
                        op.results.push({ type: '!async.token' });
                    }
                    op.results.push({ type: resultType });
                    if (!parser.accept(',')) {
                        break;
                    }
                }
                parser.expect(')');
            } else {
                const resultType = parser.parseType();
                if (op.results.length < 1) {
                    op.results.push({ type: '!async.token' });
                }
                op.results.push({ type: resultType });
            }
        } else if (op.results.length === 1 && !op.results[0].type) {
            op.results[0].type = '!async.token';
        }
        parser.parseOptionalAttrDictWithKeyword(op.attributes);
        if (parser.match('{')) {
            const region = {};
            parser.parseRegion(region);
            op.regions.push(region);
        }
        return true;
    }

    _parseFuncOp(parser, op) {
        parser.parseOptionalVisibilityKeyword(op.attributes);
        parser.parseSymbolName('sym_name', op.attributes);
        const type = {};
        type.inputs = parser.parseFunctionArgumentList();
        parser.parseOptionalAttrDictWithKeyword(op.attributes);
        type.results = [];
        if (parser.accept('->')) {
            if (parser.accept('(')) {
                while (!parser.match(')')) {
                    type.results.push({ type: parser.parseType() });
                    if (!parser.accept(',')) {
                        break;
                    }
                }
                parser.expect(')');
            } else {
                type.results.push({ type: parser.parseType() });
            }
        }
        op.attributes.push({ name: 'function_type', value: type });
        parser.parseOptionalAttrDictWithKeyword(op.attributes);
        if (parser.match('{')) {
            const region = {};
            parser.parseRegion(region);
            op.regions.push(region);
        }
        return true;
    }

    _parseAwaitOp(parser, op) {
        op.operands.push({ value: parser.expect('%') });
        parser.expect(':');
        const operandType = parser.parseType();
        if (op.operands.length > 0) {
            op.operands[0].type = operandType;
        }
        if (operandType && operandType.startsWith('!async.value')) {
            const match = operandType.match(/!async\.value<(.+)>/);
            if (match && op.results.length > 0) {
                [, op.results[0].type] = match;
            }
        }
        parser.parseOptionalAttrDictWithKeyword(op.attributes);
        return true;
    }
};

mlir.ArithDialect = class extends mlir.Dialect {

    constructor(operations) {
        super('arith', operations);
    }

    parseOperation(parser, opName, op) {
        if (opName === 'arith.select') {
            return this._parseSelectOp(parser, op);
        }
        return super.parseOperation(parser, opName, op);
    }

    _parseSelectOp(parser, op) {
        op.operands = parser.parseArguments();
        if (parser.match('{')) {
            parser.parseAttributeDict(op.attributes);
        }
        if (parser.accept(':')) {
            const condType = parser.parseType();
            if (parser.accept(',')) {
                const resultType = parser.parseType();
                if (op.operands.length > 0) {
                    op.operands[0].type = condType;
                }
                if (op.operands.length > 1) {
                    op.operands[1].type = resultType;
                    op.operands[2].type = resultType;
                }
                if (op.results.length > 0) {
                    op.results[0].type = resultType;
                } else {
                    op.results.push({ type: resultType });
                }
            } else if (op.results.length > 0) {
                op.results[0].type = condType;
            } else {
                op.results.push({ type: condType });
            }
        }
        return true;
    }
};

mlir.BuiltinDialect = class extends mlir.Dialect {

    constructor(operations) {
        super('builtin', operations);
    }

    parseOperation(parser, opName, op) {
        if (opName === 'builtin.func') {
            return this._parseFuncOp(parser, op);
        }
        if (opName === 'builtin.call' || opName === 'builtin.call_indirect') {
            parser.parseSymbolName('callee', op.attributes);
            op.operands = parser.parseArguments();
            if (parser.accept(':')) {
                parser.parseArgumentTypes(op.operands);
            }
            if (parser.accept('->')) {
                parser.parseArgumentTypes(op.results);
            }
            return true;
        }
        return super.parseOperation(parser, opName, op);
    }

    _parseFuncOp(parser, op) {
        parser.parseOptionalVisibilityKeyword(op.attributes);
        parser.parseSymbolName('sym_name', op.attributes);
        const type = {};
        type.inputs = parser.parseFunctionArgumentList();
        parser.parseOptionalAttrDictWithKeyword(op.attributes);
        type.results = [];
        if (parser.accept('->')) {
            for (const result of parser.parseFunctionResultList()) {
                type.results.push(result);
            }
        }
        op.attributes.push({ name: 'function_type', value: type });
        parser.parseOptionalAttrDictWithKeyword(op.attributes);
        if (parser.match('{')) {
            const region = {};
            parser.parseRegion(region);
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
        return true;
    }
};

mlir.BufferizationDialect = class extends mlir.Dialect {

    constructor(operations) {
        super('bufferization', operations);
    }

    parseOperation(parser, opName, op) {
        if (opName === 'bufferization.alloc_tensor') {
            if (!parser.accept('(')) {
                return false;
            }
            while (!parser.match(')')) {
                if (parser.match('%')) {
                    const operand = {};
                    operand.value = parser.expect('%');
                    op.operands.push(operand);
                    if (!parser.accept(',')) {
                        break;
                    }
                } else {
                    break;
                }
            }
            parser.expect(')');
            if (parser.accept('id', 'copy')) {
                parser.expect('(');
                const copyOperand = {};
                copyOperand.value = parser.expect('%');
                copyOperand.name = 'copy';
                op.operands.push(copyOperand);
                parser.expect(')');
            }
            if (parser.accept('id', 'size_hint')) {
                parser.expect('=');
                const sizeHintOperand = {};
                sizeHintOperand.value = parser.expect('%');
                sizeHintOperand.name = 'size_hint';
                op.operands.push(sizeHintOperand);
            }
            parser.parseOptionalAttrDict(op.attributes);
            if (parser.accept(':')) {
                const type = parser.parseType();
                if (op.results.length === 0) {
                    op.results.push({ type });
                } else {
                    op.results[0].type = type;
                }
            }
            return true;
        }
        return super.parseOperation(parser, opName, op);
    }
};

mlir.SCFDialect = class extends mlir.Dialect {

    constructor(operations) {
        super('scf', operations);
    }

    parseOperation(parser, opName, op) {
        if (opName === 'scf.for') {
            return this._parseForOp(parser, op);
        }
        if (opName === 'scf.if') {
            return this._parseIfOp(parser, op);
        }
        if (opName === 'scf.while') {
            return this._parseWhileOp(parser, op);
        }
        if (opName === 'scf.forall') {
            return this._parseForallOp(parser, op);
        }
        if (opName === 'scf.forall.in_parallel') {
            return this._parseInParallelOp(parser, op);
        }
        if (opName === 'scf.parallel') {
            return this._parseParallelOp(parser, op);
        }
        return super.parseOperation(parser, opName, op);
    }

    _parseForOp(parser, op) {
        // scf.for [unsigned] %inductionVar = %lb to %ub step %step [iter_args(...)] [: type] { region }
        // Check for optional "unsigned" keyword
        if (parser.accept('id', 'unsigned')) {
            op.attributes.push({ name: 'unsignedCmp', value: true });
        }
        // Parse induction variable: %inductionVar
        if (!parser.match('%')) {
            return false;
        }
        const inductionVar = parser.expect('%');
        // Parse '='
        if (!parser.accept('=')) {
            return false;
        }
        // Parse lower bound: %lb
        if (parser.match('%')) {
            op.operands.push({ value: parser.expect('%') });
        } else {
            return false;
        }
        // Parse 'to' keyword
        if (!parser.accept('id', 'to')) {
            return false;
        }
        // Parse upper bound: %ub
        if (parser.match('%')) {
            op.operands.push({ value: parser.expect('%') });
        } else {
            return false;
        }
        // Parse 'step' keyword
        if (!parser.accept('id', 'step')) {
            return false;
        }
        // Parse step: %step
        if (parser.match('%')) {
            op.operands.push({ value: parser.expect('%') });
        } else {
            return false;
        }
        // Parse optional iter_args
        if (parser.accept('id', 'iter_args')) {
            // iter_args(%arg = %init, ...) -> (type, ...)
            if (parser.accept('(')) {
                while (!parser.accept(')')) {
                    // Parse %arg = %init
                    if (parser.match('%')) {
                        parser.expect('%'); // Skip the loop-carried variable name
                    }
                    if (parser.accept('=')) {
                        if (parser.match('%')) {
                            op.operands.push({ value: parser.expect('%') });
                        } else {
                            // Handle non-SSA values (constants, etc.)
                            const value = parser.parseValue();
                            if (value) {
                                op.operands.push(value);
                            }
                        }
                    }
                    parser.accept(',');
                }
            }
            // Parse optional -> (result types)
            if (parser.accept('->')) {
                // Parse result types
                const resultTypes = [];
                if (parser.accept('(')) {
                    while (!parser.accept(')')) {
                        const resultType = parser.parseType();
                        resultTypes.push(resultType);
                        parser.accept(',');
                    }
                } else {
                    const resultType = parser.parseType();
                    resultTypes.push(resultType);
                }
                if (op.results.length > 0) {
                    for (let i = 0; i < resultTypes.length && i < op.results.length; i++) {
                        op.results[i].type = resultTypes[i];
                    }
                } else {
                    for (let i = 0; i < resultTypes.length; i++) {
                        op.results.push({ value: `%${i}`, type: resultTypes[i] });
                    }
                }
            }
        }
        // Parse optional type: : type
        if (parser.accept(':')) {
            parser.parseType(); // Parse and discard the induction variable type
        }
        // Parse region: { ... }
        if (parser.match('{')) {
            const region = {};
            parser.parseRegion(region);
            // Set the induction variable as the first block argument
            if (region.blocks && region.blocks.length > 0) {
                if (!region.blocks[0].arguments) {
                    region.blocks[0].arguments = [];
                }
                if (region.blocks[0].arguments.length > 0) {
                    region.blocks[0].arguments[0] = { value: inductionVar };
                } else {
                    region.blocks[0].arguments.push({ value: inductionVar });
                }
            }
            op.regions.push(region);
        }
        return true;
    }

    _parseIfOp(parser, op) {
        // scf.if %condition [-> (result_types)] { ... } [else { ... }]
        // Parse condition
        if (parser.match('%')) {
            op.operands.push({ value: parser.expect('%') });
        } else {
            return false;
        }
        // Parse optional result types
        if (parser.accept('->')) {
            const resultTypes = [];
            if (parser.accept('(')) {
                while (!parser.accept(')')) {
                    const resultType = parser.parseType();
                    resultTypes.push(resultType);
                    parser.accept(',');
                }
            } else {
                const resultType = parser.parseType();
                resultTypes.push(resultType);
            }
            // Set types on existing results or create new ones
            if (op.results.length > 0) {
                for (let i = 0; i < resultTypes.length && i < op.results.length; i++) {
                    op.results[i].type = resultTypes[i];
                }
            } else {
                for (const type of resultTypes) {
                    op.results.push({ type });
                }
            }
        }
        // Parse then region
        if (parser.match('{')) {
            const region = {};
            parser.parseRegion(region);
            op.regions.push(region);
        } else {
            return false;
        }
        // Parse optional else region
        if (parser.accept('id', 'else')) {
            if (parser.match('{')) {
                const region = {};
                parser.parseRegion(region);
                op.regions.push(region);
            }
        }
        return true;
    }

    _parseWhileOp(parser, op) {
        // scf.while (%arg = %init, ...) : (type, ...) -> (type, ...) { before region } do { after region }
        // Parse operands in parentheses
        if (parser.accept('(')) {
            while (!parser.accept(')')) {
                if (parser.match('%')) {
                    parser.expect('%'); // Skip variable name
                }
                if (parser.accept('=')) {
                    if (parser.match('%')) {
                        op.operands.push({ value: parser.expect('%') });
                    } else {
                        const value = parser.parseValue();
                        if (value) {
                            op.operands.push(value);
                        }
                    }
                }
                parser.accept(',');
            }
        }
        // Parse types
        if (parser.accept(':')) {
            // Parse operand types
            if (parser.accept('(')) {
                while (!parser.accept(')')) {
                    parser.parseType();
                    parser.accept(',');
                }
            }
            // Parse optional result types
            if (parser.accept('->')) {
                const resultTypes = [];
                if (parser.accept('(')) {
                    while (!parser.accept(')')) {
                        const resultType = parser.parseType();
                        resultTypes.push(resultType);
                        parser.accept(',');
                    }
                }
                // Set types on existing results or create new ones
                if (op.results.length > 0) {
                    for (let i = 0; i < resultTypes.length && i < op.results.length; i++) {
                        op.results[i].type = resultTypes[i];
                    }
                } else {
                    for (const type of resultTypes) {
                        op.results.push({ type });
                    }
                }
            }
        }
        // Parse before region
        if (parser.match('{')) {
            const region = {};
            parser.parseRegion(region);
            op.regions.push(region);
        }
        // Parse 'do' keyword and after region
        if (parser.accept('id', 'do')) {
            if (parser.match('{')) {
                const region = {};
                parser.parseRegion(region);
                op.regions.push(region);
            }
        }
        return true;
    }

    _parseForallOp(parser, op) {
        // scf.forall (%iv, ...) in (%ub, ...) [shared_outs(...)] [-> types] { region }
        // scf.forall (%iv, ...) = (%lb, ...) to (%ub, ...) step (%step, ...) [shared_outs(...)] [-> types] { region }
        // Parse induction variables: (%iv1, %iv2, ...)
        const inductionVars = [];
        if (!parser.accept('(')) {
            return false;
        }
        while (!parser.accept(')')) {
            if (parser.match('%')) {
                inductionVars.push(parser.expect('%'));
            } else {
                return false;
            }
            if (!parser.accept(',')) {
                if (parser.match(')')) {
                    parser.accept(')');
                    break;
                }
                return false;
            }
        }
        // Check for normalized (in) vs non-normalized (=) form
        const isNormalized = parser.accept('id', 'in');
        if (!isNormalized && !parser.accept('=')) {
            return false;
        }
        // Parse bounds and steps based on form
        if (isNormalized) {
            // Normalized form: just upper bounds
            if (!parser.accept('(')) {
                return false;
            }
            while (!parser.accept(')')) {
                if (parser.match('%')) {
                    op.operands.push({ value: parser.expect('%') });
                } else if (parser.match('int')) {
                    op.operands.push({ value: parser.expect('int'), type: 'int64' });
                } else {
                    return false;
                }
                parser.accept(',');
            }
        } else {
            // Non-normalized form: lower bounds, to, upper bounds, step, steps
            // Parse lower bounds: (%lb1, %lb2, ...)
            if (!parser.accept('(')) {
                return false;
            }
            while (!parser.accept(')')) {
                if (parser.match('%')) {
                    op.operands.push({ value: parser.expect('%') });
                } else if (parser.match('int')) {
                    op.operands.push({ value: parser.expect('int'), type: 'int64' });
                } else {
                    return false;
                }
                parser.accept(',');
            }
            // Parse 'to' keyword
            if (!parser.accept('id', 'to')) {
                return false;
            }
            // Parse upper bounds: (%ub1, %ub2, ...)
            if (!parser.accept('(')) {
                return false;
            }
            while (!parser.accept(')')) {
                if (parser.match('%')) {
                    op.operands.push({ value: parser.expect('%') });
                } else if (parser.match('int')) {
                    op.operands.push({ value: parser.expect('int'), type: 'int64' });
                } else {
                    return false;
                }
                parser.accept(',');
            }
            // Parse 'step' keyword
            if (!parser.accept('id', 'step')) {
                return false;
            }
            // Parse steps: (%step1, %step2, ...)
            if (!parser.accept('(')) {
                return false;
            }
            while (!parser.accept(')')) {
                if (parser.match('%')) {
                    op.operands.push({ value: parser.expect('%') });
                } else if (parser.match('int')) {
                    op.operands.push({ value: parser.expect('int'), type: 'int64' });
                } else {
                    return false;
                }
                parser.accept(',');
            }
        }
        // Parse optional shared_outs: shared_outs(%arg = %val, ...)
        if (parser.accept('id', 'shared_outs')) {
            if (!parser.accept('(')) {
                return false;
            }
            while (!parser.accept(')')) {
                if (parser.match('%')) {
                    parser.expect('%'); // Skip arg name
                }
                if (parser.accept('=')) {
                    if (parser.match('%')) {
                        op.operands.push({ value: parser.expect('%') });
                    } else {
                        const value = parser.parseValue();
                        if (value) {
                            op.operands.push(value);
                        }
                    }
                }
                parser.accept(',');
            }
        }
        // Parse optional result types: -> (types)
        if (parser.accept('->')) {
            if (parser.accept('(')) {
                while (!parser.accept(')')) {
                    const type = parser.parseType();
                    op.results.push({ type });
                    parser.accept(',');
                }
            } else {
                const type = parser.parseType();
                op.results.push({ type });
            }
        }
        // Parse region
        if (parser.match('{')) {
            const region = {};
            parser.parseRegion(region);
            op.regions.push(region);
        } else {
            return false;
        }
        return true;
    }

    _parseParallelOp(parser, op) {
        // scf.parallel (%iv, ...) = (%lb, ...) to (%ub, ...) step (%step, ...) [init (%val, ...)] [-> types] { region }
        // Parse induction variables: (%iv1, %iv2, ...)
        const inductionVars = [];
        if (!parser.accept('(')) {
            return false;
        }
        while (!parser.accept(')')) {
            if (parser.match('%')) {
                inductionVars.push(parser.expect('%'));
            } else {
                return false;
            }
            parser.accept(',');
        }
        // Parse '='
        if (!parser.accept('=')) {
            return false;
        }
        // Parse lower bounds: (%lb1, %lb2, ...)
        if (!parser.accept('(')) {
            return false;
        }
        while (!parser.accept(')')) {
            if (parser.match('%')) {
                op.operands.push({ value: parser.expect('%') });
            } else {
                return false;
            }
            parser.accept(',');
        }
        // Parse 'to' keyword
        if (!parser.accept('id', 'to')) {
            return false;
        }
        // Parse upper bounds: (%ub1, %ub2, ...)
        if (!parser.accept('(')) {
            return false;
        }
        while (!parser.accept(')')) {
            if (parser.match('%')) {
                op.operands.push({ value: parser.expect('%') });
            } else {
                return false;
            }
            parser.accept(',');
        }
        // Parse 'step' keyword
        if (!parser.accept('id', 'step')) {
            return false;
        }
        // Parse steps: (%step1, %step2, ...)
        if (!parser.accept('(')) {
            return false;
        }
        while (!parser.accept(')')) {
            if (parser.match('%')) {
                op.operands.push({ value: parser.expect('%') });
            } else {
                return false;
            }
            parser.accept(',');
        }
        // Parse optional init values: init (%val1, %val2, ...)
        if (parser.accept('id', 'init')) {
            if (!parser.accept('(')) {
                return false;
            }
            while (!parser.accept(')')) {
                if (parser.match('%')) {
                    op.operands.push({ value: parser.expect('%') });
                } else {
                    const value = parser.parseValue();
                    if (value) {
                        op.operands.push(value);
                    }
                }
                parser.accept(',');
            }
        }
        // Parse optional result types: -> (types)
        if (parser.accept('->')) {
            if (parser.accept('(')) {
                while (!parser.accept(')')) {
                    const type = parser.parseType();
                    op.results.push({ type });
                    parser.accept(',');
                }
            } else {
                const type = parser.parseType();
                op.results.push({ type });
            }
        }
        // Parse region
        if (parser.match('{')) {
            const region = {};
            parser.parseRegion(region);
            // Set induction vars as block arguments
            if (region.blocks && region.blocks.length > 0 && inductionVars.length > 0) {
                if (!region.blocks[0].arguments) {
                    region.blocks[0].arguments = [];
                }
                for (const iv of inductionVars) {
                    region.blocks[0].arguments.push({ value: iv });
                }
            }
            op.regions.push(region);
        } else {
            return false;
        }
        return true;
    }

    _parseInParallelOp(parser, op) {
        // scf.forall.in_parallel { region }
        // Just parse the region
        if (parser.match('{')) {
            const region = {};
            parser.parseRegion(region);
            op.regions.push(region);
        } else {
            return false;
        }
        return true;
    }

};

mlir.ShapeDialect = class extends mlir.Dialect {

    constructor(operations) {
        super('shape', operations);
    }

    parseType(parser, dialectType) {
        if (dialectType === '!shape.shape' || dialectType === '!shape.witness' ||
            dialectType === '!shape.size' || dialectType === '!shape.value_shape') {
            return dialectType;
        }
        return null;
    }

    parseOperation(parser, opName, op) {
        if (opName === 'shape.func') {
            return this._parseFuncOp(parser, op);
        }
        if (opName === 'shape.assuming') {
            return this._parseAssumingOp(parser, op);
        }
        if (opName === 'shape.const_shape') {
            return this._parseConstShapeOp(parser, op);
        }
        if (opName === 'shape.reduce') {
            return this._parseReduceOp(parser, op);
        }
        if (opName === 'shape.function_library') {
            return this._parseFunctionLibraryOp(parser, op);
        }
        return super.parseOperation(parser, opName, op);
    }

    _parseFuncOp(parser, op) {
        parser.parseOptionalVisibilityKeyword(op.attributes);
        parser.parseSymbolName('sym_name', op.attributes);
        const type = {};
        type.inputs = parser.parseFunctionArgumentList();
        parser.parseOptionalAttrDictWithKeyword(op.attributes);
        type.results = [];
        if (parser.accept('->')) {
            for (const result of parser.parseFunctionResultList()) {
                type.results.push(result);
            }
        }
        op.attributes.push({ name: 'function_type', value: type });
        parser.parseOptionalAttrDictWithKeyword(op.attributes);
        if (parser.match('{')) {
            const region = {};
            parser.parseRegion(region);
            op.regions.push(region);
        }
        return true;
    }

    _parseAssumingOp(parser, op) {
        if (!parser.match('%')) {
            return false;
        }
        op.operands.push({ value: parser.expect('%') });
        if (parser.accept('->')) {
            parser.parseArgumentTypes(op.results);
        }
        if (parser.match('{')) {
            const region = {};
            parser.parseRegion(region);
            op.regions.push(region);
        }
        parser.parseOptionalAttrDict(op.attributes);
        return true;
    }

    _parseConstShapeOp(parser, op) {
        parser.parseOptionalAttrDict(op.attributes);
        const extents = parser.parseAttributeValue();
        op.attributes.push({ name: 'shape', value: extents });
        if (parser.accept(':')) {
            const type = parser.parseType();
            op.results.push({ type });
        }
        return true;
    }

    _parseReduceOp(parser, op) {
        if (!parser.match('(')) {
            return false;
        }
        parser.accept('(');
        while (parser.match('%')) {
            op.operands.push({ value: parser.expect('%') });
            if (!parser.accept(',')) {
                break;
            }
        }
        parser.accept(')');
        if (parser.accept(':')) {
            const shapeType = parser.parseType();
            if (op.operands.length > 0) {
                op.operands[0].type = shapeType;
            }
        }
        if (parser.accept('->')) {
            parser.parseArgumentTypes(op.results);
            for (let i = 1; i < op.operands.length && i - 1 < op.results.length; i++) {
                op.operands[i].type = op.results[i - 1].type;
            }
        }
        if (parser.match('{')) {
            const region = {};
            parser.parseRegion(region);
            op.regions.push(region);
        }
        parser.parseOptionalAttrDict(op.attributes);
        return true;
    }

    _parseFunctionLibraryOp(parser, op) {
        parser.parseSymbolName('sym_name', op.attributes);
        parser.parseOptionalAttrDictWithKeyword(op.attributes);
        if (parser.match('{')) {
            const region = {};
            parser.parseRegion(region);
            op.regions.push(region);
        }
        return true;
    }
};

mlir.SparseTensorDialect = class extends mlir.Dialect {

    constructor(operations) {
        super('sparse_tensor', operations);
    }

    parseOperation(parser, opName, op) {
        if (opName === 'sparse_tensor.iterate') {
            return this._parseIterateOp(parser, op);
        }
        if (opName === 'sparse_tensor.coiterate') {
            return this._parseCoIterateOp(parser, op);
        }
        return super.parseOperation(parser, opName, op);
    }

    _parseIterateOp(parser, op) {
        if (!parser.match('%')) {
            return false;
        }
        parser.expect('%');
        if (!parser.accept('id', 'in')) {
            return false;
        }
        if (!parser.match('%')) {
            return false;
        }
        op.operands.push({ value: parser.expect('%') });
        if (parser.accept('id', 'at')) {
            parser.accept('(');
            while (parser.match('%') || parser.match('id')) {
                parser.expect();
                if (!parser.accept(',')) {
                    break;
                }
            }
            parser.accept(')');
        }
        if (parser.accept('id', 'iter_args')) {
            parser.accept('(');
            while (parser.match('%')) {
                op.operands.push({ value: parser.expect('%') });
                if (parser.accept('=')) {
                    parser.expect('%');
                }
                if (!parser.accept(',')) {
                    break;
                }
            }
            parser.accept(')');
        }
        if (parser.accept(':')) {
            parser.parseType();
        }
        if (parser.accept('->')) {
            parser.parseArgumentTypes(op.results);
        }
        if (parser.match('{')) {
            const region = {};
            parser.parseRegion(region);
            op.regions.push(region);
        }
        parser.parseOptionalAttrDict(op.attributes);
        return true;
    }

    _parseCoIterateOp(parser, op) {
        if (!parser.accept('(')) {
            return false;
        }
        while (parser.match('%')) {
            op.operands.push({ value: parser.expect('%') });
            if (!parser.accept(',')) {
                break;
            }
        }
        parser.accept(')');
        if (parser.accept('id', 'at')) {
            parser.accept('(');
            while (parser.match('%') || parser.match('id')) {
                parser.expect();
                if (!parser.accept(',')) {
                    break;
                }
            }
            parser.accept(')');
        }
        if (parser.accept('id', 'iter_args')) {
            parser.accept('(');
            while (parser.match('%')) {
                op.operands.push({ value: parser.expect('%') });
                if (parser.accept('=')) {
                    parser.expect('%');
                }
                if (!parser.accept(',')) {
                    break;
                }
            }
            parser.accept(')');
        }
        if (parser.accept(':')) {
            parser.accept('(');
            while (!parser.match(')')) {
                parser.parseType();
                if (!parser.accept(',')) {
                    break;
                }
            }
            parser.accept(')');
        }
        if (parser.accept('->')) {
            parser.parseArgumentTypes(op.results);
        }
        while (parser.accept('id', 'case')) {
            while (parser.match('%') || parser.match('id')) {
                parser.expect();
                if (!parser.accept(',')) {
                    break;
                }
            }
            if (parser.match('{')) {
                const region = {};
                parser.parseRegion(region);
                op.regions.push(region);
            }
        }
        parser.parseOptionalAttrDict(op.attributes);
        return true;
    }
};

mlir.FuncDialect = class extends mlir.Dialect {

    constructor(operations) {
        super('func', operations);
    }

    parseOperation(parser, opName, op) {
        if (opName === 'func.func') {
            parser.parseOptionalVisibilityKeyword(op.attributes);
            parser.parseSymbolName('sym_name', op.attributes);
            const type = {};
            type.inputs = parser.parseFunctionArgumentList();
            parser.parseOptionalAttrDictWithKeyword(op.attributes);
            type.results = [];
            if (parser.accept('->')) {
                for (const result of parser.parseFunctionResultList()) {
                    type.results.push(result);
                }
            }
            op.attributes.push({ name: 'function_type', value: type });
            parser.parseOptionalAttrDictWithKeyword(op.attributes);
            if (parser.match('{')) {
                const region = {};
                parser.parseRegion(region);
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
            return true;
        }
        return super.parseOperation(parser, opName, op);
    }
};

mlir.GpuDialect = class extends mlir.Dialect {

    constructor(operations) {
        super('gpu', operations);
        this.registerCustomParser('LaunchFuncOperands', this._parseLaunchFuncOperands.bind(this));
        this.registerCustomParser('AsyncDependencies', this._parseAsyncDependencies.bind(this));
    }

    parseOperation(parser, opName, op) {
        if (opName === 'gpu.func') {
            parser.parseOptionalVisibilityKeyword(op.attributes);
            parser.parseSymbolName('sym_name', op.attributes);
            const type = {};
            type.inputs = parser.parseFunctionArgumentList();
            parser.parseOptionalAttrDictWithKeyword(op.attributes);
            type.results = [];
            if (parser.accept('->')) {
                for (const result of parser.parseFunctionResultList()) {
                    type.results.push(result);
                }
            }
            op.attributes.push({ name: 'function_type', value: type });
            if (parser.match('id', 'kernel')) {
                parser.expect();
                op.attributes.push({ name: 'gpu.kernel', value: true });
            }
            parser.parseOptionalAttrDictWithKeyword(op.attributes);
            if (parser.match('{')) {
                const region = {};
                parser.parseRegion(region);
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
            return true;
        }
        if (opName === 'gpu.launch') {
            if (parser.accept('[')) {
                while (!parser.match(']')) {
                    if (parser.match('%')) {
                        const dep = {};
                        dep.value = parser.expect('%');
                        op.operands.push(dep);
                        if (!parser.accept(',')) {
                            break;
                        }
                    } else {
                        break;
                    }
                }
                parser.expect(']');
            }
            if (parser.accept('id', 'clusters')) {
                this._parseSizeAssignment(parser, op);
                parser.expect('id', 'in');
                this._parseSizeAssignment(parser, op);
            }
            parser.expect('id', 'blocks');
            this._parseSizeAssignment(parser, op);
            parser.expect('id', 'in');
            this._parseSizeAssignment(parser, op);
            parser.expect('id', 'threads');
            this._parseSizeAssignment(parser, op);
            parser.expect('id', 'in');
            this._parseSizeAssignment(parser, op);
            if (parser.accept('id', 'dynamic_shared_memory_size')) {
                const operand = {};
                operand.value = parser.expect('%');
                operand.name = 'dynamic_shared_memory_size';
                op.operands.push(operand);
            }
            if (parser.accept('id', 'module')) {
                parser.expect('(');
                const moduleSymbol = parser.expect('@');
                op.attributes.push({ name: 'module', value: moduleSymbol });
                parser.expect(')');
            }
            if (parser.accept('id', 'function')) {
                parser.expect('(');
                const funcSymbol = parser.expect('@');
                op.attributes.push({ name: 'function', value: funcSymbol });
                parser.expect(')');
            }
            if (parser.accept('id', 'workgroup')) {
                parser.expect('(');
                while (!parser.match(')')) {
                    parser.expect('%');
                    parser.expect(':');
                    parser.parseType();
                    if (!parser.accept(',')) {
                        break;
                    }
                }
                parser.expect(')');
            }
            if (parser.accept('id', 'private')) {
                parser.expect('(');
                while (!parser.match(')')) {
                    parser.expect('%');
                    parser.expect(':');
                    parser.parseType();
                    if (!parser.accept(',')) {
                        break;
                    }
                }
                parser.expect(')');
            }
            if (parser.match('{')) {
                const region = {};
                parser.parseRegion(region);
                op.regions.push(region);
            }
            parser.parseOptionalAttrDict(op.attributes);
            return true;
        }
        return super.parseOperation(parser, opName, op);
    }

    _parseSizeAssignment(parser, op) {
        // Parse: (%id, %id, %id) or (%id = %val, %id = %val, %id = %val)
        parser.expect('(');
        while (!parser.match(')')) {
            if (parser.match('%')) {
                parser.expect('%');
                if (parser.accept('=')) {
                    const operand = {};
                    operand.value = parser.expect('%');
                    op.operands.push(operand);
                }
                if (!parser.accept(',')) {
                    break;
                }
            } else {
                break;
            }
        }
        parser.expect(')');
    }

    _parseLaunchFuncOperands(parser /*, args */) {
        if (parser.match('id', 'args')) {
            parser.expect();
            parser.expect('(');
            while (!parser.match(')')) {
                parser.expect('%');
                parser.expect(':');
                parser.parseType();
                if (!parser.match(')')) {
                    parser.expect(',');
                }
            }
            parser.expect(')');
        }
    }

    _parseAsyncDependencies(parser /*, args */) {
        if (parser.match('[')) {
            parser.skip('[', ']');
        }
    }
};

mlir.ArmSMEDialect = class extends mlir.Dialect {

    constructor(operations) {
        super('arm_sme', operations);
    }
};

mlir.ArmNeonDialect = class extends mlir.Dialect {

    constructor(operations) {
        super('arm_neon', operations);
    }
};

mlir.ArmSVEDialect = class extends mlir.Dialect {

    constructor(operations) {
        super('arm_sve', operations);
    }
};

mlir.NVVMDialect = class extends mlir.Dialect {

    constructor(operations) {
        super('nvvm', operations);
    }
};

mlir.OpenMPDialect = class extends mlir.Dialect {

    constructor(operations) {
        super('omp', operations);
    }
};

mlir.LLVMDialect = class extends mlir.Dialect {

    constructor(operations) {
        super('llvm', operations);
    }

    parseType(parser, dialectType) {
        // Handle LLVM pointer types: !llvm.ptr (opaque) or !llvm.ptr<T> (typed, legacy)
        if (dialectType === '!llvm.ptr') {
            if (parser.match('<')) {
                // Typed pointer: !llvm.ptr<i8>, !llvm.ptr<f32>, etc.
                const content = parser.skip('<', '>');
                return dialectType + content;
            }
            // Opaque pointer: !llvm.ptr
            return dialectType;
        }
        // Handle other LLVM types: !llvm.struct, !llvm.array, !llvm.vec, etc.
        if (dialectType.startsWith('!llvm.')) {
            if (parser.match('<')) {
                const content = parser.skip('<', '>');
                return dialectType + content;
            }
            return dialectType;
        }
        return null;
    }

    parseOperation(parser, opName, op) {
        if (opName === 'llvm.func') {
            return this._parseLLVMFuncOp(parser, op);
        }
        if (opName === 'llvm.mlir.global') {
            return this._parseLLVMGlobalOp(parser, op);
        }
        if (opName === 'llvm.getelementptr') {
            return this._parseLLVMGEPOp(parser, op);
        }
        if (opName === 'llvm.alloca') {
            return this._parseLLVMAllocaOp(parser, op);
        }
        if (opName === 'llvm.call') {
            return this._parseLLVMCallOp(parser, op);
        }
        if (opName === 'llvm.icmp' || opName === 'llvm.fcmp') {
            return this._parseLLVMCmpOp(parser, op);
        }
        if (opName.startsWith('llvm.intr.')) {
            return this._parseLLVMIntrinsicOp(parser, op);
        }
        return super.parseOperation(parser, opName, op);
    }

    _parseLLVMGlobalOp(parser, op) {
        const linkageKeywords = ['external', 'available_externally', 'linkonce', 'linkonce_odr', 'weak', 'weak_odr', 'appending', 'internal', 'private', 'extern_weak', 'common'];
        if (parser.match('id') && linkageKeywords.includes(parser._token.value)) {
            op.attributes.push({ name: 'linkage', value: parser.expect('id') });
        }
        const visibilityKeywords = ['default', 'hidden', 'protected'];
        if (parser.match('id') && visibilityKeywords.includes(parser._token.value)) {
            op.attributes.push({ name: 'visibility_', value: parser.expect('id') });
        }
        const unnamedAddrKeywords = ['unnamed_addr', 'local_unnamed_addr'];
        if (parser.match('id') && unnamedAddrKeywords.includes(parser._token.value)) {
            op.attributes.push({ name: 'unnamed_addr', value: parser.expect('id') });
        }
        if (parser.accept('id', 'thread_local')) {
            op.attributes.push({ name: 'thread_local_', value: true });
        }
        if (parser.accept('id', 'constant')) {
            op.attributes.push({ name: 'constant', value: true });
        }
        if (parser.match('@')) {
            const symbol = parser.expect('@');
            op.attributes.push({ name: 'sym_name', value: symbol });
        }
        parser.expect('(');
        if (!parser.match(')')) {
            const value = parser.parseAttributeValue();
            if (parser.accept(':')) {
                parser.parseType();
            }
            op.attributes.push({ name: 'value', value });
        }
        parser.expect(')');
        if (parser.accept('id', 'comdat')) {
            parser.expect('(');
            const comdat = parser.expect('@');
            parser.expect(')');
            op.attributes.push({ name: 'comdat', value: comdat });
        }
        if (parser.match('{')) {
            parser.parseAttributeDict(op.attributes);
        }
        if (parser.accept(':')) {
            const type = parser.parseType();
            op.results = [{ type }];
        }
        if (parser.match('{')) {
            const region = {};
            parser.parseRegion(region);
            op.regions.push(region);
        }
        return true;
    }

    _parseLLVMGEPOp(parser, op) {
        const noWrapFlagsKeywords = ['inbounds', 'nusw', 'nuw'];
        const noWrapFlags = [];
        while (parser.match('id') && noWrapFlagsKeywords.includes(parser._token.value)) {
            noWrapFlags.push(parser.expect('id'));
        }
        if (noWrapFlags.length > 0) {
            op.attributes.push({ name: 'noWrapFlags', value: noWrapFlags.join(' ') });
        }
        const base = parser.parseValue();
        op.operands.push(base);
        parser.expect('[');
        const rawConstantIndices = [];
        while (!parser.match(']')) {
            if (parser.match('int')) {
                const constIndex = parser.expect('int');
                rawConstantIndices.push(constIndex);
            } else {
                const operand = parser.parseValue();
                op.operands.push(operand);
                rawConstantIndices.push(-2147483648);
            }
            parser.accept(',');
        }
        parser.expect(']');
        if (rawConstantIndices.length > 0) {
            op.attributes.push({ name: 'rawConstantIndices', value: rawConstantIndices });
        }
        if (parser.match('{')) {
            parser.parseAttributeDict(op.attributes);
        }
        parser.expect(':');
        parser.parseArgumentTypes(op.operands);
        if (parser.accept('->')) {
            const resultType = parser.parseType();
            op.results = [{ type: resultType }];
        }
        parser.expect(',');
        const elemType = parser.parseType();
        op.attributes.push({ name: 'elem_type', value: elemType });
        return true;
    }

    _parseLLVMAllocaOp(parser, op) {
        if (parser.accept('id', 'inalloca')) {
            op.attributes.push({ name: 'inalloca', value: true });
        }
        const arraySize = parser.parseValue();
        op.operands.push(arraySize);
        parser.expect('id', 'x');
        const elemType = parser.parseType();
        op.attributes.push({ name: 'elem_type', value: elemType });
        if (parser.match('{')) {
            parser.parseAttributeDict(op.attributes);
        }
        parser.expect(':');
        parser.parseArgumentTypes(op.operands);
        if (parser.accept('->')) {
            const resultType = parser.parseType();
            op.results = [{ type: resultType }];
        }
        return true;
    }

    _parseLLVMCallOp(parser, op) {
        const cconvKeywords = ['ccc', 'fastcc', 'coldcc', 'cc', 'webkit_jscc', 'anyregcc', 'preserve_mostcc', 'preserve_allcc', 'preserve_nonecc', 'cxx_fast_tlscc', 'tailcc', 'swiftcc', 'swifttailcc', 'cfguard_checkcc', 'ghccc'];
        if (parser.match('id') && cconvKeywords.includes(parser._token.value)) {
            op.attributes.push({ name: 'CConv', value: parser.expect('id') });
        }
        const tailcallKeywords = ['tail', 'musttail', 'notail'];
        if (parser.match('id') && tailcallKeywords.includes(parser._token.value)) {
            op.attributes.push({ name: 'tailcallkind', value: parser.expect('id') });
        }
        let isDirect = false;
        if (parser.match('@')) {
            const callee = parser.expect('@');
            op.attributes.push({ name: 'callee', value: callee });
            isDirect = true;
        } else if (parser.match('%')) {
            const calleePtr = parser.parseValue();
            op.operands.push(calleePtr);
        }
        parser.expect('(');
        while (!parser.match(')')) {
            const arg = parser.parseValue();
            op.operands.push(arg);
            parser.accept(',');
        }
        parser.expect(')');
        if (parser.accept('id', 'vararg')) {
            parser.expect('(');
            const varCalleeType = parser.parseType();
            op.attributes.push({ name: 'var_callee_type', value: varCalleeType });
            parser.expect(')');
        }
        if (parser.match('{')) {
            parser.parseAttributeDict(op.attributes);
        }
        parser.expect(':');
        if (!isDirect) {
            parser.parseType();
            parser.expect(',');
        }
        parser.parseArgumentTypes(op.operands);
        if (parser.accept('->')) {
            const resultType = parser.parseType();
            op.results = [{ type: resultType }];
        }
        return true;
    }

    _parseLLVMCmpOp(parser, op) {
        const predicate = parser.expect('string');
        op.attributes.push({ name: 'predicate', value: predicate });
        const lhs = parser.parseValue();
        op.operands.push(lhs);
        parser.expect(',');
        const rhs = parser.parseValue();
        op.operands.push(rhs);
        if (parser.match('{')) {
            parser.parseAttributeDict(op.attributes);
        }
        parser.expect(':');
        const type = parser.parseType();
        for (const operand of op.operands) {
            operand.type = type;
        }
        return true;
    }

    _parseLLVMIntrinsicOp(parser, op) {
        parser.expect('(');
        while (!parser.match(')')) {
            const operand = parser.parseValue();
            op.operands.push(operand);
            parser.accept(',');
        }
        parser.expect(')');
        if (parser.match('{')) {
            parser.parseAttributeDict(op.attributes);
        }
        parser.expect(':');
        parser.parseArgumentTypes(op.operands);
        if (parser.accept('->')) {
            const resultType = parser.parseType();
            op.results = [{ type: resultType }];
        }
        return true;
    }

    _parseLLVMFuncOp(parser, op) {
        const linkageKeywords = ['external', 'available_externally', 'linkonce', 'linkonce_odr', 'weak', 'weak_odr', 'appending', 'internal', 'private', 'extern_weak', 'common'];
        if (parser.match('id') && linkageKeywords.includes(parser._token.value)) {
            op.attributes.push({ name: 'linkage', value: parser.expect('id') });
        }
        const visibilityKeywords = ['default', 'hidden', 'protected'];
        if (parser.match('id') && visibilityKeywords.includes(parser._token.value)) {
            op.attributes.push({ name: 'visibility_', value: parser.expect('id') });
        }
        const unnamedAddrKeywords = ['unnamed_addr', 'local_unnamed_addr'];
        if (parser.match('id') && unnamedAddrKeywords.includes(parser._token.value)) {
            op.attributes.push({ name: 'unnamed_addr', value: parser.expect('id') });
        }
        const cconvKeywords = ['ccc', 'fastcc', 'coldcc', 'cc', 'webkit_jscc', 'anyregcc', 'preserve_mostcc', 'preserve_allcc', 'preserve_nonecc', 'cxx_fast_tlscc', 'tailcc', 'swiftcc', 'swifttailcc', 'cfguard_checkcc', 'ghccc', 'arm_apcscc', 'arm_aapcscc', 'arm_aapcs_vfpcc', 'aarch64_vector_pcs', 'aarch64_sve_vector_pcs', 'aarch64_sme_preservemost_from_x0', 'aarch64_sme_preservemost_from_x2', 'msp430_intrcc', 'avr_intrcc', 'avr_signalcc', 'ptx_kernel', 'ptx_device', 'spir_func', 'spir_kernel', 'intel_ocl_bicc', 'x86_64_sysvcc', 'win64cc', 'x86_fastcallcc', 'x86_stdcallcc', 'x86_thiscallcc', 'x86_vectorcallcc', 'x86_intrcc', 'amdgpu_vs', 'amdgpu_gs', 'amdgpu_ps', 'amdgpu_cs', 'amdgpu_kernel', 'x86_regcallcc', 'amdgpu_hs', 'msp430_builtincc', 'amdgpu_ls', 'amdgpu_es', 'aarch64_vfpcc', 'aarch64_sve_vfpcc', 'wasm_emscripten_invokecc', 'amdgpu_gfx', 'm68k_intrcc'];
        if (parser.match('id') && cconvKeywords.includes(parser._token.value)) {
            op.attributes.push({ name: 'CConv', value: parser.expect('id') });
        }
        parser.parseSymbolName('sym_name', op.attributes);
        const type = {};
        type.inputs = parser.parseFunctionArgumentList();
        if (parser.accept('->')) {
            type.results = parser.parseFunctionResultList();
        } else {
            type.results = [];
        }
        op.attributes.push({ name: 'function_type', value: type });
        if (parser.accept('id', 'vscale_range')) {
            parser.expect('(');
            const minRange = parser.expect();
            parser.expect(',');
            const maxRange = parser.expect();
            parser.expect(')');
            op.attributes.push({ name: 'vscale_range', value: `(${minRange}, ${maxRange})` });
        }
        if (parser.accept('id', 'comdat')) {
            parser.expect('(');
            const comdat = parser.expect('@');
            parser.expect(')');
            op.attributes.push({ name: 'comdat', value: comdat });
        }
        parser.parseOptionalAttrDictWithKeyword(op.attributes);
        if (parser.match('{')) {
            const region = {};
            parser.parseRegion(region);
            op.regions.push(region);
        }
        return true;
    }
};

mlir.StdxDialect = class extends mlir.Dialect {

    constructor(operations) {
        super('stdx', operations);
    }
};

mlir.VMDialect = class extends mlir.Dialect {

    constructor(operations) {
        super('vm', operations);
    }

    parseOperation(parser, opName, op) {
        if (opName === 'vm.func') {
            parser.parseOptionalVisibilityKeyword(op.attributes);
            parser.parseSymbolName('sym_name', op.attributes);
            const type = {};
            type.inputs = parser.parseFunctionArgumentList();
            parser.parseOptionalAttrDictWithKeyword(op.attributes);
            type.results = [];
            if (parser.accept('->')) {
                for (const result of parser.parseFunctionResultList()) {
                    type.results.push(result);
                }
            }
            op.attributes.push({ name: 'function_type', value: type });
            parser.parseOptionalAttrDictWithKeyword(op.attributes);
            if (parser.match('{')) {
                const region = {};
                parser.parseRegion(region);
                op.regions.push(region);
            }
            return true;
        }
        return super.parseOperation(parser, opName, op);
    }
};

mlir.MathDialect = class extends mlir.Dialect {

    constructor(operations) {
        super('math', operations);
    }
};

mlir.TMTensorDialect = class extends mlir.Dialect {

    constructor(operations) {
        super('tm_tensor', operations);
    }
};

mlir.MLProgramDialect = class extends mlir.Dialect {

    constructor(operations) {
        super('ml_program', operations);
    }

    parseOperation(parser, opName, op) {
        if (opName === 'ml_program.global') {
            if (parser.match('id', 'private') || parser.match('id', 'public') || parser.match('id', 'nested')) {
                parser.expect('id');
            }
            if (parser.match('id', 'mutable')) {
                const mutable = parser.expect('id');
                op.attributes.push({ name: 'is_mutable', value: mutable });
            }
            parser.parseSymbolName('sym_name', op.attributes);
            if (parser.accept('(')) {
                const initialValue = parser.parseAttributeValue();
                op.attributes.push({ name: 'value', value: initialValue });
                parser.accept(')');
            }
            if (parser.accept(':')) {
                const type = parser.parseType();
                op.attributes.push({ name: 'type', value: type });
            }
            return true;
        }
        return super.parseOperation(parser, opName, op);
    }
};

mlir.IREEGPUDialect = class extends mlir.Dialect {

    constructor(operations) {
        super('iree_gpu', operations);
    }
};

mlir.TFDeviceDialect = class extends mlir.Dialect {

    constructor(operations) {
        super('tf_device', operations);
    }
};

mlir.TFExecutorDialect = class extends mlir.Dialect {

    constructor(operations) {
        super('tf_executor', operations);
    }

    parseType(parser, dialectType) {
        if (dialectType === '!tf_executor.control' || dialectType === '!tf_executor.token') {
            return dialectType;
        }
        return null;
    }

    parseOperation(parser, opName, op) {
        if (opName === 'tf_executor.graph') {
            return this._parseGraphOp(parser, op);
        }
        if (opName === 'tf_executor.island') {
            return this._parseIslandOp(parser, op);
        }
        return super.parseOperation(parser, opName, op);
    }

    _parseGraphOp(parser, op) {
        if (parser.match('{')) {
            const region = {};
            parser.parseRegion(region);
            op.regions.push(region);
            if (region.blocks && region.blocks.length > 0) {
                const [block] = region.blocks;
                if (block.operations && block.operations.length > 0) {
                    const lastOp = block.operations[block.operations.length - 1];
                    if (lastOp.name === 'tf_executor.fetch' && lastOp.operands) {
                        for (const operand of lastOp.operands) {
                            if (operand.type && operand.type !== '!tf_executor.control') {
                                op.results.push({ type: operand.type });
                            }
                        }
                    }
                }
            }
        }
        parser.parseOptionalAttrDict(op.attributes);
        return true;
    }

    _parseIslandOp(parser, op) {
        // Parse: tf_executor.island wraps "tf.SomeOp"(...) {...} : (...) -> (...)
        // or: tf_executor.island {...}
        // or: tf_executor.island(%control_inputs) {...}
        if (parser.match('(')) {
            op.operands = parser.parseArguments();
        }
        if (parser.accept('id', 'wraps')) {
            // Parse the wrapped operation
            const wrappedOp = parser.parseGenericOperation();
            op.attributes.push({ name: 'wrappedOp', value: wrappedOp });
            // Note: The island's results are already set by parseOperation from the LHS
            // We just need to ensure there's a control token type for the last result
            // The wrapped operation's result types should match the island's first N-1 results
        } else if (parser.match('{')) {
            // Parse region-based island
            const region = {};
            parser.parseRegion(region);
            op.regions.push(region);
        }
        parser.parseOptionalAttrDict(op.attributes);
        return true;
    }
};

mlir.TFFrameworkDialect = class extends mlir.Dialect {

    constructor(operations) {
        super('tf_framework', operations);
    }
};

mlir.TFRDialect = class extends mlir.Dialect {

    constructor(operations) {
        super('tfr', operations);
    }
};

mlir.TFRTDialect = class extends mlir.Dialect {

    constructor(operations) {
        super('tfrt', operations);
    }

    parseType(parser, dialectType) {
        const simpleTypes = ['chain', 'string', 'dist_context'];
        for (const simpleType of simpleTypes) {
            if (dialectType === `!tfrt.${simpleType}`) {
                return dialectType;
            }
        }
        if (dialectType.startsWith('!tfrt.tensor')) {
            if (parser.match('<')) {
                const content = parser.skip('<', '>');
                return dialectType + content;
            }
            return dialectType;
        }
        return null;
    }
};

mlir.TileDialect = class extends mlir.Dialect {

    constructor(operations) {
        super('tile', operations);
    }

    parseOperation(parser, opName, op) {
        // tile.contract has format: tile.contract agg, combo, operands... attributes : types -> result
        // Example: %1 = tile.contract add, mul, %0, %arg0, %arg1 {sink = #map0, srcs = [#map1, #map2]} : tensor<f32>, tensor<1x256xf32>, tensor<256x512xf32> -> tensor<1x512xf32>
        if (opName === 'tile.contract') {
            // Parse aggregation kind (add, mul, etc.)
            if (parser.match('id')) {
                const agg = parser.expect('id');
                op.attributes.push({ name: 'agg', value: agg });
            }
            parser.accept(',');
            // Parse combination kind (add, mul, etc.)
            if (parser.match('id')) {
                const combo = parser.expect('id');
                op.attributes.push({ name: 'combo', value: combo });
            }
            parser.accept(',');
            // Parse operands
            op.operands = parser.parseArguments();
            // Parse attributes
            if (parser.match('{')) {
                parser.parseAttributeDict(op.attributes);
            }
            // Parse types
            if (parser.accept(':')) {
                parser.parseArgumentTypes(op.operands);
            }
            if (parser.accept('->')) {
                parser.parseArgumentTypes(op.results);
            }
            return true;
        }
        return super.parseOperation(parser, opName, op);
    }
};

mlir.ToyDialect = class extends mlir.Dialect {

    constructor(operations) {
        super('toy', operations);
    }

    parseOperation(parser, opName, op) {
        if (opName === 'toy.func') {
            return this._parseFuncOp(parser, op);
        }
        if (opName === 'toy.constant') {
            const value = parser.parseValue();
            op.attributes.push({ name: 'value', value: value.value === undefined ? value : value.value });
            if (parser.accept(':')) {
                parser.parseArgumentTypes(op.results);
            }
            return true;
        }
        if (opName === 'toy.mul' || opName === 'toy.add') {
            op.operands = parser.parseArguments();
            parser.parseOptionalAttrDictWithKeyword(op.attributes);
            if (parser.accept(':')) {
                // The type signature can be either:
                // 1. Simple: `: tensor<*xf64>` - just the result type
                // 2. Function-style: `: (tensor<*xf64>, tensor<*xf64>) -> tensor<*xf64>`
                const type = parser.parseType();
                if (type.startsWith('(') && type.includes('->')) {
                    // Function-style type signature
                    const parts = type.match(/\((.*?)\)\s*->\s*(.+)/);
                    if (parts) {
                        const inputTypes = parts[1].split(',').map((t) => t.trim());
                        for (let i = 0; i < op.operands.length && i < inputTypes.length; i++) {
                            op.operands[i].type = inputTypes[i];
                        }
                        // Add result type to existing result (which already has SSA opName)
                        if (op.results.length > 0) {
                            op.results[0].type = parts[2].trim();
                        }
                    }
                } else {
                    // Simple type signature - type applies to both operands and result
                    for (const operand of op.operands) {
                        operand.type = type;
                    }
                    if (op.results.length > 0) {
                        op.results[0].type = type;
                    }
                }
            }
            return true;
        }
        return super.parseOperation(parser, opName, op);
    }

    _parseFuncOp(parser, op) {
        parser.parseOptionalVisibilityKeyword(op.attributes);
        parser.parseSymbolName('sym_name', op.attributes);
        const type = {};
        type.inputs = parser.parseFunctionArgumentList();
        parser.parseOptionalAttrDictWithKeyword(op.attributes);
        type.results = [];
        if (parser.accept('->')) {
            for (const result of parser.parseFunctionResultList()) {
                type.results.push(result);
            }
        }
        op.attributes.push({ name: 'function_type', value: type });
        parser.parseOptionalAttrDictWithKeyword(op.attributes);
        if (parser.match('{')) {
            const region = {};
            parser.parseRegion(region);
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
        return true;
    }
};

mlir.SdfgDialect = class extends mlir.Dialect {

    constructor(operations) {
        super('sdfg', operations);
    }

    parseType(parser, dialectType) {
        if (dialectType === '!sdfg.array' || dialectType === '!sdfg.stream' ||
            dialectType === '!sdir.array' || dialectType === '!sdir.stream' || dialectType === '!sdir.stream_array') {
            if (parser.match('<')) {
                const content = parser.skip('<', '>');
                return dialectType + content;
            }
            return dialectType;
        }
        if (dialectType.startsWith('!sdfg.array<') || dialectType.startsWith('!sdfg.stream<') ||
            dialectType.startsWith('!sdir.array<') || dialectType.startsWith('!sdir.stream<') || dialectType.startsWith('!sdir.stream_array<')) {
            return dialectType;
        }
        return null;
    }

    parseOperation(parser, opName, op) {
        if (opName === 'sdfg.sdfg' || opName === 'sdfg.nested_sdfg' || opName === 'sdir.sdfg') {
            parser.parseOptionalAttrDict(op.attributes);
            const type = {};
            type.inputs = parser.parseFunctionArgumentList();
            parser.parseOptionalAttrDictWithKeyword(op.attributes);
            type.results = [];
            if (parser.accept('->')) {
                type.results = parser.parseFunctionArgumentList();
            }
            op.attributes.push({ name: 'function_type', value: type });
            parser.parseOptionalAttrDictWithKeyword(op.attributes);
            if (parser.match('{')) {
                const region = {};
                parser.parseRegion(region);
                op.regions.push(region);
            }
            return true;
        }
        if (opName === 'sdfg.tasklet' || opName === 'sdir.tasklet') {
            parser.parseOptionalAttrDict(op.attributes);
            op.sym_name = parser.parseOptionalSymbolName();
            if (parser.match('(')) {
                op.operands = parser.parseArguments();
            }
            parser.parseOptionalAttrDictWithKeyword(op.attributes);
            if (parser.accept('->')) {
                if (parser.accept('(')) {
                    while (!parser.accept(')')) {
                        const type = parser.parseType();
                        op.results.push({ type });
                        parser.accept(',');
                    }
                } else {
                    const type = parser.parseType();
                    op.results.push({ type });
                }
            }
            parser.parseOptionalAttrDictWithKeyword(op.attributes);
            if (parser.match('{')) {
                const region = {};
                parser.parseRegion(region);
                op.regions.push(region);
            }
            return true;
        }
        if (opName === 'sdfg.consume') {
            if (parser.match('(')) {
                op.operands = parser.parseArguments();
            }
            parser.parseOptionalAttrDictWithKeyword(op.attributes);
            if (parser.accept('->')) {
                if (parser.accept('(')) {
                    while (!parser.accept(')')) {
                        const type = parser.parseType();
                        op.results.push({ type });
                        parser.accept(',');
                    }
                } else {
                    const type = parser.parseType();
                    op.results.push({ type });
                }
            }
            parser.parseOptionalAttrDictWithKeyword(op.attributes);
            if (parser.match('{')) {
                const region = {};
                parser.parseRegion(region);
                op.regions.push(region);
            }
            return true;
        }
        if (opName === 'sdfg.state' || opName === 'sdir.state') {
            parser.parseOptionalAttrDict(op.attributes);
            op.sym_name = parser.parseOptionalSymbolName();
            parser.parseOptionalAttrDictWithKeyword(op.attributes);
            const region = {};
            parser.parseRegion(region);
            op.regions.push(region);
            return true;
        }
        if (opName === 'sdfg.alloc' || opName === 'sdir.alloc' || opName === 'sdir.alloc_transient' || opName === 'sdir.alloc_stream') {
            parser.parseOptionalAttrDict(op.attributes);
            if (parser.match('(')) {
                op.operands = parser.parseArguments();
            }
            parser.parseOptionalAttrDictWithKeyword(op.attributes);
            if (parser.accept(':')) {
                const type = parser.parseType();
                op.results.push({ type });
            }
            return true;
        }
        if (opName === 'sdfg.store' || opName === 'sdir.store') {
            parser.parseOptionalAttrDict(op.attributes);
            const value = parser.expect('%');
            op.operands.push({ value });
            parser.accept(',');
            const array = parser.expect('%');
            op.operands.push({ value: array });
            if (parser.accept('[')) {
                while (!parser.match(']')) {
                    if (parser.match('%')) {
                        const idx = parser.expect('%');
                        op.operands.push({ value: idx });
                    } else {
                        parser.expect();
                    }
                    if (parser.match(',')) {
                        parser.accept(',');
                    }
                }
                parser.accept(']');
            }
            parser.parseOptionalAttrDictWithKeyword(op.attributes);
            if (parser.accept(':')) {
                const valueType = parser.parseType();
                parser.accept('->');
                const arrayType = parser.parseType();
                if (op.operands.length > 0) {
                    op.operands[0].type = valueType;
                }
                if (op.operands.length > 1) {
                    op.operands[1].type = arrayType;
                }
            }
            return true;
        }
        if (opName === 'sdfg.load' || opName === 'sdir.load') {
            parser.parseOptionalAttrDict(op.attributes);
            const array = parser.expect('%');
            op.operands.push({ value: array });
            if (parser.accept('[')) {
                while (!parser.match(']')) {
                    if (parser.match('%')) {
                        const idx = parser.expect('%');
                        op.operands.push({ value: idx });
                    } else {
                        parser.expect();
                    }
                    if (parser.match(',')) {
                        parser.accept(',');
                    }
                }
                parser.accept(']');
            }
            parser.parseOptionalAttrDictWithKeyword(op.attributes);
            if (parser.accept(':')) {
                const arrayType = parser.parseType();
                parser.accept('->');
                const resultType = parser.parseType();
                if (op.operands.length > 0) {
                    op.operands[0].type = arrayType;
                }
                op.results.push({ type: resultType });
            }
            return true;
        }
        if (opName === 'sdfg.map' || opName === 'sdir.map') {
            parser.parseOptionalAttrDict(op.attributes);
            const params = [];
            if (parser.accept('(')) {
                while (!parser.accept(')')) {
                    if (parser.match('%')) {
                        const param = parser.expect('%');
                        params.push(param);
                    }
                    if (parser.match(',')) {
                        parser.accept(',');
                    }
                }
            }
            if (parser.accept('=')) {
                if (parser.accept('(')) {
                    while (!parser.accept(')')) {
                        parser.expect();
                    }
                }
            }
            if (parser.match('id', 'to')) {
                parser.accept('id', 'to');
                if (parser.accept('(')) {
                    while (!parser.accept(')')) {
                        parser.expect();
                    }
                }
            }
            if (parser.match('id', 'step')) {
                parser.accept('id', 'step');
                if (parser.accept('(')) {
                    while (!parser.accept(')')) {
                        parser.expect();
                    }
                }
            }
            parser.parseOptionalAttrDictWithKeyword(op.attributes);
            if (parser.match('{')) {
                const region = {};
                parser.parseRegion(region);
                op.regions.push(region);
            }
            return true;
        }
        if (opName === 'sdfg.consume' || opName === 'sdir.consume') {
            parser.parseOptionalAttrDict(op.attributes);
            if (parser.accept('(')) {
                op.operands = parser.parseArguments();
            }
            if (parser.accept('->')) {
                if (parser.accept('(')) {
                    while (!parser.accept(')')) {
                        parser.expect();
                    }
                }
            }
            parser.parseOptionalAttrDictWithKeyword(op.attributes);
            if (parser.match('{')) {
                const region = {};
                parser.parseRegion(region);
                op.regions.push(region);
            }
            return true;
        }
        if (opName === 'sdfg.edge' || opName === 'sdir.edge') {
            parser.parseOptionalAttrDict(op.attributes);
            if (parser.match('(')) {
                op.operands = parser.parseArguments();
            }
            parser.parseOptionalAttrDictWithKeyword(op.attributes);
            if (parser.match('@')) {
                const src = parser.expect('@');
                op.attributes.push({ name: 'src', value: src });
            }
            parser.accept('->');
            if (parser.match('@')) {
                const dst = parser.expect('@');
                op.attributes.push({ name: 'dst', value: dst });
            }
            return true;
        }
        if (opName === 'sdfg.sym' || opName === 'sdir.sym') {
            if (parser.accept('(')) {
                const expr = parser.expect('string');
                op.attributes.push({ name: 'expr', value: expr });
                parser.accept(')');
            }
            parser.parseOptionalAttrDictWithKeyword(op.attributes);
            if (parser.accept(':')) {
                const type = parser.parseType();
                op.results.push({ type });
            }
            return true;
        }
        if (opName === 'sdfg.copy' || opName === 'sdir.copy') {
            parser.parseOptionalAttrDict(op.attributes);
            op.operands = parser.parseArguments();
            if (parser.accept('->')) {
                const dst = parser.parseArguments();
                op.operands.push(...dst);
            }
            if (parser.accept(':')) {
                const type = parser.parseType();
                for (const operand of op.operands) {
                    if (!operand.type) {
                        operand.type = type;
                    }
                }
            }
            return true;
        }
        if (opName === 'sdfg.libcall' || opName === 'sdir.libcall') {
            parser.parseOptionalAttrDictWithKeyword(op.attributes);
            if (parser.match('string')) {
                const libname = parser.expect('string');
                op.attributes.push({ name: 'libname', value: libname });
            }
            op.operands = parser.parseArguments();
            if (parser.accept(':')) {
                if (parser.accept('(')) {
                    while (!parser.accept(')')) {
                        parser.parseType();
                        parser.accept(',');
                    }
                }
                if (parser.accept('->')) {
                    const resultType = parser.parseType();
                    op.results.push({ type: resultType });
                }
            }
            return true;
        }
        if (opName === 'sdir.get_access') {
            if (parser.match('%')) {
                const value = parser.expect();
                op.operands.push({ value });
            }
            return true;
        }
        if (opName === 'sdir.call') {
            const callee = parser.parseOptionalSymbolName();
            if (callee) {
                op.attributes.push({ name: 'callee', value: callee });
            }
            if (parser.match('(')) {
                op.operands = parser.parseArguments();
            }
            if (parser.accept(':')) {
                if (parser.accept('(')) {
                    while (!parser.accept(')')) {
                        parser.parseType();
                        parser.accept(',');
                    }
                }
                if (parser.accept('->')) {
                    const resultType = parser.parseType();
                    op.results.push({ type: resultType });
                }
            }
            return true;
        }
        if (opName === 'sdfg.alloc_symbol' || opName === 'sdir.alloc_symbol') {
            if (parser.accept('(')) {
                const sym = parser.expect('string');
                op.attributes.push({ name: 'sym', value: sym });
                parser.accept(')');
            }
            return true;
        }
        if (opName === 'sdfg.return') {
            if (parser.match('%')) {
                op.operands = parser.parseArguments();
                if (parser.accept(':')) {
                    parser.parseArgumentTypes(op.operands);
                }
            }
            return true;
        }
        if (opName === 'sdfg.stream_push' || opName === 'sdir.stream_push') {
            parser.parseOptionalAttrDict(op.attributes);
            const value = parser.expect('%');
            op.operands.push({ value });
            parser.accept(',');
            const stream = parser.expect('%');
            op.operands.push({ value: stream });
            parser.parseOptionalAttrDictWithKeyword(op.attributes);
            if (parser.accept(':')) {
                const valueType = parser.parseType();
                parser.accept('->');
                const streamType = parser.parseType();
                if (op.operands.length > 0) {
                    op.operands[0].type = valueType;
                }
                if (op.operands.length > 1) {
                    op.operands[1].type = streamType;
                }
            }
            return true;
        }
        if (opName === 'sdfg.stream_pop' || opName === 'sdir.stream_pop') {
            parser.parseOptionalAttrDict(op.attributes);
            const stream = parser.expect('%');
            op.operands.push({ value: stream });
            parser.parseOptionalAttrDictWithKeyword(op.attributes);
            if (parser.accept(':')) {
                const streamType = parser.parseType();
                parser.accept('->');
                const resultType = parser.parseType();
                if (op.operands.length > 0) {
                    op.operands[0].type = streamType;
                }
                op.results.push({ type: resultType });
            }
            return true;
        }
        if (opName === 'sdfg.stream_length' || opName === 'sdir.stream_length') {
            parser.parseOptionalAttrDict(op.attributes);
            const stream = parser.expect('%');
            op.operands.push({ value: stream });
            parser.parseOptionalAttrDictWithKeyword(op.attributes);
            if (parser.accept(':')) {
                const streamType = parser.parseType();
                parser.accept('->');
                const resultType = parser.parseType();
                if (op.operands.length > 0) {
                    op.operands[0].type = streamType;
                }
                op.results.push({ type: resultType });
            }
            return true;
        }
        if (opName === 'sdfg.view_cast' || opName === 'sdir.view_cast') {
            parser.parseOptionalAttrDict(op.attributes);
            const input = parser.expect('%');
            op.operands.push({ value: input });
            parser.parseOptionalAttrDictWithKeyword(op.attributes);
            if (parser.accept(':')) {
                const inputType = parser.parseType();
                parser.accept('->');
                const resultType = parser.parseType();
                if (op.operands.length > 0) {
                    op.operands[0].type = inputType;
                }
                op.results.push({ type: resultType });
            }
            return true;
        }
        if (opName === 'sdfg.subview' || opName === 'sdir.subview') {
            parser.parseOptionalAttrDict(op.attributes);
            const input = parser.expect('%');
            op.operands.push({ value: input });
            while (parser.accept('[')) {
                while (!parser.accept(']')) {
                    parser.expect();
                }
            }
            parser.parseOptionalAttrDictWithKeyword(op.attributes);
            if (parser.accept(':')) {
                const inputType = parser.parseType();
                parser.accept('->');
                const resultType = parser.parseType();
                if (op.operands.length > 0) {
                    op.operands[0].type = inputType;
                }
                op.results.push({ type: resultType });
            }
            return true;
        }
        return super.parseOperation(parser, opName, op);
    }
};

mlir.TFLDialect = class extends mlir.Dialect {

    constructor(operations) {
        super('tfl', operations);
        // Operations that use parseOneResultSameOperandTypeOp in tfl_ops.cc
        // Format: operands attr-dict : single-type
        this._binaryOps = new Set([
            'add', 'sub', 'mul', 'div', 'floor_div', 'pow', 'squared_difference',
            'less', 'less_equal', 'greater', 'greater_equal', 'not_equal',
            'logical_and', 'logical_or'
        ]);
    }

    parseOperation(parser, opName, op) {
        const opKind = opName.substring('tfl.'.length);
        if (this._binaryOps.has(opKind)) {
            // Parse: operands attr-dict : type
            op.operands = parser.parseArguments();
            parser.parseOptionalAttrDict(op.attributes);
            if (parser.accept(':')) {
                const type = parser.parseType();
                // All operands and result share the same type
                for (const operand of op.operands) {
                    operand.type = type;
                }
                if (op.results.length > 0) {
                    op.results[0].type = type;
                }
            }
            return true;
        }

        return super.parseOperation(parser, opName, op);
    }
};

mlir.TFDialect = class extends mlir.Dialect {

    constructor(operations) {
        super('tf', operations);
    }

    parseType(parser, dialectType) {
        if (dialectType === '!tf.resource') {
            if (parser.match('<')) {
                const content = parser.skip('<', '>');
                return dialectType + content;
            }
            return dialectType;
        }
        if (dialectType.startsWith('!tf.resource<')) {
            return dialectType;
        }
        if (dialectType === '!tf.variant') {
            if (parser.match('<')) {
                const content = parser.skip('<', '>');
                return dialectType + content;
            }
            return dialectType;
        }
        if (dialectType.startsWith('!tf.variant<')) {
            return dialectType;
        }
        const simpleTypes = ['string', 'control'];
        for (const simpleType of simpleTypes) {
            if (dialectType === `!tf.${simpleType}`) {
                return dialectType;
            }
        }
        return null;
    }
};

mlir.TFTypeDialect = class extends mlir.Dialect {

    constructor(operations) {
        super('tf_type', operations);
    }

    parseType(parser, dialectType) {
        const simpleTypes = [
            'string', 'resource', 'variant',
            'qint8', 'qint16', 'qint32', 'quint8', 'quint16',
            'f32ref', 'f64ref', 'uint4ref', 'int4ref', 'uint8ref', 'int8ref',
            'uint16ref', 'int16ref', 'uint32ref', 'int32ref', 'uint64ref', 'int64ref',
            'stringref', 'boolref', 'quint8ref', 'qint8ref', 'quint16ref', 'qint16ref',
            'qint32ref', 'bfloat16ref', 'complex64ref', 'complex128ref', 'halfref',
            'resourceref', 'variantref',
            'float8e4m3fnref', 'float8e5m2ref', 'float8e4m3fnuzref',
            'float8e4m3b11fnuzref', 'float8e5m2fnuzref'
        ];
        for (const simpleType of simpleTypes) {
            if (dialectType === `!tf_type.${simpleType}`) {
                if (!parser.match('<')) {
                    return dialectType;
                }
            }
        }
        if (dialectType.startsWith('!tf_type.resource') || dialectType.startsWith('!tf_type.variant')) {
            if (parser.accept('<')) {
                const subtypes = [];
                while (!parser.match('>')) {
                    subtypes.push(parser.parseType());
                    parser.accept(',');
                }
                parser.expect('>');
                return `${dialectType}<${subtypes.join(', ')}>`;
            }
            return dialectType;
        }
        return null;
    }
};

mlir.CheckDialect = class extends mlir.Dialect {
    constructor(operations) {
        super('check', operations);
    }

    parseOperation(parser, opName, op) {
        if (opName === 'check.expect_eq_const' || opName === 'check.expect_almost_eq_const' ||
            opName === 'check.expect_eq' || opName === 'check.expect_almost_eq' ||
            opName === 'check.expect_close' || opName === 'check.expect_serialized_eq') {
            op.operands = parser.parseArguments();
            if (parser.accept(',')) {
                const expectedValue = parser.parseValue();
                op.attributes.push({ name: 'expected', value: expectedValue.value });
            }
            if (parser.accept(':')) {
                parser.parseArgumentTypes(op.operands);
            }
            if (parser.match('{')) {
                parser.parseAttributeDict(op.attributes);
            }
            return true;
        }
        return super.parseOperation(parser, opName, op);
    }
};

mlir.TransformDialect = class extends mlir.Dialect {

    constructor(operations) {
        super('transform', operations);
    }

    parseType(parser, dialectType) {
        const simpleTypes = ['any_op', 'any_param', 'any_value', 'op', 'param', 'affine_map', 'type'];
        for (const simpleType of simpleTypes) {
            if (dialectType === `!transform.${simpleType}`) {
                return dialectType;
            }
            if (dialectType.startsWith(`!transform.${simpleType}<`)) {
                return dialectType;
            }
        }
        return null;
    }

    parseOperation(parser, opName, op) {
        if (opName === 'transform.named_sequence') {
            return this._parseNamedSequenceOp(parser, op);
        }
        if (opName.startsWith('transform.test.')) {
            const parsed = this._parseTestOp(parser, op);
            if (parsed === false) {
                const hasSchema = super.parseOperation(parser, opName, op);
                if (hasSchema) {
                    return true;
                }
                if (parser.match('%')) {
                    op.operands = parser.parseArguments();
                }
                parser.parseOptionalAttrDict(op.attributes);
                if (parser.accept(':')) {
                    if (parser.accept('(')) {
                        const types = [];
                        if (!parser.match(')')) {
                            do {
                                types.push(parser.parseType());
                            } while (parser.accept(','));
                        }
                        parser.expect(')');
                        for (let i = 0; i < op.operands.length && i < types.length; i++) {
                            op.operands[i].type = types[i];
                        }
                        if (parser.accept('->')) {
                            const resultType = parser.parseType();
                            if (op.results.length > 0) {
                                op.results[0].type = resultType;
                            }
                        }
                    }
                }
                return true;
            }
            return true;
        }
        const hasSchema = super.parseOperation(parser, opName, op);
        if (hasSchema) {
            return true;
        }
        parser.parseGenericOperationAfterOpName(op);
        return true;
    }

    _parseNamedSequenceOp(parser, op) {
        parser.parseOptionalVisibilityKeyword(op.attributes);
        parser.parseSymbolName('sym_name', op.attributes);
        const type = {};
        type.inputs = parser.parseFunctionArgumentList();
        if (parser.accept('->')) {
            type.results = parser.parseFunctionResultList();
        } else {
            type.results = [];
        }
        op.attributes.push({ name: 'function_type', value: type });
        parser.parseOptionalAttrDictWithKeyword(op.attributes);
        if (parser.match('{')) {
            const region = {};
            parser.parseRegion(region);
            op.regions.push(region);
        }
        return true;
    }

    _parseTestOp(parser, op) {
        const keywords = ['before', 'after', 'into', 'tile_sizes', 'mapping'];
        if (parser.match('%') || parser.match('keyword') || parser.match('eof')) {
            return false;
        }
        const hasSpecialSyntax = parser.match('(') || parser.match('{') || parser.match(':') || parser.match('[') || parser.match('=');
        let hasKeyword = false;
        if (parser.match('id')) {
            hasKeyword = keywords.includes(parser._token.value);
        }
        if (!hasSpecialSyntax && !hasKeyword) {
            return false;
        }
        let foundKeyword = false;
        while (parser.match('%') || (parser.match('id') && !foundKeyword)) {
            if (parser.match('%')) {
                const value = parser.expect('%');
                op.operands.push({ value });
                parser.accept(',');
            } else if (parser.match('id')) {
                const id = parser._token.value;
                if (keywords.includes(id)) {
                    const keyword = parser.expect('id');
                    op.attributes.push({ name: 'keyword', value: keyword });
                    foundKeyword = true;
                } else {
                    break;
                }
            }
        }
        if (parser.accept(':')) {
            if (parser.accept('(')) {
                let idx = 0;
                while (!parser.match(')')) {
                    const type = parser.parseType();
                    if (idx < op.operands.length) {
                        op.operands[idx].type = type;
                    }
                    idx++;
                    parser.accept(',');
                }
                parser.expect(')');
                // Parse remaining types after comma
                while (parser.accept(',')) {
                    const type = parser.parseType();
                    if (idx < op.operands.length) {
                        op.operands[idx].type = type;
                    }
                    idx++;
                }
            } else {
                // Standard type list
                parser.parseArgumentTypes(op.operands);
            }
        }
        parser.parseOptionalAttrDict(op.attributes);
        return true;
    }
};

mlir.TestDialect = class extends mlir.Dialect {

    constructor(operations) {
        super('test', operations);
    }

    parseOperation(parser, opName, op) {
        const hasSchema = super.parseOperation(parser, opName, op);
        if (hasSchema) {
            return true;
        }
        parser.parseGenericOperationAfterOpName(op);
        return true;
    }
};

mlir.TritonDialect = class extends mlir.Dialect {

    constructor(operations) {
        super('tt', operations);
    }

    parseType(parser, dialectType) {
        if (dialectType.startsWith('!tt.')) {
            if (parser.match('<')) {
                const content = parser.skip('<', '>');
                return dialectType + content;
            }
        }
        return dialectType;
    }

    parseOperation(parser, opName, op) {
        if (opName === 'tt.reshape') {
            this._operations.get(opName).hasParseOperation = false;
            op.operands = parser.parseArguments();
            if (parser.accept('id', 'allow_reorder')) {
                op.attributes.push({ name: 'allow_reorder', value: true });
            }
            if (parser.accept('id', 'efficient_layout')) {
                op.attributes.push({ name: 'efficient_layout', value: true });
            }
            if (parser.match('{')) {
                parser.parseAttributeDict(op.attributes);
            }
            if (parser.accept(':')) {
                parser.parseArgumentTypes(op.operands);
            }
            if (parser.accept('->')) {
                parser.parseArgumentTypes(op.results);
            }
            return true;
        }
        if (opName === 'tt.func') {
            parser.parseOptionalVisibilityKeyword(op.attributes);
            parser.parseSymbolName('sym_name', op.attributes);
            const type = {};
            type.inputs = parser.parseFunctionArgumentList();
            parser.parseOptionalAttrDictWithKeyword(op.attributes);
            type.results = [];
            if (parser.accept('->')) {
                for (const result of parser.parseFunctionResultList()) {
                    type.results.push(result);
                }
            }
            op.attributes.push({ name: 'function_type', value: type });
            parser.parseOptionalAttrDictWithKeyword(op.attributes);
            if (parser.match('{')) {
                const region = {};
                parser.parseRegion(region);
                op.regions.push(region);
            }
            if (op.regions.length > 0) {
                const region = op.regions[op.regions.length - 1];
                if (region.blocks.length > 0) {
                    const block = region.blocks[region.blocks.length - 1];
                    if (block.operations.length > 0) {
                        const lastOp = block.operations[block.operations.length - 1];
                        if (lastOp.name === 'tt.return') {
                            type.results = lastOp.operands;
                            block.operations.pop();
                        }
                    }
                }
            }
            return true;
        }
        return super.parseOperation(parser, opName, op);
    }
};

mlir.TritonGPUDialect = class extends mlir.Dialect {

    constructor(operations) {
        super('ttg', operations);
    }

    parseOperation(parser, opName, op) {
        if (opName === 'ttg.warp_specialize') {
            parser.expect('(');
            while (!parser.match(')')) {
                const operand = {};
                operand.name = parser.expect('%');
                op.operands.push(operand);
                if (!parser.match(')')) {
                    parser.expect(',');
                }
            }
            parser.expect(')');
            parser.parseOptionalAttrDictWithKeyword(op.attributes);
            parser.expect('id', 'default');
            const defaultRegion = {};
            parser.parseRegion(defaultRegion);
            op.regions.push(defaultRegion);
            const partitionNumWarps = [];
            let partitionIndex = 0;
            while (parser.match('id', `partition${partitionIndex}`)) {
                parser.expect('id', `partition${partitionIndex}`);
                const args = parser.parseFunctionArgumentList();
                parser.expect('id', 'num_warps');
                parser.expect('(');
                const numWarps = parser.expect();
                partitionNumWarps.push(parseInt(numWarps, 10));
                parser.expect(')');
                const partitionRegion = {};
                partitionRegion.arguments = args;
                parser.parseRegion(partitionRegion);
                if (!op.regions[1]) {
                    op.regions[1] = { blocks: [{ operations: [] }] };
                }
                partitionIndex++;
            }
            parser.expect(':');
            const type = {};
            type.inputs = [];
            type.results = [];
            parser.expect('(');
            while (!parser.match(')')) {
                type.inputs.push(parser.parseType());
                if (!parser.match(')')) {
                    parser.expect(',');
                }
            }
            parser.expect(')');
            parser.expect('->');
            parser.expect('(');
            while (!parser.match(')')) {
                type.results.push(parser.parseType());
                if (!parser.match(')')) {
                    parser.expect(',');
                }
            }
            parser.expect(')');
            op.attributes.push({ name: 'function_type', value: type });
            if (partitionNumWarps.length > 0) {
                op.attributes.push({
                    name: 'partitionNumWarps',
                    value: { type: 'array', element_type: 'i32', value: partitionNumWarps }
                });
            }
            return true;
        }
        return super.parseOperation(parser, opName, op);
    }

    parseType(parser, dialectType) {
        if (dialectType.startsWith('!ttg.')) {
            if (parser.match('<')) {
                const content = parser.skip('<', '>');
                return dialectType + content;
            }
            return dialectType;
        }
        return null;
    }
};

mlir.GluonDialect = class extends mlir.Dialect {

    constructor(operations) {
        super('gluon', operations);
    }
};

mlir.TritonNvidiaGPUDialect = class extends mlir.Dialect {

    constructor(operations) {
        super('ttng', operations);
    }

    parseType(parser, dialectType) {
        if (dialectType.startsWith('!ttng.')) {
            if (parser.match('<')) {
                const content = parser.skip('<', '>');
                return dialectType + content;
            }
            return dialectType;
        }
        return null;
    }
};

mlir.ProtonDialect = class extends mlir.Dialect {

    constructor(operations) {
        super('proton', operations);
    }
};

mlir.MichelsonDialect = class extends mlir.Dialect {

    constructor(operations) {
        super('michelson', operations);
    }

    parseType(parser, dialectType) {
        const simpleTypes = ['int', 'bytes', 'operation', 'nat', 'string', 'unit', 'bool', 'mutez', 'timestamp', 'address', 'key', 'signature', 'chain_id', 'key_hash'];
        for (const simpleType of simpleTypes) {
            if (dialectType === `!michelson.${simpleType}`) {
                return dialectType;
            }
        }
        if (dialectType.startsWith('!michelson.pair') || dialectType.startsWith('!michelson.list') ||
            dialectType.startsWith('!michelson.option') || dialectType.startsWith('!michelson.or') ||
            dialectType.startsWith('!michelson.map') || dialectType.startsWith('!michelson.big_map') ||
            dialectType.startsWith('!michelson.set') || dialectType.startsWith('!michelson.contract') ||
            dialectType.startsWith('!michelson.lambda')) {
            if (parser.match('<')) {
                const content = parser.skip('<', '>');
                return dialectType + content;
            }
            return dialectType;
        }
        return null;
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
        this.operations = new Map();
        if (data) {
            const operations = JSON.parse(data);
            for (const op of operations) {
                const [dialectName] = op.name.split('.');
                if (!this.operations.has(dialectName)) {
                    this.operations.set(dialectName, []);
                }
                this.operations.get(dialectName).push(op);
            }
        }
    }

    type(name) {
        const [dialectName] = name.split('.');
        const operations = this.operations.get(dialectName);
        if (operations) {
            const op = operations.find((op) => op.name === name);
            if (op) {
                return op;
            }
        }
        return { name };
    }
};

mlir.Error = class extends Error {

    constructor(message) {
        super(message);
        this.name = 'Error loading MLIR model.';
    }
};

export const ModelFactory = mlir.ModelFactory;
