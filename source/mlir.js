
// Experimental
// contributor @tucan9389

const mlir = {};

mlir.ModelFactory = class {

    match(context) {
        const stream = context.stream;
        if (stream && stream.length > 4) {
            const buffer = stream.peek(4);
            const signature = String.fromCharCode.apply(null, buffer);
            if (signature === 'ML\xEFR') {
                context.type = 'mlir.binary';
                return;
            }
        }
        try {
            const reader = context.read('text', 0x10000);
            for (let line = reader.read('\n'); line !== undefined; line = reader.read('\n')) {
                if (/module\s+(\w+\s+)?{/.test(line) || /tensor<\w+>/.test(line) || /func\s*@\w+/.test(line)) {
                    context.type = 'mlir.text';
                    return;
                }
            }
        } catch {
            // continue regardless of error
        }
    }

    async open(context) {
        switch (context.type) {
            case 'mlir.text': {
                const decoder = context.read('text.decoder');
                const parser = new mlir.Parser(decoder);
                const obj = await parser.read();
                return new mlir.Model(obj);
            }
            case 'mlir.binary': {
                const reader = new mlir.BytecodeReader(context);
                reader.read();
                throw new mlir.Error('Invalid file content. File contains MLIR bytecode data.');
            }
            default: {
                throw new mlir.Error(`Unsupported MLIR format '${context.type}'.`);
            }
        }
    }
};

mlir.Model = class {

    constructor(obj) {
        this.format = 'MLIR';
        this.graphs = [];
        this.metadata = [];
        for (const op of obj.operations) {
            if (op.name.endsWith('.func')) {
                const graph = new mlir.Graph(op);
                this.graphs.push(graph);
            }
            if (op.name.endsWith('.module')) {
                for (const region of op.regions) {
                    for (const block of region.blocks) {
                        for (const op of block.operations) {
                            if (op.name.endsWith('.func')) {
                                const graph = new mlir.Graph(op);
                                this.graphs.push(graph);
                            }
                        }
                    }
                }
            }
        }
        if (obj.definitions) {
            for (const attribute of obj.definitions) {
                const metadata = new mlir.Argument(attribute.name, attribute.value, attribute.type);
                this.metadata.push(metadata);
            }
        }
    }
};

mlir.Graph = class {

    constructor(func) {
        const attr = Object.fromEntries(func.attributes.map((attr) => [attr.name, attr.value]));
        this.name = attr.sym_name || '';
        this.type = func.name;
        this.inputs = [];
        this.outputs = [];
        this.nodes = [];
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
                    const operands = op.operands || [];
                    for (let i = 0; i < operands.length; i++) {
                        const input = op.operands[i];
                        if (input.value instanceof Uint8Array) {
                            operation.operands.push({
                                name: input.name || i.toString(),
                                value: input.value,
                                type: input.type
                            });
                        } else if (Number.isInteger(input.value)) {
                            operation.operands.push({
                                name: input.name || i.toString(),
                                value: input.value,
                                type: 'int64'
                            });
                        } else if (typeof input.value === 'boolean') {
                            operation.operands.push({
                                name: input.name || i.toString(),
                                value: input.value,
                                type: 'boolean'
                            });
                        } else if (Array.isArray(input.value)) {
                            operation.operands.push({
                                name: input.name || i.toString(),
                                value: input.value
                            });
                        } else {
                            const value = values.map(input);
                            value.to.push(operation);
                            const args = [{ name: input.value, type: input.type }];
                            operation.operands.push({
                                name: input.name || i.toString(),
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
                        operation.results.push({
                            name: output.name || i.toString(),
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
        const tensors = new Map();
        const tensor = (arg) => {
            if (!tensors.has(arg.name)) {
                tensors.set(arg.name, new mlir.Value(arg.name, arg.type, null, arg.value));
            }
            return tensors.get(arg.name);
        };
        for (const input of this.inputs) {
            for (const arg of input.value) {
                tensors.set(arg.name, arg);
            }
        }
        for (const output of this.outputs) {
            for (const arg of output.value) {
                tensors.set(arg.name, arg);
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
            // const type = op.type; 'program:' + op.type;
            // const metadata = this._metadata.type(type);
            // if (metadata && Array.isArray(metadata.inputs)) {
            //     let index = 1;
            //     const map = new Map(metadata.inputs.map((input) => [ input.name, index++ ]));
            //     op.inputs.sort((a, b) => (map.get(a.name) || map.size) - (map.get(b.name) || map.size));
            // }
            const node = new mlir.Node(op);
            this.nodes.push(node);
        }
    }
};

mlir.Argument = class {

    constructor(name, value, type) {
        this.name = name;
        this.value = value;
        this.type = type || null;
        switch (this.type) {
            case 'i64': this.type = 'int64'; break;
            case 'si64': this.type = 'int64'; break;
            case 'i32': this.type = 'int32'; break;
            case 'f32': this.type = 'float32'; break;
            case 'f64': this.type = 'float64'; break;
            case null:
            case 'attribute':
            case 'boolean':
            case 'string':
            case 'int64':
            case 'tensor':
                break;
            default:
                throw new mlir.Error(`Unsupported argument type '${this.type}'.`);
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

    constructor(op) {
        if (!op.type) {
            throw new mlir.Error('Undefined node type.');
        }
        this.type = { name: op.type || '', identifier: op.identifier || '' };
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
                        if (token.value === 'tensor') {
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
                            return new mlir.Token('tensor', `${token.value}${v}`);
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

    constructor(decoder) {
        this._tokenizer = new mlir.Tokenizer(decoder);
        this._token = this._tokenizer.read();
        this._state = {
            defaultDialectStack: ['builtin']
        };
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
                // type-alias-def ::= `!` alias-name `=` type
                throw new mlir.Error('Type alias definition is not implemented.');
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
                const input = {};
                input.value = this._token.value;
                if (this._match('keyword', 'loc')) {
                    this._parseAttributeValue();
                } else {
                    this._read('%');
                    this._read(':');
                    input.type = this._parseType();
                    // attribute
                    if (this._match('{')) {
                        input.attributes = [];
                        this.parseAttributeDict(input.attributes);
                    }
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
            output.type = this._parseType();
            outputs.push(output);
        }
        return outputs;
    }

    _skipSymbolBetween(open, close) {
        let value = '';
        if (this._match(open)) {
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
            const region = {};
            this.parseRegion(region);
            op.regions.push(region);
            if (op.regions.length > 0) {
                const region = op.regions[op.regions.length - 1];
                if (region.blocks.length > 0) {
                    const block = region.blocks[region.blocks.length - 1];
                    if (block.operations.length > 0) {
                        const op = block.operations[block.operations.length - 1];
                        type.results = op.operands;
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
        if (this._match('}')) {
            return op;
        }
        if (op.name === 'torch.constant.none') {
            return op;
        }
        if (this._match('{')) {
            this.parseAttributeDict(op.attributes);
        }
        // (%a, %b)
        // condition: start with `(%`, `%`, or `()`
        op.operands = this._parseArguments();
        if (op.name.endsWith('.for')) {
            this._read('=');
            this._read();
            this._read('id', 'to');
            this._read();
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
        // successor-list?
        // condition: start with `[`, end with `]`
        this._skipSymbolBetween('[', ']');
        // dictionary-properties?
        // condition: start with `<`, end with `>`
        this._skipSymbolBetween('<', '>');
        // region-list?
        // condition: start with `({^`, or (operation, end with `)`
        if (this._eat('(') && this._match('{')) {
            let count = 1;
            while (count > 0) {
                if (this._match('(')) {
                    count++;
                } else if (this._match(')')) {
                    count--;
                }
                this._read();
            }
        }
        // dictionary-attribute?
        // condition: start with `{`, end with `}`
        if (this._match('{')) {
            if (op.attributes.length === 0 || (op.attributes.length === 1 && op.attributes[0].name === 'predicate')) {
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
                }
                let attribute = {};
                if (this._eat('=')) {
                    attribute = this._parseValue();
                    if (this._eat(':')) {
                        attribute.type = this._parseType();
                    }
                }
                attribute.name = name;
                attributes.push(attribute);
                this._eat(',');
            }
        }
    }

    parseRegion(region) {
        region.blocks = Array.isArray(region.blocks) ? region.blocks : [];
        const block = {};
        this.parseBlock(block);
        region.blocks.push(block);
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
                location.file = this._read('string');
                if (this._eat(':')) {
                    location.line = this._read('int');
                    if (this._eat(':')) {
                        location.col = this._read('int');
                    }
                }
            } else if (this._match('#')) {
                location.alias = this._read();
            } else if (this._match('id', 'unknown')) {
                this._read();
            } else {
                throw new mlir.Error(`Unexpected location '${this._token.value}' ${this._tokenizer.location()}`);
            }
            this._read(')');
            return location;
        }
        return null;
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
        while (!this._match(')') && !this._match('->') && !this._match('{')) {
            const input = {};
            if (this._token.kind === 'id' && this._token.value !== 'dense') {
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
                } else {
                    input.name = identifier;
                    this._read('=');
                }
            }
            if (this._match('%')) {
                input.value = this._read();
                if (open && this._eat(':')) {
                    input.type = this._parseType();
                }
            } else if (this._match('keyword', 'loc')) {
                continue;
            } else {
                const value = this._parseValue();
                input.type = value.type;
                input.value = value.value;
                if (open && this._eat(':')) {
                    input.type = this._parseType();
                }
            }
            inputs.push(input);
            if (!this._eat(',')) {
                break;
            }
        }
        if (open) {
            this._read(')');
        }
        return inputs;
    }

    _parseType() {
        if (this._token.kind === 'id') {
            if (this._token.value === 'none' ||
                this._token.value === 'i32' ||
                this._token.value === 'i64' ||
                this._token.value === 'si64' ||
                this._token.value === 'f32' ||
                this._token.value === 'f64' ||
                this._token.value === 'index') {
                return this._read('id');
            }
        }
        if (this._match('tensor')) {
            return this._read();
        }
        if (this._match('id', 'memref')) {
            let value = this._read('id');
            value += this._skipSymbolBetween('<', '>');
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
        if (this._eat('[')) {
            const list = [];
            while (!this._eat(']')) {
                list.push(this._parseValue().value);
                this._eat(',');
            }
            if (this._eat('id', 'x')) {
                list[0] = Array.from(list);
                const second = [];
                this._read('[');
                while (!this._eat(']')) {
                    second.push(this._parseValue().value);
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
            if (this._match('<')) {
                value.value += this._read('<');
                while (!this._match('>')) {
                    value.value += this._read();
                }
                value.value += this._read('>');
            }
            return value;
        }
        if (this._match('tensor')) {
            value.value = this._parseType();
            value.type = 'type';
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
        throw new mlir.Error(`Unexpected attribute value ${this._tokenizer.location()}`);
    }

    _match(kind, value) {
        return (this._token.kind === kind && (!value || this._token.value === value));
    }

    _read(kind, value) {
        if (kind && this._token.kind !== kind) {
            throw new mlir.Error(`Expected token of type '${kind}', but got '${this._token.kind}' ${this._tokenizer.location()}`);
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

    constructor(context) {
        this._reader = new mlir.BinaryReader(context);
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

    constructor(context) {
        this._reader = context.read('binary');
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
            case 'f16': return 'float16';
            case 'f32': return 'float32';
            case 'f64': return 'float64';
            case 'i1': return 'boolean';
            case 'i8': return 'int8';
            case 'i16': return 'int16';
            case 'i32': return 'int32';
            case 'i64': return 'int64';
            case 'si8': return 'int8';
            case 'si16': return 'int16';
            case 'si32': return 'int32';
            case 'si64': return 'int64';
            case 'ui8': return 'uint8';
            case 'ui16': return 'uint16';
            case 'ui32': return 'uint32';
            case 'ui64': return 'uint64';
            default: throw new mlir.Error(`Unknown data type '${value}'.`);
        }
    }

    static valueType(type) {
        if (type === undefined) {
            return null;
        }
        // eg. tensor<?x3x2x2xf32>
        if (type.startsWith('tensor<') && type.endsWith('>')) {
            const spec = type.substring(7, type.length - 1).trim();
            if (spec.startsWith('!')) {
                return mlir.Utility.valueType(spec);
            }
            const index = type.lastIndexOf('x');
            let dataType = spec;
            let shape = [];
            if (index > -1) {
                dataType = type.substring(index + 1, type.length - 1);
                if (!Number.isInteger(parseInt(dataType, 10))) {
                    shape = type.substring(7, index).split('x').map((dim) => parseInt(dim, 10)).map((dim) => isNaN(dim) ? '?' : dim);
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

mlir.Error = class extends Error {

    constructor(message) {
        super(message);
        this.name = 'Error loading MLIR model.';
    }
};

export const ModelFactory = mlir.ModelFactory;
