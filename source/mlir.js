
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
                const obj = parser.read();
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
        this.graphs = obj.functions.map((func) => new mlir.Graph(func));
    }
};

mlir.Graph = class {

    constructor(func) {
        this.name = func.name;
        this.inputs = [];
        this.outputs = [];
        this.nodes = [];
        const valueType = (type) => {
            if (type === undefined) {
                return null;
            }
            // eg. tensor<?x3x2x2xf32>
            if (type.startsWith('tensor<') && type.endsWith('>')) {
                const spec = type.substring(7, type.length - 1).trim();
                if (spec.startsWith('!')) {
                    return valueType(spec);
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
            return type;
        };
        // inputs of function
        for (let i = 0; i < func.inputs.length; i++) {
            const input = func.inputs[i];
            const inputType = func.inputTypes[i];
            const type = valueType(inputType);
            const value = new mlir.Value(input, type, '', null);
            const argument = new mlir.Argument(input, [value]);
            this.inputs.push(argument);
        }
        // outputs of function
        for (let i = 0; i < func.outputTypes.length; i++) {
            const output = Array.isArray(func.outputs) && func.outputs.length > 0 ? func.outputs[i] : `%return/${i}`;
            const outputType = func.outputTypes[i];
            const type = valueType(outputType);
            const value = new mlir.Value(output, type, '', null);
            const argument = new mlir.Argument(output, [value]);
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
        const operations = func.operations.map((op) => {
            const operation = {
                type: op.name,
                attributes: new Map(),
                inputs: [],
                outputs: [],
                delete: false,
            };
            // convert attributes to proper types
            operation.attributes = op.attributes;
            // for (const [key, value] of Object.entries(op.attributes)) {
            //     operation.attributes[key] = convertValue(value);
            // }
            for (let j = 0; j < (op.inputs ? op.inputs.length : 0); j++) {
                const input = op.inputs[j];
                const inputType = op.inputTypes[j];
                const value = values.map(input);
                value.to.push(operation);
                const args = [{ name: input, value: inputType }];
                operation.inputs.push({
                    name: input,
                    arguments: args
                });
            }
            for (let j = 0; j < (op.outputs ? op.outputs.length : 0); j++) {
                const output = op.outputs[j];
                const outputType = op.outputTypes[j];
                const value = values.map(output);
                value.type = valueType(outputType);
                value.from.push(operation);
                operation.outputs.push({
                    name: output,
                    arguments: [value]
                });
            }
            return operation;
        });

        // // operations - constant ops
        // for (const op of operations) {
        //     if (op.type === 'const' && op.inputs.length === 0 &&
        //         op.outputs.length === 1 && op.outputs[0].arguments.length === 1) {
        //         const argument = op.outputs[0].arguments[0];
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
        //         if (input.arguments.length > 1 && input.arguments.some((argument) => argument.const)) {
        //             if (input.arguments.every((argument) => argument.value instanceof mlir.Tensor)) {
        //                 continue;
        //             }
        //             for (const argument of input.arguments) {
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
        //         if (input.arguments.every((argument) => argument.value === undefined || argument.value instanceof coreml.Tensor)) {
        //             return true;
        //         }
        //         if (input.arguments.length === 1) {
        //             const argument = input.arguments[0];
        //             op.attributes[input.name] = argument.value;
        //             return false;
        //         }
        //         op.attributes[input.name] = input.arguments.map((argument) => argument.value[0]);
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
            op.inputs = op.inputs.map((input) => new mlir.Argument(input.name, input.arguments.map((argument) => tensor(argument))));
            op.outputs = op.outputs.map((output) => new mlir.Argument(output.name, output.arguments.map((argument) => tensor(argument))));
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
        this.type = { name: op.type || '' };
        this.name = op.name || '';
        this.inputs = op.inputs || [];
        this.outputs = op.outputs || [];
        this.attributes = [];
        if (op.attributes) {
            for (const [name, value] of op.attributes) {
                const attribute = new mlir.Argument(name, value, 'string');
                this.attributes.push(attribute);
            }
        }
    }
};

mlir.Tensor = class {

    constructor(type, data) {
        this.type = type;  // mlir.TensorType
        this.values = data;
        switch (this.type.dataType) {
            case 'float32': this.encoding = '|'; break;
            default: this.encoding = '<'; break;
        }
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

mlir.TokenType = {
    IDENTIFIER: 'IDENTIFIER',
    BOOLEAN_LITERAL: 'BOOLEAN_LITERAL',
    INTEGER_LITERAL: 'INTEGER_LITERAL',
    HEXADECIMAL_LITERAL: 'HEXADECIMAL_LITERAL',
    FLOAT_LITERAL: 'FLOAT_LITERAL',
    STRING_LITERAL: 'STRING_LITERAL',
    SYMBOL_REF_ID: 'SYMBOL_REF_ID',
    ATTRIBUTE_ALIAS: '#',
    TYPE: 'TYPE',
    DENSE: 'DENSE',
    VALUE_ID: '%',
    CARET_ID: '^',
    KEYWORD: 'KEYWORD',
    EOF: 'EOF',
};

mlir.Token = class {

    constructor(type, value) {
        this.type = type;
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
                    return new mlir.Token(mlir.TokenType.KEYWORD, '.');
                case '-':
                    if (/[0-9]/.test(this._peek())) {
                        return this._number();
                    } else if (this._peek() === '>') {
                        this._read();
                        this._read();
                        return new mlir.Token('->', '->');
                    }
                    this._read();
                    return new mlir.Token(mlir.TokenType.KEYWORD, '-');
                case '+':
                    if (/[0-9]/.test(this._peek())) {
                        return this._number();
                    }
                    this._read();
                    return new mlir.Token(mlir.TokenType.KEYWORD, '+');
                case '"':
                    return this._stringLiteral();
                case '@':
                    return this._symbolRefId();
                case '%':
                    return this._valueId();
                case '#':
                    return this._attributeAlias();
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
                    this._read();
                    return new mlir.Token(',', ',');
                case '(':
                    this._read();
                    return new mlir.Token('(', '(');
                case ')':
                    this._read();
                    return new mlir.Token(')', ')');
                case '{':
                    this._read();
                    return new mlir.Token('{', '{');
                case '}':
                    this._read();
                    return new mlir.Token('}', '}');
                case '[':
                    this._read();
                    return new mlir.Token('[', '[');
                case ']':
                    this._read();
                    return new mlir.Token(']', ']');
                case '<':
                    this._read();
                    return new mlir.Token('<', '<');
                case '>':
                    this._read();
                    return new mlir.Token('>', '>');
                default:
                    if (/[a-zA-Z_$]/.test(this._current) || /[-.]/.test(this._current)) {
                        return this._identifier();
                    }
                    if (/[0-9]/.test(this._current)) {
                        let result = '';
                        const type = mlir.TokenType.INTEGER_LITERAL;
                        while (this._current && /[0-9]/.test(this._current)) {
                            result += this._read();
                        }
                        if (this._current === 'x') {
                            // Read the rest of the shape
                            do {
                                result += this._read();
                            } while (this._current && /[0-9x]/.test(this._current));
                            return new mlir.Token(mlir.TokenType.SHAPE, result);
                        }
                        return new mlir.Token(type, parseInt(result, 10));
                    }
                    return new mlir.Token(mlir.TokenType.KEYWORD, this._read());
            }
        }
        return new mlir.Token(mlir.TokenType.EOF, null);
    }

    location() {
        let line = 1;
        let column = 1;
        this._decoder.position = 0;
        let c = '';
        do {
            if (this._decoder.position === this._position) {
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
        if (this._eat('/')) {
            if (this._current === '/') {
                while (this._current && this._current !== '\n') {
                    this._read();
                }
                this._skipWhitespace();
                this._skipComment();
            } else if (this._current === '*') {
                while (this._current) {
                    this._read();
                    if (this._eat('*') && this._eat('/')) {
                        break;
                    }
                }
                this._skipWhitespace();
                this._skipComment();
            }
        }
    }

    _number() {
        let result = '';
        let type = mlir.TokenType.INTEGER_LITERAL;
        while (this._current && /[0-9]/.test(this._current)) {
            result += this._read();
        }
        if (this._current === 'x') {
            result += this._read();
            type = mlir.TokenType.HEXADECIMAL_LITERAL;
            while (this._current && /[0-9a-fA-F]/.test(this._current)) {
                result += this._read();
            }
        } else if (this._current === '.') {
            result += this._read();
            type = mlir.TokenType.FLOAT_LITERAL;
            while (this._current && /[0-9]/.test(this._current)) {
                result += this._read();
            }
            if (this._current === 'e' || this._current === 'E') {
                result += this._read();
                if (this._current === '+' || this._current === '-') {
                    result += this._read();
                }
                while (this._current && /[0-9]/.test(this._current)) {
                    result += this._read();
                }
                if (type === mlir.TokenType.INTEGER_LITERAL && /[.eE]/.test(this._current)) {
                    type = mlir.TokenType.FLOAT_LITERAL;
                }
                if (type === mlir.TokenType.FLOAT_LITERAL && !/[.eE]/.test(this._current)) {
                    return new mlir.Token(type, parseFloat(result));
                }
                if (type === mlir.TokenType.HEXADECIMAL_LITERAL && !/[x]/.test(this._current)) {
                    return new mlir.Token(type, parseInt(result, 16));
                }
                return new mlir.Token(type, result);
            }
        }
        return new mlir.Token(type, parseInt(result, 10));
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
            return new mlir.Token(mlir.TokenType.STRING_LITERAL, result);
        }
        throw new mlir.Error('Unterminated string literal');
    }

    _identifier() {
        let result = '';
        let opened = 0;
        let wasOpened = false;
        while (this._current) {
            if (!opened) {
                if (this._current && (/[a-zA-Z_$<>\-.*]/.test(this._current) || /[0-9]/.test(this._current))) {
                    if (this._current === '<') {
                        opened += 1;
                        wasOpened = true;
                    }
                    result += this._read();
                } else {
                    break;
                }
            } else if (!this._current) {
                break;
            } else if (this._current === '>') {
                result += this._read();
                opened -= 1;
                if (opened === 0) {
                    break;
                }
            } else {
                if (this._current === '<') {
                    opened += 1;
                }
                result += this._read();
            }
        }
        if (wasOpened) {
            if (result.startsWith('dense')) {
                return new mlir.Token(mlir.TokenType.DENSE, result);
            }
            return new mlir.Token(mlir.TokenType.TYPE, result);
        }
        if (result.endsWith('func')) {
            return new mlir.Token(mlir.TokenType.KEYWORD, result);
        }
        switch (result) {
            case 'module':
            case 'func':
            case 'loc':
                return new mlir.Token(mlir.TokenType.KEYWORD, result);
            case 'true':
            case 'false':
                return new mlir.Token(mlir.TokenType.BOOLEAN_LITERAL, result === 'true');
            case 'unknown':
                return new mlir.Token(mlir.TokenType.IDENTIFIER, result);
            default:
                return new mlir.Token(mlir.TokenType.IDENTIFIER, result);
        }
    }

    _attributeAlias() {
        let result = '#';
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
        return new mlir.Token(mlir.TokenType.ATTRIBUTE_ALIAS, result);
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
        return new mlir.Token(mlir.TokenType.SYMBOL_REF_ID, result);
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
        return new mlir.Token(mlir.TokenType.VALUE_ID, result);
    }

    _caretId() {
        let result = '^';
        this._read();
        if (this._current === ':' && this._peek() !== ':') {
            result += this._read();
            return new mlir.Token(mlir.TokenType.CARET_ID, result);
        }
        while (this._current && (/[a-zA-Z_$]/.test(this._current) || /[0-9]/.test(this._current) || /[-.]/.test(this._current))) {
            result += this._read();
        }
        if (this._current === ':' && this._peek() === ':') {
            result += this._read();
            result += this._read();
            result += this._caretId().value;
        }
        return new mlir.Token(mlir.TokenType.CARET_ID, result);
    }
};

mlir.Parser = class {

    constructor(decoder) {
        this._tokenizer = new mlir.Tokenizer(decoder);
        this._current = this._tokenizer.read();
    }

    read() {
        // https://mlir.llvm.org/docs/LangRef/#top-level-productions
        const graph = {
            functions: [],
            operations: [],
            attributes: new Map()
        };
        while (this._match(mlir.TokenType.ATTRIBUTE_ALIAS)) {
            const attribute = this._eat(mlir.TokenType.ATTRIBUTE_ALIAS);
            this._read('=');
            graph.attributes.set(attribute.value, this._parseAttributeValue());
        }
        const module = this._eat(mlir.TokenType.KEYWORD, 'module');
        if (module) {
            const symbol = this._eat(mlir.TokenType.SYMBOL_REF_ID);
            if (symbol) {
                graph.name = symbol.value;
            }
            const attributes = this._eat(mlir.TokenType.IDENTIFIER, 'attributes');
            if (attributes) {
                graph.attributes = this._parseAttributes();
            }
            this._read('{');
        }
        // functions or operations
        const terminal = module ? '}' : mlir.TokenType.EOF;
        while (this._current.type !== terminal) {
            if (this._current.type === mlir.TokenType.KEYWORD && (this._current.value === 'func' || this._current.value === 'func.func')) {
                // function
                const func = this._parseFunction();
                graph.functions.push(func);
            } else {
                // operation
                const op = this._parseOperation();
                graph.operations.push(op);
            }
        }
        if (module) {
            this._read('}');
        }
        return graph;
    }

    _parseFunction() {
        // func keyword
        const func = {};
        const type = this._read(mlir.TokenType.KEYWORD);
        func.type = type.value;
        func.visibility = null;
        if (this._current.type !== mlir.TokenType.SYMBOL_REF_ID) {
            func.visibility = this._current.value;
            this._read(this._current.type);
        }
        func.name = this._parseFunctionName();
        const inputs = this._parseFunctionInputs();
        func.attributes = new Map();
        // attributes
        if (this._eat(mlir.TokenType.IDENTIFIER, 'attributes')) {
            for (const [key, value] of this._parseAttributes()) {
                func.attributes.set(key, value);
            }
        }
        const outputTypes = [];
        if (this._eat('->')) {
            for (const output of this._parseFunctionOutputs()) {
                outputTypes.push(output.type);
            }
        }
        // attributes
        if (this._eat(mlir.TokenType.IDENTIFIER, 'attributes')) {
            for (const [key, value] of this._parseAttributes()) {
                func.attributes.set(key, value);
            }
        }
        this._read('{');
        // operations
        func.operations = [];
        while (this._current.type !== '}') {
            const operation = this._parseOperation();
            func.operations.push(operation);
        }
        this._read('}');
        if (this._match(mlir.TokenType.KEYWORD, 'loc')) {
            this._parseAttributeValue();
        }
        func.inputs = inputs.map((input) => input.name);
        func.inputTypes = inputs.map((input) => input.type);
        func.outputTypes = outputTypes;
        if (func.operations.length > 0) {
            const ret = func.operations[func.operations.length - 1];
            func.outputs = ret.inputs;
            func.operations.pop();
        }
        return func;
    }

    _parseFunctionName() {
        const name = this._current.value;
        this._read(mlir.TokenType.SYMBOL_REF_ID);
        return name;
    }

    _parseFunctionInputs() {
        this._read('(');
        const inputs = [];
        while (!this._eat(')')) {
            const input = {
                name: this._current.value,
            };
            if (this._match(mlir.TokenType.KEYWORD, 'loc')) {
                this._parseAttributeValue();
            } else {
                this._read(mlir.TokenType.VALUE_ID);
                this._read(':');
                input.type = this._current.value;
                if (!this._eat(mlir.TokenType.TYPE)) {
                    this._eat(mlir.TokenType.IDENTIFIER);
                }
                // attribute
                if (this._match('{')) {
                    input.attributes = this._parseAttributes();
                }
                inputs.push(input);
                this._eat(',');
            }
        }
        return inputs;
    }

    _parseFunctionOutputs() {
        const outputs = [];
        if (this._eat('(')) {
            while (!this._eat(')')) {
                const output = {
                    type: this._current.value,
                };
                if (!this._eat(mlir.TokenType.TYPE)) {
                    this._eat(mlir.TokenType.IDENTIFIER);
                }
                // attribute
                if (this._current.type === '{') {
                    output.attributes = this._parseAttributes();
                }
                outputs.push(output);
                this._eat(',');
            }
        } else {
            const output = {
                type: this._current.value,
            };
            if (!this._eat(mlir.TokenType.TYPE)) {
                this._eat(mlir.TokenType.IDENTIFIER);
            }
            outputs.push(output);
        }
        return outputs;
    }

    _parseOperation() {
        // %3
        const operation = {};
        operation.outputs = this._parseReturnValues();
        // =
        this._eat('=');
        // 'add'
        operation.name = this._parseOperationName();
        if (this._current.type === '}') {
            return operation;
        }
        const skipSymbolBetween = (openingTokenType, closingTokenType) => {
            let count = 1;
            while (count > 0) {
                if (this._current.type === openingTokenType) {
                    count++;
                } else if (this._current.type === closingTokenType) {
                    count--;
                }
                this._read(this._current.type);
            }
        };
        // (%a, %b)
        // condition: start with `(%`, `%`, or `()`
        const inputs = this._parseInputArguments();
        // successor-list?
        // condition: start with `[`, end with `]`
        if (this._eat('[')) {
            skipSymbolBetween('[', ']');
        }
        // dictionary-properties?
        // condition: start with `<`, end with `>`
        if (this._eat('<')) {
            skipSymbolBetween('<', '>');
        }
        // region-list?
        // condition: start with `({^`, or (operation, end with `)`
        if (this._eat('(') && this._current.type === '{') {
            skipSymbolBetween('(', ')');
        }
        // dictionary-attribute?
        // condition: start with `{`, end with `}`
        operation.attributes = this._parseAttributes();
        // : (f32, tensor<1xf32>)
        let inputTypes = [];
        if (this._eat(':')) {
            inputTypes = this._parseInputArgumentTypes();
        }
        const outputTypes = [];
        if (operation.name.endsWith('constant') && this._current.type !== '->') {
            // constant
            operation.outputTypes = outputTypes;
            operation.isConstant = true;
            // operation.data = this._parseConstantData();
            return operation;
        }
        // -> f32
        if (this._eat('->')) {
            outputTypes.push(...this._parseOutputType());
        }
        if (this._match(mlir.TokenType.KEYWORD, 'loc')) {
            this._parseAttributeValue();
        }
        let body = null;
        if (this._eat('{')) {
            let braceCount = 0;
            braceCount++;
            body = '{ ';
            while (braceCount > 0) {
                if (this._current.type === '{') {
                    braceCount++;
                } else if (this._current.type === '}') {
                    braceCount--;
                }
                if (braceCount > 0) {
                    body += this._current.value;
                    if (this._current.type === '{' || this._current.type === '}') {
                        body += '\n';
                    } else if (this._current.type !== mlir.TokenType.WHITESPACE) {
                        body += ' ';
                    }
                }
                this._read(this._current.type);
            }
            body += '}';
        }
        for (const [key, value] of this._parseAttributes()) {
            operation.attributes.set(key, value);
        }
        operation.inputs = inputs;
        operation.inputTypes = inputTypes;
        operation.outputTypes = outputTypes;
        operation.body = body;
        return operation;
    }

    _parseReturnValues() {
        const outputs = [];
        if (this._eat('(')) {
            while (!this._eat(')')) {
                const value = this._eat(mlir.TokenType.VALUE_ID);
                if (value) {
                    outputs.push(value.value);
                }
                this._eat(',');
            }
        } else {
            const value = this._eat(mlir.TokenType.VALUE_ID);
            if (value) {
                outputs.push(value.value);
            }
            if (this._eat(',')) {
                while (this._current.type === mlir.TokenType.VALUE_ID) {
                    const value = this._read(mlir.TokenType.VALUE_ID);
                    outputs.push(value.value);
                    this._eat(',');
                }
            }
        }
        const result = [];
        for (const output of outputs) {
            if (output.split(':').length === 2) {
                const [valueId, length] = output.split(':');
                for (let i = 0; i < length; i++) {
                    result.push(`${valueId}#${i}`);
                }
            } else {
                result.push(output);
            }
        }
        return result;
    }

    _parseOperationName() {
        let value = '';
        switch (this._current.type) {
            case mlir.TokenType.STRING_LITERAL:
                value = this._current.value;
                this._read(mlir.TokenType.STRING_LITERAL);
                break;
            case mlir.TokenType.IDENTIFIER:
                value = this._current.value;
                this._read(mlir.TokenType.IDENTIFIER);
                if (this._current.type === mlir.TokenType.IDENTIFIER) {
                    value += this._current.value;
                    this._read(mlir.TokenType.IDENTIFIER);
                }
                break;
            default:
                throw new mlir.Error(`Unexpected operation '${this._current.value}' ${this._tokenizer.location()}`);
        }
        return value;
    }

    _parseInputArguments() {
        const inputs = [];
        this._eat('(');
        while (this._current.type !== ')' &&
               this._current.type !== ':' &&
               this._current.type !== '->' &&
               this._current.type !== '}' &&
               this._current.type !== mlir.TokenType.IDENTIFIER &&
               this._current.type !== mlir.TokenType.STRING_LITERAL) {
            const value = this._eat(mlir.TokenType.VALUE_ID);
            if (value) {
                inputs.push(value.value);
            } else {
                const dense = this._eat(mlir.TokenType.DENSE);
                inputs.push(dense.value);
                return inputs;
            }
            this._eat(',');
        }
        this._eat(')');
        return inputs;
    }

    _parseInputArgumentTypes() {
        const inputTypes = [];
        this._eat('(');
        while (this._current.type === mlir.TokenType.TYPE || (this._current.type === mlir.TokenType.IDENTIFIER && this._current.value === 'none')) {
            inputTypes.push(this._current.value);
            this._read(this._current.type);
            this._eat(',');
        }
        this._eat(')');
        return inputTypes;
    }

    _parseOutputArguments() {
        const outputs = [];
        const outputTypes = [];
        this._read('(');
        while (!this._eat(')')) {
            const value = this._eat(mlir.TokenType.VALUE_ID);
            if (value) {
                outputs.push(value.value);
            }
            if (this._eat(':')) {
                const type = this._read(mlir.TokenType.TYPE);
                outputTypes.push(type.value);
            }
            this._eat(',');
        }
        return { outputs, outputTypes };
    }

    _parseOutputType() {
        const outputTypes = [];
        if (this._eat('(')) {
            while (!this._eat(')')) {
                outputTypes.push(this._current.value);
                if (!this._eat(mlir.TokenType.TYPE)) {
                    if (this._current.type === mlir.TokenType.IDENTIFIER && (this._current.value === 'none' || /[^f\\d+$]/.test(this._current.value) || /[^i\\d+$]/.test(this._current.value))) {
                        this._read(mlir.TokenType.IDENTIFIER);
                    }
                }
                this._eat(',');
            }
        } else {
            outputTypes.push(this._current.value);
            if (!this._eat(mlir.TokenType.TYPE)) {
                if (this._current.type === mlir.TokenType.IDENTIFIER && (this._current.value === 'none' || /[^f\\d+$]/.test(this._current.value) || /[^i\\d+$]/.test(this._current.value))) {
                    this._read(mlir.TokenType.IDENTIFIER);
                }
            }
        }
        return outputTypes;
    }

    _parseAttributes() {
        const attributes = new Map();
        if (this._eat('{')) {
            while (!this._eat('}')) {
                const name = this._read(mlir.TokenType.IDENTIFIER).value;
                if (this._eat('=')) {
                    let value = '';
                    let openingCount = 0;
                    while (openingCount !== 0 || (this._current.type !== ',' && this._current.type !== '}')) {
                        switch (this._current.type) {
                            case '[':
                            case '{':
                            case '(':
                                openingCount++;
                                break;
                            case ']':
                            case '}':
                            case ')':
                                openingCount--;
                                break;
                            default:
                                break;
                        }
                        value += `${this._current.value} `;
                        this._read(this._current.type);
                    }
                    attributes.set(name, value.trim());
                } else {
                    attributes.set(name, name);
                }
                this._eat(',');
            }
        }
        return attributes;
    }

    _parseAttributeValue() {
        const loc = this._eat(mlir.TokenType.KEYWORD, 'loc');
        if (loc) {
            const args = [];
            if (this._eat('(')) {
                while (!this._eat(')')) {
                    if (this._match(mlir.TokenType.IDENTIFIER)) {
                        args.push(this._eat(mlir.TokenType.IDENTIFIER).value);
                    } else if (this._match(mlir.TokenType.ATTRIBUTE_ALIAS)) {
                        args.push(this._eat(mlir.TokenType.ATTRIBUTE_ALIAS).value);
                    } else {
                        throw new mlir(`Unexpected token '${this._current}.`);
                    }
                }
            }
            return {
                name: loc.value,
                args
            };
        }
        const alias = this._eat(mlir.TokenType.ATTRIBUTE_ALIAS);
        if (alias) {
            let name = alias.value;
            if (this._eat(mlir.TokenType.KEYWORD, '.')) {
                name += `.${this._read(mlir.TokenType.IDENTIFIER)}`;
            }
            if (this._eat('<')) {
                while (!this._eat('>')) {
                    this._current = this._tokenizer.read();
                }
            }
            return {
                name
            };
        }
        throw new mlir.Error('Unexpected attribute value.');
    }

    _match(type, value) {
        if (this._current.type === type && (!value || this._current.value === value)) {
            return true;
        }
        return false;
    }

    _eat(type, value) {
        if (this._current.type === type && (!value || this._current.value === value)) {
            return this._read(type, value);
        }
        return null;
    }

    _read(type, value) {
        if (this._current.type !== type) {
            throw new mlir.Error(`Expected token of type '${type}', but got '${this._current.type}' ${this._tokenizer.location()}`);
        }
        if (value && this._current.value !== value) {
            throw new mlir.Error(`Expected token with value '${value}', but got '${this._current.value}' ${this._tokenizer.location()}`);
        }
        const current = this._current;
        this._current = this._tokenizer.read();
        return current;
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

mlir.Error = class extends Error {

    constructor(message) {
        super(message);
        this.name = 'Error loading MLIR model.';
    }
};

export const ModelFactory = mlir.ModelFactory;

