
// Experimental
// contributor @tucan9389

const mlir = {};

mlir.ModelFactory = class {

    match(context) {
        try {
            const reader = context.read('text', 0x10000);
            for (let line = reader.read('\n'); line !== undefined; line = reader.read('\n')) {
                if (/module\s+(\w+\s+)?{/.test(line) || /tensor<\w+>/.test(line)) {
                    context.type = 'mlir';
                    return;
                }
            }
        } catch {
            // continue regardless of error
        }
    }

    async open(context) {
        const decoder = context.read('text.decoder');
        const parser = new mlir.Parser(decoder);
        const obj = parser.read();
        return new mlir.Model(obj);
    }
};

mlir.Model = class {

    constructor(obj) {
        this.format = 'MLIR';
        this.graphs = obj.functions.map((func) => new mlir.Graph(func, ''));
    }
};

mlir.Graph = class {

    constructor(func, group) {
        this.inputs = [];  // [mlir.Argument]
        this.outputs = []; // [mlir.Argument]
        this.nodes = [];   // [mlir.Node]
        const valueType = (type) => {
            if (type === undefined) {
                return null;
            }
            // eg. tensor<?x3x2x2xf32>
            if (type.startsWith('tensor<')) {
                const shapeString = type.substring(7, type.length - 1);
                if (!/^[0-9xfiq?*]+$/i.test(shapeString)) {
                    return type;
                }
                const parts = shapeString.split('x');
                const dataType = parts[parts.length - 1];
                const shape = parts
                    .slice(0, -1)
                    .map((dimension) => {
                        const parsedDimension = parseInt(dimension.trim(), 10);
                        return isNaN(parsedDimension) ? '?' : parsedDimension;
                    });
                return new mlir.TensorType(dataType, new mlir.TensorShape(shape));
            }
            return type;
        };
        // inputs of function
        for (let i = 0; i < func.inputs.length; i++) {
            const input = func.inputs[i];
            const inputType = func.inputTypes[i];
            const type = valueType(inputType);
            const value = new mlir.Value(input, type, "input desc", null);
            const argument = new mlir.Argument(input, [value]);
            this.inputs.push(argument);
        }
        // outputs of function
        for (let i = 0; i < func.outputTypes.length; i++) {
            const output = `%return/${i}`;
            const outputType = func.outputTypes[i];
            const type = valueType(outputType);
            const value = new mlir.Value(output, type, "output desc", null);
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
                attributes: {},
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
            const type = op.type; // 'program:' + op.type;
            // const metadata = this._metadata.type(type);
            // if (metadata && Array.isArray(metadata.inputs)) {
            //     let index = 1;
            //     const map = new Map(metadata.inputs.map((input) => [ input.name, index++ ]));
            //     op.inputs.sort((a, b) => (map.get(a.name) || map.size) - (map.get(b.name) || map.size));
            // }
            const node = new mlir.Node(/*this._metadata, */group, type, null, null, op.attributes, op.inputs, op.outputs);
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

    constructor(group, type, name, description, attributes, inputs, outputs) {
        if (!type) {
            throw new mlir.Error('Undefined node type.');
        }
        if (group) {
            this.group = group;
        }
        this.type = { name: type || '' };      // string (metadata.type(type) || { name: type }
        this.name = name || '';                // string
        this.description = description || '';  // string
        this.inputs = inputs || [];            // [mlir.Parameter]
        this.outputs = outputs || [];          // [mlir.Parameter]
        this.attributes = [];                  // [mlir.Attribute]
        if (attributes) {
            for (const [name, value] of Object.entries(attributes)) {
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
    TYPE: 'TYPE',
    DENSE: 'DENSE',
    VALUE_ID: '%',
    CARET_ID: '^',
    COLON: ':',
    COMMA: ',',
    EQUAL: '=',
    LPAREN: '(',
    RPAREN: ')',
    ARROW: '->',
    LBRACKET: '[',
    RBRACKET: ']',
    LBRACE: '{',
    RBRACE: '}',
    LESS_THAN: '<',
    GREATER_THAN: '>',
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
                        return new mlir.Token(mlir.TokenType.ARROW, '->');
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
                case '^':
                    return this._caretId();
                case '=':
                    if (this._peek() === '=') {
                        this._read();
                        this._read();
                        return new mlir.Token(mlir.TokenType.EQUAL_EQUAL, '==');
                    }
                    this._read();
                    return new mlir.Token(mlir.TokenType.EQUAL, '=');
                case ':':
                    if (this._peek() === ':') {
                        this._read();
                        this._read();
                        return new mlir.Token(mlir.TokenType.DOUBLE_COLON, '::');
                    }
                    this._read();
                    return new mlir.Token(mlir.TokenType.COLON, ':');
                case ',':
                    this._read();
                    return new mlir.Token(mlir.TokenType.COMMA, ',');
                case '(':
                    this._read();
                    return new mlir.Token(mlir.TokenType.LPAREN, '(');
                case ')':
                    this._read();
                    return new mlir.Token(mlir.TokenType.RPAREN, ')');
                case '{':
                    this._read();
                    return new mlir.Token(mlir.TokenType.LBRACE, '{');
                case '}':
                    this._read();
                    return new mlir.Token(mlir.TokenType.RBRACE, '}');
                case '[':
                    this._read();
                    return new mlir.Token(mlir.TokenType.LBRACKET, '[');
                case ']':
                    this._read();
                    return new mlir.Token(mlir.TokenType.RBRACKET, ']');
                case '<':
                    this._read();
                    return new mlir.Token(mlir.TokenType.LESS_THAN, '<');
                case '>':
                    this._read();
                    return new mlir.Token(mlir.TokenType.GREATER_THAN, '>');
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

    _match(value) {
        if (this._current === value) {
            this._read();
            return true;
        }
        return false;
    }

    _skipComment() {
        if (this._match('/')) {
            if (this._current === '/') {
                while (this._current && this._current !== '\n') {
                    this._read();
                }
                this._skipWhitespace();
                this._skipComment();
            } else if (this._current === '*') {
                while (this._current) {
                    this._read();
                    if (this._match('*') && this._match('/')) {
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
            if (this._match('\\')) {
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
        if (this._match('"')) {
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
            default:
                return new mlir.Token(mlir.TokenType.IDENTIFIER, result);
        }
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
        const hasModule = this._match(mlir.TokenType.KEYWORD, 'module');
        let attributes = {};
        if (hasModule) {
            // Attributes
            if (this._current.value === 'attributes') {
                this._read(mlir.TokenType.IDENTIFIER, 'attributes');
                attributes = Object.assign(attributes, this._parseAttribute());
            }
            this._read(mlir.TokenType.LBRACE);
        }
        const graph = {
            functions: [],
            operations: [],
            attributes,
        };
        // functions or operations
        const terminal = hasModule ? mlir.TokenType.RBRACE : mlir.TokenType.EOF;
        while (this._current.type !== terminal) {
            if (this._current.type === mlir.TokenType.KEYWORD && this._current.value.endsWith('func')) {
                // function
                const func = this._parseFunction();
                graph.functions.push(func);
            } else {
                // operation
                const op = this._parseOperation();
                graph.operations.push(op);
            }
        }
        if (hasModule) {
            this._read(mlir.TokenType.RBRACE);
        }
        return graph;
    }

    _parseFunction() {
        // func keyword
        this._read(mlir.TokenType.KEYWORD);
        let visibility = null;
        if (this._current.type !== mlir.TokenType.SYMBOL_REF_ID) {
            visibility = this._current.value;
            this._read(this._current.type);
        }
        const name = this._parseFunctionName();
        const inputs = this._parseFunctionInputs();
        let attributes = {};
        // attributes
        if (this._match(mlir.TokenType.IDENTIFIER, 'attributes')) {
            attributes = Object.assign(attributes, this._parseAttribute());
        }
        let outputs = {};
        if (this._match(mlir.TokenType.ARROW)) {
            outputs = Object.assign(outputs, this._parseFunctionOutputs());
        }
        // attributes
        if (this._match(mlir.TokenType.IDENTIFIER, 'attributes')) {
            attributes = Object.assign(attributes, this._parseAttribute());
        }
        this._read(mlir.TokenType.LBRACE);
        // operations
        const operations = [];
        while (this._current.type !== mlir.TokenType.RBRACE) {
            const operation = this._parseOperation();
            operations.push(operation);
        }
        this._read(mlir.TokenType.RBRACE);
        return {
            name,
            inputs: inputs.map((input) => input.name),
            inputTypes: inputs.map((input) => input.type),
            outputTypes: outputs,
            operations,
            attributes,
            visibility,
        };
    }

    _parseFunctionName() {
        const name = this._current.value;
        this._read(mlir.TokenType.SYMBOL_REF_ID);
        return name;
    }

    _parseFunctionInputs() {
        this._read(mlir.TokenType.LPAREN);
        const inputs = [];
        while (!this._match(mlir.TokenType.RPAREN)) {
            const input = {
                name: this._current.value,
            };
            this._read(mlir.TokenType.VALUE_ID);
            this._read(mlir.TokenType.COLON);
            input.type = this._current.value;
            if (!this._match(mlir.TokenType.TYPE)) {
                this._match(mlir.TokenType.IDENTIFIER);
            }
            // attribute
            if (this._current.type === mlir.TokenType.LBRACE) {
                input.attributes = this._parseAttribute();
            }
            inputs.push(input);
            this._match(mlir.TokenType.COMMA);
        }
        return inputs;
    }

    _parseFunctionOutputs() {
        const outputs = [];
        if (this._match(mlir.TokenType.LPAREN)) {
            while (!this._match(mlir.TokenType.RPAREN)) {
                const output = {
                    type: this._current.value,
                };
                if (!this._match(mlir.TokenType.TYPE)) {
                    this._match(mlir.TokenType.IDENTIFIER);
                }
                // attribute
                if (this._current.type === mlir.TokenType.LBRACE) {
                    output.attributes = this._parseAttribute();
                }
                outputs.push(output);
                this._match(mlir.TokenType.COMMA);
            }
        } else {
            const output = {
                type: this._current.value,
            };
            if (!this._match(mlir.TokenType.TYPE)) {
                this._match(mlir.TokenType.IDENTIFIER);
            }
            outputs.push(output);
        }
        return outputs;
    }

    _parseOperation() {
        // %3
        const outputs = this._parseReturnValues();
        // =
        this._match(mlir.TokenType.EQUAL);
        // "add"
        const operationName = this._parseOperationName();
        if (this._current.type === mlir.TokenType.RBRACE) {
            // early return
            return {
                outputs,
                name: operationName,
            };
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
        const { inputs } = this._parseInputArguments();
        // successor-list?
        // condition: start with `[`, end with `]`
        if (this._match(mlir.TokenType.LBRACKET)) {
            skipSymbolBetween(mlir.TokenType.LBRACKET, mlir.TokenType.RBRACKET);
        }
        // dictionary-properties?
        // condition: start with `<`, end with `>`
        if (this._match(mlir.TokenType.LESS_THAN)) {
            skipSymbolBetween(mlir.TokenType.LESS_THAN, mlir.TokenType.GREATER_THAN);
        }
        // region-list?
        // condition: start with `({^`, or (operation, end with `)`
        if (this._match(mlir.TokenType.LPAREN) && this._current.type === mlir.TokenType.LBRACE) {
            skipSymbolBetween(mlir.TokenType.LPAREN, mlir.TokenType.RPAREN);
        }
        // dictionary-attribute?
        // condition: start with `{`, end with `}`
        let attributes = this._parseAttribute();
        // : (f32, tensor<1xf32>)
        let inputTypes = [];
        if (this._match(mlir.TokenType.COLON)) {
            inputTypes = this._parseInputArgumentTypes();
        }
        const outputTypes = [];
        if (operationName.endsWith('constant') && this._current.type !== mlir.TokenType.ARROW) {
            // constant
            const result = {
                name: operationName,
                attributes,
                // data: this._parseConstantData(),
                outputs,
                outputTypes,
                isConstant: true,
            };
            return result;
        }
        // -> f32
        if (this._match(mlir.TokenType.ARROW)) {
            outputTypes.push(...this._parseOutputType());
        }
        let body = null;
        if (this._match(mlir.TokenType.LBRACE)) {
            let braceCount = 0;
            braceCount++;
            body = '{ ';
            while (braceCount > 0) {
                if (this._current.type === mlir.TokenType.LBRACE) {
                    braceCount++;
                } else if (this._current.type === mlir.TokenType.RBRACE) {
                    braceCount--;
                }
                if (braceCount > 0) {
                    body += this._current.value;
                    if (this._current.type === mlir.TokenType.LBRACE || this._current.type === mlir.TokenType.RBRACE) {
                        body += '\n';
                    } else if (this._current.type !== mlir.TokenType.WHITESPACE) {
                        body += ' ';
                    }
                }
                this._read(this._current.type);
            }
            body += '}';
        }
        attributes = Object.assign(attributes, this._parseAttribute());
        const result = {
            name: operationName,
            attributes,
            inputs,
            inputTypes,
            outputs,
            outputTypes,
            body,
        };
        return result;
    }

    _parseReturnValues() {
        const outputs = [];
        if (this._match(mlir.TokenType.LPAREN)) {
            while (!this._match(mlir.TokenType.RPAREN)) {
                const value = this._match(mlir.TokenType.VALUE_ID);
                if (value) {
                    outputs.push(value.value);
                }
                this._match(mlir.TokenType.COMMA);
            }
        } else {
            const value = this._match(mlir.TokenType.VALUE_ID);
            if (value) {
                outputs.push(value.value);
            }
            if (this._match(mlir.TokenType.COMMA)) {
                while (this._current.type === mlir.TokenType.VALUE_ID) {
                    const value = this._read(mlir.TokenType.VALUE_ID);
                    outputs.push(value.value);
                    this._match(mlir.TokenType.COMMA);
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
        this._match(mlir.TokenType.LPAREN);
        while (this._current.type !== mlir.TokenType.RPAREN &&
               this._current.type !== mlir.TokenType.COLON &&
               this._current.type !== mlir.TokenType.ARROW &&
               this._current.type !== mlir.TokenType.RBRACE &&
               this._current.type !== mlir.TokenType.IDENTIFIER &&
               this._current.type !== mlir.TokenType.STRING_LITERAL) {
            const value = this._match(mlir.TokenType.VALUE_ID);
            if (value) {
                inputs.push(value.value);
            } else {
                const dense = this._match(mlir.TokenType.DENSE);
                inputs.push(dense.value);
                return { inputs };
            }
            this._match(mlir.TokenType.COMMA);
        }
        this._match(mlir.TokenType.RPAREN);
        return { inputs };
    }

    _parseInputArgumentTypes() {
        const inputTypes = [];
        this._match(mlir.TokenType.LPAREN);
        while (this._current.type === mlir.TokenType.TYPE || (this._current.type === mlir.TokenType.IDENTIFIER && this._current.value === 'none')) {
            inputTypes.push(this._current.value);
            this._read(this._current.type);
            this._match(mlir.TokenType.COMMA);
        }
        this._match(mlir.TokenType.RPAREN);
        return inputTypes;
    }

    _parseOutputArguments() {
        const outputs = [];
        const outputTypes = [];
        this._read(mlir.TokenType.LPAREN);
        while (!this._match(mlir.TokenType.RPAREN)) {
            const value = this._match(mlir.TokenType.VALUE_ID);
            if (value) {
                outputs.push(value.value);
            }
            if (this._match(mlir.TokenType.COLON)) {
                const type = this._read(mlir.TokenType.TYPE);
                outputTypes.push(type.value);
            }
            this._match(mlir.TokenType.COMMA);
        }
        return { outputs, outputTypes };
    }

    _parseOutputType() {
        const outputTypes = [];
        if (this._match(mlir.TokenType.LPAREN)) {
            while (!this._match(mlir.TokenType.RPAREN)) {
                outputTypes.push(this._current.value);
                if (!this._match(mlir.TokenType.TYPE)) {
                    if (this._current.type === mlir.TokenType.IDENTIFIER && (this._current.value === 'none' || /[^f\\d+$]/.test(this._current.value) || /[^i\\d+$]/.test(this._current.value))) {
                        this._read(mlir.TokenType.IDENTIFIER);
                    }
                }
                this._match(mlir.TokenType.COMMA);
            }
        } else {
            outputTypes.push(this._current.value);
            if (!this._match(mlir.TokenType.TYPE)) {
                if (this._current.type === mlir.TokenType.IDENTIFIER && (this._current.value === 'none' || /[^f\\d+$]/.test(this._current.value) || /[^i\\d+$]/.test(this._current.value))) {
                    this._read(mlir.TokenType.IDENTIFIER);
                }
            }
        }
        return outputTypes;
    }

    _parseAttribute() {
        const attributes = {};
        if (this._match(mlir.TokenType.LBRACE)) {
            while (!this._match(mlir.TokenType.RBRACE)) {
                const name = this._read(mlir.TokenType.IDENTIFIER).value;
                if (this._match(mlir.TokenType.EQUAL)) {
                    let value = '';
                    let openingCount = 0;
                    while (openingCount !== 0 || (this._current.type !== mlir.TokenType.COMMA && this._current.type !== mlir.TokenType.RBRACE)) {
                        switch (this._current.type) {
                            case mlir.TokenType.LBRACKET:
                            case mlir.TokenType.LBRACE:
                            case mlir.TokenType.LPAREN:
                                openingCount++;
                                break;
                            case mlir.TokenType.RBRACKET:
                            case mlir.TokenType.RBRACE:
                            case mlir.TokenType.RPAREN:
                                openingCount--;
                                break;
                            default:
                                break;
                        }
                        value += `${this._current.value} `;
                        this._read(this._current.type);
                    }
                    attributes[name] = value.trim();
                } else {
                    attributes[name] = name;
                }
                this._match(mlir.TokenType.COMMA);
            }
        }
        return attributes;
    }

    _match(type, value) {
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
            case 'i8': return 'int8';
            case 'i16': return 'int16';
            case 'i32': return 'int32';
            case 'i64': return 'int64';
            case 'i1': return 'boolean';
            default: throw new mlir.Error(`Unknown data type '${value}'.`);
        }
    }
};

mlir.Error = class extends Error {

    constructor(message) {
        super(message);
        this.name = 'Error loading MLIR model.';
    }
};

export const ModelFactory = mlir.ModelFactory;

