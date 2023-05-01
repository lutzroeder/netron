/**
 * owner: @lutzroeder
 * contributors: @tucan9389
 **/

var mlir = {};
var text = require('./text');

mlir.ModelFactory = class {

    match(context) {
        const stream = context.stream;
        if (stream) {
            const reader = text.Reader.open(stream, 2048);
            for (; ;) {
                const line = reader.read();
                if (line === undefined) {
                    break;
                }
                if (line.indexOf('module ') !== -1) {
                    return 'mlir';
                }
            }
        }
        return null;
    }

    open(context) {
        const stream = context.stream;
        const decoder = text.Decoder.open(stream);
        const parser = new mlir.Parser(decoder);
        const obj = parser.read();
        const model = new mlir.Model(obj);
        return Promise.resolve(model);
    }
};

mlir.Model = class {

    constructor(obj) {
        this._format = 'MLIR';
        const group = '';
        this._graphs = obj.functions.map(func => new mlir.Graph(func, group));
    }

    get format() {
        return this._format;
    }

    get graphs() {
        return this._graphs;
    }
};

const TokenType = {
    IDENTIFIER: 'IDENTIFIER',
    BOOLEAN_LITERAL: 'BOOLEAN_LITERAL',
    INTEGER_LITERAL: 'INTEGER_LITERAL',
    HEXADECIMAL_LITERAL: 'HEXADECIMAL_LITERAL',
    FLOAT_LITERAL: 'FLOAT_LITERAL',
    STRING_LITERAL: 'STRING_LITERAL',
    SYMBOL_REF_ID: 'SYMBOL_REF_ID',
    TYPE: 'TYPE',
    DENSE: 'DENSE',
    VALUE_ID: 'VALUE_ID',
    CARET_ID: 'CARET_ID',
    COLON: 'COLON',
    COMMA: 'COMMA',
    EQUAL: 'EQUAL',
    LPAREN: 'LPAREN',
    RPAREN: 'RPAREN',
    ARROW: 'ARROW',
    LBRACKET: 'LBRACKET',
    RBRACKET: 'RBRACKET',
    LBRACE: 'LBRACE',
    RBRACE: 'RBRACE',
    LESS_THAN: 'LESS_THAN',
    GREATER_THAN: 'GREATER_THAN',
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
        this.currentChar = this._decoder.decode();
        this.nextChar = this._decoder.decode();
    }

    advance() {
        this.currentChar = this.nextChar;
        this.nextChar = this._decoder.decode();
    }

    peek() {
        return this.nextChar;
    }

    skipWhitespace() {
        while (this.currentChar === ' ' || this.currentChar === '\t' || this.currentChar === '\n' || this.currentChar === '\r' || this.currentChar === '\f') {
            this.advance();
        }
    }

    skipComment() {
        if (this.currentChar === '/') {
            this.advance();
            if (this.currentChar === '/') {
                while (this.currentChar && this.currentChar !== '\n') {
                    this.advance();
                }
                this.skipWhitespace();
                this.skipComment();
            } else if (this.currentChar === '*') {
                while (this.currentChar) {
                    this.advance();
                    if (this.currentChar === '*') {
                        this.advance();
                        if (this.currentChar === '/') {
                            this.advance();
                            break;
                        }
                    }
                }
                this.skipWhitespace();
                this.skipComment();
            }
        }
    }

    number() {
        let result = '';
        let type = TokenType.INTEGER_LITERAL;

        while (this.currentChar && /[0-9]/.test(this.currentChar)) {
            result += this.currentChar;
            this.advance();
        }

        if (this.currentChar === 'x') {
            result += this.currentChar;
            this.advance();
            type = TokenType.HEXADECIMAL_LITERAL;
            while (this.currentChar && /[0-9a-fA-F]/.test(this.currentChar)) {
                result += this.currentChar;
                this.advance();
            }
        } else if (this.currentChar === '.') {
            result += this.currentChar;
            this.advance();
            type = TokenType.FLOAT_LITERAL;
            while (this.currentChar && /[0-9]/.test(this.currentChar)) {
                result += this.currentChar;
                this.advance();
            }
            if (this.currentChar === 'e' || this.currentChar === 'E') {
                result += this.currentChar;
                this.advance();
                if (this.currentChar === '+' || this.currentChar === '-') {
                    result += this.currentChar;
                    this.advance();
                }
                while (this.currentChar && /[0-9]/.test(this.currentChar)) {
                    result += this.currentChar;
                    this.advance();
                }

                if (type === TokenType.INTEGER_LITERAL && /[.eE]/.test(this.currentChar)) {
                    type = TokenType.FLOAT_LITERAL;
                }

                if (type === TokenType.FLOAT_LITERAL && !/[.eE]/.test(this.currentChar)) {
                    return new mlir.Token(type, parseFloat(result));
                }

                if (type === TokenType.HEXADECIMAL_LITERAL && !/[x]/.test(this.currentChar)) {
                    return new mlir.Token(type, parseInt(result, 16));
                }

                return new mlir.Token(type, result);
            }
        }

        return new mlir.Token(type, parseInt(result, 10));
    }

    stringLiteral() {
        let result = '';
        this.advance();

        while (this.currentChar && this.currentChar !== '"') {
            if (this.currentChar === '\\') {
                this.advance();
                switch (this.currentChar) {
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
                        result += this.currentChar;
                        break;
                }
            } else {
                result += this.currentChar;
            }
            this.advance();
        }

        if (this.currentChar === '"') {
            this.advance();
            return new mlir.Token(TokenType.STRING_LITERAL, result);
        }

        throw new Error('Unterminated string literal');
    }

    identifier() {
        let result = '';
        let opened = 0;
        let wasOpened = false;
        while (true) {
            if (!opened) {
                if (this.currentChar &&
                    (/[a-zA-Z_$<>\-.\*]/.test(this.currentChar) ||
                        /[0-9]/.test(this.currentChar))) {
                    if (this.currentChar === '<') {
                        opened += 1;
                        wasOpened = true;
                    }
                    result += this.currentChar;
                    this.advance();
                } else {
                    break;
                }
            } else { // opened
                if (!this.currentChar) {
                    break;
                } else if (this.currentChar === '>') {
                    result += this.currentChar;
                    this.advance();
                    opened -= 1;
                    if (opened === 0) {
                        break;
                    }
                } else if (this.currentChar !== '>') {
                    if (this.currentChar === '<') {
                        opened += 1;
                    }
                    result += this.currentChar;
                    this.advance();
                }
            }
        }

        if (wasOpened) {
            if (result.startsWith('dense')) {
                return new mlir.Token(TokenType.DENSE, result);
            }
            return new mlir.Token(TokenType.TYPE, result);

        }

        if (result.endsWith('func')) {
            return new mlir.Token(TokenType.KEYWORD, result);
        }

        switch (result) {
            case 'module':
            case 'func':
            case 'loc':
                return new mlir.Token(TokenType.KEYWORD, result);
            case 'true':
            case 'false':
                return new mlir.Token(TokenType.BOOLEAN_LITERAL, result === 'true');
            default:
                return new mlir.Token(TokenType.IDENTIFIER, result);
        }
    }


    symbolRefId() {
        let result = '@';
        this.advance();
        if (this.currentChar === '"') {
            result += this.stringLiteral().value;
        } else {
            while (
                this.currentChar &&
                (/[a-zA-Z_$]/.test(this.currentChar) ||
                    /[0-9]/.test(this.currentChar) ||
                    /[-.]/.test(this.currentChar))
            ) {
                result += this.currentChar;
                this.advance();
            }
            if (this.currentChar === ':' && this.peek() === ':') {
                result += this.currentChar;
                this.advance();
                result += this.currentChar;
                this.advance();
                result += this.symbolRefId().value;
            }
        }
        return new mlir.Token(TokenType.SYMBOL_REF_ID, result);
    }

    valueId() {
        let result = '';
        if (this.currentChar === '%') {
            result = '%';
        } else if (this.currentChar === '$') {
            result = '$';
        }
        this.advance();
        while (
            this.currentChar &&
            (/[a-zA-Z_$]/.test(this.currentChar) ||
                /[0-9]/.test(this.currentChar) ||
                /[-.]/.test(this.currentChar))
        ) {
            result += this.currentChar;
            this.advance();
        }
        return new mlir.Token(TokenType.VALUE_ID, result);
    }

    caretId() {
        let result = '^';
        this.advance();

        if (this.currentChar === ':' && this.peek() !== ':') {
            result += this.currentChar;
            this.advance();
            return new mlir.Token(TokenType.CARET_ID, result);
        }

        while (
            this.currentChar &&
            (/[a-zA-Z_$]/.test(this.currentChar) ||
                /[0-9]/.test(this.currentChar) ||
                /[-.]/.test(this.currentChar))
        ) {
            result += this.currentChar;
            this.advance();
        }

        if (this.currentChar === ':' && this.peek() === ':') {
            result += this.currentChar;
            this.advance();
            result += this.currentChar;
            this.advance();
            result += this.caretId().value;
        }

        return new mlir.Token(TokenType.CARET_ID, result);
    }

    numberOrShape() {
        let result = '';
        const type = TokenType.INTEGER_LITERAL;

        while (this.currentChar && /[0-9]/.test(this.currentChar)) {
            result += this.currentChar;
            this.advance();
        }

        if (this.currentChar === 'x') {
            // Read the rest of the shape
            do {
                result += this.currentChar;
                this.advance();
            } while (this.currentChar && /[0-9x]/.test(this.currentChar));
            return new mlir.Token(TokenType.SHAPE, result);
        }

        return new mlir.Token(type, parseInt(result, 10));
    }

    nextToken() {
        while (this.currentChar) {
            if (this.currentChar === ' ' || this.currentChar === '\t' || this.currentChar === '\n' || this.currentChar === '\r' || this.currentChar === '\f') {
                this.skipWhitespace();
                continue;
            }
            if (this.currentChar === '/') {
                this.skipComment();
                continue;
            }
            if (/[0-9]/.test(this.currentChar)) {
                return this.numberOrShape();
            }
            if (this.currentChar === '.') {
                if (/[0-9]/.test(this.peek())) {
                    return this.number();
                }
                return new mlir.Token(TokenType.KEYWORD, '.');
            }
            if (this.currentChar === '-') {
                if (/[0-9]/.test(this.peek())) {
                    return this.number();
                } else if (this.peek() === '>') {
                    this.advance();
                    this.advance();
                    return new mlir.Token(TokenType.ARROW, '->');
                }
                this.advance();
                return new mlir.Token(TokenType.KEYWORD, '-');
            }
            if (this.currentChar === '+') {
                if (/[0-9]/.test(this.peek())) {
                    return this.number();
                }
                this.advance();
                return new mlir.Token(TokenType.KEYWORD, '+');
            }
            if (this.currentChar === '"') {
                return this.stringLiteral();
            }
            if (
                /[a-zA-Z_$]/.test(this.currentChar) ||
                /[-.]/.test(this.currentChar)
            ) {
                return this.identifier();
            }
            if (this.currentChar === '@') {
                return this.symbolRefId();
            }
            if (this.currentChar === '%') {
                return this.valueId();
            }
            if (this.currentChar === '^') {
                return this.caretId();
            }
            if (this.currentChar === '=') {
                if (this.peek() === '=') {
                    this.advance();
                    this.advance();
                    return new mlir.Token(TokenType.EQUAL_EQUAL, '==');
                }
                this.advance();
                return new mlir.Token(TokenType.EQUAL, '=');

            }
            if (this.currentChar === ':') {
                if (this.peek() === ':') {
                    this.advance();
                    this.advance();
                    return new mlir.Token(TokenType.DOUBLE_COLON, '::');
                }
                this.advance();
                return new mlir.Token(TokenType.COLON, ':');
            }
            if (this.currentChar === ',') {
                this.advance();
                return new mlir.Token(TokenType.COMMA, ',');
            }
            if (this.currentChar === '(') {
                this.advance();
                return new mlir.Token(TokenType.LPAREN, '(');
            }
            if (this.currentChar === ')') {
                this.advance();
                return new mlir.Token(TokenType.RPAREN, ')');
            }
            if (this.currentChar === '{') {
                this.advance();
                return new mlir.Token(TokenType.LBRACE, '{');
            }
            if (this.currentChar === '}') {
                this.advance();
                return new mlir.Token(TokenType.RBRACE, '}');
            }
            if (this.currentChar === '[') {
                this.advance();
                return new mlir.Token(TokenType.LBRACKET, '[');
            }
            if (this.currentChar === ']') {
                this.advance();
                return new mlir.Token(TokenType.RBRACKET, ']');
            }
            if (this.currentChar === '<') {
                this.advance();
                return new mlir.Token(TokenType.LESS_THAN, '<');
            }
            if (this.currentChar === '>') {
                this.advance();
                return new mlir.Token(TokenType.GREATER_THAN, '>');
            }

            const result = this.currentChar;
            this.advance();
            return new mlir.Token(TokenType.KEYWORD, result);
        }

        return new mlir.Token(TokenType.EOF, null);
    }
};

mlir.Parser = class {

    constructor(decoder) {
        this.tokenizer = new mlir.Tokenizer(decoder);
        this.currentToken = this.tokenizer.nextToken();
    }

    read() {
        this.consumeToken(TokenType.KEYWORD, 'module');

        let attributes = {};
        // Attributes
        if (this.currentToken.value === 'attributes') {
            this.consumeToken(TokenType.IDENTIFIER, 'attributes');
            attributes = Object.assign(attributes, this.parseAttribute());
        }

        this.consumeToken(TokenType.LBRACE);

        const graph = {
            functions: [],
            operations: [],
            attributes: attributes,
        };

        // functions or operations
        while (this.currentToken.type !== TokenType.RBRACE) {
            if (this.currentToken.type === TokenType.KEYWORD && this.currentToken.value.endsWith('func')) {
                // function
                const func = this.parseFunction();
                graph.functions.push(func);
            } else {
                // operation
                const op = this.parseOperation();
                graph.operations.push(op);
            }
        }

        this.consumeToken(TokenType.RBRACE);

        return graph;
    }

    parseFunction() {
        // func keyword
        this.consumeToken(TokenType.KEYWORD);

        const name = this.parseFunctionName();

        const inputs = this.parseFunctionInputs();

        let attributes = {};

        // Attributes
        if (this.currentToken.value === 'attributes') {
            this.consumeToken(TokenType.IDENTIFIER, 'attributes');
            attributes = Object.assign(attributes, this.parseAttribute());
        }

        let outputs = {};

        if (this.currentToken.type === TokenType.ARROW) {
            outputs = Object.assign(outputs, this.parseFunctionOutputs());
        }

        // Attributes
        if (this.currentToken.value === 'attributes') {
            this.consumeToken(TokenType.IDENTIFIER, 'attributes');
            attributes = Object.assign(attributes, this.parseAttribute());
        }

        this.consumeToken(TokenType.LBRACE);

        // Operations
        const operations = [];
        while (this.currentToken.type !== TokenType.RBRACE) {
            const operation = this.parseOperation();
            operations.push(operation);
        }

        this.consumeToken(TokenType.RBRACE);

        return {
            name: name,
            inputs: inputs.map (input => input.name),
            inputTypes: inputs.map (input => input.type),
            outputTypes: outputs,
            operations: operations,
        };
    }

    parseFunctionName() {
        const name = this.currentToken.value;
        this.consumeToken(TokenType.SYMBOL_REF_ID);
        return name;
    }

    parseFunctionInputs() {
        this.consumeToken(TokenType.LPAREN);
        const inputs = [];
        while (this.currentToken.type !== TokenType.RPAREN) {
            const input = {
                name: this.currentToken.value,
            };

            this.consumeToken(TokenType.VALUE_ID);
            this.consumeToken(TokenType.COLON);
            input.type = this.currentToken.value;
            if (this.currentToken.type === TokenType.TYPE) {
                this.consumeToken(TokenType.TYPE);
            } else if (this.currentToken.type === TokenType.IDENTIFIER) {
                this.consumeToken(TokenType.IDENTIFIER);
            }

            // attribute
            if (this.currentToken.type === TokenType.LBRACE) {
                input.attributes = this.parseAttribute();
            }
            inputs.push(input);

            if (this.currentToken.type === TokenType.COMMA) {
                this.consumeToken(TokenType.COMMA);
            }
        }
        this.consumeToken(TokenType.RPAREN);
        return inputs;
    }

    parseFunctionOutputs() {
        this.consumeToken(TokenType.ARROW);
        const outputs = [];

        if (this.currentToken.type === TokenType.LPAREN) {
            this.consumeToken(TokenType.LPAREN);
            while (this.currentToken.type !== TokenType.RPAREN) {
                const output = {
                    type: this.currentToken.value,
                };
                if (this.currentToken.type === TokenType.TYPE) {
                    this.consumeToken(TokenType.TYPE);
                } else if (this.currentToken.type === TokenType.IDENTIFIER) {
                    this.consumeToken(TokenType.IDENTIFIER);
                }

                // attribute
                if (this.currentToken.type === TokenType.LBRACE) {
                    output.attributes = this.parseAttribute();
                }
                outputs.push(output);

                if (this.currentToken.type === TokenType.COMMA) {
                    this.consumeToken(TokenType.COMMA);
                }
            }
            this.consumeToken(TokenType.RPAREN);
        } else {
            const output = {
                type: this.currentToken.value,
            };
            if (this.currentToken.type === TokenType.TYPE) {
                this.consumeToken(TokenType.TYPE);
            } else if (this.currentToken.type === TokenType.IDENTIFIER) {
                this.consumeToken(TokenType.IDENTIFIER);
            }

            outputs.push(output);
        }

        return outputs;
    }


    parseOperationName() {
        let operationName;

        if (this.currentToken.type === TokenType.STRING_LITERAL) {
            operationName = this.currentToken.value;
            this.consumeToken(TokenType.STRING_LITERAL);
        } else if (this.currentToken.type === TokenType.IDENTIFIER) {
            operationName = this.currentToken.value;
            this.consumeToken(TokenType.IDENTIFIER);
            if (this.currentToken.type === TokenType.IDENTIFIER) {
                operationName += this.currentToken.value;
                this.consumeToken(TokenType.IDENTIFIER);
            }
        } else {
            throw new Error(`Unexpected token for operation name: ${JSON.stringify(this.currentToken)}`);
        }

        return operationName;
    }

    parseInputArguments() {
        const inputs = [];

        if (this.currentToken.type === TokenType.LPAREN) {
            this.consumeToken(TokenType.LPAREN);
        }

        const validTerminatingTokens = [
            TokenType.RPAREN,
            TokenType.COLON,
            TokenType.ARROW,
            TokenType.LBRACE,
            TokenType.IDENTIFIER,
            TokenType.STRING_LITERAL
        ];

        while (!validTerminatingTokens.includes(this.currentToken.type)) {
            if (this.currentToken.type === TokenType.VALUE_ID) {
                inputs.push(this.currentToken.value);
                this.consumeToken(TokenType.VALUE_ID);
            } else if (this.currentToken.type === TokenType.DENSE) {
                inputs.push(this.currentToken.value);
                this.consumeToken(TokenType.DENSE);
                return { inputs };
            }

            if (this.currentToken.type === TokenType.COMMA) {
                this.consumeToken(TokenType.COMMA);
            }
        }

        if (this.currentToken.type === TokenType.RPAREN) {
            this.consumeToken(TokenType.RPAREN);
        }

        return { inputs };
    }

    parseInputArgumentTypes() {
        const inputTypes = [];

        if (this.currentToken.type === TokenType.LPAREN) {
            this.consumeToken(TokenType.LPAREN);
        }

        while (this.currentToken.type === TokenType.TYPE || (this.currentToken.type === TokenType.IDENTIFIER && this.currentToken.value === 'none')) {
            inputTypes.push(this.currentToken.value);
            this.consumeToken(this.currentToken.type);
            if (this.currentToken.type === TokenType.COMMA) {
                this.consumeToken(TokenType.COMMA);
            }
        }

        if (this.currentToken.type === TokenType.RPAREN) {
            this.consumeToken(TokenType.RPAREN);
        }

        return { inputTypes };
    }


    parseOutputArguments() {
        const outputs = [];
        const outputTypes = [];

        this.consumeToken(TokenType.LPAREN);

        while (this.currentToken.type !== TokenType.RPAREN) {
            if (this.currentToken.type === TokenType.VALUE_ID) {
                outputs.push(this.currentToken.value);
                this.consumeToken(TokenType.VALUE_ID);
            }

            if (this.currentToken.type === TokenType.COLON) {
                this.consumeToken(TokenType.COLON);
                outputTypes.push(this.currentToken.value);
                this.consumeToken(TokenType.TYPE);
            }

            if (this.currentToken.type === TokenType.COMMA) {
                this.consumeToken(TokenType.COMMA);
            }
        }

        this.consumeToken(TokenType.RPAREN);

        return { outputs, outputTypes };
    }

    parseOutputType() {
        const outputTypes = [];

        if (this.currentToken.type === TokenType.LPAREN) {
            this.consumeToken(TokenType.LPAREN);

            while (this.currentToken.type !== TokenType.RPAREN) {
                outputTypes.push(this.currentToken.value);
                if (this.currentToken.type === TokenType.TYPE) {
                    this.consumeToken(TokenType.TYPE);
                } else if (this.currentToken.type === TokenType.IDENTIFIER && this.currentToken.value === 'none') {
                    this.consumeToken(TokenType.IDENTIFIER);
                }

                if (this.currentToken.type === TokenType.COMMA) {
                    this.consumeToken(TokenType.COMMA);
                }
            }

            this.consumeToken(TokenType.RPAREN);
        } else {
            outputTypes.push(this.currentToken.value);
            if (this.currentToken.type === TokenType.TYPE) {
                this.consumeToken(TokenType.TYPE);
            } else if (this.currentToken.type === TokenType.IDENTIFIER && this.currentToken.value === 'none') {
                this.consumeToken(TokenType.IDENTIFIER);
            }
        }

        return outputTypes;
    }

    parseOperationBody() {
        let bodyContent = '';
        let braceCount = 0;

        this.consumeToken(TokenType.LBRACE);
        braceCount++;
        bodyContent += '{ ';

        while (braceCount > 0) {
            if (this.currentToken.type === TokenType.LBRACE) {
                braceCount++;
            } else if (this.currentToken.type === TokenType.RBRACE) {
                braceCount--;
            }

            if (braceCount > 0) {
                bodyContent += this.currentToken.value;
                if (this.currentToken.type === TokenType.LBRACE || this.currentToken.type === TokenType.RBRACE) {
                    bodyContent += '\n';
                } else if (this.currentToken.type !== TokenType.WHITESPACE) {
                    bodyContent += ' ';
                }
            }

            this.consumeToken(this.currentToken.type);
        }

        bodyContent += '}';

        return bodyContent;
    }

    parseReturnValues() {
        const outputs = [];

        if (this.currentToken.type === TokenType.LPAREN) {
            this.consumeToken(TokenType.LPAREN);

            while (this.currentToken.type !== TokenType.RPAREN) {
                if (this.currentToken.type === TokenType.VALUE_ID) {
                    outputs.push(this.currentToken.value);
                    this.consumeToken(TokenType.VALUE_ID);
                }

                if (this.currentToken.type === TokenType.COMMA) {
                    this.consumeToken(TokenType.COMMA);
                }
            }

            this.consumeToken(TokenType.RPAREN);
        } else if (this.currentToken.type === TokenType.VALUE_ID) {
            outputs.push(this.currentToken.value);
            this.consumeToken(TokenType.VALUE_ID);

            if (this.currentToken.type === TokenType.COMMA) {
                this.consumeToken(TokenType.COMMA);

                while (this.currentToken.type === TokenType.VALUE_ID) {
                    outputs.push(this.currentToken.value);
                    this.consumeToken(TokenType.VALUE_ID);

                    if (this.currentToken.type === TokenType.COMMA) {
                        this.consumeToken(TokenType.COMMA);
                    }
                }
            }
        }

        return outputs;
    }

    parseAttribute() {
        const attributes = {};
        if (this.currentToken.type !== TokenType.LBRACE) {
            return attributes;
        }
        this.consumeToken(TokenType.LBRACE);


        while (this.currentToken.type !== TokenType.RBRACE) {
            if (this.currentToken.type === TokenType.IDENTIFIER) {
                const attributeName = this.currentToken.value;
                this.consumeToken(TokenType.IDENTIFIER);

                if (this.currentToken.type === TokenType.EQUAL) {
                    this.consumeToken(TokenType.EQUAL);
                    const attributeValue = this.parseAttributeValue();
                    attributes[attributeName] = attributeValue;
                } else {
                    attributes[attributeName] = attributeName;
                }

                if (this.currentToken.type === TokenType.COMMA) {
                    this.consumeToken(TokenType.COMMA);
                }

            } else {
                throw new Error(`Unexpected token '${this.currentToken.value}' when parsing operation attribute: ${JSON.stringify(this.currentToken)}`);
            }
        }

        this.consumeToken(TokenType.RBRACE);

        return attributes;
    }

    parseAttributeValue() {
        let value = '';

        let openingCount = 0;
        const openingChars = [TokenType.LBRACKET, TokenType.LBRACE, TokenType.LPAREN];
        const closingChars = [TokenType.RBRACKET, TokenType.RBRACE, TokenType.RPAREN];

        while (
            !(openingCount === 0 && (this.currentToken.type === TokenType.COMMA || this.currentToken.type === TokenType.RBRACE))
        ) {
            if (openingChars.includes(this.currentToken.type)) {
                openingCount++;
            } else if (closingChars.includes(this.currentToken.type)) {
                openingCount--;
            }

            value += this.currentToken.value + ' ';
            this.consumeToken(this.currentToken.type);
        }

        return value.trim();
    }

    parseOperation() {
        // %3
        const outputs = this.parseReturnValues();
        // =
        if (this.currentToken.type == TokenType.EQUAL) {
            this.consumeToken(TokenType.EQUAL);
        }
        // "add"
        const operationName = this.parseOperationName();
        // (%a, %b)
        const { inputs } = this.parseInputArguments();

        // TODO: parsing ^bb
        if (this.currentToken.type === TokenType.LPAREN) {
            this.consumeToken(TokenType.LPAREN);
            let count = 1;
            while (count > 0) {
                if (this.currentToken.type === TokenType.LPAREN) {
                    count++;
                } else if (this.currentToken.type === TokenType.RPAREN) {
                    count--;
                }
                this.consumeToken(this.currentToken.type);
            }
        }

        // : (f32, tensor<1xf32>)
        let inputTypes = [];
        let attributes = {};

        attributes = Object.assign(attributes, this.parseAttribute());
        if (this.currentToken.type === TokenType.COLON) {
            this.consumeToken(TokenType.COLON);
            ({ inputTypes } = this.parseInputArgumentTypes());
        }

        const outputTypes = [];
        if (operationName.endsWith('constant') && this.currentToken.type !== TokenType.ARROW) {
            // constant
            const result = {
                name: operationName,
                attributes: attributes,
                // data: this.parseConstantData(),
                outputs: outputs,
                outputTypes: outputTypes,
                isConstant: true,
            };

            return result;
        }
        // -> f32
        if (this.currentToken.type === TokenType.ARROW) {
            this.consumeToken(TokenType.ARROW);
            outputTypes.push(...this.parseOutputType());
        }

        let body = null;
        if (this.currentToken.type === TokenType.LBRACE) {
            body = this.parseOperationBody();
        }

        attributes = Object.assign(attributes, this.parseAttribute());

        const result = {
            name: operationName,
            attributes: attributes,
            inputs: inputs,
            inputTypes: inputTypes,
            outputs: outputs,
            outputTypes: outputTypes,
            body: body,
        };

        return result;

    }

    peekNextToken() {
        const savedToken = this.currentToken;
        const nextToken = this.tokenizer.nextToken();
        this.currentToken = savedToken;
        return nextToken;
    }

    consumeToken(expectedType, expectedValue) {
        if (this.currentToken.type === expectedType) {
            if (expectedValue !== undefined && this.currentToken.value !== expectedValue) {
                throw new Error(`Expected token with value '${expectedValue}', but got '${this.currentToken.value}': ${JSON.stringify(this.currentToken)}`);
            }
            this.currentToken = this.tokenizer.nextToken();
        } else {
            throw new Error(`Expected token of type '${expectedType}', but got '${this.currentToken.type}': ${JSON.stringify(this.currentToken)}`);
        }
    }
};

mlir.Graph = class {

    constructor(func, group) {
        this._inputs = [];  // [mlir.Parameter]
        this._outputs = []; // [mlir.Parameter]
        this._nodes = [];   // [mlir.Node]

        // ---------------------------------------------------------------------
        // inputs of function
        for (let i=0; i<func.inputs.length; i++) {
            const input = func.inputs[i];
            const inputType = func.inputTypes[i];
            const type = mlir.Utility.valueType(inputType);
            const inputArgument = new mlir.Argument(input, type, "input desc", null);
            const inputParameter = new mlir.Parameter(input, true, [inputArgument]);
            this._inputs.push(inputParameter);
        }

        // outputs of function
        for (let i=0; i<func.outputTypes.length; i++) {
            const output = "%return" + "/" + i;
            const outputType = func.outputTypes[i];
            const type = mlir.Utility.valueType(outputType);
            const outputArgument = new mlir.Argument(output, type, "output desc", null);
            const outputParameter = new mlir.Parameter(output, true, [outputArgument]);
            this._outputs.push(outputParameter);
        }

        // ---------------------------------------------------------------------
        // operations
        // `args` is map of edges. `args` will be converted to mlir.Arguemnts.
        const args = new Map();
        const arg = (name) => {
            if (!args.has(name)) {
                args.set(name, { name: name, to: [], from: [] });
            }
            return args.get(name);
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

            // TODO: convert attributes to proper types
            operation.attributes = op.attributes;
            // for (const entry of Object.entries(op.attributes)) {
            //     const key = entry[0];
            //     const value = entry[1];
            //     operation.attributes[key] = convertValue(value);
            // }

            for (let j=0; j<(op.inputs ? op.inputs.length : 0); j++) {
                const input = op.inputs[j];
                const inputType = op.inputTypes[j];

                const value = arg(input);
                value.to.push(operation);
                const args =  [ { name: input, value: inputType } ];

                operation.inputs.push({
                    name: input,
                    arguments: args
                });
            }

            for (let j=0; j<(op.outputs ? op.outputs.length : 0); j++) {
                const output = op.outputs[j];
                const outputType = op.outputTypes[j];

                const value = arg(output);
                value.type = mlir.Utility.valueType(outputType);
                value.from.push(operation);

                operation.outputs.push({
                    name: output,
                    arguments: [ value ]
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
                tensors.set(arg.name, new mlir.Argument(arg.name, arg.type, null, arg.value));
            }
            return tensors.get(arg.name);
        };
        for (const input of this._inputs) {
            for (const arg of input.arguments) {
                tensors.set(arg.name, arg);
            }
        }
        for (const output of this._outputs) {
            for (const arg of output.arguments) {
                tensors.set(arg.name, arg);
            }
        }

        for (const op of operations) {
            if (op.delete) {
                continue;
            }
            op.inputs  = op.inputs.map((input)  => new mlir.Parameter(input.name,  true, input.arguments.map((argument)  => tensor(argument))));
            op.outputs = op.outputs.map((output) => new mlir.Parameter(output.name, true, output.arguments.map((argument) => tensor(argument))));
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
            this._nodes.push(node);
        }
    }

    get inputs() {
        return this._inputs;
    }

    get outputs() {
        return this._outputs;
    }

    get nodes() {
        return this._nodes;
    }
};

mlir.Parameter = class {

    constructor(name, visible, args) {
        this._name = name;       // string
        this._visible = visible; // bool
        this._arguments = args;  // [mlir.Argument]
    }

    get name() {
        return this._name;
    }

    get visible() {
        return this._visible == false ? false : true;
    }

    get arguments() {
        return this._arguments;
    }
};

mlir.Argument = class {

    constructor(name, type, description, initializer) {
        if (typeof name !== 'string') {
            throw new mlir.Error("Invalid argument identifier '" + JSON.stringify(name) + "'.");
        }
        this._name = name;          // string
        this._type = type || null;  // mlir.TensorType
        this._description = description || null;
        this._initializer = initializer || null;
    }

    get name() {
        return this._name;
    }

    set name(value) {
        this._name = value;
    }

    get type() {
        if (this._initializer) {
            return this._initializer.type;
        }
        return this._type;
    }

    set type(value) {
        this._type = value;
    }

    get description() {
        return this._description;
    }

    set description(value) {
        this._description = value;
    }

    get quantization() {
        if (this._initializer) {
            return this._initializer.quantization;
        }
        return null;
    }

    get initializer() {
        return this._initializer;
    }
};

mlir.Node = class {

    constructor(group, type, name, description, attributes, inputs, outputs) {
        if (!type) {
            throw new Error('Undefined node type.');
        }
        if (group) {
            this._group = group;
        }
        this._type = { name: type };             // string (metadata.type(type) || { name: type }
        this._name = name || '';                // string
        this._description = description || '';  // string
        this._inputs = inputs;                  // [mlir.Parameter]
        this._outputs = outputs;                // [mlir.Parameter]
        this._attributes = [];                  // [mlir.Attribute]
        if (attributes) {
            for (const key of Object.keys(attributes)) {
                const schema = {}; // metadata.attribute(type, key);
                const value = attributes[key];
                const attribute = new mlir.Attribute(key, value);
                this._attributes.push(attribute);
            }
        }
    }

    get type() {
        return this._type;
    }

    get name() {
        return this._name;
    }

    get description() {
        return this._description;
    }

    get metadata() {
        return this._metadata;
    }

    get group() {
        return this._group ? this._group : null;
    }

    get inputs() {
        return this._inputs;
    }

    get outputs() {
        return this._outputs;
    }

    get attributes() {
        return this._attributes;
    }
};

mlir.Attribute = class {

    constructor(name, value) {
        this._name = name;
        this._type = 'string';
        this._value = value;
        this._visible = true;
    }

    get name() {
        return this._name;
    }

    get type() {
        return this._type;
    }

    get value() {
        return this._value;
    }

    get visible() {
        return this._visible == false ? false : true;
    }
};

mlir.Tensor = class {

    constructor(type, data) {
        this._type = type;  // mlir.TensorType
        this._data = data;
    }

    get type() {
        return this._type;
    }

    get quantization() {
        // TODO
        return null;
    }

    get layout() {
        switch (this._type.dataType) {
            case 'float32': return '|';
            default: return '<';
        }
    }

    get values() {
        return this._data;
    }
};

mlir.TensorType = class {

    constructor(dataType, shape) {
        this._dataType = dataType;                       // string
        this._shape = shape || new mlir.TensorShape([]); // mlir.TensorShape
    }

    get dataType() {
        return this._dataType;
    }

    get shape() {
        return this._shape;
    }

    toString() {
        return this.dataType + this._shape.toString();
    }
};

mlir.TensorShape = class {

    constructor(dimensions) {
        this._dimensions = dimensions;
    }

    get dimensions() {
        return this._dimensions;
    }

    toString() {
        if (!this._dimensions || this._dimensions.length == 0) {
            return '';
        }
        return '[' + this._dimensions.map((dimension) => dimension.toString()).join(',') + ']';
    }
};


mlir.Utility = class {

    static valueType(typeString) {
        if (typeString === undefined) {
            return null;
        }

        // eg. tensor<?x3x2x2xf32>
        if (typeString.startsWith('tensor<')) {
            const shapeString = typeString.substring(7, typeString.length - 1);
            if (!/^[0-9xfiq?*]+$/i.test(shapeString)) {
                return typeString;
            }
            const parts = shapeString.split('x');
            const dataType = parts[parts.length - 1];
            const shape = parts
                .slice(0, -1)
                .map((dimension) => {
                    const parsedDimension = parseInt(dimension.trim());
                    return isNaN(parsedDimension) ? '?' : parsedDimension;
                });
            return new mlir.TensorType(dataType, new mlir.TensorShape(shape));
        }
        return typeString;
    }
};

mlir.Error = class extends Error {

    constructor(message) {
        super(message);
        this.name = 'Error loading MLIR model.';
    }
};

if (typeof module !== 'undefined' && typeof module.exports === 'object') {
    module.exports.ModelFactory = mlir.ModelFactory;
}
