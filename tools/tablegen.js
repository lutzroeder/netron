
import * as fs from 'fs/promises';
import * as path from 'path';

const tablegen = {};

tablegen.Location = class {

    constructor(file, line, column) {
        this.file = file;
        this.line = line;
        this.column = column;
    }

    toString() {
        return `${this.file}:${this.line}:${this.column}`;
    }
};

tablegen.Token = class {

    constructor(type, value, location) {
        this.type = type;
        this.value = value;
        this.location = location;
    }
};

tablegen.Tokenizer = class {

    constructor(text, filename) {
        this._text = text;
        this._filename = filename;
        this._position = 0;
        this._line = 1;
        this._column = 1;
        this._token = this._tokenize();
    }

    location() {
        return new tablegen.Location(this._filename, this._line, this._column);
    }

    current() {
        return this._token;
    }

    peek(offset = 0) {
        const pos = this._position + offset;
        return pos < this._text.length ? this._text[pos] : null;
    }

    read() {
        const prev = this._token;
        this._token = this._tokenize();
        return prev;
    }

    _tokenize() {
        this._skipWhitespace();
        if (this._position >= this._text.length) {
            return new tablegen.Token('eof', null, this.location());
        }
        const location = this.location();
        const c = this.peek();
        if (c === '"') {
            return this._readString(location);
        }
        if (c === '[' && this.peek(1) === '{') {
            return this._readCodeBlock(location);
        }
        if (c === ':' && this.peek(1) === ':') {
            this._next();
            this._next();
            return new tablegen.Token('::', '::', location);
        }
        if (c === '.' && this.peek(1) === '.' && this.peek(2) === '.') {
            this._next();
            this._next();
            this._next();
            return new tablegen.Token('...', '...', location);
        }
        if (c === '-' && this.peek(1) === '>') {
            this._next();
            this._next();
            return new tablegen.Token('->', '->', location);
        }
        if (c === '=' && this.peek(1) === '=') {
            this._next();
            this._next();
            return new tablegen.Token('==', '==', location);
        }
        if (c === '!' && this.peek(1) === '=') {
            this._next();
            this._next();
            return new tablegen.Token('!=', '!=', location);
        }
        if (c === '<' && this.peek(1) === '<') {
            this._next();
            this._next();
            return new tablegen.Token('<<', '<<', location);
        }
        if (this._isDigit(c) || (c === '-' && this._isDigit(this.peek(1)))) {
            return this._readNumber(location);
        }
        if (this._isIdentifierStart(c)) {
            return this._readIdentifier(location);
        }
        if ('{}[]()<>,;:=!#$.?^'.includes(c)) {
            this._next();
            return new tablegen.Token(c, c, location);
        }
        throw new tablegen.Error(`Unexpected character '${c}' at ${location}`);
    }

    _next() {
        if (this._position < this._text.length) {
            if (this._text[this._position] === '\n') {
                this._line++;
                this._column = 1;
            } else {
                this._column++;
            }
            this._position++;
        }
    }

    _skipWhitespace() {
        while (this._position < this._text.length) {
            const c = this.peek();
            if (/\s/.test(c)) {
                this._next();
                continue;
            }
            if (c === '/' && this.peek(1) === '/') {
                this._next();
                this._next();
                while (this._position < this._text.length && this.peek() !== '\n') {
                    this._next();
                }
                continue;
            }
            if (c === '/' && this.peek(1) === '*') {
                this._next();
                this._next();
                while (this._position < this._text.length) {
                    if (this.peek() === '*' && this.peek(1) === '/') {
                        this._next();
                        this._next();
                        break;
                    }
                    this._next();
                }
                continue;
            }
            break;
        }
    }

    _isDigit(c) {
        return c !== null && /[0-9]/.test(c);
    }

    _isIdentifierStart(c) {
        return c !== null && /[a-zA-Z_]/.test(c);
    }

    _isIdentifierChar(c) {
        return c !== null && /[a-zA-Z0-9_]/.test(c);
    }

    _readString(location) {
        let value = '';
        this._next(); // opening "
        while (this._position < this._text.length && this.peek() !== '"') {
            if (this.peek() === '\\') {
                this._next();
                const c = this.peek();
                switch (c) {
                    case 'n': value += '\n'; break;
                    case 't': value += '\t'; break;
                    case 'r': value += '\r'; break;
                    case '\\': value += '\\'; break;
                    case '"': value += '"'; break;
                    default: value += c; break;
                }
                this._next();
            } else {
                value += this.peek();
                this._next();
            }
        }
        if (this.peek() === '"') {
            this._next();
        }
        return new tablegen.Token('string', value, location);
    }

    _readCodeBlock(location) {
        let value = '';
        this._next(); // [
        this._next(); // {

        // Track nested [{ }] blocks
        let depth = 1;
        while (this._position < this._text.length && depth > 0) {
            const c = this.peek();
            const next = this.peek(1);

            // Check for nested [{ to increase depth
            if (c === '[' && next === '{') {
                depth++;
                value += c;
                this._next();
                value += next;
                this._next();
            } else if (c === '}' && next === ']') {
                depth--;
                if (depth === 0) {
                    this._next(); // consume }
                    this._next(); // consume ]
                    break;
                }
                value += c;
                this._next();
                value += next;
                this._next();
            } else {
                value += c;
                this._next();
            }
        }
        return new tablegen.Token('code', value.trim(), location);
    }

    _readNumber(location) {
        let value = '';
        if (this.peek() === '-') {
            value += this.peek();
            this._next();
        }
        // Hexadecimal
        if (this.peek() === '0' && this.peek(1) === 'x') {
            value += this.peek();
            this._next();
            value += this.peek();
            this._next();
            while (this._position < this._text.length && /[0-9a-fA-F]/.test(this.peek())) {
                value += this.peek();
                this._next();
            }
            return new tablegen.Token('number', parseInt(value, 16), location);
        }
        // Binary
        if (this.peek() === '0' && this.peek(1) === 'b') {
            value += this.peek();
            this._next();
            value += this.peek();
            this._next();
            while (this._position < this._text.length && /[01]/.test(this.peek())) {
                value += this.peek();
                this._next();
            }
            return new tablegen.Token('number', parseInt(value.substring(2), 2), location);
        }
        // Decimal
        while (this._position < this._text.length && this._isDigit(this.peek())) {
            value += this.peek();
            this._next();
        }
        return new tablegen.Token('number', parseInt(value, 10), location);
    }

    _readIdentifier(location) {
        let value = '';
        while (this._position < this._text.length && this._isIdentifierChar(this.peek())) {
            value += this.peek();
            this._next();
        }
        // Handle member access with dots (e.g., ElementwiseMappable.traits)
        while (this.peek() === '.' && this._isIdentifierStart(this.peek(1))) {
            value += this.peek(); // add dot
            this._next();
            while (this._position < this._text.length && this._isIdentifierChar(this.peek())) {
                value += this.peek();
                this._next();
            }
        }
        const keywords = [
            'assert', 'bit', 'bits', 'class', 'code', 'dag', 'def', 'defm',
            'defset', 'defvar', 'dump', 'else', 'false', 'field', 'foreach',
            'if', 'in', 'include', 'int', 'let', 'list', 'multiclass',
            'string', 'then', 'true'
        ];
        const type = keywords.includes(value) ? value : 'id';
        const tokenValue = (type === 'true' || type === 'false') ? (value === 'true') : value;
        return new tablegen.Token(type, tokenValue, location);
    }
};

tablegen.Value = class {

    constructor(type, value) {
        this.type = type; // 'int', 'string', 'bit', 'bits', 'list', 'dag', 'code', 'def'
        this.value = value;
    }
};

tablegen.DAG = class {

    constructor(operator, operands) {
        this.operator = operator; // string or Value
        this.operands = operands; // array of {value, name}
    }
};

tablegen.Type = class {

    constructor(name) {
        this.name = name;
        this.args = []; // template arguments
    }

    toString() {
        if (this.args.length === 0) {
            return this.name;
        }
        return `${this.name}<${this.args.map((a) => a.toString()).join(', ')}>`;
    }
};

tablegen.Field = class {

    constructor(name, type, value) {
        this.name = name;
        this.type = type; // tablegen.Type
        this.value = value; // tablegen.Value or null
    }
};

tablegen.Record = class {

    constructor(name, parser = null) {
        this.name = name;
        this.parents = [];
        this.fields = new Map();
        this.templateArgs = [];
        this.location = null;
        this.parser = parser;
    }

    getField(name) {
        return this.fields.get(name);
    }

    hasField(name) {
        return this.fields.has(name);
    }

    resolveField(name, visited = new Set()) {
        if (!visited.has(this.name)) {
            visited.add(this.name);
            const field = this.fields.get(name);
            if (field) {
                return field;
            }
            for (const parent of this.parents) {
                const parentRecord = this.parser.classes.get(parent.name);
                if (parentRecord) {
                    const parentField = parentRecord.resolveField(name, visited);
                    if (parentField) {
                        return parentField;
                    }
                }
            }
        }
        return null;
    }

    getValueAsString(fieldName) {
        const field = this.resolveField(fieldName);
        if (!field || !field.value) {
            return null;
        }
        if (field.value.type === 'string') {
            return field.value.value.replace(/^"|"$/g, '');
        }
        return null;
    }

    getValueAsDef(fieldName) {
        const field = this.resolveField(fieldName);
        if (!field || !field.value) {
            return null;
        }
        if (field.value.type === 'def' && typeof field.value.value === 'string') {
            const defName = field.value.value;
            return this.parser.defs.get(defName) || this.parser.classes.get(defName);
        }
        return null;
    }
};

tablegen.Reader = class {

    constructor() {
        this.classes = new Map();
        this.defs = new Map();
        this.includes = new Set();
        this.paths = [];
    }

    async parse(files, paths) {
        this.paths = paths || [];
        for (const file of files) {
            // eslint-disable-next-line no-await-in-loop
            await this._parseFile(file);
        }
    }

    async access(file) {
        try {
            await fs.access(file);
            return true;
        } catch {
            // continue regardless of error
        }
        return false;
    }

    async _parseFile(file) {
        let location = null;
        for (const current of this.paths) {
            const test = path.join(current, file);
            // eslint-disable-next-line no-await-in-loop
            if (await this.access(test)) {
                location = path.resolve(test);
                break;
            }
        }
        if (!location) {
            throw new tablegen.Error(`File not found '${file}'.`);
        }
        if (!this.includes.has(location)) {
            this.includes.add(location);
            const content = await fs.readFile(location, 'utf-8');
            this._tokenizer = new tablegen.Tokenizer(content, location);
            while (!this._match('eof')) {
                switch (this._tokenizer.current().type) {
                    case 'include':
                        // eslint-disable-next-line no-await-in-loop
                        await this._parseInclude();
                        break;
                    case 'class': this._parseClass(); break;
                    case 'def': this._parseDef(); break;
                    case 'defm': this._parseDefm(); break;
                    case 'let': this._parseLet(); break;
                    case 'multiclass': this._parseMulticlass(); break;
                    case 'defvar': this._parseDefvar(); break;
                    case 'defset': this._parseDefset(); break;
                    case 'foreach': this._parseForeach(); break;
                    default:
                        this._read();
                        break;
                }
            }
        }
    }

    async _parseInclude() {
        this._read();
        const file = this._expect('string');
        const tokenizer = this._tokenizer;
        await this._parseFile(file);
        this._tokenizer = tokenizer;
    }

    _parseClass() {
        this._read();
        let name = null;
        if (this._match('id')) {
            name = this._expect('id');
        } else if (this._match('number')) {
            name = String(this._expect('number'));
        } else {
            throw new tablegen.Error(`Expected class name but got '${this._tokenizer.current().type}' at ${this._tokenizer.location()}`);
        }
        const record = new tablegen.Record(name, this);
        record.location = this._tokenizer.location();

        if (this._match('<')) {
            record.templateArgs = this._parseTemplateParams();
        }

        if (this._match(':')) {
            record.parents = this._parseParentClassList();
        }

        if (this._match('{')) {
            this._parseRecordBody(record);
        }

        this.classes.set(name, record);
    }

    _parseDef() {
        this._read();
        let name = '';
        if (this._match('id')) {
            name = this._read();
        } else if (this._match('number')) {
            name = this._read().toString();
        }
        const def = new tablegen.Record(name, this);
        def.location = this._tokenizer.location();
        if (this._match(':')) {
            def.parents = this._parseParentClassList();
        }
        if (this._match('{')) {
            this._parseRecordBody(def);
        }
        if (name) {
            this.defs.set(name, def);
        }
    }

    _parseDefm() {
        this._read();
        this._skipUntil([';', 'def', 'class', 'defm', 'let', 'multiclass']);
        this._eat(';');
    }

    _parseMulticlass() {
        this._read();
        let depth = 0;
        while (!this._match('eof')) {
            if (this._eat('{')) {
                depth++;
            } else if (this._eat('}')) {
                depth--;
                if (depth === 0) {
                    break;
                }
            } else {
                this._read();
            }
        }
    }

    _parseLet() {
        this._read();
        // Skip let statements
        this._skipUntil(['in', '{', ';']);
        this._eat('in');
        this._eat(';');
    }

    _parseDefvar() {
        this._read(); // consume 'defvar'
        this._expect('id'); // variable name
        this._expect('=');
        // Parse the value (could be complex expression)
        this._parseValue();
        this._expect(';');
        // Note: defvar defines local variables used in templates
        // We don't need to store them as they're only used during TableGen evaluation
        // Just consume the syntax for now
    }

    _parseDefset() {
        this._read();
        // Skip defset
        let depth = 0;
        while (!this._match('eof')) {
            if (this._eat('{')) {
                depth++;
            } else if (this._eat('}')) {
                depth--;
                if (depth === 0) {
                    break;
                }
            } else {
                this._read();
            }
        }
    }

    _parseForeach() {
        this._read();
        // Skip foreach
        let depth = 0;
        while (!this._match('eof')) {
            if (this._eat('{')) {
                depth++;
            } else if (this._eat('}')) {
                depth--;
                if (depth === 0) {
                    break;
                }
            } else {
                this._read();
            }
        }
    }

    _parseTemplateParams() {
        this._read(); // <
        const params = [];
        while (!this._match('>') && !this._match('eof')) {
            const type = this._parseType();
            const name = this._expect('id');
            let defaultValue = null;
            if (this._match('=')) {
                this._read();
                defaultValue = this._parseValue();
            }
            params.push({ name, type, defaultValue });
            if (this._match(',')) {
                this._read();
            }
        }
        this._expect('>');
        return params;
    }

    _parseParentClassList() {
        this._read();
        const parents = [];
        while (!this._match('{') && !this._match(';') && !this._match('eof')) {
            const parent = this._parseType();
            parents.push(parent);
            if (!this._eat(',')) {
                break;
            }
        }
        return parents;
    }

    _parseRecordBody(record) {
        this._read();
        while (!this._match('}') && !this._match('eof')) {
            if (this._match('let')) {
                this._read();
                const name = this._expect('id');
                this._expect('=');
                const value = this._parseValue();
                const field = new tablegen.Field(name, null, value);
                record.fields.set(name, field);
                this._eat(';');
            } else if (this._match('defvar')) {
                this._read();
                const name = this._expect('id');
                this._expect('=');
                const value = this._parseValue();
                const field = new tablegen.Field(name, null, value);
                record.fields.set(name, field);
                this._eat(';');
            } else if (this._match('assert')) {
                this._read();
                // Parse assert condition, message
                this._parseValue(); // condition
                this._eat(',');
                this._parseValue(); // message
                this._eat(';');
            } else if (this._match('bit') || this._match('bits') || this._match('int') ||
                       this._match('string') || this._match('list') || this._match('dag') ||
                       this._match('code') || this._match('id')) {
                const type = this._parseType();
                const name = this._expect('id');
                let value = null;
                if (this._eat('=')) {
                    value = this._parseValue();
                }
                const field = new tablegen.Field(name, type, value);
                record.fields.set(name, field);
                this._eat(';');
            } else {
                this._read();
            }
        }
        this._expect('}');
    }

    _parseType() {
        let typeName = '';

        if (this._match('bit') || this._match('bits') || this._match('int') ||
            this._match('string') || this._match('list') || this._match('dag') ||
            this._match('code')) {
            typeName = this._read();
        } else if (this._match('id')) {
            typeName = this._read();
        } else if (this._match('number')) {
            typeName = this._read().toString();
            if (this._match('id')) {
                typeName += this._read();
            }
        } else {
            throw new tablegen.Error(`Expected type at ${this._tokenizer.location()}`);
        }
        const type = new tablegen.Type(typeName);
        if (this._eat('<')) {
            let depth = 1;
            const argsTokens = [];
            while (depth > 0 && !this._match('eof')) {
                if (this._eat('<')) {
                    argsTokens.push(this._tokenizer.current());
                    depth++;
                } else if (this._eat('>')) {
                    depth--;
                    if (depth === 0) {
                        break;
                    }
                    argsTokens.push(this._tokenizer.current());
                } else {
                    argsTokens.push(this._tokenizer.current());
                    this._read();
                }
            }
            type.args = this._parseTemplateArgList(argsTokens);
        }
        return type;
    }

    _parseTemplateArgList(tokens) {
        const args = [];
        let current = '';
        for (const token of tokens) {
            if (token.type === ',') {
                if (current.trim()) {
                    args.push(current.trim());
                }
                current = '';
            } else {
                current += token.value === null ? token.type : token.value;
            }
        }
        current = current.trim();
        if (current) {
            args.push(current);
        }
        return args;
    }

    _parseValue() {
        let value = this._parsePrimaryValue();
        while (this._match('#') || (value && value.type === 'string' && this._match('string'))) {
            if (this._match('#')) {
                this._read();
            }
            const right = this._parsePrimaryValue();
            if (value && value.type === 'string' && right && right.type === 'string') {
                value = new tablegen.Value('string', value.value + right.value);
            } else {
                value = right;
            }
        }

        return value;
    }

    _parsePrimaryValue() {
        if (this._match('string')) {
            const value = this._read();
            return new tablegen.Value('string', value);
        }
        if (this._match('number')) {
            const value = this._read();
            return new tablegen.Value('int', value);
        }
        if (this._match('true') || this._match('false')) {
            const value = this._read();
            return new tablegen.Value('bit', value);
        }
        if (this._match('code')) {
            const value = this._read();
            return new tablegen.Value('code', value);
        }
        if (this._eat('[')) {
            const items = [];
            while (!this._match(']') && !this._match('eof')) {
                items.push(this._parseValue());
                this._eat(',');
            }
            this._expect(']');
            // Skip optional type annotation: []<Type>
            if (this._match('<')) {
                this._skipBalanced('<', '>');
            }
            return new tablegen.Value('list', items);
        }
        if (this._eat('(')) {
            let operator = null;
            if (this._match('id')) {
                operator = this._read();
            }
            const operands = [];
            while (!this._match(')') && !this._match('eof')) {
                const value = this._parseValue();
                let name = null;
                if (this._eat(':')) {
                    if (this._eat('$')) {
                        // Allow keywords as identifiers in this context
                        if (this._match('id') || this._isKeyword(this._tokenizer.current().type)) {
                            name = this._read();
                        }
                    }
                }
                const operand = { value, name };
                operands.push(operand);
                this._eat(',');
            }
            this._expect(')');
            const dag = new tablegen.DAG(operator, operands);
            return new tablegen.Value('dag', dag);
        }
        if (this._eat('{')) {
            // Anonymous record { field = value, ... }
            const fields = new Map();
            while (!this._match('}') && !this._match('eof')) {
                const name = this._expect('id');
                this._expect('=');
                const value = this._parseValue();
                fields.set(name, value);
                this._eat(',');
            }
            this._expect('}');
            return new tablegen.Value('record', fields);
        }
        if (this._eat('!')) {
            const keywords = [
                'assert', 'bit', 'bits', 'class', 'code', 'dag', 'def', 'defm',
                'defset', 'defvar', 'dump', 'else', 'false', 'field', 'foreach',
                'if', 'in', 'include', 'int', 'let', 'list', 'multiclass',
                'string', 'then', 'true'
            ];
            let op = null;
            if (this._match('id') || keywords.includes(this._tokenizer.current().type)) {
                op = this._read();
            } else {
                throw new tablegen.Error(`Expected operator after '!' but got '${this._tokenizer.current().type}' at ${this._tokenizer.location()}`);
            }
            if (this._match('<')) {
                this._skipBalanced('<', '>');
            }
            const args = [];
            if (this._eat('(')) {
                if (op === 'cond') {
                    while (!this._match(')') && !this._match('eof')) {
                        const condition = this._parseValue();
                        this._expect(':');
                        const value = this._parseValue();
                        args.push({ condition, value });
                        this._eat(',');
                    }
                } else {
                    while (!this._match(')') && !this._match('eof')) {
                        args.push(this._parseValue());
                        this._eat(',');
                    }
                }
                this._expect(')');
            }
            let field = null;
            if (this._eat('.')) {
                field = this._expect('id');
            }
            return new tablegen.Value('bang', { op, args, field });
        }
        if (this._match('id')) {
            let value = this._read();
            if (this._match('<')) {
                this._skipBalanced('<', '>');
            }
            if (this._eat('.')) {
                const field = this._expect('id');
                value = `${value}.${field}`;
            }
            if (this._eat('::')) {
                const suffix = this._expect('id');
                return new tablegen.Value('def', `${value}::${suffix}`);
            }
            return new tablegen.Value('def', value);
        }
        if (this._eat('?')) {
            return new tablegen.Value('uninitialized', null);
        }
        throw new tablegen.Error(`Unexpected value at ${this._tokenizer.location()}`);
    }

    _read() {
        return this._tokenizer.read().value;
    }

    _match(type) {
        return this._tokenizer.current().type === type;
    }

    _eat(type) {
        if (this._match(type)) {
            this._read();
            return true;
        }
        return false;
    }

    _expect(type) {
        if (this._match(type)) {
            return this._read();
        }
        throw new tablegen.Error(`Expected '${type}' but got '${this._tokenizer.current().type}' at ${this._tokenizer.location()}`);
    }

    _isKeyword(tokenType) {
        const keywords = [
            'assert', 'bit', 'bits', 'class', 'code', 'dag', 'def', 'defm',
            'defset', 'defvar', 'dump', 'else', 'false', 'field', 'foreach',
            'if', 'in', 'include', 'int', 'let', 'list', 'multiclass',
            'string', 'then', 'true'
        ];
        return keywords.includes(tokenType);
    }

    _skipBalanced(open, close) {
        if (!this._match(open)) {
            return;
        }
        this._read(); // consume opening token
        let depth = 1;
        while (depth > 0 && !this._match('eof')) {
            if (this._match(open)) {
                depth++;
            } else if (this._match(close)) {
                depth--;
            }
            this._read();
        }
    }

    _skipUntil(types) {
        while (!this._match('eof') && !types.includes(this._tokenizer.current().type)) {
            this._read();
        }
    }
};

tablegen.Error = class extends Error {

    constructor(message) {
        super(message);
        this.name = 'TableGen Error';
    }
};

export const Reader = tablegen.Reader;