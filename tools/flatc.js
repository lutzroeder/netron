
import * as fs from 'fs/promises';
import * as path from 'path';

const flatc = {};

flatc.Object = class {

    constructor(parent, name) {
        this.parent = parent;
        this.name = name;
        this.metadata = new Map();
    }

    get root() {
        return this.parent.root;
    }

    resolve() {
        if (!this.resolved) {
            for (const key of this.metadata.keys()) {
                switch (key) {
                    case 'bit_flags':
                    case 'deprecated':
                    case 'force_align':
                    case 'key':
                    case 'required':
                        break;
                    default:
                        throw new flatc.Error(`Unsupported attribute '${key}'.`);
                }
            }
            this.resolved = true;
        }
    }

    find(name, type) {
        return this.parent ? this.parent.find(name, type) : undefined;
    }
};

flatc.Namespace = class extends flatc.Object {

    constructor(parent, name) {
        super(parent, name);
        this.children = new Map();
        this.root_type = new Map();
    }

    resolve() {
        if (!this.resolved) {
            for (const child of this.children.values()) {
                child.resolve();
            }
            if (this.root_type.size > 0) {
                for (const [name, value] of this.root_type) {
                    const type = this.find(name, flatc.Type);
                    if (!type) {
                        throw new flatc.Error(`Failed to resolve root type '${name}'.`);
                    }
                    if (value) {
                        type.file_identifier = value;
                    }
                    this.root.root_type.add(type);
                }
                this.root_type.clear();
            }
            super.resolve();
        }
    }

    find(name, type) {
        if (type === flatc.Type) {
            const parts = name.split('.');
            const typeName = parts.pop();
            const namespaceName = parts.join('.');
            if (namespaceName === '') {
                if (this.children.has(typeName)) {
                    return this.children.get(typeName);
                }
            }
            const namespace = this.parent.find(namespaceName, flatc.Namespace);
            if (namespace) {
                if (namespace.children.has(typeName)) {
                    return namespace.children.get(typeName);
                }
            }
            const parents = this.name.split('.');
            while (parents.length > 1) {
                parents.pop();
                const parentNamespaceName = parents.join('.') + (namespaceName ? `.${namespaceName}` : '');
                const namespace = this.parent.find(parentNamespaceName, flatc.Namespace);
                if (namespace) {
                    if (namespace.children.has(typeName)) {
                        return namespace.children.get(typeName);
                    }
                }
            }
        }
        return super.find(name, type);
    }
};

flatc.Type = class extends flatc.Object {

    constructor(parent, name) {
        super(parent, name);
        if (parent instanceof flatc.Namespace) {
            if (parent.children.has(name)) {
                throw new flatc.Error(`Duplicate identifier '${name}'.`);
            }
            parent.children.set(name, this);
        }
    }
};

flatc.Enum = class extends flatc.Type {

    constructor(parent, name, base) {
        super(parent, name);
        this.base = base;
        this.values = new Map();
    }

    resolve() {
        if (!this.resolved) {
            if (this.base instanceof flatc.TypeReference) {
                this.base = this.base.resolve(this);
                this.defaultValue = this.base.defaultValue;
            }
            let index = 0;
            for (const key of this.values.keys()) {
                if (this.values.get(key) === undefined) {
                    this.values.set(key, index);
                }
                index = this.values.get(key) + 1;
            }
            this.keys = new Map(Array.from(this.values).map(([key, value]) => [value, key]));
            super.resolve();
        }
    }
};

flatc.Alias = class extends flatc.Type {

    constructor(parent, name, type) {
        super(parent, name);
        this.type = type;
    }
};

flatc.Union = class extends flatc.Type {

    constructor(parent, name) {
        super(parent, name);
        this.values = [];
    }

    resolve() {
        if (!this.resolved) {
            let index = 1;
            for (const value of this.values) {
                value.index = value.index === undefined ? index : value.index;
                index = value.index + 1;
                const name = this.parent.find(value.name, flatc.Type);
                const type = value.type ? this.parent.find(value.type, flatc.Type) : null;
                value.type = name;
                if (!name && type) {
                    value.type = new flatc.Alias(this.parent, value.name, type);
                    type.aliases.add(value.type);
                    this.root.aliases.push(value);
                }
            }
            super.resolve();
        }
    }
};

flatc.Table = class extends flatc.Type {

    constructor(parent, name) {
        super(parent, name);
        this.fields = new Map();
        this.aliases = new Set();
    }

    resolve() {
        if (!this.resolved) {
            let offset = 4;
            for (const field of this.fields.values()) {
                field.resolve();
                field.offset = offset;
                offset += (field.type instanceof flatc.Union) ? 4 : 2;
            }
            super.resolve();
        }
    }
};

flatc.Struct = class extends flatc.Type {

    constructor(parent, name) {
        super(parent, name);
        this.fields = new Map();
        this.size = -1;
    }

    resolve() {
        if (!this.resolved) {
            let offset = 0;
            for (const field of this.fields.values()) {
                field.resolve();
                const fieldType = field.type instanceof flatc.Enum ? field.type.base : field.type;
                if (field.repeated) {
                    if (field.length === undefined) {
                        const name = `${this.parent.name}.${this.name}`;
                        throw new flatc.Error(`Struct '${name}' may contain only scalar or struct fields.`);
                    }
                    const size = fieldType.size;
                    field.offset = (offset % size) === 0 ? offset : (Math.floor(offset / size) + 1) * size;
                    offset = field.offset + (field.length * size);
                } else if (fieldType instanceof flatc.PrimitiveType && field.type !== 'string') {
                    const size = fieldType.size;
                    field.offset = (offset % size) === 0 ? offset : (Math.floor(offset / size) + 1) * size;
                    offset = field.offset + size;
                } else if (field.type instanceof flatc.Struct) {
                    field.type.resolve();
                    const align = 8;
                    field.offset = (offset % align) === 0 ? offset : (Math.floor(offset / align) + 1) * align;
                    offset += field.type.size;
                } else {
                    throw new flatc.Error('Structs may contain only scalar or struct fields.');
                }
            }
            this.size = offset;
            super.resolve();
        }
    }
};

flatc.Field = class extends flatc.Object {

    constructor(parent, name, type, defaultValue) {
        super(parent, name);
        this.type = type;
        this.defaultValue = defaultValue;
    }

    resolve() {
        if (!this.resolved) {
            if (this.type instanceof flatc.TypeReference) {
                if (this.type.repeated) {
                    this.repeated = true;
                    this.length = this.type.length;
                }
                this.type = this.type.resolve(this);
                if (this.defaultValue === undefined) {
                    let type = this.type instanceof flatc.Enum ? this.type.base : this.type;
                    if (type instanceof flatc.TypeReference) {
                        type = type.resolve(this);
                    }
                    if (type instanceof flatc.PrimitiveType) {
                        this.defaultValue = type.defaultValue;
                    }
                } else if (this.type instanceof flatc.Enum) {
                    this.type.resolve();
                    if (this.type.values.has(this.defaultValue)) {
                        this.defaultValue = this.type.values.get(this.defaultValue);
                    } else if (!new Set(this.type.values.values()).has(this.defaultValue)) {
                        throw new flatc.Error(`Unsupported enum value '${this.defaultValue}'.`);
                    }
                }
            }
            super.resolve();
        }
    }
};

flatc.PrimitiveType = class extends flatc.Type {

    constructor(name, defaultValue, size) {
        super(null, name);
        this.defaultValue = defaultValue;
        this.size = size;
    }

    static get(name) {
        if (!this._map) {
            this._map = new Map();
            const register = (names, defaultValue, size) => {
                const type = new flatc.PrimitiveType(names[0], defaultValue, size);
                for (const name of names) {
                    this._map.set(name, type);
                }
            };
            register(['bool'], false, 1);
            register(['int8', 'byte'], 0, 1);
            register(['uint8', 'ubyte'], 0, 1);
            register(['int16', 'short'], 0, 2);
            register(['uint16', 'ushort'], 0, 2);
            register(['int32', 'int'], 0, 4);
            register(['uint32', 'uint'], 0, 4);
            register(['int64', 'long'], 0n, 8);
            register(['uint64', 'ulong'], 0n, 8);
            register(['float32', 'float'], 0.0, 4);
            register(['float64', 'double'], 0, 4);
            register(['string'], null, undefined);
        }
        return this._map.get(name);
    }
};

flatc.TypeReference = class {

    constructor(name, repeated, length) {
        this.name = name;
        this.repeated = repeated;
        this.length = length;
    }

    resolve(context) {
        const primitiveType = flatc.PrimitiveType.get(this.name);
        if (primitiveType) {
            return primitiveType;
        }
        const type = context.parent.find(this.name, flatc.Type);
        if (type) {
            return type;
        }
        throw new flatc.Error(`Falied to resolve type '${this.name}'.`);
    }
};

flatc.Parser = class {

    constructor(text, file, root) {
        // https://google.github.io/flatbuffers/flatbuffers_grammar.html
        this._tokenizer = new flatc.Tokenizer(text, file);
        this._root = root;
        this._context = root.defineNamespace('');
    }

    include() {
        const includes = [];
        while (!this._tokenizer.match('eof') && this._tokenizer.eat('id', 'include')) {
            includes.push(this._tokenizer.string());
            this._tokenizer.expect(';');
        }
        return includes;
    }

    parse() {
        const attributes = [];
        while (!this._tokenizer.match('eof')) {
            if (this._tokenizer.eat('id', 'namespace')) {
                let name = this._tokenizer.identifier();
                while (this._tokenizer.eat('.')) {
                    name += `.${this._tokenizer.identifier()}`;
                }
                this._tokenizer.expect(';');
                this._context = this._root.defineNamespace(name);
                continue;
            }
            if (this._tokenizer.eat('id', 'table')) {
                const name = this._tokenizer.identifier();
                const table = new flatc.Table(this._context, name);
                this._parseMetadata(table.metadata);
                this._tokenizer.expect('{');
                while (!this._tokenizer.eat('}')) {
                    const field = this._parseField(table);
                    table.fields.set(field.name, field);
                    this._tokenizer.expect(';');
                }
                continue;
            }
            if (this._tokenizer.eat('id', 'struct')) {
                const name = this._tokenizer.identifier();
                const struct = new flatc.Struct(this._context, name);
                this._parseMetadata(struct.metadata);
                this._tokenizer.expect('{');
                while (!this._tokenizer.eat('}')) {
                    const field = this._parseField(struct);
                    struct.fields.set(field.name, field);
                    this._tokenizer.expect(';');
                }
                continue;
            }
            if (this._tokenizer.eat('id', 'enum')) {
                const name = this._tokenizer.identifier();
                this._tokenizer.expect(':');
                const base = this._parseTypeReference();
                if (base.repeated) {
                    throw new flatc.Error(`Underlying enum type must be integral ${this._tokenizer.location()}`);
                }
                const type = new flatc.Enum(this._context, name, base);
                this._parseMetadata(type.metadata);
                this._tokenizer.expect('{');
                while (!this._tokenizer.eat('}')) {
                    const key = this._tokenizer.identifier();
                    const value = this._tokenizer.eat('=') ? this._tokenizer.integer() : undefined;
                    type.values.set(key, value);
                    this._parseMetadata(new Map());
                    if (this._tokenizer.eat(',')) {
                        continue;
                    }
                }
                continue;
            }
            if (this._tokenizer.eat('id', 'union')) {
                const name = this._tokenizer.identifier();
                const union = new flatc.Union(this._context, name);
                this._parseMetadata(union.metadata);
                this._tokenizer.expect('{');
                while (!this._tokenizer.eat('}')) {
                    const name = this._tokenizer.identifier();
                    const type = this._tokenizer.eat(':') ? this._tokenizer.identifier() : null;
                    const index = this._tokenizer.eat('=') ? this._tokenizer.integer() : undefined;
                    union.values.push({ name, type, index });
                    this._parseMetadata(new Map());
                    if (this._tokenizer.eat(',')) {
                        continue;
                    }
                }
                continue;
            }
            if (this._tokenizer.eat('id', 'rpc_service')) {
                throw new flatc.Error(`Unsupported keyword 'rpc_service' ${this._tokenizer.location()}`);
            }
            if (this._tokenizer.eat('id', 'root_type')) {
                const root_type = this._tokenizer.identifier();
                this._root_type = this._root_type || root_type;
                this._tokenizer.eat(';');
                continue;
            }
            if (this._tokenizer.eat('id', 'file_extension')) {
                const value = this._tokenizer.string();
                this._file_extension = value;
                this._tokenizer.eat(';');
                continue;
            }
            if (this._tokenizer.eat('id', 'file_identifier')) {
                const value = this._tokenizer.string();
                if (value.length !== 4) {
                    throw new flatc.Error(`'file_identifier' must be exactly 4 characters ${this._tokenizer.location()}`);
                }
                this._file_identifier = value;
                this._tokenizer.eat(';');
                continue;
            }
            if (this._tokenizer.eat('id', 'attribute')) {
                const token = this._tokenizer.read();
                switch (token.type) {
                    case 'string':
                        attributes.push(token.value);
                        break;
                    case 'id':
                        attributes.push(token.token);
                        break;
                    default:
                        throw new flatc.Error(`Unexpected attribute token '${token.token}' ${this._tokenizer.location()}`);
                }
                this._tokenizer.expect(';');
                continue;
            }
            if (this._tokenizer.eat('{')) {
                throw new flatc.Error(`Unsupported object ${this._tokenizer.location()}`);
            }
            throw new flatc.Error(`Unexpected token '${this._tokenizer.peek().token}' ${this._tokenizer.location()}`);
        }
        if (this._root_type) {
            this._context.root_type.set(this._root_type, this._file_identifier);
        }
    }

    _parseTypeReference() {
        const token = this._tokenizer.read();
        if (token.type === 'id') {
            return new flatc.TypeReference(token.token, false);
        }
        if (token.type === '[') {
            const identifier = this._tokenizer.read();
            if (identifier.type === 'id') {
                if (this._tokenizer.eat(':')) {
                    const length = this._parseScalar(); // array length
                    this._tokenizer.expect(']');
                    return new flatc.TypeReference(identifier.token, true, length);
                }
                this._tokenizer.expect(']');
                return new flatc.TypeReference(identifier.token, true);
            }
        }
        throw new flatc.Error(`Expected type instead of '${token.token}' ${this._tokenizer.location()}`);
    }

    _parseField(parent) {
        const name = this._tokenizer.identifier();
        this._tokenizer.expect(':');
        const type = this._parseTypeReference();
        const defaultValue = this._tokenizer.eat('=') ? this._parseScalar() : undefined;
        const field = new flatc.Field(parent, name, type, defaultValue);
        this._parseMetadata(field.metadata);
        return field;
    }

    _parseMetadata(metadata) {
        if (this._tokenizer.eat('(')) {
            while (!this._tokenizer.eat(')')) {
                const key = this._tokenizer.identifier();
                const value = this._tokenizer.eat(':') ? this._parseSingleValue() : undefined;
                metadata.set(key, value);
                if (this._tokenizer.eat(',')) {
                    continue;
                }
            }
        }
    }

    _parseScalar() {
        const token = this._tokenizer.read();
        switch (token.type) {
            case 'boolean':
            case 'integer':
            case 'float':
                return token.value;
            case 'id':
                return token.token;
            default:
                throw new flatc.Error(`Expected scalar instead of '${token.token}'${this._tokenizer.location()}`);
        }
    }

    _parseSingleValue() {
        const token = this._tokenizer.read();
        switch (token.type) {
            case 'string':
            case 'boolean':
            case 'integer':
            case 'float':
                return token.value;
            default:
                throw new flatc.Error(`Expected single value instead of '${token.token}'${this._token.location()}`);
        }
    }
};

flatc.Tokenizer = class {

    constructor(text, file) {
        this._text = text;
        this._file = file;
        this._position = 0;
        this._lineStart = 0;
        this._line = 0;
        this._token = { type: '', value: '' };
    }

    peek() {
        if (!this._cache) {
            this._token = this._tokenize();
            this._cache = true;
        }
        return this._token;
    }

    read() {
        if (!this._cache) {
            this._token = this._tokenize();
        }
        const next = this._position + this._token.token.length;
        while (this._position < next) {
            if (flatc.Tokenizer._isNewline(this._get(this._position))) {
                this._position = this._newLine(this._position);
                this._lineStart = this._position;
                this._line++;
            } else {
                this._position++;
            }
        }
        this._cache = false;
        return this._token;
    }

    match(type, value) {
        const token = this.peek();
        if (token.type === type && (!value || token.token === value)) {
            return true;
        }
        return false;
    }

    eat(type, value) {
        const token = this.peek();
        if (token.type === type && (!value || token.token === value)) {
            this.read();
            return true;
        }
        return false;
    }

    expect(type, value) {
        const token = this.peek();
        if (token.type !== type) {
            throw new flatc.Error(`Unexpected '${token.token}' instead of '${type}'${this.location()}`);
        }
        if (value && token.token !== value) {
            throw new flatc.Error(`Unexpected '${token.token}' instead of '${value}'${this.location()}`);
        }
        this.read();
    }

    string() {
        const token = this.read();
        if (token.type === 'string') {
            return token.value;
        }
        throw new flatc.Error(`Expected string instead of '${token.token}' ${this.location()}`);
    }

    identifier() {
        const token = this.read();
        if (token.type === 'id') {
            return token.token;
        }
        throw new flatc.Error(`Expected identifier instead of '${token.token}' ${this.location()}`);
    }

    integer() {
        const token = this.read();
        if (token.type === 'integer') {
            return token.value;
        }
        throw new flatc.Error(`Expected integer instead of '${token.token}' ${this.location()}`);
    }

    location() {
        const line = this._line + 1;
        const column = this._position - this._lineStart + 1;
        return `at ${this._file}:${line}:${column}`;
    }

    _tokenize() {
        if (this._token.type !== '\n') {
            this._skipWhitespace();
        }
        if (this._position >= this._text.length) {
            return { type: 'eof', value: '' };
        }
        const content = this._text.slice(this._position);

        const boolean_constant = content.match(/^(true|false)/);
        if (boolean_constant) {
            const [content] = boolean_constant;
            return { type: 'boolean', token: content, value: content === 'true' };
        }

        const identifier = content.match(/^[a-zA-Z_][a-zA-Z0-9_.]*/);
        if (identifier) {
            return { type: 'id', token: identifier[0] };
        }

        const string_constant = content.match(/^".*?"/) || content.match(/^'.*?'/);
        if (string_constant) {
            const [content] = string_constant;
            return { type: 'string', token: content, value: content.substring(1, content.length - 1) };
        }

        const dec_float_constant = content.match(/^[-+]?(([.][0-9]+)|([0-9]+[.][0-9]*)|([0-9]+))([eE][-+]?[0-9]+)?/);
        if (dec_float_constant) {
            const [content] = dec_float_constant;
            if (content.indexOf('.') !== -1 || content.indexOf('e') !== -1) {
                return { type: 'float', token: content, value: parseFloat(content) };
            }
        }

        const hex_float_constant = content.match(/^[-+]?0[xX](([.][0-9a-fA-F]+)|([0-9a-fA-F]+[.][0-9a-fA-F]*)|([0-9a-fA-F]+))([pP][-+]?[0-9]+)/);
        if (hex_float_constant) {
            throw new flatc.Error('Unsupported hexadecimal constant.');
        }

        const dec_integer_constant = content.match(/^[-+]?[0-9]+/);
        if (dec_integer_constant) {
            const [content] = dec_integer_constant;
            return { type: 'integer', token: content, value: parseInt(content, 10) };
        }
        const hex_integer_constant = content.match(/^[-+]?0[xX][0-9a-fA-F]+/);
        if (hex_integer_constant) {
            throw new flatc.Error('Unsupported hexadecimal constant.');
        }

        const c = this._get(this._position);
        switch (c) {
            case ';':
            case ':':
            case '{':
            case '}':
            case '[':
            case ']':
            case '(':
            case ')':
            case '=':
            case ',':
                return { type: c, token: c };
            default:
                throw new flatc.Error(`Unsupported character '${c}' ${this.location()}`);
        }
    }

    _get(position) {
        return position >= this._text.length ? '\0' : this._text[position];
    }

    _skipLine() {
        while (this._position < this._text.length) {
            if (flatc.Tokenizer._isNewline(this._get(this._position))) {
                break;
            }
            this._position++;
        }
    }

    _skipWhitespace() {
        while (this._position < this._text.length) {
            const c = this._get(this._position);
            if (flatc.Tokenizer._isSpace(c)) {
                this._position++;
                continue;
            }
            if (flatc.Tokenizer._isNewline(c)) {
                // Implicit Line Continuation
                this._position = this._newLine(this._position);
                this._lineStart = this._position;
                this._line++;
                continue;
            }
            if (c === '/') {
                const c1 = this._get(this._position + 1);
                if (c1 === '/') {
                    this._skipLine();
                    continue;
                }
                if (c1 === '*') {
                    this._position += 2;
                    while (this._get(this._position) !== '*' || this._get(this._position + 1) !== '/') {
                        this._position++;
                        if ((this._position + 2) > this._text.length) {
                            throw new flatc.Error('Unexpected end of file in comment.');
                        }
                    }
                    this._position += 2;
                    continue;
                }
            }
            break;
        }
    }

    static _isSpace(c) {
        switch (c) {
            case ' ':
            case '\t':
            case '\v': // 11
            case '\f': // 12
            case '\xA0': // 160
                return true;
            default:
                return false;
        }
    }

    static _isNewline(c) {
        switch (c) {
            case '\n':
            case '\r':
            case '\u2028': // 8232
            case '\u2029': // 8233
                return true;
            default:
                return false;
        }
    }

    _newLine(position) {
        if ((this._get(position) === '\n' && this._get(position + 1) === '\r') ||
            (this._get(position) === '\r' && this._get(position + 1) === '\n')) {
            return position + 2;
        }
        return position + 1;
    }
};

flatc.Root = class extends flatc.Object {

    constructor(root) {
        super(null, root);
        this._namespaces = new Map();
        this._files = new Set();
        this.root_type = new Set();
    }

    async load(paths, files) {
        for (const file of files) {
            /* eslint-disable no-await-in-loop */
            await this._parseFile(paths, file);
            /* eslint-enable no-await-in-loop */
        }
        this.resolve();
    }

    resolve() {
        if (!this.resolved) {
            this.aliases = [];
            for (const namespace of this._namespaces.values()) {
                namespace.resolve();
            }
            for (const value of this.aliases) {
                if (value.type instanceof flatc.Alias && value.type.type.aliases.size <= 1) {
                    value.type.type.aliases.delete(value.type);
                    value.type.parent.children.delete(value.type.name);
                    value.type = value.type.type;
                }
            }
            delete this.aliases;
            super.resolve();
        }
    }

    get root() {
        return this;
    }

    get namespaces() {
        return this._namespaces;
    }

    set(name, value) {
        this.metadata.set(name, value);
    }

    get(name) {
        return this.metadata.get(name);
    }

    defineNamespace(name) {
        if (!this._namespaces.has(name)) {
            this._namespaces.set(name, new flatc.Namespace(this, name));
        }
        return this._namespaces.get(name);
    }

    find(name, type) {
        if (type === flatc.Namespace) {
            if (this._namespaces.has(name)) {
                return this._namespaces.get(name);
            }
        }
        return super.find(name, type);
    }

    async _parseFile(paths, file) {
        if (!this._files.has(file)) {
            this._files.add(file);
            const content = await fs.readFile(file, 'utf-8');
            const parser = new flatc.Parser(content, file, this);
            const includes = parser.include();
            for (const include of includes) {
                /* eslint-disable no-await-in-loop */
                const includeFile = await this._resolve(paths, file, include);
                /* eslint-enable no-await-in-loop */
                if (includeFile) {
                    /* eslint-disable no-await-in-loop */
                    await this._parseFile(paths, includeFile);
                    /* eslint-enable no-await-in-loop */
                    continue;
                }
                throw new flatc.Error(`Include '${include}' not found.`);
            }
            parser.parse();
        }
    }

    async _resolve(paths, origin, target) {
        const access = async (path) => {
            try {
                await fs.access(path);
                return true;
            } catch {
                return false;
            }
        };
        const file = path.join(path.dirname(origin), target);
        const exists = await access(file);
        if (exists) {
            return file;
        }
        for (const current of paths) {
            const file = path.join(current, target);
            /* eslint-disable no-await-in-loop */
            const exists = await access(file);
            /* eslint-enable no-await-in-loop */
            if (exists) {
                return file;
            }
        }
        return null;
    }
};

flatc.Generator = class {

    constructor(root, text) {
        this._root = root;
        this._text = text;
        this._builder = new flatc.Generator.StringBuilder();
        const namespaces = Array.from(this._root.namespaces.values()).filter((namespace) => namespace.name !== '').map((namespace) => namespace.name);
        const exports = new Set (namespaces.map((namespace) => namespace.split('.')[0]));
        for (const value of exports) {
            this._builder.add('');
            this._builder.add(`export const ${value} = {};`);
        }
        for (const namespace of this._root.namespaces.values()) {
            this._buildNamespace(namespace);
        }
        this._content = this._builder.toString();
    }

    get content() {
        return this._content;
    }

    _buildNamespace(namespace) {
        if (namespace.name !== '') {
            const parts = namespace.name.split('.');
            for (let i = 2; i <= parts.length; i++) {
                const name = `${parts.slice(0, i).join('.')}`;
                this._builder.add('');
                this._builder.add(`${name} = ${name} || {};`);
            }
        }
        for (const child of namespace.children.values()) {
            if (child instanceof flatc.Table) {
                this._buildTable(child);
            } else if (child instanceof flatc.Struct) {
                this._buildStruct(child);
            } else if (child instanceof flatc.Union) {
                this._buildUnion(child);
            } else if (child instanceof flatc.Alias) {
                this._buildAlias(child);
            } else if (child instanceof flatc.Enum) {
                this._buildEnum(child);
            } else {
                throw new flatc.Error(`Unsupported type '${child.name}'.`);
            }
        }
    }

    _buildTable(type) {

        const typeName = `${type.parent.name}.${type.name}`;
        const typeReference = `${typeName}`;

        /* eslint-disable indent */
        this._builder.add('');
        this._builder.add(`${typeReference} = class ${type.name} {`);
        this._builder.indent();

            if (this._root.root_type.has(type)) {

                const file_identifier = type.file_identifier;
                if (file_identifier) {
                    this._builder.add('');
                    this._builder.add('static identifier(reader) {');
                    this._builder.indent();
                        this._builder.add(`return reader.identifier === '${file_identifier}';`);
                    this._builder.outdent();
                    this._builder.add('}');
                }

                this._builder.add('');
                this._builder.add('static create(reader) {');
                this._builder.indent();
                    this._builder.add(`return ${typeReference}.decode(reader, reader.root);`);
                this._builder.outdent();
                this._builder.add('}');

                if (this._text) {
                    this._builder.add('');
                    this._builder.add('static createText(reader) {');
                    this._builder.indent();
                        this._builder.add(`return ${typeReference}.decodeText(reader, reader.root);`);
                    this._builder.outdent();
                    this._builder.add('}');
                }
            }

            this._builder.add('');
            if (type.aliases.size > 0) {
                this._builder.add('static decode(reader, position, $) {');
            } else if (type.fields.size === 0) {
                this._builder.add('static decode(/* reader, position */) {');
            } else {
                this._builder.add('static decode(reader, position) {');
            }
            this._builder.indent();
                this._builder.add(type.aliases.size > 0 ? `$ = $ || new ${typeReference}();` : `const $ = new ${typeReference}();`);
                for (const field of type.fields.values()) {
                    const fieldType = field.type instanceof flatc.Enum ? field.type.base : field.type;
                    if (field.repeated) {
                        if (fieldType instanceof flatc.PrimitiveType) {
                            switch (field.type.name) {
                                case 'int64': {
                                    this._builder.add(`$.${field.name} = reader.int64s_(position, ${field.offset});`);
                                    break;
                                }
                                case 'uint64': {
                                    this._builder.add(`$.${field.name} = reader.uint64s_(position, ${field.offset});`);
                                    break;
                                }
                                case 'string': {
                                    this._builder.add(`$.${field.name} = reader.strings_(position, ${field.offset});`);
                                    break;
                                }
                                case 'bool': {
                                    this._builder.add(`$.${field.name} = reader.bools_(position, ${field.offset});`);
                                    break;
                                }
                                default: {
                                    const arrayType = `${fieldType.name[0].toUpperCase() + fieldType.name.substring(1)}Array`;
                                    this._builder.add(`$.${field.name} = reader.array(position, ${field.offset}, ${arrayType});`);
                                    break;
                                }
                            }
                        } else if (fieldType instanceof flatc.Union) {
                            const unionType = `${field.type.parent.name}.${field.type.name}`;
                            this._builder.add(`$.${field.name} = reader.unions(position, ${field.offset}, ${unionType});`);
                        } else if (fieldType instanceof flatc.Struct) {
                            const fieldType = `${field.type.parent.name}.${field.type.name}`;
                            this._builder.add(`$.${field.name} = reader.structs(position, ${field.offset}, ${fieldType});`);
                        } else {
                            const fieldType = `${field.type.parent.name}.${field.type.name}`;
                            this._builder.add(`$.${field.name} = reader.tables(position, ${field.offset}, ${fieldType});`);
                        }
                    } else if (fieldType instanceof flatc.PrimitiveType) {
                        const n = fieldType.name === 'uint64' || fieldType.name === 'int64' ? 'n' : '';
                        this._builder.add(`$.${field.name} = reader.${fieldType.name}_(position, ${field.offset}, ${field.defaultValue}${n});`);
                    } else if (fieldType instanceof flatc.Union) {
                        const unionType = `${field.type.parent.name}.${field.type.name}`;
                        this._builder.add(`$.${field.name} = reader.union(position, ${field.offset}, ${unionType});`);
                    } else if (fieldType instanceof flatc.Struct) {
                        const fieldType = `${field.type.parent.name}.${field.type.name}`;
                        this._builder.add(`$.${field.name} = reader.struct(position, ${field.offset}, ${fieldType});`);
                    } else {
                        const fieldType = `${field.type.parent.name}.${field.type.name}`;
                        this._builder.add(`$.${field.name} = reader.table(position, ${field.offset}, ${fieldType});`);
                    }
                }
                this._builder.add('return $;');
            this._builder.outdent();
            this._builder.add('}');

            if (this._text) {
                this._builder.add('');
                if (type.aliases.size > 0) {
                    this._builder.add('static decodeText(reader, json, $) {');
                } else if (type.fields.size === 0) {
                    this._builder.add('static decodeText(/* reader, json */) {');
                } else {
                    this._builder.add('static decodeText(reader, json) {');
                }
                this._builder.indent();
                    this._builder.add(type.aliases.size > 0 ? `$ = $ || new ${typeReference}();` : `const $ = new ${typeReference}();`);
                    for (const field of type.fields.values()) {
                        if (field.repeated) {
                            if (field.type instanceof flatc.PrimitiveType) {
                                switch (field.type.name) {
                                    case 'int64':
                                    case 'uint64':
                                    case 'string':
                                    case 'bool': {
                                        this._builder.add(`$.${field.name} = reader.array(json.${field.name});`);
                                        break;
                                    }
                                    default: {
                                        const type = `${field.type.name[0].toUpperCase() + field.type.name.substring(1)}Array`;
                                        this._builder.add(`$.${field.name} = reader.array(json.${field.name}, ${type});`);
                                        break;
                                    }
                                }
                            } else if (field.type instanceof flatc.Union) {
                                throw new flatc.Error('Not implemented.');
                            } else if (field.type instanceof flatc.Struct) {
                                const fieldType = `${field.type.parent.name}.${field.type.name}`;
                                this._builder.add(`$.${field.name} = ${fieldType}.decode(reader, position + ${field.offset});`);
                            } else {
                                const fieldType = `${field.type.parent.name}.${field.type.name}`;
                                this._builder.add(`$.${field.name} = reader.objects(json.${field.name}, ${fieldType});`);
                            }
                        } else if (field.type instanceof flatc.PrimitiveType) {
                            switch (field.type.name) {
                                case 'int64': {
                                    this._builder.add(`$.${field.name} = reader.int64(json.${field.name}, ${field.defaultValue}n);`);
                                    break;
                                }
                                case 'uint64': {
                                    this._builder.add(`$.${field.name} = reader.uint64(json.${field.name}, ${field.defaultValue}n);`);
                                    break;
                                }
                                default: {
                                    this._builder.add(`$.${field.name} = reader.value(json.${field.name}, ${field.defaultValue});`);
                                    break;
                                }
                            }
                        } else if (field.type instanceof flatc.Enum) {
                            const enumName = `${field.type.parent.name}.${field.type.name}`;
                            this._builder.add(`$.${field.name} = ${enumName}[json.${field.name}];`);
                        } else if (field.type instanceof flatc.Union) {
                            const unionType = `${field.type.parent.name}.${field.type.name}`;
                            this._builder.add(`$.${field.name} = ${unionType}.decodeText(reader, json.${field.name}, json.${field.name}_type);`);
                        } else { // struct | table
                            const fieldType = `${field.type.parent.name}.${field.type.name}`;
                            this._builder.add(`$.${field.name} = reader.object(json.${field.name}, ${fieldType});`);
                        }
                    }
                    this._builder.add('return $;');
                this._builder.outdent();
                this._builder.add('}');
            }

        this._builder.outdent();
        this._builder.add('};');
        /* eslint-enable indent */
    }

    _buildStruct(type) {

        const typeName = `${type.parent.name}.${type.name}`;
        const typeReference = `${typeName}`;

        /* eslint-disable indent */
        this._builder.add('');
        this._builder.add(`${typeReference} = class ${type.name} {`);
        this._builder.indent();

            this._builder.add('');
            this._builder.add(type.fields.size === 0 ? 'static decode(/* reader, position */) {' : 'static decode(reader, position) {');
            this._builder.indent();
                this._builder.add(`const $ = new ${typeReference}();`);
                for (const field of type.fields.values()) {
                    if (field.repeated) {
                        if (field.length === undefined) {
                            throw new flatc.Error(`Struct '${typeName}' may contain only scalar or struct fields.`);
                        }
                        this._builder.add(`$.${field.name} = undefined; // not implemented`);
                    } else if (field.type instanceof flatc.PrimitiveType) {
                        this._builder.add(`$.${field.name} = reader.${field.type.name}(position + ${field.offset});`);
                    } else if (field.type instanceof flatc.Enum) {
                        this._builder.add(`$.${field.name} = reader.${field.type.base.name}(position + ${field.offset});`);
                    } else if (field.type instanceof flatc.Struct) {
                        const fieldType = `${field.type.parent.name}.${field.type.name}`;
                        this._builder.add(`$.${field.name} = ${fieldType}.decode(reader, position + ${field.offset});`);
                    } else {
                        throw new flatc.Error(`Struct '${typeName}' may contain only scalar or struct fields.`);
                    }
                }
                this._builder.add('return $;');
            this._builder.outdent();
            this._builder.add('}');

            if (this._text) {
                this._builder.add('');
                this._builder.add(type.fields.size === 0 ? 'static decodeText(/* reader, json */) {' : 'static decodeText(reader, json) {');
                this._builder.indent();
                    this._builder.add(`const $ = new ${typeReference}();`);
                    for (const field of type.fields.values()) {
                        if (field.repeated) {
                            throw new flatc.Error(`Struct '${typeName}' may contain only scalar or struct fields.`);
                        } else if (field.type instanceof flatc.PrimitiveType) {
                            this._builder.add(`$.${field.name} = json.${field.name};`);
                        } else if (field.type instanceof flatc.Enum) {
                            throw new flatc.Error('Not implemented.');
                        } else if (field.type instanceof flatc.Struct) {
                            const fieldType = `${field.type.parent.name}.${field.type.name}`;
                            this._builder.add(`$.${field.name} = ${fieldType}.decodeText(reader, json.${field.name});`);
                        } else {
                            throw new flatc.Error(`Struct '${typeName}' may contain only scalar or struct fields.`);
                        }
                    }
                    this._builder.add('return $;');
                this._builder.outdent();
                this._builder.add('}');
            }

        this._builder.outdent();
        this._builder.add('};');
        /* eslint-enable indent */
    }

    _buildAlias(type) {

        const typeName = `${type.parent.name}.${type.name}`;
        const typeReference = `${typeName}`;

        /* eslint-disable indent */
        this._builder.add('');
        this._builder.add(`${typeReference} = class ${type.name} {`);
        this._builder.indent();

            this._builder.add('');
            this._builder.add('static decode(reader, position) {');
            this._builder.indent();
                this._builder.add(`const $ = new ${typeReference}();`);
                this._builder.add(`${type.type.parent.name}.${type.type.name}.decode(reader, position, $);`);
                this._builder.add('return $;');
            this._builder.outdent();
            this._builder.add('}');

            if (this._text) {
                this._builder.add('');
                this._builder.add('static decodeText(reader, json) {');
                this._builder.indent();
                    this._builder.add(`const $ = new ${typeReference}();`);
                    this._builder.add(`${type.type.parent.name}.${type.type.name}.decode(reader, json, $);`);
                    this._builder.add('return $;');
                this._builder.outdent();
                this._builder.add('}');
            }

        this._builder.outdent();
        this._builder.add('};');
        /* eslint-enable indent */
    }

    _buildUnion(type) {

        /* eslint-disable indent */
        this._builder.add('');
        this._builder.add(`${type.parent.name}.${type.name} = class {`);

        this._builder.indent();
            this._builder.add('');
            this._builder.add(type.values.length === 0 ? 'static decode(/* reader, position, type */) {' : 'static decode(reader, position, type) {');
            this._builder.indent();
                this._builder.add('switch (type) {');
                this._builder.indent();
                    for (const value of type.values) {
                        const valueType = `${value.type.parent.name}.${value.type.name}`;
                        this._builder.add(`case ${value.index}: return ${valueType}.decode(reader, position);`);
                    }
                    this._builder.add('default: return undefined;');
                this._builder.outdent();
                this._builder.add('}');
            this._builder.outdent();
            this._builder.add('}');

            if (this._text) {
                this._builder.add('');
                this._builder.add(type.values.length === 0 ? 'static decodeText(/* reader, json, type */) {' : 'static decodeText(reader, json, type) {');
                this._builder.indent();
                    this._builder.add('switch (type) {');
                    this._builder.indent();
                        for (const value of type.values) {
                            const valueType = `${value.type.parent.name}.${value.type.name}`;
                            this._builder.add(`case '${value.name}': return ${valueType}.decodeText(reader, json);`);
                        }
                        this._builder.add('default: return undefined;');
                    this._builder.outdent();
                    this._builder.add('}');
                this._builder.outdent();
                this._builder.add('}');
            }

        this._builder.outdent();
        this._builder.add('};');
        /* eslint-enable indent */
    }

    _buildEnum(type) {

        /* eslint-disable indent */
        this._builder.add('');
        this._builder.add(`${type.parent.name}.${type.name} = {`);
        this._builder.indent();
            const keys = Array.from(type.values.keys());
            for (let i = 0; i < keys.length; i++) {
                const key = keys[i];
                this._builder.add(`${key}: ${type.values.get(key)}${i === keys.length - 1 ? '' : ','}`);
            }
        this._builder.outdent();
        this._builder.add('};');
        /* eslint-enable indent */
    }
};

flatc.Generator.StringBuilder = class {

    constructor() {
        this._indentation = '';
        this._lines = [''];
        this._newline = true;
    }

    indent() {
        this._indentation += '    ';
    }

    outdent() {
        if (this._indentation.length === 0) {
            throw new flatc.Error('Invalid indentation.');
        }
        this._indentation = this._indentation.substring(0, this._indentation.length - 4);
    }

    add(text, newline) {
        if (this._newline) {
            if (text !== '') {
                this._lines.push(this._indentation);
            }
        }
        this._lines[this._lines.length - 1] = this._lines[this._lines.length - 1] + text + (newline === false ? '' : '\n');
        this._newline = newline === false ? false : true;
    }

    toString() {
        return this._lines.join('');
    }
};

flatc.Error = class extends Error {

    constructor(message) {
        super(message);
        this.name = 'FlatBuffers Compiler Error';
    }
};

const main = async (args) => {
    const options = { verbose: false, root: 'default', out: '', text: false, paths: [], files: [] };
    while (args.length > 0) {
        const arg = args.shift();
        switch (arg) {
            case '--verbose':
                options.verbose = true;
                break;
            case '--out':
                options.out = args.shift();
                break;
            case '--root':
                options.root = args.shift();
                break;
            case '--text':
                options.text = true;
                break;
            case '--path':
                options.paths.push(args.shift());
                break;
            default:
                if (arg.startsWith('-')) {
                    throw new flatc.Error(`Invalid command line argument '${arg}'.`);
                }
                options.files.push(arg);
                break;
        }
    }
    try {
        const root = new flatc.Root(options.root);
        await root.load(options.paths, options.files);
        const generator = new flatc.Generator(root, options.text);
        if (options.out) {
            await fs.writeFile(options.out, generator.content, 'utf-8');
        }
    } catch (error) {
        if (error instanceof flatc.Error && !options.verbose) {
            process.stderr.write(`${error.message}\n`);
        } else {
            process.stderr.write(`${error.stack}\n`);
        }
        process.exit(1);
    }
    process.exit(0);
};

if (typeof process === 'object' &&
    Array.isArray(process.argv) && process.argv.length > 1 &&
    path.basename(process.argv[1]) === 'flatc.js') {
    const args = process.argv.slice(2);
    await main(args);
}

export const Root = flatc.Root;
export const Namespace = flatc.Namespace;
export const Type = flatc.Type;
export const Enum = flatc.Enum;
