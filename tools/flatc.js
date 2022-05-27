
const flatc = {};
const fs = require('fs');
const path = require('path');

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
                    case 'force_align':
                    case 'deprecated':
                    case 'key':
                    case 'required':
                        break;
                    default:
                        throw new flatc.Error("Unsupported attribute '" + key + "'.");
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
        this.root_type = new Set();
    }

    resolve() {
        if (!this.resolved) {
            for (const child of this.children.values()) {
                child.resolve();
            }
            if (this.root_type.size > 0) {
                for (const root_type of this.root_type) {
                    const type = this.find(root_type, flatc.Type);
                    if (!type) {
                        throw new flatc.Error("Failed to resolve root type '" + root_type + "'.");
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
                const namespace = this.parent.find(parents.join('.') + '.' + namespaceName, flatc.Namespace);
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
                throw new flatc.Error("Duplicate identifier '" + name + "'.");
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
            this.keys = new Map(Array.from(this.values).map((pair) => [ pair[1], pair[0] ]));
            super.resolve();
        }
    }
};

flatc.Union = class extends flatc.Type {

    constructor(parent, name) {
        super(parent, name);
        this.values = new Map();
    }

    resolve() {
        if (!this.resolved) {
            let index = 1;
            for (const key of this.values.keys()) {
                if (this.values.get(key) === undefined) {
                    this.values.set(key, index);
                }
                index = this.values.get(key) + 1;
            }
            const map = new Map();
            for (const pair of this.values) {
                const type = this.parent.find(pair[0], flatc.Type);
                map.set(pair[1], type);
            }
            this.values = map;
            super.resolve();
        }
    }
};


flatc.Table = class extends flatc.Type {

    constructor(parent, name) {
        super(parent, name);
        this.fields = new Map();
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
                if (field.type instanceof flatc.PrimitiveType && field.type !== 'string') {
                    const size = field.type.size;
                    field.offset = (offset % size != 0) ? (Math.floor(offset / size) + 1) * size : offset;
                    offset = field.offset + field.type.size;
                }
                else if (field.type instanceof flatc.Struct) {
                    field.type.resolve();
                    const align = 8;
                    field.offset = (offset % align != 0) ? (Math.floor(offset / align) + 1) * align : offset;
                    offset += field.type.size;
                }
                else {
                    throw flatc.Error('Structs may contain only scalar or struct fields.');
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
                }
                else if (this.type instanceof flatc.Enum) {
                    this.type.resolve();
                    if (this.type.values.has(this.defaultValue)) {
                        this.defaultValue = this.type.values.get(this.defaultValue);
                    }
                    else if (!new Set(this.type.values.values()).has(this.defaultValue)) {
                        throw new flatc.Error("Unsupported enum value '" + this.defaultValue + "'.");
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
            register([ 'bool' ], false, 1);
            register([ 'int8', 'byte' ], 0, 1);
            register([ 'uint8', 'ubyte' ], 0, 1);
            register([ 'int16', 'short' ], 0, 2);
            register([ 'uint16', 'ushort' ], 0, 2);
            register([ 'int32', 'int' ], 0, 4);
            register([ 'uint32', 'uint' ], 0, 4);
            register([ 'int64', 'long' ], 0, 8);
            register([ 'uint64', 'ulong' ], 0, 8);
            register([ 'float32', 'float' ], 0.0, 4);
            register([ 'float64', 'double' ], 0, 4);
            register([ 'string' ], null, undefined);
        }
        return this._map.get(name);
    }
};

flatc.TypeReference = class {

    constructor(name, repeated) {
        this.name = name;
        this.repeated = repeated;
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
        throw new flatc.Error("Falied to resolve type '" + this.name + "'.");
    }
};

flatc.Parser = class {

    constructor(text, file, root) {
        // https://google.github.io/flatbuffers/flatbuffers_grammar.html
        this._tokenizer = new flatc.Parser.Tokenizer(text, file);
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
                    name += '.' + this._tokenizer.identifier();
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
                    throw new flatc.Error('Underlying enum type must be integral' + this._tokenizer.location());
                }
                const type = new flatc.Enum(this._context, name, base);
                this._parseMetadata(type.metadata);
                this._tokenizer.expect('{');
                while (!this._tokenizer.eat('}')) {
                    const key = this._tokenizer.identifier();
                    const value = this._tokenizer.eat('=') ? this._tokenizer.integer() : undefined;
                    type.values.set(key, value);
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
                    const key = this._tokenizer.eat(':') ? this._tokenizer.identifier() : name;
                    const value = this._tokenizer.eat('=') ? this._tokenizer.integer() : undefined;
                    union.values.set(key, value);
                    if (this._tokenizer.eat(',')) {
                        continue;
                    }
                }
                continue;
            }
            if (this._tokenizer.eat('id', 'rpc_service')) {
                throw new flatc.Error("Unsupported keyword 'rpc_service'." + this._tokenizer.location());
            }
            if (this._tokenizer.eat('id', 'root_type')) {
                this._context.root_type.add(this._tokenizer.identifier());
                this._tokenizer.eat(';');
                continue;
            }
            if (this._tokenizer.eat('id', 'file_extension')) {
                const value = this._tokenizer.string();
                this._root.file_extension = value;
                this._tokenizer.eat(';');
                continue;
            }
            if (this._tokenizer.eat('id', 'file_identifier')) {
                const value = this._tokenizer.string();
                if (value.length !== 4) {
                    throw new flatc.Error("'file_identifier' must be exactly 4 characters " + this._tokenizer.location());
                }
                this._root.file_identifier = value;
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
                        throw new flatc.Error("Unexpected attribute token '" + token.token + "'" + this._tokenizer.location());
                }
                this._tokenizer.expect(';');
                continue;
            }
            if (this._tokenizer.eat('{')) {
                throw new flatc.Error('Unsupported object.' + this._tokenizer.location());
            }
            throw new flatc.Error("Unexpected token '" + this._tokenizer.peek().token + "'" + this._tokenizer.location());
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
                this._tokenizer.expect(']');
                return new flatc.TypeReference(identifier.token, true);
            }
        }
        throw new flatc.Error("Expected type instead of '" + token.token + "'" + this._tokenizer.location());
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
                throw new flatc.Error("Expected scalar instead of '" + token.token + "'" + this._tokenizer.location());
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
                throw new flatc.Error("Expected single value instead of '" + token.token + "'" + this._token.location());
        }
    }
};

flatc.Parser.Tokenizer = class {

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
            if (flatc.Parser.Tokenizer._isNewline(this._get(this._position))) {
                this._position = this._newLine(this._position);
                this._lineStart = this._position;
                this._line++;
            }
            else {
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
            throw new flatc.Error("Unexpected '" + token.token + "' instead of '" + type + "'" + this.location());
        }
        if (value && token.token !== value) {
            throw new flatc.Error("Unexpected '" + token.token + "' instead of '" + value + "'" + this.location());
        }
        this.read();
    }

    string() {
        const token = this.read();
        if (token.type === 'string') {
            return token.value;
        }
        throw new flatc.Error("Expected string instead of '" + token.token + "'" + this.location());
    }

    identifier() {
        const token = this.read();
        if (token.type === 'id') {
            return token.token;
        }
        throw new flatc.Error("Expected identifier instead of '" + token.token + "'" + this.location());
    }

    integer() {
        const token = this.read();
        if (token.type === 'integer') {
            return token.value;
        }
        throw new flatc.Error("Expected integer instead of '" + token.token + "'" + this.location());
    }

    location() {
        return ' at ' + this._file + ':' + (this._line + 1).toString() + ':' + (this._position - this._lineStart + 1).toString();
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
            const content = boolean_constant[0];
            return { type: 'boolean', token: content, value: content === 'true' };
        }

        const identifier = content.match(/^[a-zA-Z_][a-zA-Z0-9_.]*/);
        if (identifier) {
            return { type: 'id', token: identifier[0] };
        }

        const string_constant = content.match(/^".*?"/) || content.match(/^'.*?'/);
        if (string_constant) {
            const content = string_constant[0];
            return { type: 'string', token: content, value: content.substring(1, content.length - 1) };
        }

        const dec_float_constant = content.match(/^[-+]?(([.][0-9]+)|([0-9]+[.][0-9]*)|([0-9]+))([eE][-+]?[0-9]+)?/);
        if (dec_float_constant) {
            const content = dec_float_constant[0];
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
            const content = dec_integer_constant[0];
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
                throw new flatc.Error("Unsupported character '" + c + "' " + this.location());
        }
    }

    _get(position) {
        return position >= this._text.length ? '\0' : this._text[position];
    }

    _skipLine() {
        while (this._position < this._text.length) {
            if (flatc.Parser.Tokenizer._isNewline(this._get(this._position))) {
                break;
            }
            this._position++;
        }
    }

    _skipWhitespace() {
        while (this._position < this._text.length) {
            const c = this._get(this._position);
            if (flatc.Parser.Tokenizer._isSpace(c)) {
                this._position++;
                continue;
            }
            if (flatc.Parser.Tokenizer._isNewline(c)) {
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
        switch(c) {
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

    constructor(root, paths, files) {
        super(null, root);
        this._namespaces = new Map();
        this._files = new Set();
        this.root_type = new Set();
        for (const file of files) {
            this._parseFile(paths, file);
        }
        this.resolve();
    }

    resolve() {
        if (!this.resolved) {
            for (const namespace of this._namespaces.values()) {
                namespace.resolve();
            }
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

    _parseFile(paths, file) {
        if (!this._files.has(file)) {
            this._files.add(file);
            const content = fs.readFileSync(file, 'utf-8');
            const parser = new flatc.Parser(content, file, this);
            const includes = parser.include();
            for (const include of includes) {
                const includeFile = this._resolve(paths, file, include);
                if (includeFile) {
                    this._parseFile(paths, includeFile);
                    continue;
                }
                throw new flatc.Error("Include '" + include + "' not found.");
            }
            parser.parse();
        }
    }

    _resolve(paths, origin, target) {
        const file = path.join(path.dirname(origin), target);
        if (fs.existsSync(file)) {
            return file;
        }
        for (const current of paths) {
            const file = path.join(current, target);
            if (fs.existsSync(file)) {
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
        this._builder.add("var $root = flatbuffers.get('" + this._root.name + "');");
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
            for (let i = 1; i <= parts.length; i++) {
                const name = '$root.' + parts.slice(0, i).join('.');
                this._builder.add('');
                this._builder.add(name + ' = ' + name + ' || {};');
            }
        }
        for (const child of namespace.children.values()) {
            if (child instanceof flatc.Table) {
                this._buildTable(child);
            }
            else if (child instanceof flatc.Struct) {
                this._buildStruct(child);
            }
            else if (child instanceof flatc.Union) {
                this._buildUnion(child);
            }
            else if (child instanceof flatc.Enum) {
                this._buildEnum(child);
            }
        }
    }

    _buildTable(type) {

        const typeName = type.parent.name + '.' + type.name;
        const typeReference = '$root.' + typeName;

        /* eslint-disable indent */
        this._builder.add('');
        this._builder.add(typeReference + ' = class ' + type.name + ' {');
        this._builder.indent();

            if (this._root.root_type.has(type)) {

                const file_identifier = this._root.file_identifier;
                if (file_identifier) {
                    this._builder.add('');
                    this._builder.add('static identifier(reader) {');
                    this._builder.indent();
                        this._builder.add("return reader.identifier === '" + file_identifier + "';");
                    this._builder.outdent();
                    this._builder.add('}');
                }

                this._builder.add('');
                this._builder.add('static create(reader) {');
                this._builder.indent();
                    this._builder.add('return ' + typeReference + '.decode(reader, reader.root);');
                this._builder.outdent();
                this._builder.add('}');

                if (this._text) {
                    this._builder.add('');
                    this._builder.add('static createText(reader) {');
                    this._builder.indent();
                        this._builder.add('return ' + typeReference + '.decodeText(reader, reader.root);');
                    this._builder.outdent();
                    this._builder.add('}');
                }
            }

            this._builder.add('');
            this._builder.add(type.fields.size !== 0 ? 'static decode(reader, position) {' : 'static decode(/* reader, position */) {');
            this._builder.indent();
                this._builder.add('const $ = new ' + typeReference + '();');
                for (const field of type.fields.values()) {
                    const fieldType = field.type instanceof flatc.Enum ? field.type.base : field.type;
                    if (field.repeated) {
                        if (fieldType instanceof flatc.PrimitiveType) {
                            switch (field.type.name) {
                                case 'int64': {
                                    this._builder.add('$.' + field.name + ' = reader.int64s_(position, ' + field.offset + ');');
                                    break;
                                }
                                case 'uint64': {
                                    this._builder.add('$.' + field.name + ' = reader.uint64s_(position, ' + field.offset + ');');
                                    break;
                                }
                                case 'string': {
                                    this._builder.add('$.' + field.name + ' = reader.strings_(position, ' + field.offset + ');');
                                    break;
                                }
                                case 'bool': {
                                    this._builder.add('$.' + field.name + ' = reader.bools_(position, ' + field.offset + ');');
                                    break;
                                }
                                default: {
                                    const arrayType = fieldType.name[0].toUpperCase() + fieldType.name.substring(1) + 'Array';
                                    this._builder.add('$.' + field.name + ' = reader.typedArray(position, ' + field.offset + ', ' + arrayType + ');');
                                    break;
                                }
                            }
                        }
                        else if (fieldType instanceof flatc.Union) {
                            const unionType = '$root.' + field.type.parent.name + '.' + field.type.name;
                            this._builder.add('$.' + field.name + ' = reader.unionArray(position, ' + field.offset + ', ' + ',' + unionType + '.decode);');
                        }
                        else if (fieldType instanceof flatc.Struct) {
                            const fieldType = '$root.' + field.type.parent.name + '.' + field.type.name;
                            this._builder.add('$.' + field.name + ' = reader.structArray(position, ' + field.offset + ', ' + field.size + ',' + fieldType + '.decode);');
                        }
                        else {
                            const fieldType = '$root.' + field.type.parent.name + '.' + field.type.name;
                            this._builder.add('$.' + field.name + ' = reader.tableArray(position, ' + field.offset + ', ' + fieldType + '.decode);');
                        }
                    }
                    else if (fieldType instanceof flatc.PrimitiveType) {
                        this._builder.add('$.' + field.name + ' = reader.' + fieldType.name + '_(position, ' + field.offset + ', ' + field.defaultValue + ');');
                    }
                    else if (fieldType instanceof flatc.Union) {
                        const unionType = '$root.' + field.type.parent.name + '.' + field.type.name;
                        this._builder.add('$.' + field.name + ' = reader.union(position, ' + field.offset + ', ' + unionType + '.decode);');
                    }
                    else if (fieldType instanceof flatc.Struct) {
                        const fieldType = '$root.' + field.type.parent.name + '.' + field.type.name;
                        this._builder.add('$.' + field.name + ' = reader.struct(position, ' + field.offset + ', ' + fieldType + '.decode);');
                    }
                    else {
                        const fieldType = '$root.' + field.type.parent.name + '.' + field.type.name;
                        this._builder.add('$.' + field.name + ' = reader.table(position, ' + field.offset + ', ' + fieldType + '.decode);');
                    }
                }
                this._builder.add('return $;');
            this._builder.outdent();
            this._builder.add('}');

            if (this._text) {
                this._builder.add('');
                this._builder.add(type.fields.size !== 0 ? 'static decodeText(reader, json) {' : 'static decodeText(/* reader, json */) {');
                this._builder.indent();
                    this._builder.add('const $ = new ' + typeReference + '();');
                    for (const field of type.fields.values()) {
                        if (field.repeated) {
                            if (field.type instanceof flatc.PrimitiveType) {
                                switch (field.type.name) {
                                    case 'int64':
                                    case 'uint64':
                                    case 'string':
                                    case 'bool': {
                                        this._builder.add('$.' + field.name + ' = reader.array(json.' + field.name + ');');
                                        break;
                                    }
                                    default: {
                                        const arrayType = field.type.name[0].toUpperCase() + field.type.name.substring(1) + 'Array';
                                        this._builder.add('$.' + field.name + ' = reader.typedArray(json.' + field.name + ', ' + arrayType + ');');
                                        break;
                                    }
                                }
                            }
                            else if (field.type instanceof flatc.Union) {
                                throw new flatc.Error('Not implemented.');
                            }
                            else if (field.type instanceof flatc.Struct) {
                                const fieldType = '$root.' + field.type.parent.name + '.' + field.type.name;
                                this._builder.add('$.' + field.name + ' = ' + fieldType + '.decode(reader, position + ' + field.offset + ');');
                            }
                            else {
                                const fieldType = '$root.' + field.type.parent.name + '.' + field.type.name;
                                this._builder.add('$.' + field.name + ' = reader.objectArray(json.' + field.name + ', ' + fieldType + '.decodeText);');
                            }
                        }
                        else if (field.type instanceof flatc.PrimitiveType) {
                            this._builder.add('$.' + field.name + ' = reader.value(json.' + field.name + ', ' + field.defaultValue + ');');
                        }
                        else if (field.type instanceof flatc.Enum) {
                            const enumName = '$root.' + field.type.parent.name + '.' + field.type.name;
                            this._builder.add('$.' + field.name + ' = ' + enumName + '[json.' + field.name + '];');
                        }
                        else if (field.type instanceof flatc.Union) {
                            const unionType = '$root.' + field.type.parent.name + '.' + field.type.name;
                            this._builder.add('$.' + field.name + ' = ' + unionType + '.decodeText(reader, json.' + field.name + ', json.' + field.name + '_type' + ');');
                        }
                        else { // struct | table
                            const fieldType = '$root.' + field.type.parent.name + '.' + field.type.name;
                            this._builder.add('$.' + field.name + ' = reader.object(json.' + field.name + ', ' + fieldType + '.decodeText);');
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

        const typeName = type.parent.name + '.' + type.name;
        const typeReference = '$root.' + typeName;

        /* eslint-disable indent */
        this._builder.add('');
        this._builder.add(typeReference + ' = class ' + type.name + ' {');
        this._builder.indent();

            this._builder.add('');
            this._builder.add(type.fields.size !== 0 ? 'static decode(reader, position) {' : 'static decode(/* reader, position */) {');
            this._builder.indent();
                this._builder.add('const $ = new ' + typeReference + '();');
                for (const field of type.fields.values()) {
                    if (field.repeated) {
                        throw new flatc.Error("Struct '" + typeName + "' may contain only scalar or struct fields.");
                    }
                    else if (field.type instanceof flatc.PrimitiveType) {
                        this._builder.add('$.' + field.name + ' = reader.' + field.type.name + '(position + ' + field.offset + ');');
                    }
                    else if (field.type instanceof flatc.Enum) {
                        throw new flatc.Error('Not implemented.');
                    }
                    else if (field.type instanceof flatc.Struct) {
                        const fieldType = '$root.' + field.type.parent.name + '.' + field.type.name;
                        this._builder.add('$.' + field.name + ' = ' + fieldType + '.decode(reader, position + ' + field.offset + ');');
                    }
                    else {
                        throw new flatc.Error("Struct '" + typeName + "' may contain only scalar or struct fields.");
                    }
                }
                this._builder.add('return $;');
            this._builder.outdent();
            this._builder.add('}');

            if (this._text) {
                this._builder.add('');
                this._builder.add(type.fields.size !== 0 ? 'static decodeText(reader, json) {' : 'static decodeText(/* reader, json */) {');
                this._builder.indent();
                    this._builder.add('const $ = new ' + typeReference + '();');
                    for (const field of type.fields.values()) {
                        if (field.repeated) {
                            throw new flatc.Error("Struct '" + typeName + "' may contain only scalar or struct fields.");
                        }
                        else if (field.type instanceof flatc.PrimitiveType) {
                            this._builder.add('$.' + field.name + ' = json.' + field.name + ';');
                        }
                        else if (field.type instanceof flatc.Enum) {
                            throw new flatc.Error('Not implemented.');
                        }
                        else if (field.type instanceof flatc.Struct) {
                            const fieldType = '$root.' + field.type.parent.name + '.' + field.type.name;
                            this._builder.add('$.' + field.name + ' = ' + fieldType + '.decodeText(reader, json.' + field.name + ');');
                        }
                        else {
                            throw new flatc.Error("Struct '" + typeName + "' may contain only scalar or struct fields.");
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

    _buildUnion(type) {

        /* eslint-disable indent */
        this._builder.add('');
        this._builder.add('$root.' + type.parent.name + '.' + type.name + ' = class ' + '{');

        this._builder.indent();
            this._builder.add('');
            this._builder.add(type.values.size !== 0 ? 'static decode(reader, position, type) {' : 'static decode(/* reader, position, type */) {');
            this._builder.indent();
                this._builder.add('switch (type) {');
                this._builder.indent();
                    for (const pair of type.values) {
                        const valueType = '$root.' + pair[1].parent.name + '.' + pair[1].name;
                        this._builder.add('case ' + pair[0] + ': return ' + valueType + '.decode(reader, position);');
                    }
                    this._builder.add('default: return undefined;');
                this._builder.outdent();
                this._builder.add('}');
            this._builder.outdent();
            this._builder.add('}');

            this._builder.add('');
            this._builder.add(type.values.size !== 0 ? 'static decodeText(reader, json, type) {' : 'static decodeText(/* reader, json, type */) {');
            this._builder.indent();
                this._builder.add('switch (type) {');
                this._builder.indent();
                    for (const pair of type.values) {
                        const valueType = '$root.' + pair[1].parent.name + '.' + pair[1].name;
                        this._builder.add('case \'' + pair[1].name + '\': return ' + valueType + '.decodeText(reader, json);');
                    }
                    this._builder.add('default: return undefined;');
                this._builder.outdent();
                this._builder.add('}');
            this._builder.outdent();
            this._builder.add('}');

        this._builder.outdent();
        this._builder.add('};');
        /* eslint-enable indent */
    }

    _buildEnum(type) {

        /* eslint-disable indent */
        this._builder.add('');
        this._builder.add('$root.' + type.parent.name + '.' + type.name + ' = {');
        this._builder.indent();
            const keys = Array.from(type.values.keys());
            for (let i = 0; i < keys.length; i++) {
                const key = keys[i];
                this._builder.add(key + ': ' + type.values.get(key) + (i === keys.length - 1 ? '' : ','));
            }
        this._builder.outdent();
        this._builder.add('};');
        /* eslint-enable indent */
    }
};

flatc.Generator.StringBuilder = class {

    constructor() {
        this._indentation = '';
        this._lines = [];
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

const main = (args) => {

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
                    throw new flatc.Error("Invalid command line argument '" + arg + "'.");
                }
                options.files.push(arg);
                break;
        }
    }

    try {
        const root = new flatc.Root(options.root, options.paths, options.files);
        const generator = new flatc.Generator(root, options.text);
        if (options.out) {
            fs.writeFileSync(options.out, generator.content, 'utf-8');
        }
    }
    catch (err) {
        if (err instanceof flatc.Error && !options.verbose) {
            process.stderr.write(err.message + '\n');
        }
        else {
            process.stderr.write(err.stack + '\n');
        }
        return 1;
    }
    return 0;
};

if (typeof process === 'object' && Array.isArray(process.argv) &&
    process.argv.length > 1 && process.argv[1] === __filename) {
    const args = process.argv.slice(2);
    const code = main(args);
    process.exit(code);
}

if (typeof module !== 'undefined' && typeof module.exports === 'object') {
    module.exports.Root = flatc.Root;
    module.exports.Namespace = flatc.Namespace;
    module.exports.Type = flatc.Type;
    module.exports.Enum = flatc.Enum;
}
