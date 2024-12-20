
import * as fs from 'fs/promises';
import * as path from 'path';

const protoc = {};

protoc.Object = class {

    constructor(parent, name) {
        this.parent = parent;
        this.name = name;
        this.options = new Map();
    }

    get fullName() {
        const path = [this.name];
        let context = this.parent;
        while (context) {
            if (context.name) {
                path.unshift(context.name);
            }
            context = context.parent;
        }
        return path.join('.');
    }
};

protoc.Namespace = class extends protoc.Object {

    constructor(parent, name) {
        super(parent, name);
        this.children = new Map();
        if (!(this instanceof protoc.Root || this instanceof protoc.PrimitiveType)) {
            if (parent.get(this.name)) {
                throw new protoc.Error(`Duplicate name '${this.name}' in '${parent.name}'.`);
            }
            parent.children.set(this.name, this);
        }
    }

    defineNamespace(path) {
        path = path.split('.');
        if (path && path.length > 0 && path[0] === '') {
            throw new protoc.Error('Invalid path.');
        }
        let parent = this;
        while (path.length > 0) {
            const part = path.shift();
            if (parent.children && parent.children.get(part)) {
                parent = parent.children.get(part);
                if (!(parent instanceof protoc.Namespace)) {
                    throw new protoc.Error('Invalid path.');
                }
            } else {
                parent = new protoc.Namespace(parent, part);
            }
        }
        return parent;
    }

    defineType(name) {
        const parts = name.split('.');
        const typeName = parts.pop();
        const parent = this.defineNamespace(parts.join('.'));
        const type = parent.get(name);
        if (type) {
            if (type instanceof protoc.Type) {
                return type;
            }
            throw new protoc.Error('Invalid type');
        }
        return new protoc.Type(parent, typeName);
    }

    get(name) {
        return this.children.get(name) || null;
    }

    find(path, filterType, parentAlreadyChecked) {
        if (path.length === 0) {
            return this;
        }
        if (path[0] === '') {
            return this.root.find(path.slice(1), filterType);
        }
        let found = this.get(path[0]);
        if (found) {
            if (path.length === 1) {
                if (found instanceof filterType) {
                    return found;
                }
            } else if (found instanceof protoc.Namespace && (found = found.find(path.slice(1), filterType, true))) {
                return found;
            }
        } else {
            for (const child of this.children.values()) {
                if (child instanceof protoc.Namespace && (found = child.find(path, filterType, true))) {
                    return found;
                }
            }
        }

        if (!this.parent || parentAlreadyChecked) {
            return null;
        }
        return this.parent.find(path, filterType);
    }

    findType(path) {
        const type = this.find(path.split('.'), protoc.Type);
        if (!type) {
            throw new protoc.Error(`Type or enum '${path}' not found in '${this.name}'.`);
        }
        return type;
    }

    static isReservedId(reserved, id) {
        if (reserved) {
            for (let i = 0; i < reserved.length; ++i) {
                if (typeof reserved[i] !== 'string' && reserved[i][0] <= id && reserved[i][1] > id) {
                    return true;
                }
            }
        }
        return false;
    }

    static isReservedName(reserved, name) {
        if (reserved) {
            for (let i = 0; i < reserved.length; ++i) {
                if (reserved[i] === name) {
                    return true;
                }
            }
        }
        return false;
    }
};

protoc.Root = class extends protoc.Namespace {

    constructor(alias) {
        super(null, '');
        this.alias = alias;
        this._files = new Set();
        this._library = new Map();
        this._library.set('google/protobuf/any.proto', () => {
            const type = this.defineType('google.protobuf.Any');
            type.defineField('type_url', 1, 'string');
            type.defineField('value', 2, 'bytes');
        });
        this._library.set('google/protobuf/wrappers.proto', () => {
            const type = this.defineType('google.protobuf.BoolValue');
            type.defineField('value', 1, 'bool');
        });
    }

    async load(paths, files) {
        for (const file of files) {
            /* eslint-disable no-await-in-loop */
            const resolved = await this._resolve(file, '', paths);
            /* eslint-enable no-await-in-loop */
            if (resolved) {
                /* eslint-disable no-await-in-loop */
                await this._loadFile(paths, resolved);
                /* eslint-enable no-await-in-loop */
            } else {
                throw new protoc.Error(`File '${file}' not found.`);
            }
        }
        return this;
    }

    async _loadFile(paths, file, weak) {
        if (!this._files.has(file)) {
            this._files.add(file);
            if (this._library.has(file)) {
                const callback = this._library.get(file);
                callback();
            } else {
                try {
                    await this._parseFile(paths, file);
                } catch (error) {
                    if (!weak) {
                        throw error;
                    }
                }
            }
        }
    }

    async _parseFile(paths, file) {
        const source = await fs.readFile(file, 'utf-8');
        const parser = new protoc.Parser(source, file, this);
        const parsed = parser.parse();
        for (const item of parsed.imports) {
            /* eslint-disable no-await-in-loop */
            const resolved = await this._resolve(item, file, paths);
            /* eslint-enable no-await-in-loop */
            if (!resolved) {
                throw new protoc.Error(`File '${item}' not found.`);
            }
            /* eslint-disable no-await-in-loop */
            await this._loadFile(paths, resolved);
            /* eslint-enable no-await-in-loop */
        }
        for (const item of parsed.weakImports) {
            /* eslint-disable no-await-in-loop */
            const resolved = await this._resolve(item, file, paths);
            /* eslint-enable no-await-in-loop */
            if (resolved) {
                /* eslint-disable no-await-in-loop */
                await this._loadFile(paths, resolved);
                /* eslint-enable no-await-in-loop */
            }
        }
    }

    async _resolve(target, source, paths) {
        const file = path.resolve(source, target);
        const posix = file.split(path.sep).join(path.posix.sep);
        const index = posix.lastIndexOf('google/protobuf/');
        if (index > -1) {
            const name = posix.substring(index);
            if (this._library.has(name)) {
                return name;
            }
        }
        const access = async (path) => {
            try {
                await fs.access(path);
                return true;
            } catch {
                return false;
            }
        };
        const exists = await access(file);
        if (exists) {
            return file;
        }
        for (const dir of paths) {
            const file = path.resolve(dir, target);
            /* eslint-disable no-await-in-loop */
            const exists = await access(file);
            if (exists) {
                return file;
            }
            /* eslint-enable no-await-in-loop */
        }
        return null;
    }
};

protoc.Type = class extends protoc.Namespace {

    constructor(parent, name) {
        super(parent, name);
        this.fields = new Map();
        this.oneofs = new Map();
        this.extensions = [];
        this.reserved = [];
    }

    get(name) {
        return this.fields.get(name) || this.oneofs.get(name) || this.children.get(name) || null;
    }

    defineField(name, id, type, rule, extend) {
        return new protoc.Field(this, name, id, type, rule, extend);
    }
};

protoc.Enum = class extends protoc.Type {

    constructor(parent, name) {
        super(parent, name);
        this.valuesById = new Map();
        this.values = {};
        this.reserved = [];
    }

    add(name, id) {
        if (!Number.isInteger(id)) {
            throw new protoc.Error('Identifier must be an integer.');
        }
        if (this.values[name] !== undefined) {
            throw new protoc.Error(`Duplicate name '${name}' in '${this.name}'.`);
        }
        if (protoc.Namespace.isReservedId(this.reserved, id)) {
            throw new protoc.Error(`Identifier '${id}' is reserved in '${this.name}'.`);
        }
        if (protoc.Namespace.isReservedName(this.reserved, name)) {
            throw new protoc.Error(`Name '${name}' is reserved in '${this.name}'.`);
        }
        if (this.valuesById.has(id)) {
            if (!this.options.has('allow_alias')) {
                throw new protoc.Error(`Duplicate identifier '${id}' in '${this.name}'.`);
            }
        } else {
            this.valuesById.set(id, name);
        }
        this.values[name] = id;
    }
};

protoc.PrimitiveType = class extends protoc.Type {

    constructor(name, long, mapKey, packed, defaultValue) {
        super(null, name);
        this.long = long;
        this.mapKey = mapKey;
        this.packed = packed;
        this.defaultValue = defaultValue;
    }

    static get(name) {
        if (!this._map) {
            this._map = new Map();
            const register = (type) => this._map.set(type.name, type);
            register(new protoc.PrimitiveType('double', false, false, true, 0));
            register(new protoc.PrimitiveType('float', false, false, true, 0));
            register(new protoc.PrimitiveType('int32', false, true, true, 0));
            register(new protoc.PrimitiveType('uint32', false, true, true, 0));
            register(new protoc.PrimitiveType('sint32', false, true, true, 0));
            register(new protoc.PrimitiveType('fixed32', false, true, true, 0));
            register(new protoc.PrimitiveType('sfixed32', false, true, true, 0));
            register(new protoc.PrimitiveType('int64', true, true, true, 0));
            register(new protoc.PrimitiveType('uint64', true, true, true, 0));
            register(new protoc.PrimitiveType('sint64', true, true, true, 0));
            register(new protoc.PrimitiveType('fixed64', true, true, true, 0));
            register(new protoc.PrimitiveType('sfixed64', true, true, true, 0));
            register(new protoc.PrimitiveType('bool', false, true, true, false));
            register(new protoc.PrimitiveType('string', false, true, false, ''));
            register(new protoc.PrimitiveType('bytes', false, true, false, []));
        }
        return this._map.get(name);
    }
};

protoc.Field = class extends protoc.Object {

    constructor(parent, name, id, type, rule, extend) {
        super(parent instanceof protoc.OneOf ? parent.parent : parent, name);
        if (!Number.isInteger(id) || id < 0) {
            throw new protoc.Error('Identifier must be a non-negative integer.');
        }
        if (rule && rule !== 'required' && rule !== 'optional' && rule !== 'repeated') {
            throw new protoc.Error('Rule must be a string.');
        }
        this.id = id;
        this._type = type;
        this.rule = rule && rule !== 'optional' ? rule : undefined;
        this.extend = extend;
        this.required = rule === 'required';
        this.repeated = rule === 'repeated';
        if (parent instanceof protoc.OneOf) {
            this.partOf = parent;
            parent.oneof.set(this.name, this);
            parent = parent.parent;
        }
        if (parent.get(this.name)) {
            throw new protoc.Error(`Duplicate name '${this.name}' in '${parent.name}'.`);
        }
        if (protoc.Namespace.isReservedId(parent.reserved, this.id)) {
            throw new protoc.Error(`Identifier '${this.id}' is reserved in '${parent.name}'.`);
        }
        if (protoc.Namespace.isReservedName(parent.reserved, this.name)) {
            throw new protoc.Error(`Name '${this.name}' is reserved in '${parent.name}'.`);
        }
        parent.fields.set(this.name, this);
    }

    get type() {
        if (typeof this._type === 'string') {
            this._type = protoc.PrimitiveType.get(this._type) || this.parent.findType(this._type);
        }
        return this._type;
    }

    get defaultValue() {
        const type = this.type;
        let value = null;
        if (type instanceof protoc.PrimitiveType) {
            value = type.defaultValue;
        } else if (type instanceof protoc.Enum) {
            value = type.values[Object.keys(type.values)[0]];
        }
        if (this.options.has('default')) {
            value = this.options.get('default');
            if (type instanceof protoc.Enum && typeof value === 'string') {
                value = type.values[value];
            }
        }
        if (type === 'bytes' && typeof value === 'string') {
            throw new protoc.Error('Unsupported bytes field.');
        }
        return value;
    }
};

protoc.OneOf = class extends protoc.Object {

    constructor(parent, name) {
        super(parent, name);
        this.oneof = new Map();
        if (parent.get(this.name)) {
            throw new protoc.Error(`Duplicate name '${this.name}' in '${parent.name}'.`);
        }
        parent.oneofs.set(this.name, this);
    }
};

protoc.MapField = class extends protoc.Field {

    constructor(parent, name, id, keyType, type) {
        super(parent, name, id, type);
        this.keyType = protoc.PrimitiveType.get(keyType);
    }
};

protoc.Parser = class {

    constructor(text, file, root) {
        this._context = root;
        this._tokenizer = new protoc.Tokenizer(text, file);
        this._head = true;
        this._imports = [];
        this._weakImports = [];
    }

    parse() {
        let token = '';
        while ((token = this._tokenizer.next()) !== null) {
            switch (token) {
                case 'package':
                    if (!this._head) {
                        throw this._parseError(token);
                    }
                    this._parsePackage();
                    break;
                case 'import':
                    if (!this._head) {
                        throw this._parseError(token);
                    }
                    this._parseImport();
                    break;
                case 'syntax':
                    if (!this._head) {
                        throw this._parseError(token);
                    }
                    this._parseSyntax();
                    break;
                case 'option':
                    this._parseOption(this._context, token);
                    this._tokenizer.expect(';');
                    break;
                default:
                    if (this._parseCommon(this._context, token)) {
                        this._head = false;
                        continue;
                    }
                    throw this._parseError(token);
            }
        }

        return { package: this._package, imports: this._imports, weakImports: this._weakImports };
    }

    _parseId(token, acceptNegative) {
        switch (token) {
            case 'max':
            case 'Max':
            case 'MAX':
                return 0x1fffffff;
            case '0':
                return 0;
            default: {
                if (!acceptNegative && token.charAt(0) === "-") {
                    throw this._parseError(token, 'id');
                }
                if (/^-?[1-9][0-9]*$/.test(token)) {
                    return parseInt(token, 10);
                }
                if (/^-?0[x][0-9a-fA-F]+$/.test(token)) {
                    return parseInt(token, 16);
                }
                if (/^-?0[0-7]+$/.test(token)) {
                    return parseInt(token, 8);
                }
                throw this._parseError(token, 'id');
            }
        }
    }

    _parsePackage() {
        if (this._package) {
            throw this._parseError("package");
        }
        this._package = this._tokenizer.next();
        if (!protoc.Parser._isTypeReference(this._package)) {
            throw this._parseError(this._package, 'name');
        }
        this._context = this._context.defineNamespace(this._package);
        this._tokenizer.expect(";");
    }

    _parseImport() {
        let token = this._tokenizer.peek();
        if (token === 'weak') {
            this._tokenizer.next();
            token = this._readString();
            this._tokenizer.expect(";");
            this._weakImports.push(token);
        } else {
            if (token === 'public') {
                this._tokenizer.next();
            }
            token = this._readString();
            this._tokenizer.expect(";");
            this._imports.push(token);
        }
    }

    _parseSyntax() {
        this._tokenizer.expect("=");
        this._syntax = this._readString();
        if (this._syntax !== 'proto2' && this._syntax !== 'proto3') {
            throw this._parseError(this._syntax, 'syntax');
        }
        this._tokenizer.expect(";");
    }

    _parseCommon(parent, token) {
        switch (token) {
            case 'option':
                this._parseOption(parent, token);
                this._tokenizer.expect(";");
                return true;
            case 'message':
                this._parseType(parent, token);
                return true;
            case 'enum':
                this._parseEnum(parent, token);
                return true;
            case 'extend':
                this._parseExtend(parent, token);
                return true;
            case 'service':
                throw new protoc.Error(`Keyword '${token}' is not supported ${this._tokenizer.location()}`);
            default:
                return false;
        }
    }

    _ifBlock(obj, ifCallback, elseCallback) {
        if (obj) {
            obj.file = this._file;
        }
        if (this._tokenizer.eat("{")) {
            let token = '';
            while ((token = this._tokenizer.next()) !== "}") {
                ifCallback(token);
            }
            this._tokenizer.eat(";");
        } else {
            if (elseCallback) {
                elseCallback();
            }
            this._tokenizer.expect(";");
        }
    }

    _parseType(parent, token) {
        token = this._tokenizer.next();
        if (!protoc.Parser._isName(token)) {
            throw this._parseError(token, 'type');
        }
        const type = new protoc.Type(parent, token);
        this._ifBlock(type, (token) => {
            if (this._parseCommon(type, token)) {
                return;
            }
            switch (token) {
                case 'map':
                    this._parseMapField(type, token);
                    break;
                case 'required':
                case 'optional':
                case 'repeated':
                    this._parseField(type, token);
                    break;
                case 'oneof':
                    this._parseOneOf(type, token);
                    break;
                case 'reserved':
                    this._readRanges(type.reserved, true);
                    break;
                case 'extensions':
                    this._readRanges(type.extensions);
                    break;
                default:
                    if (this._syntax !== 'proto3' || !protoc.Parser._isTypeReference(token)) {
                        throw this._parseError(token);
                    }
                    this._tokenizer.push(token);
                    this._parseField(type, 'optional');
                    break;
            }
        });
    }

    _parseField(parent, rule, extend) {
        const type = this._tokenizer.next();
        if (type === "group") {
            this._parseGroup(parent, rule);
            return;
        }
        if (!protoc.Parser._isTypeReference(type)) {
            throw this._parseError(type, 'type');
        }
        const name = this._tokenizer.next();
        if (!protoc.Parser._isName(name)) {
            throw this._parseError(name, 'name');
        }
        this._tokenizer.expect("=");
        const id = this._parseId(this._tokenizer.next());
        const field = new protoc.Field(parent, name, id, type, rule, extend);
        this._ifBlock(field, (token) => {
            if (token === "option") {
                this._parseOption(field, token);
                this._tokenizer.expect(";");
            } else {
                throw this._parseError(token);
            }
        }, () => {
            this._parseInlineOptions(field);
        });
    }

    _parseGroup(parent, rule) {
        let name = this._tokenizer.next();
        if (!protoc.Parser._isName(name)) {
            throw this._parseError(name, 'name');
        }
        const fieldName = name.charAt(0).toLowerCase() + name.substring(1);
        if (name === fieldName) {
            name = name.charAt(0).toUpperCase() + name.substring(1);
        }
        this._tokenizer.expect("=");
        const id = this._parseId(this._tokenizer.next());
        const type = new protoc.Type(name);
        type.group = true;
        const field = new protoc.Field(parent, fieldName, id, name, rule);
        field.file = this._file;
        this._ifBlock(type, (token) => {
            switch (token) {
                case "option":
                    this._parseOption(type, token);
                    this._tokenizer.expect(";");
                    break;
                case "required":
                case "optional":
                case "repeated":
                    this._parseField(type, token);
                    break;
                default:
                    throw this._parseError(token); // there are no groups with proto3 semantics
            }
        });
        parent.add(type).add(field);
    }

    _parseMapField(parent) {
        this._tokenizer.expect("<");
        const keyType = this._tokenizer.next();
        const resolvedKeyType = protoc.PrimitiveType.get(keyType);
        if (!resolvedKeyType || !resolvedKeyType.mapKey) {
            throw this._parseError(keyType, 'type');
        }
        this._tokenizer.expect(",");
        const valueType = this._tokenizer.next();
        if (!protoc.Parser._isTypeReference(valueType)) {
            throw this._parseError(valueType, 'type');
        }
        this._tokenizer.expect(">");
        const name = this._tokenizer.next();
        if (!protoc.Parser._isName(name)) {
            throw this._parseError(name, 'name');
        }
        this._tokenizer.expect("=");
        const id = this._parseId(this._tokenizer.next());
        const field = new protoc.MapField(parent, name, id, keyType, valueType);
        this._ifBlock(field, (token) => {
            if (token === "option") {
                this._parseOption(field, token);
                this._tokenizer.expect(";");
            } else {
                throw this._parseError(token);
            }
        }, () => {
            this._parseInlineOptions(field);
        });
    }

    _parseOneOf(parent, token) {

        token = this._tokenizer.next();
        if (!protoc.Parser._isName(token)) {
            throw this._parseError(token, 'name');
        }
        const oneof = new protoc.OneOf(parent, token);
        this._ifBlock(oneof, (token) => {
            if (token === "option") {
                this._parseOption(oneof, token);
                this._tokenizer.expect(";");
            } else {
                this._tokenizer.push(token);
                this._parseField(oneof, 'optional');
            }
        });
    }

    _parseEnum(parent, token) {
        token = this._tokenizer.next();
        if (!protoc.Parser._isName(token)) {
            throw this._parseError(token, 'name');
        }
        const obj = new protoc.Enum(parent, token);
        this._ifBlock(obj, (token) => {
            switch (token) {
                case "option":
                    this._parseOption(obj, token);
                    this._tokenizer.expect(";");
                    break;
                case "reserved":
                    this._readRanges(obj.reserved, true);
                    break;
                default:
                    this._parseEnumValue(obj, token);
                    break;
            }
        });
    }

    _parseEnumValue(parent, token) {
        if (!protoc.Parser._isName(token)) {
            throw this._parseError(token, 'name');
        }
        this._tokenizer.expect("=");
        const value = this._parseId(this._tokenizer.next(), true);
        const dummy = {};
        this._ifBlock(dummy, (token) => {
            if (token === "option") {
                this._parseOption(dummy, token); // skip
                this._tokenizer.expect(";");
            } else {
                throw this._parseError(token);
            }

        }, () => {
            this._parseInlineOptions(dummy); // skip
        });
        parent.add(token, value);
    }

    _parseExtend(parent, token) {
        token = this._tokenizer.next();
        if (!protoc.Parser._isTypeReference(token)) {
            throw this._parseError(token, 'reference');
        }
        const reference = token;
        this._ifBlock(null, (token) => {
            switch (token) {
                case "required":
                case "repeated":
                case "optional":
                    this._parseField(parent, token, reference);
                    break;
                default:
                    if (this._syntax === 'proto3' || !protoc.Parser._isTypeReference(token)) {
                        throw this._parseError(token);
                    }
                    this._tokenizer.push(token);
                    this._parseField(parent, 'optional', reference);
                    break;
            }
        });
    }

    _parseOption(parent, token) {
        const custom = this._tokenizer.eat("(");
        token = this._tokenizer.next();
        if (!protoc.Parser._isTypeReference(token)) {
            throw this._parseError(token, 'name');
        }
        let name = token;
        if (custom) {
            this._tokenizer.expect(")");
            name = `(${name})`;
            token = this._tokenizer.peek();
            if (/^(?:\.[a-zA-Z_][a-zA-Z_0-9]*)+$/.test(token)) {
                name += token;
                this._tokenizer.next();
            }
        }
        this._tokenizer.expect("=");
        this._parseOptionValue(parent, name);
    }

    _parseOptionValue(parent, name) {
        if (this._tokenizer.eat('{')) {
            while (!this._tokenizer.eat('}')) {
                const token = this._tokenizer.next();
                if (!protoc.Parser._isName(token)) {
                    throw this._parseError(token, 'name');
                }
                if (this._tokenizer.peek() === '{') {
                    this._parseOptionValue(parent, `${name}.${token}`);
                } else {
                    this._tokenizer.expect(':');
                    if (this._tokenizer.peek() === '{') {
                        this._parseOptionValue(parent, `${name}.${token}`);
                    } else {
                        parent.options.set(`${name}.${token}`, this._readValue());
                    }
                }
                this._tokenizer.eat(',');
            }
        } else {
            parent.options.set(name, this._readValue());
        }
    }

    _parseInlineOptions(parent) {
        if (this._tokenizer.eat('[')) {
            do {
                this._parseOption(parent, 'option');
            }
            while (this._tokenizer.eat(','));
            this._tokenizer.expect(']');
        }
        return parent;
    }

    _readString() {
        const values = [];
        let token = '';
        do {
            if ((token = this._tokenizer.next()) !== '"' && token !== "'") {
                throw this._parseError(token);
            }
            values.push(this._tokenizer.next());
            this._tokenizer.expect(token);
            token = this._tokenizer.peek();
        }
        while (token === '"' || token === "'");
        return values.join('');
    }

    _readValue() {
        const token = this._tokenizer.next();
        switch (token) {
            case "'":
            case '"':
                this._tokenizer.push(token);
                return this._readString();
            case 'true':
            case 'TRUE':
                return true;
            case 'false':
            case 'FALSE':
                return false;
            default: {
                const value = this._parseNumber(token);
                if (value !== undefined) {
                    return value;
                }
                if (protoc.Parser._isTypeReference(token)) {
                    return token;
                }
                throw this._parseError(token, 'value');
            }
        }
    }

    _readRanges(target, acceptStrings) {
        do {
            let token = '';
            if (acceptStrings && ((token = this._tokenizer.peek()) === '"' || token === "'")) {
                target.push(this._readString());
            } else {
                const start = this._parseId(this._tokenizer.next());
                const end = this._tokenizer.eat('to') ? this._parseId(this._tokenizer.next()) : start;
                target.push([start, end]);
            }
        }
        while (this._tokenizer.eat(','));
        this._tokenizer.expect(';');
    }

    _parseNumber(token) {
        let sign = 1;
        if (token.charAt(0) === '-') {
            sign = -1;
            token = token.substring(1);
        }
        switch (token) {
            case 'inf':
            case 'INF':
            case 'Inf': {
                return sign * Infinity;
            }
            case 'nan':
            case 'NAN':
            case 'Nan':
            case 'NaN': {
                return NaN;
            }
            case '0': {
                return 0;
            }
            default: {
                if (/^[1-9][0-9]*$/.test(token)) {
                    return sign * parseInt(token, 10);
                }
                if (/^0[x][0-9a-fA-F]+$/.test(token)) {
                    return sign * parseInt(token, 16);
                }
                if (/^0[0-7]+$/.test(token)) {
                    return sign * parseInt(token, 8);
                }
                if (/^(?![eE])[0-9]*(?:\.[0-9]*)?(?:[eE][+-]?[0-9]+)?$/.test(token)) {
                    return sign * parseFloat(token);
                }
                return undefined;
            }
        }
    }

    static _isName(value) {
        return /^[a-zA-Z_][a-zA-Z_0-9]*$/.test(value);
    }

    static _isTypeReference(value) {
        return /^(?:\.?[a-zA-Z_][a-zA-Z_0-9]*)(?:\.[a-zA-Z_][a-zA-Z_0-9]*)*$/.test(value);
    }

    _parseError(token, name) {
        name = name || 'token';
        const location = this._tokenizer.location();
        return new protoc.Error(`Invalid ${name} '${token}' ${location}.`);
    }
};

protoc.Tokenizer = class {

    constructor(text, file) {
        this._text = text;
        this._file = file;
        this._position = 0;
        this._length = text.length;
        this._line = 1;
        this._stack = [];
        this._delimiter = null;
    }

    get file() {
        return this._file;
    }

    get line() {
        return this._line;
    }

    next() {
        if (this._stack.length > 0) {
            return this._stack.shift();
        }
        if (this._delimiter) {
            return this._readString();
        }

        let repeat = false;
        let prev = '';
        let curr = '';
        do {
            if (this._position === this._length) {
                return null;
            }
            repeat = false;
            while (/\s/.test(curr = this._get(this._position))) {
                if (curr === '\n') {
                    this._line++;
                }
                this._position++;
                if (this._position === this._length) {
                    return null;
                }
            }
            if (this._get(this._position) === '/') {
                this._position++;
                if (this._position === this._length) {
                    throw this._readError('Invalid comment');
                }
                if (this._get(this._position) === '/') {
                    while (this._get(++this._position) !== '\n') {
                        if (this._position === this._length) {
                            return null;
                        }
                    }
                    this._position++;
                    this._line++;
                    repeat = true;
                } else if ((curr = this._get(this._position)) === '*') {
                    do {
                        if (curr === '\n') {
                            this._line++;
                        }
                        this._position++;
                        if (this._position === this._length) {
                            throw this._readError('Invalid comment');
                        }
                        prev = curr;
                        curr = this._get(this._position);
                    } while (prev !== '*' || curr !== '/');
                    this._position++;
                    repeat = true;
                } else {
                    return '/';
                }
            }
        }
        while (repeat);

        let end = this._position;
        const delimRe = /[\s{}=;:[\],'"()<>]/g;
        delimRe.lastIndex = 0;
        const delim = delimRe.test(this._get(end++));
        if (!delim) {
            while (end < this._length && !delimRe.test(this._get(end))) {
                end++;
            }
        }
        const position = this._position;
        this._position = end;
        const token = this._text.substring(position, this._position);
        if (token === '"' || token === "'") {
            this._delimiter = token;
        }
        return token;
    }

    peek() {
        if (!this._stack.length) {
            const token = this.next();
            if (token === null) {
                return null;
            }
            this.push(token);
        }
        return this._stack[0];
    }

    push(value) {
        this._stack.push(value);
    }

    expect(value) {
        const token = this.peek();
        if (token !== value) {
            throw this._readError(`Unexpected '${token}' instead of '${value}'`);
        }
        this.next();
    }

    eat(value) {
        const token = this.peek();
        if (token === value) {
            this.next();
            return true;
        }
        return false;
    }

    _get(pos) {
        return this._text.charAt(pos);
    }

    static _unescape(str) {
        return str.replace(/\\(.?)/g, ($0, $1) => {
            switch ($1) {
                case '\\':
                case '':
                    return $1;
                case '0':
                    return '\0';
                case 'r':
                    return '\r';
                case 'n':
                    return '\n';
                case 't':
                    return '\t';
                default:
                    return '';
            }
        });
    }

    _readString() {
        const re = this._delimiter === "'" ? /(?:'([^'\\]*(?:\\.[^'\\]*)*)')/g : /(?:"([^"\\]*(?:\\.[^"\\]*)*)")/g;
        re.lastIndex = this._position - 1;
        const match = re.exec(this._text);
        if (!match) {
            throw this._readError('Invalid string');
        }
        this._position = re.lastIndex;
        this.push(this._delimiter);
        this._delimiter = null;
        return protoc.Tokenizer._unescape(match[1]);
    }

    _readError(message) {
        const location = `at ${this._file}:${this._line}.`;
        return new protoc.Error(`${message} ${location}`);
    }

    location() {
        return `at ${this.file}:${this.line}.`;
    }
};

protoc.Generator = class {

    constructor(root, options) {
        this._root = root;
        this._options = options;
        this._builder = new protoc.Generator.StringBuilder();
        const scopes = Array.from(this._root.children.values()).map((child) => child.fullName);
        const exports = new Set(scopes.map((scope) => scope.split('.')[0]));
        this._builder.add('');
        for (const value of exports) {
            this._builder.add(`export const ${value} = {};`);
        }
        this._buildContent(this._root);
        this._content = this._builder.toString();
    }

    get content() {
        return this._content;
    }

    _buildContent(namespace) {
        for (const child of namespace.children.values()) {
            if (child instanceof protoc.Enum) {
                this._builder.add('');
                this._buildEnum(child);
            } else if (child instanceof protoc.Type) {
                this._builder.add('');
                this._buildType(child);
            } else if (child instanceof protoc.Namespace) {
                const name = child.fullName.split('.').map((name) => protoc.Generator._escapeName(name)).join('.');
                if (name.indexOf('.') !== -1) {
                    this._builder.add('');
                    this._builder.add(`${name} = {};`);
                }
                this._buildContent(child);
            } else {
                throw new protoc.Error('Unsupportd namespace child.');
            }
        }
    }

    _buildEnum(type) {
        /* eslint-disable indent */
        const name = type.fullName.split('.').map((name) => protoc.Generator._escapeName(name)).join('.');
        this._builder.add(`${name} = {`);
        this._builder.indent();
            const keys = Object.keys(type.values);
            for (let i = 0; i < keys.length; i++) {
                const key = keys[i];
                const value = type.values[key];
                this._builder.add(`${JSON.stringify(key)}: ${value}${(i === keys.length - 1) ? '' : ','}`);
            }
        this._builder.outdent();
        this._builder.add("};");
        /* eslint-enable indent */
    }

    _buildType(type) {

        const name = type.fullName.split('.').map((name) => protoc.Generator._escapeName(name)).join('.');
        this._builder.add(`${name} = class ${protoc.Generator._escapeName(type.name)} {`);
        this._builder.indent();

        this._buildConstructor(type);

        for (const oneof of type.oneofs.values()) {
            /* eslint-disable indent */
            this._builder.add('');
            this._builder.add(`get ${oneof.name}() {`);
            this._builder.indent();
                this._builder.add(`${name}.${oneof.name}Set = ${name}.${oneof.name}Set || new Set([${Array.from(oneof.oneof.keys()).map(JSON.stringify).join(', ')}]);`);
                this._builder.add(`return Object.keys(this).find((key) => ${name}.${oneof.name}Set.has(key) && this[key] !== null);`);
            this._builder.outdent();
            this._builder.add('}');
            /* eslint-enable indent */
        }

        if (this._options.binary) {
            this._builder.add('');
            this._buildDecodeFunction(type);
        }

        if (this._options.text) {
            this._builder.add('');
            this._buildDecodeTextFunction(type);
        }

        if (this._options.json) {
            this._builder.add('');
            this._buildDecodeJsonFunction(type);
        }

        this._builder.outdent();
        this._builder.add('};');

        let first = true;
        for (const field of type.fields.values()) {
            if (field.partOf || field.repeated || field instanceof protoc.MapField) {
                continue;
            }
            if (first) {
                this._builder.add('');
                first = false;
            }
            if (field.type.long) {
                if (field.type.name === 'uint64' || field.type.name === 'fixed64') {
                    this._builder.add(`${name}.prototype${protoc.Generator._propertyReference(field.name)} = ${field.defaultValue}n;`);
                } else {
                    this._builder.add(`${name}.prototype${protoc.Generator._propertyReference(field.name)} = ${field.defaultValue}n;`);
                }
            } else if (field.type.name === 'bytes') {
                this._builder.add(`${name}.prototype${protoc.Generator._propertyReference(field.name)} = new Uint8Array(${JSON.stringify(Array.prototype.slice.call(field.defaultValue))});`);
            } else {
                this._builder.add(`${name}.prototype${protoc.Generator._propertyReference(field.name)} = ${JSON.stringify(field.defaultValue)};`);
            }
        }

        this._buildContent(type);
    }

    _buildConstructor(type) {
        const filter = (field) => field instanceof protoc.MapField || field.repeated;
        const fields = Array.from(type.fields.values()).filter(filter);
        if (fields.length === 0) {
            return;
        }
        /* eslint-disable indent */
        this._builder.add('');
        this._builder.add('constructor() {');
        this._builder.indent();
            for (const field of fields) {
                if (field instanceof protoc.MapField) {
                    this._builder.add(`this${protoc.Generator._propertyReference(field.name)} = {};`);
                } else if (field.repeated) {
                    this._builder.add(`this${protoc.Generator._propertyReference(field.name)} = [];`);
                }
            }
        this._builder.outdent();
        this._builder.add('}');
        /* eslint-enable indent */
    }

    _buildDecodeFunction(type) {
        /* eslint-disable indent */
        const fieldTypeName = (field) => `${field.type.fullName}`;
        this._builder.add('static decode(reader, length) {');
        this._builder.indent();
            this._builder.add(`const message = new ${type.fullName}();`);
            this._builder.add('const end = length === undefined ? reader.length : reader.position + length;');
            this._builder.add("while (reader.position < end) {");
            this._builder.indent();
                this._builder.add("const tag = reader.uint32();");
                if (type.group) {
                    this._builder.add("if ((tag&7) === 4)");
                    this._builder.indent();
                        this._builder.add("break;");
                    this._builder.outdent();
                }
                this._builder.add("switch (tag >>> 3) {");
                this._builder.indent();
                    for (const field of type.fields.values()) {
                        const variable = `message${protoc.Generator._propertyReference(field.name)}`;
                        this._builder.add(`case ${field.id}:`);
                        this._builder.indent();
                            if (field instanceof protoc.MapField) {
                                const value = field.type instanceof protoc.PrimitiveType ?
                                    `reader.${field.type.name}()` :
                                    `${fieldTypeName(field)}.decode(reader, reader.uint32())`;
                                this._builder.add(`reader.entry(${variable}, () => reader.${field.keyType.name}(), () => ${value});`);
                            } else if (field.repeated) {
                                if (field.type.name === 'float' || field.type.name === 'double') {
                                    this._builder.add(`${variable} = reader.${field.type.name}s(${variable}, tag);`);
                                } else if (field.type instanceof protoc.Enum) {
                                    this._builder.add(`${variable} = reader.array(${variable}, () => reader.int32(), tag);`);
                                } else if (field.type instanceof protoc.PrimitiveType && field.type.packed) {
                                    this._builder.add(`${variable} = reader.array(${variable}, () => reader.${field.type.name}(), tag);`);
                                } else if (field.type instanceof protoc.PrimitiveType) {
                                    this._builder.add(`${variable}.push(reader.${field.type.name}());`);
                                } else if (field.type.group) {
                                    this._builder.add(`${variable}.push(${fieldTypeName(field)}.decode(reader));`);
                                } else {
                                    this._builder.add(`${variable}.push(${fieldTypeName(field)}.decode(reader, reader.uint32()));`);
                                }
                            } else if (field.type instanceof protoc.Enum) {
                                this._builder.add(`${variable} = reader.int32();`);
                            } else if (field.type instanceof protoc.PrimitiveType) {
                                this._builder.add(`${variable} = reader.${field.type.name}();`);
                            } else if (field.type.group) {
                                this._builder.add(`${variable} = ${fieldTypeName(field)}.decode(reader);`);
                            } else {
                                this._builder.add(`${variable} = ${fieldTypeName(field)}.decode(reader, reader.uint32());`);
                            }
                            this._builder.add('break;');
                        this._builder.outdent();
                    }

                    this._builder.add('default:');
                    this._builder.indent();
                        this._builder.add("reader.skipType(tag & 7);");
                        this._builder.add("break;");
                    this._builder.outdent();
                this._builder.outdent();
                this._builder.add("}");
            this._builder.outdent();
            this._builder.add('}');

            for (const field of Array.from(type.fields.values()).filter((field) => field.required)) {
                this._builder.add(`if (!Object.prototype.hasOwnProperty.call(message, '${field.name}')) {`);
                this._builder.indent();
                    this._builder.add(`throw new Error("Excepted '${field.name}'.");`);
                this._builder.outdent();
                this._builder.add('}');
            }

            this._builder.add('return message;');
        this._builder.outdent();
        this._builder.add('}');
        /* eslint-enable indent */
    }

    _buildDecodeTextFunction(type) {
        /* eslint-disable indent */
        this._builder.add('static decodeText(reader) {');
        this._builder.indent();
            if (type.fullName === 'google.protobuf.Any') {
                this._builder.add(`return reader.any(() => new ${type.fullName}());`);
            } else {
                this._builder.add(`const message = new ${type.fullName}();`);
                this._builder.add('reader.start();');
                this._builder.add('while (!reader.end()) {');
                this._builder.indent();
                    this._builder.add('const tag = reader.tag();');
                    this._builder.add('switch (tag) {');
                    this._builder.indent();
                        for (const field of type.fields.values()) {
                            const variable = `message${protoc.Generator._propertyReference(field.name)}`;
                            this._builder.add(`case "${field.name}":`);
                            this._builder.indent();
                                // Map fields
                                if (field instanceof protoc.MapField) {
                                    const value = field.type instanceof protoc.PrimitiveType ?
                                        `reader.${field.type.name}()` :
                                        `${field.type.fullName}.decodeText(reader)`;
                                    this._builder.add(`reader.entry(${variable}, () => reader.${field.keyType.name}(), () => ${value});`);
                                } else if (field.repeated) { // Repeated fields
                                    if (field.type instanceof protoc.Enum) {
                                        this._builder.add(`reader.array(${variable}, () => reader.enum(${field.type.fullName}));`);
                                    } else if (field.type instanceof protoc.PrimitiveType) {
                                        this._builder.add(`reader.array(${variable}, () => reader.${field.type.name}());`);
                                    } else if (field.type.fullName === 'google.protobuf.Any') {
                                        this._builder.add(`reader.anyarray(${variable}, () => new ${field.type.fullName}());`);
                                    } else {
                                        this._builder.add(`${variable}.push(${field.type.fullName}.decodeText(reader));`);
                                    }
                                // Non-repeated
                                } else if (field.type instanceof protoc.Enum) {
                                    this._builder.add(`${variable} = reader.enum(${field.type.fullName});`);
                                } else if (field.type instanceof protoc.PrimitiveType) {
                                    this._builder.add(`${variable} = reader.${field.type.name}();`);
                                } else {
                                    this._builder.add(`${variable} = ${field.type.fullName}.decodeText(reader);`);
                                }
                                this._builder.add("break;");
                            this._builder.outdent();
                        }
                        this._builder.add("default:");
                        this._builder.indent();
                            this._builder.add("reader.field(tag, message);");
                            this._builder.add("break;");
                        this._builder.outdent();
                    this._builder.outdent();
                    this._builder.add('}');
                this._builder.outdent();
                this._builder.add('}');
                for (const field of Array.from(type.fields.values()).filter((field) => field.required)) {
                    this._builder.add(`if (!Object.prototype.hasOwnProperty.call(message, "${field.name}")) {`);
                    this._builder.indent();
                        this._builder.add(`throw new Error("Excepted '${field.name}'.");`);
                    this._builder.outdent();
                    this._builder.add('}');
                }
                this._builder.add('return message;');
            }
        this._builder.outdent();
        this._builder.add('}');
        /* eslint-enable indent */
    }

    _buildDecodeJsonFunction(type) {
        /* eslint-disable indent */
        this._builder.add('static decodeJson(obj) {');
        this._builder.indent();
            if (type.fullName === 'google.protobuf.Any') {
                throw new protoc.Error('Any fields not implemented.');
            } else {
                this._builder.add(`const message = new ${type.fullName}();`);
                for (const field of type.fields.values()) {
                    const json = field.name.replace(/_([a-z])/g, (_, letter) => letter.toUpperCase());
                    const source = `obj.${json}`;
                    const target = `message${protoc.Generator._propertyReference(field.name)}`;
                    if (!field.required) {
                        this._builder.add(`if ('${json}' in obj) {`);
                        this._builder.indent();
                    }
                    if (field instanceof protoc.MapField) {
                        throw new protoc.Error('Map fields not implemented.');
                    } else if (field.repeated) {
                        if (field.type instanceof protoc.PrimitiveType) {
                            if (field.type.name === 'float' || field.type.name === 'double' || field.type.name === 'int32' || field.type.name === 'uint32') {
                                this._builder.add(`${target} = ${source}.map((obj) => Number(obj));`);
                            } else if (field.type.name === 'int64' || field.type.name === 'uint64') {
                                this._builder.add(`${target} = ${source}.map((obj) => BigInt(obj));`);
                            } else if (field.type.name === 'bytes') {
                                this._builder.add(`${target} = ${source}.map((obj) => new Uint8Array(atob(obj)));`);
                            } else if (field.type.name === 'string' || field.type.name === 'bool') {
                                this._builder.add(`${target} = ${source};`);
                            } else {
                                throw new protoc.Error(`Repeated primitive field type '${field.type.name}' not implemented.`);
                            }
                        } else if (field.type instanceof protoc.Enum) {
                            this._builder.add(`${target} = ${source}.map((key) => ${field.type.fullName}[key]);`);
                        } else {
                            this._builder.add(`${target} = ${source}.map((obj) => ${field.type.fullName}.decodeJson(obj));`);
                        }
                    } else if (field.type instanceof protoc.PrimitiveType) {
                        if (field.type.name === 'float' || field.type.name === 'double' || field.type.name === 'int32' || field.type.name === 'uint32') {
                            this._builder.add(`${target} = Number(${source});`);
                        } else if (field.type.name === 'int64') {
                            this._builder.add(`${target} = BigInt(${source});`);
                        } else if (field.type.name === 'bytes') {
                            this._builder.add(`${target} = new Uint8Array(atob(${source}));`);
                        } else if (field.type.name === 'string' || field.type.name === 'bool') {
                            this._builder.add(`${target} = ${source};`);
                        } else {
                            throw new protoc.Error(`Primitive field type '${field.type.name}' not implemented.`);
                        }
                    } else if (field.type instanceof protoc.Enum) {
                        this._builder.add(`${target} = ${field.type.fullName}[${source}];`);
                    } else {
                        this._builder.add(`${target} = ${field.type.fullName}.decodeJson(${source});`);
                    }
                    if (!field.required) {
                        this._builder.outdent();
                        this._builder.add('}');
                    }
               }
                this._builder.add('return message;');
            }
        this._builder.outdent();
        this._builder.add('}');
        /* eslint-enable indent */
    }

    static _isKeyword(name) {
        return /^(?:do|if|in|for|let|new|try|var|case|else|enum|eval|false|null|this|true|void|with|break|catch|class|const|super|throw|while|yield|delete|export|import|public|return|static|switch|typeof|default|extends|finally|package|private|continue|debugger|function|arguments|interface|protected|implements|instanceof)$/.test(name);
    }

    static _escapeName(name) {
        return protoc.Generator._isKeyword(name) ? `${name}_` : name;
    }

    static _propertyReference(name) {
        if (!/^[$\w_]+$/.test(name)) {
            return `["${name.replace(/\\/g, '\\\\').replace(/"/g, "\\\"")}"]`;
        }
        return `.${name}`;
    }
};

protoc.Generator.StringBuilder = class {

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
            throw new protoc.Error('Invalid indentation.');
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

protoc.Error = class extends Error {

    constructor(message) {
        super(message);
        this.name = 'Protocol Buffers Compiler Error';
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
            case '--binary':
                options.binary = true;
                break;
            case '--text':
                options.text = true;
                break;
            case '--json':
                options.json = true;
                break;
            case '--path':
                options.paths.push(args.shift());
                break;
            default:
                if (arg.startsWith('-')) {
                    throw new protoc.Error(`Invalid command line argument '${arg}'.`);
                }
                options.files.push(arg);
                break;
        }
    }

    try {
        const root = new protoc.Root(options.root);
        await root.load(options.paths, options.files);
        const generator = new protoc.Generator(root, options);
        if (options.out) {
            await fs.writeFile(options.out, generator.content, 'utf-8');
        }
    } catch (error) {
        if (error instanceof protoc.Error && !options.verbose) {
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
    path.basename(process.argv[1]) === 'protoc.js') {
    const args = process.argv.slice(2);
    await main(args);
}

export const Root = protoc.Root;
