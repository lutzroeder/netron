
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

    constructor(text, file) {
        this._text = text;
        this._file = file;
        this._position = 0;
        this._line = 1;
        this._column = 1;
        this._keywords = new Set([
            'assert', 'bit', 'bits', 'class', 'code', 'dag', 'def', 'defm', 'defset', 'defvar',
            'dump', 'else', 'false', 'field', 'foreach', 'if', 'in', 'include', 'int',
            'let', 'list', 'multiclass', 'string', 'then', 'true'
        ]);
        this._token = this._tokenize();
    }

    location() {
        return new tablegen.Location(this._file, this._line, this._column);
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
        if (this._isDigit(c)) {
            let pos = 1;
            while (this.peek(pos) && this._isDigit(this.peek(pos))) {
                pos++;
            }
            if (this.peek(pos) && /[a-zA-Z_]/.test(this.peek(pos))) {
                return this._readIdentifier(location);
            }
            return this._readNumber(location);
        }
        if (c === '-' && this._isDigit(this.peek(1))) {
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
                let depth = 1;
                while (this._position < this._text.length && depth > 0) {
                    if (this.peek() === '/' && this.peek(1) === '*') {
                        this._next();
                        this._next();
                        depth++;
                    } else if (this.peek() === '*' && this.peek(1) === '/') {
                        this._next();
                        this._next();
                        depth--;
                    } else {
                        this._next();
                    }
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

        let depth = 1;
        let lineStart = true;
        let lineContent = '';
        while (this._position < this._text.length && depth > 0) {
            const c = this.peek();
            const next = this.peek(1);

            if (c === '[' && next === '{') {
                const trimmedLine = lineContent.trim();
                if (lineStart || trimmedLine === '' || /^(let|def|class)\s/.test(trimmedLine)) {
                    depth++;
                }
                value += c;
                this._next();
                value += next;
                this._next();
                lineContent += c + next;
            } else if (c === '}' && next === ']') {
                depth--;
                if (depth === 0) {
                    this._next();
                    this._next();
                    break;
                }
                value += c;
                this._next();
                value += next;
                this._next();
                lineContent += c + next;
            } else {
                if (c === '\n') {
                    lineStart = true;
                    lineContent = '';
                } else if (c !== ' ' && c !== '\t') {
                    lineStart = false;
                }
                if (c !== '\n') {
                    lineContent += c;
                }
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
        while (this.peek() === '.' && this._isIdentifierStart(this.peek(1))) {
            value += this.peek(); // add dot
            this._next();
            while (this._position < this._text.length && this._isIdentifierChar(this.peek())) {
                value += this.peek();
                this._next();
            }
        }
        const type = this._keywords.has(value) ? 'keyword' : 'id';
        return new tablegen.Token(type, value, location);
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
        this.args = [];
    }

    toString() {
        if (this.args.length === 0) {
            return this.name;
        }
        return `${this.name}<${this.args.map((a) => a.toString()).join(', ')}>`;
    }
};

tablegen.RecordVal = class {

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
        this.templateBindings = new Map(); // parameter name -> bound value
        this.location = null;
        this.parser = parser;
    }

    getValue(name) {
        return this.fields.get(name) || null;
    }

    // Unified resolution function that handles both parent field copying and post-inheritance resolution
    // Inspired by LLVM's Resolver pattern to avoid code duplication
    _resolveInit(init, resolver, visited = new Set()) {
        if (!init || typeof init !== 'object') {
            return init;
        }
        const key = JSON.stringify(init);
        if (visited.has(key)) {
            return init;
        }
        visited.add(key);
        // Resolve def/id type references (template parameters or field access)
        if ((init.type === 'def' || init.type === 'id') && typeof init.value === 'string') {
            // Handle field access syntax (e.g., "meta.mnemonic" or "addrKind.name")
            if (init.value.includes('.')) {
                const [baseName, ...fieldParts] = init.value.split('.');
                // Try to resolve the base
                let current = resolver.resolveReference(baseName);
                if (current) {
                    // Walk through field access chain
                    for (const fieldName of fieldParts) {
                        // Handle def reference
                        if (current && current.type === 'def' && typeof current.value === 'string') {
                            const defName = current.value;
                            const def = this.parser.getDef(defName) || this.parser.getClass(defName);
                            if (def) {
                                const field = def.getValue(fieldName);
                                if (field && field.value) {
                                    current = field.value;
                                } else {
                                    return init; // Field not found
                                }
                            } else {
                                return init; // Def not found
                            }
                        } else if (current && current.type === 'dag' && current.value) {
                            // Handle DAG (anonymous class instantiation)
                            const className = current.value.operator;
                            const templateArgs = current.value.operands.map((op) => op.value);
                            // Instantiate an anonymous class with template arguments
                            // Used for resolving field access like meta.mnemonic where meta is ROCDL_TrLoadOpMeta<...>
                            let instantiated = null;
                            const baseClass = this.parser.getClass(className);
                            if (baseClass) {
                                // Create an anonymous record that inherits from the class
                                instantiated = new tablegen.Record(`<anonymous ${className}>`, this.parser);
                                instantiated.parents = [{ name: className, args: templateArgs }];
                                this.parser.addSubClass(instantiated);
                                instantiated.resolveReferences();
                            }
                            if (instantiated) {
                                const field = instantiated.getValue(fieldName);
                                if (field && field.value) {
                                    current = field.value;
                                } else {
                                    return init;
                                }
                            } else {
                                return init;
                            }
                        } else {
                            return init; // Can't resolve further
                        }
                    }

                    // Successfully resolved the field access chain
                    return this._resolveInit(current, resolver, visited);
                }
            } else {
                // Simple reference without dots
                const resolved = resolver.resolveReference(init.value);
                if (resolved) {
                    return this._resolveInit(resolved, resolver, visited);
                }
            }
        }
        // Recursively resolve nested structures
        if (init.type === 'dag' && init.value) {
            return {
                type: 'dag',
                value: {
                    operator: init.value.operator,
                    operands: init.value.operands.map((op) => ({
                        value: this._resolveInit(op.value, resolver, visited),
                        name: op.name
                    }))
                }
            };
        }
        if (init.type === 'list' && Array.isArray(init.value)) {
            return {
                type: 'list',
                value: init.value.map((v) => this._resolveInit(v, resolver, visited))
            };
        }
        if (init.type === 'bang' && init.value) {
            return {
                type: 'bang',
                value: {
                    op: init.value.op,
                    args: init.value.args.map((arg) => this._resolveInit(arg, resolver, visited))
                }
            };
        }
        if (init.type === 'concat' && Array.isArray(init.value)) {
            const resolvedParts = init.value.map((v) => this._resolveInit(v, resolver, visited));
            // Flatten nested concats recursively: if a resolved part is itself a concat,
            // extract its parts to avoid nested concat structures that cause double evaluation
            const flattenedParts = [];
            const flattenConcat = (part) => {
                if (part && part.type === 'concat' && Array.isArray(part.value)) {
                    for (const subPart of part.value) {
                        flattenConcat(subPart);
                    }
                } else {
                    flattenedParts.push(part);
                }
            };
            for (const part of resolvedParts) {
                flattenConcat(part);
            }
            return {
                type: 'concat',
                value: flattenedParts
            };
        }
        // For other types, return shallow copy for parent field copying, as-is for post-inheritance
        return resolver.shouldCopy ? { ...init } : init;
    }

    // Helper to deep copy a field and resolve parameter references in a specific context
    _copyAndResolveField(field, bindings, parentClass) {
        const resolver = {
            shouldCopy: true,
            resolveReference: (name) => bindings.get(name) || parentClass.templateBindings.get(name) || null
        };
        return {
            name: field.name,
            type: field.type,
            value: this._resolveInit(field.value, resolver)
        };
    }

    // Resolve template parameter references in field values
    // This matches C++ Record::resolveReferences() behavior
    // After inheriting from a templated class, substitute all template parameter
    // references in field values with their bound values from templateBindings
    resolveReferences() {
        if (this.templateBindings.size === 0) {
            return; // No template parameters to substitute
        }
        const findTemplateBinding = (paramName, record = this, visited = new Set()) => {
            if (visited.has(record.name)) {
                return null;
            }
            visited.add(record.name);
            if (record.templateBindings.has(paramName)) {
                return record.templateBindings.get(paramName);
            }
            for (const parent of record.parents) {
                const parentClass = this.parser.classes.get(parent.name);
                if (parentClass) {
                    const paramIndex = parentClass.templateArgs.findIndex((arg) => arg.name === paramName);
                    if (paramIndex !== -1 && parent.args && parent.args[paramIndex]) {
                        return parent.args[paramIndex];
                    }
                    const binding = findTemplateBinding(paramName, parentClass, visited);
                    if (binding) {
                        return binding;
                    }
                }
            }
            return null;
        };
        const resolver = {
            shouldCopy: false,
            resolveReference: (name) => {
                const binding = findTemplateBinding(name);
                if (binding) {
                    return binding;
                }
                const field = this.getValue(name);
                if (field && field.value) {
                    return field.value;
                }
                return null;
            }
        };
        for (const [, field] of this.fields) {
            if (field.value) {
                const resolved = this._resolveInit(field.value, resolver);
                if (resolved !== field.value) {
                    field.value = resolved;
                }
            }
        }
    }

    getValueAsString(fieldName) {
        const field = this.getValue(fieldName);
        if (!field || !field.value) {
            return null;
        }
        const evaluated = this.evaluateValue(field.value);
        if (typeof evaluated === 'string') {
            return evaluated;
        }
        return null;
    }

    getValueAsBit(fieldName) {
        const field = this.getValue(fieldName);
        if (!field || !field.value) {
            return null;
        }
        const evaluated = this.evaluateValue(field.value);
        if (typeof evaluated === 'boolean') {
            return evaluated;
        }
        if (typeof evaluated === 'number') {
            return evaluated !== 0;
        }
        return null;
    }

    getValueAsDag(fieldName) {
        const field = this.getValue(fieldName);
        if (!field || !field.value) {
            return null;
        }
        const evaluated = this.evaluateValue(field.value);
        if (evaluated && typeof evaluated === 'object' && evaluated.operator) {
            return evaluated;
        }
        return null;
    }

    getValueAsDef(fieldName) {
        const field = this.getValue(fieldName);
        if (!field || !field.value) {
            return null;
        }
        if (field.value.type === 'def' && typeof field.value.value === 'string') {
            const defName = field.value.value;
            return this.parser.getDef(defName) || this.parser.getClass(defName);
        }
        return null;
    }

    isEnumAttr() {
        const enumBaseClasses = [
            'EnumAttr',  // Wrapper for dialect-specific enums
            'IntEnumAttr', 'I32EnumAttr', 'I64EnumAttr',
            'BitEnumAttr', 'I8BitEnumAttr', 'I16BitEnumAttr', 'I32BitEnumAttr', 'I64BitEnumAttr',
            'IntEnum', 'I32IntEnum', 'I64IntEnum',  // Integer enum types
            'BitEnum', 'I8BitEnum', 'I16BitEnum', 'I32BitEnum', 'I64BitEnum'  // Bit enum types
        ];
        const checkParents = (record, visited = new Set()) => {
            if (record && !visited.has(record.name)) {
                visited.add(record.name);
                if (enumBaseClasses.includes(record.name)) {
                    return true;
                }
                for (const parent of record.parents) {
                    const parentClass = this.parser.getClass(parent.name);
                    if (parentClass && checkParents(parentClass, visited)) {
                        return true;
                    }
                }
            }
            return false;
        };
        return checkParents(this);
    }

    isEnumProp() {
        const enumPropBaseClasses = [
            'EnumProp',  // Property wrapping an enum
            'EnumPropWithAttrForm'  // Property with attribute form
        ];
        const checkParents = (record, visited = new Set()) => {
            if (record && !visited.has(record.name)) {
                visited.add(record.name);
                if (enumPropBaseClasses.includes(record.name)) {
                    return true;
                }
                for (const parent of record.parents) {
                    const parentClass = this.parser.getClass(parent.name);
                    if (parentClass && checkParents(parentClass, visited)) {
                        return true;
                    }
                }
            }
            return false;
        };
        return checkParents(this);
    }

    // Get enum cases from an enum attribute or property definition
    getEnumCases() {
        if (!this.isEnumAttr() && !this.isEnumProp()) {
            return null;
        }

        // Handle EnumProp - the first argument is the underlying enum
        if (this.isEnumProp()) {
            for (const parent of this.parents) {
                if (parent.name === 'EnumProp' || parent.name === 'EnumPropWithAttrForm') {
                    if (parent.args && parent.args.length >= 1) {
                        const [enumInfoArg] = parent.args;
                        if (enumInfoArg && enumInfoArg.type === 'def' && typeof enumInfoArg.value === 'string') {
                            const enumName = enumInfoArg.value;
                            const underlyingEnum = this.parser.getDef(enumName) || this.parser.getClass(enumName);
                            if (underlyingEnum) {
                                // Recursively get cases from the underlying enum
                                return underlyingEnum.getEnumCases();
                            }
                        }
                    }
                }
                // Recursively search parent classes
                const parentClass = this.parser.getClass(parent.name);
                if (parentClass && parentClass.isEnumProp()) {
                    const cases = parentClass.getEnumCases();
                    if (cases) {
                        return cases;
                    }
                }
            }
        }

        // Helper to search for EnumAttr in the parent hierarchy
        const findEnumAttrParent = (record, visited = new Set()) => {
            if (!record || visited.has(record.name)) {
                return null;
            }
            visited.add(record.name);
            for (const parent of record.parents) {
                if (parent.name === 'EnumAttr') {
                    return parent;
                }
                // Recursively search parent classes
                const parentClass = this.parser.getClass(parent.name);
                if (parentClass) {
                    const found = findEnumAttrParent(parentClass, visited);
                    if (found) {
                        return found;
                    }
                }
            }
            return null;
        };
        const enumAttrParent = findEnumAttrParent(this);
        // Pattern 1a: EnumAttr<Dialect, EnumInfo (as def), name>
        // The 2nd argument is a reference to the underlying enum (e.g., GPU_Dimension)
        if (enumAttrParent && enumAttrParent.args && enumAttrParent.args.length >= 2) {
            const [dialectArg, enumInfoArg] = enumAttrParent.args;
            if (enumInfoArg && enumInfoArg.type === 'def' && typeof enumInfoArg.value === 'string') {
                // Get the expected namespace from the dialect
                let expectedNamespace = null;
                if (dialectArg && dialectArg.type === 'def') {
                    const dialectDef = this.parser.getDef(dialectArg.value) || this.parser.getClass(dialectArg.value);
                    if (dialectDef) {
                        expectedNamespace = dialectDef.getValueAsString('cppNamespace');
                    }
                }
                // Find the enum with the matching name and namespace
                // Try all defs with this name in case of conflicts (different namespaces)
                const enumName = enumInfoArg.value;
                let underlyingEnum = null;
                if (expectedNamespace) {
                    // Normalize namespace for comparison (handle both "vector" and "::mlir::vector" formats)
                    const normalizeNamespace = (ns) => {
                        if (!ns) {
                            return null;
                        }
                        // Remove leading :: and extract the last component
                        const parts = ns.replace(/^::/, '').split('::');
                        return parts[parts.length - 1];
                    };
                    const normalizedExpected = normalizeNamespace(expectedNamespace);

                    // Search through all defs to find one with matching name AND namespace
                    for (const def of this.parser.defs) {
                        if (def.name === enumName) {
                            const defNamespace = def.getValueAsString('cppNamespace');
                            const normalizedDef = normalizeNamespace(defNamespace);
                            if (normalizedDef === normalizedExpected) {
                                underlyingEnum = def;
                                break;
                            }
                        }
                    }
                }

                // Fallback to regular lookup if namespace matching fails
                if (!underlyingEnum) {
                    underlyingEnum = this.parser.getDef(enumName) || this.parser.getClass(enumName);
                }

                if (underlyingEnum) {
                    return underlyingEnum.getEnumCases();
                }
            }
            // Pattern 1b: EnumAttr<Dialect, EnumInfo (as DAG), name>
            // The 2nd argument is a DAG that instantiates an enum class template
            // e.g., EnumAttr<SPIRV_Dialect, SPIRV_I32Enum<"Scope", "desc", [cases]>, "scope">
            if (enumInfoArg && enumInfoArg.type === 'dag' && enumInfoArg.value) {
                // The DAG operator is the enum class template (e.g., SPIRV_I32Enum)
                // We need to find the actual cases by looking at this record's parent args
                // which should have the instantiated template parameters
                // Search through this record's parents to find one with the cases list
                for (const parent of this.parents) {
                    if (parent.args && parent.args.length >= 3) {
                        // Look for a list argument that contains enum case defs
                        for (const arg of parent.args) {
                            if (arg.type === 'list' && Array.isArray(arg.value)) {
                                // Check if this looks like an enum case list
                                const [firstItem] = arg.value;
                                if (firstItem && firstItem.type === 'def') {
                                    // Try to extract cases from this list
                                    const cases = [];
                                    for (const caseValue of arg.value) {
                                        if (caseValue.type === 'def' && typeof caseValue.value === 'string') {
                                            const caseDef = this.parser.getDef(caseValue.value) || this.parser.getClass(caseValue.value);
                                            if (caseDef) {
                                                const str = caseDef.getValueAsString('str');
                                                if (str) {
                                                    cases.push(str);
                                                }
                                            }
                                        }
                                    }
                                    if (cases.length > 0) {
                                        return cases;
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        // Pattern 2: I64EnumAttr<name, summary, [cases]>
        // Cases are in the 3rd template argument
        for (const parent of this.parents) {
            // The 3rd argument should be the cases list
            if (parent.args && parent.args.length >= 3) {
                const [,,casesArg] = parent.args;
                if (casesArg && casesArg.type === 'list' && Array.isArray(casesArg.value)) {
                    const cases = [];
                    for (const caseValue of casesArg.value) {
                        // Each case can be either a DAG or a def reference
                        if (caseValue.type === 'dag' && caseValue.value) {
                            // DAG format: I64EnumAttrCase<"symbol", value, "string">
                            // The string representation is in the 3rd operand, or 1st if only 2 operands
                            const operands = caseValue.value.operands;
                            if (operands && operands.length > 0) {
                                // Try the 3rd operand first (string), fall back to 1st (symbol)
                                const strOperand = operands.length >= 3 ? operands[2] : operands[0];
                                if (strOperand && strOperand.value) {
                                    const str = this.evaluateValue(strOperand.value);
                                    if (str && typeof str === 'string') {
                                        cases.push(str);
                                    }
                                }
                            }
                        } else if (caseValue.type === 'def' && typeof caseValue.value === 'string') {
                            // Def reference format
                            const caseDef = this.parser.getDef(caseValue.value) || this.parser.getClass(caseValue.value);
                            if (caseDef) {
                                const str = caseDef.getValueAsString('str');
                                if (str) {
                                    cases.push(str);
                                }
                            }
                        }
                    }
                    return cases.length > 0 ? cases : null;
                }
            }
        }
        // Pattern 3: LLVM_EnumAttr<name, cppType, summary, [cases]>
        // Cases are in the 4th template argument
        for (const parent of this.parents) {
            if (parent.args && parent.args.length >= 4) {
                const [,,,casesArg] = parent.args;
                if (casesArg && casesArg.type === 'list' && Array.isArray(casesArg.value)) {
                    const cases = [];
                    for (const caseValue of casesArg.value) {
                        // Each case is a def reference
                        if (caseValue.type === 'def' && typeof caseValue.value === 'string') {
                            const caseDef = this.parser.getDef(caseValue.value) || this.parser.getClass(caseValue.value);
                            if (caseDef) {
                                const str = caseDef.getValueAsString('str');
                                if (str) {
                                    cases.push(str);
                                }
                            }
                        }
                    }
                    return cases.length > 0 ? cases : null;
                }
            }
        }
        return null;
    }

    evaluateValue(value) {
        if (!value) {
            return null;
        }
        // Handle named values (e.g., { name: 'clauses', value: { type: 'list', ... } })
        // These come from named arguments in template instantiation
        if (!value.type && value.name && value.value) {
            return this.evaluateValue(value.value);
        }
        switch (value.type) {
            case 'string':
                return value.value.replace(/^"|"$/g, '');
            case 'code':
                return value.value;
            case 'int':
                return parseInt(value.value, 10);
            case 'bit':
                return value.value === 'true' || value.value === '1';
            case 'list':
                return value.value.map((v) => this.evaluateValue(v));
            case 'concat': {
                const parts = value.value.map((v) => this.evaluateValue(v)).filter((v) => v !== null && v !== undefined && v !== '');
                return parts.join('');
            }
            case 'id': {
                const fieldName = value.value;
                if (this.templateBindings.has(fieldName)) {
                    return this.evaluateValue(this.templateBindings.get(fieldName));
                }
                const field = this.getValue(fieldName);
                if (field && field.value) {
                    return this.evaluateValue(field.value);
                }
                return null;
            }
            case 'def': {
                const defName = typeof value.value === 'string' ? value.value : value.value.value;
                if (defName.includes('.')) {
                    const parts = defName.split('.');
                    const [baseName, ...fieldPath] = parts;
                    let baseValue = null;
                    if (this.templateBindings.has(baseName)) {
                        baseValue = this.evaluateValue(this.templateBindings.get(baseName));
                    } else {
                        const field = this.getValue(baseName);
                        if (field && field.value) {
                            baseValue = this.evaluateValue(field.value);
                        }
                    }
                    if (typeof baseValue === 'string') {
                        const def = this.parser.getDef(baseValue) || this.parser.getClass(baseValue);
                        if (def) {
                            let current = def;
                            for (const fieldName of fieldPath) {
                                const field = current.getValue(fieldName);
                                if (!field || !field.value) {
                                    return null;
                                }
                                const evaluated = current.evaluateValue(field.value);
                                // If it's another def name, continue navigation
                                if (typeof evaluated === 'string' && (this.parser.getDef(evaluated) || this.parser.getClass(evaluated))) {
                                    current = this.parser.getDef(evaluated) || this.parser.getClass(evaluated);
                                } else {
                                    return evaluated;
                                }
                            }
                            return null;
                        }
                    }
                    return null;
                }
                if (this.templateBindings.has(defName)) {
                    return this.evaluateValue(this.templateBindings.get(defName));
                }
                const field = this.getValue(defName);
                if (field && field.value) {
                    return this.evaluateValue(field.value);
                }
                const def = this.parser.getDef(defName) || this.parser.getClass(defName);
                return def ? defName : null;
            }
            case 'bang': {
                const { op, args } = value.value;
                switch (op) {
                    case 'if': {
                        if (args.length < 2) {
                            return null;
                        }
                        const condition = this.evaluateValue(args[0]);
                        if (condition) {
                            return args.length > 1 ? this.evaluateValue(args[1]) : null;
                        }
                        return args.length > 2 ? this.evaluateValue(args[2]) : null;
                    }
                    case 'empty': {
                        if (args.length < 1) {
                            return true;
                        }
                        const val = this.evaluateValue(args[0]);
                        if (Array.isArray(val)) {
                            return val.length === 0;
                        }
                        if (typeof val === 'string') {
                            return val.length === 0;
                        }
                        return val === null || val === undefined;
                    }
                    case 'interleave': {
                        if (args.length < 2) {
                            return '';
                        }
                        const list = this.evaluateValue(args[0]);
                        const separator = this.evaluateValue(args[1]);
                        if (!Array.isArray(list)) {
                            return '';
                        }
                        return list.filter((x) => x !== null && x !== '').join(separator || '');
                    }
                    case 'not':
                        if (args.length < 1) {
                            return true;
                        }
                        return !this.evaluateValue(args[0]);
                    case 'or':
                        for (const arg of args) {
                            if (this.evaluateValue(arg)) {
                                return true;
                            }
                        }
                        return false;
                    case 'and':
                        for (const arg of args) {
                            if (!this.evaluateValue(arg)) {
                                return false;
                            }
                        }
                        return true;
                    case 'foldl': {
                        // !foldl(init, list, acc, item, expr)
                        // Fold left: iterate over list, accumulating results
                        if (args.length < 5) {
                            return null;
                        }
                        let accumulator = this.evaluateValue(args[0]); // init value
                        const list = this.evaluateValue(args[1]); // list to fold over
                        // args[2] is the accumulator variable name (not evaluated)
                        // args[3] is the item variable name (not evaluated)
                        // args[4] is the expression to evaluate
                        if (!Array.isArray(list)) {
                            return accumulator;
                        }
                        for (const item of list) {
                            const accName = args[2].value; // variable name
                            const itemName = args[3].value; // variable name
                            const prevAcc = this.fields.get(accName);
                            const prevItem = this.fields.get(itemName);
                            let accValue = null;
                            if (accumulator && typeof accumulator === 'object' && accumulator.operator) {
                                // It's a DAG object
                                accValue = new tablegen.Value('dag', accumulator);
                            } else if (Array.isArray(accumulator)) {
                                // It's a list
                                accValue = new tablegen.Value('list', accumulator);
                            } else if (typeof accumulator === 'string') {
                                accValue = new tablegen.Value('string', accumulator);
                            } else if (typeof accumulator === 'number') {
                                accValue = new tablegen.Value('int', accumulator);
                            } else {
                                accValue = accumulator;
                            }

                            this.fields.set(accName, new tablegen.RecordVal(accName, null, accValue));
                            let itemValue = item;
                            if (typeof item === 'string') {
                                itemValue = new tablegen.Value('def', item);
                            } else if (typeof item === 'number') {
                                itemValue = new tablegen.Value('int', item);
                            } else if (typeof item === 'boolean') {
                                itemValue = new tablegen.Value('bit', item ? '1' : '0');
                            } else if (item && typeof item === 'object' && !item.type) {
                                itemValue = new tablegen.Value('dag', item);
                            }
                            this.fields.set(itemName, new tablegen.RecordVal(itemName, null, itemValue));
                            accumulator = this.evaluateValue(args[4]);
                            if (prevAcc) {
                                this.fields.set(accName, prevAcc);
                            } else {
                                this.fields.delete(accName);
                            }
                            if (prevItem) {
                                this.fields.set(itemName, prevItem);
                            } else {
                                this.fields.delete(itemName);
                            }
                        }
                        return accumulator;
                    }
                    case 'foreach': {
                        // !foreach(item, list, expr)
                        // Map: iterate over list, transforming each item
                        if (args.length < 3) {
                            return [];
                        }
                        const itemName = args[0].value;
                        const list = this.evaluateValue(args[1]);
                        const results = [];
                        if (Array.isArray(list)) {
                            for (const item of list) {
                                const prevItem = this.fields.get(itemName);
                                let itemValue = item;
                                if (typeof item === 'string') {
                                    itemValue = new tablegen.Value('def', item);
                                } else if (typeof item === 'number') {
                                    itemValue = new tablegen.Value('int', item);
                                } else if (typeof item === 'boolean') {
                                    itemValue = new tablegen.Value('bit', item ? '1' : '0');
                                } else if (item && typeof item === 'object' && !item.type) {
                                    itemValue = new tablegen.Value('dag', item);
                                }
                                this.fields.set(itemName, new tablegen.RecordVal(itemName, null, itemValue));
                                const result = this.evaluateValue(args[2]);
                                results.push(result);
                                if (prevItem) {
                                    this.fields.set(itemName, prevItem);
                                } else {
                                    this.fields.delete(itemName);
                                }
                            }
                        }
                        return results;
                    }
                    case 'filter': {
                        // !filter(item, list, predicate)
                        // Filter: keep items where predicate is true
                        if (args.length < 3) {
                            return [];
                        }
                        const itemName = args[0].value;
                        const list = this.evaluateValue(args[1]);
                        const results = [];
                        if (Array.isArray(list)) {
                            for (const item of list) {
                                const prevItem = this.fields.get(itemName);
                                // Wrap item in a Value so it can be used in expressions
                                let itemValue = item;
                                if (typeof item === 'string') {
                                    // If it's a def name, wrap it as a 'def' Value
                                    itemValue = new tablegen.Value('def', item);
                                } else if (typeof item === 'number') {
                                    itemValue = new tablegen.Value('int', item);
                                } else if (typeof item === 'boolean') {
                                    itemValue = new tablegen.Value('bit', item ? '1' : '0');
                                } else if (item && typeof item === 'object' && !item.type) {
                                    // If it's a raw object (like a DAG), wrap it
                                    itemValue = new tablegen.Value('dag', item);
                                }
                                // If item is already a Value, use it as is
                                this.fields.set(itemName, new tablegen.RecordVal(itemName, null, itemValue));
                                const keep = this.evaluateValue(args[2]);
                                if (keep) {
                                    results.push(item);
                                }
                                if (prevItem) {
                                    this.fields.set(itemName, prevItem);
                                } else {
                                    this.fields.delete(itemName);
                                }
                            }
                        }
                        return results;
                    }
                    case 'con': {
                        // !con(dag1, dag2, ...)
                        // Concatenate dags - merge operands from multiple dags
                        if (args.length === 0) {
                            return new tablegen.DAG('ins', []);
                        }
                        let operator = 'ins';
                        const allOperands = [];
                        for (const arg of args) {
                            let dagToProcess = null;
                            const evaluated = this.evaluateValue(arg);
                            if (evaluated && typeof evaluated === 'object') {
                                if (evaluated.operator && evaluated.operands) {
                                    dagToProcess = evaluated;
                                } else if (evaluated.type === 'dag' && evaluated.value) {
                                    dagToProcess = evaluated.value;
                                }
                            }
                            if (!dagToProcess && arg.type === 'dag') {
                                dagToProcess = arg.value;
                            }
                            if (dagToProcess && dagToProcess.operands) {
                                if (operator === 'ins' && dagToProcess.operator) {
                                    operator = dagToProcess.operator;
                                }
                                allOperands.push(...dagToProcess.operands);
                            }
                        }
                        return new tablegen.DAG(operator, allOperands);
                    }
                    case 'listconcat': {
                        // !listconcat(list1, list2, ...)
                        // Concatenate multiple lists into one
                        const result = [];
                        for (const arg of args) {
                            const list = this.evaluateValue(arg);
                            if (Array.isArray(list)) {
                                result.push(...list);
                            }
                        }
                        return result;
                    }
                    case 'listremove': {
                        // !listremove(list, items_to_remove)
                        // Remove all occurrences of items_to_remove from list
                        if (args.length < 2) {
                            return [];
                        }
                        const list = this.evaluateValue(args[0]);
                        const toRemove = this.evaluateValue(args[1]);
                        if (!Array.isArray(list)) {
                            return [];
                        }
                        const removeSet = new Set();
                        if (Array.isArray(toRemove)) {
                            for (const item of toRemove) {
                                removeSet.add(JSON.stringify(item));
                            }
                        } else {
                            removeSet.add(JSON.stringify(toRemove));
                        }
                        return list.filter((item) => !removeSet.has(JSON.stringify(item)));
                    }
                    case 'cast': {
                        // !cast<type>(value)
                        // Cast value to type - for now just convert to string
                        if (args.length < 1) {
                            return null;
                        }
                        const val = this.evaluateValue(args[0]);
                        if (val === null || val === undefined) {
                            return null;
                        }
                        return String(val);
                    }
                    case 'eq': {
                        // !eq(a, b)
                        // Return true if a equals b
                        if (args.length < 2) {
                            return false;
                        }
                        const a = this.evaluateValue(args[0]);
                        const b = this.evaluateValue(args[1]);
                        return a === b;
                    }
                    case 'ne': {
                        // !ne(a, b)
                        // Return true if a not equals b
                        if (args.length < 2) {
                            return false;
                        }
                        const a = this.evaluateValue(args[0]);
                        const b = this.evaluateValue(args[1]);
                        return a !== b;
                    }
                    case 'lt': {
                        // !lt(a, b)
                        // Return true if a < b
                        if (args.length < 2) {
                            return false;
                        }
                        const a = this.evaluateValue(args[0]);
                        const b = this.evaluateValue(args[1]);
                        return a < b;
                    }
                    case 'le': {
                        // !le(a, b)
                        // Return true if a <= b
                        if (args.length < 2) {
                            return false;
                        }
                        const a = this.evaluateValue(args[0]);
                        const b = this.evaluateValue(args[1]);
                        return a <= b;
                    }
                    case 'gt': {
                        // !gt(a, b)
                        // Return true if a > b
                        if (args.length < 2) {
                            return false;
                        }
                        const a = this.evaluateValue(args[0]);
                        const b = this.evaluateValue(args[1]);
                        return a > b;
                    }
                    case 'ge': {
                        // !ge(a, b)
                        // Return true if a >= b
                        if (args.length < 2) {
                            return false;
                        }
                        const a = this.evaluateValue(args[0]);
                        const b = this.evaluateValue(args[1]);
                        return a >= b;
                    }
                    case 'range': {
                        // !range(n) or !range(start, end) or !range(start, end, step)
                        // Generate a list of integers
                        if (args.length === 0) {
                            return [];
                        }
                        let start = 0;
                        let end = 0;
                        let step = 1;
                        if (args.length === 1) {
                            end = this.evaluateValue(args[0]);
                        } else if (args.length === 2) {
                            start = this.evaluateValue(args[0]);
                            end = this.evaluateValue(args[1]);
                        } else {
                            start = this.evaluateValue(args[0]);
                            end = this.evaluateValue(args[1]);
                            step = this.evaluateValue(args[2]);
                        }
                        const result = [];
                        if (step > 0) {
                            for (let i = start; i < end; i += step) {
                                result.push(i);
                            }
                        } else if (step < 0) {
                            for (let i = start; i > end; i += step) {
                                result.push(i);
                            }
                        }
                        return result;
                    }
                    case 'listsplat': {
                        // !listsplat(element, n)
                        // Create a list with n copies of element
                        if (args.length < 2) {
                            return [];
                        }
                        const [element] = args; // Don't evaluate yet, keep as Value
                        const count = this.evaluateValue(args[1]);
                        const result = [];
                        for (let i = 0; i < count; i++) {
                            result.push(element);
                        }
                        return result;
                    }
                    case 'cond': {
                        // !cond(condition1: value1, condition2: value2, ..., true: defaultValue)
                        // Evaluate conditions in order, return first matching value
                        for (let i = 0; i < args.length; i++) {
                            const arg = args[i];
                            if (arg.condition) {
                                const condition = this.evaluateValue(arg.condition);
                                if (condition === true || condition === 1) {
                                    return this.evaluateValue(arg.value);
                                }
                            }
                        }
                        return null;
                    }
                    case 'dag': {
                        // !dag(operator, operands_list, names_list)
                        // Construct a DAG from operator, operands, and names
                        if (args.length < 2) {
                            return new tablegen.DAG('ins', []);
                        }
                        const operatorArg = this.evaluateValue(args[0]);
                        const operator = typeof operatorArg === 'string' ? operatorArg : 'ins';
                        const operandsList = this.evaluateValue(args[1]);
                        const namesList = args.length > 2 ? this.evaluateValue(args[2]) : [];
                        const operands = [];
                        if (Array.isArray(operandsList)) {
                            for (let i = 0; i < operandsList.length; i++) {
                                const value = operandsList[i];
                                const name = Array.isArray(namesList) && i < namesList.length ? namesList[i] : '';
                                operands.push({ value, name });
                            }
                        }
                        return new tablegen.DAG(operator, operands);
                    }
                    case 'tolower': {
                        // !tolower(string)
                        // Convert string to lowercase
                        if (args.length < 1) {
                            return null;
                        }
                        const str = this.evaluateValue(args[0]);
                        if (typeof str === 'string') {
                            return str.toLowerCase();
                        }
                        return null;
                    }
                    case 'toupper': {
                        // !toupper(string)
                        // Convert string to uppercase
                        if (args.length < 1) {
                            return null;
                        }
                        const str = this.evaluateValue(args[0]);
                        if (typeof str === 'string') {
                            return str.toUpperCase();
                        }
                        return null;
                    }
                    case 'strconcat': {
                        // !strconcat(str1, str2, ...)
                        // Concatenate strings
                        const parts = [];
                        for (const arg of args) {
                            const val = this.evaluateValue(arg);
                            if (val !== null && val !== undefined) {
                                parts.push(String(val));
                            }
                        }
                        return parts.join('');
                    }
                    case 'subst': {
                        // !subst(pattern, replacement, string)
                        // Replace all occurrences of pattern with replacement in string
                        if (args.length < 3) {
                            return null;
                        }
                        const pattern = this.evaluateValue(args[0]);
                        const replacement = this.evaluateValue(args[1]);
                        const str = this.evaluateValue(args[2]);
                        if (typeof str === 'string' && typeof pattern === 'string') {
                            const rep = replacement !== null && replacement !== undefined ? String(replacement) : '';
                            // Use split/join for global replacement
                            return str.split(pattern).join(rep);
                        }
                        return str;
                    }
                    default:
                        return null;
                }
            }
            case 'dag':
                return value.value;
            case 'uninitialized':
                return null;
            default:
                return null;
        }
    }
};

tablegen.Reader = class {

    constructor() {
        this._paths = [];
        this._includes = new Set();
        this._defs = new Map();
        this.defs = [];
        this.classes = new Map();
    }

    async parse(files, paths) {
        this._paths = paths || [];
        for (const file of files) {
            // eslint-disable-next-line no-await-in-loop
            await this._parseFile(file);
        }
    }

    getDef(name) {
        return this._defs.get(name);
    }

    getClass(name) {
        return this.classes.get(name);
    }

    // Add a subclass to a record, processing template parameters and copying fields
    // This mirrors LLVM's TGParser::AddSubClass behavior
    addSubClass(record) {
        // Track which fields were explicitly defined in the def via 'let' statements
        // These should never be overwritten by parent class fields
        const explicitFields = new Set(record.fields.keys());
        // Step 1: Build initial template bindings for this record from its immediate parents
        const recordBindings = new Map();
        for (const parent of record.parents) {
            const parentClass = this.classes.get(parent.name);
            if (parentClass && parentClass.templateArgs && parentClass.templateArgs.length > 0) {
                const templateArgs = parent.args || [];
                for (let i = 0; i < parentClass.templateArgs.length; i++) {
                    const param = parentClass.templateArgs[i];
                    let boundValue = null;
                    // Check for named argument, positional argument, or default value
                    const namedArg = templateArgs.find((arg) => arg.name === param.name);
                    if (namedArg) {
                        boundValue = namedArg.value;
                    } else if (i < templateArgs.length) {
                        const arg = templateArgs[i];
                        boundValue = arg.name ? arg.value : arg;
                    } else if (param.defaultValue) {
                        boundValue = param.defaultValue;
                    }
                    if (boundValue) {
                        recordBindings.set(param.name, boundValue);
                    }
                }
            }
        }
        record.templateBindings = recordBindings;
        // Step 2: Process parents and flatten fields
        // Helper to resolve Init values using a set of bindings
        // Uses record._resolveInit to avoid code duplication
        const resolveInitValue = (value, bindings) => {
            const resolver = {
                shouldCopy: false,
                resolveReference: (name) => bindings.get(name) || null
            };
            return record._resolveInit(value, resolver);
        };
        const processParent = (parent, currentBindings, visited = new Set()) => {
            if (visited.has(parent.name)) {
                return;
            }
            visited.add(parent.name);
            const parentClass = this.classes.get(parent.name);
            if (!parentClass) {
                return;
            }
            const parentBindings = new Map();
            if (parentClass.templateArgs && parent.args) {
                for (let i = 0; i < parentClass.templateArgs.length && i < parent.args.length; i++) {
                    const paramName = parentClass.templateArgs[i].name;
                    const argValue = parent.args[i];
                    const resolvedArg = resolveInitValue(argValue, currentBindings);
                    parentBindings.set(paramName, resolvedArg);
                }
            }
            for (const grandparent of parentClass.parents) {
                processParent(grandparent, parentBindings, visited);
            }
            for (const [fieldName, field] of parentClass.fields) {
                // Only protect fields that were explicitly defined in the def via 'let'
                // Fields inherited from grandparents should be overwritten by parent class fields
                if (explicitFields.has(fieldName)) {
                    // Check if the explicit field is empty/uninitialized - if so, copy from parent
                    const existingField = record.fields.get(fieldName);
                    const existingIsUninit = existingField.value?.type === 'uninitialized';
                    const existingIsEmptyDag = existingField.value?.type === 'dag' && existingField.value?.value?.operands?.length === 0;
                    const existingIsEmptyString = existingField.value?.type === 'string' && existingField.value?.value === '';
                    const existingIsFalseBit = existingField.value?.type === 'int' && existingField.value?.value === 0 && existingField.type?.name === 'bit';
                    if (existingIsUninit || existingIsEmptyDag || existingIsEmptyString || existingIsFalseBit) {
                        const resolvedField = record._copyAndResolveField(field, parentBindings, parentClass);
                        record.fields.set(fieldName, resolvedField);
                    }
                    // Otherwise, keep the explicit field (child's 'let' wins)
                } else {
                    // Field is not explicit - copy from parent (overwrites grandparent values)
                    const resolvedField = record._copyAndResolveField(field, parentBindings, parentClass);
                    record.fields.set(fieldName, resolvedField);
                }
            }
        };
        for (const parent of record.parents) {
            processParent(parent, recordBindings, new Set());
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
        for (const current of this._paths) {
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
        if (!this._includes.has(location)) {
            this._includes.add(location);
            const content = await fs.readFile(location, 'utf-8');
            this._tokenizer = new tablegen.Tokenizer(content, location);
            while (!this._match('eof')) {
                const token = this._tokenizer.current();
                if (token.type === 'keyword') {
                    switch (token.value) {
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
                } else {
                    this._read();
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
        // Don't process parent classes for class definitions - only for defs/instances
        // Classes are templates that get instantiated later with concrete template arguments
        // Processing parents here with unbound template parameters causes incorrect resolutions
        // this.addSubClass(record);
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
        this.addSubClass(def);
        def.resolveReferences();
        if (name) {
            this._defs.set(name, def);
            this.defs.push(def);
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
        const location = this._tokenizer.location();
        this._read();
        const iterVarName = this._expect('id');
        this._expect('=');
        const listValue = this._parseForeachListValue();
        this._expect('keyword', 'in');
        const loop = { location, iterVarName, listValue, entries: [], hasDefvar: false };
        if (this._match('{')) {
            this._read();
            this._parseForeachBody(loop);
            this._expect('}');
        } else {
            this._parseForeachBodyStatement(loop);
        }
        this._resolveForeachLoop(loop, new Map());
    }

    _parseForeachListValue() {
        const values = [];
        if (this._eat('[')) {
            while (!this._match(']') && !this._match('eof')) {
                const value = this._parseListItem();
                if (value && value.type === 'dag') {
                    const instantiated = this._instantiateClassTemplate(value.value);
                    if (instantiated) {
                        values.push(instantiated);
                    } else {
                        values.push(value);
                    }
                } else {
                    values.push(value);
                }
                this._eat(',');
            }
            this._expect(']');
        } else if (this._eat('!')) {
            const op = this._expect('id');
            if (op === 'range' && this._eat('(')) {
                const args = [];
                while (!this._match(')') && !this._match('eof')) {
                    args.push(this._parseValue());
                    this._eat(',');
                }
                this._expect(')');
                if (args.length >= 1) {
                    let start = 0;
                    let end = 0;
                    if (args.length === 1) {
                        end = this._evaluateSimpleValue(args[0]);
                    } else {
                        start = this._evaluateSimpleValue(args[0]);
                        end = this._evaluateSimpleValue(args[1]);
                    }
                    for (let i = start; i < end; i++) {
                        values.push(new tablegen.Value('int', i));
                    }
                }
            } else {
                while (!this._match('keyword', 'in') && !this._match('eof')) {
                    this._read();
                }
            }
        } else if (this._eat('{')) {
            const start = this._expect('number');
            if (this._eat('-') || this._eat('...')) {
                const end = this._expect('number');
                for (let i = start; i <= end; i++) {
                    values.push(new tablegen.Value('int', i));
                }
            }
            this._expect('}');
        } else {
            while (!this._match('keyword', 'in') && !this._match('eof')) {
                this._read();
            }
        }
        return values;
    }

    _instantiateClassTemplate(dag) {
        if (!dag || !dag.operator) {
            return null;
        }
        const className = typeof dag.operator === 'string' ? dag.operator : dag.operator.value;
        const classRecord = this.classes.get(className);
        if (!classRecord) {
            return null;
        }
        const fields = new Map();
        const bindings = new Map();
        if (classRecord.templateArgs && dag.operands) {
            for (let i = 0; i < classRecord.templateArgs.length && i < dag.operands.length; i++) {
                const paramName = classRecord.templateArgs[i].name;
                const argValue = dag.operands[i].value;
                bindings.set(paramName, argValue);
            }
        }
        for (const [fieldName, field] of classRecord.fields) {
            let resolvedValue = field.value;
            if (resolvedValue && resolvedValue.type === 'def' && bindings.has(resolvedValue.value)) {
                resolvedValue = bindings.get(resolvedValue.value);
            } else if (resolvedValue && resolvedValue.type === 'bang') {
                resolvedValue = this._evaluateBangOp(resolvedValue, bindings);
            }
            fields.set(fieldName, resolvedValue);
        }
        return new tablegen.Value('record_instance', { className, fields });
    }

    _evaluateBangOp(value, bindings) {
        if (!value || value.type !== 'bang') {
            return value;
        }
        const { op, args } = value.value;
        if (op === 'tolower' && args && args.length === 1) {
            let [arg] = args;
            if (arg.type === 'def' && bindings.has(arg.value)) {
                arg = bindings.get(arg.value);
            }
            if (arg.type === 'string') {
                const str = String(arg.value).replace(/^"|"$/g, '');
                return new tablegen.Value('string', str.toLowerCase());
            }
        }
        return value;
    }

    _evaluateSimpleValue(value) {
        if (!value) {
            return 0;
        }
        if (value.type === 'int') {
            return typeof value.value === 'number' ? value.value : parseInt(value.value, 10);
        }
        if (typeof value === 'number') {
            return value;
        }
        return 0;
    }

    _parseForeachBody(loop) {
        while (!this._match('}') && !this._match('eof')) {
            this._parseForeachBodyStatement(loop);
            // If we found defvar, skip the rest of the body since we can't properly expand this loop
            if (loop.hasDefvar) {
                this._skipUntilClosingBrace();
                return;
            }
        }
    }

    // Skip tokens until we reach the matching closing brace (but don't consume it)
    _skipUntilClosingBrace() {
        let depth = 1;
        while (depth > 0 && !this._match('eof')) {
            if (this._match('{')) {
                depth++;
                this._read();
            } else if (this._match('}')) {
                depth--;
                if (depth === 0) {
                    // Don't consume the final } - let the caller handle it
                    return;
                }
                this._read();
            } else {
                this._read();
            }
        }
    }

    _parseForeachBodyStatement(loop) {
        const token = this._tokenizer.current();
        if (token.type === 'keyword') {
            switch (token.value) {
                case 'def':
                    loop.entries.push({ type: 'def', data: this._parseDefTemplate() });
                    break;
                case 'defm':
                    this._parseDefm();
                    break;
                case 'let':
                    this._parseLet();
                    break;
                case 'defvar':
                    loop.hasDefvar = true;
                    this._parseDefvar();
                    break;
                case 'foreach':
                    loop.entries.push({ type: 'foreach', data: this._parseForeachTemplate() });
                    break;
                case 'if':
                    loop.entries.push({ type: 'foreach', data: this._parseIfAsLoop() });
                    break;
                default:
                    this._read();
                    break;
            }
        } else {
            this._read();
        }
    }

    _parseIfAsLoop() {
        const location = this._tokenizer.location();
        this._read();
        const condition = this._parseValue();
        this._expect('keyword', 'then');
        const loop = {
            location,
            iterVarName: null,
            listValue: [],
            entries: [],
            condition,
            hasDefvar: false,
            isConditional: true
        };
        if (this._match('{')) {
            this._read();
            this._parseForeachBody(loop);
            this._expect('}');
        } else {
            this._parseForeachBodyStatement(loop);
        }
        if (this._match('keyword', 'else')) {
            this._read();
            if (this._match('{')) {
                this._read();
                let depth = 1;
                while (depth > 0 && !this._match('eof')) {
                    if (this._eat('{')) {
                        depth++;
                    } else if (this._eat('}')) {
                        depth--;
                    } else {
                        this._read();
                    }
                }
            }
        }
        return loop;
    }

    _parseDefTemplate() {
        this._read();
        const nameTemplate = [];
        while (!this._match(':') && !this._match('{') && !this._match(';') && !this._match('eof')) {
            if (this._match('id')) {
                const value = this._read();
                if (this._eat('.')) {
                    const field = this._expect('id');
                    nameTemplate.push({ type: 'field_access', base: value, field });
                } else {
                    nameTemplate.push({ type: 'id', value });
                }
            } else if (this._match('string')) {
                nameTemplate.push({ type: 'string', value: this._read() });
            } else if (this._match('number')) {
                nameTemplate.push({ type: 'number', value: this._read() });
            } else if (this._eat('#')) {
                nameTemplate.push({ type: 'concat' });
            } else {
                break;
            }
        }
        let parents = [];
        if (this._match(':')) {
            parents = this._parseParentClassList();
        }
        let bodyFields = new Map();
        if (this._match('{')) {
            this._read(); // consume '{'
            bodyFields = this._parseRecordBodyFields();
            this._expect('}');
        }
        this._eat(';');
        return { nameTemplate, parents, bodyFields };
    }

    _parseForeachTemplate() {
        const location = this._tokenizer.location();
        this._read();
        const iterVarName = this._expect('id');
        this._expect('=');
        const listValue = this._parseForeachListValue();
        this._expect('keyword', 'in');
        const loop = { location, iterVarName, listValue, entries: [], hasDefvar: false };
        if (this._match('{')) {
            this._read();
            this._parseForeachBody(loop);
            this._expect('}');
        } else {
            this._parseForeachBodyStatement(loop);
        }
        return loop;
    }

    _parseRecordBodyFields() {
        const fields = new Map();
        while (!this._match('}') && !this._match('eof')) {
            if (this._match('keyword', 'let')) {
                this._read();
                const name = this._expect('id');
                this._expect('=');
                const value = this._parseValue();
                this._eat(';');
                fields.set(name, { name, type: null, value });
            } else if (this._match('keyword', 'field')) {
                this._read();
                const type = this._parseType();
                const name = this._expect('id');
                let value = null;
                if (this._eat('=')) {
                    value = this._parseValue();
                }
                this._eat(';');
                fields.set(name, { name, type, value });
            } else if (this._match('keyword', 'assert') || this._match('keyword', 'dump')) {
                // Skip assert and dump statements
                this._skipUntil([';']);
                this._eat(';');
            } else if (this._match('id') || this._match('keyword')) {
                // Type followed by field name
                const type = this._parseType();
                const name = this._expect('id');
                let value = null;
                if (this._eat('=')) {
                    value = this._parseValue();
                }
                this._eat(';');
                fields.set(name, { name, type, value });
            } else {
                this._read();
            }
        }
        return fields;
    }

    _resolveForeachLoop(loop, substitutions) {
        if (loop.hasDefvar) {
            return;
        }
        if (loop.entries.length === 0) {
            return;
        }

        if (loop.isConditional) {
            const conditionResult = this._evaluateCondition(loop.condition, substitutions);
            if (conditionResult === false || conditionResult === null) {
                return;
            }
            for (const entry of loop.entries) {
                if (entry.type === 'def') {
                    this._instantiateDef(entry.data, substitutions);
                } else if (entry.type === 'foreach') {
                    this._resolveForeachLoop(entry.data, substitutions);
                }
            }
            return;
        }
        if (loop.listValue.length === 0) {
            return;
        }
        for (const listItem of loop.listValue) {
            const currentSubs = new Map(substitutions);
            if (loop.iterVarName) {
                currentSubs.set(loop.iterVarName, listItem);
            }
            for (const entry of loop.entries) {
                if (entry.type === 'def') {
                    this._instantiateDef(entry.data, currentSubs);
                } else if (entry.type === 'foreach') {
                    this._resolveForeachLoop(entry.data, currentSubs);
                }
            }
        }
    }

    _evaluateCondition(condition, substitutions) {
        if (!condition) {
            return null;
        }
        if (condition.type === 'bang' && condition.value) {
            const { op, args } = condition.value;
            if (op === 'ne' && args.length === 2) {
                const a = this._evaluateSimpleExpr(args[0], substitutions);
                const b = this._evaluateSimpleExpr(args[1], substitutions);
                if (a !== null && b !== null) {
                    return a !== b;
                }
            }
            if (op === 'eq' && args.length === 2) {
                const a = this._evaluateSimpleExpr(args[0], substitutions);
                const b = this._evaluateSimpleExpr(args[1], substitutions);
                if (a !== null && b !== null) {
                    return a === b;
                }
            }
        }
        // Can't evaluate complex conditions
        return null;
    }

    // Evaluate a simple expression for condition evaluation
    _evaluateSimpleExpr(expr, substitutions) {
        if (!expr) {
            return null;
        }
        if (expr.type === 'string') {
            return String(expr.value).replace(/^"|"$/g, '');
        }
        if (expr.type === 'int') {
            return typeof expr.value === 'number' ? expr.value : parseInt(expr.value, 10);
        }
        if ((expr.type === 'def' || expr.type === 'id') && substitutions.has(expr.value)) {
            return this._evaluateSimpleExpr(substitutions.get(expr.value), substitutions);
        }
        return null;
    }

    _instantiateDef(template, substitutions) {
        let name = '';
        for (const part of template.nameTemplate) {
            if (part.type === 'concat') {
                continue;
            } else if (part.type === 'field_access') {
                if (substitutions.has(part.base)) {
                    const subValue = substitutions.get(part.base);
                    name += this._getFieldValue(subValue, part.field);
                } else {
                    name += `${part.base}.${part.field}`;
                }
            } else if (part.type === 'id') {
                if (substitutions.has(part.value)) {
                    const subValue = substitutions.get(part.value);
                    name += this._valueToString(subValue);
                } else {
                    name += part.value;
                }
            } else if (part.type === 'string') {
                name += part.value;
            } else if (part.type === 'number') {
                name += String(part.value);
            }
        }
        const def = new tablegen.Record(name, this);
        def.location = this._tokenizer.location();
        def.parents = template.parents.map((parent) => ({
            name: parent.name,
            args: parent.args ? parent.args.map((arg) => this._substituteValue(arg, substitutions)) : []
        }));
        for (const [fieldName, field] of template.bodyFields) {
            const resolvedValue = field.value ? this._substituteValue(field.value, substitutions) : null;
            def.fields.set(fieldName, new tablegen.RecordVal(fieldName, field.type, resolvedValue));
        }
        this.addSubClass(def);
        def.resolveReferences();
        if (name) {
            this._defs.set(name, def);
            this.defs.push(def);
        }
    }

    _valueToString(value) {
        if (!value) {
            return '';
        }
        if (typeof value === 'string') {
            return value;
        }
        if (typeof value === 'number') {
            return String(value);
        }
        if (value.type === 'string') {
            return String(value.value).replace(/^"|"$/g, '');
        }
        if (value.type === 'int') {
            return String(value.value);
        }
        if (value.type === 'id' || value.type === 'def') {
            return String(value.value);
        }
        return '';
    }

    _getFieldValue(value, fieldName) {
        if (!value) {
            return '';
        }
        if (value.type === 'record_instance' && value.value && value.value.fields) {
            const fieldValue = value.value.fields.get(fieldName);
            return this._valueToString(fieldValue);
        }
        return '';
    }

    _substituteValue(value, substitutions) {
        if (!value) {
            return value;
        }
        if (value.type === 'def' || value.type === 'id') {
            const varName = value.value;
            if (substitutions.has(varName)) {
                return substitutions.get(varName);
            }
            if (typeof varName === 'string' && varName.includes('.')) {
                const [base, field] = varName.split('.', 2);
                if (substitutions.has(base)) {
                    const baseValue = substitutions.get(base);
                    const fieldValue = this._getFieldValue(baseValue, field);
                    if (fieldValue) {
                        return new tablegen.Value('string', fieldValue);
                    }
                }
            }
        }
        if (value.type === 'list' && Array.isArray(value.value)) {
            return {
                type: 'list',
                value: value.value.map((v) => this._substituteValue(v, substitutions))
            };
        }
        if (value.type === 'dag' && value.value) {
            return {
                type: 'dag',
                value: {
                    operator: value.value.operator,
                    operands: value.value.operands.map((op) => ({
                        value: this._substituteValue(op.value, substitutions),
                        name: op.name
                    }))
                }
            };
        }
        if (value.type === 'concat' && Array.isArray(value.value)) {
            return {
                type: 'concat',
                value: value.value.map((v) => this._substituteValue(v, substitutions))
            };
        }
        return value;
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
            if (this._match('keyword', 'let')) {
                this._read();
                const name = this._expect('id');
                this._expect('=');
                const value = this._parseValue();
                const field = new tablegen.RecordVal(name, null, value);
                record.fields.set(name, field);
                this._eat(';');
            } else if (this._match('keyword', 'defvar')) {
                this._read();
                const name = this._expect('id');
                this._expect('=');
                const value = this._parseValue();
                const field = new tablegen.RecordVal(name, null, value);
                record.fields.set(name, field);
                this._eat(';');
            } else if (this._match('keyword', 'assert')) {
                this._read();
                this._parseValue(); // condition
                this._eat(',');
                this._parseValue(); // message
                this._eat(';');
            } else if (this._match('keyword', 'bit') || this._match('keyword', 'bits') || this._match('keyword', 'int') ||
                       this._match('keyword', 'string') || this._match('keyword', 'list') || this._match('keyword', 'dag') ||
                       this._match('keyword', 'code') || this._match('id')) {
                const type = this._parseType();
                // Skip if next token is not an id (handles edge cases in complex nested structures)
                if (!this._match('id')) {
                    if (this._match(',') || this._match('>') || this._match(')') || this._match(']')) {
                        continue;
                    }
                }
                const name = this._expect('id');
                let value = null;
                if (this._eat('=')) {
                    value = this._parseValue();
                }
                const field = new tablegen.RecordVal(name, type, value);
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

        if (this._match('keyword', 'bit') || this._match('keyword', 'bits') || this._match('keyword', 'int') ||
            this._match('keyword', 'string') || this._match('keyword', 'list') || this._match('keyword', 'dag') ||
            this._match('keyword', 'code')) {
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
            type.args = this._parseTemplateArgList();
        }
        return type;
    }

    _parseTemplateArgList() {
        // Parse template arguments directly from token stream
        // Supports both positional (arg1, arg2) and named (name=value) arguments
        const args = [];
        while (!this._match('>') && !this._match('eof')) {
            // Check if this is a named argument: id = value
            if (this._match('id')) {
                const name = this._read();
                if (this._match('=')) {
                    // Named argument
                    this._read(); // Consume '='
                    const value = this._parseValue();
                    args.push({ name, value });
                } else {
                    // Positional argument that starts with an id
                    // Reconstruct the value - id might be part of concat, field access, etc.
                    let value = new tablegen.Value('def', name);
                    // Handle < > for template instantiation - manually parse with depth tracking
                    if (this._eat('<')) {
                        const nestedArgs = [];
                        let depth = 1;
                        let currentArg = [];
                        while (depth > 0 && !this._match('eof')) {
                            if (this._match('<')) {
                                currentArg.push(this._read());
                                depth++;
                            } else if (this._match('>')) {
                                if (depth === 1) {
                                    // End of this template arg list
                                    if (currentArg.length > 0) {
                                        // Parse accumulated tokens properly
                                        const argStr = currentArg.join(' ');
                                        // Try to parse as number first
                                        if (/^-?\d+$/.test(argStr)) {
                                            nestedArgs.push(new tablegen.Value('int', argStr));
                                        } else {
                                            nestedArgs.push(new tablegen.Value('def', argStr));
                                        }
                                    }
                                    this._read(); // consume the >
                                    depth--;
                                } else {
                                    currentArg.push(this._read());
                                    depth--;
                                }
                            } else if (this._match(',') && depth === 1) {
                                // Argument separator at current depth
                                if (currentArg.length > 0) {
                                    // Parse accumulated tokens properly
                                    const argStr = currentArg.join(' ');
                                    // Try to parse as number first
                                    if (/^-?\d+$/.test(argStr)) {
                                        nestedArgs.push(new tablegen.Value('int', argStr));
                                    } else {
                                        nestedArgs.push(new tablegen.Value('def', argStr));
                                    }
                                    currentArg = [];
                                }
                                this._read(); // consume comma
                            } else {
                                // Regular token - add to current arg
                                currentArg.push(this._read());
                            }
                        }
                        // Convert to DAG operands
                        const operands = nestedArgs.map((arg) => ({ value: arg, name: null }));
                        value = new tablegen.Value('dag', new tablegen.DAG(name, operands));
                    }
                    // Handle field access
                    if (this._eat('.')) {
                        const field = this._expect('id');
                        if (value.type === 'def') {
                            value = new tablegen.Value('def', `${name}.${field}`);
                        }
                    }
                    // Handle :: suffix
                    if (this._eat('::')) {
                        const suffix = this._expect('id');
                        if (value.type === 'def') {
                            value = new tablegen.Value('def', `${name}::${suffix}`);
                        }
                    }
                    // Handle # concatenation
                    if (this._match('#')) {
                        const values = [value];
                        while (this._match('#')) {
                            this._read();
                            values.push(this._parsePrimaryValue());
                        }
                        value = new tablegen.Value('concat', values);
                    }
                    args.push(value);
                }
            } else {
                const value = this._parseValue();
                args.push(value);
            }
            if (!this._eat(',')) {
                break;
            }
        }
        this._expect('>');
        return args;
    }

    _parseValue() {
        const values = [];
        values.push(this._parsePrimaryValue());
        while (this._match('#') || (values[values.length - 1] && values[values.length - 1].type === 'string' && this._match('string'))) {
            if (this._match('#')) {
                this._read();
            }
            values.push(this._parsePrimaryValue());
        }
        if (values.length === 1) {
            return values[0];
        }
        return new tablegen.Value('concat', values);
    }

    _parseListItem() {
        // Handle $variable as a standalone value in list/dag context
        if (this._eat('$')) {
            const name = this._expect('id');
            return new tablegen.Value('var', name);
        }
        // Special handling for dag-like constructs (id followed by <...>)
        // These need to be parsed as DAGs for trait information
        if (this._match('id')) {
            const name = this._read();
            if (this._eat('<')) {
                const templateArgs = this._parseTemplateArgList();
                const operands = templateArgs.map((arg) => {
                    if (arg && typeof arg === 'object' && arg.name && arg.value) {
                        return { value: arg.value, name: arg.name };
                    }
                    return { value: arg, name: null };
                });
                const result = new tablegen.Value('dag', new tablegen.DAG(name, operands));
                if (this._eat('.')) {
                    result.field = this._expect('id');
                }
                return result;
            }
            // Not a template instantiation, but might be an identifier with suffixes
            // Put the token back conceptually by creating a def value and checking for suffixes
            let result = new tablegen.Value('def', name);
            // Check for subscripts, field access, etc.
            while (this._eat('[')) {
                const index = this._parseValue();
                this._expect(']');
                result = new tablegen.Value('bang', { op: 'subscript', args: [result, index], field: null });
            }
            if (this._eat('.')) {
                const field = this._expect('id');
                if (result.type === 'def') {
                    result = new tablegen.Value('def', `${result.value}.${field}`);
                } else {
                    result.value.field = field;
                }
            }
            return result;
        }
        return this._parseValue();
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
                items.push(this._parseListItem());
                this._eat(',');
            }
            this._expect(']');
            if (this._match('<')) {
                this._skip('<', '>');
            }
            return new tablegen.Value('list', items);
        }
        if (this._eat('(')) {
            let operator = null;
            let operatorName = null;
            if (this._match('id')) {
                operator = this._read();
                // Handle template arguments on the operator: (NativeCodeCallVoid<...> ...)
                if (this._match('<')) {
                    this._skip('<', '>');
                }
                // Handle operator binding: (OpQ:$op ...)
                if (this._eat(':') && this._eat('$')) {
                    if (this._match('id') || this._match('keyword')) {
                        operatorName = this._read();
                    }
                }
            }
            const operands = [];
            while (!this._match(')') && !this._match('eof')) {
                if (this._eat(',')) {
                    continue;
                }
                // Use _parseListItem() to handle template instantiations like Arg<TTG_MemDescType>
                const value = this._parseListItem();
                let name = null;
                if (this._eat(':') && this._eat('$')) {
                    if (this._match('id') || this._match('keyword')) {
                        name = this._read();
                    }
                }
                const operand = { value, name };
                operands.push(operand);
                this._eat(',');
            }
            this._expect(')');
            const dag = new tablegen.DAG(operator, operands);
            if (operatorName) {
                dag.operatorName = operatorName;
            }
            return new tablegen.Value('dag', dag);
        }
        if (this._eat('{')) {
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
            let op = null;
            if (this._match('id') || this._match('keyword')) {
                op = this._read();
            } else {
                throw new tablegen.Error(`Expected operator after '!' but got '${this._tokenizer.current().type}' at ${this._tokenizer.location()}`);
            }
            if (this._match('<')) {
                this._skip('<', '>');
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
        if (this._match('id') || this._match('keyword')) {
            const value = this._read();
            let result = new tablegen.Value('def', value);
            // Handle various suffixes: templates, subscripts, field access, scope resolution
            while (true) {
                if (this._match('<')) {
                    // Template arguments are skipped here - they're parsed properly in
                    // DAG context by _parseListItem() which creates DAG values with template args
                    this._skip('<', '>');
                } else if (this._eat('[')) {
                    // Array subscripting: x[0]
                    const index = this._parseValue();
                    this._expect(']');
                    result = new tablegen.Value('bang', { op: 'subscript', args: [result, index], field: null });
                } else if (this._eat('.')) {
                    // Field access: x.field or x[0].field
                    const field = this._expect('id');
                    if (result.type === 'def') {
                        result = new tablegen.Value('def', `${result.value}.${field}`);
                    } else {
                        // For subscript results, add field access
                        result.value.field = field;
                    }
                } else if (this._eat('::')) {
                    // Scope resolution
                    const suffix = this._expect('id');
                    if (result.type === 'def') {
                        result = new tablegen.Value('def', `${result.value}::${suffix}`);
                    }
                    break;
                } else {
                    break;
                }
            }
            return result;
        }
        if (this._eat('?')) {
            return new tablegen.Value('uninitialized', null);
        }
        if (this._eat('$')) {
            const name = this._expect('id');
            return new tablegen.Value('var', name);
        }
        throw new tablegen.Error(`Unexpected value at ${this._tokenizer.location()}`);
    }

    _read() {
        return this._tokenizer.read().value;
    }

    _match(type, value) {
        const token = this._tokenizer.current();
        return token.type === type && (!value || token.value === value);
    }

    _eat(type, value) {
        if (this._match(type, value)) {
            this._read();
            return true;
        }
        return false;
    }

    _expect(type, value) {
        if (this._match(type, value)) {
            return this._read();
        }
        const token = this._tokenizer.current();
        throw new tablegen.Error(`Expected '${type}' but got '${token.type}' at ${this._tokenizer.location()}`);
    }

    _skip(open, close) {
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
