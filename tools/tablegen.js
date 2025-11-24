
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
        const type = this._keywords.has(value) ? value : 'id';
        value = type === 'true' || type === 'false' ? value === 'true' : value;
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

    // Returns the RecordVal for the given field name, or null if not found
    // Matches C++ Record::getValue() which returns nullptr for missing fields
    getValue(name) {
        return this.fields.get(name) || null;
    }

    // Returns the Init value for the given field name, or throws if not found
    // Matches C++ Record::getValueInit() which throws PrintFatalError for missing fields
    getValueInit(fieldName) {
        const recordVal = this.getValue(fieldName);
        if (!recordVal || !recordVal.value) {
            throw new Error(`Record '${this.name}' does not have a field named '${fieldName}'`);
        }
        return recordVal.value;
    }

    // Bind template parameters from parent class instantiation
    bindTemplateParameters() {
        for (const parent of this.parents) {
            const parentClass = this.parser.classes.get(parent.name);
            if (parentClass && parentClass.templateArgs && parentClass.templateArgs.length > 0) {
                // Match template arguments to template parameters
                const templateArgs = parent.args || [];

                // Process both positional and named arguments
                for (let i = 0; i < parentClass.templateArgs.length; i++) {
                    const param = parentClass.templateArgs[i];
                    let boundValue = null;

                    // Check for named argument
                    const namedArg = templateArgs.find((arg) => arg.name === param.name);
                    if (namedArg) {
                        boundValue = namedArg.value;
                    } else if (i < templateArgs.length) {
                        // Use positional argument
                        const arg = templateArgs[i];
                        boundValue = arg.name ? arg.value : arg;
                    } else if (param.defaultValue) {
                        // Use default value
                        boundValue = param.defaultValue;
                    }

                    if (boundValue) {
                        this.templateBindings.set(param.name, boundValue);
                    }
                }
            }
        }
    }

    // Flatten parent class fields into this record (eager resolution)
    // This matches the C++ TableGen behavior where parent fields are copied
    // during record construction, eliminating the need for runtime parent walking
    flattenParentFields() {
        const visited = new Set();
        const parentFields = new Map();

        // Recursively collect fields from all parent classes
        const collectParentFields = (record) => {
            if (!record || visited.has(record.name)) {
                return;
            }
            visited.add(record.name);

            // Process parents first (depth-first) so closer parents override
            for (const parent of record.parents) {
                const parentClass = this.parser.classes.get(parent.name);
                if (parentClass) {
                    collectParentFields(parentClass);
                }
            }

            // Add this record's fields to the parent fields map
            // Later entries override earlier ones
            for (const [name, field] of record.fields) {
                parentFields.set(name, field);
            }
        };

        // Collect fields from parents (not including this record itself)
        for (const parent of this.parents) {
            const parentClass = this.parser.classes.get(parent.name);
            if (parentClass) {
                collectParentFields(parentClass);
            }
        }

        // Now merge parent fields into this record
        // Own fields take precedence over parent fields
        for (const [name, parentField] of parentFields) {
            if (!this.fields.has(name)) {
                this.fields.set(name, parentField);
            }
        }
    }

    // Returns evaluated string value, or null if field doesn't exist or wrong type
    // Unlike C++ getValueAsString which throws, this returns null for convenience
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

    // Returns evaluated bit value, or null if field doesn't exist or wrong type
    // Unlike C++ getValueAsBit which throws, this returns null for convenience
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

    // Returns evaluated dag value, or null if field doesn't exist or wrong type
    // Unlike C++ getValueAsDag which throws, this returns null for convenience
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

    evaluateValue(value) {
        if (!value) {
            return null;
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
                // First check if this is a template parameter binding
                if (this.templateBindings.has(fieldName)) {
                    return this.evaluateValue(this.templateBindings.get(fieldName));
                }
                // Otherwise, resolve as a field
                const field = this.getValue(fieldName);
                if (field && field.value) {
                    return this.evaluateValue(field.value);
                }
                return null;
            }
            case 'def': {
                const defName = typeof value.value === 'string' ? value.value : value.value.value;

                // Check if this includes field access (e.g., "clause.arguments")
                if (defName.includes('.')) {
                    const parts = defName.split('.');
                    const [baseName, ...fieldPath] = parts;

                    // First try to resolve the base as a template binding
                    let baseValue = null;
                    if (this.templateBindings.has(baseName)) {
                        baseValue = this.evaluateValue(this.templateBindings.get(baseName));
                    } else {
                        // Try to resolve as a field
                        const field = this.getValue(baseName);
                        if (field && field.value) {
                            baseValue = this.evaluateValue(field.value);
                        }
                    }

                    // If baseValue is a def name, look it up
                    if (typeof baseValue === 'string') {
                        const def = this.parser.getDef(baseValue) || this.parser.getClass(baseValue);
                        if (def) {
                            // Navigate through the field path
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

                // Check if this is a template parameter binding first
                if (this.templateBindings.has(defName)) {
                    return this.evaluateValue(this.templateBindings.get(defName));
                }
                // Otherwise, resolve as a field or def
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

                        // For each item in the list, evaluate the expression
                        // with acc=accumulator and item=current
                        for (const item of list) {
                            // Create temporary fields for acc and item variables
                            const accName = args[2].value; // variable name
                            const itemName = args[3].value; // variable name

                            // Store current values
                            const prevAcc = this.fields.get(accName);
                            const prevItem = this.fields.get(itemName);

                            // Set up context for expression evaluation
                            // The accumulator needs to be wrapped appropriately
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

                            // Evaluate the expression
                            accumulator = this.evaluateValue(args[4]);

                            // Restore previous values
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
                        // args[0] is the item variable name (not evaluated)
                        const itemName = args[0].value;
                        const list = this.evaluateValue(args[1]);
                        // args[2] is the expression to evaluate for each item

                        if (!Array.isArray(list)) {
                            return [];
                        }

                        const results = [];
                        for (const item of list) {
                            // Store current value
                            const prevItem = this.fields.get(itemName);

                            // Set up context - wrap item in a Value so it can be used in expressions
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

                            // Evaluate expression
                            const result = this.evaluateValue(args[2]);
                            results.push(result);

                            // Restore previous value
                            if (prevItem) {
                                this.fields.set(itemName, prevItem);
                            } else {
                                this.fields.delete(itemName);
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

                        if (!Array.isArray(list)) {
                            return [];
                        }

                        const results = [];
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

                        return results;
                    }
                    case 'con': {
                        // !con(dag1, dag2, ...)
                        // Concatenate dags - merge operands from multiple dags
                        if (args.length === 0) {
                            // Return empty dag - just the DAG, not wrapped
                            return new tablegen.DAG('ins', []);
                        }

                        // Get the operator from the first arg (if it's a dag)
                        let operator = 'ins';
                        const allOperands = [];

                        // Process all arguments
                        for (const arg of args) {
                            // For !con, we need to access field values directly
                            // without full evaluation to get the DAG structure
                            let dagToProcess = null;

                            // First try to evaluate
                            const evaluated = this.evaluateValue(arg);

                            // Handle different types of evaluated values
                            if (evaluated && typeof evaluated === 'object') {
                                if (evaluated.operator && evaluated.operands) {
                                    // It's a DAG object directly
                                    dagToProcess = evaluated;
                                } else if (evaluated.type === 'dag' && evaluated.value) {
                                    // It's a Value wrapping a DAG
                                    dagToProcess = evaluated.value;
                                }
                            }

                            // If evaluation didn't work, try direct field access
                            if (!dagToProcess && arg.type === 'dag') {
                                dagToProcess = arg.value;
                            }

                            // Add operands from this dag
                            if (dagToProcess && dagToProcess.operands) {
                                if (operator === 'ins' && dagToProcess.operator) {
                                    operator = dagToProcess.operator;
                                }
                                allOperands.push(...dagToProcess.operands);
                            }
                        }

                        // Return just the DAG object, not wrapped in a Value
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
                    default:
                        return null;
                }
            }
            case 'dag':
                // Return the DAG object directly
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
        this._defs = new Map(); // Internal map for lookup during parsing (can have collisions)
        this.classes = new Map();
        this.defs = []; // Public list of all defs including duplicates
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

    getDef(name) {
        return this._defs.get(name);
    }

    getClass(name) {
        return this.classes.get(name);
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
        // Bind template parameters after parsing is complete
        def.bindTemplateParameters();
        // Flatten parent fields into this record (eager resolution)
        def.flattenParentFields();
        if (name) {
            this._defs.set(name, def);
            this.defs.push(def); // Also add to list to preserve duplicates
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
                const field = new tablegen.RecordVal(name, null, value);
                record.fields.set(name, field);
                this._eat(';');
            } else if (this._match('defvar')) {
                this._read();
                const name = this._expect('id');
                this._expect('=');
                const value = this._parseValue();
                const field = new tablegen.RecordVal(name, null, value);
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
                    // Handle < > for template instantiation
                    if (this._match('<')) {
                        this._skip('<', '>');
                    }
                    // Handle field access
                    if (this._eat('.')) {
                        const field = this._expect('id');
                        value = new tablegen.Value('def', `${name}.${field}`);
                    }
                    // Handle :: suffix
                    if (this._eat('::')) {
                        const suffix = this._expect('id');
                        value = new tablegen.Value('def', `${name}::${suffix}`);
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
                // Positional argument that doesn't start with an id
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
                return new tablegen.Value('dag', new tablegen.DAG(name, operands));
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
            if (this._match('id')) {
                operator = this._read();
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
                    if (this._match('id') || this._isKeyword(this._tokenizer.current().type)) {
                        name = this._read();
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
        if (this._match('id') || this._isKeyword(this._tokenizer.current().type)) {
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
