
// Experimental
// contributor @tucan9389

const mlir = {};

mlir.ModelFactory = class {

    async match(context) {
        const stream = context.stream;
        if (stream && stream.length > 4) {
            const buffer = stream.peek(4);
            const signature = String.fromCharCode.apply(null, buffer);
            if (signature === 'ML\xEFR') {
                return context.set('mlir.binary');
            }
        }
        try {
            const reader = await context.read('text', 0x10000);
            for (let line = reader.read('\n'); line !== undefined; line = reader.read('\n')) {
                if (/module\s+(@\w+|\w+|attributes)/.test(line) ||
                    /tensor<[\w\d]+>/.test(line) ||
                    /func[.\s]*@\w+/.test(line) ||
                    /%\w+\s*=\s*"[\w.]+/.test(line) ||
                    /%\w+\s*=\s*\w+\./.test(line) ||
                    /#\w+\s*=\s*loc\s*\(/.test(line) ||
                    /\w+\.\w+\s+@\w+/.test(line) ||
                    /:\s*![\w.]+/.test(line) ||
                    /(%\w+|\w{2,}|[)])\s*:\s*(\[|tensor<)/.test(line) ||
                    /->\s*(![\w.]+|\(|tensor<)/.test(line)) {
                    return context.set('mlir.text');
                }
            }
        } catch {
            // continue regardless of error
        }
        return null;
    }

    async open(context) {
        const metadata = await mlir.Metadata.open(context);
        switch (context.type) {
            case 'mlir.text': {
                const decoder = await context.read('text.decoder');
                const parser = new mlir.Parser(decoder, metadata);
                const obj = await parser.read();
                return new mlir.Model(metadata, obj);
            }
            case 'mlir.binary': {
                const reader = await context.read('binary');
                const parser = new mlir.BytecodeReader(reader);
                parser.read();
                throw new mlir.Error('File contains unsupported MLIR bytecode data.');
            }
            default: {
                throw new mlir.Error(`Unsupported MLIR format '${context.type}'.`);
            }
        }
    }
};

mlir.Model = class {

    constructor(metadata, obj) {
        this.format = 'MLIR';
        this.modules = [];
        this.metadata = [];
        for (const op of obj.operations) {
            if (op.name.endsWith('.func')) {
                const graph = new mlir.Graph(metadata, op);
                this.modules.push(graph);
            }
            if (op.name.endsWith('.module')) {
                for (const region of op.regions) {
                    for (const block of region.blocks) {
                        for (const op of block.operations) {
                            if (op.name.endsWith('.func')) {
                                const graph = new mlir.Graph(metadata, op);
                                this.modules.push(graph);
                            }
                        }
                    }
                }
            }
        }
        if (obj.definitions) {
            for (const attribute of obj.definitions) {
                const value = typeof attribute.value === 'string' ? attribute.value : JSON.stringify(attribute.value);
                const metadata = new mlir.Argument(attribute.name, value, 'attribute');
                this.metadata.push(metadata);
            }
        }
    }
};

mlir.Graph = class {

    constructor(metadata, func) {
        const attr = Object.fromEntries(func.attributes.map((attr) => [attr.name, attr.value]));
        this.name = attr.sym_name || '';
        this.type = func.name === 'func' || func.name.endsWith('.func') ? 'function' : '';
        this.description = func.name;
        this.inputs = [];
        this.outputs = [];
        this.nodes = [];
        const tensors = new Map();
        const tensor = (arg) => {
            if (!tensors.has(arg.name)) {
                tensors.set(arg.name, new mlir.Value(arg.name, arg.type, null, arg.value));
            }
            return tensors.get(arg.name);
        };
        const function_type = attr.function_type;
        for (let i = 0; i < function_type.inputs.length; i++) {
            const input = function_type.inputs[i];
            const name = input.name || i.toString();
            const type = mlir.Utility.valueType(input.type);
            const valueName = input.value || input.name || `%arg${i}`;
            const value = new mlir.Value(valueName, type, '', null);
            const argument = new mlir.Argument(name, [value]);
            this.inputs.push(argument);
        }
        for (let i = 0; i < function_type.results.length; i++) {
            const output = function_type.results[i];
            const name = output.name || i.toString();
            const type = mlir.Utility.valueType(output.type);
            const valueName = output.value || output.name || `%result${i}`;
            const value = new mlir.Value(valueName, type, '', null);
            const argument = new mlir.Argument(name, [value]);
            this.outputs.push(argument);
        }
        const values = new Map();
        values.map = (name) => {
            if (!values.has(name)) {
                values.set(name, { name, to: [], from: [] });
            }
            return values.get(name);
        };
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
                    const opMetadata = metadata.type(op.name);
                    const operands = op.operands || [];
                    for (let i = 0; i < operands.length; i++) {
                        const input = op.operands[i];
                        const inputName = input.name || (opMetadata && opMetadata.inputs && opMetadata.inputs[i] ? opMetadata.inputs[i].name : null) || i.toString();
                        if (input.value instanceof Uint8Array) {
                            operation.operands.push({
                                name: inputName,
                                value: input.value,
                                type: input.type
                            });
                        } else if (Number.isInteger(input.value)) {
                            operation.operands.push({
                                name: inputName,
                                value: input.value,
                                type: 'int64'
                            });
                        } else if (typeof input.value === 'boolean') {
                            operation.operands.push({
                                name: inputName,
                                value: input.value,
                                type: 'boolean'
                            });
                        } else if (Array.isArray(input.value)) {
                            operation.operands.push({
                                name: inputName,
                                value: input.value
                            });
                        } else {
                            const value = values.map(input);
                            value.to.push(operation);
                            const args = [{ name: input.value, type: input.type }];
                            operation.operands.push({
                                name: inputName,
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
                        const outputName = output.name || (opMetadata && opMetadata.outputs && opMetadata.outputs[i] ? opMetadata.outputs[i].name : null) || i.toString();
                        operation.results.push({
                            name: outputName,
                            value: [value]
                        });
                    }
                    operations.push(operation);
                }
            }
        }
        for (const input of this.inputs) {
            for (const arg of input.value) {
                if (!tensors.has(arg.name)) {
                    tensors.set(arg.name, arg);
                }
            }
        }
        for (const output of this.outputs) {
            for (let i = 0; i < output.value.length; i++) {
                const arg = output.value[i];
                if (tensors.has(arg.name)) {
                    output.value[i] = tensors.get(arg.name);
                } else {
                    tensors.set(arg.name, arg);
                }
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
            const node = new mlir.Node(metadata, op);
            this.nodes.push(node);
        }
    }
};

mlir.Argument = class {

    constructor(name, value, type) {
        this.name = name;
        this.value = value;
        this.type = type || null;
        // Normalize common type aliases and accept extended MLIR types
        if (this.type) {
            switch (this.type) {
                case 'i64': case 'si64': this.type = 'int64'; break;
                case 'i32': case 'si32': this.type = 'int32'; break;
                case 'i16': case 'si16': this.type = 'int16'; break;
                case 'i8': case 'si8': this.type = 'int8'; break;
                case 'i1': this.type = 'boolean'; break;
                case 'f32': case 'float32': this.type = 'float32'; break;
                case 'f64': case 'float64': this.type = 'float64'; break;
                case 'f16': this.type = 'float16'; break;
                case null:
                case 'attribute':
                case 'boolean':
                case 'string':
                case 'int64':
                case 'int32':
                case 'int16':
                case 'int8':
                case 'float16':
                case 'tensor':
                case 'type':
                case 'dense':
                    break;
                default:
                    // Accept other MLIR types without normalization
                    if (/^[usi]i?[0-9]+$/.test(this.type) || /^f[0-9]+$/.test(this.type) ||
                        this.type === 'bf16' || this.type === 'index' || this.type === 'none' ||
                        this.type.startsWith('!') || this.type.startsWith('tensor<') ||
                        this.type.startsWith('memref<') || this.type.startsWith('vector<')) {
                        break;
                    }
                    throw new mlir.Error(`Unsupported argument type '${this.type}'.`);
            }
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

    constructor(metadata, op) {
        if (!op.type) {
            throw new mlir.Error('Undefined node type.');
        }
        this.type = { ...metadata.type(op.identifier || '') };
        this.type.name = op.type || '';
        this.type.identifier = op.identifier || '';
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

    constructor(kind, value, text) {
        this.kind = kind;
        this.value = value;
        this.text = text;
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
                        return this._identifier();
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
        if (v === '0' && this._current === 'x' && /[0-9a-fA-F]/.test(this._peek())) {
            v += this._read();
            while (this._current && /[0-9a-fA-F]/.test(this._current)) {
                v += this._read();
            }
            return new mlir.Token(type, parseInt(v, 16), v);
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
            }
            return new mlir.Token(type, parseFloat(v), v);
        }
        return new mlir.Token(type, parseInt(v, 10), v);
    }

    _stringLiteral() {
        let result = '';
        this._read();
        while (this._current && this._current !== '"') {
            if (this._eat('\\')) {
                const hexDigit = /[0-9a-fA-F]/;
                if (hexDigit.test(this._current) && this._next && hexDigit.test(this._next)) {
                    const hex = this._current + this._next;
                    result += String.fromCharCode(parseInt(hex, 16));
                    this._read();
                    this._read();
                    continue;
                }
                switch (this._current) {
                    case 'n':
                        result += '\n';
                        this._read();
                        break;
                    case 'r':
                        result += '\r';
                        this._read();
                        break;
                    case 't':
                        result += '\t';
                        this._read();
                        break;
                    case '"':
                    case '\\':
                        result += this._current;
                        this._read();
                        break;
                    default:
                        throw new mlir.Error(`Unknown escape sequence '\\${this._current}' in string literal`);
                }
            } else {
                result += this._current;
                this._read();
            }
        }
        if (this._eat('"')) {
            return new mlir.Token('string', result);
        }
        throw new mlir.Error('Unterminated string literal');
    }

    _identifier() {
        let result = '';
        while (this._current && (/[a-zA-Z_$\-.]/.test(this._current) || /[0-9]/.test(this._current))) {
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

    constructor(decoder, metadata) {
        this._tokenizer = new mlir.Tokenizer(decoder);
        this._token = this._tokenizer.read();
        this._state = {
            defaultDialectStack: ['builtin']
        };
        this._dialects = new Map();
        const operations = metadata.operations;
        this._dialects.set('builtin', new mlir.BuiltinDialect(operations));
        this._dialects.set('bufferization', new mlir.BufferizationDialect(operations));
        this._dialects.set('stablehlo', new mlir.StableHLODialect(operations));
        this._dialects.set('vhlo', new mlir.VhloDialect(operations));
        this._dialects.set('interpreter', new mlir.InterpreterDialect(operations));
        this._dialects.set('affine', new mlir.AffineDialect(operations));
        this._dialects.set('asuka', new mlir.AsukaDialect(operations));
        this._dialects.set('arith', new mlir.ArithDialect(operations));
        this._dialects.set('async', new mlir.AsyncDialect(operations));
        this._dialects.set('cf', new mlir.CFDialect(operations));
        this._dialects.set('emitc', new mlir.Dialect('emitc', operations));
        this._dialects.set('complex', new mlir.Dialect('complex', operations));
        this._dialects.set('index', new mlir.Dialect('index', operations));
        this._dialects.set('pdl', new mlir.Dialect('pdl', operations));
        this._dialects.set('ptr', new mlir.Dialect('ptr', operations));
        this._dialects.set('ub', new mlir.Dialect('ub', operations));
        this._dialects.set('amdgpu', new mlir.Dialect('amdgpu', operations));
        this._dialects.set('nvgpu', new mlir.Dialect('nvgpu', operations));
        this._dialects.set('nvvm', new mlir.NVVMDialect(operations));
        this._dialects.set('rocdl', new mlir.Dialect('rocdl', operations));
        this._dialects.set('nvws', new mlir.Dialect('nvws', operations));
        this._dialects.set('tti', new mlir.Dialect('tti', operations));
        this._dialects.set('omp', new mlir.OpenMPDialect(operations));
        this._dialects.set('proton', new mlir.ProtonDialect(operations));
        this._dialects.set('proton_gpu', new mlir.Dialect('proton_gpu', operations));
        this._dialects.set('arm_sme', new mlir.ArmSMEDialect(operations));
        this._dialects.set('arm_neon', new mlir.ArmNeonDialect(operations));
        this._dialects.set('arm_sve', new mlir.ArmSVEDialect(operations));
        this._dialects.set('shard', new mlir.Dialect('shard', operations));
        this._dialects.set('amx', new mlir.Dialect('amx', operations));
        this._dialects.set('smt', new mlir.Dialect('smt', operations));
        this._dialects.set('iree_codegen', new mlir.Dialect('iree_codegen', operations));
        this._dialects.set('iree_encoding', new mlir.Dialect('iree_encoding', operations));
        this._dialects.set('test', new mlir.TestDialect(operations));
        this._dialects.set('scf', new mlir.SCFDialect(operations));
        this._dialects.set('shape', new mlir.ShapeDialect(operations));
        this._dialects.set('sparse_tensor', new mlir.SparseTensorDialect(operations));
        this._dialects.set('func', new mlir.FuncDialect(operations));
        this._dialects.set('gpu', new mlir.GpuDialect(operations));
        this._dialects.set('llvm', new mlir.LLVMDialect(operations));
        this._dialects.set('xegpu', new mlir.Dialect('xegpu', operations));
        this._dialects.set('memref', new mlir.MemRefDialect(operations));
        this._dialects.set('vector', new mlir.VectorDialect(operations));
        this._dialects.set('x86vector', new mlir.Dialect('x86vector', operations));
        this._dialects.set('onnx', new mlir.ONNXDialect(operations));
        this._dialects.set('krnl', new mlir.Dialect('krnl', operations));
        this._dialects.set('torch', new mlir.TorchDialect(operations));
        this._dialects.set('torch_c', new mlir.Dialect('torch_c', operations));
        this._dialects.set('hal', new mlir.HALDialect(operations));
        this._dialects.set('util', new mlir.UtilDialect(operations));
        this._dialects.set('mhlo', new mlir.MhloDialect(operations));
        this._dialects.set('chlo', new mlir.Dialect('chlo', operations));
        this._dialects.set('flow', new mlir.FlowDialect(operations));
        this._dialects.set('stream', new mlir.StreamDialect(operations));
        this._dialects.set('iree_vector_ext', new mlir.Dialect('iree_vector_ext', operations));
        this._dialects.set('iree_tensor_ext', new mlir.IREETensorExtDialect(operations));
        this._dialects.set('linalg', new mlir.LinalgDialect(operations));
        this._dialects.set('iree_linalg_ext', new mlir.Dialect('iree_linalg_ext', operations));
        this._dialects.set('quant', new mlir.QuantDialect(operations));
        this._dialects.set('tensor', new mlir.Dialect('tensor', operations));
        this._dialects.set('tosa', new mlir.TosaDialect(operations));
        this._dialects.set('tf', new mlir.TFDialect(operations));
        this._dialects.set('tf_saved_model', new mlir.Dialect('tf_saved_model', operations));
        this._dialects.set('tf_type', new mlir.TFTypeDialect(operations));
        this._dialects.set('tf_device', new mlir.TFDeviceDialect(operations));
        this._dialects.set('tf_executor', new mlir.TFExecutorDialect(operations));
        this._dialects.set('tf_framework', new mlir.TFFrameworkDialect(operations));
        this._dialects.set('tfr', new mlir.TFRDialect(operations));
        this._dialects.set('tfrt', new mlir.TFRTDialect(operations));
        this._dialects.set('tfrt_fallback', new mlir.Dialect('tfrt_fallback', operations));
        this._dialects.set('tfl', new mlir.Dialect('tfl', operations));
        this._dialects.set('stdx', new mlir.StdxDialect(operations));
        this._dialects.set('vm', new mlir.VMDialect(operations));
        this._dialects.set('math', new mlir.MathDialect(operations));
        this._dialects.set('tm_tensor', new mlir.TMTensorDialect(operations));
        this._dialects.set('ml_program', new mlir.MLProgramDialect(operations));
        this._dialects.set('iree_gpu', new mlir.IREEGPUDialect(operations));
        this._dialects.set('tile', new mlir.TileDialect(operations));
        this._dialects.set('irdl', new mlir.IRDLDialect(operations));
        this._dialects.set('transform', new mlir.TransformDialect(operations));
        this._dialects.set('wasmssa', new mlir.Dialect('wasmssa', operations));
        this._dialects.set('spirv', new mlir.SPIRVDialect(operations));
        this._dialects.set('spv', this._dialects.get('spirv'));
        this._dialects.set('toy', new mlir.ToyDialect(operations));
        this._dialects.set('top', new mlir.Dialect('top', operations));
        this._dialects.set('tpu', new mlir.Dialect('tpu', operations));
        this._dialects.set('sdfg', new mlir.SdfgDialect(operations));
        this._dialects.set('sdir', this._dialects.get('sdfg'));
        this._dialects.set('check', new mlir.CheckDialect(operations));
        this._dialects.set('tt', new mlir.TritonDialect(operations));
        this._dialects.set('ttg', new mlir.TritonGPUDialect(operations));
        this._dialects.set('triton_gpu', this._dialects.get('ttg'));
        this._dialects.set('gluon', new mlir.GluonDialect(operations));
        this._dialects.set('ttng', new mlir.TritonNvidiaGPUDialect(operations));
        this._dialects.set('nvidia_gpu', this._dialects.get('ttng'));
        this._dialects.set('michelson', new mlir.MichelsonDialect(operations));
        this._redirect = new Map([
            ['builtin.constant', 'arith.constant'],
            ['builtin.return', 'func.return'],
            ['builtin.view', 'memref.view'],
            ['builtin.dealloc', 'memref.dealloc'],
            ['builtin.addi', 'arith.addi'],
            ['builtin.subi', 'arith.subi'],
            ['builtin.muli', 'arith.muli'],
            ['builtin.divi_signed', 'arith.divsi'],
            ['builtin.divi_unsigned', 'arith.divui'],
            ['builtin.divsi', 'arith.divsi'],
            ['builtin.divui', 'arith.divui'],
            ['builtin.andi', 'arith.andi'],
            ['builtin.ori', 'arith.ori'],
            ['builtin.xori', 'arith.xori'],
            ['builtin.shli', 'arith.shli'],
            ['builtin.shrsi', 'arith.shrsi'],
            ['builtin.shrui', 'arith.shrui'],
            ['builtin.addf', 'arith.addf'],
            ['builtin.subf', 'arith.subf'],
            ['builtin.mulf', 'arith.mulf'],
            ['builtin.divf', 'arith.divf'],
            ['builtin.splat', 'vector.splat'],
            ['scf.splat', 'vector.splat'],
            ['flow.constant', 'flow.tensor.constant']
        ]);
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
            if (this.match('eof')) {
                break;
            }
            if (this.match('#')) { // attribute-alias-def
                const name = this.expect();
                this.expect('=');
                const value = this.parseAttributeValue();
                block.definitions.push({ name, value });
                continue;
            }
            if (this.match('!')) { // type-alias-def
                const name = this.expect();
                this.expect('=');
                const type = this.parseType();
                block.definitions.push({ name, type });
                continue;
            }
            if (this.match('{-#')) { // file metadata
                throw new mlir.Error('File metadata is not implemented.');
            }
            const op = this.parseOperation();
            block.operations.push(op);
        }
        return block;
    }

    parseFunctionArgumentList() {
        const inputs = [];
        if (this.accept('(')) {
            while (!this.accept(')')) {
                if (this.match(')')) {
                    break;
                }
                if (this.match('%')) {
                    const input = {};
                    input.value = this._token.value;
                    this.expect('%');
                    this.expect(':');
                    input.type = this.parseType();
                    if (this.match('{')) {
                        input.attributes = [];
                        this.parseAttributeDict(input.attributes);
                    }
                    input.loc = this.parseLocation();
                    inputs.push(input);
                } else {
                    const input = {};
                    input.value = `%arg${inputs.length}`;
                    input.type = this.parseType();
                    inputs.push(input);
                }
                if (!this.match(')')) {
                    if (!this.accept(',')) {
                        break;
                    }
                    if (this.match(')')) {
                        break;
                    }
                }
            }
        }
        return inputs;
    }

    parseFunctionResultList() {
        const outputs = [];
        if (this.accept('(')) {
            while (!this.accept(')')) {
                const output = {};
                output.value = `%result${outputs.length}`;  // Generate a name
                output.type = this.parseType();
                if (this.match('{')) {
                    output.attributes = [];
                    this.parseAttributeDict(output.attributes);
                }
                outputs.push(output);
                this.accept(',');
            }
        } else {
            const output = {};
            output.value = `%result0`;  // Generate a name
            output.type = this.parseType();
            outputs.push(output);
        }
        return outputs;
    }

    parseTypeList() {
        const types = [];
        while (!this.match(')') && !this.match('eof')) {
            const type = this.parseType();
            if (type) {
                types.push(type);
            }
            if (!this.accept(',')) {
                break;
            }
        }
        return types;
    }

    skip(open, close) {
        let value = '';
        if (this.match(open)) {
            value += this.expect();
            let count = 1;
            while (count > 0) {
                if (this.match('eof')) {
                    throw new mlir.Error(`Unexpected end of file while looking for '${close}' ${this.location()}`);
                }
                if (this.match(open)) {
                    count++;
                } else if (this.match(close)) {
                    count--;
                }
                value += this.expect();
            }
        }
        return value;
    }

    parseOperation() {
        const results = [];
        if (this.match('%')) {
            do {
                if (!this.match('%')) {
                    break;
                }
                const value = this.expect();
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
            } while (this.accept(','));
            this.expect('=');
        }
        let op = null;
        if (this.match('id')) {
            op = this.parseCustomOperation(results);
        } else if (this.match('string')) {
            op = this.parseGenericOperation();
        } else {
            throw new mlir.Error(`Unexpected operation name '${this._token.value}' ${this.location()}`);
        }
        if (!op) {
            throw new mlir.Error(`Failed to parse operation ${this.location()}`);
        }
        op.results = results;
        return op;
    }

    parseGenericOperationAfterOpName(op) {
        op.attributes = op.attributes || [];
        op.operands = op.operands || [];
        op.regions = op.regions || [];
        op.results = op.results || [];
        if (this.match('}')) {
            op.loc = this.parseLocation();
            return op;
        }
        if (!op.operands.length) {
            op.operands = this.parseArguments();
        }
        // Parse successor blocks (for branch operations like cf.br)
        if (this.match('^')) {
            op.successors = [];
            while (this.match('^')) {
                const successor = {};
                successor.label = this.expect('^');
                if (this.accept('(')) {
                    successor.arguments = [];
                    while (!this.accept(')')) {
                        if (this.match('%')) {
                            const value = this.expect('%');
                            successor.arguments.push({ value });
                            if (!this.accept(',')) {
                                break;
                            }
                        } else {
                            break;
                        }
                    }
                    const hasTypes = this.accept(':');
                    if (!hasTypes) {
                        this.accept(')');
                    }
                    if (hasTypes) {
                        let idx = 0;
                        while (idx < successor.arguments.length && !this.match(',') && !this.match('[') && !this.match('{') && !this.match('^')) {
                            const type = this.parseType();
                            if (successor.arguments[idx]) {
                                successor.arguments[idx].type = type;
                            }
                            idx++;
                            this.accept(',');
                        }
                        this.accept(')');
                    }
                }
                op.successors.push(successor);
                if (!this.accept(',')) {
                    break;
                }
            }
        }
        while (this.match('[')) {
            this.skip('[', ']');
        }
        this.skip('<', '>');
        let parsedRegionList = false;
        if (this.match('(')) {
            const savedToken = this._token;
            this.accept('(');
            if (this.match('{') || this.match('id') || this.match('^')) {
                parsedRegionList = true;
            } else {
                this._token = savedToken;
            }
        }
        if (parsedRegionList) {
            while (!this.match(')')) {
                let entryLabel = null;
                const entryArgs = [];
                if (this.match('id') && !this.match('{')) {
                    entryLabel = this.expect('id');
                    if (this.accept('(')) {
                        while (!this.accept(')')) {
                            const value = this.expect('%');
                            this.expect(':');
                            const type = this.parseType();
                            entryArgs.push({ value, type });
                            this.accept(',');
                        }
                    }
                }
                if (!this.match('{')) {
                    throw new mlir.Error(`Expected '{' for region in region-list, but got '${this._token.value}' ${this.location()}`);
                }
                const region = {};
                this.parseRegion(region);
                if (entryLabel) {
                    region.entryLabel = entryLabel;
                    region.entryArgs = entryArgs;
                }
                op.regions.push(region);
                if (!this.accept(',') && !this.match(')')) {
                    throw new mlir.Error(`Expected ',' or ')' after region in region-list, but got '${this._token.value}' ${this.location()}`);
                }
            }
            this.expect(')');
        }
        if (this.match('{')) {
            if (parsedRegionList || op.attributes.length === 0 || (op.attributes.length === 1 && op.attributes[0].name === 'predicate')) {
                this.parseAttributeDict(op.attributes);
            } else {
                const region = {};
                this.parseRegion(region);
                op.regions.push(region);
            }
        }
        if (this.accept(':')) {
            this.parseArgumentTypes(op.operands);
        }
        if (this.accept('->') || this.accept('id', 'to')) {
            this.parseArgumentTypes(op.results);
        }
        if (this.accept('id', 'attributes')) {
            if (this.match('{')) {
                this.parseAttributeDict(op.attributes);
            }
        }
        if (this.match('{')) {
            const region = {};
            this.parseRegion(region);
            op.regions.push(region);
            while (this.accept(',') && this.match('{')) {
                const region = {};
                this.parseRegion(region);
                op.regions.push(region);
            }
            if (op.name.endsWith('.if') && this.match('id', 'else')) {
                this.expect('id', 'else');
                const region = {};
                this.parseRegion(region);
                op.regions.push(region);
            }
        }
        op.loc = this.parseLocation();
        return op;
    }

    parseCustomOperation(results) {
        const opNameInfo = this.parseCustomOperationName();
        const op = { name: opNameInfo, results, attributes: [], operands: [], regions: [] };
        if (op.name.startsWith('triton_gpu.')) {
            op.name = op.name.replace('triton_gpu.', 'ttg.');
        }
        if (op.name.startsWith('nvidia_gpu.')) {
            op.name = op.name.replace('nvidia_gpu.', 'ttng.');
        }
        if (this._redirect.has(op.name)) {
            op.name = this._redirect.get(op.name);
        }
        const index = op.name.indexOf('.');
        if (index === -1) {
            throw new mlir.Error(`No dialect found '${op.name}' ${this.location()}`);
        }
        const dialectName = op.name.substring(0, index);
        if (!this._dialects.has(dialectName)) {
            throw new mlir.Error(`Unsupported dialect '${dialectName}' ${this.location()}`);
        }
        const dialect = this._dialects.get(dialectName);
        this._state.defaultDialectStack.push(dialectName);
        if (dialect.parseOperation(this, op.name, op)) {
            op.loc = this.parseLocation() || {};
            this._state.defaultDialectStack.pop();
            return op;
        }
        this._state.defaultDialectStack.pop();
        throw new mlir.Error(`Unsupported custom operation '${op.name}' ${this.location()}`);
    }

    parseCustomOperationName() {
        let opName = this.expect('id');
        if (opName.indexOf('.') === -1) {
            const dialect = this._state.defaultDialectStack[this._state.defaultDialectStack.length - 1];
            opName = `${dialect}.${opName}`;
        }
        return opName;
    }

    parseGenericOperation() {
        const op = { name: this.expect('string'), attributes: [], operands: [], regions: [], results: [] };
        return this.parseGenericOperationAfterOpName(op);
    }

    parseOptionalVisibilityKeyword(attributes) {
        if (this.match('id', 'private') || this.match('id', 'public') || this.match('id', 'nested')) {
            const value = this.expect();
            attributes.push({ name: 'sym_visibility', value });
        }
    }

    parseSymbolName(name, attributes) {
        const value = this.expect('@');
        attributes.push({ name, value });
    }

    parseOptionalSymbolName() {
        if (this.match('@')) {
            return this.expect('@');
        }
        return null;
    }

    parseOptionalAttrDictWithKeyword(attributes) {
        if (this.accept('id', 'attributes')) {
            this.parseAttributeDict(attributes);
        }
    }

    parseOptionalAttrDict(attributes) {
        if (this.match('{')) {
            this.parseAttributeDict(attributes);
        }
    }

    parseAttributeDict(attributes) {
        if (this.accept('{')) {
            while (!this.accept('}')) {
                let name = null;
                if (this.match('id') || this.match('string') || this.match('keyword')) {
                    name = this.expect();
                } else if (this.match('[')) {
                    const arrayValue = this.parseValue();
                    attributes.push({ name: 'array', value: arrayValue.value });
                    this.accept(',');
                    continue;
                } else if (!this.match('=') && !this.match(':') && !this.match('}')) {
                    throw new mlir.Error(`Expected attribute name or '}', but got '${this._token.value}' ${this.location()}`);
                }
                let attribute = {};
                if (this.accept('=') || this.accept(':')) {
                    attribute = this.parseValue();
                    if (this.accept(':')) {
                        attribute.type = this.parseType();
                    }
                } else if (name) {
                    attribute = { name };
                    attributes.push(attribute);
                    this.accept(',');
                    continue;
                } else {
                    break;
                }

                attribute.name = name;
                attributes.push(attribute);
                if (!this.accept(',') && !this.match('}')) {
                    throw new mlir.Error(`Expected ',' or '}' after attribute, but got '${this._token.value}' ${this.location()}`);
                }
            }
        }
    }

    parseRegion(region) {
        region.blocks = Array.isArray(region.blocks) ? region.blocks : [];
        const block = {};
        this.parseBlock(block);
        region.blocks.push(block);

        let hasMultipleBlocks = false;
        while ((this._token.kind === '^' || (this._token.kind === 'id' && this._token.value && this._token.value.startsWith('^'))) && !this.match('}')) {
            hasMultipleBlocks = true;
            const nextBlock = {};
            nextBlock.operations = [];
            nextBlock.arguments = [];
            if (this._token.kind === '^') {
                nextBlock.name = this.expect('^');
            } else {
                nextBlock.name = this.expect('id');
            }
            if (this.accept('(')) {
                while (!this.accept(')')) {
                    const value = this.expect('%');
                    this.expect(':');
                    const type = this.parseType();
                    const arg = { value, type };
                    const loc = this.parseLocation();
                    if (loc) {
                        arg.loc = loc;
                    }
                    nextBlock.arguments.push(arg);
                    this.accept(',');
                }
            }
            if (nextBlock.name && nextBlock.name.endsWith(':')) {
                nextBlock.name = nextBlock.name.slice(0, -1);
            } else {
                this.expect(':');
            }
            while (!(this._token.kind === '^' || (this._token.kind === 'id' && this._token.value && this._token.value.startsWith('^'))) && !this.match('}')) {
                const op = this.parseOperation();
                nextBlock.operations.push(op);
            }
            region.blocks.push(nextBlock);
        }
        if (hasMultipleBlocks && this.match('}')) {
            this.expect('}');
        }
        return region;
    }

    parseBlock(block) {
        block.operations = Array.isArray(block.operations) ? block.operations : [];
        block.arguments = Array.isArray(block.arguments) ? block.arguments : [];
        this.expect('{');
        if (this._token.kind === '^' || (this._token.kind === 'id' && this._token.value && this._token.value.startsWith('^'))) {
            if (this._token.kind === '^') {
                block.name = this.expect('^');
            } else {
                block.name = this.expect('id');
            }
            if (this.accept('(')) {
                while (!this.accept(')') && !this.match('^')) {
                    const value = this.expect('%');
                    this.expect(':');
                    const type = this.parseType();
                    const arg = { value, type };
                    const loc = this.parseLocation();
                    if (loc) {
                        arg.loc = loc;
                    }
                    block.arguments.push(arg);
                    this.accept(',');
                }
            }
            if (block.name && block.name.endsWith(':')) {
                block.name = block.name.slice(0, -1);
            } else {
                this.expect(':');
            }
        }
        while (!this.accept('}')) {
            if (this._token.kind === '^' || (this._token.kind === 'id' && this._token.value && this._token.value.startsWith('^'))) {
                break;
            }
            const op = this.parseOperation();
            block.operations.push(op);
        }
        block.loc = this.parseLocation();
        return block;
    }

    parseLocation() {
        if (this.accept('keyword', 'loc')) {
            const location = {};
            this.expect('(');
            if (this.match('string')) {
                const str = this.expect('string');
                if (this.match('(')) {
                    location.name = str;
                    this.expect('(');
                    location.child = this.parseLocationContent();
                    this.expect(')');
                } else {
                    location.file = str;
                    if (this.accept(':')) {
                        location.line = this.expect('int');
                        if (this.accept(':')) {
                            location.col = this.expect('int');
                        }
                    }
                }
            } else if (this.match('#')) {
                location.alias = this.expect();
            } else if (this.match('id', 'unknown')) {
                this.expect();
                location.unknown = true;
            } else if (this.match('id', 'callsite')) {
                this.expect('id', 'callsite');
                this.expect('(');
                location.type = 'callsite';
                location.callee = this.parseLocationContent();
                this.expect('id', 'at');
                location.caller = this.parseLocationContent();
                this.expect(')');
            } else if (this.match('id', 'fused')) {
                this.expect('id', 'fused');
                location.type = 'fused';
                if (this.accept('<')) {
                    location.metadata = this.parseValue();
                    this.expect('>');
                }
                this.expect('[');
                location.locations = [];
                do {
                    location.locations.push(this.parseLocationContent());
                } while (this.accept(','));
                this.expect(']');
            } else {
                throw new mlir.Error(`Unexpected location '${this._token.value}' ${this.location()}`);
            }
            this.expect(')');
            return location;
        }
        return null;
    }

    parseLocationContent() {
        if (this.match('#')) {
            return { alias: this.expect() };
        }
        if (this.match('keyword', 'loc')) {
            return this.parseLocation();
        }
        throw new mlir.Error(`Expected location content, got '${this._token.value}' ${this.location()}`);
    }

    parseOperationName() {
        switch (this._token.kind) {
            case 'string':
                return this.expect();
            case 'id':
                return this.expect('id');
            default:
                throw new mlir.Error(`Unexpected operation name '${this._token.value}' ${this.location()}`);
        }
    }

    parseArguments() {
        const inputs = [];
        if (this.match('{')) {
            return inputs;
        }
        const open = this.accept('(');
        // eslint-disable-next-line no-unmodified-loop-condition
        while (!this.match(')') && !this.match('->') && !this.match('{') && !this.match('}') && !this.match('[') && !this.match('=') && !this.match('^') && !(this.match(':') && !open)) {
            const input = {};
            if (this._token.kind === 'id' && this._token.value !== 'dense' && this._token.value !== 'dense_resource') {
                const identifier = this.expect('id');
                if (this.accept('(')) {
                    const args = this.parseArguments();
                    for (let i = 0; i < args.length; i++) {
                        const arg = args[i];
                        arg.name = `${identifier}.${i}`;
                        inputs.push(arg);
                    }
                    if (this.accept(':')) {
                        this.parseArgumentTypes(inputs);
                    }
                    this.expect(')');
                    continue;
                } else if (this.match('=')) {
                    input.name = identifier;
                    this.expect('=');
                } else if (this.match(':')) {
                    // Named argument syntax: identifier: value (e.g., init: %init)
                    input.name = identifier;
                    this.expect(':');
                } else {
                    input.value = identifier;
                    inputs.push(input);
                    if (!this.accept(',')) {
                        break;
                    }
                    continue;
                }
            }
            if (this.match('%')) {
                input.value = this.expect();
                if (open && this.accept(':')) {
                    input.type = this.parseType();
                }
            } else if (this.match('keyword', 'loc')) {
                break;
            } else {
                const value = this.parseValue();
                input.type = value.type;
                input.value = value.value;
                if (open && this.accept(':')) {
                    input.type = this.parseType();
                }
            }
            inputs.push(input);
            this.accept(',');
        }
        if (open) {
            this.expect(')');
        }
        return inputs;
    }

    _lookaheadMatch(kind, value) {
        const saved = this._token;
        const savedPos = this._tokenizer._position;
        this._token = this._tokenizer.read();
        const result = this.match(kind, value);
        this._token = saved;
        this._tokenizer._position = savedPos;
        return result;
    }

    _lookahead(fn) {
        const saved = this._token;
        const savedPos = this._tokenizer._position;
        const result = fn();
        this._token = saved;
        this._tokenizer._position = savedPos;
        return result;
    }

    parseElementTypeFromPrefix(prefix, dimensions) {
        if (/^[0-9?]/.test(prefix)) {
            let i = 0;
            while (i < prefix.length) {
                if (prefix[i] === '?') {
                    dimensions.push('?');
                    i++;
                } else if (/[0-9]/.test(prefix[i])) {
                    let numStr = '';
                    while (i < prefix.length && /[0-9]/.test(prefix[i])) {
                        numStr += prefix[i];
                        i++;
                    }
                    dimensions.push(parseInt(numStr, 10));
                } else {
                    break;
                }

                if (i < prefix.length && prefix[i] === 'x') {
                    i++;
                } else {
                    break;
                }
            }

            prefix = prefix.substring(i);
        }

        // Handle nested types like memref<4xvector<16xf32>> or tensor<20x20xcomplex<f32>>
        if (prefix === 'complex') {
            if (this.match('<')) {
                this.expect('<');
                const elementType = this.parseType();
                this.expect('>');
                return `complex<${elementType}>`;
            }
        } else if (prefix === 'tensor' || prefix === 'vector' || prefix === 'memref') {
            if (this.match('<')) {
                this.expect('<');

                const nestedDimInfo = this.parseDimensionListRanked();

                let nestedElementType = null;
                if (nestedDimInfo.elementTypePrefix) {
                    nestedElementType = this.parseElementTypeFromPrefix(nestedDimInfo.elementTypePrefix, nestedDimInfo.dimensions);
                    if (!nestedElementType) {
                        if (this.match('?') || this.match('int')) {
                            const moreDims = this.parseDimensionListRanked();
                            nestedDimInfo.dimensions.push(...moreDims.dimensions);
                            if (moreDims.elementTypePrefix) {
                                nestedElementType = this.parseElementTypeFromPrefix(moreDims.elementTypePrefix, nestedDimInfo.dimensions);
                            } else {
                                nestedElementType = this.parseType();
                            }
                        } else {
                            nestedElementType = this.parseType();
                        }
                    }
                } else {
                    nestedElementType = this.parseType();
                }

                this.expect('>');

                let nestedTypeStr = `${prefix}<`;
                if (nestedDimInfo.unranked) {
                    nestedTypeStr += '*x';
                } else if (nestedDimInfo.dimensions.length > 0) {
                    nestedTypeStr += `${nestedDimInfo.dimensions.join('x')}x`;
                }
                nestedTypeStr += `${nestedElementType}>`;

                return nestedTypeStr;
            }
        }
        if (prefix.startsWith('!')) {
            this._token = new mlir.Token('!', prefix, prefix);
            return this.parseType();
        }
        return prefix;
    }

    parseDimensionListRanked() {
        const dimensions = [];

        if (this.match('*')) {
            this.expect('*');
            if (this.match('id')) {
                const token = this._token.value;
                if (token === 'x' || token.startsWith('x')) {
                    this.expect('id');
                    return { unranked: true, dimensions: [], elementTypePrefix: token === 'x' ? null : token.substring(1) };
                }
            }
            return { unranked: true, dimensions: [], elementTypePrefix: null };
        }

        while (true) {
            if (this.match('[')) {
                this.expect('[');
                if (this.match('int')) {
                    dimensions.push(`[${this.expect('int')}]`);
                } else if (this.match('?')) {
                    dimensions.push('[?]');
                    this.expect('?');
                }
                this.expect(']');
            } else if (this.match('?')) {
                dimensions.push('?');
                this.expect('?');
            } else if (this.match('int')) {
                const tokenText = this._token.text;
                if (tokenText && tokenText.length > 2 && tokenText.startsWith('0x') && /[fiub]/.test(tokenText[2])) {
                    dimensions.push(0);
                    this._token = new mlir.Token('id', tokenText.substring(1), tokenText.substring(1));
                } else {
                    dimensions.push(parseInt(this.expect('int'), 10));
                }
            } else {
                break;
            }

            if (this.match('id')) {
                const token = this._token.value;
                if (token === 'x') {
                    this.expect('id', 'x');
                } else if (token.startsWith('x')) {
                    this.expect('id');
                    const rest = token.substring(1);
                    return { unranked: false, dimensions, elementTypePrefix: rest };
                } else {
                    break;
                }
            } else {
                break;
            }
        }

        return { unranked: false, dimensions, elementTypePrefix: null };
    }

    parseTensorType() {
        this.expect('id', 'tensor');
        this.expect('<');
        const dimInfo = this.parseDimensionListRanked();
        let elementType = null;
        if (dimInfo.elementTypePrefix) {
            elementType = this.parseElementTypeFromPrefix(dimInfo.elementTypePrefix, dimInfo.dimensions);
            if (!elementType) {
                if (this.match('?') || this.match('int')) {
                    const moreDims = this.parseDimensionListRanked();
                    dimInfo.dimensions.push(...moreDims.dimensions);
                    if (moreDims.elementTypePrefix) {
                        elementType = this.parseElementTypeFromPrefix(moreDims.elementTypePrefix, dimInfo.dimensions);
                    } else {
                        elementType = this.parseType();
                    }
                } else {
                    elementType = this.parseType();
                }
            }
        } else {
            elementType = this.parseType();
        }
        let encoding = null;
        if (this.accept(',')) {
            encoding = this.parseAttributeValue();
        }
        this.expect('>');
        let typeStr = 'tensor<';
        if (dimInfo.unranked) {
            typeStr += '*x';
        } else if (dimInfo.dimensions.length > 0) {
            typeStr += `${dimInfo.dimensions.join('x')}x`;
        }
        typeStr += elementType;
        if (encoding) {
            const enc = typeof encoding === 'object' ? JSON.stringify(encoding) : encoding;
            typeStr += `, ${enc}`;
        }
        typeStr += '>';
        return typeStr;
    }

    parseMemRefType() {
        this.expect('id', 'memref');
        this.expect('<');
        const dimInfo = this.parseDimensionListRanked();
        let elementType = null;
        if (dimInfo.elementTypePrefix) {
            elementType = this.parseElementTypeFromPrefix(dimInfo.elementTypePrefix, dimInfo.dimensions);
            if (!elementType) {
                if (this.match('?') || this.match('int')) {
                    const moreDims = this.parseDimensionListRanked();
                    dimInfo.dimensions.push(...moreDims.dimensions);
                    if (moreDims.elementTypePrefix) {
                        elementType = this.parseElementTypeFromPrefix(moreDims.elementTypePrefix, dimInfo.dimensions);
                    } else {
                        elementType = this.parseType();
                    }
                } else {
                    elementType = this.parseType();
                }
            }
        } else {
            elementType = this.parseType();
        }

        const extras = [];
        while (this.accept(',')) {
            const extra = this.parseAttributeValue();
            extras.push(extra);
        }

        this.expect('>');

        let memorySpaceBraces = '';
        if (this.match('{') && this._lookaheadMatch('%')) {
            memorySpaceBraces = this.skip('{', '}');
        }

        let typeStr = 'memref<';
        if (dimInfo.unranked) {
            typeStr += '*x';
        } else if (dimInfo.dimensions.length > 0) {
            typeStr += `${dimInfo.dimensions.join('x')}x`;
        }
        typeStr += elementType;
        if (extras.length > 0) {
            const content = extras.map((e) => typeof e === 'object' ? JSON.stringify(e) : e).join(', ');
            typeStr += `, ${content}`;
        }
        typeStr += '>';
        typeStr += memorySpaceBraces;

        return typeStr;
    }

    parseVectorType() {
        this.expect('id', 'vector');
        this.expect('<');
        const dimInfo = this.parseDimensionListRanked();
        let elementType = null;
        if (dimInfo.elementTypePrefix) {
            elementType = this.parseElementTypeFromPrefix(dimInfo.elementTypePrefix, dimInfo.dimensions);
            if (!elementType) {
                if (this.match('?') || this.match('int')) {
                    const moreDims = this.parseDimensionListRanked();
                    dimInfo.dimensions.push(...moreDims.dimensions);
                    if (moreDims.elementTypePrefix) {
                        elementType = this.parseElementTypeFromPrefix(moreDims.elementTypePrefix, dimInfo.dimensions);
                    } else {
                        elementType = this.parseType();
                    }
                } else {
                    elementType = this.parseType();
                }
            }
        } else {
            elementType = this.parseType();
        }

        this.expect('>');

        let typeStr = 'vector<';
        if (dimInfo.dimensions.length > 0) {
            typeStr += `${dimInfo.dimensions.join('x')}x`;
        }
        typeStr += elementType;
        typeStr += '>';

        return typeStr;
    }

    parseComplexType() {
        this.expect('id', 'complex');
        this.expect('<');
        const elementType = this.parseType();
        this.expect('>');
        return `complex<${elementType}>`;
    }

    parseTupleType() {
        this.expect('id', 'tuple');
        this.expect('<');
        const types = [];
        while (!this.match('>')) {
            types.push(this.parseType());
            if (this.match(',')) {
                this.expect(',');
            }
        }
        this.expect('>');
        return `tuple<${types.join(', ')}>`;
    }

    parseType() {
        if (this._token.kind === 'id') {
            const value = this._token.value;
            if (value === 'none' || value === 'index' || /^[su]?i[0-9]+$/.test(value) ||
                /^f[0-9]+$/.test(value) || value === 'bf16' || value === 'tf32' || value.startsWith('f8') || value.startsWith('f6') || value.startsWith('f4')) {
                return this.expect('id');
            }
            if (value === 'tensor') {
                return this.parseTensorType();
            }
            if (value === 'vector') {
                return this.parseVectorType();
            }
            if (value === 'memref') {
                return this.parseMemRefType();
            }
            if (value === 'complex') {
                return this.parseComplexType();
            }
            if (value === 'tuple') {
                return this.parseTupleType();
            }
        }
        if (this.match('!')) {
            const value = this.expect();  // Read !dialect.typename
            if (value && value.startsWith('!')) {
                const match = value.match(/^!([^.<]+)/);
                if (match) {
                    const [,dialectName] = match;
                    if (!this._dialects.has(dialectName)) {
                        throw new mlir.Error(`Unsupported dialect '${dialectName}' ${this.location()}`);
                    }
                    const dialect = this._dialects.get(dialectName);
                    const parsedType = dialect.parseType(this, value);
                    if (!parsedType) {
                        throw new mlir.Error(`Failed to parse type '${value}' ${this.location()}`);
                    }
                    return parsedType;
                }
            }
            throw new mlir.Error(`Invalid type '${value}' ${this.location()}`);
        }
        if (this.match('(')) {
            let value = this.skip('(', ')');
            if (this.match('->')) {
                value += this.expect();
                if (this.match('(')) {
                    value += this.skip('(', ')');
                } else {
                    const resultType = this.parseType();
                    if (resultType) {
                        value += resultType;
                    }
                }
            }
            return value;
        }
        if (this.match('<')) {
            return this.skip('<', '>');
        }
        throw new mlir.Error(`Invalid type '${this._token.value}' ${this.location()}`);
    }

    parseArgumentTypes(args) {
        let index = 0;
        const open = this.accept('(');
        if (open) {
            while (this._token.kind !== ')') {
                const type = this.parseType();
                if (!type) {
                    break;
                }
                if (index < args.length) {
                    args[index].type = type;
                } else {
                    const arg = {};
                    arg.type = type;
                    arg.value = `%${index}`;
                    args.push(arg);
                }
                index++;
                if (!this.accept(',')) {
                    break;
                }
            }
            this.expect(')');
        } else {
            while (!this.accept(')') &&
                this._token.kind !== '->' &&
                this._token.value !== 'loc' &&
                this._token.value !== 'return' && this._token.value !== 'func.return' &&
                this._token.value !== 'to' &&
                this._token.value !== 'into' &&
                this._token.kind !== '}' &&
                this._token.kind !== '%' &&
                this._token.kind !== '^') {
                const type = this.parseType();
                if (!type) {
                    break;
                }
                if (index < args.length) {
                    args[index].type = type;
                } else {
                    const input = {};
                    input.type = type;
                    input.value = `%${index}`;
                    args.push(input);
                }
                index++;
                if (!this.accept(',')) {
                    break;
                }
            }
        }
    }

    parseValue() {
        const value = {};
        if (this.match('string')) {
            value.value = this.expect();
            value.type = 'string';
            return value;
        }
        if (this.match('int')) {
            value.value = this.expect();
            value.type = 'int64';
            return value;
        }
        if (this.match('float')) {
            value.value = this.expect();
            value.type = 'float32';
            return value;
        }
        if (this.match('boolean')) {
            value.value = this.expect();
            value.type = 'boolean';
            return value;
        }
        if (this.match('%')) {
            value.value = this.expect();
            return value;
        }
        if (this.match('@')) {
            value.value = this.expect();
            return value;
        }
        if (this.match('id', 'DEFAULT')) {
            value.value = this.expect();
            return value;
        }
        if (this.match('id', 'affine_map') || this.match('id', 'affine_set')) {
            const name = this.expect();
            const args = this.skip('<', '>');
            if (this.match('(')) {
                const dimArgs = this.skip('(', ')');
                if (this.match('[')) {
                    const symbolArgs = this.skip('[', ']');
                    return { name, args, dimArgs, symbolArgs };
                }
                return { name, args, dimArgs };
            }
            return { name, args };
        }
        if (this.match('id', 'array')) {
            this.expect('id', 'array');
            this.expect('<');
            const arrayType = this.parseType();
            const arrayValues = [];
            if (this.accept(':')) {
                while (!this.match('>')) {
                    const val = this.parseValue();
                    arrayValues.push(val.value === undefined ? val : val.value);
                    this.accept(',');
                }
            }
            this.expect('>');
            return { value: arrayValues, type: arrayType };
        }
        if (this.accept('[')) {
            const list = [];
            while (!this.accept(']')) {
                const item = this.parseValue();
                if (this.accept(':')) {
                    this.parseType();
                }
                list.push(item.value);
                this.accept(',');
            }
            if (this.accept('id', 'x')) {
                list[0] = Array.from(list);
                const second = [];
                this.expect('[');
                while (!this.accept(']')) {
                    const item = this.parseValue();
                    if (this.accept(':')) {
                        this.parseType();
                    }
                    second.push(item.value);
                    this.accept(',');
                }
                list.push(second);
            }
            return { value: list };
        }
        if (this.match('{')) {
            const attributes = [];
            this.parseAttributeDict(attributes);
            const obj = {};
            for (const attribute of attributes) {
                obj[attribute.name] = attribute.value;
            }
            return { value: obj };
        }
        if (this.match('#')) {
            value.value = this.expect('#');
            if (this.match('<')) {
                value.value += this.skip('<', '>');
            }
            if (this.match('(')) {
                value.value += this.skip('(', ')');
            }
            return value;
        }
        if (this.match('tensor')) {
            value.value = this.parseType();
            value.type = 'type';
            return value;
        }
        if (this.match('id', 'dense_resource')) {
            this.expect('id', 'dense_resource');
            this.expect('<');
            const resourceHandle = this.expect();
            this.expect('>');
            return { value: resourceHandle, type: 'dense' };
        }
        if (this.match('id', 'dense')) {
            this.expect('id', 'dense');
            this.expect('<');
            value.type = 'dense';
            if (this.match('>')) {
                this.expect('>');
                value.value = null;
                return value;
            }
            if (this.match('string')) {
                const hexStr = this.expect();
                if (hexStr.startsWith('"0x') || hexStr.startsWith('0x')) {
                    const cleanHex = hexStr.replace(/"/g, '').substring(2);
                    const data = new Uint8Array(cleanHex.length >> 1);
                    for (let i = 0; i < data.length; i++) {
                        const index = i << 1;
                        data[i] = parseInt(cleanHex.substring(index, index + 2), 16);
                    }
                    value.value = data;
                } else {
                    value.value = hexStr;
                }
            } else if (this.match('[')) {
                const arrayValue = this.parseValue();
                value.value = arrayValue.value;
            } else if (this.match('(')) {
                this.expect('(');
                const real = this.parseValue();
                this.accept(',');
                const imag = this.parseValue();
                this.expect(')');
                value.value = { real: real.value, imag: imag.value };
            } else {
                const scalarValue = this.parseValue();
                value.value = scalarValue.value;
            }
            this.expect('>');
            return value;
        }
        if (this._token.kind === 'id') {
            const tokenValue = this._token.value;
            if (tokenValue === 'tensor' || tokenValue === 'vector' || tokenValue === 'memref' ||
                tokenValue === 'none' || tokenValue === 'index' || /^[su]?i[0-9]+$/.test(tokenValue) ||
                /^f[0-9]+$/.test(tokenValue) || tokenValue === 'bf16' || tokenValue === 'tf32' ||
                tokenValue.startsWith('f8')) {
                const type = this.parseType();
                return { value: type, type: 'type' };
            }
        }
        if (this.match('!')) {
            const type = this.parseType();
            return { value: type, type: 'type' };
        }
        if (this.match('id')) {
            value.value = this.expect('id');
            return value;
        }
        if (this.match('(')) {
            this.expect('(');
            const real = this.parseValue();
            this.accept(',');
            const imag = this.parseValue();
            this.expect(')');
            return { value: { real: real.value, imag: imag.value }, type: 'complex' };
        }
        throw new mlir.Error(`Unexpected value '${this._token.value}' ${this.location()}`);
    }

    parseAttributeValue() {
        if (this.match('keyword', 'loc')) {
            return this.parseLocation();
        }
        if (this.match('id', 'affine_map') || this.match('id', 'affine_set')) {
            const name = this.expect();
            const args = this.skip('<', '>');
            return { name, args };
        }
        if (this.match('#')) {
            const name = this.expect();
            if (this.accept('<')) {
                const bracketStack = ['<'];
                while (bracketStack.length > 0) {
                    if (this.match('eof')) {
                        throw new mlir.Error(`Unexpected end of file while parsing attribute ${this.location()}`);
                    }
                    const token = this._token.kind;
                    if (token === '<' || token === '{' || token === '[' || token === '(') {
                        bracketStack.push(token);
                        this._token = this._tokenizer.read();
                    } else if (token === '>') {
                        if (bracketStack[bracketStack.length - 1] === '<') {
                            bracketStack.pop();
                            if (bracketStack.length === 0) {
                                this.expect('>');
                            } else {
                                this._token = this._tokenizer.read();
                            }
                        } else {
                            this._token = this._tokenizer.read();
                        }
                    } else if (token === '}') {
                        if (bracketStack[bracketStack.length - 1] === '{') {
                            bracketStack.pop();
                        }
                        this._token = this._tokenizer.read();
                    } else if (token === ']') {
                        if (bracketStack[bracketStack.length - 1] === '[') {
                            bracketStack.pop();
                        }
                        this._token = this._tokenizer.read();
                    } else if (token === ')') {
                        if (bracketStack[bracketStack.length - 1] === '(') {
                            bracketStack.pop();
                        }
                        this._token = this._tokenizer.read();
                    } else {
                        this._token = this._tokenizer.read();
                    }
                }
            }
            return { name };
        }
        if (this.match('<')) {
            this.expect('<');
            const dict = {};
            while (!this.match('>')) {
                if (this.match('id')) {
                    const key = this.expect('id');
                    if (this.accept('=')) {
                        const value = this.parseAttributeValue();
                        dict[key] = value;
                    }
                }
                this.accept(',');
            }
            this.expect('>');
            return dict;
        }
        if (this.match('(')) {
            const parts = [];
            let depth = 0;
            let seenArrow = false;
            while (true) {
                if (this.match('(')) {
                    depth++;
                    parts.push(this.expect());
                } else if (this.match(')')) {
                    parts.push(this.expect());
                    depth--;
                    if (depth === 0) {
                        if (seenArrow) {
                            break;
                        } else if (!this.match('->')) {
                            break;
                        }
                    }
                } else if (this.match('->')) {
                    seenArrow = true;
                    parts.push(this.expect());
                } else {
                    parts.push(this.expect());
                }
            }
            return { affine_map: parts.join(' ') };
        }
        if (this._token.kind === 'id') {
            const value = this._token.value;
            if (value === 'tensor' || value === 'vector' || value === 'memref' ||
                value === 'none' || value === 'index' || /^[su]?i[0-9]+$/.test(value) ||
                /^f[0-9]+$/.test(value) || value === 'bf16') {
                return this.parseType();
            }
        }
        if (this.match('!')) {
            return this.parseType();
        }
        return this.parseValue();
    }

    match(kind, value) {
        return (this._token.kind === kind && (!value || this._token.value === value));
    }

    expect(kind, value) {
        if (kind && this._token.kind !== kind) {
            throw new mlir.Error(`Expected token of type '${kind}', but got '${this._token.value}' ${this.location()}`);
        }
        if (value && this._token.value !== value) {
            throw new mlir.Error(`Expected token with value '${value}', but got '${this._token.value}' ${this.location()}`);
        }
        const token = this._token;
        this._token = this._tokenizer.read();
        return token.value;
    }

    accept(kind, value) {
        if (this.match(kind, value)) {
            return this.expect();
        }
        return null;
    }

    get token() {
        return this._token;
    }

    location() {
        return this._tokenizer.location();
    }
};

mlir.BytecodeReader = class {

    constructor(reader) {
        this._reader = new mlir.BinaryReader(reader);
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
        const parseCustomEntry = (/* entry, reader, entryType */) => {
        };
        const parseAsmEntry = (/* entry, reader, entryType */) => {
        };
        const resolveEntries = (range, entryType) => {
            for (const entry of this.attributes) {
                reader.seek(offset + entry.offset);
                if (entry.hasCustomEncoding) {
                    parseCustomEntry(entry, reader);
                } else {
                    parseAsmEntry(entry, reader, entryType);
                }
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

    constructor(reader) {
        this._reader = reader;
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
            case 'index': return 'int64';
            case 'f16': return 'float16';
            case 'f32': return 'float32';
            case 'f64': return 'float64';
            case 'bf16': return 'bfloat16';
            case 'fp8': return 'float8';
            case 'fp8e4m3': return 'float8e4m3';
            case 'fp8_e4m3': return 'float8e4m3';
            case 'fp8e4m3fn': return 'float8e4m3fn';
            case 'fp8e5m2': return 'float8e5m2';
            case 'fp8_e5m2': return 'float8e5m2';
            case 'f4E2M1FN': return 'float4e2m1fn';
            case 'f6E2M3FN': return 'float6e2m3fn';
            case 'f6E3M2FN': return 'float6e3m2fn';
            case 'f8E3M4': return 'float8e3m4';
            case 'f8E4M3': return 'float8e4m3';
            case 'f8E4M3B11FNUZ': return 'float8e4m3b11fnuz';
            case 'f8E4M3FN': return 'float8e4m3fn';
            case 'f8E4M3FNUZ': return 'float8e4m3fnuz';
            case 'f8E5M2': return 'float8e5m2';
            case 'f8E5M2FNUZ': return 'float8e5m2fnuz';
            case 'f8E8M0FNU': return 'float8e8m0fnu';
            case 'float8': return 'float8';
            case 'tf32': return 'tensorfloat32';
            case 'i1': return 'boolean';
            case 'i2': return 'int2';
            case 'i4': return 'int4';
            case 'i8': return 'int8';
            case 'i16': return 'int16';
            case 'i32': return 'int32';
            case 'i64': return 'int64';
            case 'si8': return 'int8';
            case 'si16': return 'int16';
            case 'si32': return 'int32';
            case 'si64': return 'int64';
            case 'ui1': return 'uint1';
            case 'ui2': return 'uint2';
            case 'ui4': return 'uint4';
            case 'ui8': return 'uint8';
            case 'ui16': return 'uint16';
            case 'ui32': return 'uint32';
            case 'ui64': return 'uint64';
            case 'complex<f32>': return 'complex64';
            case 'complex<f64>': return 'complex128';
            case 'b8': return 'int8';
            case 'boolean': return 'boolean';
            default:
                if (value && value.startsWith('!')) {
                    // Workaround for !quant.uniform
                    return value;
                }
                throw new mlir.Error(`Unknown data type '${value}'.`);
        }
    }

    static valueType(type) {
        if (type === undefined) {
            return null;
        }
        // Handle dialect-specific types like !tosa.shape<3>, !quant.uniform<...>, etc.
        // These should be returned as-is without trying to decompose them
        if (type.startsWith('!') && !type.startsWith('!torch.vtensor<')) {
            return type;
        }
        // eg. tensor<?x3x2x2xf32> or tensor<20x20xcomplex<f64>>
        if (type.startsWith('tensor<') && type.endsWith('>')) {
            const spec = type.substring(7, type.length - 1).trim();
            if (spec.startsWith('!')) {
                return mlir.Utility.valueType(spec);
            }
            // Parse dimensions from left to right until we hit the element type
            // Dimensions are numbers, '?', or '*' separated by 'x'
            let i = 0;
            const shape = [];
            while (i < spec.length) {
                if (spec[i] === '?' || spec[i] === '*') {
                    shape.push('?');
                    i++;
                } else if (/[0-9]/.test(spec[i])) {
                    let numStr = '';
                    while (i < spec.length && /[0-9]/.test(spec[i])) {
                        numStr += spec[i];
                        i++;
                    }
                    const dim = parseInt(numStr, 10);
                    if (isNaN(dim)) {
                        // This shouldn't happen, but handle it gracefully
                        shape.push('?');
                    } else {
                        shape.push(dim);
                    }
                } else {
                    // Not a dimension, rest is the element type
                    break;
                }
                // Skip 'x' separator
                if (i < spec.length && spec[i] === 'x') {
                    i++;
                } else {
                    break;
                }
            }
            let dataType = spec.substring(i);
            // Strip encoding if present (e.g., "f32, #stablehlo.type_extensions<...>")
            const encodingIndex = dataType.indexOf(',');
            if (encodingIndex !== -1) {
                dataType = dataType.substring(0, encodingIndex).trim();
            }
            return new mlir.TensorType(dataType, new mlir.TensorShape(shape));
        }
        if (type.startsWith('!torch.vtensor<') && type.endsWith('>')) {
            const spec = type.substring(15, type.length - 1);
            const index = spec.lastIndexOf(',');
            const shapeStr = spec.substring(0, index).replace(/\?/g, '"?"');
            const shape = JSON.parse(shapeStr);
            const dataType = spec.substring(index + 1);
            return new mlir.TensorType(dataType, new mlir.TensorShape(shape));
        }
        // Handle tuple types: tuple<tensor<f32>, tensor<i32>>
        if (type.startsWith('tuple<') && type.endsWith('>')) {
            // const spec = type.substring(6, type.length - 1).trim();
            // For now, return as-is since we don't have a TupleType class
            // In the future, we could parse the comma-separated types
            return type;
        }
        return type;
    }
};

// Dialect Plugin System

mlir.AssemblyFormatParser = class {

    constructor(format) {
        this._format = format || '';
        this._pos = 0;
    }

    parse() {
        const directives = [];
        this._skipWhitespace();
        while (this._pos < this._format.length) {
            const directive = this._parseDirective();
            if (directive) {
                directives.push(directive);
            }
            this._skipWhitespace();
        }
        return directives;
    }

    _parseDirective() {
        const ch = this._format[this._pos];
        // Optional group: (...)?
        // Check if this is the start of an optional group by looking ahead for `?` after matching `)`
        if (ch === '(') {
            // Look ahead to see if this is an optional group
            const savedPos = this._pos;
            this._pos++; // skip '('
            let depth = 1;
            let tempPos = this._pos;
            // Find matching ')'
            while (tempPos < this._format.length && depth > 0) {
                if (this._format[tempPos] === '(') {
                    depth++;
                } else if (this._format[tempPos] === ')') {
                    depth--;
                }
                tempPos++;
            }
            // Check if there's a '?' after the ')'
            tempPos = this._skipWhitespaceAt(tempPos);
            const isOptional = tempPos < this._format.length && this._format[tempPos] === '?';

            if (isOptional) {
                // This is an optional group - parse it as a single directive
                const elements = [];
                let anchorElement = null;
                this._skipWhitespace();
                let loopCount = 0;
                const maxLoops = 100;
                while (this._pos < this._format.length && this._format[this._pos] !== ')') {
                    if (loopCount++ > maxLoops) {
                        throw new Error(`Infinite loop detected in optional group parsing at position ${this._pos}`);
                    }
                    const startPos = this._pos;
                    const elem = this._parseDirective();
                    if (elem) {
                        // Check if this element has an anchor marker
                        if (elem.anchor) {
                            anchorElement = elem.name || elem.type;
                        }
                        elements.push(elem);
                    }
                    // Safety check: ensure we're making progress
                    if (this._pos === startPos) {
                        // If we didn't advance, skip one character to prevent infinite loop
                        this._pos++;
                    }
                    this._skipWhitespace();
                }
                this._pos++; // skip ')'
                this._skipWhitespace();
                this._pos++; // skip '?'
                return { type: 'optional_group', elements, anchor: anchorElement };
            }
            // Not an optional group, restore position and treat '(' as literal
            this._pos = savedPos;
        }
        // Literal: `keyword`
        if (ch === '`') {
            this._pos++;
            const value = this._parseUntil('`');
            this._pos++; // skip closing `
            // MLIR reference: Empty literals (`` or ` `) are whitespace, not literals
            if (value.length === 0 || value === ' ' || value === '\\n') {
                return null; // Skip whitespace elements
            }
            return { type: 'literal', value };
        }
        // Variable reference: $name with optional anchor ^
        if (ch === '$') {
            this._pos++;
            const name = this._parseIdentifier();
            // Check for anchor marker ^
            let hasAnchor = false;
            if (this._pos < this._format.length && this._format[this._pos] === '^') {
                hasAnchor = true;
                this._pos++; // consume '^'
            }
            // Check if it's a special keyword that maps to a directive
            if (name === 'operands') {
                return { type: 'operands', anchor: hasAnchor };
            }
            if (name === 'results') {
                return { type: 'results', anchor: hasAnchor };
            }
            if (name === 'regions') {
                return { type: 'regions', anchor: hasAnchor };
            }
            if (name === 'successors') {
                return { type: 'successors', anchor: hasAnchor };
            }
            // Otherwise, it's a named operand/attribute reference
            return { type: 'operand_ref', name, anchor: hasAnchor };
        }
        // Check for keywords
        const remaining = this._format.substring(this._pos);
        if (remaining.startsWith('type(')) {
            this._pos += 'type'.length;
            const args = this._parseParenList();
            return { type: 'type', args };
        }
        if (remaining.startsWith('qualified(')) {
            this._pos += 'qualified'.length;
            const args = this._parseParenList();
            return { type: 'qualified', args };
        }
        if (remaining.startsWith('attr-dict-with-keyword')) {
            this._pos += 'attr-dict-with-keyword'.length;
            return { type: 'attr_dict_with_keyword' };
        }
        if (remaining.startsWith('attr-dict')) {
            this._pos += 'attr-dict'.length;
            return { type: 'attr_dict' };
        }
        if (remaining.startsWith('operands')) {
            this._pos += 'operands'.length;
            return { type: 'operands' };
        }
        if (remaining.startsWith('results')) {
            this._pos += 'results'.length;
            return { type: 'results' };
        }
        if (remaining.startsWith('regions')) {
            this._pos += 'regions'.length;
            return { type: 'regions' };
        }
        if (remaining.startsWith('successors')) {
            this._pos += 'successors'.length;
            return { type: 'successors' };
        }
        if (remaining.startsWith('functional-type')) {
            this._pos += 'functional-type'.length;
            const args = this._parseParenList();
            return { type: 'functional_type', args };
        }
        if (remaining.startsWith('custom<')) {
            this._pos += 'custom<'.length;
            const parser = this._parseUntil('>');
            this._pos++; // consume '>'
            const args = this._parseParenList();
            return { type: 'custom', parser, args };
        }
        if (remaining.startsWith('oilist(')) {
            this._pos += 'oilist'.length;
            this._parseParenList();
            return null;
        }
        if (/^[:()[\]{}<>,=|]/.test(ch)) {
            this._pos++;
            return { type: 'literal', value: ch };
        }
        this._pos++;
        return null;
    }

    _parseIdentifier() {
        let name = '';
        while (this._pos < this._format.length) {
            const ch = this._format[this._pos];
            if (/[a-zA-Z0-9_]/.test(ch)) {
                name += ch;
                this._pos++;
            } else {
                break;
            }
        }
        return name;
    }

    _parseUntil(terminator) {
        let value = '';
        while (this._pos < this._format.length && this._format[this._pos] !== terminator) {
            value += this._format[this._pos];
            this._pos++;
        }
        return value;
    }

    _parseParenList() {
        if (this._format[this._pos] !== '(') {
            return [];
        }
        this._pos++;
        const items = [];
        let parenDepth = 1;
        while (this._pos < this._format.length && parenDepth > 0) {
            this._skipWhitespace();
            const startPos = this._pos;
            if (this._format[this._pos] === '$') {
                this._pos++;
                const item = this._parseIdentifier();
                if (item) {
                    items.push(`$${item}`);
                }
            } else if (this._format[this._pos] === '(') {
                parenDepth++;
                this._pos++;
            } else if (this._format[this._pos] === ')') {
                parenDepth--;
                if (parenDepth > 0) {
                    this._pos++;
                }
            } else if (this._format[this._pos] === ',') {
                this._pos++;
            } else {
                const item = this._parseIdentifier();
                if (item) {
                    items.push(item);
                }
                if (this._pos === startPos) {
                    this._pos++;
                }
            }
        }
        if (this._pos < this._format.length && this._format[this._pos] === ')') {
            this._pos++;
        }
        return items;
    }

    _skipWhitespace() {
        while (this._pos < this._format.length && /\s/.test(this._format[this._pos])) {
            this._pos++;
        }
    }

    _skipWhitespaceAt(pos) {
        while (pos < this._format.length && /\s/.test(this._format[pos])) {
            pos++;
        }
        return pos;
    }
};

mlir.Dialect = class {

    constructor(name, operations) {
        this._name = name;
        this._operations = new Map();
        this._customParsers = new Map();
        this.registerCustomParser('SameOperandsAndResultType', this._parseSameOperandsAndResultType.bind(this));
        this.registerCustomParser('VariadicSameOperandsAndResultType', this._parseVariadicSameOperandsAndResultType.bind(this));
        this.registerCustomParser('ComplexOpType', this._parseComplexOpType.bind(this));
        this.registerCustomParser('SelectOpType', this._parseSelectOpType.bind(this));
        this.registerCustomParser('TupleOpType', this._parseTupleOpType.bind(this));
        this.registerCustomParser('PairwiseOpType', this._parsePairwiseOpType.bind(this));
        this.registerCustomParser('ConvolutionDimensions', this._parseConvolutionDimensions.bind(this));
        this.registerCustomParser('DotDimensionNumbers', this._parseDotDimensionNumbers.bind(this));
        this.registerCustomParser('PrecisionConfig', this._parsePrecisionConfig.bind(this));
        this.registerCustomParser('PrecisionConfigAndAlgorithm', this._parsePrecisionConfigAndAlgorithm.bind(this));
        this.registerCustomParser('WindowAttributes', this._parseWindowAttributes.bind(this));
        this.registerCustomParser('SliceRanges', this._parseSliceRanges.bind(this));
        this.registerCustomParser('CustomCallTarget', this._parseCustomCallTarget.bind(this));
        this.registerCustomParser('VariadicOperandWithAttribute', this._parseVariadicOperandWithAttribute.bind(this));
        this.registerCustomParser('DynamicIndexList', this._parseDynamicIndexList.bind(this));
        this.registerCustomParser('SymbolVisibility', this._parseSymbolVisibility.bind(this));
        this.registerCustomParser('OptionalUnitAttr', this._parseOptionalUnitAttr.bind(this));
        for (const op of operations.get(name) || []) {
            if (op.assemblyFormat) {
                const parser = new mlir.AssemblyFormatParser(op.assemblyFormat);
                const directives = parser.parse();
                this._operations.set(op.name, { metadata: op, directives });
            }
        }
    }

    registerCustomParser(name, parserFn) {
        this._customParsers.set(name, parserFn);
    }

    parseType(/* parser, dialectType */) {
        // Default implementation: return null to indicate no custom parsing
        // Subclasses should override this to provide dialect-specific type parsing
        return null;
    }

    parseOperation(parser, opName, op) {
        const name = opName.replace(/^"|"$/g, '');
        const opInfo = this._operations.get(name);
        if (!opInfo) {
            return false;
        }
        for (const directive of opInfo.directives) {
            switch (directive.type) {
                case 'literal':
                    parser.expect(null, directive.value);
                    break;
                case 'operand_ref': {
                    // Parse operand/attribute reference like $lhs, $rhs, or $value
                    // Check if this is an attribute reference
                    const refName = directive.name;
                    let isAttribute = false;
                    let isVariadic = false;
                    let isSuccessor = false;
                    if (opInfo.metadata && opInfo.metadata.successors) {
                        const successorInfo = opInfo.metadata.successors.find((succ) => succ.name === refName);
                        if (successorInfo) {
                            isSuccessor = true;
                        }
                    }
                    if (!isSuccessor && parser.match('^')) {
                        isSuccessor = true;
                    }
                    // Check if this is an attribute
                    if (!isSuccessor && opInfo.metadata && opInfo.metadata.attributes) {
                        const attrInfo = opInfo.metadata.attributes.find((attr) => attr.name === refName);
                        if (attrInfo) {
                            isAttribute = true;
                        }
                    }
                    // Check if this is a Variadic operand
                    if (!isAttribute && !isSuccessor && opInfo.metadata && opInfo.metadata.inputs) {
                        const inputInfo = opInfo.metadata.inputs.find((inp) => inp.name === refName);
                        if (inputInfo && inputInfo.type === 'Variadic') {
                            isVariadic = true;
                        }
                    }
                    // Handle successors: parse block labels like ^bb1
                    if (isSuccessor) {
                        if (parser.match('^')) {
                            if (!op.successors) {
                                op.successors = [];
                            }
                            const successor = {};
                            successor.label = parser.expect('^');
                            // Parse successor arguments with types if present
                            // Format: ^label(%val1, %val2, ... : type1, type2, ...)
                            if (parser.accept('(')) {
                                successor.arguments = [];
                                // Parse all values first
                                while (!parser.match(':') && !parser.match(')')) {
                                    if (parser.match('%')) {
                                        const arg = {};
                                        arg.value = parser.expect('%');
                                        successor.arguments.push(arg);
                                        parser.accept(',');
                                    } else {
                                        break;
                                    }
                                }
                                // Parse types if present
                                if (parser.accept(':')) {
                                    let idx = 0;
                                    while (idx < successor.arguments.length && !parser.match(')')) {
                                        const type = parser.parseType();
                                        successor.arguments[idx].type = type;
                                        idx++;
                                        parser.accept(',');
                                    }
                                }
                                parser.accept(')');
                            }
                            op.successors.push(successor);
                        }
                    } else if (isAttribute) {
                        const attrValue = parser.parseValue();
                        if (attrValue) {
                            // Check for optional type annotation after constant values (e.g., 0.0 : f64, dense<...> : tensor<...>)
                            // Parse the type annotation for typed attributes
                            if ((attrValue.type === 'int64' || attrValue.type === 'float32' || attrValue.type === 'boolean' || attrValue.type === 'dense') &&
                                parser.accept(':')) {
                                parser.parseType();
                            }
                            // For attributes, we only store the value, not the internal "type" field
                            // The type field here is just metadata about how the value was parsed (e.g., 'dense')
                            op.attributes.push({ name: refName, value: attrValue.value });
                        }
                    } else if (isVariadic) {
                        // For Variadic operands, parse comma-separated list
                        // Parse comma-separated operands until we hit a delimiter
                        while (!parser.match(')') && !parser.match(']') && !parser.match('}') && !parser.match(':')) {
                            if (parser.match('%')) {
                                const input = {};
                                input.value = parser.expect();
                                op.operands.push(input);
                            } else {
                                break;
                            }
                            // Skip optional comma
                            parser.accept(',');
                        }
                    } else if (parser.match('%')) {
                        const input = {};
                        input.value = parser.expect();
                        op.operands.push(input);
                    } else if ((refName === 'region' || refName === 'regions' || refName.endsWith('Region') || refName.endsWith('Regions')) && parser.match('{')) {
                        // Parse region reference like $region or $bodyRegion
                        const region = {};
                        parser.parseRegion(region);
                        op.regions.push(region);
                    } else if (parser.match('{')) {
                        // If we see '{' but haven't identified this as an attribute/operand/successor,
                        // it's likely a region that wasn't explicitly named "region"
                        const region = {};
                        parser.parseRegion(region);
                        op.regions.push(region);
                    } else if (parser.match('@')) {
                        // Symbol reference like @my_func - add as attribute instead of operand
                        const value = parser.expect('@');
                        if (directive.name) {
                            op.attributes.push({ name: directive.name, value });
                        } else {
                            op.attributes.push({ name: 'callee', value });
                        }
                    } else if (parser.match('id')) {
                        const input = {};
                        input.value = parser.expect('id');
                        op.operands.push(input);
                    } else if (parser.match('int')) {
                        const input = {};
                        input.value = parser.expect('int');
                        op.operands.push(input);
                    } else if (!parser.match(':') && !parser.match(')') && !parser.match(']')) {
                        // Try to parse as a general value, but not if we're at a delimiter
                        const input = parser.parseValue();
                        if (input) {
                            op.operands.push(input);
                        }
                    }
                    break;
                }
                case 'operands':
                    op.operands = parser.parseArguments();
                    break;
                case 'results':
                    op.results = parser.parseArguments();
                    break;
                case 'type':
                case 'qualified':
                    if (directive.args && directive.args.length > 0) {
                        let [arg] = directive.args;
                        if (arg === 'type' && directive.args.length > 1) {
                            [, arg] = directive.args;
                        }
                        if (arg === 'results' || arg === '$results') {
                            parser.parseArgumentTypes(op.results);
                        } else if (arg === 'operands' || arg === '$operands') {
                            parser.parseArgumentTypes(op.operands);
                        } else {
                            const opMetadata = opInfo.metadata;
                            let isResult = false;
                            let isVariadic = false;
                            if (opMetadata && opMetadata.outputs) {
                                for (const output of opMetadata.outputs) {
                                    if (output.name === arg || `$${output.name}` === arg) {
                                        isResult = true;
                                        isVariadic = output.type === 'Variadic' || output.isVariadic || false;
                                        break;
                                    }
                                }
                            }
                            if (!isResult && opMetadata && opMetadata.inputs) {
                                for (const input of opMetadata.inputs) {
                                    if (input.name === arg || `$${input.name}` === arg) {
                                        isVariadic = input.type === 'Variadic' || input.isVariadic || false;
                                        break;
                                    }
                                }
                            }
                            if (isResult) {
                                if (isVariadic) {
                                    parser.parseArgumentTypes(op.results);
                                } else {
                                    const type = parser.parseType();
                                    if (op.results.length === 0) {
                                        op.results.push({ type });
                                    } else {
                                        op.results[0].type = type;
                                    }
                                }
                            } else if (isVariadic) {
                                parser.parseArgumentTypes(op.operands);
                            } else if (op.operands.length > 0) {
                                op.operands[op.operands.length - 1].type = parser.parseType();
                            } else {
                                parser.parseType();
                            }
                        }
                    } else {
                        parser.parseArgumentTypes(op.operands);
                    }
                    break;
                case 'attr_dict_with_keyword':
                    if (parser.match('id') && parser.token.value === 'attributes') {
                        parser.expect('id');
                    }
                    parser.parseAttributeDict(op.attributes);
                    break;
                case 'attr_dict':
                    parser.parseAttributeDict(op.attributes);
                    break;
                case 'regions':
                    // Parse regions
                    while (parser.match('{')) {
                        const region = {};
                        parser.parseRegion(region);
                        op.regions.push(region);
                    }
                    break;
                case 'successors':
                    // Skip successors for now
                    if (parser.match('[')) {
                        parser.skip('[', ']');
                    }
                    break;
                case 'functional_type': {
                    // functional-type(operands, results) parses: (input_types) -> (result_types)
                    // Note: ':' before functional-type should be handled by a separate literal directive
                    // Parse input types: (type1, type2, ...)
                    parser.parseArgumentTypes(op.operands);
                    // Parse arrow and result types
                    if (parser.accept('->') || parser.accept('id', 'to')) {
                        if (op.results.length > 0) {
                            parser.parseArgumentTypes(op.results);
                        } else {
                            op.results = parser.parseArguments();
                        }
                    }
                    break;
                }
                case 'custom': {
                    const fn = this._customParsers.get(directive.parser);
                    if (!fn) {
                        break;
                    }
                    const result = fn(parser, directive.args);
                    if (result) {
                        if (result.kind === 'SameOperandsAndResultType' && result.type) {
                            for (const operand of op.operands) {
                                if (!operand.type) {
                                    operand.type = result.type;
                                }
                            }
                            for (const res of op.results) {
                                if (!res.type) {
                                    res.type = result.type;
                                }
                            }
                        } else if (result.kind === 'PairwiseOpType') {
                            if (result.operandTypes && op.operands.length > 0) {
                                for (let i = 0; i < Math.min(result.operandTypes.length, op.operands.length); i++) {
                                    if (!op.operands[i].type) {
                                        op.operands[i].type = result.operandTypes[i];
                                    }
                                }
                            }
                            if (result.resultTypes && op.results.length > 0) {
                                for (let i = 0; i < Math.min(result.resultTypes.length, op.results.length); i++) {
                                    if (!op.results[i].type) {
                                        op.results[i].type = result.resultTypes[i];
                                    }
                                }
                            }
                        } else if (result.kind === 'PrecisionConfig' && result.precision && result.precision.length > 0) {
                            op.attributes.push({ name: 'precision_config', value: result.precision });
                        } else if (result.kind === 'PrecisionConfigAndAlgorithm') {
                            if (result.precision && result.precision.length > 0) {
                                op.attributes.push({ name: 'precision_config', value: result.precision });
                            }
                            if (result.algorithm) {
                                op.attributes.push({ name: 'algorithm', value: result.algorithm });
                            }
                        }
                    }
                    break;
                }
                case 'optional_group': {
                    let shouldParse = false;

                    const [firstElem] = directive.elements;
                    if (firstElem) {
                        if (firstElem.type === 'literal') {
                            // Check if literal matches current token
                            // Single-char literals like '(', '{', etc. are token kinds
                            // Multi-char literals like 'ins', 'outs' are identifier/keyword values
                            if (firstElem.value.length === 1 && /[(){}[\],:<>]/.test(firstElem.value)) {
                                shouldParse = parser.match(firstElem.value);
                            } else {
                                shouldParse = parser.match('id', firstElem.value) || parser.match('keyword', firstElem.value);
                            }
                        } else if (firstElem.type === 'operand_ref') {
                            shouldParse = parser.match('%');
                        } else if (firstElem.type === 'operands') {
                            shouldParse = parser.match('(') || parser.match('%');
                        }
                    }

                    if (shouldParse) {
                        // Parse all elements in the group
                        for (const elem of directive.elements) {
                            // Recursively handle each element by switching on its type
                            switch (elem.type) {
                                case 'literal':
                                    parser.expect(null, elem.value);
                                    break;
                                case 'operand_ref': {
                                    const refName = elem.name;
                                    let isAttribute = false;
                                    let isVariadic = false;
                                    if (opInfo.metadata && opInfo.metadata.attributes) {
                                        const attrInfo = opInfo.metadata.attributes.find((attr) => attr.name === refName);
                                        if (attrInfo) {
                                            isAttribute = true;
                                        }
                                    }
                                    if (!isAttribute && opInfo.metadata && opInfo.metadata.inputs) {
                                        const inputInfo = opInfo.metadata.inputs.find((inp) => inp.name === refName);
                                        if (inputInfo && inputInfo.type === 'Variadic') {
                                            isVariadic = true;
                                        }
                                    }
                                    if (isAttribute) {
                                        const value = parser.parseValue();
                                        op.attributes.push({ name: refName, value: value.value === undefined ? value : value.value });
                                    } else if (isVariadic) {
                                        while (parser.match('%')) {
                                            const operand = parser.parseValue();
                                            op.operands.push(operand);
                                            if (!parser.accept(',')) {
                                                break;
                                            }
                                        }
                                    } else {
                                        const operand = parser.parseValue();
                                        op.operands.push(operand);
                                    }
                                    break;
                                }
                                case 'operands': {
                                    // Parse operands directive
                                    op.operands = parser.parseArguments();
                                    break;
                                }
                                case 'type': {
                                    // Parse type for operands/results
                                    if (elem.args && elem.args.length > 0) {
                                        const [arg] = elem.args;
                                        if (arg.startsWith('$')) {
                                            const varName = arg.substring(1);
                                            // Check if it's an operand or result
                                            if (varName === 'result' || varName === 'results') {
                                                // Need to check metadata to determine if this is an input or output
                                                let isInput = false;
                                                if (opInfo.metadata && opInfo.metadata.inputs) {
                                                    isInput = opInfo.metadata.inputs.some((inp) => inp.name === varName);
                                                }
                                                if (isInput) {
                                                    parser.parseArgumentTypes(op.operands);
                                                } else {
                                                    parser.parseArgumentTypes(op.results);
                                                }
                                            } else if (varName === 'operands') {
                                                // Parse types for all operands
                                                parser.parseArgumentTypes(op.operands);
                                            } else {
                                                // Parse type for specific operand
                                                const type = parser.parseType();
                                                if (op.operands.length > 0) {
                                                    op.operands[op.operands.length - 1].type = type;
                                                }
                                            }
                                        }
                                    }
                                    break;
                                }
                                case 'results': {
                                    // Parse Variadic results/operands
                                    // Just consume tokens for now to test if parsing itself is the issue
                                    while (parser.match('%')) {
                                        parser.expect(); // Just consume the token
                                        parser.accept(',');
                                    }
                                    break;
                                }
                                case 'custom': {
                                    const fn = this._customParsers.get(elem.parser);
                                    if (!fn) {
                                        break;
                                    }
                                    const result = fn(parser, elem.args);
                                    if (result && result.kind === 'SameOperandsAndResultType' && result.type) {
                                        for (const operand of op.operands) {
                                            if (!operand.type) {
                                                operand.type = result.type;
                                            }
                                        }
                                        for (const res of op.results) {
                                            if (!res.type) {
                                                res.type = result.type;
                                            }
                                        }
                                    } else if (result && result.kind === 'PairwiseOpType') {
                                        if (result.operandTypes && op.operands.length > 0) {
                                            for (let i = 0; i < Math.min(result.operandTypes.length, op.operands.length); i++) {
                                                if (!op.operands[i].type) {
                                                    op.operands[i].type = result.operandTypes[i];
                                                }
                                            }
                                        }
                                        if (result.resultTypes && op.results.length > 0) {
                                            for (let i = 0; i < Math.min(result.resultTypes.length, op.results.length); i++) {
                                                if (!op.results[i].type) {
                                                    op.results[i].type = result.resultTypes[i];
                                                }
                                            }
                                        }
                                    }
                                    break;
                                }
                                default: {
                                    throw new mlir.Error(`Unsupported directive type '${elem.type}' ${parser.location()}.`);
                                }
                            }
                        }
                    }
                    // If shouldParse is false, we skip the entire group
                    break;
                }
                default: {
                    throw new mlir.Error(`Unsupported directive type '${directive.type}' ${parser.location()}.`);
                }
            }
        }
        return true;
    }

    // Custom Type Parsers - return type information

    _parseSameOperandsAndResultType(parser, args) {
        // Parse: single type that applies to all operands and results
        // Handles operations where all operands and results have the same type
        //
        // Args format: [type($lhs), type($rhs), ..., type($result)]
        // We parse ONE type and DON'T consume commas between args
        // The args tell us which operands/results to assign the type to, but we don't parse multiple types
        const type = parser.parseType();
        return { kind: 'SameOperandsAndResultType', type, args };
    }

    _parseVariadicSameOperandsAndResultType(parser, /*, args */) {
        // Like SameOperandsAndResultType but for variadic operands
        const type = parser.parseType();
        return { kind: 'VariadicSameOperandsAndResultType', type };
    }

    _parseComplexOpType(parser, /*, args */) {
        // Parse: complex_type or (type1, type2) -> complex_type
        // For complex operations where operands are real/imag components
        const type = parser.parseType();
        return { kind: 'ComplexOpType', type };
    }

    _parseSelectOpType(parser, /*, args */) {
        // Parse: pred_type, result_type OR (pred, on_true, on_false) -> result_type
        const firstType = parser.parseType();
        if (parser.accept(',')) {
            // Two types: pred_type, result_type
            const secondType = parser.parseType();
            return { kind: 'SelectOpType', predType: firstType, resultType: secondType };
        }
        // Function type
        return { kind: 'SelectOpType', functionType: firstType };
    }

    _parseTupleOpType(parser, /*, args */) {
        // Parse tuple operation types
        const type = parser.parseType();
        return { kind: 'TupleOpType', type };
    }

    _parsePairwiseOpType(parser /*, args */) {
        const operandTypes = [];
        const resultTypes = [];
        while (true) {
            const type = parser.parseType();
            if (!type) {
                break;
            }
            operandTypes.push(type);
            resultTypes.push(type);
            if (!parser.accept(',')) {
                break;
            }
        }

        return { kind: 'PairwiseOpType', operandTypes, resultTypes };
    }

    _parseConvolutionDimensions(parser, /*, args */) {
        // Parse: [b, f, 0, 1]x[0, 1, i, o]->[b, f, 0, 1]
        const result = {
            kind: 'ConvolutionDimensions',
            input: [],
            kernel: [],
            output: []
        };

        // Parse input dimensions: [b, f, 0, 1]
        if (parser.accept('[')) {
            while (!parser.match(']')) {
                if (parser.match('int') || parser.match('number')) {
                    result.input.push(parseInt(parser.expect(), 10));
                } else if (parser.match('id')) {
                    result.input.push(parser.expect('id'));
                } else {
                    // Unexpected token - break to avoid infinite loop
                    break;
                }
                parser.accept(',');
            }
            parser.accept(']');
        }

        // Parse 'x'
        if (parser.accept('id', 'x')) {
            // Parse kernel dimensions: [0, 1, i, o]
            if (parser.accept('[')) {
                while (!parser.match(']')) {
                    if (parser.match('int') || parser.match('number')) {
                        result.kernel.push(parseInt(parser.expect(), 10));
                    } else if (parser.match('id')) {
                        result.kernel.push(parser.expect('id'));
                    } else {
                        // Unexpected token - break to avoid infinite loop
                        break;
                    }
                    parser.accept(',');
                }
                parser.accept(']');
            }
        }

        // Parse '->'
        if (parser.accept('->')) {
            // Parse output dimensions: [b, f, 0, 1]
            if (parser.accept('[')) {
                while (!parser.match(']')) {
                    if (parser.match('int') || parser.match('number')) {
                        result.output.push(parseInt(parser.expect(), 10));
                    } else if (parser.match('id')) {
                        result.output.push(parser.expect('id'));
                    } else {
                        // Unexpected token - break to avoid infinite loop
                        break;
                    }
                    parser.accept(',');
                }
                parser.accept(']');
            }
        }

        return result;
    }

    _parseDotDimensionNumbers(parser /*, args */) {
        // Parse: contracting_dims = [0, 1] x [1, 0], batch_dims = [2] x [2]
        const result = {
            kind: 'DotDimensionNumbers',
            lhs_batching_dimensions: [],
            rhs_batching_dimensions: [],
            lhs_contracting_dimensions: [],
            rhs_contracting_dimensions: []
        };

        // Grammar: [batching_dims = [...] x [...] `,`] contracting_dims = [...] x [...]
        // Note: batching_dims MUST have trailing comma if present, contracting_dims must NOT

        const parsePair = () => {
            const first = [];
            const second = [];
            // Parse first array
            if (parser.accept('[')) {
                while (!parser.match(']')) {
                    if (parser.match('int')) {
                        first.push(parseInt(parser.expect('int'), 10));
                        parser.accept(',');
                    } else {
                        // Skip non-integer token and advance to avoid infinite loop
                        parser.expect();
                        parser.accept(',');
                    }
                }
                parser.accept(']');
            }
            // Parse 'x'
            if (parser.accept('id', 'x')) {
                // Parse second array
                if (parser.accept('[')) {
                    while (!parser.match(']')) {
                        if (parser.match('int')) {
                            second.push(parseInt(parser.expect('int'), 10));
                            parser.accept(',');
                        } else {
                            // Skip non-integer token and advance to avoid infinite loop
                            parser.expect();
                            parser.accept(',');
                        }
                    }
                    parser.accept(']');
                }
            }
            return { first, second };
        };

        // Optional: batching_dims = [...] x [...],
        if (parser.match('id', 'batching_dims') || parser.match('id', 'batch_dims')) {
            parser.expect('id');
            parser.accept('=');
            const pair = parsePair();
            result.lhs_batching_dimensions = pair.first;
            result.rhs_batching_dimensions = pair.second;
            parser.accept(','); // Required comma after batching_dims
        }

        // Required: contracting_dims = [...] x [...]
        if (parser.match('id', 'contracting_dims')) {
            parser.expect('id');
            parser.accept('=');
            const pair = parsePair();
            result.lhs_contracting_dimensions = pair.first;
            result.rhs_contracting_dimensions = pair.second;
            // Do NOT consume trailing comma - it belongs to the next parser
        }

        return result;
    }

    _parsePrecisionConfig(parser /*, args */) {
        // Parse: precision = [DEFAULT, DEFAULT]
        // Grammar: [`precision` `=` `[` precision_list `]`]
        // Note: This is for stablehlo.dot - simpler than PrecisionConfigAndAlgorithm
        const result = {
            kind: 'PrecisionConfig',
            precision: []
        };

        // Check if precision keyword is present
        if (!parser.match('id', 'precision')) {
            return result; // No precision config
        }

        parser.expect('id', 'precision');
        parser.expect('=');
        parser.expect('[');
        while (!parser.match(']')) {
            if (parser.match('id')) {
                result.precision.push(parser.expect('id'));
                parser.accept(',');
            } else {
                // Skip unexpected token to avoid infinite loop
                parser.expect();
                parser.accept(',');
            }
        }
        parser.expect(']');

        return result;
    }

    _parsePrecisionConfigAndAlgorithm(parser /*, args */) {
        // Parse: precision = [DEFAULT, DEFAULT], algorithm = {...}
        // Grammar: [`,` (precision | algorithm) [`,` algorithm]]
        const result = {
            kind: 'PrecisionConfigAndAlgorithm',
            precision: [],
            algorithm: null
        };

        // Optional leading comma
        if (!parser.accept(',')) {
            return result; // No comma means no precision/algorithm config
        }

        // Try parsing "algorithm = ..." first
        if (parser.match('id', 'algorithm')) {
            parser.expect('id');
            parser.accept('=');
            result.algorithm = parser.parseAttributeValue();
            return result;
        }

        // Parse "precision = [...]"
        if (parser.match('id', 'precision')) {
            parser.expect('id');
            parser.accept('=');
            if (parser.accept('[')) {
                while (!parser.match(']')) {
                    if (parser.match('id')) {
                        result.precision.push(parser.expect('id'));
                        parser.accept(',');
                    } else {
                        // Skip unexpected token and advance to avoid infinite loop
                        parser.expect();
                        parser.accept(',');
                    }
                }
                parser.accept(']');
            }

            // Optional ", algorithm = ..."
            if (parser.accept(',')) {
                if (parser.match('id', 'algorithm')) {
                    parser.expect('id');
                    parser.accept('=');
                    result.algorithm = parser.parseAttributeValue();
                }
            }
        }

        return result;
    }

    _parseWindowAttributes(parser, /*, args */) {
        // Parse: stride = [2, 2], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], etc.
        const result = {
            kind: 'WindowAttributes',
            stride: [],
            pad: [],
            lhs_dilate: [],
            rhs_dilate: [],
            window_reversal: []
        };

        while (!parser.match('}')) {
            if (parser.match('id')) {
                const key = parser.expect('id');
                if (parser.accept('=')) {
                    const parseArray = () => {
                        const arr = [];
                        if (parser.accept('[')) {
                            while (!parser.match(']')) {
                                if (parser.match('[')) {
                                    // Nested array like [[0, 0], [1, 1]]
                                    arr.push(parseArray());
                                } else if (parser.match('int') || parser.match('number')) {
                                    arr.push(parseInt(parser.expect(), 10));
                                } else if (parser.match('id')) {
                                    arr.push(parser.expect('id'));
                                } else {
                                    // Unexpected token - break to avoid infinite loop
                                    break;
                                }
                                parser.accept(',');
                            }
                            parser.accept(']');
                        }
                        return arr;
                    };

                    result[key] = parseArray();
                }
                parser.accept(',');
            } else {
                break;
            }
        }

        return result;
    }

    _parseSliceRanges(parser, /*, args */) {
        // Parse: [start:limit:stride, start:limit:stride, ...]
        const result = {
            kind: 'SliceRanges',
            start_indices: [],
            limit_indices: [],
            strides: []
        };

        if (parser.accept('[')) {
            while (!parser.match(']')) {
                // Parse start
                if (parser.match('int')) {
                    result.start_indices.push(parseInt(parser.expect('int'), 10));
                }
                parser.accept(':');
                // Parse limit
                if (parser.match('int')) {
                    result.limit_indices.push(parseInt(parser.expect('int'), 10));
                }
                // Parse stride (optional, defaults to 1)
                if (parser.accept(':')) {
                    if (parser.match('int')) {
                        result.strides.push(parseInt(parser.expect('int'), 10));
                    }
                } else {
                    result.strides.push(1);
                }
                parser.accept(',');
            }
            parser.accept(']');
        }

        return result;
    }

    _parseCustomCallTarget(parser, /*, args */) {
        // Parse: symbol_name (e.g., @my_custom_function)
        if (parser.match('@')) {
            const target = parser.expect('@');
            return { kind: 'CustomCallTarget', target };
        } else if (parser.match('string')) {
            const target = parser.expect('string');
            return { kind: 'CustomCallTarget', target };
        }
        throw new mlir.Error(`Expected '@' or string for CustomCallTarget at ${parser.location()}`);
    }

    _parseDynamicIndexList(parser, /*, args */) {
        // Parse: [index1, index2, ...] or empty
        // Used for vector.insert, vector.extract, etc.
        const result = {
            kind: 'DynamicIndexList',
            indices: []
        };

        if (parser.accept('[')) {
            while (!parser.match(']')) {
                if (parser.match('int') || parser.match('number')) {
                    result.indices.push(parseInt(parser.expect(), 10));
                } else if (parser.match('%')) {
                    // Dynamic index
                    result.indices.push(parser.expect('%'));
                } else {
                    break;
                }
                parser.accept(',');
            }
            parser.accept(']');
        }

        return result;
    }

    _parseVariadicOperandWithAttribute(parser, /*, args */) {
        // Parse variadic operands with optional attributes
        // Example: %operand1, %operand2 {attr = value}, %operand3
        const result = {
            kind: 'VariadicOperandWithAttribute',
            operands: []
        };

        while (parser.match('%') || parser.match('id')) {
            const operand = {
                value: null,
                attributes: []
            };

            if (parser.match('%')) {
                operand.value = parser.expect('%');
            } else {
                operand.value = parser.expect('id');
            }

            // Check for inline attributes
            if (parser.match('{')) {
                parser.parseAttributeDict(operand.attributes);
            }

            result.operands.push(operand);

            if (!parser.accept(',')) {
                break;
            }
        }

        return result;
    }

    _parseSymbolVisibility(parser /*, args */) {
        if (parser.match('id', 'private') || parser.match('id', 'public') || parser.match('id', 'nested')) {
            const visibility = parser.expect('id');
            return { kind: 'SymbolVisibility', visibility };
        }
        return { kind: 'SymbolVisibility', visibility: null };
    }

    _parseOptionalUnitAttr(parser /*, args */) {
        if (parser.match('id') || parser.match('%') || parser.match('(')) {
            return { kind: 'OptionalUnitAttr', present: true };
        }
        return { kind: 'OptionalUnitAttr', present: false };
    }
};

mlir.StableHLODialect = class extends mlir.Dialect {

    constructor(operations) {
        super('stablehlo', operations);
    }

    parseType(parser, dialectType) {
        if (dialectType === '!stablehlo.token') {
            return dialectType;
        }
        return null;
    }

    parseOperation(parser, opName, op) {
        const name = opName.replace(/^"|"$/g, '');
        if (name === 'stablehlo.constant') {
            if (parser.accept('(') && parser.accept(')')) {
                if (parser.match('{')) {
                    parser.parseAttributeDict(op.attributes);
                }
                if (parser.accept(':')) {
                    parser.parseArgumentTypes(op.operands);
                }
                if (parser.accept('->')) {
                    parser.parseArgumentTypes(op.results);
                }
            } else {
                if (parser.match('{')) {
                    parser.parseAttributeDict(op.attributes);
                }
                const value = parser.parseValue();
                if (value) {
                    op.attributes.push({ name: 'value', value: value.value });
                }
                // Parse result type: : tensor<...>
                if (parser.accept(':')) {
                    parser.parseArgumentTypes(op.results);
                }
            }
            return true;
        }
        if (name === 'stablehlo.concatenate') {
            op.operands = parser.parseArguments();
            if (parser.accept('id', 'dim')) {
                parser.accept('=');
                const dimValue = parser.parseAttributeValue();
                op.attributes.push({ name: 'dimension', value: dimValue });
            }
            if (parser.accept(':')) {
                parser.parseArgumentTypes(op.operands);
            }
            if (parser.accept('->')) {
                parser.parseArgumentTypes(op.results);
            }
            return true;
        }
        if (name === 'stablehlo.compare') {
            // Parse comparison predicate (EQ, NE, GE, GT, LE, LT)
            if (parser.match('id')) {
                const predicate = parser.expect('id');
                op.attributes.push({ name: 'comparison_direction', value: predicate });
            }
            parser.accept(',');
            op.operands = parser.parseArguments();
            parser.accept(',');
            if (parser.match('id')) {
                const compareType = parser.expect('id');
                op.attributes.push({ name: 'compare_type', value: compareType });
            }
            // Parse type signature
            if (parser.accept(':')) {
                parser.parseArgumentTypes(op.operands);
            }
            if (parser.accept('->')) {
                parser.parseArgumentTypes(op.results);
            }
            return true;
        }
        if (name === 'stablehlo.iota' || name === 'stablehlo.dynamic_iota') {
            if (name === 'stablehlo.dynamic_iota') {
                op.operands = parser.parseArguments();
                parser.accept(',');
            }
            if (parser.accept('id', 'dim')) {
                parser.accept('=');
                const dimValue = parser.parseAttributeValue();
                op.attributes.push({ name: 'iota_dimension', value: dimValue });
            }
            if (parser.accept(':')) {
                parser.parseArgumentTypes(op.operands);
            }
            if (parser.accept('->')) {
                parser.parseArgumentTypes(op.results);
            }
            return true;
        }
        if (name === 'stablehlo.dynamic_slice') {
            op.operands = parser.parseArguments();
            parser.accept(',');
            if (parser.accept('id', 'sizes')) {
                parser.accept('=');
                const sizes = parser.parseAttributeValue();
                op.attributes.push({ name: 'slice_sizes', value: sizes });
            }
            if (parser.accept(':')) {
                parser.parseArgumentTypes(op.operands);
            }
            if (parser.accept('->')) {
                parser.parseArgumentTypes(op.results);
            }
            return true;
        }
        if (name === 'stablehlo.get_dimension_size') {
            op.operands = parser.parseArguments();
            parser.accept(',');
            if (parser.accept('id', 'dim')) {
                parser.accept('=');
                const dimValue = parser.parseAttributeValue();
                op.attributes.push({ name: 'dimension', value: dimValue });
            }
            if (parser.accept(':')) {
                parser.parseArgumentTypes(op.operands);
            }
            if (parser.accept('->')) {
                parser.parseArgumentTypes(op.results);
            }
            return true;
        }
        if (name === 'stablehlo.reduce_precision') {
            op.operands = parser.parseArguments();
            parser.accept(',');
            if (parser.accept('id', 'format')) {
                parser.accept('=');
                const formatToken = parser.expect('id');
                const match = formatToken.match(/^e(\d+)m(\d+)$/);
                if (match) {
                    op.attributes.push({ name: 'exponent_bits', value: parseInt(match[1], 10) });
                    op.attributes.push({ name: 'mantissa_bits', value: parseInt(match[2], 10) });
                }
            }
            if (parser.accept(':')) {
                const type = parser.parseType();
                if (op.operands.length > 0) {
                    op.operands[0].type = type;
                }
                if (op.results.length > 0) {
                    op.results[0].type = type;
                }
            }
            return true;
        }
        if (name === 'stablehlo.while' && parser.match('(')) {
            parser.accept('(');
            while (!parser.match(')')) {
                const arg = {};
                arg.value = parser.expect('%');
                if (parser.accept('=')) {
                    arg.name = arg.value;
                    arg.value = parser.expect('%');
                }
                op.operands.push(arg);
                parser.accept(',');
            }
            parser.expect(')');
            if (parser.accept(':')) {
                let index = 0;
                while (!parser.match('id', 'cond') && !parser.match('id', 'attributes') && index < op.operands.length * 2) {
                    const type = parser.parseType();
                    if (index < op.operands.length) {
                        op.operands[index].type = type;
                    }
                    if (index < op.results.length) {
                        op.results[index].type = type;
                    }
                    index++;
                    if (!parser.accept(',')) {
                        break;
                    }
                }
            }
            if (parser.accept('id', 'attributes')) {
                if (parser.match('{')) {
                    parser.parseAttributeDict(op.attributes);
                }
            }
            if (parser.accept('id', 'cond')) {
                const condRegion = {};
                parser.parseRegion(condRegion);
                op.regions.push(condRegion);
            }
            if (parser.accept('id', 'do')) {
                const bodyRegion = {};
                parser.parseRegion(bodyRegion);
                op.regions.push(bodyRegion);
            }
            return true;
        }
        if (name === 'stablehlo.optimization_barrier') {
            if (parser.accept('(') && parser.accept(')')) {
                return true;
            }
            while (parser.match('%')) {
                const operand = { value: parser.expect('%') };
                op.operands.push(operand);
                if (!parser.accept(',')) {
                    break;
                }
            }
            if (parser.accept(':')) {
                let index = 0;
                while (!parser.match('keyword', 'loc') && !parser.match('}') && index < 100) {
                    const type = parser.parseType();
                    if (!type) {
                        break;
                    }
                    if (index < op.operands.length) {
                        op.operands[index].type = type;
                    }
                    if (index < op.results.length) {
                        op.results[index].type = type;
                    }
                    index++;
                    if (!parser.accept(',')) {
                        break;
                    }
                }
            }
            return true;
        }
        if ((name === 'stablehlo.reduce' || name === 'stablehlo.scan') && parser.match('(')) {
            return this._parseReduceLikeOp(parser, op);
        }
        return super.parseOperation(parser, opName, op);
    }

    _parseReduceLikeOp(parser, op) {
        op.operands = parser.parseArguments();

        while (parser.accept(',')) {
            const moreOperands = parser.parseArguments();
            op.operands.push(...moreOperands);
        }

        if (parser.accept('id', 'applies')) {
            const innerOpName = parser.expect('id');
            op.attributes.push({ name: 'body_op', value: innerOpName });

            if (parser.accept('id', 'across')) {
                if (parser.accept('id', 'dimensions')) {
                    parser.accept('=');
                    const dims = parser.parseAttributeValue();
                    op.attributes.push({ name: 'dimensions', value: dims });
                }
            }

            if (parser.match('{')) {
                parser.parseAttributeDict(op.attributes);
            }

            if (parser.accept(':')) {
                parser.parseArgumentTypes(op.operands);
            }

            if (parser.accept('->')) {
                parser.parseArgumentTypes(op.results);
            }

            return true;
        }

        if (parser.match('(')) {
            // Generic form with parenthesized region-list: ({ ... })
            if (parser.accept('(') && parser.match('{')) {
                let regionCount = 0;
                while (!parser.match(')')) {
                    if (regionCount++ > 10) {
                        throw new mlir.Error(`Too many regions in region-list (>10) - possible infinite loop at ${parser.location()}, current token: '${parser.token.value}'`);
                    }
                    if (!parser.match('{')) {
                        throw new mlir.Error(`Expected '{' for region in region-list, got '${parser.token.value}' at ${parser.location()}`);
                    }
                    const region = {};
                    parser.parseRegion(region);
                    op.regions.push(region);
                    if (!parser.accept(',') && !parser.match(')')) {
                        throw new mlir.Error(`Expected ',' or ')' after region, got '${parser.token.value}' at ${parser.location()}`);
                    }
                }
                parser.expect(')');
            }

            // Parse attributes dictionary { dimensions = ... }
            if (parser.match('{')) {
                parser.parseAttributeDict(op.attributes);
            }

            // Parse type signature : (...) -> (...)
            if (parser.accept(':')) {
                parser.parseArgumentTypes(op.operands);
            }

            if (parser.accept('->') || parser.accept('id', 'to')) {
                if (op.results.length > 0) {
                    parser.parseArgumentTypes(op.results);
                } else {
                    op.results = parser.parseArguments();
                }
            }

            return true;
        }

        // Handle "across dimensions = [...]"
        if (parser.accept('id', 'across')) {
            if (parser.accept('id', 'dimensions')) {
                parser.expect('=');
                parser.skip('[', ']');
            }
        }

        // Type signature
        if (parser.accept(':')) {
            parser.parseArgumentTypes(op.operands);
        }

        if (parser.accept('->') || parser.accept('id', 'to')) {
            if (op.results.length > 0) {
                parser.parseArgumentTypes(op.results);
            } else {
                op.results = parser.parseArguments();
            }
        }

        // Handle regions
        if (parser.match('id') && !parser.match('keyword', 'loc')) {
            // Labeled region: reducer(...) { ... } or reducer(...) (...) { ... }
            const label = parser.expect('id');
            const region = { blocks: [] };
            const block = { operations: [], arguments: [], name: label };

            while (parser.accept('(')) {
                while (!parser.accept(')')) {
                    const value = parser.expect('%');
                    parser.expect(':');
                    const type = parser.parseType();
                    block.arguments.push({ value, type });
                    parser.accept(',');
                }
            }

            parser.expect('{');
            while (!parser.accept('}')) {
                const innerOp = parser.parseOperation();
                block.operations.push(innerOp);
            }

            block.loc = parser.parseLocation();
            region.blocks.push(block);
            op.regions.push(region);
        } else if (parser.accept('(') && parser.match('{')) {
            // Parenthesized region list: ({ ... })
            const region = {};
            parser.parseRegion(region);
            op.regions.push(region);
            parser.expect(')');
        } else if (parser.match('{')) {
            // Simple region: { ... }
            const region = {};
            parser.parseRegion(region);
            op.regions.push(region);
        }

        return true;
    }
};

mlir.VhloDialect = class extends mlir.Dialect {

    constructor(operations) {
        super('vhlo', operations);
    }

    parseOperation(parser, opName, op) {
        const name = opName.replace(/^"|"$/g, '');

        if (name === 'vhlo.constant_v1') {
            if (parser.match('{')) {
                parser.parseAttributeDict(op.attributes);
            }
            const value = parser.parseValue();
            if (value) {
                op.attributes.push({ name: 'value', value: value.value });
            }
            // Parse result type: : tensor<...>
            if (parser.accept(':')) {
                parser.parseArgumentTypes(op.results);
            }
            return true;
        }

        if (name === 'vhlo.return_v1') {
            op.operands = parser.parseArguments();
            if (parser.accept(':')) {
                parser.parseArgumentTypes(op.operands);
            }
            return true;
        }

        if (name === 'vhlo.func_v1') {
            parser.parseOptionalVisibilityKeyword(op.attributes);
            parser.parseSymbolName(opName, op.attributes);
            if (parser.accept('(')) {
                op.attributes.push({ name: 'inputs', value: parser.parseTypeList() });
                parser.accept(')');
            }
            if (parser.accept('->')) {
                const outputType = parser.parseType();
                if (outputType) {
                    const outputs = outputType.value === 'tuple' ? outputType.params : [outputType];
                    op.attributes.push({ name: 'outputs', value: outputs });
                }
            }
            if (parser.match('{')) {
                parser.parseAttributeDict(op.attributes);
            }
            op.regions = [parser.parseRegion()];
            return true;
        }

        return false;
    }
};

mlir.InterpreterDialect = class extends mlir.Dialect {

    constructor(operations) {
        super('interpreter', operations);
    }
};

mlir.AffineDialect = class extends mlir.Dialect {

    constructor(operations) {
        super('affine', operations);
    }

    parseOperation(parser, opName, op) {
        const name = opName.replace(/^"|"$/g, '');
        // Special handling for affine.for - similar to scf.for but with affine expressions
        if (name === 'affine.for') {
            return this._parseForOp(parser, op);
        }
        // Special handling for affine.if - has condition before region
        if (name === 'affine.if') {
            // affine.if #set(...) { region }
            if (parser.match('#')) {
                const condition = parser.parseValue();
                op.attributes.push({ name: 'condition', value: condition });
            }
            const region = {};
            parser.parseRegion(region);
            op.regions.push(region);
            if (parser.match('id', 'else')) {
                parser.expect('id', 'else');
                const elseRegion = {};
                parser.parseRegion(elseRegion);
                op.regions.push(elseRegion);
            }
            return true;
        }
        // Special handling for affine.apply and affine.min
        if (name === 'affine.apply' || name === 'affine.min') {
            // affine.apply and affine.min have special syntax:
            // affine.apply #map(args) or affine.apply affine_map<...>()[args]
            // No type annotations, just the map application
            if (parser.match('#') || parser.match('id', 'affine_map') || parser.match('id', 'affine_set')) {
                const value = parser.parseValue();
                op.attributes.push({ name: 'map', value });
            }
            return true;
        }
        if (name === 'affine.store') {
            return this._parseStoreOp(parser, op);
        }
        if (name === 'affine.load') {
            return this._parseLoadOp(parser, op);
        }
        return super.parseOperation(parser, name, op);
    }

    _parseForOp(parser, op) {
        // affine.for %i = lowerBound to upperBound [step constant] { region }
        // Parse induction variable
        if (!parser.match('%')) {
            return false;
        }
        const inductionVar = parser.expect('%');
        // Parse '='
        if (!parser.accept('=')) {
            return false;
        }
        // Parse lower bound (can be constant, SSA value, or affine expression)
        this._parseAffineBound(parser, op, 'lowerBound');
        // Parse 'to' keyword
        if (!parser.accept('id', 'to')) {
            return false;
        }
        // Parse upper bound
        this._parseAffineBound(parser, op, 'upperBound');
        // Parse optional 'step' keyword and value
        if (parser.accept('id', 'step')) {
            if (parser.match('int')) {
                const step = parser.expect('int');
                op.attributes.push({ name: 'step', value: step });
            }
        }
        // Parse region
        if (parser.match('{')) {
            const region = {};
            parser.parseRegion(region);
            // Set the induction variable as the first block argument
            if (region.blocks && region.blocks.length > 0) {
                if (!region.blocks[0].arguments) {
                    region.blocks[0].arguments = [];
                }
                if (region.blocks[0].arguments.length > 0) {
                    region.blocks[0].arguments[0] = { value: inductionVar };
                } else {
                    region.blocks[0].arguments.push({ value: inductionVar });
                }
            }
            op.regions.push(region);
        }
        return true;
    }

    _parseAffineBound(parser, op, boundName) {
        // Parse affine bound: can be int, SSA value, or affine expression
        // Examples: 0, %N, min #map(%arg), max #map(), #map(%args)
        if (parser.match('int')) {
            const value = parser.expect('int');
            op.attributes.push({ name: boundName, value });
        } else if (parser.accept('id', 'min') || parser.accept('id', 'max')) {
            // min/max affine expression
            const expr = parser.parseValue();
            if (expr) {
                op.attributes.push({ name: boundName, value: expr });
            }
        } else if (parser.match('#')) {
            // Affine map reference
            const expr = parser.parseValue();
            if (expr) {
                op.attributes.push({ name: boundName, value: expr });
            }
        } else if (parser.match('%')) {
            op.operands.push({ value: parser.expect('%') });
        }
    }

    _parseStoreOp(parser, op) {
        if (parser.match('%')) {
            const value = parser.expect('%');
            op.operands.push({ value });
        } else {
            const value = parser.parseValue();
            op.operands.push(value);
        }
        if (!parser.accept('id', 'to')) {
            parser.accept(',');
        }
        const address = parser.expect('%');
        op.operands.push({ value: address });
        parser.skip('[', ']');
        if (parser.accept(':')) {
            const type = parser.parseType();
            op.operands[1].type = type;
        }
        return true;
    }

    _parseLoadOp(parser, op) {
        const address = parser.expect('%');
        op.operands.push({ value: address });
        parser.skip('[', ']');
        if (parser.accept(':')) {
            const type = parser.parseType();
            op.operands[0].type = type;
        }
        return true;
    }
};

mlir.MemRefDialect = class extends mlir.Dialect {

    constructor(operations) {
        super('memref', operations);
    }

    parseOperation(parser, opName, op) {
        const name = opName.replace(/^"|"$/g, '');
        if (name === 'memref.tensor_load') {
            op.operands = parser.parseArguments();
            if (parser.accept(':')) {
                parser.parseArgumentTypes(op.operands);
            }
            return true;
        }
        if (name === 'memref.store') {
            return this._parseStoreOp(parser, op);
        }
        if (name === 'memref.load') {
            return this._parseLoadOp(parser, op);
        }
        return super.parseOperation(parser, opName, op);
    }

    _parseStoreOp(parser, op) {
        if (parser.match('%')) {
            const value = parser.expect('%');
            op.operands.push({ value });
        } else {
            const value = parser.parseValue();
            op.operands.push(value);
        }
        if (!parser.accept('id', 'to')) {
            parser.accept(',');
        }
        const address = parser.expect('%');
        op.operands.push({ value: address });
        parser.skip('[', ']');
        if (parser.accept(':')) {
            const type = parser.parseType();
            op.operands[1].type = type;
        }
        return true;
    }

    _parseLoadOp(parser, op) {
        const address = parser.expect('%');
        op.operands.push({ value: address });
        parser.skip('[', ']');
        if (parser.accept(':')) {
            const type = parser.parseType();
            op.operands[0].type = type;
        }
        return true;
    }
};

mlir.VectorDialect = class extends mlir.Dialect {

    constructor(operations) {
        super('vector', operations);
    }

    parseOperation(parser, opName, op) {
        const name = opName.replace(/^"|"$/g, '');
        if (name === 'vector.splat') {
            op.operands = parser.parseArguments();
            if (parser.accept(':')) {
                parser.parseArgumentTypes(op.results);
            }
            return true;
        }
        if (name === 'vector.transfer_read' || name === 'vector.transfer_write') {
            return this._parseTransferOp(parser, op);
        }
        // Handle old vector.extract syntax (pre-2023) without 'from' keyword
        // Old: %r = vector.extract %v[0] : vector<4xf32>
        // New: %r = vector.extract %v[0] : f32 from vector<4xf32>
        if (name === 'vector.extract' && !op.isGeneric) {
            return this._parseExtractOp(parser, op);
        }
        return super.parseOperation(parser, opName, op);
    }

    _parseExtractOp(parser, op) {
        // Parse: %source [ indices ] : result_type [from source_type]
        // The 'from' keyword indicates new syntax (current MLIR)
        // Old syntax (pre-2023): %r = vector.extract %v[0] : vector<4xf32>
        // New syntax: %r = vector.extract %v[0] : f32 from vector<4xf32>

        // Parse source operand
        const source = parser.expect('%');
        op.operands.push({ value: source });

        // Parse indices: [0, 1, ...]
        if (parser.accept('[')) {
            while (!parser.match(']')) {
                if (parser.match('int') || parser.match('number')) {
                    parser.expect(); // Consume index but don't store (indices are in static_position attribute)
                } else if (parser.match('%')) {
                    const dynIndex = parser.expect('%');
                    op.operands.push({ value: dynIndex }); // Dynamic indices are operands
                } else {
                    break;
                }
                parser.accept(',');
            }
            parser.accept(']');
        }

        // Parse optional attributes
        if (parser.match('{')) {
            parser.parseAttributeDict(op.attributes);
        }

        // Parse type signature: : result_type [from source_type]
        if (parser.accept(':')) {
            const resultType = parser.parseType();

            // Check for 'from' keyword (new syntax)
            if (parser.accept('id', 'from')) {
                const sourceType = parser.parseType();
                op.operands[0].type = sourceType;
                op.results.push({ type: resultType });
            } else {
                // Old syntax: the type after ':' is the source type
                // Result type is extracted element type (scalar or sub-vector)
                op.operands[0].type = resultType;
                // We don't set result type - let the default inference handle it
            }
        }

        return true;
    }

    _parseTransferOp(parser, op) {
        // Parse: vector.transfer_read %source[%i, %j, ...], %padding {attrs} : memref_type, vector_type
        //    or: vector.transfer_write %value, %dest[%i, %j, ...] {attrs} : vector_type, memref_type

        // First operand: source/value
        const first = parser.expect('%');
        op.operands.push({ value: first });

        // Check if indices follow first operand or second operand
        const hasIndicesAfterFirst = parser.match('[');
        if (hasIndicesAfterFirst) {
            parser.skip('[', ']');
        }

        // Comma
        parser.accept(',');

        // Second operand: padding value or destination
        const second = parser.expect('%');
        op.operands.push({ value: second });

        // If indices didn't follow first operand, they follow second operand
        if (!hasIndicesAfterFirst && parser.match('[')) {
            parser.skip('[', ']');
        }

        // Optional attribute dictionary
        if (parser.match('{')) {
            parser.parseAttributeDict(op.attributes);
        }

        // Type signature: : memref_type, vector_type
        if (parser.accept(':')) {
            const type1 = parser.parseType();
            op.operands[0].type = type1;
            parser.accept(',');
            const type2 = parser.parseType();
            // For transfer_read, type2 is the result type
            // For transfer_write, type2 is just the vector type
            if (op.results.length > 0) {
                op.results[0].type = type2;
            }
            op.operands[1].type = type2;
        }

        return true;
    }
};

mlir.TorchDialect = class extends mlir.Dialect {

    constructor(operations) {
        super('torch', operations);
    }

    parseType(parser, dialectType) {
        const simpleTypes = [
            'int', 'float', 'bool', 'str', 'none', 'Device', 'Generator',
            'qint8', 'quint8', 'qint16', 'qint32', 'quint4x2', 'quint2x4',
            'LinearParams', 'number', 'any'
        ];
        for (const simpleType of simpleTypes) {
            if (dialectType === `!torch.${simpleType}`) {
                return dialectType;
            }
        }
        if (dialectType.startsWith('!torch.nn.Module')) {
            if (parser.match('<')) {
                const content = parser.skip('<', '>');
                return dialectType + content;
            }
            return dialectType;
        }
        if (dialectType.startsWith('!torch.vtensor') || dialectType.startsWith('!torch.tensor') ||
            dialectType.startsWith('!torch.list') || dialectType.startsWith('!torch.tuple') ||
            dialectType.startsWith('!torch.union') || dialectType.startsWith('!torch.optional') ||
            dialectType.startsWith('!torch.dict')) {
            if (parser.match('<')) {
                const content = parser.skip('<', '>');
                return dialectType + content;
            }
            return dialectType;
        }
        return null;
    }

    parseOperation(parser, opName, op) {
        const name = opName.replace(/^"|"$/g, '');
        if (name === 'torch.constant.int') {
            if (parser.match('int')) {
                const value = parser.expect('int');
                op.attributes.push({ name: 'value', value });
            }
            return true;
        }

        if (name === 'torch.operator') {
            if (parser.match('string')) {
                const operatorName = parser.expect('string');
                op.attributes.push({ name: 'name', value: operatorName });
            }
            parser.parseGenericOperationAfterOpName(op);
            return true;
        }

        if (name.startsWith('torch.aten.') || name.startsWith('torch.prim.')) {
            parser.parseGenericOperationAfterOpName(op);
            return true;
        }

        return super.parseOperation(parser, opName, op);
    }
};

mlir.HALDialect = class extends mlir.Dialect {

    constructor(operations) {
        super('hal', operations);
    }

    parseType(parser, dialectType) {
        const simpleTypes = [
            'allocator', 'buffer', 'buffer_view', 'channel', 'command_buffer',
            'device', 'event', 'executable', 'fence', 'file', 'semaphore'
        ];
        for (const simpleType of simpleTypes) {
            if (dialectType === `!hal.${simpleType}`) {
                return dialectType;
            }
        }
        return null;
    }

    parseOperation(parser, opName, op) {
        const name = opName.replace(/^"|"$/g, '');

        if (name === 'hal.tensor.cast') {
            op.operands = parser.parseArguments();
            if (parser.accept(':')) {
                const type = parser.parseType();
                if (op.operands.length > 0) {
                    op.operands[0].type = type;
                }
            }
            if (parser.accept('->')) {
                parser.parseArgumentTypes(op.results);
            }
            return true;
        }

        if (name === 'hal.constant') {
            parser.parseOptionalAttrDictWithKeyword(op.attributes);
            const value = parser.parseValue();
            op.attributes.push({ name: 'value', value: value.value === undefined ? value : value.value });
            if (parser.accept(':')) {
                parser.parseArgumentTypes(op.results);
            }
            return true;
        }

        // Handle hal.device.switch with attribute-based case matching FIRST (before generic hal.device handler)
        // Format: hal.device.switch<%device : !hal.device> #hal.device.match.xxx<...> { ... }
        if (name === 'hal.device.switch') {
            // Parse <%operand : type> if present
            if (parser.accept('<')) {
                while (!parser.accept('>')) {
                    const operand = parser.expect('%');
                    op.operands.push({ value: operand });
                    if (parser.accept(':')) {
                        const type = parser.parseType();
                        if (op.operands.length > 0) {
                            op.operands[op.operands.length - 1].type = type;
                        }
                    }
                    parser.accept(',');
                }
            }
            // Parse result type if present (-> type or : type)
            if (parser.accept('->') || parser.accept(':')) {
                const resultType = parser.parseType();
                op.results = [{ type: resultType }];
            }
            // Parse case regions: #attribute { region }, #attribute { region }, ...
            while (parser.match('#')) {
                const region = {};
                // Parse the case attribute
                const caseAttr = parser.parseAttributeValue();
                region.caseAttribute = caseAttr;
                // Parse the region
                if (parser.match('{')) {
                    parser.parseRegion(region);
                }
                op.regions.push(region);
                // Consume optional comma between cases
                parser.accept(',');
            }
            return true;
        }
        // Handle hal.executable.create with both old (layouts) and new (affinity) syntax
        if (name === 'hal.executable.create') {
            // Parse named parameters: device(...), target(...), and either layouts(...) or affinity(...)
            while (parser.match('id') && !parser.match(':') && !parser.match('loc')) {
                const paramName = parser.expect('id');
                if (parser.accept('(')) {
                    let parenDepth = 1;
                    let paramValue = '';
                    while (parenDepth > 0 && !parser.match('eof')) {
                        if (parser.match('(')) {
                            parenDepth++;
                            paramValue += parser.expect();
                        } else if (parser.match(')')) {
                            parenDepth--;
                            if (parenDepth > 0) {
                                paramValue += parser.expect();
                            } else {
                                parser.expect(')');
                            }
                        } else {
                            paramValue += parser.expect();
                        }
                    }
                    // Normalize old 'layouts' parameter to 'affinity' for consistency
                    const normalizedName = paramName === 'layouts' ? 'affinity' : paramName;
                    op.attributes.push({ name: normalizedName, value: paramValue });
                } else {
                    break;
                }
            }
            // Parse result type
            if (parser.accept(':')) {
                parser.parseArgumentTypes(op.results);
            }
            return true;
        }
        // Handle operations with <%operand : type> syntax and/or named parameters
        // e.g., hal.allocator.compute_size<%allocator : !hal.allocator> shape([...]) type(...) encoding(...) : index
        // or hal.executable_layout.lookup device(%device : !hal.device) layouts([[...]]) : !hal.executable_layout
        // Exclude hal.executable, hal.interface, and hal.device.switch which have special handling
        if ((name.startsWith('hal.allocator.') || name.startsWith('hal.buffer') ||
             name.startsWith('hal.command_buffer') || name.startsWith('hal.executable_layout') ||
             name.startsWith('hal.executable.') || name.startsWith('hal.descriptor_set_layout') ||
             name.startsWith('hal.device')) &&
            name !== 'hal.executable' && name !== 'hal.interface' && name !== 'hal.device.switch' &&
            name !== 'hal.executable.variant' && name !== 'hal.executable.entry_point' && name !== 'hal.interface.binding' &&
            name !== 'hal.executable.create') {
            // Parse <%operand : type> if present
            if (parser.accept('<')) {
                while (!parser.accept('>')) {
                    const operand = parser.expect('%');
                    op.operands.push({ value: operand });
                    if (parser.accept(':')) {
                        const type = parser.parseType();
                        if (op.operands.length > 0) {
                            op.operands[op.operands.length - 1].type = type;
                        }
                    }
                    parser.accept(',');
                }
            }
            // Parse named parameters like shape([...]) type(...) encoding(...)
            // Also handle bracket expressions between parameters like layout(...)[%c0]
            // Stop when we hit a colon (result type) or something that doesn't look like a parameter
            // Named parameters don't have dots, so if we see an id with a dot, it's likely the next operation
            while (parser.match('[') ||
                   (parser.match('id') && !parser.match('id', 'attributes') &&
                    !parser.match(':') && !parser.match('loc') &&
                    parser.token.value && parser.token.value.indexOf('.') === -1)) {
                // Handle bracket expressions (e.g., [%c0])
                if (parser.match('[')) {
                    parser.skip('[', ']');
                    continue;
                }
                // Try to parse a named parameter (id followed by '(')
                const paramName = parser.expect('id');
                if (parser.accept('(')) {
                    // This is a named parameter, parse the value
                    // Track depth separately for () and []
                    let parenDepth = 1;  // We've already consumed the opening (
                    let paramValue = '';
                    while (parenDepth > 0 && !parser.match('eof')) {
                        if (parser.match('(')) {
                            parenDepth++;
                            paramValue += parser.expect();
                        } else if (parser.match(')')) {
                            parenDepth--;
                            if (parenDepth > 0) {
                                paramValue += parser.expect();
                            } else {
                                // This is the closing ), consume it
                                parser.expect(')');
                            }
                        } else {
                            paramValue += parser.expect();
                        }
                    }
                    op.attributes.push({ name: paramName, value: paramValue });
                } else {
                    // Not a named parameter - we've consumed an id token that doesn't belong to us
                    // This shouldn't happen with proper MLIR, but break gracefully
                    break;
                }
            }
            // Parse result types if present (: type1, type2, ...)
            if (parser.accept(':')) {
                parser.parseArgumentTypes(op.results);
            }
            // Parse optional = <value> (default value or attribute)
            if (parser.accept('=')) {
                const value = parser.parseValue();
                op.attributes.push({ name: 'default', value: value.value });
            }
            return true;
        }
        // Handle operations with visibility + symbol (similar to flow dialect)
        if (name === 'hal.executable' || name === 'hal.interface') {
            if (parser.match('id', 'private') || parser.match('id', 'public') || parser.match('id', 'nested')) {
                parser.expect('id');
            }
            if (parser.match('@')) {
                const symbol = parser.expect('@');
                op.attributes.push({ name: 'sym_name', value: symbol });
            }
            if (parser.match('id', 'attributes')) {
                parser.expect('id', 'attributes');
                parser.parseAttributeDict(op.attributes);
            }
            if (parser.match('{')) {
                const region = {};
                parser.parseRegion(region);
                op.regions.push(region);
            }
            return true;
        }
        // Handle hal.interface.binding.subspan with old syntax (symbol reference)
        // Old syntax: hal.interface.binding.subspan @io::@binding[operand] : type
        // New syntax: hal.interface.binding.subspan layout(...) binding(...) : type
        if (name === 'hal.interface.binding.subspan' && parser.match('@')) {
            // Old syntax - parse symbol reference and bracket expression
            const symbolRef = parser.expect('@');
            op.attributes.push({ name: 'layout', value: symbolRef });
            // Parse optional bracket expression [operand]
            if (parser.accept('[')) {
                while (!parser.accept(']')) {
                    if (parser.match('%')) {
                        const operand = parser.expect('%');
                        op.operands.push({ value: operand });
                    } else {
                        parser.expect();
                    }
                    parser.accept(',');
                }
            }
            // Parse result type
            if (parser.accept(':')) {
                parser.parseArgumentTypes(op.results);
            }
            return true;
        }
        // Handle operations with named parameters: hal.interface.binding, hal.executable.variant, etc.
        if (name === 'hal.interface.binding' || name === 'hal.executable.variant' || name === 'hal.executable.entry_point') {
            if (parser.match('id', 'private') || parser.match('id', 'public') || parser.match('id', 'nested')) {
                parser.expect('id');
            }
            if (parser.match('@')) {
                const symbol = parser.expect('@');
                op.attributes.push({ name: 'sym_name', value: symbol });
            }
            // Parse comma-separated named parameters
            while (parser.accept(',')) {
                if (parser.match('id')) {
                    parser.expect('id');
                    if (parser.accept('=')) {
                        // Skip the value - could be complex attribute
                        if (parser.match('#')) {
                            parser.parseValue();
                        } else {
                            parser.expect();
                        }
                    }
                }
            }
            // Parse attributes dict if present
            if (parser.match('id', 'attributes')) {
                parser.expect('id', 'attributes');
                parser.parseAttributeDict(op.attributes);
            }
            // Parse region if present
            if (parser.match('{')) {
                const region = {};
                parser.parseRegion(region);
                op.regions.push(region);
            }
            return true;
        }
        return super.parseOperation(parser, opName, op);
    }
};

mlir.UtilDialect = class extends mlir.Dialect {

    constructor(operations) {
        super('util', operations);
    }

    parseType(parser, dialectType) {
        const simpleTypes = ['buffer', 'list', 'object'];
        for (const simpleType of simpleTypes) {
            if (dialectType === `!util.${simpleType}`) {
                return dialectType;
            }
        }
        if (dialectType.startsWith('!util.ptr')) {
            if (parser.match('<')) {
                const content = parser.skip('<', '>');
                return dialectType + content;
            }
            return dialectType;
        }
        return null;
    }

    parseOperation(parser, opName, op) {
        const name = opName.replace(/^"|"$/g, '');
        // Handle util.global with visibility and symbol
        if (name === 'util.global') {
            if (parser.match('id', 'private') || parser.match('id', 'public') || parser.match('id', 'nested')) {
                parser.expect('id');
            }
            if (parser.match('@')) {
                const symbol = parser.expect('@');
                op.attributes.push({ name: 'sym_name', value: symbol });
            }
            if (parser.accept(':')) {
                const type = parser.parseType();
                op.results = [{ type }];
            }
            return true;
        }
        // Handle util.initializer with region
        if (name === 'util.initializer') {
            if (parser.match('{')) {
                const region = {};
                parser.parseRegion(region);
                op.regions.push(region);
            }
            return true;
        }
        // Handle util.unreachable with optional message
        // assemblyFormat: ($message^)? attr-dict
        if (name === 'util.unreachable') {
            // Parse optional message string
            if (parser.match('string')) {
                const message = parser.expect('string');
                op.attributes.push({ name: 'message', value: message });
            }
            // Parse attr-dict if present
            if (parser.match('{')) {
                parser.parseAttributeDict(op.attributes);
            }
            return true;
        }
        if (name === 'util.func') {
            return this._parseFuncOp(parser, op);
        }
        return super.parseOperation(parser, opName, op);
    }

    _parseFuncOp(parser, op) {
        parser.parseOptionalVisibilityKeyword(op.attributes);
        parser.parseSymbolName('sym_name', op.attributes);
        const type = {};
        type.inputs = parser.parseFunctionArgumentList();
        parser.parseOptionalAttrDictWithKeyword(op.attributes);
        type.results = [];
        if (parser.accept('->')) {
            for (const result of parser.parseFunctionResultList()) {
                type.results.push(result);
            }
        }
        op.attributes.push({ name: 'function_type', value: type });
        parser.parseOptionalAttrDictWithKeyword(op.attributes);
        if (parser.match('{')) {
            const region = {};
            parser.parseRegion(region);
            op.regions.push(region);
        }
        if (op.regions.length > 0) {
            const region = op.regions[op.regions.length - 1];
            if (region.blocks.length > 0) {
                const block = region.blocks[region.blocks.length - 1];
                if (block.operations.length > 0) {
                    const lastOp = block.operations[block.operations.length - 1];
                    type.results = lastOp.operands;
                    block.operations.pop();
                }
            }
        }
        return true;
    }
};

mlir.FlowDialect = class extends mlir.Dialect {

    constructor(operations) {
        super('flow', operations);
    }

    parseType(parser, dialectType) {
        if (dialectType === '!flow.channel') {
            return dialectType;
        }
        if (dialectType.startsWith('!flow.dispatch.tensor')) {
            if (parser.match('<')) {
                const content = parser.skip('<', '>');
                return dialectType + content;
            }
            return dialectType;
        }
        return null;
    }

    parseOperation(parser, opName, op) {
        const name = opName.replace(/^"|"$/g, '');
        // Handle operations with custom syntax not in schema or using complex custom parsers
        if ((name === 'flow.dispatch.workgroups' && parser.match('[')) || name === 'flow.ex.stream.fragment') {
            return this._parseDispatchWorkgroupsOp(parser, op);
        }
        if (name === 'flow.dispatch.tensor.load' || name === 'flow.dispatch.tensor.store') {
            return this._parseTensorLoadStoreOp(parser, op);
        }
        // flow.dispatch has complex symbol references not handled by default parser
        if (name === 'flow.dispatch') {
            return this._parseDispatchOp(parser, op);
        }
        // Handle operations with visibility + symbol that aren't in schema or need manual parsing
        if (name === 'flow.executable' || name === 'flow.dispatch.entry') {
            if (parser.match('id', 'private') || parser.match('id', 'public') || parser.match('id', 'nested')) {
                parser.expect('id');
            }
            if (parser.match('@')) {
                const symbol = parser.expect('@');
                op.attributes.push({ name: 'sym_name', value: symbol });
            }
            if (parser.match('id', 'attributes')) {
                parser.expect('id', 'attributes');
                parser.parseAttributeDict(op.attributes);
            }
            if (parser.match('{')) {
                const region = {};
                parser.parseRegion(region);
                op.regions.push(region);
            }
            return true;
        }
        if (name === 'flow.func') {
            return this._parseFuncOp(parser, op);
        }
        return super.parseOperation(parser, opName, op);
    }

    _parseFuncOp(parser, op) {
        parser.parseOptionalVisibilityKeyword(op.attributes);
        parser.parseSymbolName('sym_name', op.attributes);
        const type = {};
        type.inputs = parser.parseFunctionArgumentList();
        parser.parseOptionalAttrDictWithKeyword(op.attributes);
        type.results = [];
        if (parser.accept('->')) {
            for (const result of parser.parseFunctionResultList()) {
                type.results.push(result);
            }
        }
        op.attributes.push({ name: 'function_type', value: type });
        parser.parseOptionalAttrDictWithKeyword(op.attributes);
        if (parser.match('{')) {
            const region = {};
            parser.parseRegion(region);
            op.regions.push(region);
        }
        if (op.regions.length > 0) {
            const region = op.regions[op.regions.length - 1];
            if (region.blocks.length > 0) {
                const block = region.blocks[region.blocks.length - 1];
                if (block.operations.length > 0) {
                    const lastOp = block.operations[block.operations.length - 1];
                    type.results = lastOp.operands;
                    block.operations.pop();
                }
            }
        }
        return true;
    }

    _parseDispatchWorkgroupsOp(parser, op) {
        // Parse subscript values: [%c32, %c112, %c112]
        if (parser.accept('[')) {
            while (!parser.accept(']')) {
                parser.expect(); // read subscript value
                parser.accept(',');
            }
        }
        // Parse operands: (%0, %1)
        op.operands = parser.parseArguments();
        // Parse type signature : (...) -> (...)
        if (parser.accept(':')) {
            parser.parseArgumentTypes(op.operands);
        }
        if (parser.accept('->')) {
            if (op.results.length > 0) {
                parser.parseArgumentTypes(op.results);
            } else {
                op.results = parser.parseArguments();
            }
        }
        // Parse region with arguments: = (%arg2: type, %arg3: type) { ... }
        if (parser.accept('=')) {
            const region = {};
            region.blocks = [];
            const block = {};
            block.operations = [];
            block.arguments = [];
            // Parse region arguments
            if (parser.accept('(')) {
                while (!parser.accept(')')) {
                    const value = parser.expect('%');
                    parser.expect(':');
                    const type = parser.parseType();
                    block.arguments.push({ value, type });
                    parser.accept(',');
                }
            }
            // Some operations like flow.ex.stream.fragment have -> type after region args
            if (parser.accept('->') || parser.accept('id', 'to')) {
                parser.parseType();
            }
            // Parse region body
            parser.parseBlock(block);
            region.blocks.push(block);
            op.regions.push(region);
        }
        return true;
    }

    _parseDispatchOp(parser, op) {
        // flow.dispatch @symbol::@entry[subscripts](operands) : types -> type
        if (parser.match('@')) {
            const symbol = parser.expect('@');
            op.attributes.push({ name: 'entry_point', value: symbol });
            // Handle :: nested symbol
            if (parser.accept('id', '::') || (parser.match(':') && parser.accept(':') && parser.accept(':'))) {
                if (parser.match('@')) {
                    const nested = parser.expect('@');
                    op.attributes[op.attributes.length - 1].value += `::${nested}`;
                }
            }
        }
        // Parse subscripts [...]
        if (parser.accept('[')) {
            while (!parser.accept(']')) {
                parser.expect();
                parser.accept(',');
            }
        }
        // Parse operands
        op.operands = parser.parseArguments();
        // Parse optional attribute dictionary before type signature
        if (parser.match('{')) {
            parser.parseAttributeDict(op.attributes);
        }
        // Parse type signature
        if (parser.accept(':')) {
            parser.parseArgumentTypes(op.operands);
        }
        if (parser.accept('->') || parser.accept('id', 'to')) {
            if (op.results.length > 0) {
                parser.parseArgumentTypes(op.results);
            } else {
                op.results = parser.parseArguments();
            }
        }
        return true;
    }

    _parseTensorLoadStoreOp(parser, op) {
        // Parse: load %arg2, offsets = [...] : type -> type
        //    or: store %26, %arg4, offsets = [...] : type -> type
        // Parse operands: one or more % values separated by commas
        while (parser.match('%')) {
            const value = parser.expect('%');
            op.operands.push({ value });
            // If next is not a comma, break
            if (!parser.accept(',')) {
                break;
            }
            // If after comma, next is not %, break (we've hit named parameters)
            if (!parser.match('%')) {
                // We have a comma followed by non-%, so continue to named parameters
                break;
            }
        }
        // At this point, if we broke because of named params, we've already consumed the comma
        // Parse comma-separated named parameters: offsets = [...], sizes = [...], strides = [...]
        // Note: first parameter might not need comma-eating if we just broke from operand loop
        let needComma = !parser.match('id'); // If we're not at 'id', we need to eat commas
        while (needComma ? parser.accept(',') : true) {
            needComma = true; // After first iteration, always need comma
            if (parser.match('id')) {
                const paramName = parser.expect('id');
                if (parser.accept('=')) {
                    // Skip the parameter value (usually an array)
                    if (parser.match('[')) {
                        parser.skip('[', ']');
                    } else {
                        parser.expect(); // Read single value
                    }
                    op.attributes.push({ name: paramName, value: paramName });
                }
            } else {
                break;
            }
        }
        // Parse type signature : type -> type
        if (parser.accept(':')) {
            parser.parseArgumentTypes(op.operands);
        }
        // For tensor.load, there's a -> result type
        // For tensor.store, the -> is followed by the output tensor type (not a result)
        if (parser.accept('->') || parser.accept('id', 'to')) {
            // Just skip the type - we don't need to parse it as results
            parser.parseType();
        }
        return true;
    }
};

mlir.StreamDialect = class extends mlir.Dialect {

    constructor(operations) {
        super('stream', operations);
    }

    parseType(parser, dialectType) {
        const simpleTypes = ['binding', 'timepoint', 'file'];
        for (const simpleType of simpleTypes) {
            if (dialectType === `!stream.${simpleType}`) {
                return dialectType;
            }
        }
        if (dialectType.startsWith('!stream.resource')) {
            if (parser.match('<')) {
                const content = parser.skip('<', '>');
                return dialectType + content;
            }
            return dialectType;
        }
        return null;
    }
};

mlir.IREETensorExtDialect = class extends mlir.Dialect {

    constructor(operations) {
        super('iree_tensor_ext', operations);
    }

    parseType(parser, dialectType) {
        if (dialectType.startsWith('!iree_tensor_ext.dispatch.tensor')) {
            if (parser.match('<')) {
                const content = parser.skip('<', '>');
                return dialectType + content;
            }
            return dialectType;
        }
        return null;
    }
};

mlir.LinalgDialect = class extends mlir.Dialect {

    constructor(operations) {
        super('linalg', operations);
    }

    parseOperation(parser, opName, op) {
        const name = opName.replace(/^"|"$/g, '');

        if (name === 'linalg.generic') {
            return this._parseGenericOp(parser, op);
        }

        if (name === 'linalg.init_tensor') {
            if (parser.accept('[')) {
                const dims = [];
                while (!parser.match(']')) {
                    if (parser.match('%')) {
                        dims.push(parser.expect('%'));
                    } else if (parser.match('int')) {
                        dims.push(parser.expect('int'));
                    }
                    parser.accept(',');
                }
                parser.expect(']');
                op.attributes.push({ name: 'static_sizes', value: dims });
            }
            if (parser.accept(':')) {
                parser.parseArgumentTypes(op.results);
            }
            return true;
        }

        if (name === 'linalg.fill') {
            if (parser.match('id', 'ins') || parser.match('{')) {
                return this._parseInsOutsOp(parser, op);
            }
            if (parser.accept('(')) {
                op.operands = parser.parseArguments();
                parser.expect(')');
            }
            if (parser.match('{')) {
                parser.parseAttributeDict(op.attributes);
            }
            if (parser.accept(':')) {
                parser.parseArgumentTypes(op.operands);
            }
            if (parser.accept('->')) {
                parser.parseArgumentTypes(op.results);
            }
            return true;
        }

        if (name === 'linalg.conv') {
            if (parser.accept('(')) {
                op.operands = parser.parseArguments();
                parser.expect(')');
            }
            if (parser.match('{')) {
                parser.parseAttributeDict(op.attributes);
            }
            if (parser.accept(':')) {
                parser.parseArgumentTypes(op.operands);
            }
            return true;
        }

        if (name === 'linalg.yield') {
            op.operands = parser.parseArguments();
            if (parser.accept(':')) {
                parser.parseArgumentTypes(op.operands);
            }
            return true;
        }

        // Many linalg operations (especially named structured ops) follow the pattern:
        // linalg.<op_name> {attributes} ins(...) outs(...) [-> type]
        // These named structured ops are generated at LLVM build time from YAML
        // and won't be in our schema, so we need custom parsing for them.

        // First try default parsing (for operations in schema)
        const hasSchemaEntry = this._operations.has(name);
        if (hasSchemaEntry) {
            return super.parseOperation(parser, opName, op);
        }

        // Not in schema - try to parse as ins/outs operation
        // These are named structured ops generated from LLVM build time from YAML
        if (parser.match('{') || parser.match('id', 'ins')) {
            return this._parseInsOutsOp(parser, op);
        }

        return false;
    }

    _parseInsOutsOp(parser, op) {
        // Parse: linalg.op {attrs} ins(%0, %1 : type, type) outs(%2 : type) [-> type]

        // Parse optional attribute dictionary
        if (parser.match('{')) {
            parser.parseAttributeDict(op.attributes);
        }

        // Parse 'ins' section
        if (parser.accept('id', 'ins')) {
            if (!parser.accept('(')) {
                return false;
            }
            // Parse operands: %0, %1
            while (parser.match('%')) {
                const value = parser.expect('%');
                op.operands.push({ value });
                if (!parser.accept(',')) {
                    break;
                }
            }
            // Parse types: : type1, type2
            if (parser.accept(':')) {
                let idx = 0;
                const startIdx = 0;
                while (!parser.match(')')) {
                    const type = parser.parseType();
                    if (startIdx + idx < op.operands.length) {
                        op.operands[startIdx + idx].type = type;
                    }
                    idx++;
                    if (!parser.accept(',')) {
                        break;
                    }
                }
            }
            if (!parser.accept(')')) {
                return false;
            }
        }

        // Parse 'outs' section
        if (parser.accept('id', 'outs')) {
            if (!parser.accept('(')) {
                return false;
            }
            const outsStart = op.operands.length;
            // Parse operands: %2, %3
            while (parser.match('%')) {
                const value = parser.expect('%');
                op.operands.push({ value });
                if (!parser.accept(',')) {
                    break;
                }
            }
            // Parse types: : type1, type2
            if (parser.accept(':')) {
                let idx = 0;
                while (!parser.match(')')) {
                    const type = parser.parseType();
                    if (outsStart + idx < op.operands.length) {
                        op.operands[outsStart + idx].type = type;
                    }
                    idx++;
                    if (!parser.accept(',')) {
                        break;
                    }
                }
            }
            if (!parser.accept(')')) {
                return false;
            }
        }

        // Some linalg operations may have a region after the signature
        if (parser.match('{')) {
            const region = {};
            parser.parseRegion(region);
            op.regions.push(region);
        }

        // Parse optional return type (can come after region for linalg.generic)
        if (parser.accept('->') || parser.accept('id', 'to')) {
            if (op.results.length > 0) {
                parser.parseArgumentTypes(op.results);
            } else {
                const type = parser.parseType();
                op.results.push({ type });
            }
        }

        return true;
    }

    _parseGenericOp(parser, op) {
        if (parser.match('{') || parser.match('#')) {
            if (parser.match('#')) {
                const attrRef = parser.expect('#');
                op.attributes.push({ name: 'trait', value: attrRef });
            } else {
                parser.parseAttributeDict(op.attributes);
            }
        }
        if (parser.accept('id', 'ins')) {
            parser.expect('(');
            while (parser.match('%')) {
                const value = parser.expect('%');
                op.operands.push({ value });
                if (!parser.accept(',')) {
                    break;
                }
            }
            if (parser.accept(':')) {
                let idx = 0;
                while (!parser.match(')')) {
                    const type = parser.parseType();
                    if (idx < op.operands.length) {
                        op.operands[idx].type = type;
                    }
                    idx++;
                    if (!parser.accept(',')) {
                        break;
                    }
                }
            }
            parser.expect(')');
        }
        if (parser.accept('id', 'outs')) {
            parser.expect('(');
            const outsStart = op.operands.length;
            while (parser.match('%')) {
                const value = parser.expect('%');
                op.operands.push({ value });
                if (!parser.accept(',')) {
                    break;
                }
            }
            if (parser.accept(':')) {
                let idx = 0;
                while (!parser.match(')')) {
                    const type = parser.parseType();
                    if (outsStart + idx < op.operands.length) {
                        op.operands[outsStart + idx].type = type;
                    }
                    idx++;
                    if (!parser.accept(',')) {
                        break;
                    }
                }
            }
            parser.expect(')');
        }
        if (parser.accept('id', 'attrs')) {
            parser.expect('=');
            parser.parseAttributeDict(op.attributes);
        }
        if (parser.match('{')) {
            const region = {};
            parser.parseRegion(region);
            op.regions.push(region);
        }
        if (parser.accept('->')) {
            parser.parseArgumentTypes(op.results);
        }
        return true;
    }
};

mlir.ONNXDialect = class extends mlir.Dialect {

    constructor(operations) {
        super('onnx', operations);
    }

    parseOperation(parser, opName, op) {
        const name = opName.replace(/^"|"$/g, '');

        // onnx.Constant has custom assembly format: dense<...> : type
        // Similar to stablehlo.constant
        if (name === 'onnx.Constant') {
            // Parse attribute (e.g., dense<"0x...">, dense<[1, 2, 3]>, etc.)
            const value = parser.parseValue();
            if (value) {
                op.attributes.push({ name: 'value', value: value.value });
            }
            // Parse optional : type
            if (parser.accept(':')) {
                parser.parseArgumentTypes(op.results);
            }
            return true;
        }

        // For all other ONNX operations, use default parsing
        return super.parseOperation(parser, opName, op);
    }
};

mlir.MhloDialect = class extends mlir.Dialect {

    constructor(operations) {
        super('mhlo', operations);
        this._binaryOps = new Set([
            'add', 'subtract', 'multiply', 'divide', 'remainder',
            'maximum', 'minimum', 'and', 'or', 'xor', 'shift_left', 'shift_right_arithmetic', 'shift_right_logical',
            'atan2', 'power', 'complex'
        ]);
    }

    parseOperation(parser, opName, op) {
        const name = opName.replace(/^"|"$/g, '');
        const opKind = name.substring('mhlo.'.length);
        if (this._binaryOps.has(opKind)) {
            op.operands = parser.parseArguments();
            parser.parseOptionalAttrDictWithKeyword(op.attributes);
            if (parser.accept(':')) {
                const type = parser.parseType();
                for (const operand of op.operands) {
                    operand.type = type;
                }
                if (op.results.length > 0) {
                    op.results[0].type = type;
                }
            }
            return true;
        }

        // For all other MHLO operations, use default parsing
        return super.parseOperation(parser, opName, op);
    }
};

mlir.QuantDialect = class extends mlir.Dialect {

    constructor(operations) {
        super('quant', operations);
    }

    parseType(parser, dialectType) {
        if (dialectType.startsWith('!quant.uniform') || dialectType.startsWith('!quant.calibrated')) {
            if (parser.match('<')) {
                const content = parser.skip('<', '>');
                return dialectType + content;
            }
            return dialectType;
        }
        return null;
    }
};

mlir.TosaDialect = class extends mlir.Dialect {

    constructor(operations) {
        super('tosa', operations);
        this._binaryOps = new Set(['add', 'sub', 'mul', 'div', 'maximum', 'minimum', 'logical_and', 'logical_or', 'logical_xor', 'logical_left_shift', 'logical_right_shift', 'arithmetic_right_shift', 'equal', 'greater', 'greater_equal', 'pow']);
        this._unaryOps = new Set(['abs', 'ceil', 'floor', 'negate', 'reciprocal', 'rsqrt', 'tanh', 'sigmoid', 'exp', 'log']);
        this._reduceOps = new Set(['reduce_min', 'reduce_max', 'reduce_sum', 'reduce_prod', 'reduce_all', 'reduce_any']);
    }

    parseType(parser, dialectType) {
        if (dialectType.startsWith('!tosa.shape')) {
            if (parser.match('<')) {
                const content = parser.skip('<', '>');
                return dialectType + content;
            }
            return dialectType;
        }
        return null;
    }

    parseOperation(parser, opName, op) {
        const name = opName.replace(/^"|"$/g, '');
        const opKind = name.substring('tosa.'.length);
        if (this._binaryOps.has(opKind) || this._unaryOps.has(opKind) || this._reduceOps.has(opKind) || opKind === 'argmax' || opKind === 'rescale') {
            op.operands = parser.parseArguments();
            if (parser.match('{')) {
                parser.parseAttributeDict(op.attributes);
            }
            if (parser.accept(':')) {
                if (parser.accept('(')) {
                    parser.parseArgumentTypes(op.operands);
                    parser.accept(')');
                }
                if (parser.accept('->')) {
                    parser.parseArgumentTypes(op.results);
                }
            }
            return true;
        }
        return super.parseOperation(parser, opName, op);
    }
};

mlir.IRDLDialect = class extends mlir.Dialect {

    constructor(operations) {
        super('irdl', operations);
    }

    parseOperation(parser, opName, op) {
        const name = opName.replace(/^"|"$/g, '');
        if (name === 'irdl.operands' || name === 'irdl.results' ||
            name === 'irdl.parameters' || name === 'irdl.attributes' ||
            name === 'irdl.regions') {
            if (parser.match('(')) {
                parser.expect('(');
                while (!parser.accept(')')) {
                    if (parser.match('id') || parser.match('string')) {
                        const paramName = parser.expect();
                        parser.expect(':');
                        const paramValue = parser.expect(); // Read the SSA value like %tensor
                        op.attributes.push({ name: paramName, value: paramValue });
                    }
                    parser.accept(',');
                }
            }
            op.loc = parser.parseLocation();
            return true;
        }
        return super.parseOperation(parser, opName, op);
    }
};

mlir.SPIRVDialect = class extends mlir.Dialect {

    constructor(operations) {
        super('spirv', operations);
    }

    parseType(parser, dialectType) {
        const simpleTypes = ['sampler', 'sampled_image', 'matrix', 'image', 'rtarray'];
        for (const simpleType of simpleTypes) {
            if (dialectType === `!spirv.${simpleType}`) {
                return dialectType;
            }
        }
        if (dialectType.startsWith('!spirv.ptr') || dialectType.startsWith('!spirv.array') ||
            dialectType.startsWith('!spirv.struct') || dialectType.startsWith('!spirv.coopmatrix')) {
            if (parser.match('<')) {
                const content = parser.skip('<', '>');
                return dialectType + content;
            }
            return dialectType;
        }
        return null;
    }

    parseOperation(parser, opName, op) {
        const name = opName.replace(/^"|"$/g, '');
        // spirv.module / spv.module has addressing model and memory model before the region
        if (name === 'spirv.module' || name === 'spv.module') {
            // Parse: spv.module Logical GLSL450 [requires #spv.vce<...>] { ... }
            // Read addressing model (Logical, Physical32, Physical64, etc.)
            if (parser.match('id')) {
                const addressingModel = parser.expect('id');
                op.attributes.push({ name: 'addressing_model', value: addressingModel });
            }
            // Read memory model (GLSL450, Vulkan, OpenCL, etc.)
            if (parser.match('id')) {
                const memoryModel = parser.expect('id');
                op.attributes.push({ name: 'memory_model', value: memoryModel });
            }
            // Parse optional 'requires' clause
            if (parser.accept('id', 'requires')) {
                const vce = parser.parseAttributeValue();
                op.attributes.push({ name: 'vce_triple', value: vce });
            }
            // Parse region
            if (parser.match('{')) {
                const region = {};
                parser.parseRegion(region);
                op.regions.push(region);
            }
            return true;
        }
        // spirv.func / spv.func has symbol, function type, and optional control string
        if (name === 'spirv.func' || name === 'spv.func') {
            // Parse: spirv.func @symbol() "None" attributes {...} { ... }
            //    or: spirv.func @symbol(%arg: type) -> type "None" { ... }
            // Parse symbol (@name)
            if (parser.match('@')) {
                const symbol = parser.expect('@');
                op.attributes.push({ name: 'sym_name', value: symbol });
            }
            // Create function_type structure
            const function_type = {
                inputs: [],
                results: []
            };
            // Parse function signature (arguments and return type)
            if (parser.match('(')) {
                function_type.inputs = parser.parseFunctionArgumentList();
            }
            // Parse return type if present
            if (parser.accept('->')) {
                function_type.results = parser.parseFunctionResultList();
            }
            op.attributes.push({ name: 'function_type', value: function_type });
            // Parse control string ("None", "Inline", "DontInline", etc.)
            if (parser.match('string')) {
                const control = parser.expect('string');
                op.attributes.push({ name: 'function_control', value: control });
            }
            // Parse optional attributes
            if (parser.match('id', 'attributes')) {
                parser.expect('id', 'attributes');
                parser.parseAttributeDict(op.attributes);
            }
            // Parse region
            if (parser.match('{')) {
                const region = {};
                parser.parseRegion(region);
                op.regions.push(region);
            }
            return true;
        }
        // spirv.GlobalVariable / spv.GlobalVariable - declares a global variable
        // Format: spv.GlobalVariable @symbol [built_in("...")] [bind(n, m)] : type
        if (name === 'spirv.GlobalVariable' || name === 'spv.GlobalVariable') {
            // Parse symbol reference
            if (parser.match('@')) {
                const symbol = parser.expect('@');
                op.attributes.push({ name: 'sym_name', value: symbol });
            }
            // Parse optional built_in attribute
            if (parser.accept('id', 'built_in')) {
                parser.expect('(');
                const builtIn = parser.expect('string');
                parser.expect(')');
                op.attributes.push({ name: 'built_in', value: builtIn });
            }
            // Parse optional bind attribute
            if (parser.accept('id', 'bind')) {
                parser.expect('(');
                const binding = parser.expect();
                parser.accept(',');
                const set = parser.expect();
                parser.expect(')');
                op.attributes.push({ name: 'descriptor_set', value: set });
                op.attributes.push({ name: 'binding', value: binding });
            }
            // Parse type after colon
            if (parser.accept(':')) {
                const type = parser.parseType();
                op.results = [{ type }];
            }
            return true;
        }
        // spirv.EntryPoint / spv.EntryPoint - defines entry point with execution model and interface variables
        // Format: spv.EntryPoint "GLCompute" @func_name, @var1, @var2, ...
        if (name === 'spirv.EntryPoint' || name === 'spv.EntryPoint') {
            // Parse execution model string ("GLCompute", "Vertex", "Fragment", etc.)
            if (parser.match('string')) {
                const executionModel = parser.expect('string');
                op.attributes.push({ name: 'execution_model', value: executionModel });
            }
            // Parse comma-separated symbol references
            op.operands = [];
            while (parser.match('@')) {
                const symbol = parser.expect('@');
                op.operands.push({ value: symbol });
                parser.accept(',');
            }
            return true;
        }
        // spirv.ExecutionMode / spv.ExecutionMode - specifies execution mode for entry point
        // Format: spv.ExecutionMode @func_name "LocalSize", 8, 2, 1
        if (name === 'spirv.ExecutionMode' || name === 'spv.ExecutionMode') {
            // Parse function symbol
            if (parser.match('@')) {
                const symbol = parser.expect('@');
                op.operands.push({ value: symbol });
            }
            // Parse execution mode string
            if (parser.match('string')) {
                const mode = parser.expect('string');
                op.attributes.push({ name: 'execution_mode', value: mode });
            }
            // Parse mode parameters (comma-separated integers)
            while (parser.accept(',')) {
                if (parser.match('int') || parser.match('number') || parser.match('id')) {
                    const param = parser.expect();
                    op.operands.push({ value: param });
                } else {
                    break;
                }
            }
            return true;
        }
        // spirv.mlir.loop / spv.mlir.loop - region with multi-block control flow
        // spirv.mlir.selection / spv.mlir.selection - region with multi-block control flow
        if (name === 'spirv.mlir.loop' || name === 'spv.mlir.loop' ||
            name === 'spirv.mlir.selection' || name === 'spv.mlir.selection') {
            // These operations have regions that are parsed by the generic parser
            // Add sentinel attribute to manipulate the heuristic at line 1226:
            // Setting op.attributes.length > 0 forces the else branch (region parsing)
            op.attributes.push({ name: '_has_region', value: true });
            // Return false to let the generic parser handle the region
            return false;
        }
        // spirv.Branch / spv.Branch and other branch operations with successors
        if (name === 'spirv.Branch' || name === 'spv.Branch' ||
            name === 'spirv.BranchConditional' || name === 'spv.BranchConditional' ||
            name.includes('Branch')) {
            // Parse operands if any (for conditional branches)
            op.operands = parser.parseArguments();

            // Parse successors
            if (parser.match('^')) {
                op.successors = [];
                while (parser.match('^')) {
                    const successor = {};
                    successor.label = parser.expect('^');
                    // Parse successor arguments with types
                    // Format: ^label(%val1, %val2, ... : type1, type2, ...)
                    if (parser.accept('(')) {
                        successor.arguments = [];
                        // Parse all values first
                        while (!parser.match(':') && !parser.match(')')) {
                            if (parser.match('%')) {
                                const arg = {};
                                arg.value = parser.expect('%');
                                successor.arguments.push(arg);
                                parser.accept(',');
                            } else {
                                break;
                            }
                        }
                        // Parse types if present
                        if (parser.accept(':')) {
                            let idx = 0;
                            while (idx < successor.arguments.length && !parser.match(')')) {
                                const type = parser.parseType();
                                successor.arguments[idx].type = type;
                                idx++;
                                parser.accept(',');
                            }
                        }
                        parser.accept(')');
                    }
                    op.successors.push(successor);
                    if (!parser.accept(',')) {
                        break;
                    }
                }
            }
            return true;
        }
        // spirv.CompositeInsert with 'into' keyword
        // Format: spirv.CompositeInsert %object, %composite[indices] : object-type into composite-type
        if (name === 'spirv.CompositeInsert' || name === 'spv.CompositeInsert') {
            // Parse operands (object and composite)
            op.operands = parser.parseArguments();
            // Parse indices as attributes
            if (parser.match('[')) {
                parser.expect('[');
                const indices = [];
                while (!parser.accept(']')) {
                    const index = parser.expect();
                    if (parser.accept(':')) {
                        parser.expect(); // Skip type (e.g., i32)
                    }
                    indices.push(index);
                    parser.accept(',');
                }
                op.attributes.push({ name: 'indices', value: indices });
            }
            // Parse operand types after ':'
            if (parser.accept(':')) {
                parser.parseArgumentTypes(op.operands);
            }
            // Parse result type after 'into'
            if (parser.accept('id', 'into')) {
                const resultType = parser.parseType();
                op.results = [{ type: resultType }];
            }
            return true;
        }
        return super.parseOperation(parser, opName, op);
    }
};

mlir.CFDialect = class extends mlir.Dialect {

    constructor(operations) {
        super('cf', operations);
    }
};

mlir.AsukaDialect = class extends mlir.Dialect {

    constructor(operations) {
        super('asuka', operations);
    }

    parseOperation(parser, opName, op) {
        // const name = opName.replace(/^"|"$/g, '');
        // Asuka operations have special syntax with inline attributes:
        // asuka.dot %arg0, %arg1, batch_dims = [...] x [...], reduce_dims = [...] x [...] : (...) -> (...)
        // asuka.split %arg, dim = N : (...) -> (...)
        // asuka.softmax %arg, dim = N : (...) -> (...)
        // asuka.add %arg0, %arg1 : (...) -> (...)

        // Parse operands
        op.operands = parser.parseArguments();

        // Parse inline named attributes (like batch_dims, reduce_dims, dim)
        // These come after operands but before the type signature ':'
        while (parser.match('id') && !parser.match(':') && !parser.match('{')) {
            const attrName = parser.expect('id');
            if (parser.accept('=')) {
                let attrValue = null;
                if (parser.match('[')) {
                    // Parse array attribute
                    attrValue = parser.parseValue();
                    // Check for 'x' operator (used in batch_dims = [0] x [])
                    if (parser.match('id') && parser.token.value === 'x') {
                        parser.expect('id'); // consume 'x'
                        const secondValue = parser.parseValue();
                        attrValue = { kind: 'pair', first: attrValue, second: secondValue };
                    }
                } else {
                    // Parse scalar or other value
                    attrValue = parser.parseValue();
                }
                op.attributes.push({ name: attrName, value: attrValue });
                parser.accept(','); // optional comma between attributes
            }
        }

        // Parse type signature if present
        if (parser.accept(':')) {
            parser.parseArgumentTypes(op.operands);
        }
        if (parser.accept('->')) {
            if (op.results.length > 0) {
                parser.parseArgumentTypes(op.results);
            } else {
                op.results = parser.parseArguments();
            }
        }

        return true;
    }
};

mlir.AsyncDialect = class extends mlir.Dialect {

    constructor(operations) {
        super('async', operations);
    }

    parseType(parser, dialectType) {
        const simpleTypes = ['token', 'group', 'coro.id', 'coro.handle', 'coro.state'];
        for (const simpleType of simpleTypes) {
            if (dialectType === `!async.${simpleType}`) {
                return dialectType;
            }
        }
        if (dialectType === '!async.value') {
            if (parser.match('<')) {
                const content = parser.skip('<', '>');
                return dialectType + content;
            }
            return dialectType;
        }
        return null;
    }

    parseOperation(parser, opName, op) {
        const name = opName.replace(/^"|"$/g, '');
        if (name === 'async.execute') {
            return this._parseExecuteOp(parser, op);
        }
        if (name === 'async.func') {
            return this._parseFuncOp(parser, op);
        }
        if (name === 'async.await') {
            return this._parseAwaitOp(parser, op);
        }
        return super.parseOperation(parser, opName, op);
    }

    _parseExecuteOp(parser, op) {
        if (parser.match('[')) {
            parser.expect('[');
            while (!parser.match(']')) {
                op.operands.push({ value: parser.expect('%') });
                if (!parser.accept(',')) {
                    break;
                }
            }
            parser.expect(']');
        }
        if (parser.match('(')) {
            parser.expect('(');
            while (!parser.match(')') && !parser.match(':')) {
                op.operands.push({ value: parser.expect('%') });
                if (parser.match('id', 'as')) {
                    parser.expect('id');
                    parser.expect('%');
                }
                if (parser.match(':')) {
                    parser.parseType();
                }
                if (!parser.accept(',')) {
                    break;
                }
            }
            parser.expect(')');
        }
        if (parser.match('->')) {
            parser.expect('->');
            if (parser.match('(')) {
                parser.expect('(');
                while (!parser.match(')')) {
                    const resultType = parser.parseType();
                    if (op.results.length < 1) {
                        op.results.push({ type: '!async.token' });
                    }
                    op.results.push({ type: resultType });
                    if (!parser.accept(',')) {
                        break;
                    }
                }
                parser.expect(')');
            } else {
                const resultType = parser.parseType();
                if (op.results.length < 1) {
                    op.results.push({ type: '!async.token' });
                }
                op.results.push({ type: resultType });
            }
        } else if (op.results.length === 1 && !op.results[0].type) {
            op.results[0].type = '!async.token';
        }
        parser.parseOptionalAttrDictWithKeyword(op.attributes);
        if (parser.match('{')) {
            const region = {};
            parser.parseRegion(region);
            op.regions.push(region);
        }
        return true;
    }

    _parseFuncOp(parser, op) {
        parser.parseOptionalVisibilityKeyword(op.attributes);
        parser.parseSymbolName('sym_name', op.attributes);
        const type = {};
        type.inputs = parser.parseFunctionArgumentList();
        parser.parseOptionalAttrDictWithKeyword(op.attributes);
        type.results = [];
        if (parser.accept('->')) {
            if (parser.match('(')) {
                parser.expect('(');
                while (!parser.match(')')) {
                    type.results.push({ type: parser.parseType() });
                    if (!parser.accept(',')) {
                        break;
                    }
                }
                parser.expect(')');
            } else {
                type.results.push({ type: parser.parseType() });
            }
        }
        op.attributes.push({ name: 'function_type', value: type });
        parser.parseOptionalAttrDictWithKeyword(op.attributes);
        if (parser.match('{')) {
            const region = {};
            parser.parseRegion(region);
            op.regions.push(region);
        }
        return true;
    }

    _parseAwaitOp(parser, op) {
        op.operands.push({ value: parser.expect('%') });
        parser.expect(':');
        const operandType = parser.parseType();
        if (op.operands.length > 0) {
            op.operands[0].type = operandType;
        }
        if (operandType && operandType.startsWith('!async.value')) {
            const match = operandType.match(/!async\.value<(.+)>/);
            if (match && op.results.length > 0) {
                [, op.results[0].type] = match;
            }
        }
        parser.parseOptionalAttrDictWithKeyword(op.attributes);
        return true;
    }
};

mlir.ArithDialect = class extends mlir.Dialect {

    constructor(operations) {
        super('arith', operations);
    }

    parseOperation(parser, opName, op) {
        const name = opName.replace(/^"|"$/g, '');
        if (name === 'arith.cmpi' || name === 'arith.cmpf') {
            return this._parseCmpOp(parser, op);
        }
        if (name === 'arith.select') {
            return this._parseSelectOp(parser, op);
        }
        return super.parseOperation(parser, opName, op);
    }

    _parseCmpOp(parser, op) {
        // arith.cmpi <predicate>, <lhs>, <rhs> [attr-dict] : <type>
        // Parse predicate as bare identifier or string
        if (!parser.match('id') && !parser.match('string')) {
            return false;
        }
        const predicate = parser.match('string') ? parser.expect('string') : parser.expect('id');
        op.attributes.push({ name: 'predicate', value: predicate });

        // Parse comma
        if (!parser.accept(',')) {
            return false;
        }

        // Parse lhs operand
        if (!parser.match('%')) {
            return false;
        }
        op.operands.push({ value: parser.expect('%') });

        // Parse comma
        if (!parser.accept(',')) {
            return false;
        }

        // Parse rhs operand
        if (!parser.match('%')) {
            return false;
        }
        op.operands.push({ value: parser.expect('%') });

        // Parse optional attribute dict
        if (parser.match('{')) {
            parser.parseAttributeDict(op.attributes);
        }

        // Parse type signature: : <type>
        if (parser.accept(':')) {
            const type = parser.parseType();
            op.operands[0].type = type;
            op.operands[1].type = type;
            // Result type is i1 for comparison operations
            // Only add result if one doesn't already exist (from %0 = parsing)
            if (op.results.length === 0) {
                op.results.push({ type: 'i1' });
            } else {
                // Result already exists from %0 = parsing, just set its type
                op.results[0].type = 'i1';
            }
        }

        return true;
    }

    _parseSelectOp(parser, op) {
        op.operands = parser.parseArguments();
        if (parser.accept(':')) {
            const condType = parser.parseType();
            if (parser.accept(',')) {
                const resultType = parser.parseType();
                if (op.operands.length > 0) {
                    op.operands[0].type = condType;
                }
                if (op.operands.length > 1) {
                    op.operands[1].type = resultType;
                    op.operands[2].type = resultType;
                }
                if (op.results.length > 0) {
                    op.results[0].type = resultType;
                } else {
                    op.results.push({ type: resultType });
                }
            } else if (op.results.length > 0) {
                op.results[0].type = condType;
            } else {
                op.results.push({ type: condType });
            }
        }
        return true;
    }
};

mlir.BuiltinDialect = class extends mlir.Dialect {

    constructor(operations) {
        super('builtin', operations);
    }

    parseOperation(parser, opName, op) {
        const name = opName.replace(/^"|"$/g, '');
        if (name === 'builtin.module') {
            op.sym_name = parser.parseOptionalSymbolName();
            parser.parseOptionalAttrDictWithKeyword(op.attributes);
            const region = {};
            parser.parseRegion(region);
            op.regions.push(region);
            return true;
        }
        if (name === 'builtin.func') {
            return this._parseFuncOp(parser, op);
        }
        if (name === 'builtin.call' || name === 'builtin.call_indirect') {
            parser.parseSymbolName('callee', op.attributes);
            op.operands = parser.parseArguments();
            if (parser.accept(':')) {
                parser.parseArgumentTypes(op.operands);
            }
            if (parser.accept('->')) {
                parser.parseArgumentTypes(op.results);
            }
            return true;
        }
        return super.parseOperation(parser, opName, op);
    }

    _parseFuncOp(parser, op) {
        parser.parseOptionalVisibilityKeyword(op.attributes);
        parser.parseSymbolName('sym_name', op.attributes);
        const type = {};
        type.inputs = parser.parseFunctionArgumentList();
        parser.parseOptionalAttrDictWithKeyword(op.attributes);
        type.results = [];
        if (parser.accept('->')) {
            for (const result of parser.parseFunctionResultList()) {
                type.results.push(result);
            }
        }
        op.attributes.push({ name: 'function_type', value: type });
        parser.parseOptionalAttrDictWithKeyword(op.attributes);
        if (parser.match('{')) {
            const region = {};
            parser.parseRegion(region);
            op.regions.push(region);
        }
        if (op.regions.length > 0) {
            const region = op.regions[op.regions.length - 1];
            if (region.blocks.length > 0) {
                const block = region.blocks[region.blocks.length - 1];
                if (block.operations.length > 0) {
                    const lastOp = block.operations[block.operations.length - 1];
                    type.results = lastOp.operands;
                    block.operations.pop();
                }
            }
        }
        return true;
    }
};

mlir.BufferizationDialect = class extends mlir.Dialect {

    constructor(operations) {
        super('bufferization', operations);
    }

    parseOperation(parser, opName, op) {
        const name = opName.replace(/^"|"$/g, '');
        if (name === 'bufferization.materialize_in_destination') {
            return this._parseMaterializeInDestinationOp(parser, op);
        }
        return super.parseOperation(parser, opName, op);
    }

    _parseMaterializeInDestinationOp(parser, op) {
        // Assembly format: $source `in` (`restrict` $restrict^)? (`writable` $writable^)? $dest
        //                  attr-dict `:` functional-type(operands, results)

        // Parse source operand
        if (!parser.match('%')) {
            return false;
        }
        const source = parser.expect('%');
        op.operands.push({ value: source });
        if (!parser.accept('id', 'in')) {
            return false;
        }
        if (parser.accept('id', 'restrict')) {
            op.attributes.push({ name: 'restrict', value: true });
        }
        if (parser.accept('id', 'writable')) {
            op.attributes.push({ name: 'writable', value: true });
        }
        if (!parser.match('%')) {
            return false;
        }
        const dest = parser.expect('%');
        op.operands.push({ value: dest });
        if (parser.match('{')) {
            parser.parseAttributeDict(op.attributes);
        }
        if (parser.accept(':')) {
            if (parser.accept('(')) {
                let typeIndex = 0;
                while (!parser.accept(')')) {
                    const type = parser.parseType();
                    if (typeIndex < op.operands.length) {
                        op.operands[typeIndex].type = type;
                        typeIndex++;
                    }
                    parser.accept(',');
                }
                if (parser.accept('->')) {
                    if (parser.accept('(')) {
                        while (!parser.accept(')')) {
                            const resultType = parser.parseType();
                            if (resultType && resultType !== '()') {
                                op.results.push({ type: resultType });
                            }
                            parser.accept(',');
                        }
                    }
                }
            }
        }
        return true;
    }
};

mlir.SCFDialect = class extends mlir.Dialect {

    constructor(operations) {
        super('scf', operations);
    }

    parseOperation(parser, opName, op) {
        const name = opName.replace(/^"|"$/g, '');
        if (name === 'scf.for') {
            return this._parseForOp(parser, op);
        }
        if (name === 'scf.if') {
            return this._parseIfOp(parser, op);
        }
        if (name === 'scf.while') {
            return this._parseWhileOp(parser, op);
        }
        if (name === 'scf.yield') {
            return this._parseYieldOp(parser, op);
        }
        return super.parseOperation(parser, opName, op);
    }

    _parseForOp(parser, op) {
        // scf.for [unsigned] %inductionVar = %lb to %ub step %step [iter_args(...)] [: type] { region }
        // Check for optional "unsigned" keyword
        if (parser.accept('id', 'unsigned')) {
            op.attributes.push({ name: 'unsignedCmp', value: true });
        }
        // Parse induction variable: %inductionVar
        if (!parser.match('%')) {
            return false;
        }
        const inductionVar = parser.expect('%');
        // Parse '='
        if (!parser.accept('=')) {
            return false;
        }
        // Parse lower bound: %lb
        if (parser.match('%')) {
            op.operands.push({ value: parser.expect('%') });
        } else {
            return false;
        }
        // Parse 'to' keyword
        if (!parser.accept('id', 'to')) {
            return false;
        }
        // Parse upper bound: %ub
        if (parser.match('%')) {
            op.operands.push({ value: parser.expect('%') });
        } else {
            return false;
        }
        // Parse 'step' keyword
        if (!parser.accept('id', 'step')) {
            return false;
        }
        // Parse step: %step
        if (parser.match('%')) {
            op.operands.push({ value: parser.expect('%') });
        } else {
            return false;
        }
        // Parse optional iter_args
        if (parser.accept('id', 'iter_args')) {
            // iter_args(%arg = %init, ...) -> (type, ...)
            if (parser.accept('(')) {
                while (!parser.accept(')')) {
                    // Parse %arg = %init
                    if (parser.match('%')) {
                        parser.expect('%'); // Skip the loop-carried variable name
                    }
                    if (parser.accept('=')) {
                        if (parser.match('%')) {
                            op.operands.push({ value: parser.expect('%') });
                        } else {
                            // Handle non-SSA values (constants, etc.)
                            const value = parser.parseValue();
                            if (value) {
                                op.operands.push(value);
                            }
                        }
                    }
                    parser.accept(',');
                }
            }
            // Parse optional -> (result types)
            if (parser.accept('->')) {
                // Parse result types
                const resultTypes = [];
                if (parser.accept('(')) {
                    while (!parser.accept(')')) {
                        const resultType = parser.parseType();
                        resultTypes.push(resultType);
                        parser.accept(',');
                    }
                } else {
                    const resultType = parser.parseType();
                    resultTypes.push(resultType);
                }
                // Set types on existing results (from result list before '=')
                // or create new results if none exist
                if (op.results.length > 0) {
                    for (let i = 0; i < resultTypes.length && i < op.results.length; i++) {
                        op.results[i].type = resultTypes[i];
                    }
                } else {
                    for (const type of resultTypes) {
                        op.results.push({ type });
                    }
                }
            }
        }
        // Parse optional type: : type
        if (parser.accept(':')) {
            parser.parseType(); // Parse and discard the induction variable type
        }
        // Parse region: { ... }
        if (parser.match('{')) {
            const region = {};
            parser.parseRegion(region);
            // Set the induction variable as the first block argument
            if (region.blocks && region.blocks.length > 0) {
                if (!region.blocks[0].arguments) {
                    region.blocks[0].arguments = [];
                }
                if (region.blocks[0].arguments.length > 0) {
                    region.blocks[0].arguments[0] = { value: inductionVar };
                } else {
                    region.blocks[0].arguments.push({ value: inductionVar });
                }
            }
            op.regions.push(region);
        }
        return true;
    }

    _parseIfOp(parser, op) {
        // scf.if %condition [-> (result_types)] { ... } [else { ... }]
        // Parse condition
        if (parser.match('%')) {
            op.operands.push({ value: parser.expect('%') });
        } else {
            return false;
        }
        // Parse optional result types
        if (parser.accept('->')) {
            const resultTypes = [];
            if (parser.accept('(')) {
                while (!parser.accept(')')) {
                    const resultType = parser.parseType();
                    resultTypes.push(resultType);
                    parser.accept(',');
                }
            } else {
                const resultType = parser.parseType();
                resultTypes.push(resultType);
            }
            // Set types on existing results or create new ones
            if (op.results.length > 0) {
                for (let i = 0; i < resultTypes.length && i < op.results.length; i++) {
                    op.results[i].type = resultTypes[i];
                }
            } else {
                for (const type of resultTypes) {
                    op.results.push({ type });
                }
            }
        }
        // Parse then region
        if (parser.match('{')) {
            const region = {};
            parser.parseRegion(region);
            op.regions.push(region);
        } else {
            return false;
        }
        // Parse optional else region
        if (parser.accept('id', 'else')) {
            if (parser.match('{')) {
                const region = {};
                parser.parseRegion(region);
                op.regions.push(region);
            }
        }
        return true;
    }

    _parseWhileOp(parser, op) {
        // scf.while (%arg = %init, ...) : (type, ...) -> (type, ...) { before region } do { after region }
        // Parse operands in parentheses
        if (parser.accept('(')) {
            while (!parser.accept(')')) {
                if (parser.match('%')) {
                    parser.expect('%'); // Skip variable name
                }
                if (parser.accept('=')) {
                    if (parser.match('%')) {
                        op.operands.push({ value: parser.expect('%') });
                    } else {
                        const value = parser.parseValue();
                        if (value) {
                            op.operands.push(value);
                        }
                    }
                }
                parser.accept(',');
            }
        }
        // Parse types
        if (parser.accept(':')) {
            // Parse operand types
            if (parser.accept('(')) {
                while (!parser.accept(')')) {
                    parser.parseType();
                    parser.accept(',');
                }
            }
            // Parse optional result types
            if (parser.accept('->')) {
                const resultTypes = [];
                if (parser.accept('(')) {
                    while (!parser.accept(')')) {
                        const resultType = parser.parseType();
                        resultTypes.push(resultType);
                        parser.accept(',');
                    }
                }
                // Set types on existing results or create new ones
                if (op.results.length > 0) {
                    for (let i = 0; i < resultTypes.length && i < op.results.length; i++) {
                        op.results[i].type = resultTypes[i];
                    }
                } else {
                    for (const type of resultTypes) {
                        op.results.push({ type });
                    }
                }
            }
        }
        // Parse before region
        if (parser.match('{')) {
            const region = {};
            parser.parseRegion(region);
            op.regions.push(region);
        }
        // Parse 'do' keyword and after region
        if (parser.accept('id', 'do')) {
            if (parser.match('{')) {
                const region = {};
                parser.parseRegion(region);
                op.regions.push(region);
            }
        }
        return true;
    }

    _parseYieldOp(parser, op) {
        // scf.yield [%operand, ...] [: type, ...]
        // Parse optional operands
        if (parser.match('%')) {
            do {
                const operand = parser.parseValue();
                op.operands.push(operand);
            } while (parser.accept(','));
        }
        // Parse optional types after colon
        if (parser.accept(':')) {
            parser.parseArgumentTypes(op.operands);
        }
        return true;
    }
};

mlir.ShapeDialect = class extends mlir.Dialect {

    constructor(operations) {
        super('shape', operations);
    }

    parseType(parser, dialectType) {
        if (dialectType === '!shape.shape' || dialectType === '!shape.witness' ||
            dialectType === '!shape.size' || dialectType === '!shape.value_shape') {
            return dialectType;
        }
        return null;
    }

    parseOperation(parser, opName, op) {
        const name = opName.replace(/^"|"$/g, '');
        if (name === 'shape.func') {
            return this._parseFuncOp(parser, op);
        }
        if (name === 'shape.assuming') {
            return this._parseAssumingOp(parser, op);
        }
        if (name === 'shape.const_shape') {
            return this._parseConstShapeOp(parser, op);
        }
        if (name === 'shape.reduce') {
            return this._parseReduceOp(parser, op);
        }
        if (name === 'shape.function_library') {
            return this._parseFunctionLibraryOp(parser, op);
        }
        return super.parseOperation(parser, opName, op);
    }

    _parseFuncOp(parser, op) {
        parser.parseOptionalVisibilityKeyword(op.attributes);
        parser.parseSymbolName('sym_name', op.attributes);
        const type = {};
        type.inputs = parser.parseFunctionArgumentList();
        parser.parseOptionalAttrDictWithKeyword(op.attributes);
        type.results = [];
        if (parser.accept('->')) {
            for (const result of parser.parseFunctionResultList()) {
                type.results.push(result);
            }
        }
        op.attributes.push({ name: 'function_type', value: type });
        parser.parseOptionalAttrDictWithKeyword(op.attributes);
        if (parser.match('{')) {
            const region = {};
            parser.parseRegion(region);
            op.regions.push(region);
        }
        return true;
    }

    _parseAssumingOp(parser, op) {
        if (!parser.match('%')) {
            return false;
        }
        op.operands.push({ value: parser.expect('%') });
        if (parser.accept('->')) {
            parser.parseArgumentTypes(op.results);
        }
        if (parser.match('{')) {
            const region = {};
            parser.parseRegion(region);
            op.regions.push(region);
        }
        parser.parseOptionalAttrDict(op.attributes);
        return true;
    }

    _parseConstShapeOp(parser, op) {
        parser.parseOptionalAttrDict(op.attributes);
        const extents = parser.parseAttributeValue();
        op.attributes.push({ name: 'shape', value: extents });
        if (parser.accept(':')) {
            const type = parser.parseType();
            op.results.push({ type });
        }
        return true;
    }

    _parseReduceOp(parser, op) {
        if (!parser.match('(')) {
            return false;
        }
        parser.accept('(');
        while (parser.match('%')) {
            op.operands.push({ value: parser.expect('%') });
            if (!parser.accept(',')) {
                break;
            }
        }
        parser.accept(')');
        if (parser.accept(':')) {
            const shapeType = parser.parseType();
            if (op.operands.length > 0) {
                op.operands[0].type = shapeType;
            }
        }
        if (parser.accept('->')) {
            parser.parseArgumentTypes(op.results);
            for (let i = 1; i < op.operands.length && i - 1 < op.results.length; i++) {
                op.operands[i].type = op.results[i - 1].type;
            }
        }
        if (parser.match('{')) {
            const region = {};
            parser.parseRegion(region);
            op.regions.push(region);
        }
        parser.parseOptionalAttrDict(op.attributes);
        return true;
    }

    _parseFunctionLibraryOp(parser, op) {
        parser.parseSymbolName('sym_name', op.attributes);
        parser.parseOptionalAttrDictWithKeyword(op.attributes);
        if (parser.match('{')) {
            const region = {};
            parser.parseRegion(region);
            op.regions.push(region);
        }
        return true;
    }
};

mlir.SparseTensorDialect = class extends mlir.Dialect {

    constructor(operations) {
        super('sparse_tensor', operations);
    }

    parseOperation(parser, opName, op) {
        const name = opName.replace(/^"|"$/g, '');
        if (name === 'sparse_tensor.iterate') {
            return this._parseIterateOp(parser, op);
        }
        if (name === 'sparse_tensor.coiterate') {
            return this._parseCoIterateOp(parser, op);
        }
        return super.parseOperation(parser, opName, op);
    }

    _parseIterateOp(parser, op) {
        if (!parser.match('%')) {
            return false;
        }
        parser.expect('%');
        if (!parser.accept('id', 'in')) {
            return false;
        }
        if (!parser.match('%')) {
            return false;
        }
        op.operands.push({ value: parser.expect('%') });
        if (parser.accept('id', 'at')) {
            parser.accept('(');
            while (parser.match('%') || parser.match('id')) {
                parser.expect();
                if (!parser.accept(',')) {
                    break;
                }
            }
            parser.accept(')');
        }
        if (parser.accept('id', 'iter_args')) {
            parser.accept('(');
            while (parser.match('%')) {
                op.operands.push({ value: parser.expect('%') });
                if (parser.accept('=')) {
                    parser.expect('%');
                }
                if (!parser.accept(',')) {
                    break;
                }
            }
            parser.accept(')');
        }
        if (parser.accept(':')) {
            parser.parseType();
        }
        if (parser.accept('->')) {
            parser.parseArgumentTypes(op.results);
        }
        if (parser.match('{')) {
            const region = {};
            parser.parseRegion(region);
            op.regions.push(region);
        }
        parser.parseOptionalAttrDict(op.attributes);
        return true;
    }

    _parseCoIterateOp(parser, op) {
        if (!parser.accept('(')) {
            return false;
        }
        while (parser.match('%')) {
            op.operands.push({ value: parser.expect('%') });
            if (!parser.accept(',')) {
                break;
            }
        }
        parser.accept(')');
        if (parser.accept('id', 'at')) {
            parser.accept('(');
            while (parser.match('%') || parser.match('id')) {
                parser.expect();
                if (!parser.accept(',')) {
                    break;
                }
            }
            parser.accept(')');
        }
        if (parser.accept('id', 'iter_args')) {
            parser.accept('(');
            while (parser.match('%')) {
                op.operands.push({ value: parser.expect('%') });
                if (parser.accept('=')) {
                    parser.expect('%');
                }
                if (!parser.accept(',')) {
                    break;
                }
            }
            parser.accept(')');
        }
        if (parser.accept(':')) {
            parser.accept('(');
            while (!parser.match(')')) {
                parser.parseType();
                if (!parser.accept(',')) {
                    break;
                }
            }
            parser.accept(')');
        }
        if (parser.accept('->')) {
            parser.parseArgumentTypes(op.results);
        }
        while (parser.accept('id', 'case')) {
            while (parser.match('%') || parser.match('id')) {
                parser.expect();
                if (!parser.accept(',')) {
                    break;
                }
            }
            if (parser.match('{')) {
                const region = {};
                parser.parseRegion(region);
                op.regions.push(region);
            }
        }
        parser.parseOptionalAttrDict(op.attributes);
        return true;
    }
};

mlir.FuncDialect = class extends mlir.Dialect {

    constructor(operations) {
        super('func', operations);
    }

    parseOperation(parser, opName, op) {
        const name = opName.replace(/^"|"$/g, '');
        if (name === 'func.func') {
            parser.parseOptionalVisibilityKeyword(op.attributes);
            parser.parseSymbolName('sym_name', op.attributes);
            const type = {};
            type.inputs = parser.parseFunctionArgumentList();
            parser.parseOptionalAttrDictWithKeyword(op.attributes);
            type.results = [];
            if (parser.accept('->')) {
                for (const result of parser.parseFunctionResultList()) {
                    type.results.push(result);
                }
            }
            op.attributes.push({ name: 'function_type', value: type });
            parser.parseOptionalAttrDictWithKeyword(op.attributes);
            if (parser.match('{')) {
                const region = {};
                parser.parseRegion(region);
                op.regions.push(region);
            }
            if (op.regions.length > 0) {
                const region = op.regions[op.regions.length - 1];
                if (region.blocks.length > 0) {
                    const block = region.blocks[region.blocks.length - 1];
                    if (block.operations.length > 0) {
                        const lastOp = block.operations[block.operations.length - 1];
                        type.results = lastOp.operands;
                        block.operations.pop();
                    }
                }
            }
            return true;
        }
        if (name === 'func.call' || name === 'func.call_indirect') {
            parser.parseSymbolName('callee', op.attributes);
            op.operands = parser.parseArguments();
            if (parser.accept(':')) {
                parser.parseArgumentTypes(op.operands);
            }
            if (parser.accept('->')) {
                parser.parseArgumentTypes(op.results);
            }
            return true;
        }
        if (name === 'func.return' || name === 'return') {
            op.operands = parser.parseArguments();
            if (parser.accept(':')) {
                parser.parseArgumentTypes(op.operands);
            }
            return true;
        }
        return super.parseOperation(parser, opName, op);
    }
};

mlir.GpuDialect = class extends mlir.Dialect {

    constructor(operations) {
        super('gpu', operations);
    }

    parseOperation(parser, opName, op) {
        const name = opName.replace(/^"|"$/g, '');
        if (name === 'gpu.module') {
            op.sym_name = parser.parseOptionalSymbolName();
            parser.parseOptionalAttrDictWithKeyword(op.attributes);
            const region = {};
            parser.parseRegion(region);
            op.regions.push(region);
            return true;
        }
        if (name === 'gpu.func') {
            return this._parseFuncOp(parser, op);
        }
        return super.parseOperation(parser, opName, op);
    }

    _parseFuncOp(parser, op) {
        parser.parseOptionalVisibilityKeyword(op.attributes);
        parser.parseSymbolName('sym_name', op.attributes);
        const type = {};
        type.inputs = parser.parseFunctionArgumentList();
        parser.parseOptionalAttrDictWithKeyword(op.attributes);
        type.results = [];
        if (parser.accept('->')) {
            for (const result of parser.parseFunctionResultList()) {
                type.results.push(result);
            }
        }
        op.attributes.push({ name: 'function_type', value: type });
        parser.parseOptionalAttrDictWithKeyword(op.attributes);
        if (parser.match('{')) {
            const region = {};
            parser.parseRegion(region);
            op.regions.push(region);
        }
        if (op.regions.length > 0) {
            const region = op.regions[op.regions.length - 1];
            if (region.blocks.length > 0) {
                const block = region.blocks[region.blocks.length - 1];
                if (block.operations.length > 0) {
                    const lastOp = block.operations[block.operations.length - 1];
                    type.results = lastOp.operands;
                    block.operations.pop();
                }
            }
        }
        return true;
    }
};

mlir.ArmSMEDialect = class extends mlir.Dialect {

    constructor(operations) {
        super('arm_sme', operations);
    }
};

mlir.ArmNeonDialect = class extends mlir.Dialect {

    constructor(operations) {
        super('arm_neon', operations);
    }
};

mlir.ArmSVEDialect = class extends mlir.Dialect {

    constructor(operations) {
        super('arm_sve', operations);
    }
};

mlir.NVVMDialect = class extends mlir.Dialect {

    constructor(operations) {
        super('nvvm', operations);
    }
};

mlir.OpenMPDialect = class extends mlir.Dialect {

    constructor(operations) {
        super('omp', operations);
    }
};

mlir.LLVMDialect = class extends mlir.Dialect {

    constructor(operations) {
        super('llvm', operations);
    }

    parseType(parser, dialectType) {
        // Handle LLVM pointer types: !llvm.ptr (opaque) or !llvm.ptr<T> (typed, legacy)
        if (dialectType === '!llvm.ptr') {
            if (parser.match('<')) {
                // Typed pointer: !llvm.ptr<i8>, !llvm.ptr<f32>, etc.
                const content = parser.skip('<', '>');
                return dialectType + content;
            }
            // Opaque pointer: !llvm.ptr
            return dialectType;
        }
        // Handle other LLVM types: !llvm.struct, !llvm.array, !llvm.vec, etc.
        if (dialectType.startsWith('!llvm.')) {
            if (parser.match('<')) {
                const content = parser.skip('<', '>');
                return dialectType + content;
            }
            return dialectType;
        }
        return null;
    }

    parseOperation(parser, opName, op) {
        const name = opName.replace(/^"|"$/g, '');
        if (name === 'llvm.func') {
            return this._parseLLVMFuncOp(parser, op);
        }
        if (name === 'llvm.mlir.global') {
            return this._parseLLVMGlobalOp(parser, op);
        }
        return super.parseOperation(parser, opName, op);
    }

    _parseLLVMGlobalOp(parser, op) {
        const linkageKeywords = ['external', 'available_externally', 'linkonce', 'linkonce_odr', 'weak', 'weak_odr', 'appending', 'internal', 'private', 'extern_weak', 'common'];
        if (parser.match('id') && linkageKeywords.includes(parser._token.value)) {
            op.attributes.push({ name: 'linkage', value: parser.expect('id') });
        }
        if (parser.match('@')) {
            const symbol = parser.expect('@');
            op.attributes.push({ name: 'sym_name', value: symbol });
        }
        parser.expect('(');
        parser.expect(')');
        if (parser.match('{')) {
            parser.parseAttributeDict(op.attributes);
        }
        if (parser.accept(':')) {
            const type = parser.parseType();
            op.results = [{ type }];
        }
        return true;
    }

    _parseLLVMFuncOp(parser, op) {
        const linkageKeywords = ['external', 'available_externally', 'linkonce', 'linkonce_odr', 'weak', 'weak_odr', 'appending', 'internal', 'private', 'extern_weak', 'common'];
        if (parser.match('id') && linkageKeywords.includes(parser._token.value)) {
            op.attributes.push({ name: 'linkage', value: parser.expect('id') });
        }
        const visibilityKeywords = ['default', 'hidden', 'protected'];
        if (parser.match('id') && visibilityKeywords.includes(parser._token.value)) {
            op.attributes.push({ name: 'visibility_', value: parser.expect('id') });
        }
        const unnamedAddrKeywords = ['unnamed_addr', 'local_unnamed_addr'];
        if (parser.match('id') && unnamedAddrKeywords.includes(parser._token.value)) {
            op.attributes.push({ name: 'unnamed_addr', value: parser.expect('id') });
        }
        const cconvKeywords = ['ccc', 'fastcc', 'coldcc', 'cc', 'webkit_jscc', 'anyregcc', 'preserve_mostcc', 'preserve_allcc', 'preserve_nonecc', 'cxx_fast_tlscc', 'tailcc', 'swiftcc', 'swifttailcc', 'cfguard_checkcc', 'ghccc', 'arm_apcscc', 'arm_aapcscc', 'arm_aapcs_vfpcc', 'aarch64_vector_pcs', 'aarch64_sve_vector_pcs', 'aarch64_sme_preservemost_from_x0', 'aarch64_sme_preservemost_from_x2', 'msp430_intrcc', 'avr_intrcc', 'avr_signalcc', 'ptx_kernel', 'ptx_device', 'spir_func', 'spir_kernel', 'intel_ocl_bicc', 'x86_64_sysvcc', 'win64cc', 'x86_fastcallcc', 'x86_stdcallcc', 'x86_thiscallcc', 'x86_vectorcallcc', 'x86_intrcc', 'amdgpu_vs', 'amdgpu_gs', 'amdgpu_ps', 'amdgpu_cs', 'amdgpu_kernel', 'x86_regcallcc', 'amdgpu_hs', 'msp430_builtincc', 'amdgpu_ls', 'amdgpu_es', 'aarch64_vfpcc', 'aarch64_sve_vfpcc', 'wasm_emscripten_invokecc', 'amdgpu_gfx', 'm68k_intrcc'];
        if (parser.match('id') && cconvKeywords.includes(parser._token.value)) {
            op.attributes.push({ name: 'CConv', value: parser.expect('id') });
        }
        parser.parseSymbolName('sym_name', op.attributes);
        const type = {};
        type.inputs = parser.parseFunctionArgumentList();
        if (parser.accept('->')) {
            type.results = parser.parseFunctionResultList();
        } else {
            type.results = [];
        }
        op.attributes.push({ name: 'function_type', value: type });
        if (parser.accept('id', 'vscale_range')) {
            parser.expect('(');
            const minRange = parser.expect();
            parser.expect(',');
            const maxRange = parser.expect();
            parser.expect(')');
            op.attributes.push({ name: 'vscale_range', value: `(${minRange}, ${maxRange})` });
        }
        if (parser.accept('id', 'comdat')) {
            parser.expect('(');
            const comdat = parser.expect('@');
            parser.expect(')');
            op.attributes.push({ name: 'comdat', value: comdat });
        }
        parser.parseOptionalAttrDictWithKeyword(op.attributes);
        if (parser.match('{')) {
            const region = {};
            parser.parseRegion(region);
            op.regions.push(region);
        }
        return true;
    }
};

mlir.StdxDialect = class extends mlir.Dialect {

    constructor(operations) {
        super('stdx', operations);
    }
};

mlir.VMDialect = class extends mlir.Dialect {

    constructor(operations) {
        super('vm', operations);
    }
};

mlir.MathDialect = class extends mlir.Dialect {

    constructor(operations) {
        super('math', operations);
    }
};

mlir.TMTensorDialect = class extends mlir.Dialect {

    constructor(operations) {
        super('tm_tensor', operations);
    }
};

mlir.MLProgramDialect = class extends mlir.Dialect {

    constructor(operations) {
        super('ml_program', operations);
    }
};

mlir.IREEGPUDialect = class extends mlir.Dialect {

    constructor(operations) {
        super('iree_gpu', operations);
    }
};

mlir.TFDeviceDialect = class extends mlir.Dialect {

    constructor(operations) {
        super('tf_device', operations);
    }
};

mlir.TFExecutorDialect = class extends mlir.Dialect {

    constructor(operations) {
        super('tf_executor', operations);
    }

    parseType(parser, dialectType) {
        if (dialectType === '!tf_executor.control' || dialectType === '!tf_executor.token') {
            return dialectType;
        }
        return null;
    }

    parseOperation(parser, opName, op) {
        const name = opName.replace(/^"|"$/g, '');
        if (name === 'tf_executor.graph') {
            return this._parseGraphOp(parser, op);
        }
        if (name === 'tf_executor.island') {
            return this._parseIslandOp(parser, op);
        }
        return super.parseOperation(parser, opName, op);
    }

    _parseGraphOp(parser, op) {
        if (parser.match('{')) {
            const region = {};
            parser.parseRegion(region);
            op.regions.push(region);
            if (region.blocks && region.blocks.length > 0) {
                const [block] = region.blocks;
                if (block.operations && block.operations.length > 0) {
                    const lastOp = block.operations[block.operations.length - 1];
                    if (lastOp.name === 'tf_executor.fetch' && lastOp.operands) {
                        for (const operand of lastOp.operands) {
                            if (operand.type && operand.type !== '!tf_executor.control') {
                                op.results.push({ type: operand.type });
                            }
                        }
                    }
                }
            }
        }
        parser.parseOptionalAttrDict(op.attributes);
        return true;
    }

    _parseIslandOp(parser, op) {
        // Parse: tf_executor.island wraps "tf.SomeOp"(...) {...} : (...) -> (...)
        // or: tf_executor.island {...}
        if (parser.accept('id', 'wraps')) {
            // Parse the wrapped operation
            const wrappedOp = parser.parseGenericOperation();
            op.attributes.push({ name: 'wrappedOp', value: wrappedOp });
            // Note: The island's results are already set by parseOperation from the LHS
            // We just need to ensure there's a control token type for the last result
            // The wrapped operation's result types should match the island's first N-1 results
        } else if (parser.match('{')) {
            // Parse region-based island
            const region = {};
            parser.parseRegion(region);
            op.regions.push(region);
        }
        parser.parseOptionalAttrDict(op.attributes);
        return true;
    }
};

mlir.TFFrameworkDialect = class extends mlir.Dialect {

    constructor(operations) {
        super('tf_framework', operations);
    }
};

mlir.TFRDialect = class extends mlir.Dialect {

    constructor(operations) {
        super('tfr', operations);
    }
};

mlir.TFRTDialect = class extends mlir.Dialect {

    constructor(operations) {
        super('tfrt', operations);
    }

    parseType(parser, dialectType) {
        const simpleTypes = ['chain', 'string', 'dist_context'];
        for (const simpleType of simpleTypes) {
            if (dialectType === `!tfrt.${simpleType}`) {
                return dialectType;
            }
        }
        if (dialectType.startsWith('!tfrt.tensor')) {
            if (parser.match('<')) {
                const content = parser.skip('<', '>');
                return dialectType + content;
            }
            return dialectType;
        }
        return null;
    }
};

mlir.TileDialect = class extends mlir.Dialect {

    constructor(operations) {
        super('tile', operations);
    }

    parseOperation(parser, opName, op) {
        const name = opName.replace(/^"|"$/g, '');

        // tile.contract has format: tile.contract agg, combo, operands... attributes : types -> result
        // Example: %1 = tile.contract add, mul, %0, %arg0, %arg1 {sink = #map0, srcs = [#map1, #map2]} : tensor<f32>, tensor<1x256xf32>, tensor<256x512xf32> -> tensor<1x512xf32>
        if (name === 'tile.contract') {
            // Parse aggregation kind (add, mul, etc.)
            if (parser.match('id')) {
                const agg = parser.expect('id');
                op.attributes.push({ name: 'agg', value: agg });
            }
            parser.accept(',');
            // Parse combination kind (add, mul, etc.)
            if (parser.match('id')) {
                const combo = parser.expect('id');
                op.attributes.push({ name: 'combo', value: combo });
            }
            parser.accept(',');
            // Parse operands
            op.operands = parser.parseArguments();
            // Parse attributes
            if (parser.match('{')) {
                parser.parseAttributeDict(op.attributes);
            }
            // Parse types
            if (parser.accept(':')) {
                parser.parseArgumentTypes(op.operands);
            }
            if (parser.accept('->')) {
                parser.parseArgumentTypes(op.results);
            }
            return true;
        }
        parser.parseGenericOperationAfterOpName(op);
        return true;
    }
};

mlir.ToyDialect = class extends mlir.Dialect {

    constructor(operations) {
        super('toy', operations);
    }

    parseOperation(parser, opName, op) {
        const name = opName.replace(/^"|"$/g, '');
        if (name === 'toy.func') {
            return this._parseFuncOp(parser, op);
        }
        if (name === 'toy.constant') {
            const value = parser.parseValue();
            op.attributes.push({ name: 'value', value: value.value === undefined ? value : value.value });
            if (parser.accept(':')) {
                parser.parseArgumentTypes(op.results);
            }
            return true;
        }
        if (name === 'toy.mul' || name === 'toy.add') {
            op.operands = parser.parseArguments();
            parser.parseOptionalAttrDictWithKeyword(op.attributes);
            if (parser.accept(':')) {
                // The type signature can be either:
                // 1. Simple: `: tensor<*xf64>` - just the result type
                // 2. Function-style: `: (tensor<*xf64>, tensor<*xf64>) -> tensor<*xf64>`
                const type = parser.parseType();
                if (type.startsWith('(') && type.includes('->')) {
                    // Function-style type signature
                    const parts = type.match(/\((.*?)\)\s*->\s*(.+)/);
                    if (parts) {
                        const inputTypes = parts[1].split(',').map((t) => t.trim());
                        for (let i = 0; i < op.operands.length && i < inputTypes.length; i++) {
                            op.operands[i].type = inputTypes[i];
                        }
                        // Add result type to existing result (which already has SSA name)
                        if (op.results.length > 0) {
                            op.results[0].type = parts[2].trim();
                        }
                    }
                } else {
                    // Simple type signature - type applies to both operands and result
                    for (const operand of op.operands) {
                        operand.type = type;
                    }
                    if (op.results.length > 0) {
                        op.results[0].type = type;
                    }
                }
            }
            return true;
        }
        if (name === 'toy.transpose' || name === 'toy.reshape') {
            if (parser.accept('(')) {
                op.operands = parser.parseArguments();
                if (parser.accept(':')) {
                    parser.parseArgumentTypes(op.operands);
                }
                parser.accept(')');
            }
            parser.parseOptionalAttrDictWithKeyword(op.attributes);
            if (parser.accept('id', 'to')) {
                parser.parseArgumentTypes(op.results);
            }
            return true;
        }
        if (name === 'toy.return') {
            if (!parser.match('id', 'attributes') && parser.match('%')) {
                op.operands = parser.parseArguments();
                if (parser.accept(':')) {
                    parser.parseArgumentTypes(op.operands);
                }
            }
            parser.parseOptionalAttrDictWithKeyword(op.attributes);
            return true;
        }
        if (name === 'toy.generic_call') {
            parser.parseSymbolName('callee', op.attributes);
            if (parser.accept('(')) {
                op.operands = parser.parseArguments();
                parser.accept(')');
            }
            parser.parseOptionalAttrDictWithKeyword(op.attributes);
            if (parser.accept(':')) {
                const type = parser.parseType();
                if (type.startsWith('(') && type.includes('->')) {
                    const parts = type.match(/\((.*?)\)\s*->\s*(.+)/);
                    if (parts) {
                        const inputTypes = parts[1].split(',').map((t) => t.trim());
                        for (let i = 0; i < op.operands.length && i < inputTypes.length; i++) {
                            op.operands[i].type = inputTypes[i];
                        }
                        // Add result type to existing result (which already has SSA name)
                        if (op.results.length > 0) {
                            op.results[0].type = parts[2].trim();
                        }
                    }
                }
            }
            return true;
        }
        if (name === 'toy.print') {
            op.operands = parser.parseArguments();
            parser.parseOptionalAttrDictWithKeyword(op.attributes);
            if (parser.accept(':')) {
                parser.parseArgumentTypes(op.operands);
            }
            return true;
        }
        return super.parseOperation(parser, opName, op);
    }

    _parseFuncOp(parser, op) {
        parser.parseOptionalVisibilityKeyword(op.attributes);
        parser.parseSymbolName('sym_name', op.attributes);
        const type = {};
        type.inputs = parser.parseFunctionArgumentList();
        parser.parseOptionalAttrDictWithKeyword(op.attributes);
        type.results = [];
        if (parser.accept('->')) {
            for (const result of parser.parseFunctionResultList()) {
                type.results.push(result);
            }
        }
        op.attributes.push({ name: 'function_type', value: type });
        parser.parseOptionalAttrDictWithKeyword(op.attributes);
        if (parser.match('{')) {
            const region = {};
            parser.parseRegion(region);
            op.regions.push(region);
        }
        if (op.regions.length > 0) {
            const region = op.regions[op.regions.length - 1];
            if (region.blocks.length > 0) {
                const block = region.blocks[region.blocks.length - 1];
                if (block.operations.length > 0) {
                    const lastOp = block.operations[block.operations.length - 1];
                    type.results = lastOp.operands;
                    block.operations.pop();
                }
            }
        }
        return true;
    }
};

mlir.SdfgDialect = class extends mlir.Dialect {

    constructor(operations) {
        super('sdfg', operations);
    }

    parseType(parser, dialectType) {
        if (dialectType === '!sdfg.array' || dialectType === '!sdfg.stream' ||
            dialectType === '!sdir.array' || dialectType === '!sdir.stream' || dialectType === '!sdir.stream_array') {
            if (parser.match('<')) {
                const content = parser.skip('<', '>');
                return dialectType + content;
            }
            return dialectType;
        }
        return null;
    }

    parseOperation(parser, opName, op) {
        const name = opName.replace(/^"|"$/g, '');
        if (name === 'sdfg.sdfg' || name === 'sdfg.nested_sdfg' || name === 'sdir.sdfg') {
            parser.parseOptionalAttrDict(op.attributes);
            const type = {};
            type.inputs = parser.parseFunctionArgumentList();
            parser.parseOptionalAttrDictWithKeyword(op.attributes);
            type.results = [];
            if (parser.accept('->')) {
                type.results = parser.parseFunctionArgumentList();
            }
            op.attributes.push({ name: 'function_type', value: type });
            parser.parseOptionalAttrDictWithKeyword(op.attributes);
            if (parser.match('{')) {
                const region = {};
                parser.parseRegion(region);
                op.regions.push(region);
            }
            return true;
        }
        if (name === 'sdfg.tasklet' || name === 'sdir.tasklet') {
            parser.parseOptionalAttrDict(op.attributes);
            op.sym_name = parser.parseOptionalSymbolName();
            if (parser.match('(')) {
                op.operands = parser.parseArguments();
            }
            parser.parseOptionalAttrDictWithKeyword(op.attributes);
            if (parser.accept('->')) {
                if (parser.accept('(')) {
                    while (!parser.accept(')')) {
                        const type = parser.parseType();
                        op.results.push({ type });
                        parser.accept(',');
                    }
                } else {
                    const type = parser.parseType();
                    op.results.push({ type });
                }
            }
            parser.parseOptionalAttrDictWithKeyword(op.attributes);
            if (parser.match('{')) {
                const region = {};
                parser.parseRegion(region);
                op.regions.push(region);
            }
            return true;
        }
        if (name === 'sdfg.map' || name === 'sdfg.consume') {
            if (parser.match('(')) {
                op.operands = parser.parseArguments();
            }
            parser.parseOptionalAttrDictWithKeyword(op.attributes);
            if (parser.accept('->')) {
                if (parser.accept('(')) {
                    while (!parser.accept(')')) {
                        const type = parser.parseType();
                        op.results.push({ type });
                        parser.accept(',');
                    }
                } else {
                    const type = parser.parseType();
                    op.results.push({ type });
                }
            }
            parser.parseOptionalAttrDictWithKeyword(op.attributes);
            if (parser.match('{')) {
                const region = {};
                parser.parseRegion(region);
                op.regions.push(region);
            }
            return true;
        }
        if (name === 'sdfg.state' || name === 'sdir.state') {
            parser.parseOptionalAttrDict(op.attributes);
            op.sym_name = parser.parseOptionalSymbolName();
            parser.parseOptionalAttrDictWithKeyword(op.attributes);
            const region = {};
            parser.parseRegion(region);
            op.regions.push(region);
            return true;
        }
        if (name === 'sdfg.alloc' || name === 'sdir.alloc' || name === 'sdir.alloc_transient' || name === 'sdir.alloc_stream') {
            parser.parseOptionalAttrDict(op.attributes);
            if (parser.match('(')) {
                op.operands = parser.parseArguments();
            }
            parser.parseOptionalAttrDictWithKeyword(op.attributes);
            if (parser.accept(':')) {
                const type = parser.parseType();
                op.results.push({ type });
            }
            return true;
        }
        if (name === 'sdfg.store' || name === 'sdir.store') {
            parser.parseOptionalAttrDict(op.attributes);
            const value = parser.expect('%');
            op.operands.push({ value });
            parser.accept(',');
            const array = parser.expect('%');
            op.operands.push({ value: array });
            if (parser.accept('[')) {
                while (!parser.match(']')) {
                    if (parser.match('%')) {
                        const idx = parser.expect('%');
                        op.operands.push({ value: idx });
                    } else {
                        parser.expect();
                    }
                    if (parser.match(',')) {
                        parser.accept(',');
                    }
                }
                parser.accept(']');
            }
            parser.parseOptionalAttrDictWithKeyword(op.attributes);
            if (parser.accept(':')) {
                const valueType = parser.parseType();
                parser.accept('->');
                const arrayType = parser.parseType();
                if (op.operands.length > 0) {
                    op.operands[0].type = valueType;
                }
                if (op.operands.length > 1) {
                    op.operands[1].type = arrayType;
                }
            }
            return true;
        }
        if (name === 'sdfg.load' || name === 'sdir.load') {
            parser.parseOptionalAttrDict(op.attributes);
            const array = parser.expect('%');
            op.operands.push({ value: array });
            if (parser.accept('[')) {
                while (!parser.match(']')) {
                    if (parser.match('%')) {
                        const idx = parser.expect('%');
                        op.operands.push({ value: idx });
                    } else {
                        parser.expect();
                    }
                    if (parser.match(',')) {
                        parser.accept(',');
                    }
                }
                parser.accept(']');
            }
            parser.parseOptionalAttrDictWithKeyword(op.attributes);
            if (parser.accept(':')) {
                const arrayType = parser.parseType();
                parser.accept('->');
                const resultType = parser.parseType();
                if (op.operands.length > 0) {
                    op.operands[0].type = arrayType;
                }
                op.results.push({ type: resultType });
            }
            return true;
        }
        if (name === 'sdfg.map' || name === 'sdir.map') {
            parser.parseOptionalAttrDict(op.attributes);
            const params = [];
            if (parser.accept('(')) {
                while (!parser.accept(')')) {
                    if (parser.match('%')) {
                        const param = parser.expect('%');
                        params.push(param);
                    }
                    if (parser.match(',')) {
                        parser.accept(',');
                    }
                }
            }
            if (parser.accept('=')) {
                if (parser.accept('(')) {
                    while (!parser.accept(')')) {
                        parser.expect();
                    }
                }
            }
            if (parser.match('to')) {
                parser.accept('to');
                if (parser.accept('(')) {
                    while (!parser.accept(')')) {
                        parser.expect();
                    }
                }
            }
            if (parser.match('step')) {
                parser.accept('step');
                if (parser.accept('(')) {
                    while (!parser.accept(')')) {
                        parser.expect();
                    }
                }
            }
            parser.parseOptionalAttrDictWithKeyword(op.attributes);
            if (parser.match('{')) {
                const region = parser.parseRegion();
                op.regions.push(region);
            }
            return true;
        }
        if (name === 'sdfg.consume' || name === 'sdir.consume') {
            parser.parseOptionalAttrDict(op.attributes);
            if (parser.accept('(')) {
                op.operands = parser.parseArguments();
            }
            if (parser.accept('->')) {
                if (parser.accept('(')) {
                    while (!parser.accept(')')) {
                        parser.expect();
                    }
                }
            }
            parser.parseOptionalAttrDictWithKeyword(op.attributes);
            if (parser.match('{')) {
                const region = parser.parseRegion();
                op.regions.push(region);
            }
            return true;
        }
        if (name === 'sdfg.edge' || name === 'sdir.edge') {
            parser.parseOptionalAttrDict(op.attributes);
            if (parser.match('(')) {
                op.operands = parser.parseArguments();
            }
            parser.parseOptionalAttrDictWithKeyword(op.attributes);
            if (parser.match('@')) {
                const src = parser.expect('@');
                op.attributes.push({ name: 'src', value: src });
            }
            parser.accept('->');
            if (parser.match('@')) {
                const dst = parser.expect('@');
                op.attributes.push({ name: 'dst', value: dst });
            }
            return true;
        }
        if (name === 'sdfg.sym' || name === 'sdir.sym') {
            if (parser.accept('(')) {
                const expr = parser.expect('string');
                op.attributes.push({ name: 'expr', value: expr });
                parser.accept(')');
            }
            parser.parseOptionalAttrDictWithKeyword(op.attributes);
            if (parser.accept(':')) {
                const type = parser.parseType();
                op.results.push({ type });
            }
            return true;
        }
        if (name === 'sdfg.copy' || name === 'sdir.copy') {
            parser.parseOptionalAttrDictWithKeyword(op.attributes);
            op.operands = parser.parseArguments();
            if (parser.accept('->')) {
                const dst = parser.parseArguments();
                op.operands.push(...dst);
            }
            if (parser.accept(':')) {
                const type = parser.parseType();
                for (const operand of op.operands) {
                    if (!operand.type) {
                        operand.type = type;
                    }
                }
            }
            return true;
        }
        if (name === 'sdfg.libcall' || name === 'sdir.libcall') {
            parser.parseOptionalAttrDictWithKeyword(op.attributes);
            if (parser.match('string')) {
                const libname = parser.expect('string');
                op.attributes.push({ name: 'libname', value: libname });
            }
            op.operands = parser.parseArguments();
            if (parser.accept(':')) {
                if (parser.accept('(')) {
                    while (!parser.accept(')')) {
                        parser.parseType();
                        parser.accept(',');
                    }
                }
                if (parser.accept('->')) {
                    const resultType = parser.parseType();
                    op.results.push({ type: resultType });
                }
            }
            return true;
        }
        if (name === 'sdir.get_access') {
            op.operands = parser.parseArguments();
            return true;
        }
        if (name === 'sdir.call') {
            const callee = parser.parseOptionalSymbolName();
            if (callee) {
                op.attributes.push({ name: 'callee', value: callee });
            }
            if (parser.match('(')) {
                op.operands = parser.parseArguments();
            }
            if (parser.accept(':')) {
                if (parser.accept('(')) {
                    while (!parser.accept(')')) {
                        parser.parseType();
                        parser.accept(',');
                    }
                }
                if (parser.accept('->')) {
                    const resultType = parser.parseType();
                    op.results.push({ type: resultType });
                }
            }
            return true;
        }
        if (name === 'sdfg.alloc_symbol' || name === 'sdir.alloc_symbol') {
            if (parser.accept('(')) {
                const sym = parser.expect('string');
                op.attributes.push({ name: 'sym', value: sym });
                parser.accept(')');
            }
            return true;
        }
        if (name === 'sdfg.return') {
            if (parser.match('%')) {
                op.operands = parser.parseArguments();
                if (parser.accept(':')) {
                    parser.parseArgumentTypes(op.operands);
                }
            }
            return true;
        }
        if (name === 'sdfg.stream_push' || name === 'sdir.stream_push') {
            parser.parseOptionalAttrDict(op.attributes);
            const value = parser.expect('%');
            op.operands.push({ value });
            parser.accept(',');
            const stream = parser.expect('%');
            op.operands.push({ value: stream });
            parser.parseOptionalAttrDictWithKeyword(op.attributes);
            if (parser.accept(':')) {
                const valueType = parser.parseType();
                parser.accept('->');
                const streamType = parser.parseType();
                if (op.operands.length > 0) {
                    op.operands[0].type = valueType;
                }
                if (op.operands.length > 1) {
                    op.operands[1].type = streamType;
                }
            }
            return true;
        }
        if (name === 'sdfg.stream_pop' || name === 'sdir.stream_pop') {
            parser.parseOptionalAttrDict(op.attributes);
            const stream = parser.expect('%');
            op.operands.push({ value: stream });
            parser.parseOptionalAttrDictWithKeyword(op.attributes);
            if (parser.accept(':')) {
                const streamType = parser.parseType();
                parser.accept('->');
                const resultType = parser.parseType();
                if (op.operands.length > 0) {
                    op.operands[0].type = streamType;
                }
                op.results.push({ type: resultType });
            }
            return true;
        }
        if (name === 'sdfg.stream_length' || name === 'sdir.stream_length') {
            parser.parseOptionalAttrDict(op.attributes);
            const stream = parser.expect('%');
            op.operands.push({ value: stream });
            parser.parseOptionalAttrDictWithKeyword(op.attributes);
            if (parser.accept(':')) {
                const streamType = parser.parseType();
                parser.accept('->');
                const resultType = parser.parseType();
                if (op.operands.length > 0) {
                    op.operands[0].type = streamType;
                }
                op.results.push({ type: resultType });
            }
            return true;
        }
        if (name === 'sdfg.view_cast' || name === 'sdir.view_cast') {
            parser.parseOptionalAttrDict(op.attributes);
            const input = parser.expect('%');
            op.operands.push({ value: input });
            parser.parseOptionalAttrDictWithKeyword(op.attributes);
            if (parser.accept(':')) {
                const inputType = parser.parseType();
                parser.accept('->');
                const resultType = parser.parseType();
                if (op.operands.length > 0) {
                    op.operands[0].type = inputType;
                }
                op.results.push({ type: resultType });
            }
            return true;
        }
        if (name === 'sdfg.subview' || name === 'sdir.subview') {
            parser.parseOptionalAttrDict(op.attributes);
            const input = parser.expect('%');
            op.operands.push({ value: input });
            while (parser.accept('[')) {
                while (!parser.accept(']')) {
                    parser.expect();
                }
            }
            parser.parseOptionalAttrDictWithKeyword(op.attributes);
            if (parser.accept(':')) {
                const inputType = parser.parseType();
                parser.accept('->');
                const resultType = parser.parseType();
                if (op.operands.length > 0) {
                    op.operands[0].type = inputType;
                }
                op.results.push({ type: resultType });
            }
            return true;
        }
        return super.parseOperation(parser, opName, op);
    }
};

mlir.TFDialect = class extends mlir.Dialect {

    constructor(operations) {
        super('tf', operations);
    }

    parseType(parser, dialectType) {
        if (dialectType === '!tf.resource') {
            if (parser.match('<')) {
                const content = parser.skip('<', '>');
                return dialectType + content;
            }
            return dialectType;
        }
        if (dialectType === '!tf.variant') {
            if (parser.match('<')) {
                const content = parser.skip('<', '>');
                return dialectType + content;
            }
            return dialectType;
        }
        const simpleTypes = ['string', 'control'];
        for (const simpleType of simpleTypes) {
            if (dialectType === `!tf.${simpleType}`) {
                return dialectType;
            }
        }
        return null;
    }
};

mlir.TFTypeDialect = class extends mlir.Dialect {

    constructor(operations) {
        super('tf_type', operations);
    }

    parseType(parser, dialectType) {
        const simpleTypes = [
            'string', 'resource', 'variant',
            'qint8', 'qint16', 'qint32', 'quint8', 'quint16',
            'f32ref', 'f64ref', 'uint4ref', 'int4ref', 'uint8ref', 'int8ref',
            'uint16ref', 'int16ref', 'uint32ref', 'int32ref', 'uint64ref', 'int64ref',
            'stringref', 'boolref', 'quint8ref', 'qint8ref', 'quint16ref', 'qint16ref',
            'qint32ref', 'bfloat16ref', 'complex64ref', 'complex128ref', 'halfref',
            'resourceref', 'variantref',
            'float8e4m3fnref', 'float8e5m2ref', 'float8e4m3fnuzref',
            'float8e4m3b11fnuzref', 'float8e5m2fnuzref'
        ];
        for (const simpleType of simpleTypes) {
            if (dialectType === `!tf_type.${simpleType}`) {
                if (!parser.match('<')) {
                    return dialectType;
                }
            }
        }
        if (dialectType.startsWith('!tf_type.resource') || dialectType.startsWith('!tf_type.variant')) {
            if (parser.match('<')) {
                parser.expect('<');
                const subtypes = [];
                while (!parser.match('>')) {
                    subtypes.push(parser.parseType());
                    if (parser.match(',')) {
                        parser.expect(',');
                    }
                }
                parser.expect('>');
                return `${dialectType}<${subtypes.join(', ')}>`;
            }
            return dialectType;
        }
        return null;
    }
};

mlir.CheckDialect = class extends mlir.Dialect {
    constructor(operations) {
        super('check', operations);
    }

    parseOperation(parser, opName, op) {
        const name = opName.replace(/^"|"$/g, '');
        if (name === 'check.expect_eq_const' || name === 'check.expect_almost_eq_const' ||
            name === 'check.expect_eq' || name === 'check.expect_almost_eq' ||
            name === 'check.expect_close' || name === 'check.expect_serialized_eq') {
            op.operands = parser.parseArguments();
            if (parser.accept(',')) {
                const expectedValue = parser.parseValue();
                op.attributes.push({ name: 'expected', value: expectedValue.value });
            }
            if (parser.accept(':')) {
                parser.parseArgumentTypes(op.operands);
            }
            if (parser.match('{')) {
                parser.parseAttributeDict(op.attributes);
            }
            return true;
        }
        return super.parseOperation(parser, opName, op);
    }
};

mlir.TransformDialect = class extends mlir.Dialect {

    constructor(operations) {
        super('transform', operations);
    }

    parseType(parser, dialectType) {
        const simpleTypes = ['any_op', 'any_param', 'any_value', 'op', 'param', 'affine_map', 'type'];
        for (const simpleType of simpleTypes) {
            if (dialectType === `!transform.${simpleType}`) {
                return dialectType;
            }
        }
        return null;
    }

    parseOperation(parser, opName, op) {
        const name = opName.replace(/^"|"$/g, '');
        if (name === 'transform.named_sequence') {
            return this._parseNamedSequenceOp(parser, op);
        }
        if (name === 'transform.structured.match') {
            return this._parseStructuredMatchOp(parser, op);
        }
        if (name === 'transform.get_result') {
            return this._parseGetResultOp(parser, op);
        }
        if (name.startsWith('transform.test.')) {
            return this._parseTestOp(parser, name, op);
        }
        const hasSchema = super.parseOperation(parser, opName, op);
        if (hasSchema) {
            return true;
        }
        parser.parseGenericOperationAfterOpName(op);
        return true;
    }

    _parseNamedSequenceOp(parser, op) {
        parser.parseOptionalVisibilityKeyword(op.attributes);
        parser.parseSymbolName('sym_name', op.attributes);
        const type = {};
        type.inputs = parser.parseFunctionArgumentList();
        if (parser.accept('->')) {
            type.results = parser.parseFunctionResultList();
        } else {
            type.results = [];
        }
        op.attributes.push({ name: 'function_type', value: type });
        parser.parseOptionalAttrDictWithKeyword(op.attributes);
        if (parser.match('{')) {
            const region = {};
            parser.parseRegion(region);
            op.regions.push(region);
        }
        return true;
    }

    _parseStructuredMatchOp(parser, op) {
        // Parse: ops{["foo"]} in %arg0 : (!transform.any_op) -> !transform.any_op
        // or: ops{["foo", "bar"]} in %arg0 : (!transform.any_op) -> !transform.any_op
        if (parser.match('id', 'ops')) {
            parser.expect('id', 'ops');
            if (parser.match('{')) {
                parser.expect('{');
                const opsArray = parser.parseValue();
                op.attributes.push({ name: 'ops', value: opsArray.value });
                parser.expect('}');
            }
        }
        if (parser.accept('id', 'in')) {
            // Parse operands after 'in' keyword
            while (parser.match('%')) {
                const value = parser.expect('%');
                op.operands.push({ value });
                if (!parser.accept(',')) {
                    break;
                }
            }
        }
        if (parser.accept(':')) {
            // Parse function type: (input_types) -> result_type
            if (parser.accept('(')) {
                const inputTypes = [];
                while (!parser.match(')')) {
                    inputTypes.push(parser.parseType());
                    parser.accept(',');
                }
                parser.expect(')');
                // Apply input types to operands
                for (let i = 0; i < Math.min(inputTypes.length, op.operands.length); i++) {
                    op.operands[i].type = inputTypes[i];
                }
            }
            if (parser.accept('->')) {
                const resultType = parser.parseType();
                op.results.push({ type: resultType });
            }
        }
        parser.parseOptionalAttrDict(op.attributes);
        return true;
    }

    _parseGetResultOp(parser, op) {
        // Parse: transform.get_result %op[0] : (!transform.any_op) -> !transform.any_value
        if (parser.match('%')) {
            const value = parser.expect('%');
            op.operands.push({ value });
        }
        // Parse result index [N]
        if (parser.accept('[')) {
            const index = parser.expect('int');
            op.attributes.push({ name: 'result_number', value: parseInt(index, 10) });
            parser.expect(']');
        }
        // Parse type signature
        if (parser.accept(':')) {
            if (parser.accept('(')) {
                const inputTypes = [];
                while (!parser.match(')')) {
                    inputTypes.push(parser.parseType());
                    parser.accept(',');
                }
                parser.expect(')');
                for (let i = 0; i < Math.min(inputTypes.length, op.operands.length); i++) {
                    op.operands[i].type = inputTypes[i];
                }
            }
            if (parser.accept('->')) {
                const resultType = parser.parseType();
                op.results.push({ type: resultType });
            }
        }
        parser.parseOptionalAttrDict(op.attributes);
        return true;
    }

    _parseTestOp(parser, name, op) {
        // Parse test operations like: transform.test.move_operand_deps %op1 before %op2 : types
        // or: transform.test.move_value_defns %v1, %v2 before %op3 : types
        const keywords = ['before', 'after', 'into'];
        let foundKeyword = false;

        // Parse operands and keywords
        while (parser.match('%') || (parser.match('id') && !foundKeyword)) {
            if (parser.match('%')) {
                const value = parser.expect('%');
                op.operands.push({ value });
                parser.accept(',');
            } else if (parser.match('id')) {
                const id = parser._token.value;
                if (keywords.includes(id)) {
                    const keyword = parser.expect('id');
                    op.attributes.push({ name: 'keyword', value: keyword });
                    foundKeyword = true;
                } else {
                    break;
                }
            }
        }

        // Parse type list - handle both regular and grouped syntax
        if (parser.accept(':')) {
            // Check if types are grouped with parentheses: (type1, type2), type3
            if (parser.match('(')) {
                parser.expect('(');
                let idx = 0;
                while (!parser.match(')')) {
                    const type = parser.parseType();
                    if (idx < op.operands.length) {
                        op.operands[idx].type = type;
                    }
                    idx++;
                    parser.accept(',');
                }
                parser.expect(')');
                // Parse remaining types after comma
                while (parser.accept(',')) {
                    const type = parser.parseType();
                    if (idx < op.operands.length) {
                        op.operands[idx].type = type;
                    }
                    idx++;
                }
            } else {
                // Standard type list
                parser.parseArgumentTypes(op.operands);
            }
        }
        parser.parseOptionalAttrDict(op.attributes);
        return true;
    }
};

mlir.TestDialect = class extends mlir.Dialect {
    constructor(operations) {
        super('test', operations);
    }

    parseOperation(parser, opName, op) {
        const hasSchema = super.parseOperation(parser, opName, op);
        if (hasSchema) {
            return true;
        }
        parser.parseGenericOperationAfterOpName(op);
        return true;
    }
};

mlir.TritonDialect = class extends mlir.Dialect {

    constructor(operations) {
        super('tt', operations);
    }

    parseType(parser, dialectType) {
        if (dialectType.startsWith('!tt.')) {
            if (parser.match('<')) {
                const content = parser.skip('<', '>');
                return dialectType + content;
            }
        }
        return dialectType;
    }

    parseOperation(parser, opName, op) {
        const name = opName.replace(/^"|"$/g, '');
        if (name === 'tt.func') {
            parser.parseOptionalVisibilityKeyword(op.attributes);
            parser.parseSymbolName('sym_name', op.attributes);
            const type = {};
            type.inputs = parser.parseFunctionArgumentList();
            parser.parseOptionalAttrDictWithKeyword(op.attributes);
            type.results = [];
            if (parser.accept('->')) {
                for (const result of parser.parseFunctionResultList()) {
                    type.results.push(result);
                }
            }
            op.attributes.push({ name: 'function_type', value: type });
            parser.parseOptionalAttrDictWithKeyword(op.attributes);
            if (parser.match('{')) {
                const region = {};
                parser.parseRegion(region);
                op.regions.push(region);
            }
            if (op.regions.length > 0) {
                const region = op.regions[op.regions.length - 1];
                if (region.blocks.length > 0) {
                    const block = region.blocks[region.blocks.length - 1];
                    if (block.operations.length > 0) {
                        const lastOp = block.operations[block.operations.length - 1];
                        if (lastOp.name === 'tt.return') {
                            type.results = lastOp.operands;
                            block.operations.pop();
                        }
                    }
                }
            }
            return true;
        }
        if (name === 'tt.return') {
            op.operands = parser.parseArguments();
            if (parser.accept(':')) {
                parser.parseArgumentTypes(op.operands);
            }
            return true;
        }
        return super.parseOperation(parser, opName, op);
    }
};

mlir.TritonGPUDialect = class extends mlir.Dialect {

    constructor(operations) {
        super('ttg', operations);
    }

    parseType(parser, dialectType) {
        if (dialectType.startsWith('!ttg.')) {
            if (parser.match('<')) {
                const content = parser.skip('<', '>');
                return dialectType + content;
            }
            return dialectType;
        }
        return null;
    }
};

mlir.GluonDialect = class extends mlir.Dialect {

    constructor(operations) {
        super('gluon', operations);
    }
};

mlir.TritonNvidiaGPUDialect = class extends mlir.Dialect {

    constructor(operations) {
        super('ttng', operations);
    }

    parseType(parser, dialectType) {
        if (dialectType.startsWith('!ttng.')) {
            if (parser.match('<')) {
                const content = parser.skip('<', '>');
                return dialectType + content;
            }
            return dialectType;
        }
        return null;
    }
};

mlir.ProtonDialect = class extends mlir.Dialect {

    constructor(operations) {
        super('proton', operations);
    }
};

mlir.MichelsonDialect = class extends mlir.Dialect {

    constructor(operations) {
        super('michelson', operations);
    }

    parseType(parser, dialectType) {
        const simpleTypes = ['int', 'bytes', 'operation', 'nat', 'string', 'unit', 'bool', 'mutez', 'timestamp', 'address', 'key', 'signature', 'chain_id', 'key_hash'];
        for (const simpleType of simpleTypes) {
            if (dialectType === `!michelson.${simpleType}`) {
                return dialectType;
            }
        }
        if (dialectType.startsWith('!michelson.pair') || dialectType.startsWith('!michelson.list') ||
            dialectType.startsWith('!michelson.option') || dialectType.startsWith('!michelson.or') ||
            dialectType.startsWith('!michelson.map') || dialectType.startsWith('!michelson.big_map') ||
            dialectType.startsWith('!michelson.set') || dialectType.startsWith('!michelson.contract') ||
            dialectType.startsWith('!michelson.lambda')) {
            if (parser.match('<')) {
                const content = parser.skip('<', '>');
                return dialectType + content;
            }
            return dialectType;
        }
        return null;
    }
};

mlir.Metadata = class {

    static async open(context) {
        if (!mlir.Metadata._metadata) {
            const data = await context.request('mlir-metadata.json');
            mlir.Metadata._metadata = new mlir.Metadata(data);
        }
        return mlir.Metadata._metadata;
    }

    constructor(data) {
        this.operations = new Map();
        if (data) {
            const operations = JSON.parse(data);
            for (const op of operations) {
                const metadata = { name: op.name };
                if (op.category) {
                    metadata.category = op.category;
                }
                if (op.summary) {
                    metadata.summary = op.summary;
                }
                if (op.description) {
                    metadata.description = op.description;
                }
                if (op.inputs) {
                    metadata.inputs = op.inputs;
                }
                if (op.outputs) {
                    metadata.outputs = op.outputs;
                }
                if (op.attributes) {
                    metadata.attributes = op.attributes;
                }
                if (op.assemblyFormat) {
                    metadata.assemblyFormat = op.assemblyFormat;
                }
                const [dialectName] = op.name.split('.');
                if (!this.operations.has(dialectName)) {
                    this.operations.set(dialectName, []);
                }
                this.operations.get(dialectName).push(metadata);
            }
        }
    }

    type(name) {
        const [dialectName] = name.split('.');
        const operations = this.operations.get(dialectName);
        if (operations) {
            const op = operations.find((op) => op.name === name);
            if (op) {
                return op;
            }
        }
        return { name };
    }
};

mlir.Error = class extends Error {

    constructor(message) {
        super(message);
        this.name = 'Error loading MLIR model.';
    }
};

export const ModelFactory = mlir.ModelFactory;
