
// Experimental

import * as base from './base.js';

const mlir = {};

mlir.ModelFactory = class {

    async match(context) {
        const stream = context.stream;
        const identifier = context.identifier;
        const extension = identifier.split('.').pop().toLowerCase();
        if (stream && stream.length > 4) {
            const buffer = stream.peek(4);
            const signature = String.fromCharCode.apply(null, buffer);
            if (signature === 'ML\xEFR') {
                return context.set('mlir.binary');
            }
        }
        try {
            const reader = await context.read('text', 0x10000);
            let whitespace = true;
            for (let line = reader.read('\n'); line !== undefined; line = reader.read('\n')) {
                if (/module\s+(@\w+|\w+|attributes|\{)/.test(line) ||
                    /tensor<[\w\d]+>/.test(line) ||
                    /func[.\s]*@\w+/.test(line) ||
                    /%\w+\s*=\s*"[\w.]+/.test(line) ||
                    /%\w+\s*=\s*\w+\./.test(line) ||
                    /!\w+\s*=\s*![\w.]+</.test(line) ||
                    /#\w+\s*=\s*#[\w.]+</.test(line) ||
                    /#\w+\s*=\s*loc\s*\(/.test(line) ||
                    /\w+\.\w+(?:\s+\w+)*\s+@\w+/.test(line) ||
                    /\w+\.\w+\s+#[\w.]+</.test(line) ||
                    /\w+\.\w+\s*<?\{/.test(line) ||
                    /:\s*![\w.]+/.test(line) ||
                    /(%\w+|\w{2,}|[)])\s*:\s*(\[|tensor<)/.test(line) ||
                    /->\s*(![\w.]+|\(|tensor<)/.test(line)) {
                    return context.set('mlir.text');
                }
                if (line && !line.trim().startsWith('//')) {
                    whitespace = false;
                }
            }
            if (extension === 'mlir' && whitespace) {
                return context.set('mlir.text');
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
                const state = new mlir.ParserState();
                const parser = new mlir.Parser(state, decoder, new mlir.DialectContext(metadata));
                const block = await parser.parse();
                return new mlir.Model(metadata, 'MLIR', '', block, state.attributeAliasDefinitions);
            }
            case 'mlir.binary': {
                const binary = await context.read('binary');
                const reader = new mlir.BytecodeReader(binary, new mlir.DialectContext(metadata));
                const block = reader.read();
                const format = `MLIR Bytecode v${reader.version}`;
                const producer = reader.producer;
                const model = new mlir.Model(metadata, format, producer, block, new Map());
                return model;
            }
            default: {
                throw new mlir.Error(`Unsupported MLIR format '${context.type}'.`);
            }
        }
    }
};

mlir.Model = class {

    constructor(metadata, format, producer, block, attributeAliasDefinitions) {
        this.format = format;
        this.producer = producer || '';
        this.modules = [];
        this.functions = [];
        this.metadata = [];
        const modules = [];
        const isFunc = (name) => name.endsWith('.func') || /\.func_v\d+$/.test(name);
        const isModule = (name) => name.endsWith('.module');
        const collectModules = (operations, path, attributes) => {
            let identifier = 0;
            const funcs = [];
            const ops = [];
            for (const op of operations) {
                if (isFunc(op.name)) {
                    funcs.push(op);
                } else if (isModule(op.name)) {
                    let name = op.getAttr('sym_name');
                    name = name ? name.value : `$${identifier++}`;
                    const modulePath = [...path, name];
                    for (const region of op.regions || []) {
                        for (const blk of region.blocks || []) {
                            collectModules(blk.operations || [], modulePath, op.getAttrDictionary());
                        }
                    }
                } else {
                    ops.push(op);
                }
            }
            if (funcs.length > 0 || ops.length > 0) {
                let name = null;
                if (attributes.get('sym_name')) {
                    name = attributes.get('sym_name');
                    name = `@${name.value}`;
                }
                modules.push({ path, symName: name, funcs, ops, attributes });
            }
        };
        collectModules(block.operations, [], new Map());
        const formatPrefix = (path, symName) => {
            if (symName) {
                return symName;
            }
            if (modules.length !== 1 && path.length > 0) {
                return path.map((path) => `${path}`).join('::');
            }
            return '';
        };
        const functions = new Map();
        let identifier = 0;
        for (const module of modules) {
            const prefix = formatPrefix(module.path, module.symName);
            for (const func of module.funcs) {
                const sym_name = func.getAttr('sym_name');
                const base = sym_name ? sym_name.value : `$${identifier}`;
                identifier++;
                const name = prefix ? `${prefix}::@${base}` : `@${base}`;
                functions.set(name, { func, prefix, base, module });
            }
        }
        const context = new mlir.Context(metadata, functions);
        for (const [name, info] of functions) {
            const graph = context.graph(info.func, name);
            this.functions.push(graph);
        }
        for (const module of modules) {
            if (module.ops.length > 0 || module.attributes.size > 0) {
                const name = formatPrefix(module.path, module.symName) || '';
                const op = new mlir.Operation('builtin.module');
                op.attributes = module.attributes;
                op.regions = [{ blocks: [{ operations: module.ops, arguments: [] }] }];
                const graph = context.graph(op, name);
                this.modules.push(graph);
            }
        }
        for (const [name, attribute] of attributeAliasDefinitions) {
            let value = attribute.type;
            if (!value) {
                value = typeof attribute.value === 'string' ? attribute.value : JSON.stringify(attribute.value);
            }
            const metadata = new mlir.Argument(name, value, 'attribute');
            this.metadata.push(metadata);
        }
    }
};

mlir.Graph = class {

    constructor(metadata, func, context, name) {
        this.name = name || '';
        if (!name && func.attributes.has('sym_name')) {
            const sym_name = func.attributes.get('sym_name');
            this.name = sym_name.value;
        }
        this.type = 'graph';
        if (func.name === 'func' || func.name.endsWith('.func') || /\.func_v\d+$/.test(func.name)) {
            this.type = 'function';
        }
        this.inputs = [];
        this.outputs = [];
        this.nodes = [];
        this.metadata = [];
        const tensors = new Map();
        // Handle function inputs/outputs if function_type exists
        if (func.attributes.has('function_type')) {
            const function_type = func.attributes.get('function_type');
            const args = func.regions && func.regions[0] && func.regions[0].blocks && func.regions[0].blocks[0] && func.regions[0].blocks[0].arguments ? func.regions[0].blocks[0].arguments : [];
            const inputs = function_type.type.inputs;
            const results = function_type.type.results;
            for (let i = 0; i < inputs.length; i++) {
                const input = inputs[i];
                let name = '';
                if (args[i] && args[i].value) {
                    name = args[i].value;
                } else if (input.value) {
                    name = input.value;
                } else {
                    name = i.toString();
                }
                const type = mlir.Utility.valueType(input.type || input);
                const value = new mlir.Value(name, type, '', null);
                const argument = new mlir.Argument(name, [value]);
                this.inputs.push(argument);
            }
            for (let i = 0; i < results.length; i++) {
                const output = results[i];
                const name = output.value || i.toString();
                const type = mlir.Utility.valueType(output.type);
                const valueName = output.value || output.name || `%result${i}`;
                const value = new mlir.Value(valueName, type, '', null);
                const argument = new mlir.Argument(name, [value]);
                this.outputs.push(argument);
            }
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
                        attributes: op.getAttrDictionary(),
                        operands: [],
                        results: [],
                        regions: op.regions || [],
                        delete: false,
                    };
                    const opMetadata = op.metadata;
                    const operands = op.operands || [];
                    for (let i = 0; i < operands.length; i++) {
                        const input = op.operands[i];
                        const inputName = (opMetadata && opMetadata.operands && opMetadata.operands[i] ? opMetadata.operands[i].name : null) || input.name || i.toString();
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
                                type: new mlir.PrimitiveType('i64')
                            });
                        } else if (typeof input.value === 'boolean') {
                            operation.operands.push({
                                name: inputName,
                                value: input.value,
                                type: new mlir.PrimitiveType('i1')
                            });
                        } else if (Array.isArray(input.value)) {
                            operation.operands.push({
                                name: inputName,
                                value: input.value
                            });
                        } else if (typeof input.value === 'string' && input.value) {
                            const value = values.map(input.value);
                            value.to.push(operation);
                            const args = [{ name: input.value, type: input.type }];
                            operation.operands.push({
                                name: inputName,
                                value: args
                            });
                        } else {
                            operation.operands.push({
                                name: inputName,
                                value: input
                            });
                        }
                    }
                    const results = op.results || [];
                    for (let i = 0; i < results.length; i++) {
                        const output = results[i];
                        if (!output.value) {
                            // Skip results without value identifiers
                            continue;
                        }
                        const value = values.map(output.value);
                        value.type = mlir.Utility.valueType(output.type);
                        value.from.push(operation);
                        const outputName = (opMetadata && opMetadata.results && opMetadata.results[i] ? opMetadata.results[i].name : null) || output.name || i.toString();
                        operation.results.push({
                            name: outputName,
                            value: [value]
                        });
                    }
                    operations.push(operation);
                }
            }
        }
        // Build map of single-use constant tensors to convert to initializers
        const constantMap = new Map();
        const constantTypes = new Set([
            'tosa.const', 'stablehlo.constant', 'arith.constant',
            'mhlo.constant', 'torch.constant.tensor'
        ]);
        for (const op of operations) {
            if (constantTypes.has(op.type) &&
                op.operands.length === 0 &&
                op.results.length === 1 &&
                op.results[0].value.length === 1) {
                const [result] = op.results[0].value;
                if (result.to && result.to.length === 1) {
                    const valueAttr = op.attributes.get('value') || op.attributes.get('values');
                    if ((valueAttr instanceof mlir.DenseElementsAttr || valueAttr instanceof mlir.DenseResourceElementsAttr) &&
                        valueAttr.value !== null &&
                        valueAttr.type && valueAttr.type.toString().startsWith('tensor<')) {
                        const type = mlir.Utility.valueType(valueAttr.type);
                        if (type instanceof mlir.TensorType) {
                            constantMap.set(result.name, new mlir.Tensor(type, valueAttr.value));
                            op.delete = true;
                        }
                    }
                }
            }
        }
        const tensor = (arg) => {
            if (!tensors.has(arg.name)) {
                const initializer = constantMap.get(arg.name) || null;
                tensors.set(arg.name, new mlir.Value(arg.name, arg.type, null, initializer));
            }
            return tensors.get(arg.name);
        };
        for (const input of this.inputs) {
            for (const arg of input.value) {
                if (!tensors.has(arg.name)) {
                    tensors.set(arg.name, arg);
                }
            }
        }
        // Find return operation and connect its operands to graph outputs
        const returnOp = operations.find((op) => op.type === 'return' || op.type.endsWith('.return'));
        if (returnOp) {
            for (let i = 0; i < this.outputs.length && i < returnOp.operands.length; i++) {
                const operand = returnOp.operands[i];
                if (Array.isArray(operand.value) && operand.value.length > 0) {
                    const [returnValue] = operand.value;
                    if (returnValue && typeof returnValue.name === 'string' && returnValue.name.startsWith('%')) {
                        const output = this.outputs[i];
                        const returnType = returnValue.type ? mlir.Utility.valueType(returnValue.type) : output.value[0].type;
                        output.value[0] = new mlir.Value(returnValue.name, returnType, '', null);
                    }
                }
            }
            returnOp.delete = true;
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
        for (const op of operations.filter((op) => !op.delete)) {
            const node = new mlir.Node(metadata, op, context, tensor);
            this.nodes.push(node);
        }
        for (const [name, value] of func.attributes) {
            if (name === 'sym_name' || name === 'function_type') {
                continue;
            }
            const metadata = new mlir.Argument(name, value, 'attribute');
            this.metadata.push(metadata);
        }
    }
};

mlir.Argument = class {

    constructor(name, value, type = null) {
        this.name = name;
        this.value = value;
        this.type = type;
        // Normalize common type aliases and accept extended MLIR types
        if (this.type) {
            // Convert mlir.Type objects to strings for high-level usage
            const typeStr = this.type instanceof mlir.Type ? this.type.toString() : this.type;
            switch (typeStr) {
                case 'i64': case 'si64': this.type = 'int64'; break;
                case 'i48': case 'si48': this.type = 'int48'; break;
                case 'i32': case 'si32': this.type = 'int32'; break;
                case 'i16': case 'si16': this.type = 'int16'; break;
                case 'i8': case 'si8': this.type = 'int8'; break;
                case 'i1': this.type = 'int1'; break;
                case 'f32': case 'float32': this.type = 'float32'; break;
                case 'f64': case 'float64': this.type = 'float64'; break;
                case 'f16': this.type = 'float16'; break;
                case 'f80': this.type = 'float80'; break;
                case 'f128': this.type = 'float128'; break;
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
                case 'function':
                case 'symbol':
                case 'graph':
                    break;
                default:
                    if (/^[usi]i?[0-9]+$/.test(typeStr) || /^f[0-9]+$/.test(typeStr) ||
                        typeStr === 'bf16' || typeStr === 'index' || typeStr === 'none' ||
                        typeStr === 'unit' || typeStr.startsWith('!') || typeStr.startsWith('tensor<') ||
                        typeStr.startsWith('memref<') || typeStr.startsWith('vector<')) {
                        this.type = typeStr;
                        break;
                    }
                    throw new mlir.Error(`Unsupported argument type '${typeStr}'.`);
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

    constructor(metadata, op, context, tensor) {
        if (!op.type) {
            throw new mlir.Error('Undefined node type.');
        }
        this.type = { ...metadata.type(op.identifier || '') };
        this.type.name = op.type || '';
        this.type.identifier = op.identifier || '';
        this.name = op.name || '';
        this.inputs = [];
        this.outputs = [];
        this.attributes = [];
        this.blocks = [];
        for (const input of op.operands || []) {
            let argument = null;
            if (input.type) {
                const typeStr = input.type instanceof mlir.Type ? input.type.toString() : input.type;
                if (typeStr.startsWith('tensor<')) {
                    const type = mlir.Utility.valueType(typeStr);
                    const value = new mlir.Tensor(type, input.value);
                    argument = new mlir.Argument(input.name, value, 'tensor');
                } else {
                    argument = new mlir.Argument(input.name, input.value, input.type);
                }
            } else if (Array.isArray(input.value) && !input.value.every((value) => typeof value.name === 'string' && value.name.startsWith('%'))) {
                argument = new mlir.Argument(input.name, input.value, input.type || 'attribute');
            } else if (Array.isArray(input.value)) {
                argument = new mlir.Argument(input.name, input.value.map((arg) => tensor(arg)));
            } else {
                argument = new mlir.Argument(input.name, input.value, input.type || 'attribute');
            }
            this.inputs.push(argument);
        }
        for (const output of op.results || []) {
            const argument = new mlir.Argument(output.name, output.value.map((arg) => tensor(arg)));
            this.outputs.push(argument);
        }
        if (op.attributes) {
            for (const [name, attr] of op.attributes) {
                let value = attr;
                let type = null;
                if (attr instanceof mlir.SymbolRefAttr && context) {
                    const graph = context.function(`${value.value}`);
                    if (graph) {
                        value = graph;
                        type = 'function';
                    }
                } else if (attr instanceof mlir.DenseElementsAttr && attr.value !== null) {
                    value = new mlir.Tensor(mlir.Utility.valueType(attr.type), attr.value);
                    type = 'tensor';
                } else if (attr instanceof mlir.DenseResourceElementsAttr) {
                    value = new mlir.Tensor(mlir.Utility.valueType(attr.type), null);
                    type = 'tensor';
                } else if (attr instanceof mlir.ArrayAttr || attr instanceof mlir.DenseArrayAttr) {
                    value = attr.value;
                } else if (attr) {
                    value = attr.toString();
                }
                const attribute = new mlir.Argument(name, value, type || 'attribute');
                this.attributes.push(attribute);
            }
        }
        if (op.regions && op.regions.length > 0) {
            const opMetadata = this.type;
            for (let i = 0; i < op.regions.length; i++) {
                const region = op.regions[i];
                if (region.blocks && region.blocks.length > 0) {
                    const name = (opMetadata.regions && opMetadata.regions[i] ? opMetadata.regions[i].name : null) || i.toString();
                    const func = { name: '', attributes: new Map(), regions: [region] };
                    const graph = new mlir.Graph(metadata, func, context, '');
                    const argument = new mlir.Argument(name, graph, 'graph');
                    this.blocks.push(argument);
                }
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

mlir.Context = class {

    constructor(metadata, functions) {
        this._metadata = metadata;
        this._functions = functions; // Map of fullName -> {func, prefix, base, module}
        this._graphs = new Map();
        this._constructing = new Set();
    }

    graph(module, name) {
        if (!this._graphs.has(name)) {
            this._constructing.add(name);
            const graph = new mlir.Graph(this._metadata, module, this, name);
            this._graphs.set(name, graph);
            this._constructing.delete(name);
        }
        return this._graphs.get(name);
    }

    function(name) {
        // Return cached graph if already constructed
        if (this._graphs.has(name)) {
            return this._graphs.get(name);
        }
        // If currently constructing this graph, return placeholder to break cycle
        if (this._constructing.has(name)) {
            return { name, type: 'function', nodes: [], inputs: [], outputs: [] };
        }
        // Try to find by full name first
        if (this._functions.has(name)) {
            const info = this._functions.get(name);
            return this.graph(info.func, name);
        }
        // Try to find by base name (for callee resolution within same module)
        for (const [fullName, info] of this._functions) {
            if (info.base === name) {
                if (this._graphs.has(fullName)) {
                    return this._graphs.get(fullName);
                }
                if (this._constructing.has(fullName)) {
                    return { name: fullName, type: 'function', nodes: [], inputs: [], outputs: [] };
                }
                return this.graph(info.func, fullName);
            }
        }
        return null;
    }
};

mlir.Operation = class {

    constructor(name) {
        this.name = name;
        this.attributes = new Map();
        this.operands = [];
        this.regions = [];
        this.results = [];
        this.propertiesAttr = null;
    }

    addAttribute(name, value) {
        this.attributes.set(name, value);
    }

    addRegion() {
        const region = {};
        this.regions.push(region);
        return region;
    }

    getAttr(name) {
        if (this.propertiesAttr instanceof mlir.DictionaryAttr) {
            const value = this.propertiesAttr.get(name);
            if (value !== undefined) {
                return value;
            }
        }
        return this.attributes.get(name);
    }

    getAttrDictionary() {
        if (this.propertiesAttr instanceof mlir.DictionaryAttr) {
            const result = new Map(this.attributes);
            for (const [name, value] of this.propertiesAttr.value) {
                result.set(name, value);
            }
            return result;
        }
        return this.attributes;
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
        this._tokens = [new mlir.Token(null, null, null), new mlir.Token(null, null, null), new mlir.Token(null, null, null), new mlir.Token(null, null, null)];
        this._index = 0;
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
                    if (this._peek() !== '/' && this._peek() !== '*') {
                        this._read();
                        return this._token('/', '/');
                    }
                    this._skipComment();
                    this._position = this._currentPosition;
                    continue;
                case '.':
                    if (/[0-9]/.test(this._peek())) {
                        return this._number();
                    }
                    this._read();
                    if (this._current === '.' && this._next === '.') {
                        this._read();
                        this._read();
                        return this._token('ellipsis', '...');
                    }
                    return this._token('.', '.');
                case '-':
                    if (/[0-9]/.test(this._peek())) {
                        return this._number();
                    } else if (this._peek() === '>') {
                        this._read();
                        this._read();
                        return this._token('->', '->');
                    }
                    this._read();
                    return this._token('keyword', '-');
                case '+':
                    this._read();
                    return this._token('keyword', '+');
                case '"':
                    return this._stringLiteral();
                case '@':
                    return this._symbolRefId();
                case '%':
                    return this._valueId();
                case '#':
                    if (this._peek() === '-') {
                        const position = this._decoder.position;
                        const next = this._decoder.decode();
                        this._decoder.position = position;
                        if (next === '}') {
                            this._read();
                            this._read();
                            this._read();
                            return this._token('#-}', '#-}');
                        }
                    }
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
                        return this._token('==', '==');
                    }
                    this._read();
                    return this._token('=', '=');
                case ':':
                    if (this._peek() === ':') {
                        this._read();
                        this._read();
                        return this._token('::', '::');
                    }
                    this._read();
                    return this._token(':', ':');
                case ',':
                case '(':
                case ')':
                case '{': {
                    if (this._peek() === '-') {
                        const position = this._decoder.position;
                        const next = this._decoder.decode();
                        this._decoder.position = position;
                        if (next === '#') {
                            this._read();
                            this._read();
                            this._read();
                            return this._token('{-#', '{-#');
                        }
                    }
                    const value = this._read();
                    return this._token(value, value);
                }
                case '}':
                case '[':
                case ']':
                case '?':
                case '*':
                case '|': {
                    const value = this._read();
                    return this._token(value, value);
                }
                case '<':
                    if (this._peek() === '=') {
                        this._read();
                        this._read();
                        return this._token('<=', '<=');
                    }
                    this._read();
                    return this._token('<', '<');
                case '>':
                    if (this._peek() === '=') {
                        this._read();
                        this._read();
                        return this._token('>=', '>=');
                    }
                    this._read();
                    return this._token('>', '>');
                default:
                    if (/[a-zA-Z_$]/.test(this._current) || /[.-]/.test(this._current)) {
                        return this._identifier();
                    }
                    if (/[0-9]/.test(this._current)) {
                        return this._number();
                    }
                    throw new mlir.Error(`Unexpected character '${this._current}' ${this.location()}`);
            }
        }
        return this._token('eof', null);
    }

    resetPointer(offset) {
        if (offset < 0) {
            throw new mlir.Error('resetPointer does not support negative offsets.');
        }
        this._decoder.position = this._position;
        for (let i = 0; i < offset; i++) {
            this._decoder.decode();
        }
        this._nextPosition = this._decoder.position;
        this._next = this._decoder.decode();
        this._read();
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
        if (this._current === '*') {
            this._read();
            while (this._current) {
                if (this._current === '*') {
                    this._read();
                    if (this._current === '/') {
                        this._read();
                        return;
                    }
                } else {
                    this._read();
                }
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
            return this._token(type, parseInt(v, 16), v);
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
            return this._token(type, parseFloat(v), v);
        }
        return this._token(type, parseInt(v, 10), v);
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
            return this._token('string', result);
        }
        throw new mlir.Error('Unterminated string literal');
    }

    _identifier() {
        let result = '';
        while (this._current && (/[a-zA-Z_$\-.]/.test(this._current) || /[0-9]/.test(this._current))) {
            // Don't consume '-' if it's followed by '>' (to preserve '->' as separate token)
            if (this._current === '-' && this._peek() === '>') {
                break;
            }
            result += this._read();
        }
        switch (result) {
            case 'loc':
                return this._token('keyword', result);
            case 'true':
            case 'false':
                return this._token('boolean', result === 'true');
            case 'unknown':
                return this._token('id', result);
            default:
                return this._token('id', result);
        }
    }

    _attributeAlias() {
        let value = '#';
        this._read();
        if (this._current === '"') {
            value += this._stringLiteral().value;
        } else {
            while (this._current && (/[a-zA-Z_$]/.test(this._current) || /[0-9]/.test(this._current) || /[.-]/.test(this._current))) {
                // Don't consume '-' if it's followed by '>' (to preserve '->' as separate token)
                if (this._current === '-' && this._peek() === '>') {
                    break;
                }
                value += this._read();
            }
            if (this._current === ':' && this._peek() === ':') {
                value += this._read();
                value += this._read();
                value += this._symbolRefId().value;
            }
        }
        return this._token('#', value);
    }

    _symbolRefId() {
        let result = '@';
        this._read();
        if (this._current === '"') {
            result += this._stringLiteral().value;
        } else {
            while (this._current && (/[a-zA-Z_$]/.test(this._current) || /[0-9]/.test(this._current) || /[.-]/.test(this._current))) {
                // Don't consume '-' if it's followed by '>' (to preserve '->' as separate token)
                if (this._current === '-' && this._peek() === '>') {
                    break;
                }
                result += this._read();
            }
            if (this._current === ':' && this._peek() === ':') {
                result += this._read();
                result += this._read();
                result += this._symbolRefId().value;
            }
        }
        return this._token('@', result);
    }

    _typeAlias() {
        this._read();
        let dialectName = '';
        while (this._current && /[a-zA-Z_$0-9]/.test(this._current)) {
            dialectName += this._read();
        }
        if (!dialectName) {
            throw new mlir.Error('Invalid type alias.');
        }
        return this._token('!', `!${dialectName}`);
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
        return this._token('%', result);
    }

    _caretId() {
        let result = '^';
        this._read();
        if (this._current === ':' && this._peek() !== ':') {
            result += this._read();
            return this._token('^', result);
        }
        while (this._current && (/[a-zA-Z_$]/.test(this._current) || /[0-9]/.test(this._current) || /[.-]/.test(this._current))) {
            result += this._read();
        }
        if (this._current === ':' && this._peek() === ':') {
            result += this._read();
            result += this._read();
            result += this._caretId().value;
        }
        return this._token('^', result);
    }

    _token(kind, value, text) {
        const token = this._tokens[this._index];
        this._index = (this._index + 1) % this._tokens.length;
        token.kind = kind;
        token.value = value;
        token.text = text;
        return token;
    }
};

mlir.ParserState = class {
    constructor() {
        this.defaultDialectStack = ['builtin'];
        this.attributeAliasDefinitions = new Map();
        this.typeAliasDefinitions = new Map();
    }
};

mlir.Parser = class {

    constructor(state, decoder, context) {
        this._tokenizer = new mlir.Tokenizer(decoder);
        this._token = this._tokenizer.read();
        this._state = state;
        this._context = context;
    }

    async parse() {
        // Reference: Parser.cpp TopLevelOperationParser::parse
        // https://mlir.llvm.org/docs/LangRef/#top-level-productions
        const block = {
            operations: []
        };
        while (true) {
            if (this.match('eof')) {
                break;
            }
            if (this.match('#')) {
                this.parseAttributeAliasDef();
                continue;
            }
            if (this.match('!')) {
                this.parseTypeAliasDef();
                continue;
            }
            if (this.match('{-#')) {
                this.parseFileMetadataDictionary();
                continue;
            }
            const op = this.parseOperation();
            block.operations.push(op);
        }
        return block;
    }

    parseAttributeAliasDef() {
        const aliasName = this.expect();
        this.expect('=');
        // Handle pre-2020 bare affine map syntax: (dims) [symbols] -> (results)
        // Changed to affine_map<...> in llvm/llvm-project@4268e4f4b84b (Jan 2020)
        let attr = null;
        if (this.match('(')) {
            const dims = this.skip('(');
            const symbols = this.match('[') ? this.skip('[') : '';
            this.expect('->');
            const results = this.match('(') ? this.skip('(') : '';
            attr = { value: `affine_map<${dims}${symbols} -> ${results}>`, name: 'affine_map' };
        } else {
            attr = this.parseAttribute();
        }
        this._state.attributeAliasDefinitions.set(aliasName, attr);
    }

    parseTypeAliasDef() {
        const aliasName = this.expect('!');
        this.expect('=');
        this.accept('id', 'type');
        const type = this.parseType();
        this._state.typeAliasDefinitions.set(aliasName, type);
    }

    parseFileMetadataDictionary() {
        this.expect('{-#');
        while (!this.match('#-}') && !this.match('eof')) {
            this._token = this._tokenizer.read();
        }
        this.expect('#-}');
    }

    parseFunctionArgumentList(allowVariadic) {
        const inputs = [];
        let isVariadic = false;
        if (this.accept('(')) {
            while (!this.accept(')')) {
                if (this.match(')')) {
                    break;
                }
                if (allowVariadic && this.accept('ellipsis')) {
                    isVariadic = true;
                    this.expect(')');
                    break;
                }
                if (this.match('%')) {
                    const input = {};
                    input.value = this._token.value;
                    this.expect('%');
                    this.expect(':');
                    input.type = this.parseType();
                    if (this.match('{')) {
                        input.attributes = new Map();
                        this.parseAttributeDict(input.attributes);
                    }
                    input.loc = this.parseLocation();
                    inputs.push(input);
                } else {
                    const input = {};
                    input.value = `%arg${inputs.length}`;
                    input.type = this.parseType();
                    if (this.match('{')) {
                        input.attributes = new Map();
                        this.parseAttributeDict(input.attributes);
                    }
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
        return { arguments: inputs, isVariadic };
    }

    parseFunctionSignatureWithArguments(allowVariadic) {
        const argResult = this.parseFunctionArgumentList(allowVariadic);
        const resultTypes = [];
        const resultAttrs = [];
        if (this.accept('->')) {
            this.parseFunctionResultList(resultTypes, resultAttrs);
        }
        return { arguments: argResult.arguments, isVariadic: argResult.isVariadic, resultTypes, resultAttrs };
    }

    parseFunctionOp(op, allowVariadic) {
        this.parseOptionalVisibilityKeyword(op.attributes);
        this.parseSymbolName('sym_name', op.attributes);
        const sig = this.parseFunctionSignatureWithArguments(allowVariadic);
        const argTypes = [];
        for (const arg of sig.arguments) {
            if (arg.value !== '...') {
                argTypes.push(arg.type);
            }
        }
        const type = new mlir.FunctionType(argTypes, sig.resultTypes);
        op.addAttribute('function_type', new mlir.TypeAttrOf(type));
        if (sig.resultAttrs.some((a) => a !== null)) {
            op.addAttribute('res_attrs', sig.resultAttrs);
        }
        const argAttrs = sig.arguments.filter((a) => a.value !== '...').map((a) => a.attributes || null);
        if (argAttrs.some((a) => a !== null)) {
            op.addAttribute('arg_attrs', argAttrs);
        }
        this.parseOptionalAttrDictWithKeyword(op.attributes);
        if (this.match('{')) {
            const region = op.addRegion();
            this.parseRegion(region, sig.arguments);
        }
    }

    parseTypeList() {
        return this.parseCommaSeparatedList('none', () => this.parseType());
    }

    parseTypeListNoParens() {
        return this.parseCommaSeparatedList('none', () => this.parseType());
    }

    parseTypeListParens() {
        this.expect('(');
        if (this.accept(')')) {
            return [];
        }
        const types = this.parseTypeListNoParens();
        this.expect(')');
        return types;
    }

    skip(open) {
        const closingFor = { '<': '>', '[': ']', '(': ')', '{': '}' };
        const openingFor = { '>': '<', ']': '[', ')': '(', '}': '{' };
        let value = '';
        if (this.match(open)) {
            const stack = [open];
            value += this.expect();
            while (stack.length > 0) {
                if (this.match('eof')) {
                    throw new mlir.Error(`Unbalanced '${stack[stack.length - 1]}' ${this.location()}`);
                }
                const token = this._token.value;
                if (closingFor[token]) {
                    stack.push(token);
                } else if (openingFor[token]) {
                    if (stack[stack.length - 1] !== openingFor[token]) {
                        throw new mlir.Error(`Unbalanced '${stack[stack.length - 1]}' ${this.location()}`);
                    }
                    stack.pop();
                }
                value += this.expect();
            }
        }
        return value;
    }

    parseOperation() {
        const results = [];
        if (this.match('%')) {
            const parseNextResult = () => {
                const value = this.expect('%');
                const index = value.indexOf(':');
                if (index === -1) {
                    results.push({ value });
                } else {
                    const id = value.substring(0, index);
                    const length = parseInt(value.substring(index + 1), 10);
                    for (let i = 0; i < length; i++) {
                        results.push({ value: `${id}#${i}` });
                    }
                }
                return true;
            };
            this.parseCommaSeparatedList('none', parseNextResult);
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

    parseOptionalSSAUseList(results) {
        if (!this.match('%')) {
            return;
        }
        do {
            const value = this.expect('%');
            results.push({ value });
        } while (this.accept(',') && this.match('%'));
    }

    parseOperandList(delimiter) {
        delimiter = delimiter || 'none';
        if (delimiter === 'none') {
            if (!this.match('%')) {
                return [];
            }
        }
        const parseOneOperand = () => {
            if (this.match('%')) {
                const value = this.expect('%');
                return { value };
            }
            return null;
        };
        return this.parseCommaSeparatedList(delimiter, parseOneOperand);
    }

    resolveOperands(operands, types) {
        const count = Math.min(operands.length, types.length);
        for (let i = 0; i < count; i++) {
            operands[i].type = types[i];
        }
        for (let i = operands.length; i < types.length; i++) {
            operands.push({ type: types[i], value: `%${i}` });
        }
    }

    parseSuccessors(successors) {
        const parsed = this.parseCommaSeparatedList('square', () => {
            return { label: this.expect('^') };
        });
        for (const s of parsed) {
            successors.push(s);
        }
    }

    parseGenericOperationAfterOpName(op) {
        this.expect('(');
        this.parseOptionalSSAUseList(op.operands);
        this.expect(')');
        if (this.match('[')) {
            op.successors = [];
            this.parseSuccessors(op.successors);
        }
        if (this.accept('<')) {
            op.propertiesAttr = this.parseAttribute();
            this.expect('>');
        }
        if (this.accept('(')) {
            do {
                const region = op.addRegion();
                this.parseRegion(region);
            } while (this.accept(','));
            this.expect(')');
        }
        if (this.match('{')) {
            this.parseAttributeDict(op.attributes);
        }
        this.expect(':');
        const fnType = this.parseType();
        if (fnType instanceof mlir.FunctionType === false) {
            throw new mlir.Error(`Expected function type ${this.location()}`);
        }
        for (let i = 0; i < fnType.inputs.length && i < op.operands.length; i++) {
            op.operands[i].type = fnType.inputs[i].toString();
        }
        for (let i = 0; i < fnType.results.length; i++) {
            if (i < op.results.length) {
                op.results[i].type = fnType.results[i].toString();
            } else {
                op.results.push({ type: fnType.results[i].toString() });
            }
        }
        op.loc = this.parseLocation();
        return op;
    }

    parseCustomOperation(results) {
        const opNameInfo = this.parseCustomOperationName();
        const op = new mlir.Operation(opNameInfo);
        op.results = results;
        let opName = this._context.resolveOpName(op.name);
        const index = opName.indexOf('.');
        if (index === -1) {
            throw new mlir.Error(`No dialect found '${opName}' ${this.location()}`);
        }
        const dialectName = opName.substring(0, index);
        const dialect = this._context.getDialect(dialectName);
        if (!dialect) {
            throw new mlir.Error(`Unsupported dialect '${dialectName}'.`);
        }
        // Normalize operation name to canonical dialect name for metadata lookup
        // (e.g., spv.Load -> spirv.Load when dialect.name is spirv)
        opName = dialectName === dialect.name ? opName : opName.replace(`${dialectName}.`, `${dialect.name}.`);
        const opInfo = dialect.getOperation(opName);
        if (!opInfo) {
            // Do not remove and address the underlying root cause.
            throw new mlir.Error(`Unsupported operation '${op.name}'.`);
        }
        op.metadata = opInfo.metadata;
        const defaultDialect = (opInfo && opInfo.metadata && opInfo.metadata.defaultDialect) || '';
        this._state.defaultDialectStack.push(defaultDialect);
        if (!dialect.parseOperation(this, opName, op)) {
            this._state.defaultDialectStack.pop();
            throw new mlir.Error(`Unsupported custom operation '${op.name}' ${this.location()}`);
        }
        if (!dialect.hasParser(opName) && !dialect.hasCustomAssemblyFormat(opName) && dialect.hasAssemblyFormat(opName) && dialect.hasParseOperation(opName) !== false) {
            throw new mlir.Error(`Operation '${op.name}' has assembly format but was handled by custom dialect code.`);
        }
        op.loc = this.parseLocation() || {};
        this._state.defaultDialectStack.pop();
        return op;
    }

    parseCustomOperationName() {
        let opName = this.expect('id');
        if (opName.indexOf('.') === -1) {
            for (let i = this._state.defaultDialectStack.length - 1; i >= 0; i--) {
                let dialect = this._state.defaultDialectStack[i];
                if (dialect) {
                    // Workaround: old std.constant should be arith.constant
                    if (dialect === 'func' && opName === 'constant' && !this.match('@')) {
                        dialect = 'arith';
                    }
                    opName = `${dialect}.${opName}`;
                    break;
                }
            }
        }
        return opName;
    }

    parseGenericOperation() {
        const name = this.expect('string');
        const op = new mlir.Operation(name);
        const index = name.indexOf('.');
        if (index !== -1) {
            const dialectName = name.substring(0, index);
            const dialect = this._context.getDialect(dialectName);
            if (dialect) {
                const opInfo = dialect.getOperation(name);
                if (opInfo) {
                    op.metadata = opInfo.metadata;
                }
            }
        }
        return this.parseGenericOperationAfterOpName(op);
    }

    parseOptionalVisibilityKeyword(attributes) {
        if (this.match('id', 'private') || this.match('id', 'public') || this.match('id', 'nested')) {
            const value = this.expect();
            attributes.set('sym_visibility', value);
        }
    }

    parseSymbolName(name, attributes) {
        const value = this.expect('@').substring(1);
        attributes.set(name, new mlir.StringAttr(value));
    }

    parseOptionalSymbolName() {
        if (this.match('@')) {
            const value = this.expect('@');
            return value.substring(1);
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
                if (this.match('id') || this.match('string') || this.match('keyword') || this.match('boolean')) {
                    name = this.expect();
                } else if (this.match('[')) {
                    const arrayValue = this.parseAttribute();
                    attributes.set('array', arrayValue.value);
                    this.accept(',');
                    continue;
                } else if (!this.match('=') && !this.match(':') && !this.match('}')) {
                    throw new mlir.Error(`Expected attribute name or '}', but got '${this._token.value}' ${this.location()}`);
                }
                let attribute = {};
                if (this.accept('=') || this.accept(':')) {
                    attribute = this.parseAttribute();
                    if (this.accept(':')) {
                        attribute.type = this.parseType();
                    }
                } else if (name === null) {
                    break;
                } else {
                    attributes.set(name, new mlir.UnitAttr());
                    this.accept(',');
                    continue;
                }
                attributes.set(name, attribute);
                if (!this.accept(',') && !this.match('}')) {
                    throw new mlir.Error(`Expected ',' or '}' after attribute, but got '${this._token.value}' ${this.location()}`);
                }
            }
        }
    }

    parsePropertyDict(attributes) {
        if (this.accept('<')) {
            if (this.accept('{')) {
                while (!this.accept('}')) {
                    let name = null;
                    if (this.match('id') || this.match('string') || this.match('keyword') || this.match('boolean')) {
                        name = this.expect();
                    } else if (!this.match('=') && !this.match(':') && !this.match('}')) {
                        throw new mlir.Error(`Expected property name or '}', but got '${this._token.value}' ${this.location()}`);
                    }
                    let attribute = {};
                    if (this.accept('=') || this.accept(':')) {
                        attribute = this.parseAttribute();
                        if (this.accept(':')) {
                            attribute.type = this.parseType();
                        }
                    } else if (name === null) {
                        break;
                    } else {
                        attributes.set(name, new mlir.UnitAttr());
                        this.accept(',');
                        continue;
                    }

                    attributes.set(name, attribute);
                    if (!this.accept(',') && !this.match('}')) {
                        throw new mlir.Error(`Expected ',' or '}' after property, but got '${this._token.value}' ${this.location()}`);
                    }
                }
            }
            this.expect('>');
        }
    }

    parseRegion(region, entryArguments) {
        region.blocks = Array.isArray(region.blocks) ? region.blocks : [];
        const block = {};
        this.parseBlock(block);
        if (entryArguments && entryArguments.length > 0) {
            if (block.arguments.length === 0) {
                block.arguments = entryArguments;
            } else if (block.arguments.length !== entryArguments.length) {
                throw new mlir.Error(`Entry block has ${block.arguments.length} arguments, but function signature has ${entryArguments.length} ${this.location()}`);
            }
        }
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
        if (hasMultipleBlocks) {
            this.accept('}');
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
                const text = this.expect('string');
                let content = `"${text}"`;
                if (this.accept('(')) {
                    const child = this.parseLocationContent();
                    this.expect(')');
                    content += `(${child})`;
                } else if (this.accept(':')) {
                    const line = this.expect('int');
                    content += `:${line}`;
                    if (this.accept(':')) {
                        const col = this.expect('int');
                        content += `:${col}`;
                        if (this.accept('id', 'to')) {
                            // File range location: loc("file":L:C to L:C) or loc("file":L:C to :C)
                            if (this.accept(':')) {
                                // loc("file":L:C to :endCol) - short form
                                const endCol = this.expect('int');
                                content += ` to :${endCol}`;
                            } else if (this.match('int')) {
                                // loc("file":L:C to endLine:endCol) - full form
                                const endLine = this.expect('int');
                                content += ` to ${endLine}`;
                                if (this.accept(':')) {
                                    const endCol = this.expect('int');
                                    content += `:${endCol}`;
                                }
                            }
                        }
                    }
                }
                location.value = `loc(${content})`;
            } else if (this.match('#')) {
                const attr = this.parseExtendedAttr();
                location.value = `loc(${attr.value})`;
            } else if (this.accept('id', 'unknown')) {
                location.value = 'loc(unknown)';
            } else if (this.accept('id', 'callsite')) {
                this.expect('(');
                location.type = 'callsite';
                location.callee = this.parseLocationContent();
                this.expect('id', 'at');
                location.caller = this.parseLocationContent();
                this.expect(')');
            } else if (this.accept('id', 'fused')) {
                // Reference: LocationParser.cpp parseFusedLocation
                location.type = 'fused';
                if (this.accept('<')) {
                    location.metadata = this.parseAttribute();
                    this.expect('>');
                }
                location.locations = this.parseCommaSeparatedList('square', () => this.parseLocationContent());
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
            const attr = this.parseExtendedAttr();
            return { alias: attr.value };
        }
        if (this.match('keyword', 'loc')) {
            return this.parseLocation();
        }
        if (this.accept('id', 'unknown')) {
            return { type: 'unknown' };
        }
        if (this.accept('id', 'callsite')) {
            const location = { type: 'callsite' };
            this.expect('(');
            location.callee = this.parseLocationContent();
            this.expect('id', 'at');
            location.caller = this.parseLocationContent();
            this.expect(')');
            return location;
        }
        if (this.accept('id', 'fused')) {
            // Nested fused location inside location content
            const location = { type: 'fused' };
            if (this.accept('<')) {
                location.metadata = this.parseAttribute();
                this.expect('>');
            }
            location.locations = this.parseCommaSeparatedList('square', () => this.parseLocationContent());
            return location;
        }
        if (this.match('string')) {
            const location = {};
            location.file = this.expect('string');
            if (this.accept(':')) {
                location.line = this.expect('int');
                if (this.accept(':')) {
                    location.col = this.expect('int');
                    // Handle file range in location content: "file":L:C to L:C
                    if (this.accept('id', 'to')) {
                        if (this.accept(':')) {
                            location.endCol = this.expect('int');
                        } else if (this.match('int')) {
                            location.endLine = this.expect('int');
                            if (this.accept(':')) {
                                location.endCol = this.expect('int');
                            }
                        }
                    }
                }
            } else if (this.accept('(')) {
                // NameLoc with child location: "name"(child_location)
                location.child = this.parseLocationContent();
                this.expect(')');
            }
            return location;
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
        while (!this.match(')') && !this.match('->') && !this.match('{') && !this.match('}') && !this.match('=') && !this.match('^') && !(this.match(':') && !open)) {
            const input = {};
            if (this.match('[')) {
                this.expect('[');
                const array = [];
                while (!this.match(']')) {
                    if (this.match('%')) {
                        array.push(this.expect());
                    } else if (this.match('int')) {
                        array.push(this.parseInteger());
                    } else if (this.match('-')) {
                        this.expect('-');
                        if (this.match('int')) {
                            array.push(-this.parseInteger());
                        } else {
                            throw new mlir.Error(`Expected integer after '-' in array literal ${this.location()}`);
                        }
                    } else {
                        break;
                    }
                    if (!this.accept(',')) {
                        break;
                    }
                }
                this.expect(']');
                input.value = array;
                inputs.push(input);
                if (!this.accept(',')) {
                    break;
                }
                continue;
            }
            if (this._token.kind === 'id' && this._token.value !== 'dense' && this._token.value !== 'dense_resource') {
                const identifier = this.expect('id');
                if (this.accept('(')) {
                    const args = this.parseArguments();
                    for (let i = 0; i < args.length; i++) {
                        const arg = args[i];
                        arg.name = `${identifier}.${i}`;
                        inputs.push(arg);
                    }
                    this.resolveOperands(inputs, this.parseOptionalColonTypeList());
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
                const value = this.parseAttribute();
                input.type = value.type;
                input.value = value.value;
                if (open && this.accept(':')) {
                    input.type = this.parseType();
                }
            }
            inputs.push(input);
            if (!this.accept(',') && !this.match('id')) {
                break;
            }
        }
        if (open) {
            this.expect(')');
        }
        return inputs;
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
            if (this.accept('<')) {
                const elementType = this.parseType();
                this.expect('>');
                return new mlir.ComplexType(elementType);
            }
        } else if (prefix === 'tensor' || prefix === 'vector' || prefix === 'memref') {
            if (this.accept('<')) {
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
                // Parse optional extras for memref and tensor (layout, memory space, encoding)
                const extras = [];
                while (this.accept(',')) {
                    const extra = this.parseAttribute();
                    extras.push(extra);
                }
                this.expect('>');
                let nestedTypeStr = `${prefix}<`;
                if (nestedDimInfo.unranked) {
                    nestedTypeStr += '*x';
                } else if (nestedDimInfo.dimensions.length > 0) {
                    nestedTypeStr += `${nestedDimInfo.dimensions.join('x')}x`;
                }
                nestedTypeStr += nestedElementType;
                for (const extra of extras) {
                    let extraStr = extra;
                    if (typeof extra === 'object') {
                        extraStr = extra.value === undefined ? JSON.stringify(extra) : extra.value;
                    }
                    nestedTypeStr += `, ${extraStr}`;
                }
                nestedTypeStr += '>';
                return nestedTypeStr;
            }
        }
        if (prefix.startsWith('!')) {
            // Reuse current token by mutating it (tokens are pooled)
            this._token.kind = '!';
            this._token.value = prefix;
            this._token.text = prefix;
            return this.parseType();
        }
        // Return as PrimitiveType for known primitive types
        if (/^[su]?i[0-9]+$/.test(prefix) || /^[fb]f?[0-9]+/.test(prefix) || prefix === 'index') {
            return new mlir.PrimitiveType(prefix);
        }
        return prefix;
    }

    parseDimensionListRanked(allowDynamic, withTrailingX) {
        allowDynamic = allowDynamic === false ? false : true;
        withTrailingX = withTrailingX === false ? false : true;
        const dimensions = [];
        if (this.accept('*')) {
            if (this.match('id')) {
                const token = this._token.value;
                if (token === 'x' || token.startsWith('x')) {
                    this.expect('id');
                    return { unranked: true, dimensions: [], elementTypePrefix: token === 'x' ? null : token.substring(1) };
                }
            }
            return { unranked: true, dimensions: [], elementTypePrefix: null };
        }
        const parseDim = () => {
            if (this.accept('[')) {
                if (this.match('int')) {
                    dimensions.push(`[${this.expect('int')}]`);
                } else if (allowDynamic && this.match('?')) {
                    dimensions.push('[?]');
                    this.expect('?');
                }
                this.expect(']');
                return true;
            } else if (allowDynamic && this.match('?')) {
                dimensions.push('?');
                this.expect('?');
                return true;
            } else if (this.match('int')) {
                const text = this._token.text;
                if (text && text.length > 1 && text[1] === 'x') {
                    dimensions.push(0);
                    this._tokenizer.resetPointer(1);
                    this._token = this._tokenizer.read();
                } else {
                    dimensions.push(this.parseInteger());
                }
                return true;
            }
            return false;
        };
        const parseX = () => {
            if (this.match('id')) {
                const token = this._token.value;
                if (token === 'x') {
                    this.expect('id', 'x');
                    return { consumed: true, elementTypePrefix: null };
                } else if (token.startsWith('x')) {
                    this.expect('id');
                    const rest = token.substring(1);
                    // Check if rest is a dimension or type prefix
                    if (/^[0-9]/.test(rest) || (allowDynamic && rest === '?')) {
                        // Dimension merged with x - need to parse it
                        let remaining = rest;
                        while (remaining.length > 0) {
                            if (/^[0-9]/.test(remaining)) {
                                let i = 0;
                                while (i < remaining.length && /[0-9]/.test(remaining[i])) {
                                    i++;
                                }
                                const numPart = remaining.substring(0, i);
                                dimensions.push(parseInt(numPart, 10));
                                remaining = remaining.substring(i);
                                if (remaining.startsWith('x')) {
                                    remaining = remaining.substring(1);
                                    continue;
                                }
                                if (remaining.length > 0) {
                                    return { consumed: true, elementTypePrefix: remaining };
                                }
                                break;
                            } else if (allowDynamic && remaining === '?') {
                                dimensions.push('?');
                                break;
                            } else {
                                return { consumed: true, elementTypePrefix: remaining };
                            }
                        }
                        return { consumed: true, elementTypePrefix: null };
                    }
                    return { consumed: true, elementTypePrefix: rest };
                }
            }
            return { consumed: false, elementTypePrefix: null };
        };
        if (withTrailingX) {
            // Format: NxNxNx... (trailing x)
            while (true) {
                if (!parseDim()) {
                    break;
                }
                const xResult = parseX();
                if (!xResult.consumed) {
                    break;
                }
                if (xResult.elementTypePrefix) {
                    return { unranked: false, dimensions, elementTypePrefix: xResult.elementTypePrefix };
                }
            }
        } else if (parseDim()) {
            while (this.match('id') && this._token.value.startsWith('x')) {
                const xResult = parseX();
                if (!xResult.consumed) {
                    break;
                }
                if (xResult.elementTypePrefix) {
                    return { unranked: false, dimensions, elementTypePrefix: xResult.elementTypePrefix };
                }
                // If parseX already consumed a merged dimension, don't call parseDim again
                if (!this.match('int') && !(allowDynamic && this.match('?'))) {
                    break;
                }
                if (!parseDim()) {
                    break;
                }
            }
        }
        return { unranked: false, dimensions, elementTypePrefix: null };
    }

    parseTensorType() {
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
            encoding = this.parseAttribute();
        }
        this.expect('>');
        if (dimInfo.unranked) {
            // UnrankedTensorType - fall back to string for now
            const elementTypeStr = elementType instanceof mlir.Type ? elementType.toString() : elementType;
            return new mlir.Type(`tensor<*x${elementTypeStr}>`);
        }
        return new mlir.RankedTensorType(dimInfo.dimensions, elementType, encoding);
    }

    parseMemRefType() {
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
            const extra = this.parseAttribute();
            extras.push(extra);
        }
        this.expect('>');
        let typeStr = 'memref<';
        if (dimInfo.unranked) {
            typeStr += '*x';
        } else if (dimInfo.dimensions.length > 0) {
            typeStr += `${dimInfo.dimensions.join('x')}x`;
        }
        typeStr += elementType instanceof mlir.Type ? elementType.toString() : elementType;
        if (extras.length > 0) {
            const content = extras.map((e) => typeof e === 'object' ? JSON.stringify(e) : e).join(', ');
            typeStr += `, ${content}`;
        }
        typeStr += '>';
        return new mlir.Type(typeStr);
    }

    parseVectorType() {
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
        return new mlir.VectorType(dimInfo.dimensions, elementType);
    }

    parseComplexType() {
        this.expect('<');
        const elementType = this.parseType();
        this.expect('>');
        return new mlir.ComplexType(elementType);
    }

    parseTupleType() {
        this.expect('<');
        const types = [];
        while (!this.match('>')) {
            types.push(this.parseType());
            this.accept(',');
        }
        this.expect('>');
        const typeStrs = types.map((t) => t instanceof mlir.Type ? t.toString() : t);
        return new mlir.Type(`tuple<${typeStrs.join(', ')}>`);
    }

    parseCustomTypeWithFallback(typeT) {
        if (typeT && !this.match('!')) {
            if (typeof typeT === 'function') {
                return typeT(this);
            }
            const index = typeT.name.indexOf('.');
            if (index === -1) {
                throw new mlir.Error(`Invalid type name '${typeT.name}.`);
            }
            const dialectName = typeT.name.substring(0, index);
            const dialect = this._context.getDialect(dialectName);
            if (!dialect) {
                throw new mlir.Error(`Unsupported dialect '${dialectName}'.`);
            }
            return dialect.parseCustomTypeWithFallback(this, typeT.type);
        }
        return this.parseType();
    }

    parseCustomAttributeWithFallback(attrT, type) {
        if (attrT) {
            return attrT(this, type);
        }
        return this.parseAttribute(type);
    }

    parseType() {
        if (this.match('(')) {
            return this.parseFunctionType();
        }
        return this.parseNonFunctionType();
    }

    parseOptionalType() {
        if (this.match('(') || this.match('!')) {
            return this.parseType();
        } else if (this.match('id')) {
            switch (this._token.value) {
                case 'memref':
                case 'tensor':
                case 'complex':
                case 'tuple':
                case 'vector':
                case 'f4E2M1FN':
                case 'f6E2M3FN':
                case 'f6E3M2FN':
                case 'f8E5M2':
                case 'f8E4M3':
                case 'f8E4M3FN':
                case 'f8E5M2FNUZ':
                case 'f8E4M3FNUZ':
                case 'f8E4M3B11FNUZ':
                case 'f8E3M4':
                case 'f8E8M0FNU':
                case 'bf16':
                case 'f16':
                case 'tf32':
                case 'f32':
                case 'f64':
                case 'f80':
                case 'f128':
                case 'index':
                case 'none':
                    return this.parseType();
                default:
                    // Check for integer types (inttype in reference)
                    if (/^[su]?i[0-9]+$/.test(this._token.value)) {
                        return this.parseType();
                    }
                    break;
            }
        }
        return null;
    }

    parseNonFunctionType() {
        if (this.match('id')) {
            const value = this.expect('id');
            switch (value) {
                case 'tensor': return this.parseTensorType();
                case 'vector': return this.parseVectorType();
                case 'memref': return this.parseMemRefType();
                case 'complex': return this.parseComplexType();
                case 'tuple': return this.parseTupleType();
                case 'none': return new mlir.PrimitiveType(value);
                case 'index': return new mlir.PrimitiveType(value);
                case 'bf16':
                case 'f16':
                case 'f32':
                case 'f64':
                case 'f80':
                case 'f128':
                case 'tf32':
                case 'f8E5M2':
                case 'f8E4M3':
                case 'f8E4M3FN':
                case 'f8E5M2FNUZ':
                case 'f8E4M3FNUZ':
                case 'f8E4M3B11FNUZ':
                case 'f8E3M4':
                case 'f8E8M0FNU':
                case 'f4E2M1FN':
                case 'f6E2M3FN':
                case 'f6E3M2FN':
                    return new mlir.PrimitiveType(value);
                default:
                    if (/^[su]?i[0-9]+$/.test(value)) {
                        return new mlir.PrimitiveType(value);
                    }
                    break;
            }
        }
        if (this.match('!')) {
            return this.parseExtendedType();
        }
        throw new mlir.Error(`Invalid type '${this._token.value}' ${this.location()}`);
    }

    parseExtendedType() {
        return this.parseExtendedSymbol('!', this._state.typeAliasDefinitions, (dialectName, symbolData) => {
            const dialect = this._context.getDialect(dialectName);
            if (!dialect) {
                throw new mlir.Error(`Unsupported dialect '${dialectName}'.`);
            }
            if (symbolData) {
                return new mlir.Type(`!${dialectName}${symbolData}`);
            }
            const type = dialect.parseType(this, dialectName);
            if (type) {
                return type;
            }
            throw new mlir.Error(`Invalid type '!${dialectName}' ${this.location()}`);
        });
    }

    parseExtendedSymbol(prefix, aliases, createSymbol) {
        const token = this.expect(prefix);
        if (aliases.has(token)) {
            const alias = aliases.get(token);
            return alias instanceof mlir.Type ? alias : new mlir.Type(alias);
        }
        const identifier = token.substring(1);
        const dotIndex = identifier.indexOf('.');
        const isPrettyName = dotIndex !== -1 || identifier.endsWith('.') || this.match('.');
        const hasTrailingData = this.match('<');
        if (!hasTrailingData && !isPrettyName) {
            throw new mlir.Error(`Undefined symbol alias '${identifier}' ${this.location()}`);
        }
        let dialectName = null;
        let symbolData = null;
        if (isPrettyName) {
            if (dotIndex !== -1) {
                dialectName = identifier.substring(0, dotIndex);
            } else if (identifier.endsWith('.')) {
                dialectName = identifier.substring(0, identifier.length - 1);
            } else {
                dialectName = identifier;
                this.expect('.');
            }
            if (hasTrailingData) {
                symbolData = this.skip('<');
            }
        } else {
            // Verbose form: !dialect<...>
            dialectName = identifier;
            symbolData = this.skip('<');
        }
        return createSymbol(dialectName, symbolData);
    }

    parseFunctionType() {
        const inputs = this.parseTypeListParens();
        this.expect('->');
        const results = this.parseFunctionResultTypes();
        return new mlir.FunctionType(inputs, results);
    }

    parseCommaSeparatedList(delimiter, parseElement) {
        const results = [];
        const delimiters = {
            none: [null, null],
            paren: ['(', ')'],
            square: ['[', ']'],
            angle: ['<', '>'],
            brace: ['{', '}'],
            optionalParen: ['(', ')'],
            optionalSquare: ['[', ']'],
            optionalAngle: ['<', '>'],
            optionalBrace: ['{', '}']
        };
        const [open, close] = delimiters[delimiter] || [null, null];
        const isOptional = delimiter && delimiter.startsWith('optional');
        if (open) {
            if (isOptional) {
                if (!this.accept(open)) {
                    return results;
                }
            } else {
                this.expect(open);
            }
            if (close && this.accept(close)) {
                return results;
            }
        }
        const first = parseElement();
        if (first !== null && first !== undefined) {
            results.push(first);
        }
        while (this.accept(',')) {
            const elem = parseElement();
            if (elem !== null && elem !== undefined) {
                results.push(elem);
            }
        }
        if (close) {
            this.expect(close);
        }
        return results;
    }

    parseFunctionResultTypes() {
        if (this.match('(')) {
            return this.parseTypeListParens();
        }
        const type = this.parseNonFunctionType();
        return type ? [type] : [];
    }

    parseColonType() {
        this.expect(':');
        return this.parseType();
    }

    parseColonTypeList() {
        this.expect(':');
        return this.parseTypeList();
    }

    parseOptionalColonTypeList() {
        if (this.accept(':')) {
            return this.parseTypeList();
        }
        return [];
    }

    parseArrowTypeList() {
        this.expect('->');
        return this.parseFunctionResultTypes();
    }

    parseOptionalArrowTypeList() {
        if (this.accept('->')) {
            return this.parseFunctionResultTypes();
        }
        return [];
    }

    parseOptionalArrow() {
        return this.accept('->');
    }

    parseOptionalArrowResultTypes(op) {
        if (!this.accept('->')) {
            return false;
        }
        const types = this.parseFunctionResultTypes();
        this.resolveOperands(op.results, types);
        return true;
    }

    // Matches call_interface_impl::parseFunctionSignature from CallInterfaces.cpp
    // Parses: (type {attr}, type {attr}, ...) -> (type {attr}, ...)
    // Returns { argTypes: [...], argAttrs: [...], resultTypes: [...], resultAttrs: [...] }
    parseFunctionSignature(argOperands) {
        const argTypes = [];
        const argAttrs = [];
        const resultTypes = [];
        const resultAttrs = [];
        this.expect('(');
        if (!this.match(')')) {
            this.parseTypeAndAttrList(argTypes, argAttrs, argOperands);
        }
        this.expect(')');
        if (this.accept('->')) {
            this.parseFunctionResultList(resultTypes, resultAttrs);
        }
        return { argTypes, argAttrs, resultTypes, resultAttrs };
    }

    parseTypeAndAttrList(types, attrs, operands) {
        let index = 0;
        this.parseCommaSeparatedList('none', () => {
            const type = this.parseType();
            types.push(type);
            // Parse optional attribute dict after each type
            if (this.match('{')) {
                const attrList = new Map();
                this.parseAttributeDict(attrList);
                attrs.push(attrList);
                // Associate attrs with operand if available
                if (operands && index < operands.length) {
                    operands[index].attributes = attrList;
                }
            } else {
                attrs.push(null);
            }
            index++;
            return true;
        });
    }

    parseFunctionResultList(types, attrs) {
        if (this.accept('(')) {
            if (this.accept(')')) {
                return;
            }
            this.parseTypeAndAttrList(types, attrs);
            this.expect(')');
        } else {
            const type = this.parseType();
            types.push(type);
            attrs.push(null);
        }
    }

    parseSSAUse(allowResultNumber) {
        allowResultNumber = allowResultNumber === undefined ? true : allowResultNumber;
        const result = {};
        result.name = this._token.value;
        result.number = 0;
        result.location = this.location();
        this.expect('%');
        if (this.match('#')) {
            if (!allowResultNumber) {
                throw new mlir.Error(`Result number not allowed in argument list ${this.location()}`);
            }
            const hashValue = this.expect('#');
            const numberStr = hashValue.substring(1);
            const number = parseInt(numberStr, 10);
            if (isNaN(number)) {
                throw new mlir.Error(`Invalid SSA value result number ${this.location()}`);
            }
            result.number = number;
        }
        return result;
    }

    parseOperand(allowResultNumber) {
        allowResultNumber = allowResultNumber === undefined ? true : allowResultNumber;
        return this.parseSSAUse(allowResultNumber);
    }

    parseOptionalOperand(allowResultNumber = true) {
        if (this.match('%')) {
            return this.parseOperand(allowResultNumber);
        }
        return null;
    }

    parseAttribute(type = null) {
        if (this.match('id', 'affine_map') || this.match('id', 'affine_set')) {
            const name = this.expect();
            const args = this.skip('<');
            return { value: `${name}${args}` };
        }
        if (this.match('[')) {
            this.expect('[');
            const elements = [];
            while (!this.accept(']')) {
                const item = this.parseAttribute();
                elements.push(item);
                this.accept(',');
            }
            // Handle special `[a] x [b]` syntax (dialect-specific)
            if (this.accept('id', 'x')) {
                const firstArray = new mlir.ArrayAttr(elements);
                this.expect('[');
                const second = [];
                while (!this.accept(']')) {
                    const item = this.parseAttribute();
                    second.push(item);
                    this.accept(',');
                }
                const secondArray = new mlir.ArrayAttr(second);
                return new mlir.ArrayAttr([firstArray, secondArray]);
            }
            return new mlir.ArrayAttr(elements);
        }
        if (this.match('boolean')) {
            const value = this.expect();
            return new mlir.TypedAttr(value, new mlir.PrimitiveType('i1'));
        }
        if (this.match('id', 'dense')) {
            return this.parseDenseElementsAttr(type);
        }
        if (this.match('id', 'dense_resource')) {
            return this.parseDenseResourceElementsAttr(type);
        }
        if (this.match('id', 'array')) {
            return this.parseDenseArrayAttr(type);
        }
        if (this.match('{')) {
            const attributes = new Map();
            this.parseAttributeDict(attributes);
            return new mlir.DictionaryAttr(attributes);
        }
        if (this.match('#')) {
            const attr = this.parseExtendedAttr();
            if (!type && this.accept(':')) {
                attr.type = this.parseType();
            }
            return attr;
        }
        const parseType = (type, defaultType) => {
            if (type) {
                return type;
            }
            return this.accept(':') ? this.parseType() : defaultType;
        };
        if (this.match('float')) {
            const value = this.expect();
            type = parseType(type, new mlir.PrimitiveType('f64'));
            return new mlir.TypedAttr(value, type);
        }
        if (this.match('int')) {
            const value = this.expect();
            type = parseType(type, new mlir.PrimitiveType('i64'));
            return new mlir.TypedAttr(value, type);
        }
        if (this.match('keyword', '-')) {
            this.expect();
            if (this.match('int')) {
                const value = `-${this.expect()}`;
                type = parseType(type, new mlir.PrimitiveType('i64'));
                return new mlir.TypedAttr(value, type);
            }
            if (this.match('float')) {
                const value = `-${this.expect()}`;
                type = parseType(type, new mlir.PrimitiveType('f64'));
                return new mlir.TypedAttr(value, type);
            }
            throw new mlir.Error(`Expected integer or float after '-' ${this.location()}`);
        }
        if (this.match('keyword', 'loc')) {
            return this.parseLocation();
        }
        if (this.match('id', 'sparse')) {
            return this.parseSparseElementsAttr(type);
        }
        if (this.match('id', 'strided')) {
            return this.parseStridedLayoutAttr();
        }
        if (this.match('id', 'distinct')) {
            return this.parseDistinctAttr(type);
        }
        if (this.match('string')) {
            const value = this.expect();
            type = parseType(type, new mlir.PrimitiveType('string'));
            return new mlir.TypedAttr(value, type);
        }
        if (this.match('@')) {
            const value = this.parseOptionalSymbolName();
            // Handle scoped/nested symbol references like @module::@function
            let fullSymbol = value;
            while (this.accept('::')) {
                if (this.match('@')) {
                    const nested = this.parseOptionalSymbolName();
                    if (nested) {
                        fullSymbol += `::@${nested}`;
                    }
                } else {
                    break;
                }
            }
            return new mlir.SymbolRefAttr(fullSymbol);
        }
        if (this.match('id', 'unit')) {
            this.expect('id');
            return { value: 'unit', type: new mlir.PrimitiveType('unit') };
        }
        if (this._token.kind === 'id') {
            const tokenValue = this._token.value;
            if (tokenValue === 'tensor' || tokenValue === 'vector' || tokenValue === 'memref' ||
                tokenValue === 'none' || tokenValue === 'index' || /^[su]?i[0-9]+$/.test(tokenValue) ||
                /^f[0-9]+$/.test(tokenValue) || tokenValue === 'bf16' || tokenValue === 'tf32' ||
                tokenValue.startsWith('f8')) {
                const type = this.parseType();
                return { value: type, type: new mlir.PrimitiveType('type') };
            }
        }
        if (this.match('!')) {
            const type = this.parseType();
            return { value: type, type: new mlir.PrimitiveType('type') };
        }
        if (this.match('%')) {
            const value = this.expect();
            return { value };
        }
        if (this.match('id', 'DEFAULT')) {
            const value = this.expect();
            return { value };
        }
        if (this.match('<')) {
            const value = this.skip('<');
            return { value };
        }
        if (this.match('id')) {
            const value = this.expect('id');
            if (this.match('<')) {
                return { value: value + this.skip('<') };
            }
            return { value };
        }
        const parsedType = this.parseOptionalType();
        if (parsedType) {
            return new mlir.TypeAttrOf(parsedType);
        }
        throw new mlir.Error(`Unexpected attribute token '${this._token.value}' ${this.location()}`);
    }

    // Reference: DialectSymbolParser.cpp parseExtendedSymbol
    // Parses #dialect.symbol or #alias - does NOT parse trailing : type
    parseExtendedAttr() {
        const name = this.expect('#');
        // Check if this is an attribute alias reference (no '.' means it's an alias, not a dialect attribute)
        // Reference: Parser.cpp parseLocationAlias - looks up in state.symbols.attributeAliasDefinitions
        if (!name.includes('.') && this._state.attributeAliasDefinitions.has(name)) {
            return this._state.attributeAliasDefinitions.get(name);
        }
        let value = name;
        if (this.match('<')) {
            const body = this.skip('<');
            value = name + body;
        } else if (this.match('(')) {
            const body = this.skip('(');
            value = name + body;
        }
        return { value };
    }

    parseDenseElementsAttr(attrType) {
        this.expect('id');
        this.expect('<');
        let literalParser = null;
        if (!this.accept('>')) {
            literalParser = new mlir.TensorLiteralParser(this);
            literalParser.parse(/* allowHex */ true);
            this.expect('>');
        }
        const type = this.parseElementsLiteralType(attrType);
        const value = literalParser ? literalParser.getAttr(type) : null;
        return new mlir.DenseElementsAttr(value, type);
    }

    parseDenseResourceElementsAttr(attrType) {
        this.expect('id', 'dense_resource');
        this.expect('<');
        const handle = this.expect();
        this.expect('>');
        let type = attrType;
        if (!type) {
            this.expect(':');
            type = this.parseType();
        }
        return new mlir.DenseResourceElementsAttr(handle, type);
    }

    parseDenseArrayAttr(/* attrType */) {
        this.expect('id', 'array');
        this.expect('<');
        const arrayType = this.parseType();
        const arrayValues = [];
        if (this.accept(':')) {
            while (!this.match('>')) {
                const val = this.parseAttribute();
                arrayValues.push(val && val.value !== undefined ? val.value : val);
                this.accept(',');
            }
        }
        this.expect('>');
        return new mlir.DenseArrayAttr(arrayValues, arrayType);
    }

    parseSparseElementsAttr(attrType) {
        this.expect('id'); // consume 'sparse'
        this.expect('<');
        let indices = null;
        let values = null;
        if (!this.accept('>')) {
            const indiceParser = new mlir.TensorLiteralParser(this);
            indiceParser.parse(/* allowHex */ false);
            indices = indiceParser._storage;
            this.expect(',');
            const valuesParser = new mlir.TensorLiteralParser(this);
            valuesParser.parse(/* allowHex */ true);
            values = valuesParser._storage;
            this.expect('>');
        }
        const type = this.parseElementsLiteralType(attrType);
        return { value: { indices, values }, type };
    }

    parseStridedLayoutAttr() {
        this.expect('id', 'strided');
        const body = this.skip('<');
        return { value: `strided${body}`, type: 'strided' };
    }

    parseDistinctAttr(type) {
        this.expect('id', 'distinct');
        const id = this.skip('[');
        this.expect('<');
        let referencedAttr = null;
        if (!this.match('>')) {
            referencedAttr = this.parseAttribute(type);
        }
        this.expect('>');
        return { value: `distinct${id}`, referencedAttr, type: 'distinct' };
    }

    parseElementsLiteralType(type) {
        if (!type) {
            this.expect(':');
            return this.parseType();
        }
        return type;
    }

    parseOptionalAttribute(type) {
        switch (this._token.kind) {
            case '@':
            case '%':
            case 'int':
            case 'float':
            case '#':
            case '[':
            case '{':
            case '<':
            case 'string':
            case 'boolean':
                return this.parseAttribute(type);
            case 'keyword':
                if (this._token.value === '-' || this._token.value === 'loc') {
                    return this.parseAttribute(type);
                }
                return null;
            case 'id':
                if (this._token.value === 'affine_map' || this._token.value === 'affine_set' ||
                    this._token.value === 'dense' || this._token.value === 'dense_resource' ||
                    this._token.value === 'array' || this._token.value === 'sparse' ||
                    this._token.value === 'strided' || this._token.value === 'distinct' ||
                    this._token.value === 'unit' || this._token.value === 'DEFAULT') {
                    return this.parseAttribute(type);
                }
                // Fall through to default for type attributes
            default: {
                const value = this.parseOptionalType(type);
                if (value) {
                    return { value, type: 'type' };
                }
                return null;
            }
        }
    }

    parseKeyword() {
        return this.accept('id');
    }

    parseOptionalKeyword(values) {
        if (this.match('id')) {
            if (values.some((v) => this._token.value === v)) {
                return this.expect('id');
            }
        }
        return null;
    }

    parseInteger() {
        const value = this.expect('int');
        return parseInt(value, 10);
    }

    parseOptionalInteger() {
        if (this.match('int')) {
            return this.parseInteger();
        }
        return null;
    }

    parseString() {
        return this.expect('string');
    }

    parseOptionalString() {
        return this.accept('string');
    }

    getToken() {
        return this._token;
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

mlir.TensorLiteralParser = class {

    constructor(parser) {
        this._parser = parser;
        this._storage = [];
        this._shape = [];
    }

    parse(allowHex) {
        // If hex is allowed, check for a string literal.
        if (allowHex && this._parser.match('string')) {
            const hexStr = this._parser.expect();
            if (hexStr.startsWith('0x')) {
                const cleanHex = hexStr.replace(/"/g, '').substring(2);
                const data = new Uint8Array(cleanHex.length >> 1);
                for (let i = 0; i < data.length; i++) {
                    const index = i << 1;
                    data[i] = parseInt(cleanHex.substring(index, index + 2), 16);
                }
                this._storage = data;
                return { storage: data, shape: null };
            }
            this._storage.push(hexStr);
            return { storage: this._storage, shape: this._shape };
        }
        if (this._parser.match('[')) {
            this._parseList(this._shape);
        } else {
            this._parseElement();
            // Single element parsed without list - shape stays empty (splat)
        }
        return { storage: this._storage, shape: this._shape };
    }

    _parseList(dims) {
        this._parser.expect('[');
        let first = true;
        let newDims = [];
        let size = 0;
        while (!this._parser.accept(']')) {
            const thisDims = [];
            if (this._parser.match('[')) {
                this._parseList(thisDims);
            } else {
                this._parseElement();
            }
            size++;
            if (!first) {
                const compareDims = (a, b) =>{
                    if (a.length !== b.length) {
                        return false;
                    }
                    for (let i = 0; i < a.length; i++) {
                        if (a[i] !== b[i]) {
                            return false;
                        }
                    }
                    return true;
                };
                // Verify consistent dimensions (reference checks prevDims == newDims)
                const dimsMatch = compareDims(thisDims, newDims);
                if (!dimsMatch) {
                    throw new mlir.Error(`Invalid tensor literal ${this._parser.location()}`);
                }
            }
            newDims = thisDims;
            first = false;
            this._parser.accept(',');
        }
        dims.length = 0;
        dims.push(size);
        dims.push(...newDims);
    }

    _parseElement() {
        if (this._parser.accept('(')) {
            this._parseElement();
            this._parser.expect(',');
            this._parseElement();
            this._parser.expect(')');
            return;
        }
        if (this._parser.match('boolean')) {
            this._storage.push(this._parser.expect());
            return;
        }
        if (this._parser.accept('keyword', '-')) {
            if (this._parser.match('int')) {
                this._storage.push(`-${this._parser.expect()}`);
                return;
            }
            if (this._parser.match('float')) {
                this._storage.push(`-${this._parser.expect()}`);
                return;
            }
            throw new mlir.Error(`Expected integer or float after '-' ${this._parser.location()}`);
        }
        if (this._parser.match('int') || this._parser.match('float')) {
            this._storage.push(this._parser.expect());
            return;
        }
        if (this._parser.match('string')) {
            this._storage.push(this._parser.expect());
            return;
        }
        throw new mlir.Error(`Expected element literal of primitive type ${this._parser.location()}`);
    }

    getShape() {
        return this._shape;
    }

    getAttr(type) {
        if (this._storage instanceof Uint8Array) {
            return this._storage;
        }
        const elementType = type.getElementType ? type.getElementType() : null;
        const numElements = type.getNumElements ? type.getNumElements() : 0;
        const isComplex = elementType instanceof mlir.ComplexType;
        const baseElemType = isComplex && elementType.elementType ? elementType.elementType : elementType;
        // Determine conversion function once based on element type
        let convert = (v) => v;
        if (baseElemType) {
            const typeStr = baseElemType.toString();
            const intMatch = typeStr.match(/^[su]?i(\d+)$/);
            if (intMatch) {
                const bitWidth = parseInt(intMatch[1], 10);
                if (bitWidth >= 64) {
                    convert = (v) => typeof v === 'bigint' ? v : BigInt(v);
                }
                // For smaller ints, values are already numbers from tokenizer
            } else if (typeStr === 'index') {
                convert = (v) => typeof v === 'bigint' ? v : BigInt(v);
            }
            // For floats and other types, values are already correct from tokenizer
        }
        // Handle zero-element tensors (e.g., tensor<2x0x3xi4>)
        if (numElements === 0) {
            return [];
        }
        // Limit splat expansion to avoid memory issues with huge tensors
        const maxSplatExpansion = 10000000;
        // Handle splats - Reference: if shape.empty() and storage has elements, it's a splat
        const isSplat = this._shape.length === 0 && this._storage.length > 0;
        if (isSplat && numElements > 1) {
            if (numElements > maxSplatExpansion) {
                // Too large to expand - return null to indicate we can't provide the data
                return null;
            }
            if (isComplex && this._storage.length === 2) {
                // Complex splat: storage has 2 elements (real, imag)
                const result = [];
                const real = convert(this._storage[0]);
                const imag = convert(this._storage[1]);
                for (let i = 0; i < numElements; i++) {
                    result.push(new base.Complex(real, imag));
                }
                return result;
            }
            // Regular splat: replicate single value
            const converted = convert(this._storage[0]);
            return new Array(numElements).fill(converted);
        }
        // Non-splat complex: convert pairs to base.Complex objects
        if (isComplex && Array.isArray(this._storage)) {
            const result = [];
            for (let i = 0; i < this._storage.length; i += 2) {
                result.push(new base.Complex(convert(this._storage[i]), convert(this._storage[i + 1])));
            }
            return result;
        }
        // Convert all values
        return this._storage.map(convert);
    }
};

mlir.AttrTypeReader = class {

    constructor(bytecodeReader) {
        this._bytecodeReader = bytecodeReader;
        this._attrEntries = [];
        this._typeEntries = [];
        this._dataStart = 0;
    }

    initialize(attrEntries, typeEntries, dataStart) {
        this._attrEntries = attrEntries;
        this._typeEntries = typeEntries;
        this._dataStart = dataStart;
    }

    readAttribute(index) {
        if (index >= this._attrEntries.length) {
            return { name: 'local', value: `<local attr ${index}>` };
        }
        const entry = this._attrEntries[index];
        if (entry.resolved !== null) {
            return entry.resolved;
        }
        const reader = this._bytecodeReader._reader;
        const savedPosition = reader.position;
        reader.seek(this._dataStart + entry.offset);
        if (entry.hasCustomEncoding) {
            if (entry.dialect.name === 'builtin') {
                entry.resolved = this._parseBuiltinAttribute(reader);
            } else {
                entry.resolved = { name: 'custom', value: `<${entry.dialect.name}>` };
            }
        } else {
            // ASM format - null-terminated string
            let str = '';
            let c = '';
            while ((c = reader.byte()) !== 0) {
                str += String.fromCharCode(c);
            }
            entry.resolved = { name: 'asm', value: str };
        }
        reader.seek(savedPosition);
        return entry.resolved;
    }

    readType(index) {
        if (index >= this._typeEntries.length) {
            return new mlir.Type(`<local type ${index}>`);
        }
        const entry = this._typeEntries[index];
        if (entry.resolved !== null) {
            return entry.resolved;
        }
        const reader = this._bytecodeReader._reader;
        const savedPosition = reader.position;
        reader.seek(this._dataStart + entry.offset);
        if (entry.hasCustomEncoding) {
            if (entry.dialect.name === 'builtin') {
                entry.resolved = this._parseBuiltinType(reader);
            } else {
                entry.resolved = new mlir.Type(`!${entry.dialect.name}.custom`);
            }
        } else {
            // ASM format - null-terminated string
            let str = '';
            let c = '';
            while ((c = reader.byte()) !== 0) {
                str += String.fromCharCode(c);
            }
            entry.resolved = new mlir.Type(str);
        }
        reader.seek(savedPosition);
        return entry.resolved;
    }

    _parseBuiltinAttribute(reader) {
        // Builtin dialect attribute type codes (from BuiltinDialectBytecode.td):
        // 0 = ArrayAttr, 1 = DictionaryAttr, 2 = StringAttr, 3 = StringAttrWithType,
        // 4 = FlatSymbolRefAttr, 5 = SymbolRefAttr, 6 = TypeAttr, 7 = UnitAttr,
        // 8 = IntegerAttr, 9 = FloatAttr, 10-16 = locations, 17 = DenseArrayAttr,
        // 18 = DenseElementsAttr, 19 = DenseStringElementsAttr, 20 = DenseResourceElementsAttr
        const typeCode = reader.varint().toNumber();
        switch (typeCode) {
            case 0: { // ArrayAttr
                const count = reader.varint().toNumber();
                const elements = [];
                for (let i = 0; i < count; i++) {
                    const attrIdx = reader.varint().toNumber();
                    elements.push(this.readAttribute(attrIdx));
                }
                return new mlir.ArrayAttr(elements);
            }
            case 1: { // DictionaryAttr
                const count = reader.varint().toNumber();
                const attrs = new Map();
                for (let i = 0; i < count; i++) {
                    const nameAttrIdx = reader.varint().toNumber();
                    const nameAttr = this.readAttribute(nameAttrIdx);
                    const valueAttrIdx = reader.varint().toNumber();
                    const valueAttr = this.readAttribute(valueAttrIdx);
                    const name = nameAttr && nameAttr.value ? nameAttr.value : `attr_${i}`;
                    attrs.set(name, valueAttr);
                }
                return { name: 'dictionary', value: attrs };
            }
            case 2: { // StringAttr
                const strIdx = reader.varint().toNumber();
                const value = this._bytecodeReader._stringReader.getString(strIdx);
                return new mlir.StringAttr(value);
            }
            case 3: { // StringAttrWithType
                const strIdx = reader.varint().toNumber();
                reader.varint(); // skip type
                const value = this._bytecodeReader._stringReader.getString(strIdx);
                return new mlir.StringAttr(value);
            }
            case 4: { // FlatSymbolRefAttr
                const strIdx = reader.varint().toNumber();
                const value = this._bytecodeReader._stringReader.getString(strIdx);
                return new mlir.SymbolRefAttr(`@${value}`);
            }
            case 5: { // SymbolRefAttr
                const rootIdx = reader.varint().toNumber();
                const root = this._bytecodeReader._stringReader.getString(rootIdx);
                const numNested = reader.varint().toNumber();
                let value = `@${root}`;
                for (let i = 0; i < numNested; i++) {
                    const nestedIdx = reader.varint().toNumber();
                    const nested = this._bytecodeReader._stringReader.getString(nestedIdx);
                    value += `::@${nested}`;
                }
                return new mlir.SymbolRefAttr(value);
            }
            case 6: { // TypeAttr
                const typeIdx = reader.varint().toNumber();
                const type = this.readType(typeIdx);
                return new mlir.TypeAttrOf(type);
            }
            case 7: { // UnitAttr
                return new mlir.UnitAttr();
            }
            case 8: { // IntegerAttr
                const typeIdx = reader.varint().toNumber();
                const type = this.readType(typeIdx);
                // Read value based on bit width (ref: BytecodeReader.cpp:1145-1171)
                // - bitWidth <= 8: single byte
                // - bitWidth <= 64: signed varint
                // - larger: word count + words
                const bitWidth = this._getIntegerBitWidth(type);
                let value = null;
                if (bitWidth <= 8) {
                    value = BigInt(reader.byte());
                } else if (bitWidth <= 64) {
                    value = reader.svarint();
                } else {
                    // Large integers: read word count, then words
                    const numWords = reader.varint().toNumber();
                    value = 0n;
                    for (let i = 0; i < numWords; i++) {
                        const word = reader.svarint();
                        value |= (word << BigInt(i * 64));
                    }
                }
                return new mlir.IntegerAttr(value, type);
            }
            case 9: { // FloatAttr
                const typeIdx = reader.varint().toNumber();
                const type = this.readType(typeIdx);
                const value = this._readFloatValue(reader, type.toString());
                return new mlir.FloatAttr(value, type);
            }
            case 10: { // CallSiteLoc
                const callerIdx = reader.varint().toNumber();
                const calleeIdx = reader.varint().toNumber();
                return { name: 'loc', value: `callsite(${callerIdx} at ${calleeIdx})` };
            }
            case 11: { // FileLineColLoc
                const filenameIdx = reader.varint().toNumber();
                const filename = this._bytecodeReader._stringReader.getString(filenameIdx);
                const line = reader.varint().toNumber();
                const col = reader.varint().toNumber();
                return { name: 'loc', value: `${filename}:${line}:${col}` };
            }
            case 12: { // FusedLoc
                const count = reader.varint().toNumber();
                for (let i = 0; i < count; i++) {
                    reader.varint(); // skip location indices
                }
                return { name: 'loc', value: 'fused' };
            }
            case 13: { // FusedLocWithMetadata
                const metadataIdx = reader.varint().toNumber();
                const count = reader.varint().toNumber();
                for (let i = 0; i < count; i++) {
                    reader.varint(); // skip location indices
                }
                return { name: 'loc', value: `fused<${metadataIdx}>` };
            }
            case 14: { // NameLoc
                const nameAttrIdx = reader.varint().toNumber();
                const childLocIdx = reader.varint().toNumber();
                return { name: 'loc', value: `name(${nameAttrIdx}, ${childLocIdx})` };
            }
            case 15: { // OpaqueLoc
                const underlyingIdx = reader.varint().toNumber();
                const fallbackIdx = reader.varint().toNumber();
                return { name: 'loc', value: `opaque(${underlyingIdx}, ${fallbackIdx})` };
            }
            case 16: { // UnknownLoc
                return { name: 'loc', value: 'unknown' };
            }
            case 17: { // DenseArrayAttr
                const typeIdx = reader.varint().toNumber();
                const type = this.readType(typeIdx);
                const size = reader.varint().toNumber();
                const blobLen = reader.varint().toNumber();
                const blob = reader.read(blobLen);
                return this._parseDenseArrayData(blob, type, size);
            }
            case 18: { // DenseElementsAttr
                const typeIdx = reader.varint().toNumber();
                const type = this.readType(typeIdx);
                const blobLen = reader.varint().toNumber();
                const blob = reader.read(blobLen);
                return new mlir.DenseElementsAttr(blob, type);
            }
            case 19: { // DenseStringElementsAttr
                const typeIdx = reader.varint().toNumber();
                const type = this.readType(typeIdx);
                const isSplat = reader.varint().toNumber() !== 0;
                const count = reader.varint().toNumber();
                const strings = [];
                for (let i = 0; i < count; i++) {
                    const strIdx = reader.varint().toNumber();
                    strings.push(this._bytecodeReader._stringReader.getString(strIdx));
                }
                return { name: 'dense_string', value: strings, type, isSplat };
            }
            case 20: { // DenseResourceElementsAttr
                const typeIdx = reader.varint().toNumber();
                const type = this.readType(typeIdx);
                const handleIdx = reader.varint().toNumber();
                return new mlir.DenseResourceElementsAttr(`resource<${handleIdx}>`, type);
            }
            default: {
                return { name: 'builtin', value: `<builtin code ${typeCode}>` };
            }
        }
    }

    _parseDenseArrayData(blob, type, size) {
        const typeStr = type.toString();
        const view = new DataView(blob.buffer, blob.byteOffset, blob.length);
        const values = [];
        if (typeStr.startsWith('i') || typeStr.startsWith('si') || typeStr.startsWith('ui')) {
            const match = typeStr.match(/[su]?i(\d+)/);
            const bitWidth = match ? parseInt(match[1], 10) : 64;
            const byteWidth = Math.ceil(bitWidth / 8);
            for (let i = 0; i < size && i * byteWidth < blob.length; i++) {
                if (bitWidth <= 8) {
                    values.push(view.getInt8(i * byteWidth));
                } else if (bitWidth <= 16) {
                    values.push(view.getInt16(i * byteWidth, true));
                } else if (bitWidth <= 32) {
                    values.push(view.getInt32(i * byteWidth, true));
                } else {
                    values.push(view.getBigInt64(i * byteWidth, true));
                }
            }
        } else if (typeStr === 'f32') {
            for (let i = 0; i < size && i * 4 < blob.length; i++) {
                values.push(view.getFloat32(i * 4, true));
            }
        } else if (typeStr === 'f64') {
            for (let i = 0; i < size && i * 8 < blob.length; i++) {
                values.push(view.getFloat64(i * 8, true));
            }
        } else if (typeStr === 'f16') {
            for (let i = 0; i < size && i * 2 < blob.length; i++) {
                values.push(view.getUint16(i * 2, true)); // Store raw bits for f16
            }
        } else {
            // Default to raw bytes
            return new mlir.DenseArrayAttr(blob, type);
        }
        return new mlir.DenseArrayAttr(values, type);
    }

    _getIntegerBitWidth(type) {
        // Extract bit width from integer type (i1, i8, i16, i32, i64, si32, ui64, etc.)
        const typeStr = type ? type.toString() : '';
        const match = typeStr.match(/^[su]?i(\d+)$/);
        if (match) {
            return parseInt(match[1], 10);
        }
        // Default to 64-bit for index and unknown types
        return 64;
    }

    _readFloatValue(reader, typeStr) {
        if (typeStr === 'f16' || typeStr === 'bf16') {
            const bits = reader.read(2);
            const view = new DataView(bits.buffer, bits.byteOffset, 2);
            return view.getUint16(0, true);
        }
        if (typeStr === 'f32') {
            const bits = reader.read(4);
            const view = new DataView(bits.buffer, bits.byteOffset, 4);
            return view.getFloat32(0, true);
        }
        // Default to 64-bit (f64, f80, f128)
        const bits = reader.read(8);
        const view = new DataView(bits.buffer, bits.byteOffset, 8);
        return view.getFloat64(0, true);
    }

    _parseBuiltinType(reader) {
        // Builtin dialect type codes (from BuiltinDialectBytecode.td):
        // See BuiltinDialectTypes enum
        const typeCode = reader.varint().toNumber();
        switch (typeCode) {
            case 0: { // IntegerType
                const widthAndSign = reader.varint().toNumber();
                const width = widthAndSign >> 2;
                const signedness = widthAndSign & 0x3;
                if (signedness === 0) {
                    return new mlir.Type(`i${width}`);
                }
                if (signedness === 1) {
                    return new mlir.Type(`si${width}`);
                }
                return new mlir.Type(`ui${width}`);
            }
            case 1: { // IndexType
                return new mlir.Type('index');
            }
            case 2: { // FunctionType
                const numInputs = reader.varint().toNumber();
                const inputs = [];
                for (let i = 0; i < numInputs; i++) {
                    const typeIdx = reader.varint().toNumber();
                    inputs.push(this.readType(typeIdx));
                }
                const numResults = reader.varint().toNumber();
                const results = [];
                for (let i = 0; i < numResults; i++) {
                    const typeIdx = reader.varint().toNumber();
                    results.push(this.readType(typeIdx));
                }
                const type = new mlir.FunctionType(inputs, results);
                return type;
            }
            case 3: { // BFloat16Type
                return new mlir.Type('bf16');
            }
            case 4: { // Float16Type
                return new mlir.Type('f16');
            }
            case 5: { // Float32Type
                return new mlir.Type('f32');
            }
            case 6: { // Float64Type
                return new mlir.Type('f64');
            }
            case 7: { // Float80Type
                return new mlir.Type('f80');
            }
            case 8: { // Float128Type
                return new mlir.Type('f128');
            }
            case 9: { // ComplexType
                const elementTypeIdx = reader.varint().toNumber();
                const elementType = this.readType(elementTypeIdx);
                return new mlir.Type(`complex<${elementType.toString()}>`);
            }
            case 10: { // MemRefType
                const shape = this._readShape(reader);
                const elementTypeIdx = reader.varint().toNumber();
                const elementType = this.readType(elementTypeIdx);
                // Skip layout and memory space for now
                return new mlir.Type(`memref<${shape.join('x')}x${elementType.toString()}>`);
            }
            case 11: { // MemRefTypeWithLayout - skip for now
                return new mlir.Type('memref<?>');
            }
            case 12: { // NoneType
                return new mlir.Type('none');
            }
            case 13: { // RankedTensorType
                const shape = this._readShape(reader);
                const elementTypeIdx = reader.varint().toNumber();
                const elementType = this.readType(elementTypeIdx);
                return new mlir.RankedTensorType(shape, elementType, null);
            }
            case 14: { // RankedTensorTypeWithEncoding
                const encodingAttrIdx = reader.varint().toNumber();
                const encoding = this.readAttribute(encodingAttrIdx);
                const shape = this._readShape(reader);
                const elementTypeIdx = reader.varint().toNumber();
                const elementType = this.readType(elementTypeIdx);
                return new mlir.RankedTensorType(shape, elementType, encoding);
            }
            case 15: { // TupleType
                const numTypes = reader.varint().toNumber();
                const types = [];
                for (let i = 0; i < numTypes; i++) {
                    const typeIdx = reader.varint().toNumber();
                    types.push(this.readType(typeIdx));
                }
                return new mlir.Type(`tuple<${types.map((t) => t.toString()).join(', ')}>`);
            }
            case 16: { // UnrankedMemRefType
                const elementTypeIdx = reader.varint().toNumber();
                const elementType = this.readType(elementTypeIdx);
                return new mlir.Type(`memref<*x${elementType.toString()}>`);
            }
            case 17: { // UnrankedTensorType
                const elementTypeIdx = reader.varint().toNumber();
                const elementType = this.readType(elementTypeIdx);
                return new mlir.Type(`tensor<*x${elementType.toString()}>`);
            }
            case 18: { // VectorType
                const shape = this._readShape(reader);
                const elementTypeIdx = reader.varint().toNumber();
                const elementType = this.readType(elementTypeIdx);
                return new mlir.VectorType(shape, elementType);
            }
            case 19: { // VectorTypeWithScalableDims - simplified
                return new mlir.Type('vector<?>');
            }
            default: {
                return new mlir.Type(`<builtin type ${typeCode}>`);
            }
        }
    }

    _readShape(reader) {
        const rank = reader.varint().toNumber();
        const shape = [];
        for (let i = 0; i < rank; i++) {
            // Dimensions are encoded as signed varints
            const dim = reader.svarint().toNumber();
            shape.push(dim);
        }
        return shape;
    }
};

mlir.StringSectionReader = class {

    constructor() {
        this._strings = [];
    }

    initialize(reader, section) {
        reader.seek(section.start);
        const lengths = new Array(reader.varint().toNumber());
        for (let i = 0; i < lengths.length; i++) {
            lengths[i] = reader.varint().toNumber();
        }
        const decoder = new TextDecoder('utf-8');
        this._strings = new Array(lengths.length);
        for (let i = 0; i < this._strings.length; i++) {
            const size = lengths[lengths.length - 1 - i];
            const buffer = reader.read(size);
            // Strings are null-terminated in bytecode, exclude the null character
            this._strings[i] = decoder.decode(buffer.subarray(0, size - 1));
        }
    }

    getString(index) {
        return index < this._strings.length ? this._strings[index] : '';
    }
};

mlir.ResourceSectionReader = class {

    constructor() {
        this._resources = [];
    }

    initialize(reader, section) {
        if (!section) {
            return;
        }
        reader.seek(section.start);
        const numExternalResourceGroups = reader.varint().toNumber();
        for (let i = 0; i < numExternalResourceGroups; i++) {
            reader.varint(); // key
            const numResources = reader.varint().toNumber();
            for (let j = 0; j < numResources; j++) {
                reader.varint(); // key
                reader.varint(); // offset
                reader.byte(); // kind
            }
        }
    }
};

mlir.BytecodeReader = class {

    constructor(reader, context) {
        this._reader = new mlir.BinaryReader(reader);
        this._context = context;
        this._valueScopes = [];
    }

    read() {
        const reader = this._reader;
        reader.read(4); // signature 'ML\xEFR'
        this.version = reader.varint().toNumber();
        this.producer = reader.string();
        this._sections = new Map();
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
                const alignment = reader.varint().toNumber();
                while (reader.position % alignment !== 0) {
                    reader.byte(); // skip 0xCB padding bytes
                }
            }
            const offset = reader.position;
            reader.skip(length);
            this._sections.set(sectionID, { start: offset, end: reader.position });
        }
        if (!this._sections.has(0) || !this._sections.has(1) ||
            !this._sections.has(2) || !this._sections.has(3) ||
            !this._sections.has(4) || (this.version >= 5 && !this._sections.has(8))) {
            throw new mlir.Error('Missing required section.');
        }
        // Initialize section readers
        this._stringReader = new mlir.StringSectionReader();
        this._stringReader.initialize(reader, this._sections.get(0));
        this._resourceReader = new mlir.ResourceSectionReader();
        this._resourceReader.initialize(reader, this._sections.get(6));
        this._parseDialectSection();
        this._parseAttrTypeSection();
        if (this._sections.has(8)) {
            this._parsePropertiesSection();
        }
        return this.parseIRSection();
    }

    _parseDialectSection() {
        const section = this._sections.get(1);
        const reader = this._reader;
        reader.seek(section.start);
        const numDialects = reader.varint().toNumber();
        this._dialects = new Array(numDialects);
        for (let i = 0; i < this._dialects.length; i++) {
            this._dialects[i] = {};
            if (this.version < 1) { // kDialectVersioning
                const entryIdx = reader.varint().toNumber();
                this._dialects[i].name = this._stringReader.getString(entryIdx);
                continue;
            }
            const nameAndIsVersioned = reader.varint();
            const dialectNameIdx = (nameAndIsVersioned >> 1n).toNumber();
            this._dialects[i].name = this._stringReader.getString(dialectNameIdx);
            if (nameAndIsVersioned & 1n) {
                const size = reader.varint().toNumber();
                this._dialects[i].version = reader.read(size);
            }
        }
        let numOps = -1;
        this._opNames = [];
        if (this.version > 4) { // kElideUnknownBlockArgLocation
            numOps = reader.varint().toNumber();
            this._opNames = new Array(numOps);
        }
        let i = 0;
        while (reader.position < section.end) {
            const dialect = this._dialects[reader.varint().toNumber()];
            const numEntries = reader.varint().toNumber();
            for (let j = 0; j < numEntries; j++) {
                const opName = {};
                if (this.version < 5) { // kNativePropertiesEncoding
                    opName.name = this._stringReader.getString(reader.varint().toNumber());
                    opName.dialect = dialect;
                } else {
                    const nameAndIsRegistered = reader.varint();
                    opName.name = this._stringReader.getString((nameAndIsRegistered >> 1n).toNumber());
                    opName.dialect = dialect;
                    opName.isRegistered = (nameAndIsRegistered & 1n) === 1n;
                }
                if (numOps < 0) {
                    this._opNames.push(opName);
                } else {
                    this._opNames[i++] = opName;
                }
            }
        }
    }

    _parseAttrTypeSection() {
        const section = this._sections.get(3);
        const reader = this._reader;
        reader.seek(section.start);
        const attrEntries = new Array(reader.varint().toNumber());
        const typeEntries = new Array(reader.varint().toNumber());
        let offset = 0;
        const parseEntries = (range) => {
            for (let i = 0; i < range.length;) {
                const dialect = this._dialects[reader.varint().toNumber()];
                const numEntries = reader.varint().toNumber();
                for (let j = 0; j < numEntries; j++) {
                    const entry = {};
                    const entrySizeWithFlag = reader.varint();
                    entry.hasCustomEncoding = (entrySizeWithFlag & 1n) === 1n;
                    entry.size = (entrySizeWithFlag >> 1n).toNumber();
                    entry.offset = offset;
                    entry.dialect = dialect;
                    entry.resolved = null;
                    offset += entry.size;
                    range[i++] = entry;
                }
            }
        };
        parseEntries(attrEntries);
        parseEntries(typeEntries);
        const dataStart = this._sections.get(2).start;
        // Initialize AttrTypeReader
        this._attrTypeReader = new mlir.AttrTypeReader(this);
        this._attrTypeReader.initialize(attrEntries, typeEntries, dataStart);
    }

    _parsePropertiesSection() {
        const section = this._sections.get(8);
        const reader = this._reader;
        reader.seek(section.start);
        const count = reader.varint().toNumber();
        this._properties = new Array(count);
        for (let i = 0; i < this._properties.length; i++) {
            const size = reader.varint().toNumber();
            const data = reader.read(size);
            this._properties[i] = data;
        }
    }

    parseIRSection() {
        const section = this._sections.get(4);
        const reader = this._reader;
        reader.seek(section.start);
        const block = { operations: [] };
        this._valueScopes = [[]];
        const regionStack = [{
            block,
            curRegion: 0,
            numRegions: 1,
            curBlock: 0,
            numBlocks: 1,
            numOpsRemaining: 0,
            numValues: 0,
            blocks: [block],
            nextValueIdx: 0,
            isTopLevel: true
        }];
        const firstBlockHeader = this.parseBlockHeader(reader);
        regionStack[0].numOpsRemaining = firstBlockHeader.numOps;
        if (firstBlockHeader.hasArgs) {
            const [scope] = this._valueScopes;
            this.parseBlockArguments(reader, block, scope, 0);
            regionStack[0].nextValueIdx = block.arguments ? block.arguments.length : 0;
        }
        while (regionStack.length > 0) {
            const state = regionStack[regionStack.length - 1];
            let pushedRegions = false;
            while (state.numOpsRemaining > 0 && !pushedRegions) {
                state.numOpsRemaining--;
                const op = this.parseOpWithoutRegions(reader, state);
                state.blocks[state.curBlock].operations.push(op);
                if (op.regions && op.regions.length > 0) {
                    for (let i = op.regions.length - 1; i >= 0; i--) {
                        const region = op.regions[i];
                        const regionReader = reader;
                        if (this.version >= 2 && op._isIsolatedFromAbove) {
                            const sectionIDAndHasAlignment = reader.byte();
                            /* const sectionID = sectionIDAndHasAlignment & 0x7F; */
                            /* const sectionLength = */ reader.varint().toNumber();
                            const hasAlignment = sectionIDAndHasAlignment & 0x80;
                            if (hasAlignment) {
                                const alignment = reader.varint().toNumber();
                                while (reader.position % alignment !== 0) {
                                    reader.byte(); // skip 0xCB padding bytes
                                }
                            }
                            // The inline IR section data starts directly with region data
                            // (numBlocks, numValues, block headers, etc.) - no nested sections
                        }
                        const numBlocks = regionReader.varint().toNumber();
                        if (numBlocks === 0) {
                            continue;
                        }
                        const numValues = regionReader.varint().toNumber();
                        const blocks = [];
                        for (let j = 0; j < numBlocks; j++) {
                            blocks.push({ operations: [], arguments: [] });
                        }
                        region.blocks = blocks;
                        if (op._isIsolatedFromAbove) {
                            this._valueScopes.push([]);
                        }
                        const scope = this._valueScopes[this._valueScopes.length - 1];
                        const valueOffset = scope.length;
                        for (let j = 0; j < numValues; j++) {
                            scope.push(null);
                        }
                        const blockHeader = this.parseBlockHeader(regionReader);
                        if (blockHeader.hasArgs) {
                            this.parseBlockArguments(regionReader, blocks[0], scope, valueOffset);
                        }
                        const numBlockArgs = blocks[0].arguments ? blocks[0].arguments.length : 0;
                        regionStack.push({
                            region,
                            curRegion: 0,
                            numRegions: 1,
                            curBlock: 0,
                            numBlocks,
                            numOpsRemaining: blockHeader.numOps,
                            numValues,
                            blocks,
                            valueOffset,
                            nextValueIdx: valueOffset + numBlockArgs,
                            isIsolated: op._isIsolatedFromAbove
                        });
                        pushedRegions = true;
                    }
                }
            }

            // If we pushed regions, continue outer loop to process them first
            if (pushedRegions) {
                continue;
            }

            // Check if we need to move to next block or pop the stack
            if (state.numOpsRemaining === 0) {
                state.curBlock++;
                if (state.curBlock < state.numBlocks) {
                    // Parse next block header
                    const blockHeader = this.parseBlockHeader(reader);
                    state.numOpsRemaining = blockHeader.numOps;
                    if (blockHeader.hasArgs) {
                        const scope = this._valueScopes[this._valueScopes.length - 1];
                        // Block arguments start at current nextValueIdx
                        const argOffset = state.nextValueIdx ?? 0;
                        this.parseBlockArguments(reader, state.blocks[state.curBlock], scope, argOffset);
                        // Update nextValueIdx to account for block arguments
                        const numBlockArgs = state.blocks[state.curBlock].arguments ? state.blocks[state.curBlock].arguments.length : 0;
                        if (state.nextValueIdx !== undefined) {
                            state.nextValueIdx += numBlockArgs;
                        }
                    }
                } else {
                    // Pop this region
                    if (state.isIsolated) {
                        this._valueScopes.pop();
                    }
                    regionStack.pop();
                }
            }
        }

        return block;
    }

    parseBlockHeader(reader) {
        const numOpsAndHasArgs = reader.varint();
        const numOps = (numOpsAndHasArgs >> 1n).toNumber();
        const hasArgs = (numOpsAndHasArgs & 1n) === 1n;
        return { numOps, hasArgs };
    }

    parseBlockArguments(reader, block, scope, valueOffset) {
        const numArgs = reader.varint().toNumber();
        block.arguments = [];
        for (let i = 0; i < numArgs; i++) {
            // Parse type and location flag: (typeIdx << 1) | hasLocation
            const typeAndLocation = reader.varint().toNumber();
            const typeIdx = typeAndLocation >> 1;
            const hasLocation = (typeAndLocation & 1) === 1;
            const type = this._attrTypeReader.readType(typeIdx);
            // Parse location if present
            let location = null;
            if (hasLocation) {
                const locIdx = reader.varint().toNumber();
                location = this._attrTypeReader.readAttribute(locIdx);
            }
            // Create block argument with name and value for graph linking
            const argName = `%${valueOffset + i}`;
            const arg = { type, location, name: argName, value: argName };
            block.arguments.push(arg);
            // Update the scope so operands can reference this argument
            if (scope && (valueOffset + i) < scope.length) {
                scope[valueOffset + i] = arg;
            }
        }
        // Use-list ordering (version >= 3) - stored after all arguments
        if (this.version >= 3) {
            // Read numUseListOrders - check for 0x00 marker which means skip
            const firstByte = reader.peek();
            if (firstByte === 0x00) {
                // Skip the 0x00 byte - appears to be a marker for "no use-lists"
                reader.byte();
            } else {
                const numUseListOrders = reader.varint().toNumber();
                for (let i = 0; i < numUseListOrders; i++) {
                    // Skip use-list entries: argIndex, then use-list data
                    /* const argIndex = */ reader.varint();
                    const indexBitWidth = reader.varint().toNumber();
                    if (indexBitWidth > 0) {
                        const numUses = reader.varint().toNumber();
                        for (let j = 0; j < numUses; j++) {
                            reader.varint(); // skip use index
                        }
                    }
                }
            }
        }
    }

    parseOpWithoutRegions(reader, state) {
        // Parse operation name index
        const opNameIdx = reader.varint().toNumber();
        const opNameEntry = this._opNames[opNameIdx];
        if (!opNameEntry) {
            throw new mlir.Error(`Invalid operation name index '${opNameIdx}' (have ${this._opNames.length} ops) at position ${reader.position}.`);
        }
        const fullName = `${opNameEntry.dialect.name}.${opNameEntry.name}`;

        // Parse operation mask
        const opMask = reader.byte();
        const kHasAttrs = 0x01;
        const kHasResults = 0x02;
        const kHasOperands = 0x04;
        const kHasSuccessors = 0x08;
        const kHasInlineRegions = 0x10;
        const kHasUseListOrders = 0x20;
        const kHasProperties = 0x40;

        const op = new mlir.Operation(fullName);
        const [dialectName] = fullName.split('.');
        const dialect = this._context.getDialect(dialectName);
        if (dialect) {
            const opInfo = dialect.getOperation(fullName);
            if (opInfo) {
                op.metadata = opInfo.metadata;
            }
        }

        // Parse location
        const locIdx = reader.varint().toNumber();
        op.location = this._attrTypeReader.readAttribute(locIdx);

        // Parse attributes
        if (opMask & kHasAttrs) {
            const dictAttrIdx = reader.varint().toNumber();
            const dictAttr = this._attrTypeReader.readAttribute(dictAttrIdx);
            if (dictAttr && dictAttr.value) {
                if (dictAttr.value instanceof Map) {
                    // Already parsed as Map from custom-encoded DictionaryAttr
                    op.attributes = dictAttr.value;
                } else if (typeof dictAttr.value === 'string') {
                    // Parse dictionary attribute from ASM string format
                    op.attributes = this.parseAttributeDict(dictAttr.value);
                }
            }
        }

        // Parse properties (version >= 5)
        if (opMask & kHasProperties) {
            if (opNameEntry.isRegistered) {
                // Native properties - read index into properties table
                const propIdx = reader.varint().toNumber();
                if (propIdx < this._properties.length) {
                    const propData = this._properties[propIdx];
                    if (propData.length > 0) {
                        this._parseNativeProperties(propData, op, fullName);
                    }
                }
            } else {
                // Unregistered operations store properties as a single dictionary attribute
                const propAttrIdx = reader.varint().toNumber();
                const propAttr = this._attrTypeReader.readAttribute(propAttrIdx);
                if (propAttr && propAttr.value) {
                    const propAttrs = this.parseAttributeDict(propAttr.value);
                    for (const [key, value] of propAttrs) {
                        op.addAttribute(key, value);
                    }
                }
            }
        }

        // Parse results
        if (opMask & kHasResults) {
            const numResults = reader.varint().toNumber();
            const scope = this._valueScopes[this._valueScopes.length - 1];
            for (let i = 0; i < numResults; i++) {
                const typeIdx = reader.varint().toNumber();
                const type = this._attrTypeReader.readType(typeIdx);
                // Use nextValueIdx from state if available, otherwise use scope.length
                const valueIdx = state && state.nextValueIdx !== undefined ? state.nextValueIdx++ : scope.length;
                const valueName = `%${valueIdx}`;
                const result = { type, name: valueName, value: valueName };
                op.results.push(result);
                // Update the reserved slot in scope
                if (valueIdx < scope.length) {
                    scope[valueIdx] = result;
                } else {
                    scope.push(result);
                }
            }
        }

        // Parse operands
        if (opMask & kHasOperands) {
            const numOperands = reader.varint().toNumber();
            for (let i = 0; i < numOperands; i++) {
                const valueIdx = reader.varint().toNumber();
                const scope = this._valueScopes[this._valueScopes.length - 1];
                if (valueIdx < scope.length && scope[valueIdx]) {
                    op.operands.push(scope[valueIdx]);
                } else {
                    op.operands.push({ name: `%${valueIdx}` });
                }
            }
        }

        // Parse successors
        if (opMask & kHasSuccessors) {
            const numSuccessors = reader.varint().toNumber();
            op.successors = [];
            for (let i = 0; i < numSuccessors; i++) {
                const blockIdx = reader.varint().toNumber();
                op.successors.push(blockIdx);
            }
        }
        // Parse use-list orders (version >= 3)
        if (this.version >= 3 && (opMask & kHasUseListOrders)) {
            const numResults = op.results.length;
            for (let i = 0; i < numResults; i++) {
                const indexBitWidth = reader.varint().toNumber();
                if (indexBitWidth > 0) {
                    const numUses = reader.varint().toNumber();
                    for (let j = 0; j < numUses; j++) {
                        reader.varint(); // use index
                    }
                }
            }
        }
        // Parse inline regions
        if (opMask & kHasInlineRegions) {
            const numRegionsAndIsIsolated = reader.varint();
            const numRegions = (numRegionsAndIsIsolated >> 1n).toNumber();
            const isIsolatedFromAbove = (numRegionsAndIsIsolated & 1n) === 1n;
            op._isIsolatedFromAbove = isIsolatedFromAbove;
            for (let i = 0; i < numRegions; i++) {
                op.regions.push({ blocks: [] });
            }
        }
        return op;
    }

    parseAttributeDict(str) {
        const attrs = new Map();
        // Parse dictionary attribute format: {key = value, ...}
        if (!str.startsWith('{') || !str.endsWith('}')) {
            return attrs;
        }
        const content = str.slice(1, -1).trim();
        if (content.length === 0) {
            return attrs;
        }
        let i = 0;
        while (i < content.length) {
            while (i < content.length && /\s/.test(content[i])) {
                i++;
            }
            if (i >= content.length) {
                break;
            }
            // Read key (alphanumeric/underscore)
            const keyStart = i;
            while (i < content.length && /[a-zA-Z0-9_]/.test(content[i])) {
                i++;
            }
            const key = content.slice(keyStart, i);
            if (!key) {
                break;
            }
            while (i < content.length && /\s/.test(content[i])) {
                i++;
            }
            if (i >= content.length || content[i] !== '=') {
                break;
            }
            i++; // skip '='
            while (i < content.length && /\s/.test(content[i])) {
                i++;
            }
            // Read value (until balanced comma or end)
            const valueStart = i;
            let depth = 0;
            let inString = false;
            while (i < content.length) {
                const c = content[i];
                if (inString) {
                    if (c === '"' && content[i - 1] !== '\\') {
                        inString = false;
                    }
                    i++;
                    continue;
                }
                if (c === '"') {
                    inString = true;
                    i++;
                    continue;
                }
                if (c === '{' || c === '[' || c === '(' || c === '<') {
                    depth++;
                } else if (c === '}' || c === ']' || c === ')' || c === '>') {
                    depth--;
                } else if (c === ',' && depth === 0) {
                    break;
                }
                i++;
            }
            const value = content.slice(valueStart, i).trim();
            attrs.set(key, { value, toString: () => value });
            if (i < content.length && content[i] === ',') {
                i++;
            }
        }
        return attrs;
    }

    _parseNativeProperties(data, op, fullName) {
        // Native properties encoding is operation-specific (generated by tablegen).
        // Without the generated code, we can only safely parse known operation types.
        // Reference: llvm-project/mlir/lib/Bytecode/Reader/BytecodeReader.cpp:1274-1282
        const propReader = new mlir.BufferReader(data);
        // Function operations: sym_name (required), function_type (required), then optional attrs
        if (fullName.endsWith('.func') || /\.func_v\d+$/.test(fullName)) {
            const symNameIdx = propReader.varint().toNumber();
            const symNameAttr = this._attrTypeReader.readAttribute(symNameIdx);
            if (symNameAttr && symNameAttr.value !== undefined) {
                const name = typeof symNameAttr.value === 'string' ? symNameAttr.value : String(symNameAttr.value);
                op.addAttribute('sym_name', new mlir.StringAttr(name));
            }
            if (propReader.position < data.length) {
                const funcTypeIdx = propReader.varint().toNumber();
                const funcTypeAttr = this._attrTypeReader.readAttribute(funcTypeIdx);
                if (funcTypeAttr instanceof mlir.TypeAttrOf && funcTypeAttr.type instanceof mlir.FunctionType) {
                    op.addAttribute('function_type', funcTypeAttr);
                }
            }
            return;
        }
        // Constant operations: single required 'value' attribute
        if (fullName.includes('.constant.') || fullName.includes('.const')) {
            const attrIdx = propReader.varint().toNumber();
            const attr = this._attrTypeReader.readAttribute(attrIdx);
            if (attr !== null && attr !== undefined) {
                op.addAttribute('value', attr);
            }
        }
        // For all other operations, skip native property parsing.
        // The encoding is operation-specific and we don't have the generated code.
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

    peek() {
        const position = this._reader.position;
        const value = this._reader.byte();
        this._reader.seek(position);
        return value;
    }

    uint64() {
        return this._reader.uint64();
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
        result >>= numBytes + 1n;
        return result;
    }

    svarint() {
        // Signed varint using zigzag encoding: (n >> 1) ^ -(n & 1)
        const n = this.varint();
        return (n >> 1n) ^ -(n & 1n);
    }

    string() {
        const reader = this._reader;
        let result = '';
        let value = -1;
        for (; ;) {
            value = reader.byte();
            if (value === 0x00) {
                break;
            }
            result += String.fromCharCode(value);
        }
        return result;
    }
};

mlir.BufferReader = class {

    constructor(data) {
        this._data = data;
        this._position = 0;
    }

    get length() {
        return this._data.length;
    }

    get position() {
        return this._position;
    }

    skip(length) {
        this._position += length;
    }

    seek(offset) {
        this._position = offset;
    }

    read(length) {
        const result = this._data.subarray(this._position, this._position + length);
        this._position += length;
        return result;
    }

    byte() {
        return this._data[this._position++];
    }

    peek() {
        return this._data[this._position];
    }

    uint64() {
        const view = new DataView(this._data.buffer, this._data.byteOffset + this._position, 8);
        this._position += 8;
        return view.getBigUint64(0, true);
    }

    varint() {
        let result = this.byte();
        if (result & 1) {
            return BigInt(result >> 1);
        }
        if (result === 0) {
            return this.uint64();
        }
        result = BigInt(result);
        let mask = 1n;
        let numBytes = 0n;
        let shift = 8n;
        while (result > 0n && (result & mask) === 0n) {
            result |= (BigInt(this.byte()) << shift);
            mask <<= 1n;
            shift += 8n;
            numBytes++;
        }
        result >>= numBytes + 1n;
        return result;
    }

    svarint() {
        // Signed varint using zigzag encoding: (n >> 1) ^ -(n & 1)
        const n = this.varint();
        return (n >> 1n) ^ -(n & 1n);
    }
};

mlir.Attribute = class {
};

mlir.TypedAttr = class extends mlir.Attribute {

    constructor(value, type) {
        super();
        this.value = value;
        this.type = type;
    }

    toString() {
        return this.value;
    }
};

mlir.StringAttr = class extends mlir.TypedAttr {

    constructor(value) {
        super(value, new mlir.PrimitiveType('string'));
    }

    toString() {
        return this.value;
    }
};

mlir.UnitAttr = class extends mlir.Attribute {

    toString() {
        return '';
    }
};

mlir.IntegerAttr = class extends mlir.Attribute {

    constructor(value, type) {
        super();
        this.value = value;
        this.type = type;
    }

    toString() {
        return this.value.toString();
    }
};

mlir.FloatAttr = class extends mlir.Attribute {

    constructor(value, type) {
        super();
        this.value = value;
        this.type = type;
    }

    toString() {
        return String(this.value);
    }
};

mlir.SymbolRefAttr = class extends mlir.Attribute {

    constructor(value) {
        super();
        this.value = value;
    }
};

mlir.DenseElementsAttr = class extends mlir.Attribute {

    constructor(value, type) {
        super();
        this.value = value;
        this.type = type;
    }
};

mlir.DenseResourceElementsAttr = class extends mlir.Attribute {

    constructor(handle, type) {
        super();
        this.handle = handle;
        this.type = type;
    }

    toString() {
        return `dense_resource<${this.handle}>`;
    }
};

mlir.ArrayAttr = class extends mlir.Attribute {

    constructor(elements) {
        super();
        this.elements = elements; // Array of Attribute objects
    }

    get value() {
        return this.elements.map((e) => e && e.value !== undefined ? e.value : e);
    }

    toString() {
        return `[${this.elements.map((e) => e && e.toString ? e.toString() : String(e)).join(', ')}]`;
    }
};

mlir.DictionaryAttr = class extends mlir.Attribute {

    constructor(value) {
        super();
        this._value = value; // Map of name -> Attribute
    }

    get value() {
        return this._value;
    }

    get(name) {
        return this._value.get(name);
    }

    toString() {
        const entries = Array.from(this._value.entries())
            .map(([k, v]) => `${k} = ${v && v.toString ? v.toString() : String(v)}`);
        return `{${entries.join(', ')}}`;
    }
};

mlir.DenseArrayAttr = class extends mlir.Attribute {

    constructor(elements, type) {
        super();
        this.elements = elements; // Array of values
        this.type = type; // Element type (e.g., i64, f32)
    }

    get value() {
        return this.elements;
    }

    toString() {
        const typeStr = this.type ? this.type.toString() : '';
        return `array<${typeStr}: ${this.elements.join(', ')}>`;
    }
};

mlir.TypeAttrOf = class extends mlir.Attribute {
    constructor(type) {
        super();
        this.type = type;  // the type IS the value
    }

    toString() {
        return this.type.toString();
    }
};

mlir.ConvDimensionNumbersAttr = class extends mlir.Attribute {
    constructor(input, kernel, output) {
        super();
        this.input = input;
        this.kernel = kernel;
        this.output = output;
    }

    toString() {
        const formatDim = (dims) => `[${dims.join(', ')}]`;
        return `${formatDim(this.input)}x${formatDim(this.kernel)}->${formatDim(this.output)}`;
    }
};

mlir.Type = class {

    constructor(value) {
        this._value = value;
    }

    get name() {
        return this._value;
    }

    toString() {
        return this._value;
    }
};

mlir.PrimitiveType = class extends mlir.Type {
};

mlir.FunctionType = class extends mlir.Type {

    constructor(inputs, results) {
        super(null);
        this.inputs = inputs || [];
        this.results = results || [];
    }

    toString() {
        const inputs = this.inputs.map((t) => t.toString());
        const results = this.results.map((t) => t.toString());
        const result = results.length === 1 ? results[0] : `(${results.join(', ')})`;
        return `(${inputs.join(', ')}) -> ${result}`;
    }
};

mlir.ComplexType = class extends mlir.Type {

    constructor(elementType) {
        super(null);
        this.elementType = elementType;
    }

    getElementType() {
        return this.elementType;
    }

    toString() {
        const elementTypeStr = this.elementType?.toString ? this.elementType.toString() : this.elementType;
        return `complex<${elementTypeStr}>`;
    }
};

mlir.RankedTensorType = class extends mlir.Type {

    constructor(shape, elementType, encoding) {
        super(null);
        this.shape = shape || [];
        this.elementType = elementType;
        this.encoding = encoding;
    }

    getElementType() {
        return this.elementType;
    }

    getShape() {
        return this.shape;
    }

    getNumElements() {
        if (this.shape.some((d) => d < 0 || d === '?')) {
            return 0; // Dynamic dimensions
        }
        return this.shape.length === 0 ? 1 : this.shape.reduce((a, b) => a * b, 1);
    }

    toString() {
        const shapeStr = this.shape.map((d) => d < 0 ? '?' : d).join('x');
        const elementTypeStr = this.elementType?.toString ? this.elementType.toString() : this.elementType;
        const prefix = shapeStr ? `${shapeStr}x` : '';
        if (this.encoding) {
            return `tensor<${prefix}${elementTypeStr}, ${this.encoding}>`;
        }
        return `tensor<${prefix}${elementTypeStr}>`;
    }
};

mlir.VectorType = class extends mlir.Type {

    constructor(shape, elementType, scalableDims) {
        super(null);
        this.shape = shape || [];
        this.elementType = elementType;
        this.scalableDims = scalableDims || [];
    }

    getElementType() {
        return this.elementType;
    }

    getShape() {
        return this.shape;
    }

    getNumElements() {
        if (this.shape.some((d) => d < 0 || d === '?')) {
            return 0; // Dynamic dimensions
        }
        return this.shape.length === 0 ? 1 : this.shape.reduce((a, b) => a * b, 1);
    }

    toString() {
        const parts = this.shape.map((d, i) => {
            const isScalable = this.scalableDims[i];
            return isScalable ? `[${d}]` : String(d);
        });
        const shapeStr = parts.join('x');
        const elementTypeStr = this.elementType?.toString ? this.elementType.toString() : this.elementType;
        const prefix = shapeStr ? `${shapeStr}x` : '';
        return `vector<${prefix}${elementTypeStr}>`;
    }
};

mlir.LLVMFunctionType = class extends mlir.Type {

    constructor(returnType, params, varArg = false) {
        super(null);
        this.returnType = returnType;
        this.params = params || [];
        this.varArg = varArg;
    }

    get inputs() {
        return this.params;
    }

    get results() {
        return this.returnType ? [this.returnType] : [];
    }

    toString() {
        const params = this.params.map((t) => t.toString());
        if (this.varArg) {
            params.push('...');
        }
        const returnType = this.returnType ? this.returnType.toString() : 'void';
        return `!llvm.func<${returnType} (${params.join(', ')})>`;
    }
};

mlir.Utility = class {

    static dataType(value) {
        if (value instanceof mlir.ComplexType) {
            const elementType = mlir.Utility.dataType(value.elementType);
            return `complex<${elementType}>`;
        }
        if (value instanceof mlir.Type) {
            value = value.toString();
        }
        switch (value) {
            case 'index': return 'int64';
            case 'f16': return 'float16';
            case 'f32': return 'float32';
            case 'f64': return 'float64';
            case 'f80': return 'float80';
            case 'f128': return 'float128';
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
            case 'i1': return 'int1';
            case 'i2': return 'int2';
            case 'i4': return 'int4';
            case 'i8': return 'int8';
            case 'i16': return 'int16';
            case 'i32': return 'int32';
            case 'i48': return 'int48';
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
            case 'b8': return 'int8';
            case 'unk': return 'unk'; // torch dialect unknown dtype
            case '!tf_type.string': return 'string';
            default:
                if (value && value.startsWith('!')) {
                    return value;
                }
                if (value && value.startsWith('vector<') && value.endsWith('>')) {
                    return value;
                }
                if (value && value.startsWith('memref<') && value.endsWith('>')) {
                    return value;
                }
                if (value && value.startsWith('tuple<') && value.endsWith('>')) {
                    return value;
                }
                if (value && value.startsWith('complex<') && value.endsWith('>')) {
                    const elementTypeStr = value.substring(8, value.length - 1);
                    const convertedElementType = mlir.Utility.dataType(elementTypeStr);
                    return `complex<${convertedElementType}>`;
                }
                if (value && /^[su]?i[0-9]+$/.test(value)) {
                    const match = value.match(/^(s|u)?i([0-9]+)$/);
                    if (match) {
                        const [, signed, widthStr] = match;
                        const width = parseInt(widthStr, 10);
                        if (signed === 'u') {
                            return `uint${width}`;
                        } else if (signed === 's') {
                            return `int${width}`;
                        }
                        return `int${width}`;
                    }
                }
                throw new mlir.Error(`Unknown data type '${value}'.`);
        }
    }

    // Parse shape dimensions from a type string like "tensor<2x3xf32>" -> [2, 3]
    // Returns null if dimensions cannot be extracted or contain dynamic dimensions
    static parseShapeDimensions(typeStr) {
        let spec = null;
        if (typeStr.startsWith('tensor<') && typeStr.endsWith('>')) {
            spec = typeStr.substring(7, typeStr.length - 1).trim();
        } else if (typeStr.startsWith('vector<') && typeStr.endsWith('>')) {
            spec = typeStr.substring(7, typeStr.length - 1).trim();
        }
        if (!spec) {
            return null;
        }
        const shape = [];
        let i = 0;
        while (i < spec.length) {
            if (spec[i] === '?' || spec[i] === '*') {
                // Dynamic dimension - cannot expand splats for dynamic shapes
                return null;
            }
            if (/[0-9]/.test(spec[i])) {
                let numStr = '';
                while (i < spec.length && /[0-9]/.test(spec[i])) {
                    numStr += spec[i];
                    i++;
                }
                const dim = parseInt(numStr, 10);
                if (!isNaN(dim)) {
                    shape.push(dim);
                }
            } else {
                break;
            }
            if (i < spec.length && spec[i] === 'x') {
                i++;
            } else {
                break;
            }
        }
        return shape;
    }

    static valueType(type) {
        if (type === undefined) {
            return null;
        }
        const typeStr = type instanceof mlir.Type ? type.toString() : type;
        if (typeStr.startsWith('!') && !typeStr.startsWith('!torch.vtensor<')) {
            return typeStr;
        }
        if (typeStr.startsWith('tensor<') && typeStr.endsWith('>')) {
            const spec = typeStr.substring(7, typeStr.length - 1).trim();
            if (spec.startsWith('!')) {
                return mlir.Utility.valueType(spec);
            }
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
                        shape.push('?');
                    } else {
                        shape.push(dim);
                    }
                } else {
                    break;
                }
                if (i < spec.length && spec[i] === 'x') {
                    i++;
                } else {
                    break;
                }
            }
            let dataType = spec.substring(i);
            const encodingIndex = dataType.indexOf(',');
            if (encodingIndex !== -1) {
                dataType = dataType.substring(0, encodingIndex).trim();
            }
            return new mlir.TensorType(dataType, new mlir.TensorShape(shape));
        }
        if (typeStr.startsWith('!torch.vtensor<') && typeStr.endsWith('>')) {
            const spec = typeStr.substring(15, typeStr.length - 1);
            let shape = null;
            let dataType = null;
            if (spec.startsWith('[')) {
                const bracketEnd = spec.indexOf(']');
                const shapeStr = spec.substring(0, bracketEnd + 1);
                const jsonStr = shapeStr.replace(/\?/g, '"?"');
                shape = JSON.parse(jsonStr);
                const rest = spec.substring(bracketEnd + 1);
                if (rest.startsWith(',')) {
                    const parts = rest.substring(1).split(',');
                    dataType = parts[0].trim();
                }
            } else if (spec.startsWith('*')) {
                if (spec.includes(',')) {
                    const parts = spec.split(',');
                    dataType = parts[1].trim();
                }
            } else {
                const parts = spec.split(',');
                dataType = parts[0].trim();
            }
            return new mlir.TensorType(dataType, shape ? new mlir.TensorShape(shape) : null);
        }
        if (typeStr.startsWith('tuple<') && typeStr.endsWith('>')) {
            return typeStr;
        }
        return typeStr;
    }
};

// Dialect Plugin System

mlir.AssemblyFormatParser = class {

    constructor(metadata) {
        this._metadata = metadata;
        this._buffer = metadata.assemblyFormat || '';
        this._pos = 0;
    }

    match(char) {
        return this._pos < this._buffer.length && this._buffer[this._pos] === char;
    }

    accept(str) {
        if (str.length === 1) {
            if (this.match(str)) {
                this._pos++;
                return true;
            }
            return false;
        }
        const remaining = this._buffer.substring(this._pos);
        if (remaining.startsWith(str)) {
            // Check that keyword is not followed by alphanumeric (to avoid "type" in "typename")
            const nextChar = this._buffer[this._pos + str.length];
            if (nextChar && /[a-zA-Z0-9_-]/.test(nextChar)) {
                return false;
            }
            this._pos += str.length;
            return true;
        }
        return false;
    }

    expect(char) {
        if (!this.match(char)) {
            throw new mlir.Error(`Expected '${char}'.`);
        }
        this._pos++;
    }

    parse() {
        const directives = [];
        this._skipWhitespace();
        while (this._pos < this._buffer.length) {
            const directive = this._parseDirective();
            directives.push(directive);
            this._skipWhitespace();
        }
        return directives;
    }

    _parseDirective() {
        const ch = this._buffer[this._pos];
        if (!ch || this._pos >= this._buffer.length) {
            throw new mlir.Error(`Unexpected end of format string.`);
        }
        // Parenthesized group: can be optional (...)?  or conditional (...):(...) or just grouping (...)
        if (this.match('(')) {
            this.accept('(');
            const elements = [];
            let anchorElement = null;

            this._skipWhitespace();
            while (!this.match(')')) {
                const elem = this._parseDirective();
                if (elem.type === 'anchor') {
                    // Standalone anchor - applies to the previous element
                    if (elements.length > 0) {
                        const prev = elements[elements.length - 1];
                        anchorElement = prev.name || prev.type;
                    }
                } else {
                    if (elem.anchor) {
                        anchorElement = elem.name || elem.type;
                    }
                    elements.push(elem);
                }
                this._skipWhitespace();
            }
            this.expect(')');
            this._skipWhitespace();
            // Check what follows to determine the group type
            if (this.accept('?')) {
                // Optional group: (...)?
                return { type: 'optional_group', elements, anchor: anchorElement };
            }
            if (this.accept(':')) {
                // Conditional alternative: (...):(...)?
                this._skipWhitespace();
                const secondAlt = [];
                let isSecondOptional = false;
                if (this.accept('(')) {
                    this._skipWhitespace();
                    while (!this.match(')')) {
                        const elem = this._parseDirective();
                        secondAlt.push(elem);
                        this._skipWhitespace();
                    }
                    this.expect(')');
                    this._skipWhitespace();
                    if (this.accept('?')) {
                        isSecondOptional = true;
                    }
                }
                return { type: 'conditional_alternative', firstAlt: elements, secondAlt, secondOptional: isSecondOptional };
            }
            return { type: 'group', elements };
        }
        // Literal: `keyword`
        if (this.accept('`')) {
            const value = this._parseUntil('`');
            this.expect('`');
            // MLIR reference: Empty literals (`` or ` `) are whitespace, not literals
            if (value.length === 0 || value === ' ' || value === '\\n') {
                return { type: 'whitespace', value }; // Return whitespace as a directive
            }
            return { type: 'literal', value };
        }
        if (this.accept('$')) {
            const name = this._parseIdentifier();
            const anchor = this.accept('^');
            const metadata = this._metadata;
            // Determine variable type from metadata first - matches reference implementation
            // Check each metadata category in priority order
            if (metadata.successors && metadata.successors.some((a) => a.name === name)) {
                return { type: 'successor_ref', name, anchor };
            }
            if (metadata.attributes && metadata.attributes.some((a) => a.name === name)) {
                return { type: 'attribute_ref', name, anchor };
            }
            if (metadata.operands && metadata.operands.some((a) => a.name === name)) {
                return { type: 'operand_ref', name, anchor };
            }
            if (metadata.regions && metadata.regions.some((a) => a.name === name)) {
                return { type: 'region_ref', name, anchor };
            }

            // Special keywords for collections (only used if not found in metadata)
            if (name === 'operands') {
                return { type: 'operands', anchor };
            }
            if (name === 'results') {
                return { type: 'results', anchor };
            }
            if (name === 'regions') {
                return { type: 'regions', anchor };
            }
            if (name === 'successors') {
                return { type: 'successors', anchor };
            }

            // Variable not found in metadata - this indicates incomplete metadata or parsing of non-existent variable
            // Assume it's an operand_ref for backward compatibility, but this should be investigated
            return { type: 'operand_ref', name, anchor };
        }
        if (this.accept('type')) {
            const args = this._parseParenList();
            const anchor = this.accept('^');
            return { type: 'type', args, anchor };
        }
        if (this.accept('qualified')) {
            const args = this._parseParenList();
            const anchor = this.accept('^');
            return { type: 'qualified', args, anchor };
        }
        if (this.accept('attr-dict-with-keyword')) {
            return { type: 'attr_dict_with_keyword' };
        }
        if (this.accept('attr-dict')) {
            return { type: 'attr_dict' };
        }
        if (this.accept('prop-dict')) {
            return { type: 'prop_dict' };
        }
        if (this.accept('functional-type')) {
            const args = this._parseParenList();
            const anchor = this.accept('^');
            return { type: 'functional_type', args, anchor };
        }
        if (this.accept('params')) {
            return { type: 'params' };
        }
        if (this.accept('struct')) {
            this.expect('(');
            const args = [];
            while (!this.match(')')) {
                this._skipWhitespace();
                if (this.match(')')) {
                    break;
                }
                const arg = this._parseDirective();
                args.push(arg);
                this._skipWhitespace();
                this.accept(',');
            }
            this.expect(')');
            return { type: 'struct', args };
        }
        if (this.accept('ref')) {
            this.expect('(');
            const arg = this._parseDirective();
            this._skipWhitespace();
            this.expect(')');
            return { type: 'ref', arg };
        }
        if (this.accept('custom')) {
            this.expect('<');
            const parser = this._parseUntil('>');
            this.expect('>');
            const args = this._parseParenList();
            const anchor = this.accept('^');
            return { type: 'custom', parser, args, anchor };
        }
        if (this.accept('oilist')) {
            this._skipWhitespace();
            this.expect('(');
            let content = '';
            let depth = 1;
            while (this._pos < this._buffer.length && depth > 0) {
                const ch = this._buffer[this._pos];
                if (ch === '(') {
                    depth++;
                    content += ch;
                    this._pos++;
                } else if (ch === ')') {
                    depth--;
                    if (depth > 0) {
                        content += ch;
                    }
                    this._pos++;
                } else {
                    content += ch;
                    this._pos++;
                }
            }
            return { type: 'oilist', content };
        }
        if (this.accept('operands')) {
            return { type: 'operands' };
        }
        if (this.accept('results')) {
            return { type: 'results' };
        }
        if (this.accept('regions')) {
            return { type: 'regions' };
        }
        if (this.accept('successors')) {
            return { type: 'successors' };
        }
        if (ch === '^') {
            this._pos++;
            return { type: 'anchor' };
        }
        if (/^[:()[\]{}<>,=|]/.test(ch)) {
            this._pos++;
            return { type: 'literal', value: ch };
        }
        const context = this._buffer.substring(Math.max(0, this._pos - 10), Math.min(this._buffer.length, this._pos + 10));
        throw new mlir.Error(`Unexpected '${ch}' in assembly format '${context}...'.`);
    }

    _parseIdentifier() {
        let name = '';
        while (this._pos < this._buffer.length) {
            const ch = this._buffer[this._pos];
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
        while (this._pos < this._buffer.length && this._buffer[this._pos] !== terminator) {
            value += this._buffer[this._pos];
            this._pos++;
        }
        return value;
    }

    _parseParenList() {
        if (!this.accept('(')) {
            return [];
        }
        this._skipWhitespace();
        if (this.accept(')')) {
            return [];
        }
        const items = [];
        const parseElement = () => {
            let element = '';
            let depth = 0;
            while (this._pos < this._buffer.length) {
                this._skipWhitespace();
                if (this.accept('"')) {
                    // String literal - consume as a unit
                    element += '"';
                    element += this._parseUntil('"');
                    element += '"';
                    this.expect('"');
                } else if (this.accept('$')) {
                    element += '$';
                    const id = this._parseIdentifier();
                    element += id;
                } else if (this.accept('(')) {
                    // Nested parentheses - include in element (e.g., type($list))
                    element += '(';
                    depth++;
                } else if (this.match(')')) {
                    if (depth === 0) {
                        // End of this element
                        break;
                    }
                    element += ')';
                    this.accept(')');
                    depth--;
                } else if (this.match(',') && depth === 0) {
                    // Comma at top level - end of this element
                    break;
                } else if (this.match('-')) {
                    // Handle hyphenated identifiers like attr-dict, functional-type
                    element += '-';
                    this.accept('-');
                } else {
                    // Plain identifier (e.g., "type" in type($list))
                    const id = this._parseIdentifier();
                    if (!id) {
                        throw new mlir.Error(`Unexpected '${this._buffer[this._pos]}' in assembly format directive list.`);
                    }
                    element += id;
                }
            }
            return element.trim();
        };
        const first = parseElement();
        if (!first) {
            throw new mlir.Error('Expected element.');
        }
        items.push(first);
        this._skipWhitespace();
        while (this.accept(',')) {
            this._skipWhitespace();
            const elem = parseElement();
            if (!elem) {
                throw new mlir.Error('Expected element after comma');
            }
            items.push(elem);
            this._skipWhitespace();
        }
        this.expect(')');
        return items;
    }

    _skipWhitespace() {
        while (this._pos < this._buffer.length && /\s/.test(this._buffer[this._pos])) {
            this._pos++;
        }
    }
};

mlir.DialectContext = class {

    constructor(metadata) {
        const operations = metadata.operations;
        this._dialects = new Map();
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
        this._dialects.set('emitc', new mlir.EmitCDialect(operations));
        this._dialects.set('complex', new mlir.Dialect(operations, 'complex'));
        this._dialects.set('index', new mlir.Dialect(operations, 'index'));
        this._dialects.set('pdl', new mlir.PDLDialect(operations));
        this._dialects.set('ptr', new mlir.PtrDialect(operations));
        this._dialects.set('ub', new mlir.Dialect(operations, 'ub'));
        this._dialects.set('amdgpu', new mlir.AMDGPUDialect(operations));
        this._dialects.set('nvgpu', new mlir.NVGPUDialect(operations));
        this._dialects.set('nvvm', new mlir.NVVMDialect(operations));
        this._dialects.set('rocdl', new mlir.ROCDLDialect(operations));
        this._dialects.set('nvws', new mlir.NVWSDialect(operations));
        this._dialects.set('tti', new mlir.Dialect(operations, 'tti'));
        this._dialects.set('omp', new mlir.OpenMPDialect(operations));
        this._dialects.set('proton', new mlir.ProtonDialect(operations));
        this._dialects.set('proton_gpu', new mlir.Dialect(operations, 'proton_gpu'));
        this._dialects.set('arm_sme', new mlir.ArmSMEDialect(operations));
        this._dialects.set('arm_neon', new mlir.ArmNeonDialect(operations));
        this._dialects.set('arm_sve', new mlir.ArmSVEDialect(operations));
        this._dialects.set('shard', new mlir.ShardDialect(operations));
        this._dialects.set('amx', new mlir.Dialect(operations, 'amx'));
        this._dialects.set('smt', new mlir.SMTDialect(operations));
        this._dialects.set('lagrad', new mlir.Dialect(operations, 'lagrad'));
        this._dialects.set('iree_codegen', new mlir.IREECodegenDialect(operations));
        this._dialects.set('iree_encoding', new mlir.Dialect(operations, 'iree_encoding'));
        this._dialects.set('test', new mlir.TestDialect(operations));
        this._dialects.set('scf', new mlir.SCFDialect(operations));
        this._dialects.set('shape', new mlir.ShapeDialect(operations));
        this._dialects.set('sparse_tensor', new mlir.SparseTensorDialect(operations));
        this._dialects.set('func', new mlir.FuncDialect(operations));
        this._dialects.set('gpu', new mlir.GpuDialect(operations));
        this._dialects.set('llvm', new mlir.LLVMDialect(operations));
        this._dialects.set('xegpu', new mlir.XeGPUDialect(operations));
        this._dialects.set('memref', new mlir.MemRefDialect(operations));
        this._dialects.set('vector', new mlir.VectorDialect(operations));
        this._dialects.set('x86vector', new mlir.Dialect(operations, 'x86vector'));
        this._dialects.set('onnx', new mlir.ONNXDialect(operations));
        this._dialects.set('krnl', new mlir.KrnlDialect(operations));
        this._dialects.set('torch', new mlir.TorchDialect(operations));
        this._dialects.set('torch_c', new mlir.Dialect(operations, 'torch_c'));
        this._dialects.set('hal', new mlir.HALDialect(operations));
        this._dialects.set('hal_loader', new mlir.HALLoaderDialect(operations));
        this._dialects.set('hal_inline', new mlir.Dialect(operations, 'hal_inline'));
        this._dialects.set('util', new mlir.UtilDialect(operations));
        this._dialects.set('mhlo', new mlir.MhloDialect(operations));
        this._dialects.set('chlo', new mlir.Dialect(operations, 'chlo'));
        this._dialects.set('thlo', new mlir.THLODialect(operations));
        this._dialects.set('flow', new mlir.FlowDialect(operations));
        this._dialects.set('stream', new mlir.StreamDialect(operations));
        this._dialects.set('iree_vector_ext', new mlir.IREEVectorExtDialect(operations));
        this._dialects.set('iree_tensor_ext', new mlir.IREETensorExtDialect(operations));
        this._dialects.set('linalg', new mlir.LinalgDialect(operations));
        this._dialects.set('iree_linalg_ext', new mlir.Dialect(operations, 'iree_linalg_ext'));
        this._dialects.set('linalg_ext', this._dialects.get('iree_linalg_ext'));
        this._dialects.set('quant', new mlir.QuantDialect(operations));
        this._dialects.set('tensor', new mlir.TensorDialect(operations));
        this._dialects.set('tosa', new mlir.TosaDialect(operations));
        this._dialects.set('tf', new mlir.TFDialect(operations));
        this._dialects.set('tf_saved_model', new mlir.Dialect(operations, 'tf_saved_model'));
        this._dialects.set('tf_type', new mlir.TFTypeDialect(operations));
        this._dialects.set('tf_device', new mlir.TFDeviceDialect(operations));
        this._dialects.set('tf_executor', new mlir.TFExecutorDialect(operations));
        this._dialects.set('tf_framework', new mlir.TFFrameworkDialect(operations));
        this._dialects.set('tfr', new mlir.TFRDialect(operations));
        this._dialects.set('corert', new mlir.CoreRTDialect(operations));
        this._dialects.set('tfrt', new mlir.TFRTDialect(operations));
        this._dialects.set('tfrt_fallback', new mlir.Dialect(operations, 'tfrt_fallback'));
        this._dialects.set('tfrt_fallback_async', new mlir.TFRTFallbackAsyncDialect(operations));
        this._dialects.set('tfl', new mlir.TFLDialect(operations));
        this._dialects.set('stdx', new mlir.StdxDialect(operations));
        this._dialects.set('vm', new mlir.VMDialect(operations));
        this._dialects.set('math', new mlir.MathDialect(operations));
        this._dialects.set('tm_tensor', new mlir.TMTensorDialect(operations));
        this._dialects.set('ml_program', new mlir.MLProgramDialect(operations));
        this._dialects.set('iree_gpu', new mlir.IREEGPUDialect(operations));
        this._dialects.set('tile', new mlir.TileDialect(operations));
        this._dialects.set('pxa', new mlir.PXADialect(operations));
        this._dialects.set('irdl', new mlir.IRDLDialect(operations));
        this._dialects.set('transform', new mlir.TransformDialect(operations));
        this._dialects.set('wasmssa', new mlir.WasmSSADialect(operations));
        this._dialects.set('spirv', new mlir.SPIRVDialect(operations));
        this._dialects.set('spv', this._dialects.get('spirv'));
        this._dialects.set('toy', new mlir.ToyDialect(operations));
        this._dialects.set('top', new mlir.Dialect(operations, 'top'));
        this._dialects.set('tpu', new mlir.Dialect(operations, 'tpu'));
        this._dialects.set('sdfg', new mlir.SdfgDialect(operations));
        this._dialects.set('sdir', this._dialects.get('sdfg'));
        this._dialects.set('check', new mlir.CheckDialect(operations));
        this._dialects.set('tt', new mlir.TritonDialect(operations));
        this._dialects.set('ttg', new mlir.TritonGPUDialect(operations));
        this._dialects.set('triton_gpu', this._dialects.get('ttg'));
        this._dialects.set('gluon', new mlir.GluonDialect(operations));
        this._dialects.set('ttng', new mlir.TritonNvidiaGPUDialect(operations));
        this._dialects.set('nvidia_gpu', this._dialects.get('ttng'));
        this._dialects.set('amdg', new mlir.TritonAMDGPUDialect(operations));
        this._dialects.set('amd_gpu', this._dialects.get('amdg'));
        this._dialects.set('michelson', new mlir.MichelsonDialect(operations));
        this._dialects.set('tensorrt', new mlir.TensorRTDialect(operations));
        this._dialects.set('executor', new mlir.ExecutorDialect(operations));
        this._dialects.set('exec', this._dialects.get('executor'));
        this._dialects.set('tfrt_test', new mlir.TFRTTestDialect(operations));
        this._dialects.set('xevm', new mlir.XeVMDialect(operations));
        this._dialects.set('vmvx', new mlir.VMVXDialect(operations));
        this._dialects.set('mlrt', new mlir.MLRTDialect(operations));
        this._dialects.set('tfrt_tensor', new mlir.TFRTTensorDialect(operations));
        this._dialects.set('tfrt_dht', new mlir.TFRTDHTDialect(operations));
        this._dialects.set('coo', new mlir.Dialect(operations, 'coo'));
        this._dialects.set('tfd', new mlir.TFDDialect(operations));
        this._dialects.set('acc', new mlir.ACCDialect(operations));
        this._dialects.set('cuda', new mlir.Dialect(operations, 'cuda'));
        this._dialects.set('trtrt', new mlir.Dialect(operations, 'trtrt'));
        this._dialects.set('plan', new mlir.PlanDialect(operations));
        this._dialects.set('kernel', new mlir.KernelDialect(operations));
        this._dialects.set('nvg', new mlir.Dialect(operations, 'nvg'));
        this._dialects.set('mpi', new mlir.Dialect(operations, 'mpi'));
        this._dialects.set('pdl_interp', new mlir.PDLInterpDialect(operations));
        this._dialects.set('standalone', new mlir.Dialect(operations, 'standalone'));
        this._dialects.set('custom', new mlir.Dialect(operations, 'custom'));
        this._dialects.set('layer', new mlir.Dialect(operations, 'layer'));
        this._dialects.set('foo', new mlir.Dialect(operations, 'foo'));
        this._dialects.set('some', new mlir.Dialect(operations, 'some'));
        this._dialects.set('ts', new mlir.Dialect(operations, 'ts'));
        this._dialects.set('tf_mlrt', new mlir.Dialect(operations, 'tf_mlrt'));
        this._dialects.set('io_parameters', new mlir.IOParametersDialect(operations));
        this._dialects.set('pcf', new mlir.PCFDialect(operations));
        this._dialects.set('linalgx', new mlir.Dialect(operations, 'linalgx'));
        this._dialects.set('xsmm', new mlir.XSMMDialect(operations));
        this._dialects.set('sdy', new mlir.SdyDialect(operations));
        this._dialects.set('mpmd', new mlir.MPMDDialect(operations));
        this._dialects.set('tfg', new mlir.TFGDialect(operations));
        this._dialects.set('vt', new mlir.Dialect(operations, 'vt'));
        this._dialects.set('testd', new mlir.Dialect(operations, 'testd'));
        this._dialects.set('cmath', new mlir.Dialect(operations, 'cmath'));
        this._dialects.set('bytecode', new mlir.Dialect(operations, 'bytecode'));
        this._dialects.set('test_irdl_to_cpp', new mlir.Dialect(operations, 'test_irdl_to_cpp'));
        this._dialects.set('iree_unregistered', new mlir.Dialect(operations, 'iree_unregistered'));
        this._dialects.set('cir', new mlir.Dialect(operations, 'cir'));
        this._dialects.set('migraphx', new mlir.Dialect(operations, 'migraphx'));
        this._dialects.set('xla', new mlir.XlaDialect(operations));
        this._dialects.set('xla_gpu', new mlir.XlaGpuDialect(operations));
        this._dialects.set('xla_cpu', new mlir.Dialect(operations, 'xla_cpu'));
        this._dialects.set('xla_framework', new mlir.Dialect(operations, 'xla_framework'));
        this._dialects.set('ifrt', new mlir.Dialect(operations, 'ifrt'));
        this._dialects.set('vifrt', new mlir.Dialect(operations, 'vifrt'));
        this._dialects.set('triton_xla', new mlir.TritonXlaDialect(operations));
        this._dialects.set('xtile', new mlir.XTileDialect(operations));
        this._redirect = new Map([
            ['builtin.func', 'func.func'],
            ['builtin.constant', 'arith.constant'],
            ['builtin.return', 'func.return'],
            ['builtin.select', 'arith.select'],
            ['scf.select', 'arith.select'],
            ['scf.call', 'func.call'],
            ['builtin.view', 'memref.view'],
            ['builtin.dealloc', 'memref.dealloc'], ['func.dealloc', 'memref.dealloc'],
            // Arith operations (from both builtin and func default dialects)
            ['builtin.addi', 'arith.addi'], ['func.addi', 'arith.addi'],
            ['builtin.subi', 'arith.subi'], ['func.subi', 'arith.subi'],
            ['builtin.muli', 'arith.muli'], ['func.muli', 'arith.muli'],
            ['builtin.divi_signed', 'arith.divsi'], ['func.divi_signed', 'arith.divsi'],
            ['builtin.divi_unsigned', 'arith.divui'], ['func.divi_unsigned', 'arith.divui'],
            ['builtin.divsi', 'arith.divsi'], ['func.divsi', 'arith.divsi'],
            ['builtin.divui', 'arith.divui'], ['func.divui', 'arith.divui'],
            ['builtin.remi_signed', 'arith.remsi'], ['func.remi_signed', 'arith.remsi'],
            ['builtin.remi_unsigned', 'arith.remui'], ['func.remi_unsigned', 'arith.remui'],
            ['builtin.andi', 'arith.andi'], ['func.andi', 'arith.andi'],
            ['builtin.ori', 'arith.ori'], ['func.ori', 'arith.ori'],
            ['builtin.xori', 'arith.xori'], ['func.xori', 'arith.xori'],
            ['builtin.shli', 'arith.shli'], ['func.shli', 'arith.shli'],
            ['builtin.shrsi', 'arith.shrsi'], ['func.shrsi', 'arith.shrsi'],
            ['builtin.shrui', 'arith.shrui'], ['func.shrui', 'arith.shrui'],
            ['builtin.addf', 'arith.addf'], ['func.addf', 'arith.addf'],
            ['builtin.subf', 'arith.subf'], ['func.subf', 'arith.subf'],
            ['builtin.mulf', 'arith.mulf'], ['func.mulf', 'arith.mulf'],
            ['builtin.divf', 'arith.divf'], ['func.divf', 'arith.divf'],
            ['builtin.cmpi', 'arith.cmpi'], ['func.cmpi', 'arith.cmpi'],
            ['builtin.cmpf', 'arith.cmpf'], ['func.cmpf', 'arith.cmpf'],
            ['builtin.index_cast', 'arith.index_cast'], ['func.index_cast', 'arith.index_cast'],
            ['builtin.sitofp', 'arith.sitofp'], ['func.sitofp', 'arith.sitofp'],
            ['builtin.fptosi', 'arith.fptosi'], ['func.fptosi', 'arith.fptosi'],
            ['builtin.truncf', 'arith.truncf'], ['func.truncf', 'arith.truncf'],
            ['builtin.extf', 'arith.extf'], ['func.extf', 'arith.extf'],
            ['builtin.splat', 'vector.splat'],
            ['func.splat', 'vector.splat'],
            ['scf.splat', 'vector.splat'],
            // Memref operations
            ['builtin.alloc', 'memref.alloc'], ['func.alloc', 'memref.alloc'],
            ['builtin.load', 'memref.load'], ['func.load', 'memref.load'],
            ['builtin.store', 'memref.store'], ['func.store', 'memref.store'],
            ['builtin.subview', 'memref.subview'], ['func.subview', 'memref.subview'],
            ['builtin.dim', 'memref.dim'], ['func.dim', 'memref.dim'],
            ['builtin.view', 'memref.view'], ['func.view', 'memref.view'],
            // Control flow operations
            ['builtin.cond_br', 'cf.cond_br'], ['func.cond_br', 'cf.cond_br'],
            ['builtin.br', 'cf.br'], ['func.br', 'cf.br'],
            ['builtin.switch', 'cf.switch'], ['func.switch', 'cf.switch'],
            ['builtin.assert', 'cf.assert'], ['func.assert', 'cf.assert'],
            // Other redirects
            ['flow.constant', 'flow.tensor.constant'],
            ['util.initializer.return', 'util.return']
        ]);
    }

    getDialect(name) {
        return this._dialects.get(name);
    }

    resolveOpName(name) {
        return this._redirect.has(name) ? this._redirect.get(name) : name;
    }
};

mlir.Dialect = class {

    constructor(operations, name) {
        this._name = name;
        this._operations = new Map();
        this._customDirectives = new Map();
        this._customTypes = new Map();
        this._customAttributes = new Map();
        this.registerCustomDirective('DynamicIndexList', this._parseDynamicIndexList.bind(this));
        this.registerCustomDirective('Offsets', this._parseOffsets.bind(this));
        this.registerCustomDirective('SymbolVisibility', this._parseSymbolVisibility.bind(this));
        this.registerCustomDirective('TypeOrAttr', this._parseTypeOrAttr.bind(this));
        this.registerCustomDirective('CopyOpRegion', this._parseCopyOpRegion.bind(this));
        this.registerCustomDirective('SizeAwareType', this._parseSizeAwareType.bind(this));
        this.registerCustomAttribute('TypedAttrInterface', this._parseTypedAttrInterface.bind(this));
        this.registerCustomAttribute('VM_ConstantIntegerValueAttr', this._parseTypedAttrInterface.bind(this));
        this.registerCustomAttribute('Util_AnySerializableAttr', this._parseTypedAttrInterface.bind(this));
        this.registerCustomAttribute('ElementsAttr', this._parseTypedAttrInterface.bind(this));
        this.registerCustomAttribute('AnyAttr', this._parseTypedAttrInterface.bind(this));
        this.registerCustomAttribute('UnitAttr', this._parseUnitAttr.bind(this));
        this.registerCustomAttribute('UnitProp', this._parseUnitAttr.bind(this));
        this.registerCustomAttribute('SymbolNameAttr', this._parseSymbolNameAttr.bind(this));
        this.registerCustomAttribute('SymbolRefAttr', this._parseSymbolRefAttr.bind(this));
        this.registerCustomAttribute('FlatSymbolRefAttr', this._parseFlatSymbolRefAttr.bind(this));
        this.registerCustomAttribute('OptionalAttr', this._parseOptionalAttr.bind(this));
        this.registerCustomAttribute('OptionalProp', this._parseOptionalAttr.bind(this));
        this.registerCustomAttribute('DefaultValuedOptionalAttr', this._parseDefaultValuedOptionalAttr.bind(this));
        this.registerCustomAttribute('DefaultValuedAttr', this._parseDefaultValuedAttr.bind(this));
        this.registerCustomAttribute('DefaultValuedEnumAttr', this._parseDefaultValuedAttr.bind(this));
        this.registerCustomAttribute('DefaultValuedProp', this._parseDefaultValuedAttr.bind(this));
        this.registerCustomAttribute('ConfinedAttr', this._parseConfinedAttr.bind(this));
        this.registerCustomAttribute('TypeAttrOf', this._parseTypeAttrOf.bind(this));
        this.registerCustomAttribute('AnyAttrOf', this._parseAnyAttrOf.bind(this));
        this.registerCustomAttribute('ArrayAttr', this._parseArrayAttr.bind(this));
        this.registerCustomAttribute('TypedArrayAttrBase', this._parseArrayAttr.bind(this));
        this.registerCustomAttribute('I64Attr', this._parseIntegerAttr.bind(this, 'i64'));
        this.registerCustomAttribute('I32Attr', this._parseIntegerAttr.bind(this, 'i32'));
        this.registerCustomAttribute('I16Attr', this._parseIntegerAttr.bind(this, 'i16'));
        this.registerCustomAttribute('I8Attr', this._parseIntegerAttr.bind(this, 'i8'));
        this.registerCustomAttribute('I1Attr', this._parseIntegerAttr.bind(this, 'i1'));
        this.registerCustomAttribute('SI64Attr', this._parseIntegerAttr.bind(this, 'si64'));
        this.registerCustomAttribute('SI32Attr', this._parseIntegerAttr.bind(this, 'si32'));
        this.registerCustomAttribute('UI64Attr', this._parseIntegerAttr.bind(this, 'ui64'));
        this.registerCustomAttribute('UI32Attr', this._parseIntegerAttr.bind(this, 'ui32'));
        this.registerCustomAttribute('IndexAttr', this._parseIntegerAttr.bind(this, 'index'));
        this.registerCustomAttribute('F64Attr', this._parseFloatAttr.bind(this, 'f64'));
        this.registerCustomAttribute('F32Attr', this._parseFloatAttr.bind(this, 'f32'));
        this.registerCustomAttribute('F16Attr', this._parseFloatAttr.bind(this, 'f16'));
        this.registerCustomAttribute('BF16Attr', this._parseFloatAttr.bind(this, 'bf16'));
        this.registerCustomAttribute('StrAttr', this._parseStrAttr.bind(this));
        this.registerCustomAttribute('TypedStrAttr', this._parseTypedAttrInterface.bind(this));
        this.registerCustomAttribute('LevelAttr', this._parseIntegerAttr.bind(this, 'index'));
        this.registerCustomType('Optional', this._parseOptional.bind(this));
        for (const metadata of operations.get(name) || []) {
            const opInfo = { metadata };
            if (metadata.assemblyFormat) {
                const parser = new mlir.AssemblyFormatParser(metadata);
                opInfo.directives = parser.parse();
            }
            this._operations.set(metadata.name, opInfo);
        }
    }

    get name() {
        return this._name;
    }

    _parseConstraint(value) {
        if (!value || typeof value !== 'string') {
            return null;
        }
        value = value.trim();
        if (!value) {
            return null;
        }

        // Tokenize
        const tokenize = (str) => {
            const tokens = [];
            let i = 0;
            while (i < str.length) {
                const ch = str[i];
                if (/\s/.test(ch)) {
                    i++;
                    continue;
                }
                if ('<>={}[](),|'.indexOf(ch) !== -1) {
                    tokens.push({ type: ch, value: ch, pos: i });
                    i++;
                    continue;
                }
                if (ch === ':' && i + 1 < str.length && str[i + 1] === ':') {
                    tokens.push({ type: '::', value: '::', pos: i });
                    i += 2;
                    continue;
                }
                if (ch === ':') {
                    i++;
                    continue;
                }
                if (ch === '"' || ch === "'") {
                    const quote = ch;
                    let j = i + 1;
                    while (j < str.length && str[j] !== quote) {
                        if (str[j] === '\\' && j + 1 < str.length) {
                            j += 2;
                        } else {
                            j++;
                        }
                    }
                    if (j < str.length) {
                        tokens.push({ type: 'string', value: str.substring(i + 1, j), pos: i });
                        i = j + 1;
                    } else {
                        tokens.push({ type: 'ident', value: str.substring(i), pos: i });
                        break;
                    }
                    continue;
                }
                if (/[a-zA-Z_0-9-]/.test(ch)) {
                    let j = i;
                    while (j < str.length && /[a-zA-Z_0-9-]/.test(str[j])) {
                        j++;
                    }
                    const ident = str.substring(i, j);
                    tokens.push({ type: 'ident', value: ident, pos: i });
                    i = j;
                    continue;
                }
                i++;
            }
            return tokens;
        };

        // Parse tokens into constraint structure
        const parseTokens = (tokens, pos) => {
            if (pos >= tokens.length) {
                return null;
            }
            const token = tokens[pos];
            if (token.type === '::') {
                // eslint-disable-next-line no-use-before-define
                return parseScopedIdentifier(tokens, pos);
            }
            if (token.type !== 'ident') {
                return null;
            }
            let name = token.value;
            let nextPos = pos + 1;
            while (nextPos < tokens.length && tokens[nextPos].type === '::') {
                nextPos++;
                if (nextPos < tokens.length && tokens[nextPos].type === 'ident') {
                    const value = tokens[nextPos++].value;
                    name += `::${value}`;
                } else {
                    break;
                }
            }
            if (nextPos >= tokens.length) {
                return { value: { name }, nextPos };
            }
            const nextToken = tokens[nextPos];
            if (nextToken.type === '{') {
                // eslint-disable-next-line no-use-before-define
                return parseEnum(tokens, pos, name);
            }
            if (nextToken.type === '<') {
                // eslint-disable-next-line no-use-before-define
                return parseGeneric(tokens, pos, name);
            }
            return { value: { name }, nextPos };
        };
        const parseScopedIdentifier = (tokens, pos) => {
            let name = '';
            let nextPos = pos;
            while (nextPos < tokens.length) {
                if (tokens[nextPos].type === '::') {
                    name += '::';
                    nextPos++;
                } else if (tokens[nextPos].type === 'ident') {
                    name += tokens[nextPos].value;
                    nextPos++;
                } else {
                    break;
                }
            }
            if (!name) {
                return null;
            }
            if (nextPos < tokens.length) {
                const nextToken = tokens[nextPos];
                if (nextToken.type === '{') {
                    // eslint-disable-next-line no-use-before-define
                    return parseEnum(tokens, pos, name);
                }
                if (nextToken.type === '<') {
                    // eslint-disable-next-line no-use-before-define
                    return parseGeneric(tokens, pos, name);
                }
            }
            return { value: { name }, nextPos };
        };

        const parseEnum = (tokens, startPos, name) => {
            let pos = startPos;
            while (pos < tokens.length && (tokens[pos].type === 'ident' || tokens[pos].type === '::')) {
                pos++;
            }
            if (pos >= tokens.length || tokens[pos].type !== '{') {
                return null;
            }
            pos++;
            const values = [];
            let currentValue = '';
            while (pos < tokens.length && tokens[pos].type !== '}') {
                const token = tokens[pos];
                if (token.type === '|') {
                    if (currentValue.trim()) {
                        values.push(currentValue.trim());
                        currentValue = '';
                    }
                    pos++;
                } else if (token.type === 'ident') {
                    if (currentValue) {
                        currentValue += ' ';
                    }
                    currentValue += token.value;
                    pos++;
                } else if (token.type === '::') {
                    currentValue += '::';
                    pos++;
                } else {
                    pos++;
                }
            }
            if (currentValue.trim()) {
                values.push(currentValue.trim());
            }
            if (pos < tokens.length && tokens[pos].type === '}') {
                pos++;
            }
            return { value: { name, values }, nextPos: pos };
        };

        const parseGeneric = (tokens, startPos, name) => {
            let pos = startPos;
            while (pos < tokens.length && (tokens[pos].type === 'ident' || tokens[pos].type === '::')) {
                pos++;
            }
            if (pos >= tokens.length || tokens[pos].type !== '<') {
                return null;
            }
            pos++;
            const args = [];
            let angleDepth = 1;
            let bracketDepth = 0;
            let currentArg = [];
            while (pos < tokens.length && (angleDepth > 0 || bracketDepth > 0)) {
                const token = tokens[pos];
                if (token.type === '<') {
                    angleDepth++;
                    currentArg.push(token);
                    pos++;
                } else if (token.type === '>') {
                    angleDepth--;
                    if (angleDepth === 0 && bracketDepth === 0) {
                        if (currentArg.length > 0) {
                            // eslint-disable-next-line no-use-before-define
                            const parsed = parseArgumentTokens(currentArg);
                            if (parsed !== null) {
                                args.push(parsed);
                            }
                        }
                        pos++;
                        break;
                    } else {
                        currentArg.push(token);
                        pos++;
                    }
                } else if (token.type === '[') {
                    bracketDepth++;
                    currentArg.push(token);
                    pos++;
                } else if (token.type === ']') {
                    bracketDepth--;
                    currentArg.push(token);
                    pos++;
                } else if (token.type === ',' && angleDepth === 1 && bracketDepth === 0) {
                    if (currentArg.length > 0) {
                        // eslint-disable-next-line no-use-before-define
                        const parsed = parseArgumentTokens(currentArg);
                        if (parsed !== null) {
                            args.push(parsed);
                        }
                        currentArg = [];
                    }
                    pos++;
                } else {
                    currentArg.push(token);
                    pos++;
                }
            }
            return { value: { name, args }, nextPos: pos };
        };
        const parseArgumentTokens = (tokens) => {
            if (!tokens || tokens.length === 0) {
                return null;
            }
            tokens = tokens.filter((t) => t.type !== undefined);
            if (tokens[0].type === '[') {
                // eslint-disable-next-line no-use-before-define
                return parseListArgument(tokens);
            }
            if (tokens[0].type === 'string') {
                return tokens[0].value;
            }
            if (tokens[0].type === 'ident' || tokens[0].type === '::') {
                const result = parseTokens(tokens, 0);
                if (result && result.nextPos === tokens.length) {
                    return result.value;
                }
            }
            let literal = '';
            for (const token of tokens) {
                if (token.type === 'ident' || token.type === 'string') {
                    if (literal && !/^[,[]\(\):\.]$/.test(literal[literal.length - 1])) {
                        literal += ' ';
                    }
                    literal += token.value;
                } else if (token.type === '::') {
                    literal += '::';
                } else if ('{}[](),.'.indexOf(token.type) !== -1) {
                    literal += token.value;
                }
            }
            return literal.trim() || null;
        };

        const parseListArgument = (tokens) => {
            if (!tokens || tokens.length === 0 || tokens[0].type !== '[') {
                return null;
            }
            let pos = 1;
            const items = [];
            let bracketDepth = 1;
            let angleDepth = 0;
            let currentItem = [];
            while (pos < tokens.length && (bracketDepth > 0 || angleDepth > 0)) {
                const token = tokens[pos];
                if (token.type === '[') {
                    bracketDepth++;
                    currentItem.push(token);
                    pos++;
                } else if (token.type === ']') {
                    bracketDepth--;
                    if (bracketDepth === 0 && angleDepth === 0) {
                        if (currentItem.length > 0) {
                            const parsed = parseArgumentTokens(currentItem);
                            if (parsed !== null) {
                                items.push(parsed);
                            }
                        }
                        break;
                    } else {
                        currentItem.push(token);
                        pos++;
                    }
                } else if (token.type === '<') {
                    angleDepth++;
                    currentItem.push(token);
                    pos++;
                } else if (token.type === '>') {
                    angleDepth--;
                    currentItem.push(token);
                    pos++;
                } else if (token.type === ',' && bracketDepth === 1 && angleDepth === 0) {
                    if (currentItem.length > 0) {
                        const parsed = parseArgumentTokens(currentItem);
                        if (parsed !== null) {
                            items.push(parsed);
                        }
                        currentItem = [];
                    }
                    pos++;
                } else {
                    currentItem.push(token);
                    pos++;
                }
            }
            return items;
        };

        const tokens = tokenize(value);
        if (!tokens || tokens.length === 0) {
            return null;
        }
        const result = parseTokens(tokens, 0);
        return result ? result.value : null;
    }

    getOperation(opName) {
        const op = this._operations.get(opName);
        if (op && !op.metadata._) {
            if (Array.isArray(op.metadata.operands)) {
                for (const input of op.metadata.operands) {
                    if (input && input.type) {
                        input.type = this._parseConstraint(input.type);
                    }
                }
            }
            if (Array.isArray(op.metadata.results)) {
                for (const output of op.metadata.results) {
                    if (output && output.type) {
                        output.type = this._parseConstraint(output.type);
                    }
                }
            }
            if (Array.isArray(op.metadata.attributes)) {
                for (const attribute of op.metadata.attributes) {
                    if (attribute && attribute.type) {
                        attribute.type = this._parseConstraint(attribute.type);
                    }
                }
            }
            if (Array.isArray(op.metadata.regions)) {
                for (const region of op.metadata.regions) {
                    if (region && region.type) {
                        region.type = this._parseConstraint(region.type);
                    }
                }
            }
            op.metadata._ = true;
        }
        return op || null;
    }

    hasParser(opName) {
        const opInfo = this.getOperation(opName);
        return opInfo ? opInfo.metadata.parser : null;
    }

    hasAssemblyFormat(opName) {
        const opInfo = this.getOperation(opName);
        return opInfo ? opInfo.metadata.assemblyFormat : false;
    }

    hasCustomAssemblyFormat(opName) {
        const opInfo = this.getOperation(opName);
        return opInfo ? opInfo.metadata.hasCustomAssemblyFormat : false;
    }

    hasParseOperation(opName) {
        const opInfo = this.getOperation(opName);
        return opInfo ? opInfo.hasParseOperation : false;
    }

    registerCustomDirective(name, parserFn) {
        this._customDirectives.set(name, parserFn);
    }

    registerCustomType(name, parserFn) {
        this._customTypes.set(name, parserFn);
    }

    registerCustomAttribute(name, parserFn) {
        this._customAttributes.set(name, parserFn);
    }

    parseType(parser, dialectName) {
        const typeName = parser.parseKeyword();
        if (!typeName) {
            return null;
        }
        let type = `!${dialectName}.${typeName}`;
        if (parser.match('<')) {
            type += parser.skip('<');
        }
        return new mlir.Type(type);
    }

    parseDirective(directive, parser, op, opInfo, directives, i) {
        const isVariadic = (type) => {
            if (type.name === 'Variadic' || type.name === 'VariadicOfVariadic') {
                return true;
            }
            if (Array.isArray(type.args) && type.args.length > 0) {
                return isVariadic(type.args[0]);
            }
            return false;
        };
        const isVariadicOfVariadic = (type) => {
            if (type.name === 'VariadicOfVariadic') {
                return true;
            }
            if (Array.isArray(type.args) && type.args.length > 0) {
                return isVariadicOfVariadic(type.args[0]);
            }
            return false;
        };
        const isOptional = (type) => {
            if (type.name === 'Optional') {
                return true;
            }
            if (Array.isArray(type.args) && type.args.length > 0) {
                return isOptional(type.args[0]);
            }
            return false;
        };
        switch (directive.type) {
            case 'whitespace':
                // Skip whitespace directives - they're just formatting hints
                break;
            case 'literal':
                parser.expect(null, directive.value);
                break;
            case 'region_ref': {
                const regionMeta = opInfo.metadata && opInfo.metadata.regions && opInfo.metadata.regions.find((r) => r.name === directive.name);
                const isVariadicRegion = regionMeta && regionMeta.type && regionMeta.type.name === 'VariadicRegion';
                if (isVariadicRegion) {
                    if (parser.match('{')) {
                        do {
                            const region = op.addRegion();
                            parser.parseRegion(region);
                        } while (parser.accept(',') && parser.match('{'));
                    }
                } else {
                    const region = op.addRegion();
                    parser.parseRegion(region);
                }
                break;
            }
            case 'successor_ref': {
                if (!op.successors) {
                    op.successors = [];
                }
                // Check if this successor is variadic from metadata or context
                const refName = directive.name;
                let isVariadicSuccessor = false;
                if (opInfo.metadata && opInfo.metadata.successors) {
                    const successorMeta = opInfo.metadata.successors.find((s) => s.name === refName);
                    if (successorMeta && successorMeta.type) {
                        // Check for VariadicSuccessor type
                        const typeStr = typeof successorMeta.type === 'string' ? successorMeta.type : successorMeta.type.name;
                        isVariadicSuccessor = typeStr && typeStr.startsWith('VariadicSuccessor');
                    }
                }
                // Also check context: if next directive is ')' literal, we're inside parentheses
                const nextDir = i + 1 < directives.length ? directives[i + 1] : null;
                const isVariadicContext = isVariadicSuccessor || (nextDir && nextDir.type === 'literal' && nextDir.value === ')');
                const parseOneSuccessor = () => {
                    const successor = {};
                    successor.label = parser.expect('^');
                    if (parser.accept('(')) {
                        successor.arguments = [];
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
                        if (parser.accept(':')) {
                            let idx = 0;
                            while (idx < successor.arguments.length && !parser.match(')')) {
                                const type = parser.parseType();
                                successor.arguments[idx].type = type.toString();
                                idx++;
                                parser.accept(',');
                            }
                        }
                        parser.accept(')');
                    }
                    op.successors.push(successor);
                };
                if (isVariadicContext) {
                    // Variadic successors: parse 0 or more successors
                    while (parser.match('^')) {
                        parseOneSuccessor();
                        if (!parser.accept(',')) {
                            break;
                        }
                    }
                } else {
                    parseOneSuccessor();
                }
                break;
            }
            case 'attribute_ref': {
                const refName = directive.name;
                // Check if attribute was already parsed in optional group lookahead
                if (op.attributes.has(refName)) {
                    break;
                }
                const attrInfo = opInfo.metadata && opInfo.metadata.attributes && opInfo.metadata.attributes.find((attr) => attr.name === refName);
                const attrType = attrInfo ? attrInfo.type : null;
                let attrValue = null;
                // Pass type to suppress : type suffix parsing (it's a separate directive in assembly format)
                if (attrType && attrType !== 'Attribute') {
                    attrValue = this.parseCustomAttributeWithFallback(parser, attrType);
                } else {
                    attrValue = parser.parseAttribute(attrType || 'Attribute');
                }
                if (attrValue) {
                    op.addAttribute(refName, attrValue);
                }
                break;
            }
            case 'operand_ref': {
                const refName = directive.name;
                let isVariadicOperand = false;
                let isVariadicOfVariadicOperand = false;
                let isOptionalOperand = false;
                if (opInfo.metadata && opInfo.metadata.operands) {
                    const input = opInfo.metadata.operands.find((inp) => inp.name === refName);
                    if (input) {
                        isVariadicOperand = isVariadic(input.type);
                        isVariadicOfVariadicOperand = isVariadicOfVariadic(input.type);
                        isOptionalOperand = isOptional(input.type);
                    }
                }
                if (isVariadicOfVariadicOperand) {
                    // Parse grouped operands: (op, op), (), (op)
                    // Reference: OpFormatGen.cpp variadicOfVariadicOperandParserCode
                    do {
                        if (!parser.accept('(')) {
                            break;
                        }
                        while (parser.match('%')) {
                            const input = {};
                            input.value = parser.expect();
                            op.operands.push(input);
                            if (!parser.accept(',')) {
                                break;
                            }
                        }
                        parser.expect(')');
                    } while (parser.accept(','));
                } else if (isVariadicOperand) {
                    while (!parser.match(')') && !parser.match(']') && !parser.match('}') && !parser.match(':') && !parser.match('{') && !parser.match('=')) {
                        if (parser.match('%')) {
                            const input = {};
                            input.value = parser.expect();
                            op.operands.push(input);
                            if (!parser.accept(',')) {
                                break;
                            }
                        } else {
                            break;
                        }
                    }
                } else if (parser.match('%')) {
                    const input = {};
                    input.value = parser.expect();
                    op.operands.push(input);
                } else if (parser.match('{')) {
                    // Only parse as region if this variable is NOT an actual operand
                    const isActualOperand = opInfo.metadata && opInfo.metadata.operands && opInfo.metadata.operands.some((inp) => inp.name === refName);
                    if (!isActualOperand) {
                        const regionMeta = opInfo.metadata && opInfo.metadata.regions && opInfo.metadata.regions.find((r) => r.name === refName);
                        const isVariadicRegion = regionMeta && regionMeta.type && regionMeta.type.startsWith('VariadicRegion');
                        if (isVariadicRegion) {
                            do {
                                const region = op.addRegion();
                                parser.parseRegion(region);
                            } while (parser.accept(',') && parser.match('{'));
                        } else {
                            const region = op.addRegion();
                            parser.parseRegion(region);
                        }
                    }
                    // If it IS an actual operand, leave the '{' for attr_dict to handle
                } else if (parser.match('@')) {
                    const value = parser.expect('@');
                    if (directive.name) {
                        op.addAttribute(directive.name, value);
                    } else {
                        op.addAttribute('callee', value);
                    }
                } else if (!isOptionalOperand && parser.match('id')) {
                    const input = {};
                    input.value = parser.expect('id');
                    op.operands.push(input);
                } else if (!isOptionalOperand && parser.match('int')) {
                    const input = {};
                    input.value = parser.expect('int');
                    op.operands.push(input);
                } else if (!isOptionalOperand && !parser.match(':') && !parser.match(')') && !parser.match(']') && !parser.match('}') && !parser.match('eof')) {
                    // Try to parse as a general value, but not if we're at a delimiter
                    const input = parser.parseAttribute();
                    if (input) {
                        op.operands.push(input);
                    }
                }
                break;
            }
            case 'operands':
                // Reference: OpFormatGen.cpp - 'operands' directive parses SSA operand values
                // Generates: parser.parseOperandList(allOperands)
                op.operands = parser.parseOperandList();
                break;
            case 'results':
                op.results = parser.parseArguments();
                break;
            case 'type':
            case 'qualified':
                if (directive.args && directive.args.length > 0) {
                    const arg = directive.args[0] === 'type' && directive.args.length > 1 ? directive.args[1] : directive.args[0];
                    if (arg === 'results' || arg === '$results') {
                        const opMetadata = opInfo.metadata;
                        // Check if "results" is an input (operand) or output (result)
                        // For scf.yield, "results" is an input (the values it yields)
                        let isInput = false;
                        if (opMetadata && opMetadata.operands) {
                            isInput = opMetadata.operands.some((inp) => inp.name === 'results');
                        }
                        if (isInput) {
                            const types = parser.parseTypeListNoParens();
                            parser.resolveOperands(op.operands, types);
                        } else {
                            let hasVariadicResult = false;
                            if (opMetadata && opMetadata.results) {
                                if (opMetadata.results.length > 1) {
                                    hasVariadicResult = true;
                                } else if (opMetadata.results.length === 1) {
                                    hasVariadicResult = isVariadic(opMetadata.results[0].type);
                                }
                            }
                            if (hasVariadicResult) {
                                const types = parser.parseTypeListNoParens();
                                parser.resolveOperands(op.results, types);
                            } else {
                                const type = parser.parseType();
                                if (op.results.length === 0) {
                                    op.results.push({ type });
                                } else {
                                    op.results[0].type = type;
                                }
                            }
                        }
                    } else if (arg === 'operands' || arg === '$operands' || arg === 'type(operands)') {
                        const types = parser.parseTypeListNoParens();
                        parser.resolveOperands(op.operands, types);
                    } else if (arg === 'type(results)') {
                        const types = parser.parseTypeListNoParens();
                        parser.resolveOperands(op.results, types);
                    } else {
                        let name = null;
                        switch (directive.type) {
                            case 'type':
                                if (!arg.startsWith('$')) {
                                    throw new mlir.Error(`Invalid operand directive argument '${arg}'.`);
                                }
                                name = arg.substring(1);
                                break;
                            case 'qualified':
                                // Reference: OpFormatGen.cpp - 'qualified' expects attribute or type directive
                                // qualified(type($name)) -> parse type of $name
                                // qualified($name) -> parse attribute $name (like attribute_ref)
                                if (arg.startsWith('type($') && arg.endsWith(')')) {
                                    name = arg.substring(6, arg.length - 1);
                                } else if (arg.startsWith('$')) {
                                    // qualified($name) - parse as attribute reference
                                    const attrName = arg.substring(1);
                                    const attr = parser.parseAttribute();
                                    if (attr) {
                                        op.addAttribute(attrName, attr.value || attr);
                                    }
                                    return; // Early return - attribute handled
                                } else {
                                    throw new mlir.Error(`Invalid operand directive argument '${arg}'.`);
                                }
                                break;
                            default:
                                throw new mlir.Error(`Unknown directive type argument '${arg}'.`);
                        }
                        const opMetadata = opInfo.metadata;
                        let isResult = false;
                        let isVariadicType = false;
                        let isVariadicOfVariadicType = false;
                        if (opMetadata && opMetadata.results) {
                            for (const output of opMetadata.results) {
                                if (output.name === name) {
                                    isResult = true;
                                    isVariadicType = isVariadic(output.type);
                                    isVariadicOfVariadicType = isVariadicOfVariadic(output.type);
                                    break;
                                }
                            }
                        }
                        let isOptionalType = false;
                        if (!isResult && opMetadata && opMetadata.operands) {
                            for (const input of opMetadata.operands) {
                                if (input.name === name) {
                                    isVariadicType = isVariadic(input.type);
                                    isVariadicOfVariadicType = isVariadicOfVariadic(input.type);
                                    isOptionalType = isOptional(input.type);
                                    break;
                                }
                            }
                        }
                        if (isResult) {
                            if (isVariadicType) {
                                const types = parser.parseTypeListNoParens();
                                parser.resolveOperands(op.results, types);
                            } else {
                                const output = opMetadata.results ? opMetadata.results.find((output) => output.name === name) : null;
                                const outputType = output && output.type ? output.type : null;
                                // For optional results, only parse type if there's one present
                                const isOptionalResult = output && isOptional(output.type);
                                if (isOptionalResult && op.results.length === 0) {
                                    // Optional result is absent, try to parse type optionally
                                    const type = parser.parseOptionalType();
                                    if (type) {
                                        op.results.push({ type });
                                    }
                                } else {
                                    const type = this.parseCustomTypeWithFallback(parser, outputType);
                                    if (op.results.length === 0) {
                                        op.results.push({ type });
                                    } else {
                                        op.results[0].type = type;
                                    }
                                }
                            }
                        } else if (isVariadicOfVariadicType) {
                            // Parse grouped types: (type, type), (), (type)
                            // Reference: OpFormatGen.cpp variadicOfVariadicTypeParserCode
                            const allTypes = [];
                            do {
                                if (!parser.accept('(')) {
                                    break;
                                }
                                if (!parser.match(')')) {
                                    const types = parser.parseTypeListNoParens();
                                    allTypes.push(...types);
                                }
                                parser.expect(')');
                            } while (parser.accept(','));
                            parser.resolveOperands(op.operands, allTypes);
                        } else if (isVariadicType) {
                            const types = parser.parseTypeListNoParens();
                            parser.resolveOperands(op.operands, types);
                        } else if (op.operands.length > 0) {
                            const input = opMetadata.operands ? opMetadata.operands.find((input) => input.name === name) : null;
                            const inputType = input && input.type ? input.type : null;
                            const type = this.parseCustomTypeWithFallback(parser, inputType);
                            op.operands[op.operands.length - 1].type = type;
                        } else if (isOptionalType) {
                            // Optional operand is absent, type is also absent - don't parse
                        } else {
                            parser.parseType();
                        }
                    }
                } else {
                    const types = parser.parseTypeListNoParens();
                    parser.resolveOperands(op.operands, types);
                }
                break;
            case 'attr_dict_with_keyword':
                if (parser.accept('id', 'attributes')) {
                    parser.parseAttributeDict(op.attributes);
                }
                break;
            case 'attr_dict':
                parser.parseAttributeDict(op.attributes);
                break;
            case 'prop_dict':
                if (parser.match('<')) {
                    parser.parsePropertyDict(op.attributes);
                }
                break;
            case 'regions':
                while (parser.match('{')) {
                    const region = op.addRegion();
                    parser.parseRegion(region);
                    if (!parser.accept(',')) {
                        break;
                    }
                }
                break;
            case 'successors': {
                op.successors = op.successors || [];
                if (parser.match('^')) {
                    op.successors.push({ label: parser.expect('^') });
                    while (parser.accept(',')) {
                        if (parser.match('^')) {
                            op.successors.push({ label: parser.expect('^') });
                        } else {
                            break;
                        }
                    }
                }
                break;
            }
            case 'functional_type': {
                const type = parser.parseFunctionType();
                if (type instanceof mlir.FunctionType === false) {
                    throw new mlir.Error('Invalid functional-type function type.');
                }
                for (let i = 0; i < type.inputs.length && i < op.operands.length; i++) {
                    if (!op.operands[i].type) {
                        op.operands[i].type = type.inputs[i].toString();
                    }
                }
                for (let i = 0; i < type.results.length; i++) {
                    if (i < op.results.length) {
                        op.results[i].type = op.results[i].type || type.results[i].toString();
                    } else {
                        op.results.push({ type: type.results[i].toString() });
                    }
                }
                break;
            }
            case 'custom': {
                const fn = this._customDirectives.get(directive.parser);
                if (!fn) {
                    throw new mlir.Error(`Custom directive parser '${directive.parser}' not implemented.`);
                }
                fn(parser, op, directive.args);
                break;
            }
            case 'oilist': {
                const clauses = directive.content.split('|').map((c) => c.trim());
                const parsedClauses = [];
                for (const clauseStr of clauses) {
                    const clauseParser = new mlir.AssemblyFormatParser({ ...opInfo.metadata, assemblyFormat: clauseStr });
                    const elements = clauseParser.parse();
                    parsedClauses.push({ elements, parsed: false, clauseStr });
                }
                // Helper to check if a clause's variables are used by later custom directives
                const isHandledByCustomDirective = (clauseStr) => {
                    const varMatches = clauseStr.matchAll(/\$(\w+)/g);
                    const clauseVars = [...varMatches].map((m) => m[1]);
                    if (clauseVars.length === 0) {
                        return false;
                    }
                    for (let j = i + 1; j < directives.length; j++) {
                        const laterDir = directives[j];
                        if (laterDir.type === 'custom' && laterDir.args && Array.isArray(laterDir.args)) {
                            const customVarNames = [];
                            for (const arg of laterDir.args) {
                                const argVarMatches = arg.matchAll(/\$(\w+)/g);
                                for (const match of argVarMatches) {
                                    customVarNames.push(match[1]);
                                }
                            }
                            if (clauseVars.some((v) => customVarNames.includes(v))) {
                                return true;
                            }
                        }
                    }
                    return false;
                };
                let progress = true;
                while (progress) {
                    progress = false;
                    for (const clause of parsedClauses) {
                        if (clause.parsed) {
                            continue;
                        }
                        if (clause.elements.length === 0) {
                            continue;
                        }
                        if (isHandledByCustomDirective(clause.clauseStr)) {
                            clause.parsed = true;
                            continue;
                        }
                        const [firstElem] = clause.elements;
                        let matches = false;
                        if (firstElem.type === 'literal') {
                            if (firstElem.value.length === 1 && /[(){}[\],:<>=]/.test(firstElem.value)) {
                                matches = parser.match(firstElem.value);
                            } else {
                                matches = parser.match('id', firstElem.value) || parser.match('keyword', firstElem.value);
                            }
                        }
                        if (matches) {
                            for (const elem of clause.elements) {
                                this.parseDirective(elem, parser, op, opInfo, directives, i);
                            }
                            clause.parsed = true;
                            progress = true;
                        }
                    }
                }
                break;
            }
            case 'optional_group': {
                let shouldParse = false;
                const firstElem = directive.elements.find((elem) => elem.type !== 'whitespace');
                if (firstElem) {
                    if (firstElem.type === 'literal') {
                        if (firstElem.value.length === 1 && /[(){}[\],:<>=?]/.test(firstElem.value)) {
                            shouldParse = parser.match(firstElem.value);
                        } else if (firstElem.value === '->') {
                            shouldParse = parser.match('->');
                        } else if (firstElem.value === '...') {
                            shouldParse = parser.match('ellipsis');
                        } else {
                            shouldParse = parser.match('id', firstElem.value) || parser.match('keyword', firstElem.value);
                        }
                    } else if (firstElem.type === 'attribute_ref') {
                        const attrInfo = opInfo.metadata && opInfo.metadata.attributes && opInfo.metadata.attributes.find((attr) => attr.name === firstElem.name);
                        const attrType = attrInfo ? attrInfo.type : null;
                        // Check if attribute type is an array (TypedArrayAttrBase, ArrayAttr, etc.)
                        const isArrayAttr = (function checkArrayAttr(t) {
                            if (!t) {
                                return false;
                            }
                            if (typeof t === 'string') {
                                return /ArrayAttr|TypedArrayAttrBase/.test(t);
                            }
                            if (t.name && /ArrayAttr|TypedArrayAttrBase/.test(t.name)) {
                                return true;
                            }
                            if (t.args && Array.isArray(t.args)) {
                                return t.args.some((arg) => checkArrayAttr(arg));
                            }
                            return false;
                        })(attrType);
                        const isIntegerAttr = (function checkIntAttr(t) {
                            if (!t) {
                                return false;
                            }
                            if (typeof t === 'string') {
                                return /I\d+Attr|SI\d+Attr|UI\d+Attr|IntegerAttr|IndexAttr/.test(t);
                            }
                            if (t.name && /I\d+Attr|SI\d+Attr|UI\d+Attr|IntegerAttr|IndexAttr/.test(t.name)) {
                                return true;
                            }
                            if (t.args && Array.isArray(t.args)) {
                                return t.args.some((arg) => checkIntAttr(arg));
                            }
                            return false;
                        })(attrType);
                        const isElementsAttr = (function checkElementsAttr(t) {
                            if (!t) {
                                return false;
                            }
                            if (typeof t === 'string') {
                                return /ElementsAttr|DenseElementsAttr|SparseElementsAttr|DenseResourceElementsAttr/.test(t);
                            }
                            if (t.name && /ElementsAttr|DenseElementsAttr|SparseElementsAttr|DenseResourceElementsAttr/.test(t.name)) {
                                return true;
                            }
                            if (t.args && Array.isArray(t.args)) {
                                return t.args.some((arg) => checkElementsAttr(arg));
                            }
                            return false;
                        })(attrType);
                        let shouldTryParse = false;
                        if (isArrayAttr) {
                            shouldTryParse = parser.match('[');
                        } else if (isIntegerAttr) {
                            shouldTryParse = parser.match('int') || parser.match('-');
                        } else if (isElementsAttr) {
                            // ElementsAttr values start with specific keywords: dense, sparse, array, dense_resource
                            shouldTryParse = parser.match('id', 'dense') || parser.match('id', 'sparse') ||
                                parser.match('id', 'array') || parser.match('id', 'dense_resource');
                        } else {
                            shouldTryParse = parser.match('id') || parser.match('#') || parser.match('@') || parser.match('string') || parser.match('[') || parser.match('int');
                        }
                        if (shouldTryParse) {
                            let result = null;
                            if (attrType && attrType !== 'Attribute') {
                                result = this.parseCustomAttributeWithFallback(parser, attrType);
                            } else {
                                result = parser.parseOptionalAttribute(attrType || 'Attribute');
                            }
                            if (result !== null) {
                                op.addAttribute(firstElem.name, result);
                                shouldParse = true;
                            }
                        }
                    } else if (firstElem.type === 'successor_ref') {
                        shouldParse = parser.match('^');
                    } else if (firstElem.type === 'region_ref') {
                        shouldParse = parser.match('{');
                    } else if (firstElem.type === 'operand_ref') {
                        let isKeywordInput = false;
                        if (opInfo.metadata && opInfo.metadata.operands) {
                            const inputInfo = opInfo.metadata.operands.find((inp) => inp.name === firstElem.name);
                            if (inputInfo) {
                                const inputType = inputInfo.type;
                                if (typeof inputType === 'string' &&
                                    (inputType.includes('Prop') || inputType.endsWith('Predicate') ||
                                        inputType.includes('Flags') || inputType.includes('Enum'))) {
                                    isKeywordInput = true;
                                }
                            }
                        }
                        if (isKeywordInput) {
                            shouldParse = parser.match('id');
                        } else {
                            shouldParse = parser.match('%');
                        }
                    } else if (firstElem.type === 'operands') {
                        shouldParse = parser.match('(') || parser.match('%');
                    } else if (firstElem.type === 'custom') {
                        const fn = this._customDirectives.get(firstElem.parser);
                        if (fn) {
                            const result = fn(parser, op, firstElem.args);
                            if (result === null) {
                                shouldParse = false;
                            } else {
                                shouldParse = 'skip_first';
                            }
                        }
                    } else if (firstElem.type === 'qualified') {
                        if (firstElem.args && firstElem.args.length > 0) {
                            const [arg] = firstElem.args;
                            if (arg.startsWith('$')) {
                                shouldParse = parser.match('#');
                            } else if (arg.startsWith('type($')) {
                                shouldParse = parser.match('!') || parser.match('id');
                            }
                        }
                    }
                }
                if (shouldParse) {
                    // Recursively parse nested elements using the same parseDirective method
                    // If shouldParse === 'skip_first', the custom directive already parsed the first element
                    const startIdx = shouldParse === 'skip_first' ? 1 : 0;
                    for (let elemIdx = startIdx; elemIdx < directive.elements.length; elemIdx++) {
                        this.parseDirective(directive.elements[elemIdx], parser, op, opInfo, directive.elements, elemIdx);
                    }
                }
                break;
            }
            case 'conditional_alternative': {
                const checkMatch = (elem) => {
                    if (elem.type === 'literal') {
                        if (elem.value.length === 1 && /[(){}[\],:<>=?]/.test(elem.value)) {
                            return parser.match(elem.value);
                        }
                        return parser.match('id', elem.value) || parser.match('keyword', elem.value);
                    }
                    if (elem.type === 'operand_ref') {
                        return parser.match('%');
                    }
                    if (elem.type === 'attribute_ref') {
                        return parser.match('id') || parser.match('int') || parser.match('float') || parser.match('[') || parser.match('@') || parser.match('#');
                    }
                    if (elem.type === 'region_ref') {
                        return parser.match('{');
                    }
                    if (elem.type === 'successor_ref') {
                        return parser.match('^');
                    }
                    if (elem.type === 'custom') {
                        // Custom directives can start with various tokens including negative integers
                        return parser.match('id') || parser.match('int') || parser.match('-') || parser.match('%') || parser.match('[') || parser.match('(') || parser.match('?');
                    }
                    return false;
                };
                const firstElem = directive.firstAlt.find((e) => e.type !== 'whitespace');
                let matchedFirst = firstElem && checkMatch(firstElem);
                let customDirectiveHandledFirst = false;
                // For custom directives, try calling them and check if they return null
                if (matchedFirst && firstElem.type === 'custom') {
                    const fn = this._customDirectives.get(firstElem.parser);
                    if (fn) {
                        const result = fn(parser, op, firstElem.args);
                        if (result === null) {
                            matchedFirst = false;
                        } else {
                            customDirectiveHandledFirst = true;
                        }
                    }
                }
                if (matchedFirst) {
                    const startIdx = customDirectiveHandledFirst ? 1 : 0;
                    for (let elemIdx = startIdx; elemIdx < directive.firstAlt.length; elemIdx++) {
                        this.parseDirective(directive.firstAlt[elemIdx], parser, op, opInfo, directive.firstAlt, elemIdx);
                    }
                } else if (directive.secondOptional) {
                    const secondElem = directive.secondAlt.find((e) => e.type !== 'whitespace');
                    const matchedSecond = secondElem && checkMatch(secondElem);
                    if (matchedSecond) {
                        for (const elem of directive.secondAlt) {
                            this.parseDirective(elem, parser, op, opInfo, directive.secondAlt, 0);
                        }
                    }
                } else if (directive.secondAlt && directive.secondAlt.length > 0) {
                    for (const elem of directive.secondAlt) {
                        this.parseDirective(elem, parser, op, opInfo, directive.secondAlt, 0);
                    }
                }
                break;
            }
            default: {
                throw new mlir.Error(`Unsupported directive type '${directive.type}' ${parser.location()}.`);
            }
        }
    }

    parseOperation(parser, opName, op) {
        const opInfo = this.getOperation(opName);
        if (!opInfo) {
            return false;
        }
        if ((this.hasParser(opName) || this.hasCustomAssemblyFormat(opName)) && !this.hasAssemblyFormat(opName)) {
            throw new mlir.Error(`Operation parser '${opName}' not implemented.`);
        }
        const directives = opInfo.directives || [];
        for (let i = 0; i < directives.length; i++) {
            opInfo.hasParseOperation = false;
            this.parseDirective(directives[i], parser, op, opInfo, directives, i);
        }
        return true;
    }

    parseCustomTypeWithFallback(parser, type) {
        if (type && this._customTypes.has(type.name)) {
            let typeT = this._customTypes.get(type.name);
            if (typeof typeT !== 'function') {
                typeT = { type, name: typeT };
            }
            return parser.parseCustomTypeWithFallback(typeT);
        }
        return parser.parseType();
    }

    parseCustomAttributeWithFallback(parser, type) {
        if (this._customAttributes.has(type.name)) {
            const attrT = this._customAttributes.get(type.name);
            return parser.parseCustomAttributeWithFallback(attrT, type);
        }
        if (Array.isArray(type.values)) {
            const value = parser.parseOptionalKeyword(type.values);
            if (value !== null) {
                return new mlir.TypedAttr(value, null);
            }
        }
        // ElementsAttr constraints (I32ElementsAttr, DenseElementsAttr, etc.) are NOT concrete types
        // They're constraints for validation - don't pass them to parser, let it parse full `: type` syntax
        const isElementsAttrConstraint = type && type.name &&
            /ElementsAttr|DenseIntOrFPElements|DenseStringElements/.test(type.name);
        return parser.parseOptionalAttribute(isElementsAttrConstraint ? null : type);
    }

    _parseOptionalAttr(parser, type) {
        if (!Array.isArray(type.args) || type.args.length === 0) {
            throw new mlir.Error(`Invalid 'OptionalAttr' type.`);
        }
        const [elementType] = type.args;
        return this.parseCustomAttributeWithFallback(parser, elementType);
    }

    _parseDefaultValuedAttr(parser, type) {
        if (!Array.isArray(type.args) || type.args.length === 0) {
            throw new mlir.Error(`Invalid 'DefaultValuedAttr' type.`);
        }
        const [elementType] = type.args;
        return this.parseCustomAttributeWithFallback(parser, elementType);
    }

    _parseDefaultValuedOptionalAttr(parser, type) {
        if (!Array.isArray(type.args) || type.args.length === 0) {
            throw new mlir.Error(`Invalid 'DefaultValuedOptionalAttr' type.`);
        }
        const [elementType] = type.args;
        return this.parseCustomAttributeWithFallback(parser, elementType);
    }

    _parseTypeAttrOf(parser, type) {
        if (!Array.isArray(type.args) || type.args.length === 0) {
            throw new mlir.Error(`Invalid 'TypeAttrOf' type.`);
        }
        const parsedType = parser.parseOptionalType();
        if (parsedType) {
            return { value: parsedType, type: 'type' };
        }
        return null;
    }

    _parseAnyAttrOf(parser) {
        // Reference: AnyAttrOf doesn't define a valueType, so LLVM passes Type{} (null)
        // This allows parseAttribute to handle the full syntax including `: type` suffix
        return parser.parseOptionalAttribute(null);
    }

    _parseArrayAttr(parser) {
        if (parser.match('[')) {
            return parser.parseOptionalAttribute();
        }
        // Handle attribute alias references that resolve to arrays
        if (parser.match('#')) {
            return parser.parseAttribute();
        }
        return null;
    }

    _parseConfinedAttr(parser, type) {
        // ConfinedAttr<BaseAttr, Constraints> - parse the base attribute type
        if (!Array.isArray(type.args) || type.args.length === 0) {
            throw new mlir.Error(`Invalid ConfinedAttr type.`);
        }
        const [baseType] = type.args;
        return this.parseCustomAttributeWithFallback(parser, baseType);
    }

    // Reference: TypedAttrInterface attributes parse via parseAttribute()
    // parseAttribute handles `: type` suffix internally when type is null
    _parseTypedAttrInterface(parser) {
        return parser.parseAttribute();
    }

    _parseUnitAttr(parser) {
        // UnitAttr is printed as 'unit' keyword when not elided
        parser.accept('id', 'unit');
        return new mlir.UnitAttr();
    }

    _parseSymbolNameAttr(parser) {
        const value = parser.parseOptionalSymbolName();
        if (value) {
            return new mlir.StringAttr(value);
        }
        return null;
    }

    _parseSymbolRefAttr(parser) {
        const value = parser.parseOptionalSymbolName();
        if (value) {
            // Handle scoped/nested symbol references like @module::@function
            let fullSymbol = value;
            while (parser.accept('::')) {
                if (parser.match('@')) {
                    const nested = parser.parseOptionalSymbolName();
                    if (nested) {
                        fullSymbol += `::@${nested}`;
                    }
                } else {
                    break;
                }
            }
            return new mlir.SymbolRefAttr(fullSymbol);
        }
        return null;
    }

    _parseFlatSymbolRefAttr(parser) {
        return this._parseSymbolRefAttr(parser);
    }

    // Reference: for typed attributes, the type is known so no : type suffix parsing
    _parseIntegerAttr(typeName, parser) {
        const type = new mlir.PrimitiveType(typeName);
        return parser.parseAttribute(type);
    }

    _parseFloatAttr(typeName, parser) {
        const type = new mlir.PrimitiveType(typeName);
        return parser.parseAttribute(type);
    }

    _parseStrAttr(parser) {
        const type = new mlir.PrimitiveType('string');
        return parser.parseAttribute(type);
    }

    _parseDynamicIndexList(parser, op, args) {
        // Determine delimiter from args (default to square brackets for backward compatibility)
        // Args format: [$dynamic_basis, $static_basis, $scalable_basis, "::mlir::AsmParser::Delimiter::Paren"]
        let openDelim = '[';
        let closeDelim = ']';
        if (args && args.length > 3) {
            const [, , , delimiterSpec] = args;
            if (typeof delimiterSpec === 'string' && delimiterSpec.includes('Paren')) {
                openDelim = '(';
                closeDelim = ')';
            }
        }
        const dynamicOperands = [];
        const staticValues = [];
        const scalableFlags = [];
        if (parser.accept(openDelim)) {
            while (!parser.match(closeDelim)) {
                const isScalable = parser.accept('[');
                if (parser.match('%')) {
                    const value = parser.expect('%');
                    dynamicOperands.push({ value });
                    staticValues.push(-9223372036854775808); // ShapedType::kDynamic
                    if (parser.accept(':')) {
                        parser.parseType();
                    }
                } else if (parser.match('int') || parser.match('number')) {
                    const intVal = parseInt(parser.expect(), 10);
                    staticValues.push(intVal);
                } else {
                    break;
                }
                scalableFlags.push(isScalable);
                if (isScalable) {
                    if (!parser.accept(']')) {
                        throw new mlir.Error(`Expected ']' for scalable index ${parser.location()}`);
                    }
                }
                parser.accept(',');
            }
            parser.expect(closeDelim);
        }
        for (const operand of dynamicOperands) {
            op.operands.push(operand);
        }
        if (args && args.length > 1) {
            const staticAttrName = args[1].replace(/^\$/, '');
            op.addAttribute(staticAttrName, staticValues);
        }
        if (args && args.length > 2 && scalableFlags.length > 0) {
            const scalableAttrName = args[2].replace(/^\$/, '');
            op.addAttribute(scalableAttrName, scalableFlags);
        }
    }

    _parseOffsets(parser, op, args) {
        const values = [];
        while (parser.match('int') || parser.match('-')) {
            if (parser.accept('-')) {
                if (parser.match('int')) {
                    values.push(-parser.parseInteger());
                } else {
                    throw new mlir.Error(`Expected integer after '-' in offsets ${parser.location()}`);
                }
            } else {
                values.push(parser.parseInteger());
            }
            if (!parser.accept(',')) {
                break;
            }
        }
        if (args && args.length > 0) {
            const attrName = args[0].replace(/^\$/, '');
            op.addAttribute(attrName, values);
        }
    }

    _parseSymbolVisibility(parser, op, args) {
        let visibility = null;
        if (parser.match('id', 'private') || parser.match('id', 'public') || parser.match('id', 'nested')) {
            visibility = parser.expect('id');
        } else if (parser.match('string')) {
            visibility = parser.expect('string');
        }
        if (visibility && args && args.length > 0) {
            const attrName = args[0].replace(/^\$/, '');
            op.addAttribute(attrName, visibility);
        }
    }

    _parseTypeOrAttr(parser, op, args) {
        const [typeArg, attrArg] = args;
        if (parser.accept('=')) {
            const attr = parser.parseAttribute();
            if (parser.accept(':')) {
                const type = parser.parseType();
                op.addAttribute(typeArg, type);
                attr.type = type;
            } else if (attr && attr.type) {
                op.addAttribute(typeArg, attr.type);
            }
            op.addAttribute(attrArg, attr);
            return;
        }
        if (parser.accept(':')) {
            const type = parser.parseType();
            op.addAttribute(typeArg, type);
            if (parser.accept('=')) {
                const attr = parser.parseAttribute();
                // Handle typed attribute with trailing : type (e.g., -2 : i8)
                if (parser.accept(':')) {
                    const attrType = parser.parseType();
                    attr.type = attrType;
                }
                op.addAttribute(attrArg, attr);
            }
            return;
        }
        throw new mlir.Error(`Expected ':' or '=' in TypeOrAttr ${parser.location()}`);
    }

    _parseSizeAwareType(parser, op, args) {
        // SizeAwareType format: type{%operand}
        // e.g., !hal.buffer{%size} or !util.buffer{%arg0}
        // args: [type($result), $allocation_size]
        const type = parser.parseType();
        parser.expect('{');
        const operand = parser.parseOperand();
        parser.expect('}');
        // Set type on result (first arg is type($result))
        if (args && args.length > 0) {
            const [typeArg] = args;
            const match = typeArg.match(/^type\(\$(\w+)\)$/);
            if (match) {
                if (op.results.length === 0) {
                    op.results.push({ type });
                } else {
                    op.results[0].type = type;
                }
            }
        }
        if (operand) {
            op.operands.push(operand);
        }
    }

    _parseCopyOpRegion(parser, op) {
        op.regions.push({ blocks: [] });
    }

    _parseEnumFlags(parser, type, separator) {
        const flags = [];
        do {
            const value = parser.expect('id');
            if (!type.values.includes(value)) {
                throw new mlir.Error(`Invalid enum value '${value}' ${parser.location()}`);
            }
            flags.push(value);
        } while (parser.accept(separator));
        return new mlir.TypedAttr(flags.join(', '));
    }

    _parseEnumFlagsAngleBracketComma(parser, type) {
        if (parser.accept('<')) {
            const value = this._parseEnumFlags(parser, type, ',');
            parser.expect('>');
            return value;
        }
        return parser.parseOptionalAttribute();
    }

    _parseEnumFlagsAngleBracketPipe(parser, type) {
        if (parser.accept('<')) {
            const value = this._parseEnumFlags(parser, type, '|');
            parser.expect('>');
            return value;
        }
        return parser.parseOptionalAttribute();
    }

    _parseOptional(parser) {
        return parser.parseOptionalType();
    }
};

mlir.HLODialect = class extends mlir.Dialect {

    constructor(operations, name) {
        super(operations, name);
        this.registerCustomDirective('SameOperandsAndResultType', this._parseSameOperandsAndResultType.bind(this));
        this.registerCustomDirective('VariadicSameOperandsAndResultType', this._parseVariadicSameOperandsAndResultType.bind(this));
        this.registerCustomDirective('ComplexOpType', this._parseComplexOpType.bind(this));
        this.registerCustomDirective('SelectOpType', this._parseSelectOpType.bind(this));
        this.registerCustomDirective('TupleOpType', this._parseTupleOpType.bind(this));
        this.registerCustomDirective('PairwiseOpType', this._parsePairwiseOpType.bind(this));
        this.registerCustomDirective('ConvolutionDimensions', this._parseConvolutionDimensions.bind(this));
        this.registerCustomDirective('DotDimensionNumbers', this._parseDotDimensionNumbers.bind(this));
        this.registerCustomDirective('PrecisionConfig', this._parsePrecisionConfig.bind(this));
        this.registerCustomDirective('PrecisionConfigAndAlgorithm', this._parsePrecisionConfigAndAlgorithm.bind(this));
        this.registerCustomDirective('WindowAttributes', this._parseWindowAttributes.bind(this));
        this.registerCustomDirective('SliceRanges', this._parseSliceRanges.bind(this));
        this.registerCustomDirective('CustomCallTarget', this._parseCustomCallTarget.bind(this));
        this.registerCustomDirective('VariadicOperandWithAttribute', this._parseVariadicOperandWithAttribute.bind(this));
    }

    _parseSameOperandsAndResultType(parser, op /*, args */) {
        const type = parser.parseType();
        if (type instanceof mlir.FunctionType) {
            for (let i = 0; i < op.operands.length; i++) {
                const inputType = type.inputs[i];
                if (inputType) {
                    op.operands[i].type = op.operands[i].type || inputType;
                }
            }
            const [resultType] = type.results;
            if (type.results[0] && op.results.length > 0 && !op.results[0].type) {
                op.results[0].type = resultType;
            }
        } else {
            for (const operand of op.operands) {
                operand.type = operand.type || type;
            }
            if (op.results.length > 0 && !op.results[0].type) {
                op.results[0].type = type;
            }
        }
    }

    _parseVariadicSameOperandsAndResultType(parser, op /*, args */) {
        // Parse type: either a bare type (all operands/results same) or function type
        const type = parser.parseType();

        if (type instanceof mlir.FunctionType) {
            for (let i = 0; i < op.operands.length; i++) {
                const inputType = type.getInput(i);
                if (inputType && !op.operands[i].type) {
                    op.operands[i].type = inputType;
                }
            }

            for (let i = 0; i < op.results.length; i++) {
                const resultType = type.getResult(i);
                if (resultType && !op.results[i].type) {
                    op.results[i].type = resultType;
                }
            }
        } else {
            const typeString = type.toString();
            for (const operand of op.operands) {
                if (!operand.type) {
                    operand.type = typeString;
                }
            }
            for (const result of op.results) {
                if (!result.type) {
                    result.type = typeString;
                }
            }
        }
    }

    _parseComplexOpType(parser, op /*, args */) {
        const type = parser.parseType();
        const typeString = type.toString();
        // Apply type to operands and results (for complex operations)
        for (const operand of op.operands) {
            if (!operand.type) {
                operand.type = typeString;
            }
        }
        for (const result of op.results) {
            if (!result.type) {
                result.type = typeString;
            }
        }
    }

    _parseSelectOpType(parser, op /*, args */) {
        const firstType = parser.parseType();
        const firstTypeString = firstType.toString();
        if (parser.accept(',')) {
            const secondType = parser.parseType();
            const secondTypeString = secondType.toString();
            // predType and resultType case
            if (op.operands.length > 0 && !op.operands[0].type) {
                op.operands[0].type = firstTypeString;
            }
            if (op.results.length > 0 && !op.results[0].type) {
                op.results[0].type = secondTypeString;
            }
        } else if (op.results.length > 0 && !op.results[0].type) {
            op.results[0].type = firstTypeString;
        }
    }

    _parseTupleOpType(parser, op /*, args */) {
        const type = parser.parseType();
        // Apply type to operands and results
        for (const operand of op.operands) {
            if (!operand.type) {
                operand.type = type;
            }
        }
        for (const result of op.results) {
            if (!result.type) {
                result.type = type;
            }
        }
    }

    _parsePairwiseOpType(parser, op /*, args */) {
        const types = [];
        while (true) {
            const type = parser.parseType();
            if (!type) {
                break;
            }
            types.push(type);
            if (!parser.accept(',')) {
                break;
            }
        }
        // Apply types pairwise to operands and results
        for (let i = 0; i < types.length && i < op.operands.length; i++) {
            if (!op.operands[i].type) {
                op.operands[i].type = types[i];
            }
        }
        for (let i = 0; i < types.length && i < op.results.length; i++) {
            if (!op.results[i].type) {
                op.results[i].type = types[i];
            }
        }
    }

    _parseDims(parser) {
        const dims = [];
        if (parser.accept('[')) {
            while (!parser.match(']')) {
                if (parser.match('int') || parser.match('number')) {
                    dims.push(parseInt(parser.expect(), 10));
                } else if (parser.match('id')) {
                    dims.push(parser.expect('id'));
                } else {
                    break;
                }
                parser.accept(',');
            }
            parser.accept(']');
        }
        return dims;
    }

    _parseConvolutionDimensions(parser, op, args) {
        const input = this._parseDims(parser);
        parser.expect('id', 'x');
        const kernel = this._parseDims(parser);
        parser.expect('->');
        const output = this._parseDims(parser);
        const name = args[0].substring(1);
        op.addAttribute(name, new mlir.ConvDimensionNumbersAttr(input, kernel, output));
    }

    _parseWindowAttributes(parser, op, args) {
        const windowAttrs = {
            stride: [],
            pad: [],
            lhs_dilate: [],
            rhs_dilate: [],
            window_reversal: []
        };
        const parseArray = () => {
            return parser.parseCommaSeparatedList('square', () => {
                if (parser.match('[')) {
                    return parseArray();
                } else if (parser.match('int') || parser.match('number')) {
                    return parseInt(parser.expect(), 10);
                } else if (parser.match('boolean')) {
                    return parser.expect('boolean');
                } else if (parser.match('id')) {
                    return parser.expect('id');
                }
                return null;
            });
        };
        while (!parser.match('}')) {
            if (parser.match('id')) {
                const key = parser.expect('id');
                if (parser.accept('=')) {
                    windowAttrs[key] = parseArray();
                }
                parser.accept(',');
            } else {
                break;
            }
        }
        if (args && args.length > 0) {
            for (let i = 0; i < args.length; i++) {
                const argName = args[i].replace(/^\$/, '');
                if (argName === 'window_strides' && windowAttrs.stride) {
                    op.addAttribute(argName, windowAttrs.stride);
                } else if (argName === 'padding' && windowAttrs.pad) {
                    op.addAttribute(argName, windowAttrs.pad);
                } else if (argName === 'lhs_dilation' && windowAttrs.lhs_dilate) {
                    op.addAttribute(argName, windowAttrs.lhs_dilate);
                } else if (argName === 'rhs_dilation' && windowAttrs.rhs_dilate) {
                    op.addAttribute(argName, windowAttrs.rhs_dilate);
                } else if (argName === 'window_reversal' && windowAttrs.window_reversal) {
                    op.addAttribute(argName, windowAttrs.window_reversal);
                }
            }
        }
    }

    _parseDotDimensionNumbers(parser, op, args) {
        const dimensions = {
            lhs_batching_dimensions: [],
            rhs_batching_dimensions: [],
            lhs_contracting_dimensions: [],
            rhs_contracting_dimensions: []
        };

        const parseIntArray = () => {
            return parser.parseCommaSeparatedList('optionalSquare', () => {
                if (parser.match('int')) {
                    return parser.parseInteger();
                }
                parser.expect();
                return null;
            });
        };

        const parsePair = () => {
            const first = parseIntArray();
            let second = [];
            if (parser.accept('id', 'x')) {
                second = parseIntArray();
            }
            return { first, second };
        };

        if (parser.match('id', 'batching_dims') || parser.match('id', 'batch_dims')) {
            parser.expect('id');
            parser.accept('=');
            const pair = parsePair();
            dimensions.lhs_batching_dimensions = pair.first;
            dimensions.rhs_batching_dimensions = pair.second;
            parser.accept(',');
        }

        if (parser.accept('id', 'contracting_dims')) {
            parser.accept('=');
            const pair = parsePair();
            dimensions.lhs_contracting_dimensions = pair.first;
            dimensions.rhs_contracting_dimensions = pair.second;
        }

        const attrName = args && args.length > 0 ? args[0].replace(/^\$/, '') : 'dot_dimension_numbers';
        op.addAttribute(attrName, dimensions);
    }

    _parsePrecisionConfig(parser, op /*, args */) {
        parser.accept(',');
        if (!parser.match('id', 'precision')) {
            return;
        }

        parser.expect('id', 'precision');
        parser.expect('=');
        const precision = parser.parseCommaSeparatedList('square', () => {
            if (parser.match('id')) {
                return parser.expect('id');
            }
            parser.expect();
            return null;
        });

        if (precision.length > 0) {
            op.addAttribute('precision_config', precision);
        }
    }

    _parsePrecisionConfigAndAlgorithm(parser, op /*, args */) {
        if (!parser.accept(',')) {
            return;
        }

        if (parser.accept('id', 'algorithm')) {
            parser.accept('=');
            const algorithm = parser.parseAttribute();
            op.addAttribute('algorithm', algorithm);
            return;
        }

        if (parser.accept('id', 'precision')) {
            parser.accept('=');
            const precision = parser.parseCommaSeparatedList('optionalSquare', () => {
                if (parser.match('id')) {
                    return parser.expect('id');
                }
                parser.expect();
                return null;
            });

            if (precision.length > 0) {
                op.addAttribute('precision_config', precision);
            }

            if (parser.accept(',')) {
                if (parser.accept('id', 'algorithm')) {
                    parser.accept('=');
                    const algorithm = parser.parseAttribute();
                    op.addAttribute('algorithm', algorithm);
                }
            }
        }
    }

    _parseSliceRanges(parser, op /*, args */) {
        const ranges = {
            start_indices: [],
            limit_indices: [],
            strides: []
        };

        if (parser.accept('[')) {
            while (!parser.match(']')) {
                if (parser.match('int')) {
                    ranges.start_indices.push(parser.parseInteger());
                }
                parser.accept(':');
                if (parser.match('int')) {
                    ranges.limit_indices.push(parser.parseInteger());
                }
                if (parser.accept(':')) {
                    if (parser.match('int')) {
                        ranges.strides.push(parser.parseInteger());
                    }
                } else {
                    ranges.strides.push(1);
                }
                parser.accept(',');
            }
            parser.accept(']');
        }
        op.addAttribute('start_indices', ranges.start_indices);
        op.addAttribute('limit_indices', ranges.limit_indices);
        op.addAttribute('strides', ranges.strides);
    }

    _parseCustomCallTarget(parser, op, args) {
        let target = null;
        if (parser.match('@')) {
            target = parser.expect('@');
        } else if (parser.match('string')) {
            target = parser.expect('string');
        } else {
            throw new mlir.Error(`Expected '@' or string for CustomCallTarget at ${parser.location()}`);
        }
        const attrName = args && args.length > 0 ? args[0].replace(/^\$/, '') : 'call_target_name';
        op.addAttribute(attrName, target);
    }

    _parseVariadicOperandWithAttribute(parser, op /*, args */) {
        while (parser.match('%')) {
            const operand = {
                value: parser.expect('%'),
                attributes: []
            };
            // Check for inline attributes
            if (parser.match('{')) {
                parser.parseAttributeDict(operand.attributes);
            }
            op.operands.push(operand);
            if (!parser.accept(',')) {
                break;
            }
        }
    }
};

mlir.StableHLODialect = class extends mlir.HLODialect {

    constructor(operations) {
        super(operations, 'stablehlo');
        this.registerCustomDirective('ExponentMantissa', this._parseExponentMantissa.bind(this));
    }

    _parseExponentMantissa(parser, op, args) {
        const keyword = parser.expect('id');
        const match = /^e(\d+)m(\d+)$/.exec(keyword);
        if (!match) {
            throw new mlir.Error(`Expected exponent mantissa in format e#m#, got '${keyword}'`);
        }
        const exponent = parseInt(match[1], 10);
        const mantissa = parseInt(match[2], 10);
        const exponentVar = args[0].replace('$', '');
        const mantissaVar = args[1].replace('$', '');
        op.addAttribute(exponentVar, exponent);
        op.addAttribute(mantissaVar, mantissa);
    }

    parseType(parser, dialectName) {
        const typeName = parser.parseKeyword();
        if (typeName === 'token') {
            return new mlir.Type(`!${dialectName}.token`);
        }
        return null;
    }

    parseOperation(parser, opName, op) {
        // Reference: stablehlo/dialect/StablehloOps.cpp parseConstantOp
        if (opName === 'stablehlo.constant') {
            if (parser.accept('(') && parser.accept(')')) {
                if (parser.accept('<')) {
                    op.propertiesAttr = parser.parseAttribute();
                    parser.expect('>');
                }
                parser.parseOptionalAttrDict(op.attributes);
                parser.expect(':');
                parser.expect('(');
                parser.expect(')');
                parser.expect('->');
                const type = parser.parseType();
                op.results.push({ type: type.toString() });
            } else {
                // Custom form: {attrs} value : type
                parser.parseOptionalAttrDict(op.attributes);
                const value = parser.parseAttribute();
                if (value) {
                    op.addAttribute('value', value);
                }
                // Parse result type - either explicit `: type` or from value's type
                const types = parser.parseOptionalColonTypeList();
                if (types.length > 0) {
                    op.results.push({ type: types[0].toString() });
                } else if (value && value.type) {
                    op.results.push({ type: value.type.toString() });
                }
            }
            return true;
        }
        if (opName === 'stablehlo.while' && parser.match('(')) {
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
                const cond = op.addRegion();
                parser.parseRegion(cond);
            }
            if (parser.accept('id', 'do')) {
                const body = op.addRegion();
                parser.parseRegion(body);
            }
            return true;
        }
        if ((opName === 'stablehlo.reduce' || opName === 'stablehlo.scan') && parser.match('(')) {
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
            op.addAttribute('body_op', innerOpName);
            if (parser.accept('id', 'across')) {
                if (parser.accept('id', 'dimensions')) {
                    parser.accept('=');
                    const dims = parser.parseAttribute();
                    op.addAttribute('dimensions', dims);
                }
            }
            if (parser.match('{')) {
                parser.parseAttributeDict(op.attributes);
            }
            parser.resolveOperands(op.operands, parser.parseOptionalColonTypeList());
            parser.resolveOperands(op.results, parser.parseOptionalArrowTypeList());
            return true;
        }
        if (parser.match('(')) {
            if (parser.accept('(') && parser.match('{')) {
                let regionCount = 0;
                while (!parser.match(')')) {
                    if (regionCount++ > 10) {
                        throw new mlir.Error(`Too many regions in region-list (>10) - possible infinite loop at ${parser.location()}, current token: '${parser.token.value}'`);
                    }
                    if (!parser.match('{')) {
                        throw new mlir.Error(`Expected '{' for region in region-list, got '${parser.token.value}' at ${parser.location()}`);
                    }
                    const region = op.addRegion();
                    parser.parseRegion(region);
                    if (!parser.accept(',') && !parser.match(')')) {
                        throw new mlir.Error(`Expected ',' or ')' after region, got '${parser.token.value}' at ${parser.location()}`);
                    }
                }
                parser.expect(')');
            }
            if (parser.match('{')) {
                parser.parseAttributeDict(op.attributes);
            }
            parser.resolveOperands(op.operands, parser.parseOptionalColonTypeList());
            if (parser.accept('->') || parser.accept('id', 'to')) {
                if (op.results.length > 0) {
                    const types = parser.parseFunctionResultTypes();
                    parser.resolveOperands(op.results, types);
                } else {
                    op.results = parser.parseArguments();
                }
            }

            return true;
        }
        if (parser.accept('id', 'across')) {
            if (parser.accept('id', 'dimensions')) {
                parser.expect('=');
                const dims = parser.parseAttribute();
                op.addAttribute('dimensions', dims.value);
            }
        }
        if (parser.match('{')) {
            parser.parseAttributeDict(op.attributes);
        }
        parser.resolveOperands(op.operands, parser.parseOptionalColonTypeList());
        if (parser.accept('->') || parser.accept('id', 'to')) {
            if (op.results.length > 0) {
                const types = parser.parseFunctionResultTypes();
                parser.resolveOperands(op.results, types);
            } else {
                op.results = parser.parseArguments();
            }
        }
        if (parser.match('id') && !parser.match('keyword', 'loc')) {
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
            const region = op.addRegion();
            parser.parseRegion(region);
            parser.expect(')');
        } else if (parser.match('{')) {
            const region = op.addRegion();
            parser.parseRegion(region);
        }
        return true;
    }
};

mlir.VhloDialect = class extends mlir.Dialect {

    constructor(operations) {
        super(operations, 'vhlo');
        this.registerCustomDirective('FunctionBody', this._parseFunctionBody.bind(this));
    }

    parseOperation(parser, opName, op) {
        if (opName === 'vhlo.constant_v1') {
            if (parser.match('{')) {
                parser.parseAttributeDict(op.attributes);
            }
            const value = parser.parseAttribute();
            if (value) {
                op.addAttribute('value', value);
            }
            parser.resolveOperands(op.results, parser.parseOptionalColonTypeList());
            return true;
        }

        if (opName === 'vhlo.return_v1') {
            op.operands = parser.parseArguments();
            parser.resolveOperands(op.operands, parser.parseOptionalColonTypeList());
            return true;
        }

        return super.parseOperation(parser, opName, op);
    }

    _parseFunctionBody(parser, op) {
        parser.parseFunctionOp(op, false);
    }
};

mlir.InterpreterDialect = class extends mlir.Dialect {

    constructor(operations) {
        super(operations, 'interpreter');
    }
};

mlir.AffineDialect = class extends mlir.Dialect {

    constructor(operations) {
        super(operations, 'affine');
    }

    parseOperation(parser, opName, op) {
        if (opName === 'affine.parallel') {
            return this._parseParallelOp(parser, op);
        }
        // Special handling for affine.for - similar to scf.for but with affine expressions
        if (opName === 'affine.for') {
            return this._parseForOp(parser, op);
        }
        // Special handling for affine.if - has condition before region
        if (opName === 'affine.if') {
            // affine.if #set(dims)[symbols] [-> (type)] { region }
            // Or: affine.if affine_set<(d0) : (constraint)>(dims)[symbols]
            if (parser.match('#')) {
                const condition = parser.parseAttribute();
                op.addAttribute('condition', condition);
            } else if (parser.match('id', 'affine_set')) {
                parser.expect('id', 'affine_set');
                const content = parser.skip('<');
                op.addAttribute('condition', `affine_set${content}`);
            }
            if (parser.accept('(')) {
                while (!parser.accept(')')) {
                    if (parser.match('%')) {
                        const operand = parser.expect('%');
                        op.operands.push({ value: operand });
                    }
                    parser.accept(',');
                }
            }
            if (parser.accept('[')) {
                while (!parser.accept(']')) {
                    if (parser.match('%')) {
                        const operand = parser.expect('%');
                        op.operands.push({ value: operand });
                    }
                    parser.accept(',');
                }
            }
            parser.parseOptionalArrowResultTypes(op);
            const region = op.addRegion();
            parser.parseRegion(region);
            if (parser.accept('id', 'else')) {
                const elseRegion = {};
                parser.parseRegion(elseRegion);
                op.regions.push(elseRegion);
            }
            parser.parseOptionalAttrDict(op.attributes);
            return true;
        }
        // Special handling for affine.apply, affine.min, and affine.max
        if (opName === 'affine.apply' || opName === 'affine.min' || opName === 'affine.max') {
            if (parser.match('#') || parser.match('id', 'affine_map') || parser.match('id', 'affine_set')) {
                const value = parser.parseAttribute();
                op.addAttribute('map', value);
            }
            if (parser.match('(')) {
                const dimOperands = parser.parseOperandList('paren');
                op.operands.push(...dimOperands);
            }
            const symOperands = parser.parseOperandList('optionalSquare');
            op.operands.push(...symOperands);
            parser.parseOptionalAttrDict(op.attributes);
            return true;
        }
        if (opName === 'affine.store') {
            return this._parseStoreOp(parser, op);
        }
        if (opName === 'affine.load') {
            return this._parseLoadOp(parser, op);
        }
        if (opName === 'affine.vector_load') {
            return this._parseVectorLoadOp(parser, op);
        }
        if (opName === 'affine.vector_store') {
            return this._parseVectorStoreOp(parser, op);
        }
        if (opName === 'affine.prefetch') {
            const memref = parser.expect('%');
            op.operands.push({ value: memref });
            parser.skip('[');
            parser.expect(',');
            const rwSpecifier = parser.parseKeyword();
            op.addAttribute('isWrite', rwSpecifier === 'write');
            parser.expect(',');
            parser.expect('id', 'locality');
            parser.expect('<');
            const locality = parser.expect('int');
            op.addAttribute('localityHint', locality);
            parser.expect('>');
            parser.expect(',');
            const cacheType = parser.parseKeyword();
            op.addAttribute('isDataCache', cacheType === 'data');
            parser.parseOptionalAttrDict(op.attributes);
            if (parser.accept(':')) {
                const type = parser.parseType();
                op.operands[0].type = type;
            }
            return true;
        }
        // C++-only operation: affine.dma_start
        // Defined in mlir/lib/Dialect/Affine/IR/AffineOps.cpp
        if (opName === 'affine.dma_start') {
            // Format: affine.dma_start %src[indices], %dst[indices], %tag[indices], %num_elements [, %stride, %num_elt_per_stride] : memref, memref, memref
            while (!parser.match(':') && !parser.match('{')) {
                if (parser.match('%')) {
                    const operand = parser.expect('%');
                    op.operands.push({ value: operand });
                }
                if (parser.match('[')) {
                    parser.skip('[');
                }
                parser.accept(',');
            }
            parser.parseOptionalAttrDict(op.attributes);
            if (parser.accept(':')) {
                do {
                    parser.parseType();
                } while (parser.accept(','));
            }
            return true;
        }
        if (opName === 'affine.dma_wait') {
            // Format: affine.dma_wait %tag[indices], %num_elements : memref
            while (!parser.match(':') && !parser.match('{')) {
                if (parser.match('%')) {
                    const operand = parser.expect('%');
                    op.operands.push({ value: operand });
                }
                if (parser.match('[')) {
                    parser.skip('[');
                }
                parser.accept(',');
            }
            parser.parseOptionalAttrDict(op.attributes);
            if (parser.accept(':')) {
                parser.parseType();
            }
            return true;
        }
        return super.parseOperation(parser, opName, op);
    }

    _parseForOp(parser, op) {
        const inductionVar = parser.expect('%');
        parser.parseLocation();
        parser.expect('=');
        this._parseAffineBound(parser, op, 'lowerBound');
        parser.expect('id', 'to');
        this._parseAffineBound(parser, op, 'upperBound');
        if (parser.accept('id', 'step')) {
            if (parser.match('int')) {
                const step = parser.expect('int');
                op.addAttribute('step', step);
            }
        }
        if (parser.accept('id', 'iter_args')) {
            if (parser.accept('(')) {
                while (!parser.accept(')')) {
                    if (parser.match('%')) {
                        parser.expect('%');
                    }
                    if (parser.accept('=')) {
                        if (parser.match('%')) {
                            op.operands.push({ value: parser.expect('%') });
                        } else {
                            const value = parser.parseAttribute();
                            if (value) {
                                op.operands.push(value);
                            }
                        }
                    }
                    parser.accept(',');
                }
            }
            parser.parseOptionalArrowResultTypes(op);
        }
        if (parser.match('{')) {
            const region = {};
            parser.parseRegion(region);
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
        parser.parseOptionalAttrDict(op.attributes);
        return true;
    }

    _parseAffineBound(parser, op, boundName) {
        // Parse affine bound following reference implementation in AffineOps.cpp parseBound()
        // Syntax: [max|min] (ssa-id | integer | affine-map dim-and-symbol-list)

        // Try parsing SSA value first (shorthand for identity map)
        if (parser.match('%')) {
            const value = parser.expect('%');
            op.operands.push({ value });
            const mapAttrName = boundName === 'lowerBound' ? 'lowerBoundMap' : 'upperBoundMap';
            op.addAttribute(mapAttrName, 'symbol_identity');
            return;
        }

        // Try parsing integer literal (shorthand for constant map)
        if (parser.match('int') || parser.match('-')) {
            const negate = parser.accept('-');
            let value = parser.parseInteger();
            if (negate) {
                value = -value;
            }
            const mapAttrName = boundName === 'lowerBound' ? 'lowerBoundMap' : 'upperBoundMap';
            op.addAttribute(mapAttrName, value);
            return;
        }

        if (!parser.accept('id', 'min')) {
            parser.accept('id', 'max');
        }
        if (parser.match('#') || parser.match('id', 'affine_map')) {
            const mapValue = parser.parseAttribute();
            if (mapValue) {
                const mapAttrName = boundName === 'lowerBound' ? 'lowerBoundMap' : 'upperBoundMap';
                op.addAttribute(mapAttrName, mapValue);

                // Parse dim and symbol operands in ()[...]  or (...)
                if (parser.accept('(')) {
                    while (!parser.accept(')')) {
                        if (parser.match('%')) {
                            const operand = parser.expect('%');
                            op.operands.push({ value: operand });
                        }
                        parser.accept(',');
                    }
                }
                if (parser.accept('[')) {
                    while (!parser.accept(']')) {
                        if (parser.match('%')) {
                            const operand = parser.expect('%');
                            op.operands.push({ value: operand });
                        }
                        parser.accept(',');
                    }
                }
                return;
            }
        }
        throw new mlir.Error(`Expected loop bound (SSA value, integer, or affine map) in affine.for ${parser.location()}`);
    }

    _parseStoreOp(parser, op) {
        if (parser.match('%')) {
            const value = parser.expect('%');
            op.operands.push({ value });
        } else {
            const value = parser.parseAttribute();
            op.operands.push(value);
        }
        if (!parser.accept('id', 'to')) {
            parser.accept(',');
        }
        const address = parser.expect('%');
        op.operands.push({ value: address });
        parser.skip('[');
        if (parser.accept(':')) {
            const type = parser.parseType();
            op.operands[1].type = type;
        }
        return true;
    }

    _parseLoadOp(parser, op) {
        const address = parser.expect('%');
        op.operands.push({ value: address });
        parser.skip('[');
        if (parser.accept(':')) {
            const type = parser.parseType();
            op.operands[0].type = type;
        }
        return true;
    }

    _parseVectorLoadOp(parser, op) {
        const memref = parser.expect('%');
        op.operands.push({ value: memref });
        parser.skip('[');
        parser.parseOptionalAttrDict(op.attributes);
        if (parser.accept(':')) {
            const memrefType = parser.parseType();
            op.operands[0].type = memrefType;
            parser.expect(',');
            const vectorType = parser.parseType();
            op.results.push({ type: vectorType });
        }
        return true;
    }

    _parseVectorStoreOp(parser, op) {
        const value = parser.expect('%');
        op.operands.push({ value });
        parser.expect(',');
        const memref = parser.expect('%');
        op.operands.push({ value: memref });
        parser.skip('[');
        parser.parseOptionalAttrDict(op.attributes);
        if (parser.accept(':')) {
            const memrefType = parser.parseType();
            op.operands[1].type = memrefType;
            parser.expect(',');
            const vectorType = parser.parseType();
            op.operands[0].type = vectorType;
        }
        return true;
    }

    _parseParallelOp(parser, op) {
        const ivs = parser.parseArguments();
        if (!parser.accept('=')) {
            return false;
        }
        parser.skip('(');
        if (!parser.accept('id', 'to')) {
            return false;
        }
        parser.skip('(');
        if (parser.accept('id', 'step')) {
            parser.skip('(');
        }
        if (parser.accept('id', 'reduce')) {
            parser.expect('(');
            while (!parser.match(')')) {
                if (parser.match('string')) {
                    parser.expect('string');
                }
                if (!parser.accept(',')) {
                    break;
                }
            }
            parser.expect(')');
        }
        if (parser.accept('->')) {
            const resultTypes = [];
            const resultAttrs = [];
            parser.parseFunctionResultList(resultTypes, resultAttrs);
        }
        if (parser.match('{')) {
            const region = {};
            parser.parseRegion(region);
            if (region.blocks && region.blocks.length > 0) {
                if (!region.blocks[0].arguments) {
                    region.blocks[0].arguments = [];
                }
                region.blocks[0].arguments = ivs;
            }
            op.regions.push(region);
        }
        parser.parseOptionalAttrDict(op.attributes);
        return true;
    }
};

mlir.MemRefDialect = class extends mlir.Dialect {

    constructor(operations) {
        super(operations, 'memref');
        this.registerCustomDirective('GlobalMemrefOpTypeAndInitialValue', this._parseGlobalMemrefOpTypeAndInitialValue.bind(this));
        // AtomicRMWKindAttr can appear as bare id (addi) or string ("addi") in different test files
        this.registerCustomAttribute('AtomicRMWKindAttr', this._parseAtomicRMWKindAttr.bind(this));
    }

    _parseAtomicRMWKindAttr(parser, type) {
        // Accept both bare identifier (addi) and string literal ("addi")
        if (parser.match('string')) {
            return parser.expect('string');
        }
        if (parser.match('id') && type.values && type.values.includes(parser.getToken().value)) {
            return parser.expect('id');
        }
        return null;
    }

    _parseGlobalMemrefOpTypeAndInitialValue(parser, op, args) {
        // Parse: type [= initializer]
        // args[0] is "$type", args[1] is "$initial_value"
        const typeArg = args && args.length > 0 ? args[0].replace(/^\$/, '') : 'type';
        const initialValueArg = args && args.length > 1 ? args[1].replace(/^\$/, '') : 'initial_value';
        const type = parser.parseType();
        op.addAttribute(typeArg, { value: type, type: 'type' });

        // Parse optional initializer: = <value> or = uninitialized
        if (parser.accept('=')) {
            if (parser.accept('id', 'uninitialized')) {
                op.addAttribute(initialValueArg, 'uninitialized');
            } else {
                // Pass the type to parseAttribute to suppress : type suffix parsing
                const initialValue = parser.parseAttribute(type);
                op.addAttribute(initialValueArg, initialValue);
            }
        }
    }

    parseOperation(parser, opName, op) {
        if (opName === 'memref.tensor_load') {
            op.operands = parser.parseArguments();
            parser.resolveOperands(op.operands, parser.parseOptionalColonTypeList());
            return true;
        }
        if (opName === 'memref.store') {
            this._operations.get(opName).hasParseOperation = false; // compatibility
            return this._parseStoreOp(parser, op);
        }
        if (opName === 'memref.alloca_scope') {
            return this._parseAllocaScopeOp(parser, op);
        }
        if (opName === 'memref.transpose') {
            return this._parseTransposeOp(parser, op);
        }
        if (opName === 'memref.generic_atomic_rmw') {
            return this._parseGenericAtomicRMWOp(parser, op);
        }
        if (opName === 'memref.prefetch') {
            const memref = parser.parseAttribute();
            op.operands.push(memref);
            if (parser.accept('[')) {
                while (!parser.match(']')) {
                    const index = parser.parseAttribute();
                    op.operands.push(index);
                    parser.accept(',');
                }
                parser.expect(']');
            }
            parser.expect(',');
            const readOrWrite = parser.expect('id');
            op.addAttribute('isWrite', readOrWrite === 'write');
            parser.expect(',');
            parser.expect('id', 'locality');
            parser.expect('<');
            const localityHint = parseInt(parser.expect('int'), 10);
            op.addAttribute('localityHint', localityHint);
            parser.expect('>');
            parser.expect(',');
            const cacheType = parser.expect('id');
            op.addAttribute('isDataCache', cacheType === 'data');
            parser.parseOptionalAttrDict(op.attributes);
            if (parser.accept(':')) {
                const type = parser.parseType();
                memref.type = type.toString();
            }
            return true;
        }
        if (opName === 'memref.dma_start' && this.hasCustomAssemblyFormat(opName) && !this.hasAssemblyFormat(opName)) {
            const srcMemRef = parser.parseOperand();
            op.operands.push(srcMemRef);
            const srcIndices = parser.parseOperandList('square');
            op.operands.push(...srcIndices);
            parser.expect(',');
            const dstMemRef = parser.parseOperand();
            op.operands.push(dstMemRef);
            const dstIndices = parser.parseOperandList('square');
            op.operands.push(...dstIndices);
            parser.expect(',');
            const numElements = parser.parseOperand();
            op.operands.push(numElements);
            parser.expect(',');
            const tagMemRef = parser.parseOperand();
            op.operands.push(tagMemRef);
            const tagIndices = parser.parseOperandList('square');
            op.operands.push(...tagIndices);
            const strideInfo = [];
            while (parser.accept(',') && parser.match('%')) {
                strideInfo.push(parser.parseOperand());
            }
            op.operands.push(...strideInfo);
            parser.expect(':');
            const srcType = parser.parseType();
            srcMemRef.type = srcType.toString();
            parser.expect(',');
            const dstType = parser.parseType();
            dstMemRef.type = dstType.toString();
            parser.expect(',');
            const tagType = parser.parseType();
            tagMemRef.type = tagType.toString();
            numElements.type = 'index';
            for (const idx of srcIndices) {
                idx.type = 'index';
            }
            for (const idx of dstIndices) {
                idx.type = 'index';
            }
            for (const idx of tagIndices) {
                idx.type = 'index';
            }
            for (const idx of strideInfo) {
                idx.type = 'index';
            }
            return true;
        }
        return super.parseOperation(parser, opName, op);
    }

    _parseGenericAtomicRMWOp(parser, op) {
        const memref = parser.parseOperand();
        const indices = parser.parseOperandList('square');
        op.operands.push(memref);
        op.operands.push(...indices);
        parser.expect(':');
        const memrefType = parser.parseType();
        memref.type = memrefType;
        for (const idx of indices) {
            idx.type = 'index';
        }
        const region = op.addRegion();
        parser.parseRegion(region);
        parser.parseOptionalAttrDict(op.attributes);
        if (memrefType && memrefType.elementType) {
            op.results.push({ type: memrefType.elementType });
        }
        return true;
    }

    _parseTransposeOp(parser, op) {
        // Reference: MemRefOps.cpp TransposeOp::parse
        // Format: $in $permutation attr-dict : type($in) `to` type(results)
        const operand = { value: parser.expect('%') };
        op.operands.push(operand);
        // Parse affine map permutation: (d0, d1) -> (d1, d0)
        // This is a bare affine map, not wrapped in affine_map<...>
        const dims = parser.skip('(');
        parser.expect('->');
        const results = parser.skip('(');
        const permutation = `affine_map<${dims} -> ${results}>`;
        op.addAttribute('permutation', permutation);
        parser.parseOptionalAttrDict(op.attributes);
        if (parser.accept(':')) {
            const srcType = parser.parseType();
            operand.type = srcType;
        }
        parser.expect('id', 'to');
        const dstType = parser.parseType();
        parser.resolveOperands(op.results, [dstType]);
        return true;
    }

    _parseAllocaScopeOp(parser, op) {
        // Reference: MemRefOps.cpp AllocaScopeOp::parse
        // Format: [-> (type, ...)] { region } [attr-dict]
        const resultTypes = parser.parseOptionalArrowTypeList();
        parser.resolveOperands(op.results, resultTypes);
        const region = op.addRegion();
        parser.parseRegion(region);
        parser.parseOptionalAttrDict(op.attributes);
        return true;
    }

    _parseStoreOp(parser, op) {
        // Parse: value, memref[indices] {attributes} : type
        // or old: value to memref[indices] : type
        if (parser.match('%')) {
            const value = parser.expect('%');
            op.operands.push({ value });
        } else {
            const value = parser.parseAttribute();
            op.operands.push(value);
        }
        // Accept either ',' (new) or 'to' (old)
        if (!parser.accept('id', 'to')) {
            parser.accept(',');
        }
        const address = parser.expect('%');
        op.operands.push({ value: address });
        parser.skip('[');
        parser.parseAttributeDict(op.attributes);
        if (parser.accept(':')) {
            const type = parser.parseType();
            op.operands[1].type = type;
        }
        return true;
    }
};

mlir.VectorDialect = class extends mlir.Dialect {

    constructor(operations) {
        super(operations, 'vector');
        this.registerCustomAttribute('Vector_CombiningKindAttr', this._parseEnumFlagsAngleBracketComma.bind(this));
        this.registerCustomAttribute('Arith_FastMathAttr', this._parseEnumFlagsAngleBracketComma.bind(this));
    }

    parseOperation(parser, opName, op) {
        if (opName === 'vector.splat') {
            op.operands = parser.parseArguments();
            parser.resolveOperands(op.results, parser.parseOptionalColonTypeList());
            return true;
        }
        if (opName === 'vector.contract') {
            if (parser.match('{')) {
                parser.skip('{');
            } else if (parser.match('#')) {
                parser.expect('#');
            }
            op.operands = parser.parseArguments();
            parser.parseOptionalAttrDict(op.attributes);
            const types = parser.parseColonTypeList();
            parser.resolveOperands(op.operands, types);
            // Reference: parseKeywordType("into", resultType)
            parser.expect('id', 'into');
            const resultType = parser.parseType();
            parser.resolveOperands(op.results, [resultType]);
            return true;
        }
        if (opName === 'vector.mask') {
            if (parser.match('%')) {
                const mask = parser.expect('%');
                op.operands.push({ value: mask, name: 'mask' });
            }
            if (parser.accept(',')) {
                const passthru = parser.expect('%');
                op.operands.push({ value: passthru, name: 'passthru' });
            }
            if (parser.match('{')) {
                const region = op.addRegion();
                parser.parseRegion(region);
            }
            parser.parseOptionalAttrDict(op.attributes);
            parser.resolveOperands(op.operands, parser.parseOptionalColonTypeList());
            parser.resolveOperands(op.results, parser.parseOptionalArrowTypeList());
            return true;
        }
        if (opName === 'vector.outerproduct') {
            return this._parseOuterProductOp(parser, op);
        }
        if (opName === 'vector.transfer_read' || opName === 'vector.transfer_write') {
            return this._parseTransferOp(parser, op);
        }
        if (opName === 'vector.extract' && !op.isGeneric) {
            this._operations.get(opName).hasParseOperation = false; // compatibility
            return this._parseExtractOp(parser, op);
        }
        return super.parseOperation(parser, opName, op);
    }

    _parseOuterProductOp(parser, op) {
        const lhs = parser.expect('%');
        op.operands.push({ value: lhs, name: 'lhs' });
        parser.expect(',');
        const rhs = parser.expect('%');
        op.operands.push({ value: rhs, name: 'rhs' });
        if (parser.accept(',')) {
            const acc = parser.expect('%');
            op.operands.push({ value: acc, name: 'acc' });
        }
        parser.parseOptionalAttrDict(op.attributes);
        if (parser.accept(':')) {
            const lhsType = parser.parseType();
            op.operands[0].type = lhsType.toString();
            parser.expect(',');
            const rhsType = parser.parseType();
            op.operands[1].type = rhsType.toString();
        }
        return true;
    }

    _parseExtractOp(parser, op) {
        // Old syntax (pre-2023): %r = vector.extract %v[0] : vector<4xf32>
        // New syntax: %r = vector.extract %v[0] : f32 from vector<4xf32>
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
        //    or: vector.transfer_read %source[%i, %j, ...], %padding, %mask {attrs} : memref_type, vector_type
        //    or: vector.transfer_write %value, %dest[%i, %j, ...] {attrs} : vector_type, memref_type
        //    or: vector.transfer_write %value, %dest[%i, %j, ...], %mask {attrs} : vector_type, memref_type

        const first = parser.expect('%');
        op.operands.push({ value: first });

        const hasIndicesAfterFirst = parser.match('[');
        if (hasIndicesAfterFirst) {
            parser.skip('[');
        }
        parser.accept(',');
        const second = parser.expect('%');
        op.operands.push({ value: second });
        if (!hasIndicesAfterFirst && parser.match('[')) {
            parser.skip('[');
        }

        // Optional mask parameter (third operand)
        if (parser.accept(',')) {
            const mask = parser.expect('%');
            op.operands.push({ value: mask });
        }
        if (parser.match('{')) {
            parser.parseAttributeDict(op.attributes);
        }
        // Type signature: : memref_type, vector_type
        if (parser.accept(':')) {
            const type1 = parser.parseType();
            op.operands[0].type = type1.toString();
            parser.accept(',');
            const type2 = parser.parseType();
            // For transfer_read, type2 is the result type
            // For transfer_write, type2 is just the vector type
            if (op.results.length > 0) {
                op.results[0].type = type2.toString();
            }
            op.operands[1].type = type2.toString();
        }

        return true;
    }
};

mlir.TensorDialect = class extends mlir.Dialect {

    constructor(operations) {
        super(operations, 'tensor');
    }

    parseOperation(parser, opName, op) {
        if (opName === 'tensor.expand_shape') {
            // The new tensor.expand_shape format includes 'output_shape':
            //   $src $reassociation `output_shape` custom<DynamicIndexList>(...) attr-dict `:` type($src) `into` type($result)
            // Old format (deprecated):
            //   $src $reassociation attr-dict `:` type($src) `into` type($result)
            this.getOperation(opName).hasParseOperation = false; // compatibility
            // Parse operand
            const operand = { value: parser.expect('%') };
            op.operands.push(operand);
            // Parse reassociation attribute [[...]]
            const reassociation = parser.parseAttribute();
            op.addAttribute('reassociation', reassociation);
            // Check for new vs old format
            if (parser.accept('id', 'output_shape')) {
                // New format: parse output_shape dynamic index list
                this._parseDynamicIndexList(parser, op, ['$output_shape', '$static_output_shape']);
            }
            // Both formats: attr-dict `:` type($src) `into` type($result)
            parser.parseOptionalAttrDict(op.attributes);
            parser.expect(':');
            operand.type = parser.parseType();
            parser.expect('id', 'into');
            const resultType = parser.parseType();
            op.results.push({ type: resultType });
            return true;
        }
        return super.parseOperation(parser, opName, op);
    }
};

mlir.TorchDialect = class extends mlir.Dialect {

    constructor(operations) {
        super(operations, 'torch');
        this.simpleTypes = new Set([
            'int', 'float', 'bool', 'str', 'none', 'Device', 'Generator',
            'qint8', 'quint8', 'qint16', 'qint32', 'quint4x2', 'quint2x4',
            'LinearParams', 'number', 'any'
        ]);
    }

    parseType(parser, dialectName) {
        const typeName = parser.parseKeyword();
        if (!typeName) {
            return null;
        }
        let type = `!${dialectName}.${typeName}`;
        if (this.simpleTypes.has(typeName)) {
            return new mlir.Type(type);
        }
        if (typeName === 'vtensor' || typeName === 'tensor' || typeName === 'list' || typeName === 'tuple' || typeName === 'union' || typeName === 'optional' || typeName === 'dict' || typeName.startsWith('nn.')) {
            if (parser.match('<')) {
                const content = parser.skip('<');
                type += content;
            }
            return new mlir.Type(type);
        }
        return null;
    }

    parseOperation(parser, opName, op) {
        if (opName === 'torch.constant.int') {
            if (parser.match('int')) {
                const value = parser.expect('int');
                op.addAttribute('value', value);
            }
            parser.parseOptionalAttrDict(op.attributes);
            return true;
        }
        if (opName === 'torch.onnx.rotary_embedding') {
            op.operands = parser.parseArguments();
            parser.resolveOperands(op.operands, parser.parseOptionalColonTypeList());
            if (parser.accept('->')) {
                const resultType = parser.parseType();
                op.results.push({ type: resultType });
            }
            return true;
        }
        if (opName === 'torch.bind_symbolic_shape') {
            const operand = { value: parser.expect('%') };
            op.operands.push(operand);
            parser.accept(',');
            // Reference: TorchOps.cpp:6363 uses parseOperandList
            const shapeSymbols = parser.parseOperandList('square');
            op.operands.push(...shapeSymbols);
            parser.accept(',');
            const shapeExpr = parser.parseAttribute();
            op.addAttribute('shape_expressions', shapeExpr.value || shapeExpr);
            parser.parseOptionalAttrDict(op.attributes);
            if (parser.accept(':')) {
                operand.type = parser.parseType();
            }
            return true;
        }
        // Reference: TorchOps.cpp InitializeGlobalSlotsOp::parse
        // Format: [ @slot0(%0 : !torch.int), @slot1(%1 : !torch.float) ]
        if (opName === 'torch.initialize.global_slots') {
            parser.parseOptionalAttrDict(op.attributes);
            parser.expect('[');
            const slotSymNames = [];
            while (!parser.accept(']')) {
                const slotSymName = parser.expect('@');
                slotSymNames.push(slotSymName);
                parser.expect('(');
                const operand = { value: parser.expect('%') };
                parser.expect(':');
                operand.type = parser.parseType();
                op.operands.push(operand);
                parser.expect(')');
            }
            op.addAttribute('slotSymNames', slotSymNames);
            return true;
        }
        if (this.hasCustomAssemblyFormat(opName) && !this.hasAssemblyFormat(opName)) {
            return this._parseDefaultTorchOp(parser, op);
        }
        return super.parseOperation(parser, opName, op);
    }

    _parseDefaultTorchOp(parser, op) {
        op.operands = parser.parseArguments();
        parser.parseOptionalAttrDict(op.attributes);
        if (parser.accept(':')) {
            parser.resolveOperands(op.operands, parser.parseTypeList());
        }
        if (parser.accept('->')) {
            // Handle both -> (type, type) and -> type, type syntaxes
            const types = parser.match('(') ? parser.parseTypeListParens() : parser.parseTypeListNoParens();
            parser.resolveOperands(op.results, types);
        }
        if (parser.match('{')) {
            const region = op.addRegion();
            parser.parseRegion(region);
            if (parser.accept('id', 'else') && parser.match('{')) {
                const elseRegion = {};
                parser.parseRegion(elseRegion);
                op.regions.push(elseRegion);
            }
        }
        return true;
    }
};

mlir.IREEDialect = class extends mlir.Dialect {

    constructor(operations, name) {
        super(operations, name);
        this.registerCustomDirective('DispatchEntryPoints', this._parseDispatchEntryPoints.bind(this));
        this.registerCustomDirective('ShapedTiedResult', this._parseShapedTiedResult.bind(this));
        this.registerCustomDirective('SymbolAlias', this._parseSymbolAlias.bind(this));
        this.registerCustomDirective('TypeAlias', this._parseTypeAlias.bind(this));
        this.registerCustomDirective('WorkgroupCountRegion', this._parseWorkgroupCountRegion.bind(this));
        this.registerCustomDirective('ShapedFunctionType', this._parseShapedFunctionType.bind(this));
    }

    // Reference: UtilOps.cpp parseShapedFunctionType
    // Format: (type{%dims}, type) -> (type{%dims}, %arg as type{%dims})
    _parseShapedFunctionType(parser /*, op, args */) {
        parser.expect('(');
        if (!parser.match(')')) {
            do {
                parser.parseType();
                if (parser.match('{')) {
                    parser.skip('{');
                }
            } while (parser.accept(','));
        }
        parser.expect(')');
        parser.expect('->');
        const parseResultTypeOrTied = () => {
            if (parser.match('%')) {
                parser.expect('%');
                if (parser.accept('id', 'as')) {
                    parser.parseType();
                }
            } else {
                parser.parseType();
            }
            if (parser.match('{')) {
                parser.skip('{');
            }
        };
        if (parser.accept('(')) {
            if (!parser.match(')')) {
                do {
                    parseResultTypeOrTied();
                } while (parser.accept(','));
            }
            parser.expect(')');
        } else {
            parseResultTypeOrTied();
        }
    }

    _parseDispatchEntryPoints(parser, op, args) {
        // Parse either:
        // - Single: @symbol or @symbol::@nested
        // - Multiple: {@symbol1, @symbol2::@nested2}
        const entryPoints = [];

        if (parser.accept('{')) {
            // Parse multiple entry points
            do {
                if (parser.match('@')) {
                    let symbol = parser.expect('@');
                    // Handle :: nested symbol reference
                    if (parser.accept('id', '::') || (parser.match(':') && parser.accept(':') && parser.accept(':'))) {
                        if (parser.match('@')) {
                            const nested = parser.expect('@');
                            symbol += `::${nested}`;
                        }
                    }
                    entryPoints.push(symbol);
                }
            } while (parser.accept(','));
            parser.expect('}');
        } else if (parser.match('@')) {
            // Parse single entry point
            let symbol = parser.expect('@');
            // Handle :: nested symbol reference
            if (parser.accept('id', '::') || (parser.match(':') && parser.accept(':') && parser.accept(':'))) {
                if (parser.match('@')) {
                    const nested = parser.expect('@');
                    symbol += `::${nested}`;
                }
            }
            entryPoints.push(symbol);
        }

        const attrName = args && args.length > 0 ? args[0].replace(/^\$/, '') : 'entry_points';
        const value = entryPoints.length === 1 ? entryPoints[0] : entryPoints;
        op.addAttribute(attrName, value);
    }

    _parseShapedTiedResult(parser, op /*, args */) {
        // Parse: %arg0 as tensor<?x?xf32>{%d0, %d1}
        //    or: tensor<?x?xf32>{%d0, %d1}
        let tiedOperand = null;
        if (parser.match('%')) {
            tiedOperand = parser.expect('%');
            parser.expect('id', 'as');
        }
        const resultType = parser.parseType();
        const dims = [];
        if (parser.accept('{')) {
            while (!parser.match('}')) {
                if (parser.match('%')) {
                    const dim = parser.expect('%');
                    dims.push(dim);
                    parser.accept(',');
                } else {
                    break;
                }
            }
            parser.expect('}');
        }
        // Add result with type and tied operand info
        op.results.push({ type: resultType, tiedOperand, dims });
    }

    _parseSymbolAlias(parser, op, args) {
        // @foo or @foo as("bar")
        const alias = parser.expect('@');
        let symName = alias;
        if (parser.accept('id', 'as')) {
            if (parser.accept('(')) {
                if (parser.match('string')) {
                    symName = parser.expect('string');
                } else if (parser.match('@')) {
                    symName = parser.expect('@');
                }
                parser.accept(')');
            }
        }
        if (args && args.length >= 2) {
            const symNameArg = args[0].replace(/^\$/, '');
            const aliasArg = args[1].replace(/^\$/, '');
            op.addAttribute(symNameArg, symName);
            op.addAttribute(aliasArg, alias);
        }
    }

    _parseTypeAlias(parser, op, args) {
        const encodingType = parser.parseType();
        let storageType = encodingType;
        if (parser.accept('id', 'as')) {
            storageType = parser.parseType();
        }
        if (args && args.length >= 1) {
            const encodingAttrName = args[0].replace(/^\$/, '');
            op.addAttribute(encodingAttrName, encodingType);
        }
        if (args && args.length >= 2) {
            const typeMatch = args[1].match(/type\(\$(\w+)\)/);
            if (typeMatch) {
                if (op.results.length > 0) {
                    op.results[0].type = storageType;
                } else {
                    op.results.push({ type: storageType });
                }
            } else if (op.operands.length > 0) {
                op.operands[0].type = storageType;
            }
        }
    }

    _parseWorkgroupCountRegion(parser, op, args) {
        if (!parser.match('id', 'workgroups')) {
            return;
        }
        parser.expect('id', 'workgroups');
        const region = { blocks: [] };
        const block = { arguments: [], operations: [] };
        if (parser.accept('(')) {
            while (!parser.match(')')) {
                const arg = { value: parser.expect('%') };
                if (parser.accept(':')) {
                    arg.type = parser.parseType();
                }
                block.arguments.push(arg);
                if (!parser.accept(',')) {
                    break;
                }
            }
            parser.expect(')');
        }
        if (parser.accept('->')) {
            parser.expect('(');
            while (!parser.match(')')) {
                parser.parseType();
                if (!parser.accept(',')) {
                    break;
                }
            }
            parser.expect(')');
        }
        if (parser.accept('{')) {
            while (!parser.match('}')) {
                const innerOp = parser.parseOperation();
                if (innerOp) {
                    block.operations.push(innerOp);
                }
                if (parser.match('}')) {
                    break;
                }
            }
            parser.expect('}');
        }
        region.blocks.push(block);
        if (args && args.length > 0) {
            const regionArgName = args[0].replace(/^\$/, '');
            if (regionArgName) {
                op.regions.push(region);
            }
        } else {
            op.regions.push(region);
        }
    }
};

mlir.HALDialect = class extends mlir.IREEDialect {

    constructor(operations) {
        super(operations, 'hal');
        this.simpleTypes = new Set(['allocator', 'buffer', 'buffer_view', 'channel', 'command_buffer', 'descriptor_set', 'descriptor_set_layout', 'device', 'event', 'executable', 'executable_layout', 'fence', 'file', 'semaphore']);
        this.registerCustomAttribute('HAL_PipelineLayoutAttr', this._parsePipelineLayoutAttr.bind(this));
        this.registerCustomDirective('ExportConditionRegion', this._parseExportConditionRegion.bind(this));
        this.registerCustomDirective('TargetConditionObjects', this._parseTargetConditionObjects.bind(this));
        this.registerCustomDirective('WorkgroupCountRegion', this._parseWorkgroupCountRegion.bind(this));
    }

    parseType(parser, dialectName) {
        const typeName = parser.parseKeyword();
        if (!typeName) {
            return null;
        }
        if (this.simpleTypes.has(typeName)) {
            // Note: !hal.buffer{%size} syntax is handled by custom<SizeAwareType> directive,
            // not by the type parser. The type parser just returns the base type.
            return new mlir.Type(`!${dialectName}.${typeName}`);
        }
        return null;
    }

    parseOperation(parser, opName, op) {
        if (opName === 'hal.tensor.cast') {
            op.operands = parser.parseArguments();
            if (parser.accept(':')) {
                const type = parser.parseType();
                if (op.operands.length > 0) {
                    op.operands[0].type = type;
                }
            }
            if (parser.accept('->')) {
                const types = parser.parseFunctionResultTypes();
                parser.resolveOperands(op.results, types);
            }
            return true;
        }
        if (opName === 'hal.constant') {
            parser.parseOptionalAttrDictWithKeyword(op.attributes);
            const value = parser.parseAttribute();
            op.addAttribute('value', value.value === undefined ? value : value.value);
            // Reference: parseOptionalColonTypeList
            parser.resolveOperands(op.results, parser.parseOptionalColonTypeList());
            return true;
        }
        if (opName === 'hal.device.switch') {
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
            if (parser.accept('->') || parser.accept(':')) {
                const resultType = parser.parseType();
                op.results = [{ type: resultType }];
            }
            while (parser.match('#')) {
                const region = {};
                const caseAttr = parser.parseAttribute();
                region.caseAttribute = caseAttr;
                if (parser.match('{')) {
                    parser.parseRegion(region);
                }
                op.regions.push(region);
                parser.accept(',');
            }
            return true;
        }
        if (opName === 'hal.executable.constant.block' && this.hasCustomAssemblyFormat(opName) && !this.hasAssemblyFormat(opName)) {
            if (parser.accept('(')) {
                while (!parser.match(')')) {
                    if (parser.match('%')) {
                        const arg = parser.expect('%');
                        parser.expect(':');
                        const type = parser.parseType();
                        op.operands.push({ value: arg, type });
                    }
                    if (!parser.accept(',')) {
                        break;
                    }
                }
                parser.expect(')');
            }
            if (parser.accept('->')) {
                const resultTypes = parser.parseFunctionResultTypes();
                op.addAttribute('function_type', new mlir.TypeAttrOf(new mlir.FunctionType([], resultTypes)));
            }
            if (parser.accept('id', 'as')) {
                if (parser.match('(')) {
                    parser.expect('(');
                    const keys = [];
                    while (!parser.match(')')) {
                        if (parser.match('string')) {
                            keys.push(parser.expect('string'));
                        }
                        if (!parser.accept(',')) {
                            break;
                        }
                    }
                    parser.expect(')');
                    op.addAttribute('keys', keys);
                } else if (parser.match('string')) {
                    const key = parser.expect('string');
                    op.addAttribute('keys', [key]);
                }
            }
            if (parser.match('{')) {
                const region = op.addRegion();
                parser.parseRegion(region);
            }
            return true;
        }
        // Handle hal.executable.create with both old (layouts) and new (affinity) syntax
        if (opName === 'hal.executable.create') {
            const opInfo = this.getOperation(opName);
            opInfo.hasParseOperation = false; // compatibility?
            const inputNames = new Set(opInfo.metadata.operands.map((input) => input.name));
            // Parse named parameters: device(...), target(...), and either layouts(...) or affinity(...)
            while (parser.match('id') && !parser.match(':') && !parser.match('loc')) {
                const paramName = parser.expect('id');
                if (parser.accept('(')) {
                    // Check if this named parameter is actually an input
                    if (inputNames.has(paramName) && parser.match('%')) {
                        // Parse as an operand: %value : type
                        const operand = parser.expect('%');
                        let operandType = null;
                        if (parser.accept(':')) {
                            operandType = parser.parseType();
                        }
                        parser.expect(')');
                        op.operands.push({ name: paramName, value: operand, type: operandType });
                    } else if (inputNames.has(paramName) && parser.match('[')) {
                        // Parse as a variadic operand: [%v1, %v2, ...]
                        parser.expect('[');
                        while (!parser.match(']')) {
                            if (parser.match('%')) {
                                const operand = parser.expect('%');
                                op.operands.push({ name: paramName, value: operand });
                            }
                            parser.accept(',');
                        }
                        parser.expect(']');
                        parser.expect(')');
                    } else {
                        // Parse as an attribute with raw content
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
                        op.addAttribute(normalizedName, paramValue);
                    }
                } else {
                    break;
                }
            }
            // Reference: parseOptionalColonTypeList
            parser.resolveOperands(op.results, parser.parseOptionalColonTypeList());
            return true;
        }
        // Handle operations with <%operand : type> syntax and/or named parameters
        // e.g., hal.allocator.compute_size<%allocator : !hal.allocator> shape([...]) type(...) encoding(...) : index
        // or hal.executable_layout.lookup device(%device : !hal.device) layouts([[...]]) : !hal.executable_layout
        // Exclude hal.executable, hal.interface, and hal.device.switch which have special handling
        if ((opName.startsWith('hal.allocator.') || opName.startsWith('hal.buffer.') || opName.startsWith('hal.buffer_view.') ||
            opName.startsWith('hal.command_buffer.') || opName.startsWith('hal.executable_layout') ||
            opName.startsWith('hal.executable.') || opName.startsWith('hal.descriptor_set_layout') ||
            opName.startsWith('hal.device.')) &&
            opName !== 'hal.device.allocator' &&
            opName !== 'hal.buffer_view.buffer' &&
            opName !== 'hal.executable' &&
            opName !== 'hal.interface' &&
            opName !== 'hal.device.switch' &&
            opName !== 'hal.device.memoize' &&
            opName !== 'hal.command_buffer.execution_barrier' &&
            opName !== 'hal.executable.entry_point' &&
            opName !== 'hal.executable.variant' &&
            opName !== 'hal.executable.lookup' &&
            opName !== 'hal.interface.binding' &&
            opName !== 'hal.executable.create' &&
            opName !== 'hal.executable.export' &&
            opName !== 'hal.executable.binary' &&
            opName !== 'hal.executable.source' &&
            opName !== 'hal.executable.condition' &&
            opName !== 'hal.executable.constant.block' &&
            opName !== 'hal.executable.constant.load') {
            // Parse <%operand : type> if present
            if (opName === 'hal.allocator.allocate' || opName === 'hal.command_buffer.create' || opName === 'hal.buffer_view.create' || opName === 'hal.command_buffer.device' || opName === 'hal.command_buffer.dispatch' || opName === 'hal.device.query') {
                this.getOperation(opName).hasParseOperation = false; // compatibility?
            }
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
            // Also exclude common operation keywords that shouldn't be treated as parameters
            const notParameterNames = new Set(['br', 'cond_br', 'return', 'yield', 'call', 'unreachable', 'assert']);
            while (parser.match('[') ||
                (parser.match('id') && !parser.match('id', 'attributes') &&
                    !parser.match(':') && !parser.match('loc') &&
                    parser.token.value && parser.token.value.indexOf('.') === -1 &&
                    !notParameterNames.has(parser.token.value))) {
                // Handle bracket expressions (e.g., [%c0])
                if (parser.match('[')) {
                    parser.skip('[');
                    continue;
                }
                const paramName = parser.expect('id');
                if (parser.accept('(')) {
                    // Check if this named parameter is actually an input from the operation metadata
                    const opInfo = this.getOperation(opName);
                    const inputNames = new Set((opInfo && opInfo.metadata && opInfo.metadata.operands || []).map((i) => i.name));
                    if (inputNames.has(paramName) && parser.match('%')) {
                        // Parse as a simple operand: %value : type
                        const operand = parser.expect('%');
                        let operandType = null;
                        if (parser.accept(':')) {
                            operandType = parser.parseType();
                        }
                        parser.expect(')');
                        op.operands.push({ name: paramName, value: operand, type: operandType });
                    } else if (inputNames.has(paramName) && parser.match('[')) {
                        // Parse as a variadic operand: [%v1, %v2, ...]
                        parser.expect('[');
                        while (!parser.match(']')) {
                            if (parser.match('%')) {
                                const operand = parser.expect('%');
                                op.operands.push({ name: paramName, value: operand });
                            }
                            parser.accept(',');
                        }
                        parser.expect(']');
                        parser.expect(')');
                    } else {
                        // Parse as an attribute with raw content
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
                        op.addAttribute(paramName, paramValue);
                    }
                } else {
                    // Not a named parameter - we've consumed an id token that doesn't belong to us
                    // This shouldn't happen with proper MLIR, but break gracefully
                    break;
                }
            }
            // Reference: parseOptionalColonTypeList
            parser.resolveOperands(op.results, parser.parseOptionalColonTypeList());
            // Handle old IREE format: !hal.buffer{%size} where {%size} follows the type
            if (parser.match('{')) {
                parser.skip('{');
            }
            if (parser.accept('=')) {
                const value = parser.parseAttribute();
                op.addAttribute('default', value.value);
            }
            parser.parseOptionalAttrDictWithKeyword(op.attributes);
            return true;
        }
        if (opName === 'hal.executable.condition' || opName === 'hal.executable.constant.block') {
            const sig = parser.parseFunctionSignatureWithArguments(false);
            const argTypes = sig.arguments.map((a) => a.type);
            const type = new mlir.FunctionType(argTypes, sig.resultTypes);
            op.addAttribute('function_type', new mlir.TypeAttrOf(type));
            parser.parseOptionalAttrDictWithKeyword(op.attributes);
            if (parser.match('{')) {
                const region = op.addRegion();
                parser.parseRegion(region, sig.arguments);
            }
            return true;
        }
        // Handle operations with visibility + symbol (similar to flow dialect)
        if (opName === 'hal.executable' || opName === 'hal.executable.source' || opName === 'hal.interface' || opName === 'hal.executable.binary') {
            this.getOperation(opName).hasParseOperation = false; // compatibility?
            if (parser.match('id', 'private') || parser.match('id', 'public') || parser.match('id', 'nested')) {
                parser.expect('id');
            }
            if (parser.match('@')) {
                parser.parseSymbolName('sym_name', op.attributes);
            }
            if (parser.accept('id', 'attributes')) {
                parser.parseAttributeDict(op.attributes);
            }
            if (parser.match('{')) {
                const region = op.addRegion();
                parser.parseRegion(region);
            }
            return true;
        }
        // Handle hal.interface.binding.subspan with old syntax (symbol reference)
        // Old syntax: hal.interface.binding.subspan @io::@binding[operand] : type
        // New syntax: hal.interface.binding.subspan layout(...) binding(...) : type
        if (opName === 'hal.interface.binding.subspan' && parser.match('@')) {
            this.getOperation(opName).hasParseOperation = false; // compatibility?
            // Old syntax - parse symbol reference and bracket expression
            const symbolRef = parser.expect('@');
            op.addAttribute('layout', symbolRef);
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
            if (parser.accept(':')) {
                const type = parser.parseType();
                if (type) {
                    op.results.push({ type: type.toString() });
                }
                if (parser.accept('{')) {
                    const dynamicDims = [];
                    if (!parser.match('}')) {
                        do {
                            const dimOperand = parser.parseSSAUse();
                            dynamicDims.push(dimOperand);
                            op.operands.push({ value: dimOperand.name, kind: 'dynamic_dim' });
                        } while (parser.accept(','));
                    }
                    parser.expect('}');
                }
            }
            return true;
        }
        // Handle operations with named parameters: hal.interface.binding, hal.executable.variant, etc.
        if (opName === 'hal.interface.binding' || opName === 'hal.executable.variant' || opName === 'hal.executable.entry_point' || opName === 'hal.executable.export') {
            this.getOperation(opName).hasParseOperation = false; // compatibility?
            if (parser.match('id', 'private') || parser.match('id', 'public') || parser.match('id', 'nested')) {
                parser.expect('id');
            }
            if (parser.match('@')) {
                parser.parseSymbolName('sym_name', op.attributes);
                parser.accept(',');
            }
            while (parser.match('id') && !parser.match('id', 'attributes') && !parser.match('{') && !parser.match('loc')) {
                const tokenValue = parser.token.value;
                if (tokenValue && tokenValue.includes('.')) {
                    break;
                }
                const paramName = parser.expect('id');
                if (paramName === 'condition') {
                    parser.expect('(');
                    const regionArgs = [];
                    while (!parser.match(')')) {
                        const arg = parser.expect('%');
                        let type = null;
                        if (parser.accept(':')) {
                            type = parser.parseType();
                        }
                        regionArgs.push({ value: arg, type });
                        if (!parser.accept(',')) {
                            break;
                        }
                    }
                    parser.expect(')');
                    parser.expect('->');
                    parser.parseType();
                    const conditionRegion = { arguments: regionArgs };
                    parser.parseRegion(conditionRegion);
                    op.regions.push(conditionRegion);
                    continue;
                }
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
                    op.addAttribute(paramName, paramValue);
                    parser.accept(',');
                } else if (parser.accept('=')) {
                    if (parser.match('#')) {
                        const value = parser.parseAttribute();
                        op.addAttribute(paramName, value.value);
                    } else if (parser.match('string')) {
                        const value = parser.expect('string');
                        op.addAttribute(paramName, value);
                    } else {
                        const value = parser.expect();
                        op.addAttribute(paramName, value);
                    }
                    if (!parser.accept(',')) {
                        break;
                    }
                } else {
                    break;
                }
            }
            if (parser.accept('->')) {
                const resultTypes = [];
                const resultAttrs = [];
                parser.parseFunctionResultList(resultTypes, resultAttrs);
            }
            if (parser.accept('id', 'attributes')) {
                parser.parseAttributeDict(op.attributes);
            }
            if (parser.accept('id', 'count')) {
                this._parseWorkgroupCountRegion(parser, op);
            }
            if (parser.match('{')) {
                const region = op.addRegion();
                parser.parseRegion(region);
            }
            if (parser.accept('id', 'attributes')) {
                parser.parseAttributeDict(op.attributes);
            }
            if (parser.accept('id', 'count')) {
                this._parseWorkgroupCountRegion(parser, op);
            }
            return true;
        }
        return super.parseOperation(parser, opName, op);
    }

    _parsePipelineLayoutAttr(parser) {
        // HAL_PipelineLayoutAttr format: <constants = N, bindings = [...], flags = ...>
        if (parser.match('<')) {
            return parser.parseAttribute();
        }
        return parser.parseOptionalAttribute();
    }

    _parseExportConditionRegion(parser, op, args) {
        parser.expect('(');
        const regionArgs = [];
        while (!parser.match(')')) {
            const arg = parser.expect('%');
            let type = null;
            if (parser.accept(':')) {
                type = parser.parseType();
            }
            regionArgs.push({ value: arg, type });
            if (!parser.accept(',')) {
                break;
            }
        }
        parser.expect(')');
        parser.expect('->');
        parser.parseType();
        const region = { arguments: regionArgs };
        parser.parseRegion(region);
        if (args && args.length >= 1) {
            const regionName = args[0].replace('$', '');
            if (regionName === 'condition' && op.regions) {
                op.regions.push(region);
            }
        }
    }

    _parseTargetConditionObjects(parser, op) {
        // #target if(...) { region } ordinal(N) = [objects], ...
        do {
            if (parser.match('#')) {
                parser.parseAttribute();
            }
            if (parser.accept('id', 'if')) {
                this._parseTargetConditionRegion(parser, op);
            }
            if (parser.accept('id', 'ordinal')) {
                parser.expect('(');
                parser.expect('int');
                parser.expect(')');
            }
            if (parser.accept('=')) {
                if (parser.match('[')) {
                    parser.skip('[');
                }
            }
        } while (parser.accept(','));
    }

    _parseTargetConditionRegion(parser, op) {
        parser.expect('(');
        while (!parser.match(')') && !parser.match('eof')) {
            parser.expect('%');
            if (parser.accept(':')) {
                parser.parseType();
            }
            if (!parser.accept(',')) {
                break;
            }
        }
        parser.expect(')');
        if (parser.accept('->')) {
            parser.parseType();
        }
        if (parser.match('{')) {
            const region = op.addRegion();
            parser.parseRegion(region);
        }
    }

    _parseWorkgroupCountRegion(parser, op) {
        // (args) -> (index, index, index) { region }
        const region = { blocks: [] };
        const block = { arguments: [], operations: [] };
        if (parser.accept('(')) {
            while (!parser.match(')') && !parser.match('eof')) {
                const arg = { value: parser.expect('%') };
                if (parser.accept(':')) {
                    arg.type = parser.parseType();
                }
                block.arguments.push(arg);
                if (!parser.accept(',')) {
                    break;
                }
            }
            parser.expect(')');
        }
        if (parser.accept('->')) {
            parser.expect('(');
            while (!parser.match(')') && !parser.match('eof')) {
                parser.parseType();
                if (!parser.accept(',')) {
                    break;
                }
            }
            parser.expect(')');
        }
        region.blocks.push(block);
        if (parser.match('{')) {
            parser.parseRegion(region);
        }
        op.regions.push(region);
    }
};

mlir.IREECodegenDialect = class extends mlir.Dialect {

    constructor(operations) {
        super(operations, 'iree_codegen');
    }

    parseOperation(parser, opName, op) {
        if (opName === 'iree_codegen.workgroup_count_hint' && this.hasCustomAssemblyFormat(opName) && !this.hasAssemblyFormat(opName)) {
            const staticSizes = [];
            parser.accept('id', 'sizes');
            parser.expect('(');
            while (!parser.match(')')) {
                if (parser.match('%')) {
                    const value = parser.expect('%');
                    op.operands.push({ value, name: 'sizes' });
                    staticSizes.push(-9223372036854775808);
                } else if (parser.match('int')) {
                    const constValue = parser.expect('int');
                    staticSizes.push(parseInt(constValue, 10));
                }
                if (!parser.accept(',')) {
                    break;
                }
            }
            parser.expect(')');
            if (staticSizes.length > 0) {
                op.addAttribute('static_sizes', staticSizes);
            }
            parser.parseOptionalAttrDict(op.attributes);
            return true;
        }
        return super.parseOperation(parser, opName, op);
    }
};

mlir.HALLoaderDialect = class extends mlir.Dialect {

    constructor(operations) {
        super(operations, 'hal_loader');
        this.registerCustomDirective('DispatchBindings', this._parseDispatchBindings.bind(this));
    }

    _parseDispatchBindings(parser, op) {
        const buffers = [];
        const bufferTypes = [];
        const bufferOffsets = [];
        const bufferLengths = [];
        do {
            parser.expect('(');
            const buffer = parser.expect('%');
            buffers.push({ value: buffer });
            parser.expect(':');
            const bufferType = parser.parseType();
            bufferTypes.push(bufferType);
            parser.expect(')');
            parser.expect('[');
            const bufferOffset = parser.expect('%');
            bufferOffsets.push({ value: bufferOffset });
            parser.expect(',');
            const bufferLength = parser.expect('%');
            bufferLengths.push({ value: bufferLength });
            parser.expect(']');
        } while (parser.accept(','));
        for (const buffer of buffers) {
            op.operands.push(buffer);
        }
        for (let i = 0; i < bufferTypes.length; i++) {
            if (i < op.operands.length) {
                op.operands[i].type = bufferTypes[i];
            }
        }
        for (const offset of bufferOffsets) {
            op.operands.push(offset);
        }
        for (const length of bufferLengths) {
            op.operands.push(length);
        }
    }
};

mlir.UtilDialect = class extends mlir.IREEDialect {

    constructor(operations) {
        super(operations, 'util');
        this.registerCustomDirective('OperandTypeList', this._parseOperandTypeList.bind(this));
        this.registerCustomDirective('TiedFunctionResultList', this._parseTiedFunctionResultList.bind(this));
        this.registerCustomDirective('TypeAlias', this._parseTypeAlias.bind(this));
        this.registerCustomDirective('TypedValueList', this._parseTypedValueList.bind(this));
        this.registerCustomDirective('RangeList', this._parseRangeList.bind(this));
        this.registerCustomDirective('ListTypeGet', this._parseListTypeGet.bind(this));
        this.registerCustomDirective('ListTypeSet', this._parseListTypeSet.bind(this));
        this.registerCustomDirective('ValueTypeList', this._parseValueTypeList.bind(this));
        this.simpleTypes = new Set(['buffer', 'list', 'object', 'ptr']);
    }

    _parseTypeAlias(parser /*, op, args */) {
        parser.parseType();
        if (parser.accept('id', 'as')) {
            parser.parseType();
        }
    }

    _parseTypedValueList(parser, op /*, args */) {
        // Reference: UtilOps.cpp parseTypedValueList
        // Format: [%val1, %val2, ...] where all values have same type (passed as ref)
        parser.expect('[');
        if (!parser.match(']')) {
            do {
                const value = parser.expect('%');
                op.operands.push({ name: 'values', value });
            } while (parser.accept(','));
        }
        parser.expect(']');
    }

    parseType(parser, dialectName) {
        const typeName = parser.parseKeyword();
        if (!typeName) {
            return null;
        }
        let type = `!${dialectName}.${typeName}`;
        if (this.simpleTypes.has(typeName)) {
            if (parser.match('<')) {
                const content = parser.skip('<');
                type += content;
            }
            return new mlir.Type(type);
        }
        return null;
    }

    _parseOperandTypeList(parser, op /*, args */) {
        parser.expect('(');
        if (!parser.match(')')) {
            let index = 0;
            do {
                const type = parser.parseType();
                if (index < op.operands.length) {
                    op.operands[index].type = type;
                }
                index++;
            } while (parser.accept(','));
        }
        parser.expect(')');
    }

    _parseTiedFunctionResultList(parser, op /*, args */) {
        const parseTiedResultOrType = () => {
            if (parser.match('%')) {
                const tiedRef = parser.expect('%');
                let tiedType = null;
                for (let i = 0; i < op.operands.length; i++) {
                    if (op.operands[i].value === tiedRef) {
                        tiedType = op.operands[i].type;
                        break;
                    }
                }
                if (parser.accept('id', 'as')) {
                    return parser.parseType();
                }
                if (tiedType) {
                    return tiedType;
                }
                return new mlir.Type('!util.unknown');
            }
            return parser.parseType();
        };
        if (parser.accept('(')) {
            let index = 0;
            if (!parser.match(')')) {
                do {
                    const type = parseTiedResultOrType();
                    if (index < op.results.length) {
                        op.results[index].type = type;
                    } else {
                        op.results.push({ type });
                    }
                    index++;
                } while (parser.accept(','));
            }
            parser.expect(')');
        } else {
            let index = 0;
            do {
                const type = parseTiedResultOrType();
                if (index < op.results.length) {
                    op.results[index].type = type;
                } else {
                    op.results.push({ type });
                }
                index++;
            } while (parser.accept(','));
        }
    }

    parseOperation(parser, opName, op) {
        if (opName === 'util.assume.int') {
            return this._parseAssumeIntOp(parser, op);
        }
        if (opName === 'util.initializer') {
            if (parser.accept('id', 'attributes')) {
                parser.parseAttributeDict(op.attributes);
            }
            if (parser.match('{')) {
                const region = op.addRegion();
                parser.parseRegion(region);
            }
            return true;
        }
        if (opName === 'util.unreachable') {
            this.getOperation(opName).hasParseOperation = false; // compatibility?
            if (parser.match('string')) {
                const message = parser.expect('string');
                op.addAttribute('message', message);
            }
            if (parser.match('{')) {
                parser.parseAttributeDict(op.attributes);
            }
            return true;
        }
        if (opName === 'util.func') {
            this._parseUtilFuncOp(parser, op);
            return true;
        }
        if (opName === 'util.unfoldable_constant') {
            if (parser.match('{')) {
                parser.parseAttributeDict(op.attributes);
            }
            const value = parser.parseAttribute();
            op.addAttribute('value', value);
            if (parser.accept(':')) {
                const type = parser.parseType();
                op.results.push({ type });
            }
            return true;
        }
        return super.parseOperation(parser, opName, op);
    }

    _parseUtilFuncOp(parser, op) {
        parser.parseOptionalVisibilityKeyword(op.attributes);
        parser.parseSymbolName('sym_name', op.attributes);
        const argResult = parser.parseFunctionArgumentList(false);
        const resultTypes = [];
        const resultAttrs = [];
        const tiedOperandIndices = [];
        // Reference: UtilOps.cpp parseTiedFunctionResultList
        // Parse result list which may contain:
        // - Regular type: tensor<...>
        // - Tied reference: %arg1 (inherits type from argument)
        // - Tied with type override: %arg2 as tensor<...>
        const parseTiedResultOrType = () => {
            if (parser.match('%')) {
                const tiedRef = parser.expect('%');
                let tiedIndex = -1;
                for (let i = 0; i < argResult.arguments.length; i++) {
                    if (argResult.arguments[i].value === tiedRef) {
                        tiedIndex = i;
                        break;
                    }
                }
                tiedOperandIndices.push(tiedIndex);
                // Check for 'as type' override
                if (parser.accept('id', 'as')) {
                    return parser.parseType();
                }
                if (tiedIndex >= 0 && argResult.arguments[tiedIndex].type) {
                    return argResult.arguments[tiedIndex].type;
                }
                return new mlir.Type('!util.unknown');
            }
            tiedOperandIndices.push(-1);
            return parser.parseType();
        };
        if (parser.accept('->')) {
            if (parser.accept('(')) {
                if (!parser.match(')')) {
                    do {
                        resultTypes.push(parseTiedResultOrType());
                        if (parser.match('{')) {
                            const attrList = new Map();
                            parser.parseAttributeDict(attrList);
                            resultAttrs.push(attrList);
                        } else {
                            resultAttrs.push(null);
                        }
                    } while (parser.accept(','));
                }
                parser.expect(')');
            } else {
                do {
                    resultTypes.push(parseTiedResultOrType());
                    resultAttrs.push(null);
                } while (parser.accept(','));
            }
        }
        const argTypes = argResult.arguments.filter((a) => a.value !== '...').map((a) => a.type);
        const type = new mlir.FunctionType(argTypes, resultTypes);
        op.addAttribute('function_type', new mlir.TypeAttrOf(type));
        if (tiedOperandIndices.some((i) => i >= 0)) {
            op.addAttribute('tied_operands', tiedOperandIndices);
        }
        if (resultAttrs.some((a) => a !== null)) {
            op.addAttribute('res_attrs', resultAttrs);
        }
        const argAttrs = argResult.arguments.filter((a) => a.value !== '...').map((a) => a.attributes || null);
        if (argAttrs.some((a) => a !== null)) {
            op.addAttribute('arg_attrs', argAttrs);
        }
        parser.parseOptionalAttrDictWithKeyword(op.attributes);
        if (parser.match('{')) {
            const region = op.addRegion();
            parser.parseRegion(region, argResult.arguments);
        }
    }

    _parseAssumeIntOp(parser, op) {
        const allOperandAssumptions = [];
        const parsedOperands = [];
        const parsedOperandTypes = [];

        do {
            const operand = parser.parseAttribute();
            parsedOperands.push(operand);
            const operandAssumptions = [];
            if (parser.accept('[')) {
                if (!parser.match(']')) {
                    do {
                        const assumption = this._parseIntAssumptionAttr(parser);
                        operandAssumptions.push(assumption);
                    } while (parser.accept(','));
                }
                parser.expect(']');
            } else if (parser.match('<')) {
                const assumption = this._parseIntAssumptionAttr(parser);
                operandAssumptions.push(assumption);
            }
            allOperandAssumptions.push(operandAssumptions);
        } while (parser.accept(','));
        parser.expect(':');
        do {
            const type = parser.parseType();
            parsedOperandTypes.push(type);
        } while (parser.accept(','));
        for (let i = 0; i < parsedOperands.length; i++) {
            const operand = parsedOperands[i];
            if (parsedOperandTypes[i]) {
                operand.type = parsedOperandTypes[i];
            }
            op.operands.push(operand);
            op.results.push({ type: parsedOperandTypes[i] || null });
        }

        op.addAttribute('assumptions', allOperandAssumptions);

        if (parser.match('{')) {
            parser.parseAttributeDict(op.attributes);
        }

        return true;
    }

    _parseIntAssumptionAttr(parser) {
        parser.expect('<');
        const assumption = {};
        if (!parser.match('>')) {
            do {
                const key = parser.expect('id');
                if (!parser.accept('=')) {
                    throw new mlir.Error(`Expected '=' after ${key} ${parser.location()}`);
                }
                const value = parser.expect('int');
                assumption[key] = value;
            } while (parser.accept(','));
        }
        parser.expect('>');
        return assumption;
    }

    _parseRangeList(parser, op, args) {
        const offsets = [];
        const lengths = [];
        do {
            parser.expect('[');
            const offset = parser.expect('%');
            offsets.push({ value: offset });
            parser.expect('id', 'for');
            const length = parser.expect('%');
            lengths.push({ value: length });
            parser.expect(']');
        } while (parser.accept(','));
        for (const offset of offsets) {
            op.operands.push(offset);
        }
        for (const length of lengths) {
            op.operands.push(length);
        }
        if (args && args.length > 0) {
            const offsetsArg = args[0].replace(/^\$/, '');
            op.addAttribute(`${offsetsArg}_count`, offsets.length);
        }
    }

    _parseListTypeGet(parser, op, args) {
        const listType = parser.parseType();
        let elementType = null;
        if (parser.accept('->')) {
            elementType = parser.parseType();
        } else if (listType && listType.value) {
            const match = listType.value.match(/!util\.list<(.+)>/);
            if (match) {
                elementType = new mlir.Type(match[1]);
            }
        }
        if (args && args.length > 0) {
            const listTypeArg = args[0].replace(/^\$/, '');
            op.addAttribute(listTypeArg, listType);
        }
        if (args && args.length > 1 && elementType) {
            const elementTypeArg = args[1].replace(/^\$/, '');
            op.addAttribute(elementTypeArg, elementType);
        }
        if (elementType) {
            op.results.push({ type: elementType });
        }
    }

    _parseListTypeSet(parser, op, args) {
        const leadingType = parser.parseType();
        let listType = null;
        let elementType = null;
        if (parser.accept('->')) {
            elementType = leadingType;
            listType = parser.parseType();
        } else if (leadingType && leadingType.value && leadingType.value.includes('!util.list<')) {
            listType = leadingType;
            const match = leadingType.value.match(/!util\.list<(.+)>/);
            if (match) {
                elementType = new mlir.Type(match[1]);
            }
        }
        if (args && args.length > 0 && listType) {
            const listTypeArg = args[0].replace(/^\$/, '');
            op.addAttribute(listTypeArg, listType);
        }
        if (args && args.length > 1 && elementType) {
            const elementTypeArg = args[1].replace(/^\$/, '');
            op.addAttribute(elementTypeArg, elementType);
        }
    }

    _parseValueTypeList(parser, op) {
        parser.expect('[');
        if (!parser.match(']')) {
            do {
                const value = parser.expect('%');
                parser.expect(':');
                const type = parser.parseType();
                op.operands.push({ value, type });
            } while (parser.accept(','));
        }
        parser.expect(']');
    }
};

mlir.FlowDialect = class extends mlir.IREEDialect {

    constructor(operations) {
        super(operations, 'flow');
        this.registerCustomDirective('DispatchWorkgroupBody', this._parseDispatchWorkgroupBody.bind(this));
        this.registerCustomDirective('DispatchWorkgroupsCountRegion', this._parseDispatchWorkgroupsCountRegion.bind(this));
        this.registerCustomDirective('ShapedFunctionType', this._parseShapedFunctionType.bind(this));
        this.registerCustomDirective('ShapedOperandList', this._parseShapedOperandList.bind(this));
    }

    parseType(parser, dialectName) {
        const typeName = parser.parseKeyword();
        if (!typeName) {
            return null;
        }
        let type = `!${dialectName}.${typeName}`;
        if (typeName === 'channel') {
            return new mlir.Type(type);
        }
        if (typeName === 'dispatch.tensor') {
            if (parser.match('<')) {
                const content = parser.skip('<');
                type += content;
            }
            return new mlir.Type(type);
        }
        return null;
    }

    parseOperation(parser, opName, op) {
        if (opName === 'flow.ex.stream.fragment') {
            return this._parseDispatchWorkgroupsOp(parser, op);
        }
        if (opName === 'flow.dispatch.region') {
            return this._parseDispatchRegionOp(parser, op);
        }
        if (opName === 'flow.dispatch.tensor.load' || opName === 'flow.dispatch.tensor.store') {
            return this._parseTensorLoadStoreOp(parser, op);
        }
        // Handle operations with visibility + symbol that aren't in schema or need manual parsing
        if (opName === 'flow.dispatch.entry') {
            if (parser.match('id', 'private') || parser.match('id', 'public') || parser.match('id', 'nested')) {
                parser.expect('id');
            }
            if (parser.match('@')) {
                parser.parseSymbolName('sym_name', op.attributes);
            }
            if (parser.accept('id', 'attributes')) {
                parser.parseAttributeDict(op.attributes);
            }
            if (parser.match('{')) {
                const region = op.addRegion();
                parser.parseRegion(region);
            }
            return true;
        }
        if (opName === 'flow.func') {
            return this._parseFlowFuncOp(parser, op);
        }
        return super.parseOperation(parser, opName, op);
    }

    // Reference: UtilOps.cpp parseShapedFunctionSignature
    _parseFlowFuncOp(parser, op) {
        parser.parseOptionalVisibilityKeyword(op.attributes);
        parser.parseSymbolName('sym_name', op.attributes);
        const argResult = parser.parseFunctionArgumentList();
        const inputs = argResult.arguments.map((a) => a.type);
        const results = [];
        if (parser.accept('->')) {
            // Parse shaped function result list with tied operand support
            const hasParens = parser.accept('(');
            if (!hasParens || !parser.match(')')) {
                do {
                    // Reference: UtilOps.cpp parseShapedFunctionResultList
                    // Try to parse tied operand: %arg0 or %arg0 as type
                    if (parser.match('%')) {
                        parser.expect('%');
                        if (parser.accept('id', 'as')) {
                            const resultType = parser.parseType();
                            results.push(resultType);
                        } else {
                            results.push(new mlir.Type('tied'));
                        }
                    } else {
                        const resultType = parser.parseType();
                        results.push(resultType);
                    }
                    if (parser.match('{')) {
                        parser.skip('{');
                    }
                    if (!hasParens) {
                        break;
                    }
                } while (parser.accept(','));
            }
            if (hasParens) {
                parser.expect(')');
            }
        }
        op.addAttribute('function_type', new mlir.TypeAttrOf(new mlir.FunctionType(inputs, results)));
        parser.parseOptionalAttrDictWithKeyword(op.attributes);
        if (parser.match('{')) {
            const region = op.addRegion();
            parser.parseRegion(region);
        }
        return true;
    }

    _parseDispatchRegionOp(parser, op) {
        // Reference: FlowOps.cpp:426 uses parseOperandList
        const workloadOperands = parser.parseOperandList('optionalSquare');
        op.operands.push(...workloadOperands);
        if (parser.accept('->')) {
            if (parser.accept('(')) {
                while (!parser.accept(')')) {
                    const result = {};
                    result.type = parser.parseType();
                    if (parser.accept('{')) {
                        while (!parser.accept('}')) {
                            const operand = {};
                            operand.value = parser.expect('%');
                            op.operands.push(operand);
                            parser.accept(',');
                        }
                    }
                    op.results.push(result);
                    parser.accept(',');
                }
            }
        }
        parser.parseOptionalAttrDictWithKeyword(op.attributes);
        if (parser.match('{')) {
            const region = op.addRegion();
            parser.parseRegion(region);
        }
        // Parse optional count region
        this._parseDispatchWorkgroupsCountRegion(parser, op);
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
        op.operands = parser.parseArguments();
        // Reference: parseOptionalColonTypeList
        parser.resolveOperands(op.operands, parser.parseOptionalColonTypeList());
        // Reference: parseOptionalArrowTypeList for results
        if (parser.accept('->')) {
            if (op.results.length > 0) {
                const types = parser.parseFunctionResultTypes();
                parser.resolveOperands(op.results, types);
            } else {
                op.results = parser.parseArguments();
            }
        }
        // Parse optional attributes before =
        if (parser.accept('id', 'attributes')) {
            parser.parseAttributeDict(op.attributes);
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

    _parseShapedFunctionType(parser, op /*, args */) {
        // Parse: (operand_types) -> result_types
        // Example: (tensor<?x?xf32>{%dim0, %dim1}, tensor<4xf32>) -> tensor<?xf32>
        if (parser.accept('(')) {
            let index = 0;
            if (!parser.match(')')) {
                do {
                    const type = parser.parseType();
                    if (type) {
                        const startIdx = Math.max(0, op.operands.length - (index + 1));
                        if (startIdx + index < op.operands.length && !op.operands[startIdx + index].type) {
                            op.operands[startIdx + index].type = type;
                        }
                        index++;
                    }
                    if (parser.accept('{')) {
                        while (!parser.accept('}')) {
                            parser.expect('%');
                            parser.accept(',');
                        }
                    }
                } while (parser.accept(','));
            }
            parser.expect(')');
        }

        // Parse arrow and result types
        // Reference: UtilOps.cpp parseShapedResultList
        if (parser.accept('->')) {
            let index = 0;
            const hasParens = parser.accept('(');
            if (!parser.match(')') && !parser.match('{') && !parser.match('loc') && !parser.match('=')) {
                do {
                    if (parser.match('%')) {
                        parser.expect('%');
                        // Handle optional "as type" for tied results
                        if (parser.accept('id', 'as')) {
                            const type = parser.parseType();
                            if (type) {
                                if (index < op.results.length) {
                                    op.results[index].type = type;
                                } else {
                                    op.results.push({ type });
                                }
                            }
                        }
                        index++;
                    } else {
                        const type = parser.parseType();
                        if (type) {
                            if (index < op.results.length) {
                                op.results[index].type = type;
                            } else {
                                op.results.push({ type });
                            }
                            index++;
                        }
                    }
                    if (parser.accept('{')) {
                        while (!parser.accept('}')) {
                            parser.expect('%');
                            parser.accept(',');
                        }
                    }
                    if (!hasParens) {
                        break;
                    }
                } while (parser.accept(','));
            }
            if (hasParens) {
                parser.expect(')');
            }
        }
    }

    _parseTensorLoadStoreOp(parser, op) {
        // Parse: load %arg2, offsets = [...] : type -> type
        //    or: store %26, %arg4, offsets = [...] : type -> type
        while (parser.match('%')) {
            const value = parser.expect('%');
            op.operands.push({ value });
            if (!parser.accept(',')) {
                break;
            }
            if (!parser.match('%')) {
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
                    if (parser.match('[')) {
                        parser.skip('[');
                    } else {
                        parser.expect();
                    }
                    op.addAttribute(paramName, paramName);
                }
            } else {
                break;
            }
        }
        // Reference: parseOptionalColonTypeList
        parser.resolveOperands(op.operands, parser.parseOptionalColonTypeList());
        // For tensor.load, there's a -> result type
        // For tensor.store, the -> is followed by the output tensor type (not a result)
        if (parser.accept('->') || parser.accept('id', 'to')) {
            parser.parseType();
        }
        return true;
    }

    _parseDispatchWorkgroupBody(parser, op /*, args */) {
        parser.expect('(');
        const regionArgs = [];
        if (!parser.match(')')) {
            do {
                const arg = parser.expect('%');
                parser.expect(':');
                const argType = parser.parseType();
                regionArgs.push({ name: arg, type: argType });
            } while (parser.accept(','));
        }
        parser.expect(')');
        const region = { blocks: [{ arguments: regionArgs, operations: [] }] };
        parser.parseRegion(region);
        op.regions.push(region);
    }

    _parseDispatchWorkgroupsCountRegion(parser, op /*, args */) {
        if (!parser.accept('id', 'count')) {
            return;
        }
        parser.expect('(');
        const regionArgs = [];
        if (!parser.match(')')) {
            do {
                const arg = parser.expect('%');
                parser.expect(':');
                const argType = parser.parseType();
                regionArgs.push({ name: arg, type: argType });
            } while (parser.accept(','));
        }
        parser.expect(')');
        parser.expect('->');
        if (parser.accept('(')) {
            parser.parseType();
            parser.accept(',');
            parser.parseType();
            parser.accept(',');
            parser.parseType();
            parser.expect(')');
        } else {
            parser.parseType();
            parser.accept(',');
            parser.parseType();
            parser.accept(',');
            parser.parseType();
        }
        const region = { blocks: [{ arguments: regionArgs, operations: [] }] };
        parser.parseRegion(region);
        op.regions.push(region);
    }

    _parseShapedOperandList(parser, op) {
        const values = [];
        const valueTypes = [];
        const valueDims = [];
        do {
            const value = parser.expect('%');
            values.push({ value });
            parser.expect(':');
            const valueType = parser.parseType();
            valueTypes.push(valueType);
            if (valueType) {
                const typeStr = valueType.toString();
                const dynamicDimCount = (typeStr.match(/\?/g) || []).length;
                if (dynamicDimCount > 0 && parser.accept('{')) {
                    for (let i = 0; i < dynamicDimCount; i++) {
                        if (i > 0) {
                            parser.accept(',');
                        }
                        const dim = parser.expect('%');
                        valueDims.push({ value: dim });
                    }
                    parser.expect('}');
                }
            }
        } while (parser.accept(','));
        for (const value of values) {
            op.operands.push(value);
        }
        for (const dim of valueDims) {
            op.operands.push(dim);
        }
        for (let i = 0; i < values.length && i < op.operands.length; i++) {
            op.operands[i].type = valueTypes[i];
        }
    }
};

mlir.StreamDialect = class extends mlir.IREEDialect {

    constructor(operations, name = 'stream') {
        super(operations, name);
        this.registerCustomDirective('DispatchOperands', this._parseDispatchOperands.bind(this));
        this.registerCustomDirective('DispatchResources', this._parseDispatchResources.bind(this));
        this.registerCustomDirective('ExplicitResourceRegion', this._parseExplicitResourceRegion.bind(this));
        this.registerCustomDirective('ShapedTypeList', this._parseShapedTypeList.bind(this));
        this.registerCustomDirective('ResourceRegion', this._parseResourceRegion.bind(this));
        this.registerCustomDirective('ParameterLoadOperations', this._parseParameterLoadOperations.bind(this));
        this.registerCustomDirective('EncodedResourceOperands', this._parseEncodedResourceOperands.bind(this));
        this.registerCustomDirective('DispatchEntryPoints', this._parseDispatchEntryPoints.bind(this));
        this.registerCustomDirective('ShapedTiedResult', this._parseShapedTiedResult.bind(this));
        this.registerCustomDirective('SymbolVisibility', this._parseSymbolVisibility.bind(this));
        this.registerCustomDirective('EncodedShapedFunctionType', this._parseEncodedShapedFunctionType.bind(this));
        this.registerCustomDirective('CollectiveParam', this._parseCollectiveParam.bind(this));
        this.registerCustomDirective('PackSliceRanges', this._parsePackSliceRanges.bind(this));
        this.registerCustomDirective('WorkgroupCountRegion', this._parseWorkgroupCountRegion.bind(this));
        this.registerCustomDirective('DispatchFunctionSignature', this._parseDispatchFunctionSignature.bind(this));
        this.registerCustomDirective('ShapedFunctionSignature', this._parseShapedFunctionSignature.bind(this));
        this.registerCustomDirective('ConstantValueList', this._parseConstantValueList.bind(this));
        this.registerCustomDirective('CmdCallOperands', this._parseCmdCallOperands.bind(this));
        this.registerCustomDirective('ParameterReference', this._parseParameterReference.bind(this));
        this.registerCustomDirective('ParameterGatherOperations', this._parseParameterGatherOperations.bind(this));
        this.registerCustomDirective('ParameterScatterOperations', this._parseParameterScatterOperations.bind(this));
        this.registerCustomDirective('SymbolAlias', this._parseSymbolAlias.bind(this));
    }

    _parseDispatchResources(parser, op /*, args */) {
        do {
            const accessMode = parser.expect('id');
            const resource = parser.expect('%');
            op.operands.push({ value: resource });
            parser.expect('[');
            const offset = parser.expect('%');
            op.operands.push({ value: offset });
            parser.expect('id', 'for');
            const length = parser.expect('%');
            op.operands.push({ value: length });
            parser.expect(']');
            parser.expect(':');
            parser.parseType();
            if (parser.match('{')) {
                parser.skip('{');
            }
            op.addAttribute('resource_access', accessMode);
        } while (parser.accept(','));
    }

    _parseShapedTypeList(parser /*, op, args */) {
        do {
            parser.parseType();
            if (parser.match('{')) {
                parser.skip('{');
            }
        } while (parser.accept(','));
    }

    _parseExplicitResourceRegion(parser, op /*, args */) {
        parser.expect('(');
        const regionArgs = [];
        if (!parser.match(')')) {
            do {
                const operand = parser.parseAttribute();
                if (operand) {
                    op.operands.push(operand);
                }
                parser.expect('id', 'as');
                const arg = parser.expect('%');
                parser.expect(':');
                const argType = parser.parseType();
                regionArgs.push({ name: arg, type: argType });
                if (parser.accept('{')) {
                    const size = parser.parseAttribute();
                    if (size) {
                        op.operands.push(size);
                    }
                    parser.expect('}');
                }
            } while (parser.accept(','));
        }
        parser.expect(')');
        const region = { blocks: [{ arguments: regionArgs, operations: [] }] };
        parser.parseRegion(region);
        op.regions.push(region);
    }

    _parseResourceRegion(parser, op /*, args */) {
        // Reference: StreamOps.cpp parseResourceRegion
        // Format: (operand as arg: type{size}, ...) -> (result_type{size}, ...) { region }
        const regionArgs = [];
        parser.expect('(');
        if (!parser.match(')')) {
            do {
                const operand = parser.expect('%');
                op.operands.push({ value: operand });
                parser.expect('id', 'as');
                const arg = parser.expect('%');
                parser.expect(':');
                const argType = parser.parseType();
                regionArgs.push({ name: arg, type: argType });
                if (parser.accept('{')) {
                    const size = parser.parseAttribute();
                    if (size) {
                        op.operands.push(size);
                    }
                    parser.expect('}');
                }
            } while (parser.accept(','));
        }
        parser.expect(')');
        const parseResultTypeOrTied = () => {
            if (parser.match('%')) {
                parser.expect('%');
                if (parser.accept('id', 'as')) {
                    const resultType = parser.parseType();
                    op.results.push({ type: resultType });
                } else {
                    op.results.push({ type: new mlir.Type('tied') });
                }
            } else {
                const resultType = parser.parseType();
                op.results.push({ type: resultType });
            }
            if (parser.accept('{')) {
                const size = parser.parseAttribute();
                if (size) {
                    op.operands.push(size);
                }
                parser.expect('}');
            }
        };
        if (parser.accept('->')) {
            if (parser.accept('(')) {
                if (!parser.match(')')) {
                    do {
                        parseResultTypeOrTied();
                    } while (parser.accept(','));
                }
                parser.expect(')');
            } else {
                parseResultTypeOrTied();
            }
        }
        if (parser.match('{')) {
            const region = { blocks: [{ arguments: regionArgs, operations: [] }] };
            parser.parseRegion(region);
            op.regions.push(region);
        }
    }

    _parseParameterLoadOperations(parser, op /*, args */) {
        // Reference: IOParametersOps.cpp parseParameterLoadOperations
        // Format: "scope"::"key"[%offset] : type{%size}, ...
        do {
            // Parse parameter reference: "scope"::"key" or just "key"
            const firstAttr = parser.expect('string');
            // Check for :: (scope::key) pattern - the lexer produces '::' as a single token
            if (parser.accept('id', '::') || parser.accept('::')) {
                const keyAttr = parser.expect('string');
                op.addAttribute('source_scope', firstAttr);
                op.addAttribute('source_key', keyAttr);
            } else {
                op.addAttribute('source_key', firstAttr);
            }
            parser.expect('[');
            const offset = parser.expect('%');
            op.operands.push({ name: 'source_offset', value: offset });
            parser.expect(']');
            parser.expect(':');
            const resultType = parser.parseType();
            op.results.push({ type: resultType });
            if (parser.accept('{')) {
                const size = parser.expect('%');
                op.operands.push({ name: 'result_size', value: size });
                parser.expect('}');
            }
        } while (parser.accept(','));
    }

    // Reference: StreamOps.cpp parseEncodedResourceOperands
    // Format: %operand : encoding_type{%dims} in resource_type{%size}
    _parseEncodedResourceOperands(parser /*, op, args */) {
        do {
            parser.expect('%');
            parser.expect(':');
            parser.parseType();
            parser.skip('{');
            parser.expect('id', 'in');
            parser.parseType();
            parser.skip('{');
        } while (parser.accept(','));
    }

    // Parse: @entry_point or {@entry_point1, @entry_point2}
    // Reference: StreamOps.cpp parseDispatchEntryPoints
    _parseDispatchEntryPoints(parser, op /*, args */) {
        if (parser.accept('{')) {
            do {
                const symbol = parser.expect('@');
                op.addAttribute('entry_point', symbol);
            } while (parser.accept(','));
            parser.expect('}');
        } else {
            const symbol = parser.expect('@');
            op.addAttribute('entry_point', symbol);
        }
    }

    _parseShapedTiedResult(parser, op /*, args */) {
        let tiedOperand = null;
        if (parser.match('%')) {
            tiedOperand = parser.expect('%');
            parser.expect('id', 'as');
        }
        const type = parser.parseType();
        op.results.push({ type, tiedOperand });
        if (parser.accept('{')) {
            const size = parser.parseAttribute();
            if (size) {
                op.operands.push(size);
            }
            parser.expect('}');
        }
    }

    _parseSymbolVisibility(parser, op /*, args */) {
        if (parser.accept('id', 'public')) {
            op.addAttribute('sym_visibility', 'public');
        } else if (parser.accept('id', 'private')) {
            op.addAttribute('sym_visibility', 'private');
        } else if (parser.accept('id', 'nested')) {
            op.addAttribute('sym_visibility', 'nested');
        }
    }

    // Parse: (types) -> (types) with sizes in {} and optional encoding type
    // Reference: StreamOps.cpp parseEncodedShapedFunctionType
    // Format: (encoding{%dims} in type{%size}, type) -> (encoding{%dims} in %operand{%size})
    _parseEncodedShapedFunctionType(parser /*, op, args */) {
        const parseEncodedType = () => {
            // Parse type or encoding type
            parser.parseType();
            parser.skip('{');
            // Check for optional 'in' keyword (encoding in resource_type)
            if (parser.accept('id', 'in')) {
                // Parse resource type or tied operand
                if (parser.match('%')) {
                    parser.expect('%');
                } else {
                    parser.parseType();
                }
                parser.skip('{');
            }
        };
        parser.expect('(');
        if (!parser.match(')')) {
            do {
                parseEncodedType();
            } while (parser.accept(','));
        }
        parser.expect(')');
        parser.expect('->');
        if (parser.accept('(')) {
            if (!parser.match(')')) {
                do {
                    parseEncodedType();
                } while (parser.accept(','));
            }
            parser.expect(')');
        } else {
            parseEncodedType();
        }
    }

    // Parse: collective operation parameters
    // Reference: StreamOps.cpp parseCollectiveParam
    // The param is optional and depends on the collective operation type (op attribute).
    // For all_gather/all_reduce: no param needed (returns without parsing)
    // For broadcast/reduce/send/recv/send_recv: parses keyword(operand) syntax
    // Since we can't easily determine op type here, we parse optional keyword(operand) pattern
    _parseCollectiveParam(parser, op /*, args */) {
        // Check for keyword(operand) pattern: source(%val), target(%val), or source_target_pair(%val)
        const keywords = ['source', 'target', 'source_target_pair'];
        for (const keyword of keywords) {
            if (parser.match('id', keyword)) {
                parser.expect('id', keyword);
                parser.expect('(');
                const param = parser.expect('%');
                op.operands.push({ name: 'param', value: param });
                parser.expect(')');
                return;
            }
        }
    }

    _parsePackSliceRanges(parser, op /*, args */) {
        while (parser.accept('[')) {
            parser.parseAttribute();
            parser.expect(',');
            parser.parseAttribute();
            parser.expect(']');
            parser.expect('=');
            const operand = parser.expect('%');
            op.operands.push({ value: operand });
            if (!parser.accept(',')) {
                break;
            }
        }
    }

    // Parse: workgroups(%x: type, %y: type, %z: type) -> (index, index, index) { ... }
    _parseWorkgroupCountRegion(parser, op /*, args */) {
        if (!parser.accept('id', 'workgroups')) {
            return;
        }
        const region = { blocks: [] };
        const block = { arguments: [], operations: [] };
        if (parser.accept('(')) {
            while (!parser.match(')')) {
                const arg = { value: parser.expect('%') };
                if (parser.accept(':')) {
                    arg.type = parser.parseType();
                }
                block.arguments.push(arg);
                if (!parser.accept(',')) {
                    break;
                }
            }
            parser.expect(')');
        }
        // Parse optional return type -> (types)
        if (parser.accept('->')) {
            parser.expect('(');
            while (!parser.match(')')) {
                parser.parseType();
                if (!parser.accept(',')) {
                    break;
                }
            }
            parser.expect(')');
        }
        region.blocks.push(block);
        if (parser.match('{')) {
            parser.parseRegion(region);
        }
        op.regions.push(region);
    }

    // Parse: dispatch function signature
    // Reference: StreamOps.cpp parseDispatchFunctionSignature
    _parseDispatchFunctionSignature(parser, op /*, args */) {
        const inputs = [];
        const results = [];
        parser.expect('(');
        if (!parser.match(')')) {
            do {
                parser.expect('%');
                // skip('[', ']') already handles checking for '[' presence
                parser.skip('[');
                parser.expect(':');
                const type = parser.parseType();
                inputs.push(type);
                parser.skip('{');
            } while (parser.accept(','));
        }
        parser.expect(')');
        const parseResultTypeOrTied = () => {
            if (parser.match('%')) {
                parser.expect('%');
                if (parser.accept('id', 'as')) {
                    return parser.parseType();
                }
                return new mlir.Type('tied');
            }
            return parser.parseType();
        };
        if (parser.accept('->')) {
            if (parser.accept('(')) {
                if (!parser.match(')')) {
                    do {
                        results.push(parseResultTypeOrTied());
                        parser.skip('{');
                    } while (parser.accept(','));
                }
                parser.expect(')');
            } else {
                results.push(parseResultTypeOrTied());
                parser.skip('{');
            }
        }
        op.addAttribute('function_type', new mlir.TypeAttrOf(new mlir.FunctionType(inputs, results)));
    }

    // Parse: shaped function signature
    _parseShapedFunctionSignature(parser, op /*, args */) {
        this._parseDispatchFunctionSignature(parser, op);
    }

    _parseConstantValueList(parser, op /*, args */) {
        do {
            const resultType = parser.parseType();
            op.results.push({ type: resultType });
            if (parser.accept('{')) {
                const size = parser.parseAttribute();
                if (size) {
                    op.operands.push(size);
                }
                parser.expect('}');
            }
            parser.expect('=');
            parser.parseAttribute();
            if (parser.accept(':')) {
                parser.parseType();
            }
        } while (parser.accept(','));
    }

    // Parse: cmd call operands
    // Reference: StreamOps.cpp parseCmdCallOperands
    _parseCmdCallOperands(parser, op /*, args */) {
        parser.expect('(');
        if (!parser.match(')')) {
            do {
                // Check for access mode keyword (ro, rw, wo)
                const accessMode = parser.accept('id', 'ro') || parser.accept('id', 'rw') || parser.accept('id', 'wo');
                if (accessMode) {
                    // Resource operand with offset/length: access operand[offset for length]
                    op.operands.push({ value: parser.expect('%') });
                    parser.expect('[');
                    op.operands.push({ value: parser.expect('%') });
                    parser.expect('id', 'for');
                    op.operands.push({ value: parser.expect('%') });
                    parser.expect(']');
                    op.addAttribute('resource_access', accessMode);
                } else {
                    // Primitive/custom operand
                    op.operands.push({ value: parser.expect('%') });
                }
            } while (parser.accept(','));
        }
        parser.expect(')');
    }

    // Parse: "scope"::"key" or "key"
    // Reference: IOParametersOps.cpp parseParameterReference
    _parseParameterReference(parser /*, op, args */) {
        parser.expect('string');
        if (parser.accept('::')) {
            parser.expect('string');
        }
    }

    // Parse: parameter gather operations
    _parseParameterGatherOperations(parser /*, op, args */) {
        do {
            // "scope"::"key"[offset] -> %target[offset for length] : type{size}
            this._parseParameterReference(parser);
            if (parser.accept('[')) {
                parser.parseAttribute();
                parser.expect(']');
            }
            parser.expect('->');
            parser.expect('%');
            if (parser.accept('[')) {
                parser.parseAttribute();
                parser.expect('id', 'for');
                parser.parseAttribute();
                parser.expect(']');
            }
            parser.expect(':');
            parser.parseType();
            if (parser.match('{')) {
                parser.skip('{');
            }
        } while (parser.accept(','));
    }

    // Parse: parameter scatter operations
    _parseParameterScatterOperations(parser /*, op, args */) {
        do {
            // %source[offset for length] : type{size} -> "scope"::"key"[offset]
            parser.expect('%');
            if (parser.accept('[')) {
                parser.parseAttribute();
                parser.expect('id', 'for');
                parser.parseAttribute();
                parser.expect(']');
            }
            parser.expect(':');
            parser.parseType();
            if (parser.match('{')) {
                parser.skip('{');
            }
            parser.expect('->');
            this._parseParameterReference(parser);
            if (parser.accept('[')) {
                parser.parseAttribute();
                parser.expect(']');
            }
        } while (parser.accept(','));
    }

    // Parse: symbol alias @name = @ref
    _parseSymbolAlias(parser, op /*, args */) {
        parser.parseSymbolName('sym_name', op.attributes);
        if (parser.accept('=')) {
            const ref = parser.expect('@');
            op.addAttribute('function_ref', ref);
        }
    }

    parseType(parser, dialectName) {
        const typeName = parser.parseKeyword();
        if (!typeName) {
            return null;
        }
        let type = `!${dialectName}.${typeName}`;
        const simpleTypes = ['binding', 'channel', 'timepoint', 'file'];
        if (simpleTypes.includes(typeName)) {
            return new mlir.Type(type);
        }
        if (typeName === 'resource') {
            if (parser.match('<')) {
                const content = parser.skip('<');
                type += content;
            }
            return new mlir.Type(type);
        }
        // Handle test.fence type (Stream_TestFence in StreamTypes.td)
        // Reference: mnemonic = "test.fence" means the type name after stream. is "test.fence"
        if (typeName === 'test') {
            if (parser.accept('.')) {
                const subtype = parser.parseKeyword();
                if (subtype === 'fence') {
                    return new mlir.Type(`!${dialectName}.test.fence`);
                }
                // Handle unknown test.X subtypes generically
                return new mlir.Type(`!${dialectName}.test.${subtype}`);
            }
            // Just "test" without subtype - return as is
            return new mlir.Type(type);
        }
        // Fallback for unknown stream types - parse generically like base Dialect
        if (parser.match('<')) {
            type += parser.skip('<');
        }
        return new mlir.Type(type);
    }

    parseOperation(parser, opName, op) {
        return super.parseOperation(parser, opName, op);
    }

    _parseDispatchOperands(parser, op /*, args */) {
        // Parse: (operand1, operand2[offset to end for length], ...)
        // args would be: [$resource_operands, $resource_operand_offsets, $resource_operand_ends, $resource_operand_lengths]
        parser.expect('(');

        if (parser.match(')')) {
            parser.expect(')');
            return;
        }

        do {
            const operand = parser.parseAttribute();
            if (operand) {
                op.operands.push(operand);
                // Slice notation: [offset to end for length]
                if (parser.accept('[')) {
                    const offset = parser.parseAttribute();
                    if (offset) {
                        op.operands.push(offset);
                    }
                    parser.expect('id', 'to');
                    const end = parser.parseAttribute();
                    if (end) {
                        op.operands.push(end);
                    }
                    parser.expect('id', 'for');
                    const length = parser.parseAttribute();
                    if (length) {
                        op.operands.push(length);
                    }
                    parser.expect(']');
                }
            }
        } while (parser.accept(','));

        parser.expect(')');
    }
};

mlir.IOParametersDialect = class extends mlir.StreamDialect {

    constructor(operations) {
        super(operations, 'io_parameters');
    }
};

mlir.PCFDialect = class extends mlir.Dialect {

    constructor(operations) {
        super(operations, 'pcf');
        this.registerCustomDirective('ParallelExecutionBody', this._parseParallelExecutionBody.bind(this));
        this.registerCustomDirective('InferNumIndexArgs', this._parseInferNumIndexArgs.bind(this));
    }

    _parseInferNumIndexArgs() {
    }

    _parseParallelExecutionBody(parser, op) {
        const inits = [];
        const dynamicSizes = [];
        const resultTypes = [];
        const isTied = [];
        const regionRefArgs = [];
        const indexArgs = [];
        if (parser.accept('->')) {
            parser.expect('(');
            while (!parser.match(')')) {
                const arg = parser.expect('%');
                parser.expect(':');
                const argType = parser.parseType();
                op.addAttribute('num_leading_args', (op.attributes.get('num_leading_args') || 0) + 1);
                regionRefArgs.push({ value: arg, type: argType });
                if (!parser.accept(',')) {
                    break;
                }
            }
            parser.expect(')');
        }
        parser.expect('id', 'execute');
        if (parser.accept('(')) {
            while (!parser.match(')')) {
                const refArg = parser.expect('%');
                regionRefArgs.push({ value: refArg });
                if (parser.accept('=')) {
                    const initOperand = parser.expect('%');
                    inits.push({ value: initOperand });
                    isTied.push(true);
                } else {
                    isTied.push(false);
                }
                if (!parser.accept(',')) {
                    break;
                }
            }
            parser.expect(')');
        }
        parser.expect('[');
        while (!parser.match(']')) {
            const indexArg = parser.expect('%');
            parser.expect(':');
            const indexType = parser.parseType();
            indexArgs.push({ value: indexArg, type: indexType });
            if (!parser.accept(',')) {
                break;
            }
        }
        parser.expect(']');
        if (regionRefArgs.length > 0 && parser.accept(':')) {
            parser.expect('(');
            let refIdx = op.attributes.get('num_leading_args') || 0;
            while (!parser.match(')')) {
                const refType = parser.parseType();
                if (refIdx < regionRefArgs.length) {
                    regionRefArgs[refIdx].type = refType;
                }
                refIdx++;
                if (!parser.accept(',')) {
                    break;
                }
            }
            parser.expect(')');
            parser.expect('->');
            parser.expect('(');
            while (!parser.match(')')) {
                const resType = parser.parseType();
                resultTypes.push(resType);
                op.results.push({ type: resType });
                if (parser.accept('{')) {
                    while (!parser.match('}')) {
                        const dim = parser.expect('%');
                        dynamicSizes.push({ value: dim });
                        if (!parser.accept(',')) {
                            break;
                        }
                    }
                    parser.expect('}');
                }
                if (!parser.accept(',')) {
                    break;
                }
            }
            parser.expect(')');
        }
        for (const init of inits) {
            op.operands.push(init);
        }
        for (const dim of dynamicSizes) {
            op.operands.push(dim);
        }
        if (isTied.length > 0) {
            op.addAttribute('is_tied', isTied);
        }
        const region = { blocks: [{ arguments: [...regionRefArgs, ...indexArgs], operations: [] }] };
        parser.parseRegion(region);
        op.regions.push(region);
    }
};

mlir.IREEVectorExtDialect = class extends mlir.Dialect {

    constructor(operations) {
        super(operations, 'iree_vector_ext');
    }

    parseOperation(parser, opName, op) {
        if (opName === 'iree_vector_ext.transfer_gather') {
            // Format: iree_vector_ext.transfer_gather %source[%indices][index_vec_list], %padding [, %mask] { attr_dict } : source_type, result_type
            // Parse source operand
            const source = parser.expect('%');
            op.operands.push({ value: source });
            parser.expect('[');
            while (!parser.accept(']')) {
                if (parser.match('%')) {
                    const index = parser.expect('%');
                    op.operands.push({ value: index, name: 'index' });
                }
                parser.accept(',');
            }
            // Parse index vectors in [...]
            // Format: [None, %operand: type, None, %operand: type, ...]
            parser.expect('[');
            const indexed = [];
            while (!parser.accept(']')) {
                if (parser.accept('id', 'None')) {
                    indexed.push(false);
                } else if (parser.match('%')) {
                    const indexVec = parser.expect('%');
                    parser.expect(':');
                    const indexVecType = parser.parseType();
                    op.operands.push({ value: indexVec, type: indexVecType, name: 'indexVec' });
                    indexed.push(true);
                }
                parser.accept(',');
            }
            op.addAttribute('indexed', indexed);
            parser.expect(',');
            const padding = parser.parseAttribute();
            op.operands.push({ value: padding, name: 'padding' });
            if (parser.accept(',')) {
                if (parser.match('%')) {
                    const mask = parser.expect('%');
                    op.operands.push({ value: mask, name: 'mask' });
                }
            }
            if (parser.match('{')) {
                parser.parseAttributeDict(op.attributes);
            }
            if (parser.accept(':')) {
                const sourceType = parser.parseType();
                op.operands[0].type = sourceType;
                parser.expect(',');
                const resultType = parser.parseType();
                op.results.push({ type: resultType });
            }
            return true;
        }
        return super.parseOperation(parser, opName, op);
    }
};

mlir.IREETensorExtDialect = class extends mlir.Dialect {

    constructor(operations) {
        super(operations, 'iree_tensor_ext');
    }

    parseType(parser, dialectName) {
        const typeName = parser.parseKeyword();
        if (typeName === 'dispatch.tensor') {
            let type = `!${dialectName}.${typeName}`;
            if (parser.match('<')) {
                const content = parser.skip('<');
                type += content;
            }
            return new mlir.Type(type);
        }
        return null;
    }
};

mlir.LinalgDialect = class extends mlir.Dialect {

    constructor(operations) {
        super(operations, 'linalg');
        this._namedStructuredOps = new Set([
            'linalg.matmul', 'linalg.batch_matmul', 'linalg.batch_reduce_matmul',
            'linalg.matvec', 'linalg.vecmat', 'linalg.dot', 'linalg.batch_matvec',
            'linalg.conv_1d', 'linalg.conv_1d_ncw_fcw', 'linalg.conv_1d_nwc_wcf',
            'linalg.conv_2d', 'linalg.conv_2d_nchw_fchw', 'linalg.conv_2d_nchw_fchw_q',
            'linalg.conv_2d_ngchw_fgchw', 'linalg.conv_2d_ngchw_gfchw', 'linalg.conv_2d_ngchw_gfchw_q',
            'linalg.conv_2d_nhwc_fhwc', 'linalg.conv_2d_nhwc_fhwc_q',
            'linalg.conv_2d_nhwc_hwcf', 'linalg.conv_2d_nhwc_hwcf_q',
            'linalg.conv_2d_nhwgc_gfhwc', 'linalg.conv_2d_nhwgc_gfhwc_q',
            'linalg.conv_3d', 'linalg.conv_3d_ncdhw_fcdhw', 'linalg.conv_3d_ndhwc_dhwcf', 'linalg.conv_3d_ndhwc_dhwcf_q',
            'linalg.depthwise_conv_1d_ncw_cw', 'linalg.depthwise_conv_1d_nwc_wc', 'linalg.depthwise_conv_1d_nwc_wcm',
            'linalg.depthwise_conv_2d_nchw_chw', 'linalg.depthwise_conv_2d_nhwc_hwc', 'linalg.depthwise_conv_2d_nhwc_hwc_q',
            'linalg.depthwise_conv_2d_nhwc_hwcm', 'linalg.depthwise_conv_2d_nhwc_hwcm_q',
            'linalg.depthwise_conv_3d_ncdhw_cdhw', 'linalg.depthwise_conv_3d_ndhwc_dhwc', 'linalg.depthwise_conv_3d_ndhwc_dhwcm',
            'linalg.pooling_nchw_max', 'linalg.pooling_nchw_sum',
            'linalg.pooling_nhwc_max', 'linalg.pooling_nhwc_max_unsigned', 'linalg.pooling_nhwc_min', 'linalg.pooling_nhwc_min_unsigned', 'linalg.pooling_nhwc_sum',
            'linalg.pooling_ncw_max', 'linalg.pooling_ncw_sum',
            'linalg.pooling_nwc_max', 'linalg.pooling_nwc_max_unsigned', 'linalg.pooling_nwc_min', 'linalg.pooling_nwc_min_unsigned', 'linalg.pooling_nwc_sum',
            'linalg.pooling_ndhwc_max', 'linalg.pooling_ndhwc_min', 'linalg.pooling_ndhwc_sum',
            'linalg.broadcast', 'linalg.reduce'
        ]);
    }

    parseOperation(parser, opName, op) {
        if (opName === 'linalg.generic') {
            return this._parseGenericOp(parser, op);
        }
        if (op.name === 'linalg.init_tensor') {
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
                op.addAttribute('static_sizes', dims);
            }
            // Reference: parseOptionalColonTypeList
            parser.resolveOperands(op.results, parser.parseOptionalColonTypeList());
            return true;
        }
        if (opName === 'linalg.fill') {
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
            parser.resolveOperands(op.operands, parser.parseOptionalColonTypeList());
            if (parser.accept('->')) {
                const types = parser.parseFunctionResultTypes();
                parser.resolveOperands(op.results, types);
            }
            return true;
        }
        if (opName === 'linalg.conv') {
            if (parser.accept('(')) {
                op.operands = parser.parseArguments();
                parser.expect(')');
            }
            if (parser.match('{')) {
                parser.parseAttributeDict(op.attributes);
            }
            parser.resolveOperands(op.operands, parser.parseOptionalColonTypeList());
            return true;
        }
        if (opName === 'linalg.yield') {
            op.operands = parser.parseArguments();
            parser.resolveOperands(op.operands, parser.parseOptionalColonTypeList());
            return true;
        }
        if (opName === 'linalg.transpose') {
            if (!this._parseInsOutsOp(parser, op)) {
                return false;
            }
            if (parser.match('id', 'permutation')) {
                parser.expect('id', 'permutation');
                parser.expect('=');
                const permutation = parser.skip('[');
                op.addAttribute('permutation', permutation);
            }
            return true;
        }
        if (opName === 'linalg.elementwise') {
            if (parser.accept('id', 'kind')) {
                parser.expect('=');
                const kind = parser.parseAttribute();
                op.addAttribute('kind', kind.value);
            }
            if (parser.accept('id', 'indexing_maps')) {
                parser.expect('=');
                const indexingMaps = parser.skip('[');
                op.addAttribute('indexing_maps', indexingMaps);
            }
            return this._parseInsOutsOp(parser, op);
        }
        if (opName === 'linalg.map') {
            return this._parseMapOp(parser, op);
        }
        if (opName === 'linalg.contract') {
            if (parser.accept('id', 'indexing_maps')) {
                parser.expect('=');
                // Handle both literal array [...] and attribute alias #alias
                if (parser.match('[')) {
                    const indexingMaps = parser.skip('[');
                    op.addAttribute('indexing_maps', indexingMaps);
                } else if (parser.match('#')) {
                    const alias = parser.expect('#');
                    op.addAttribute('indexing_maps', { alias });
                }
            }
            return this._parseInsOutsOp(parser, op);
        }
        // Many linalg operations (especially named structured ops) follow the pattern:
        // linalg.<op_name> <{properties}> [indexing_maps = [...]] {attributes} ins(...) outs(...) [-> type]
        // These named structured ops are generated at LLVM build time from YAML.
        if (this._namedStructuredOps.has(opName)) {
            parser.parsePropertyDict(op.attributes);
            if (parser.match('id', 'indexing_maps')) {
                parser.expect('id', 'indexing_maps');
                parser.expect('=');
                const indexingMaps = parser.skip('[');
                op.addAttribute('indexing_maps', indexingMaps);
            }
            return this._parseInsOutsOp(parser, op);
        }
        const opInfo = this.getOperation(opName);
        if (opInfo && opInfo.metadata.assemblyFormat) {
            return super.parseOperation(parser, opName, op);
        }
        if (parser.match('{') || parser.match('id', 'ins') || parser.match('id', 'outs')) {
            return this._parseInsOutsOp(parser, op);
        }
        return false;
    }

    _parseInsOutsOp(parser, op) {
        // Parse: linalg.op {attrs} ins(%0, %1 : type, type) outs(%2 : type) [-> type]
        if (parser.match('{')) {
            parser.parseAttributeDict(op.attributes);
        }
        if (parser.accept('id', 'ins')) {
            if (!parser.accept('(')) {
                return false;
            }
            while (parser.match('%')) {
                const value = parser.expect('%');
                op.operands.push({ value });
                if (!parser.accept(',')) {
                    break;
                }
            }
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
        if (parser.accept('id', 'outs')) {
            if (!parser.accept('(')) {
                return false;
            }
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
            if (!parser.accept(')')) {
                return false;
            }
        }
        if (parser.match('{')) {
            parser.parseAttributeDict(op.attributes);
        }
        if (parser.match('id', 'dimensions') || parser.match('id', 'permutation')) {
            const attrName = parser.expect('id');
            parser.expect('=');
            const attrValue = parser.parseAttribute();
            if (attrValue) {
                op.addAttribute(attrName, attrValue.value);
            }
        }
        const blockArguments = [];
        if (parser.match('(')) {
            parser.expect('(');
            while (!parser.match(')')) {
                const value = parser.expect('%');
                parser.expect(':');
                const type = parser.parseType();
                blockArguments.push({ value, type });
                if (!parser.accept(',')) {
                    break;
                }
            }
            parser.expect(')');
        }
        if (blockArguments.length > 0 && parser.match('{')) {
            const region = { blocks: [] };
            const block = { operations: [], arguments: blockArguments };
            parser.expect('{');
            while (!parser.accept('}')) {
                if (parser.match('^') || (parser.match('id') && parser.getToken().value && parser.getToken().value.startsWith('^'))) {
                    break;
                }
                const operation = parser.parseOperation();
                block.operations.push(operation);
            }
            region.blocks.push(block);
            op.regions.push(region);
        }
        if (parser.accept('->') || parser.accept('id', 'to')) {
            if (op.results.length > 0) {
                const types = parser.parseFunctionResultTypes();
                parser.resolveOperands(op.results, types);
            } else {
                const type = parser.parseType();
                op.results.push({ type });
            }
        }
        return true;
    }

    // Reference: llvm-project/mlir/lib/Dialect/Linalg/IR/LinalgOps.cpp MapOp::parse
    // Format: linalg.map [{ payload_op }] ins(...) outs(...) [(args)] { region }
    _parseMapOp(parser, op) {
        // Optional short form: { payload_op attr-dict }
        let hasPayloadOp = false;
        if (parser.accept('{')) {
            const payloadOpName = parser.parseOperationName();
            op.addAttribute('payload_op', payloadOpName);
            if (parser.match('{')) {
                parser.parseAttributeDict(op.attributes);
            }
            parser.expect('}');
            hasPayloadOp = true;
        }
        // Parse ins(...) and outs(...)
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
        // Parse optional block arguments: (args)
        const blockArguments = [];
        if (parser.match('(')) {
            parser.expect('(');
            while (!parser.match(')')) {
                const value = parser.expect('%');
                parser.expect(':');
                const type = parser.parseType();
                blockArguments.push({ value, type });
                if (!parser.accept(',')) {
                    break;
                }
            }
            parser.expect(')');
        }
        // Parse region { ... } if not short form
        if (!hasPayloadOp && parser.match('{')) {
            const region = op.addRegion();
            parser.parseRegion(region, blockArguments);
        }
        // Parse optional result type: -> type
        if (parser.accept('->')) {
            const types = parser.parseFunctionResultTypes();
            parser.resolveOperands(op.results, types);
        }
        return true;
    }

    _parseGenericOp(parser, op) {
        if (parser.match('{') || parser.match('#')) {
            if (parser.match('#')) {
                const attrRef = parser.expect('#');
                op.addAttribute('trait', attrRef);
            } else {
                parser.parseAttributeDict(op.attributes);
            }
        }
        if (parser.match('{')) {
            parser.parseAttributeDict(op.attributes);
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
            const region = op.addRegion();
            parser.parseRegion(region);
        }
        if (parser.accept('->')) {
            const hasParens = parser.match('(');
            const types = hasParens ? parser.parseTypeListParens() : parser.parseFunctionResultTypes();
            parser.resolveOperands(op.results, types);
        }
        return true;
    }
};

mlir.ONNXDialect = class extends mlir.Dialect {

    constructor(operations) {
        super(operations, 'onnx');
    }

    parseOperation(parser, opName, op) {
        // onnx.Constant has custom assembly format: dense<...> : type
        // Similar to stablehlo.constant
        if (opName === 'onnx.Constant') {
            // Parse attribute (e.g., dense<"0x...">, dense<[1, 2, 3]>, etc.)
            const value = parser.parseAttribute();
            if (value) {
                op.addAttribute('value', value);
            }
            // Parse optional : type
            for (const type of parser.parseOptionalColonTypeList()) {
                op.results.push({ type: type.toString() });
            }
            return true;
        }
        if (opName === 'onnx.ConstantOfShape') {
            parser.expect('(');
            const operand = parser.parseOperand();
            parser.expect(')');
            parser.parseOptionalAttrDict(op.attributes);
            parser.expect(':');
            parser.expect('(');
            const inputType = parser.parseType();
            operand.type = inputType;
            op.operands.push(operand);
            parser.expect(')');
            parser.expect('->');
            const outputType = parser.parseType();
            op.results.push({ type: outputType.toString() });
            return true;
        }
        return super.parseOperation(parser, opName, op);
    }
};

mlir.KrnlDialect = class extends mlir.Dialect {

    constructor(operations) {
        super(operations, 'krnl');
    }

    parseOperation(parser, opName, op) {
        if (opName === 'krnl.define_loops') {
            if (parser.match('int')) {
                const count = parser.expect('int');
                op.addAttribute('num_loops', count);
            }
            return true;
        }
        if (opName === 'krnl.get_linear_offset_index') {
            const memref = parser.parseAttribute();
            op.operands = [memref];
            if (parser.accept('id', 'at')) {
                parser.expect('[');
                while (!parser.match(']')) {
                    const index = parser.parseAttribute();
                    op.operands.push(index);
                    if (!parser.match(']')) {
                        parser.accept(',');
                    }
                }
                parser.expect(']');
            }
            if (parser.accept(':')) {
                parser.parseType();
            }
            return true;
        }
        if (opName === 'krnl.prefetch' && this.hasCustomAssemblyFormat(opName) && !this.hasAssemblyFormat(opName)) {
            const memref = parser.parseAttribute();
            op.operands.push(memref);
            if (parser.accept('[')) {
                while (!parser.match(']')) {
                    const index = parser.parseAttribute();
                    op.operands.push(index);
                    if (!parser.match(']')) {
                        parser.accept(',');
                    }
                }
                parser.expect(']');
            }
            parser.expect(',');
            const readOrWrite = parser.expect('id');
            op.addAttribute('isWrite', readOrWrite === 'write');
            parser.expect(',');
            parser.expect('id', 'locality');
            parser.expect('<');
            const localityHint = parser.parseInteger();
            op.addAttribute('localityHint', localityHint);
            parser.expect('>');
            parser.expect(',');
            const cacheType = parser.expect('id');
            op.addAttribute('isDataCache', cacheType === 'data');
            parser.parseOptionalAttrDict(op.attributes);
            if (parser.accept(':')) {
                const type = parser.parseType();
                op.operands[0].type = type.toString();
            }
            return true;
        }
        if (opName === 'krnl.iterate') {
            op.operands = parser.parseArguments();
            if (parser.accept('id', 'with')) {
                parser.expect('(');
                const numOptimizedLoops = op.operands.length;
                while (!parser.match(')')) {
                    parser.expect('%');
                    parser.expect('->');
                    parser.expect('%');
                    parser.expect('=');
                    parser.accept('id', 'max');
                    if (parser.match('id', 'affine_map') || parser.match('id', 'affine_set')) {
                        parser.parseAttribute();
                        if (parser.match('(')) {
                            parser.skip('(');
                        }
                        if (parser.match('[')) {
                            parser.skip('[');
                        }
                    } else {
                        parser.parseAttribute();
                    }
                    parser.expect('id', 'to');
                    parser.accept('id', 'min');
                    if (parser.match('id', 'affine_map') || parser.match('id', 'affine_set')) {
                        parser.parseAttribute();
                        if (parser.match('(')) {
                            parser.skip('(');
                        }
                        if (parser.match('[')) {
                            parser.skip('[');
                        }
                    } else {
                        parser.parseAttribute();
                    }
                    if (!parser.match(')')) {
                        parser.accept(',');
                    }
                }
                parser.expect(')');
                op.addAttribute('num_optimized_loops', numOptimizedLoops);
            }
            if (parser.accept('id', 'iter_args')) {
                parser.expect('(');
                while (!parser.match(')')) {
                    parser.expect('%');
                    parser.expect('=');
                    parser.parseAttribute();
                    if (!parser.match(')')) {
                        parser.accept(',');
                    }
                }
                parser.expect(')');
                if (parser.accept('->')) {
                    const types = parser.parseFunctionResultTypes();
                    parser.resolveOperands(op.results, types);
                }
            }
            if (parser.match('{')) {
                const region = {};
                parser.parseRegion(region);
                op.regions = [region];
            }
            return true;
        }
        return super.parseOperation(parser, opName, op);
    }
};

mlir.MhloDialect = class extends mlir.HLODialect {

    constructor(operations) {
        super(operations, 'mhlo');
    }

    parseOperation(parser, opName, op) {
        const opInfo = this.getOperation(opName);
        if (opInfo && opInfo.metadata.parser && opInfo.metadata.parser.includes('parseOneResultSameOperandTypeOp')) {
            return this.parseOneResultSameOperandTypeOp(parser, op);
        }
        if (opName === 'mhlo.constant') {
            if (parser.accept('(') && parser.accept(')')) {
                if (parser.match('{')) {
                    parser.parseAttributeDict(op.attributes);
                }
                // Reference: parseOptionalColonTypeList for operands
                parser.resolveOperands(op.operands, parser.parseOptionalColonTypeList());
                // Reference: parseOptionalArrowTypeList for results
                parser.resolveOperands(op.results, parser.parseOptionalArrowTypeList());
            } else {
                if (parser.match('{')) {
                    parser.parseAttributeDict(op.attributes);
                }
                const value = parser.parseAttribute();
                if (value) {
                    op.addAttribute('value', value);
                }
                // Reference: parseOptionalColonTypeList
                parser.resolveOperands(op.results, parser.parseOptionalColonTypeList());
            }
            return true;
        }
        if (opName === 'mhlo.compare') {
            // Use modern assemblyFormat if available, but only if this is modern syntax
            // Old format has 'attributes' keyword, new format has attr-dict directly
            if (opInfo && opInfo.metadata.assemblyFormat && !parser.match('id', 'attributes')) {
                return super.parseOperation(parser, opName, op);
            }
            // Parse: comparison_direction, lhs, rhs [, compare_type] : (type, type) -> type
            // Legacy parser for old mhlo.compare without assembly format
            if (parser.match('id')) {
                const comparisonDirection = parser.expect('id');
                op.addAttribute('comparison_direction', comparisonDirection);
                parser.expect(',');
                op.operands = parser.parseArguments();
                // Check for optional compare_type
                if (parser.accept(',') && parser.match('id')) {
                    const compareType = parser.expect('id');
                    op.addAttribute('compare_type', compareType);
                }
                parser.parseOptionalAttrDict(op.attributes);
                if (parser.accept(':')) {
                    // Reference: Parse FunctionType (type, type) -> type or just type list
                    const type = parser.parseType();
                    if (type instanceof mlir.FunctionType) {
                        parser.resolveOperands(op.operands, type.inputs);
                        parser.resolveOperands(op.results, type.results);
                    } else {
                        // Single type applied to all operands
                        for (const operand of op.operands) {
                            if (!operand.type) {
                                operand.type = type;
                            }
                        }
                    }
                }
                return true;
            }
        }
        if (opName === 'mhlo.reduce') {
            return this._parseReduceOp(parser, op);
        }
        if (opName === 'mhlo.while') {
            // mhlo.while always uses parenthesized form with named arguments
            parser.expect('(');
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
        return super.parseOperation(parser, opName, op);
    }

    _parseReduceOp(parser, op) {
        const operands = [];
        const initOperands = [];
        while (true) {
            if (!parser.accept('(')) {
                break;
            }
            const operand = { value: parser.expect('%') };
            parser.expect('id', 'init');
            parser.expect(':');
            const initOperand = { value: parser.expect('%') };
            parser.expect(')');
            operands.push(operand);
            initOperands.push(initOperand);
            parser.accept(',');
        }
        op.operands = operands.concat(initOperands);

        // Check if compact syntax: "applies <inner-op>"
        if (parser.accept('id', 'applies')) {
            const innerOpName = parser.parseCustomOperationName();
            parser.expect('id', 'across');
            parser.expect('id', 'dimensions');
            parser.expect('=');
            parser.expect('[');
            const dimensions = [];
            while (!parser.match(']')) {
                if (parser.match('int')) {
                    dimensions.push(parser.expect('int'));
                } else {
                    throw new mlir.Error(`Expected integer dimension in reduce operation ${parser.location()}`);
                }
                if (!parser.accept(',') && !parser.match(']')) {
                    throw new mlir.Error(`Expected ',' or ']' in dimensions list ${parser.location()}`);
                }
            }
            parser.expect(']');
            op.addAttribute('dimensions', dimensions);
            parser.parseOptionalAttrDict(op.attributes);
            // Parse function type: (input types) -> result types
            // Reference: Use FunctionType pattern
            if (parser.accept(':')) {
                const type = parser.parseType();
                if (type instanceof mlir.FunctionType) {
                    parser.resolveOperands(op.operands, type.inputs);
                    parser.resolveOperands(op.results, type.results);
                } else {
                    // Single type or type list for operands
                    if (Array.isArray(type)) {
                        parser.resolveOperands(op.operands, type);
                    } else {
                        for (const operand of op.operands) {
                            if (!operand.type) {
                                operand.type = type;
                            }
                        }
                    }
                    // Check for arrow followed by result types
                    if (parser.accept('->')) {
                        const resultTypes = parser.parseFunctionResultTypes();
                        parser.resolveOperands(op.results, resultTypes);
                    }
                }
            }

            // Create a region with the inner operation
            const region = { blocks: [] };
            const block = { operations: [], arguments: [] };
            // Get element type from first input
            let elementType = null;
            if (op.operands.length > 0 && op.operands[0].type) {
                const tensorMatch = op.operands[0].type.toString().match(/tensor<.*?x([^>]+)>/);
                if (tensorMatch) {
                    [, elementType] = tensorMatch;
                } else {
                    const scalarMatch = op.operands[0].type.toString().match(/tensor<([^>]+)>/);
                    if (scalarMatch) {
                        [, elementType] = scalarMatch;
                    }
                }
            }
            block.arguments.push({ value: '%lhs', type: elementType ? `tensor<${elementType}>` : null });
            block.arguments.push({ value: '%rhs', type: elementType ? `tensor<${elementType}>` : null });
            const innerOp = new mlir.Operation(innerOpName);
            innerOp.operands.push({ value: '%lhs' });
            innerOp.operands.push({ value: '%rhs' });
            innerOp.results.push({ value: '%0', type: elementType ? `tensor<${elementType}>` : null });
            block.operations.push(innerOp);
            const returnOp = new mlir.Operation('mhlo.return');
            returnOp.operands.push({ value: '%0' });
            block.operations.push(returnOp);
            region.blocks.push(block);
            op.regions.push(region);
            return true;
        }

        // Non-compact syntax: parse "across dimensions = [...] : type reducer"
        parser.expect('id', 'across');
        parser.expect('id', 'dimensions');
        parser.expect('=');
        parser.expect('[');
        const dimensions = [];
        while (!parser.match(']')) {
            if (parser.match('int')) {
                dimensions.push(parser.expect('int'));
            } else {
                throw new mlir.Error(`Expected integer dimension in reduce operation ${parser.location()}`);
            }
            if (!parser.accept(',') && !parser.match(']')) {
                throw new mlir.Error(`Expected ',' or ']' in dimensions list ${parser.location()}`);
            }
        }
        parser.expect(']');
        op.addAttribute('dimensions', dimensions);

        parser.parseOptionalAttrDict(op.attributes);

        if (parser.accept(':')) {
            const fnType = parser.parseFunctionType();
            parser.resolveOperands(op.operands, fnType.inputs);
            parser.resolveOperands(op.results, fnType.results);
        }
        parser.expect('id', 'reducer');
        const reducerArgs = [];
        while (parser.accept('(')) {
            const arg1 = { value: parser.expect('%') };
            parser.expect(':');
            arg1.type = parser.parseType();
            parser.expect(',');
            const arg2 = { value: parser.expect('%') };
            parser.expect(':');
            arg2.type = parser.parseType();
            parser.expect(')');
            reducerArgs.push(arg1, arg2);
        }
        const region = {};
        region.blocks = [];
        const block = { operations: [], arguments: reducerArgs };
        parser.expect('{');
        while (!parser.accept('}')) {
            const innerOp = parser.parseOperation();
            block.operations.push(innerOp);
        }
        region.blocks.push(block);
        op.regions.push(region);
        return true;
    }

    parseOneResultSameOperandTypeOp(parser, op) {
        op.operands = parser.parseArguments();
        parser.parseOptionalAttrDict(op.attributes);
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
};

mlir.THLODialect = class extends mlir.Dialect {

    constructor(operations) {
        super(operations, 'thlo');
    }

    parseOperation(parser, opName, op) {
        if (this.hasAssemblyFormat(opName)) {
            return super.parseOperation(parser, opName, op);
        }
        if (parser.accept('id', 'ins')) {
            parser.expect('(');
            while (parser.match('%')) {
                const value = parser.expect('%');
                let type = null;
                if (parser.accept(':')) {
                    type = parser.parseType();
                }
                op.operands.push({ value, type });
                if (!parser.accept(',')) {
                    break;
                }
            }
            parser.expect(')');
        }
        // Parse 'outs' section - format: outs(%arg1: type1, %arg2: type2)
        if (parser.accept('id', 'outs')) {
            parser.expect('(');
            while (parser.match('%')) {
                const value = parser.expect('%');
                let type = null;
                if (parser.accept(':')) {
                    type = parser.parseType();
                }
                op.operands.push({ value, type });
                if (!parser.accept(',')) {
                    break;
                }
            }
            parser.expect(')');
        }
        if (parser.match('{')) {
            parser.parseAttributeDict(op.attributes);
        }
        // Parse block arguments and region: (%arg1: type, ...) { body }
        const blockArguments = [];
        if (parser.match('(')) {
            parser.expect('(');
            while (!parser.match(')')) {
                const value = parser.expect('%');
                parser.expect(':');
                const type = parser.parseType();
                blockArguments.push({ value, type });
                if (!parser.accept(',')) {
                    break;
                }
            }
            parser.expect(')');
        }
        if (parser.match('{')) {
            const region = { blocks: [] };
            const block = { operations: [], arguments: blockArguments };
            parser.expect('{');
            while (!parser.accept('}')) {
                const operation = parser.parseOperation();
                block.operations.push(operation);
            }
            region.blocks.push(block);
            op.regions.push(region);
        }
        return true;
    }
};

mlir.QuantDialect = class extends mlir.Dialect {

    constructor(operations) {
        super(operations, 'quant');
    }

    parseType(parser, dialectName) {
        const typeName = parser.parseKeyword();
        if (typeName === 'uniform' || typeName === 'calibrated' || typeName === 'any') {
            let type = `!${dialectName}.${typeName}`;
            if (parser.match('<')) {
                const content = parser.skip('<');
                type += content;
            }
            return new mlir.Type(type);
        }
        return null;
    }
};

mlir.TosaDialect = class extends mlir.Dialect {

    constructor(operations) {
        super(operations, 'tosa');
        this._customOps = new Set([
            'tosa.apply_scale', 'tosa.argmax', 'tosa.cast_from_block_scaled',
            'tosa.cast_to_block_scaled', 'tosa.clamp', 'tosa.max_pool2d',
            'tosa.maximum', 'tosa.minimum', 'tosa.reduce_max', 'tosa.reduce_min',
            'tosa.rescale', 'tosa.resize', 'tosa.matmul_t_block_scaled'
        ]);
        this._regionOps = new Set(['tosa.cond_if', 'tosa.while_loop']);
        this.registerCustomDirective('VariableOpTypeOrInitialValue', this._parseVariableOpTypeOrInitialValue.bind(this));
    }

    parseType(parser, dialectName) {
        const typeName = parser.parseKeyword();
        if (typeName === 'shape') {
            let type = `!${dialectName}.${typeName}`;
            if (parser.match('<')) {
                const content = parser.skip('<');
                type += content;
            }
            return new mlir.Type(type);
        }
        // Reference: TosaTypesBase.td - mxint8 is a simple type without parameters
        if (typeName === 'mxint8') {
            return new mlir.Type(`!${dialectName}.mxint8`);
        }
        return null;
    }

    parseOperation(parser, opName, op) {
        if (opName === 'tosa.variable' && !this.hasAssemblyFormat(opName)) {
            parser.parseSymbolName('sym_name', op.attributes);
            if (parser.accept('=')) {
                const initialValue = parser.parseAttribute();
                op.addAttribute('initial_value', initialValue);
            }
            if (parser.accept(':')) {
                const type = parser.parseType();
                op.addAttribute('type', type);
            }
            return true;
        }
        if (this._regionOps.has(opName) && this.hasCustomAssemblyFormat(opName) && !this.hasAssemblyFormat(opName)) {
            // Reference: TosaOps.cpp IfOp::parse
            let hasBlockArgs = false;
            if (parser.match('%')) {
                const cond = parser.expect('%');
                op.operands.push({ value: cond });
            }
            if (parser.accept('(')) {
                hasBlockArgs = true;
                while (!parser.match(')')) {
                    if (parser.match('%')) {
                        const blockArg = parser.expect('%');
                        parser.expect('=');
                        const inputVal = parser.expect('%');
                        op.operands.push({ blockArg, value: inputVal });
                    }
                    if (!parser.accept(',')) {
                        break;
                    }
                }
                parser.expect(')');
            }
            if (parser.accept(':')) {
                const condType = parser.parseType();
                if (op.operands.length > 0 && !op.operands[0].type) {
                    op.operands[0].type = condType;
                }
                // If block args present, parse function type for inputs/outputs
                if (hasBlockArgs && parser.match('(')) {
                    const functionType = parser.parseFunctionType();
                    if (functionType) {
                        parser.resolveOperands(op.operands.slice(1), functionType.inputs);
                        parser.resolveOperands(op.results, functionType.results);
                    }
                } else if (parser.accept('->')) {
                    const resultTypes = parser.parseFunctionResultTypes();
                    parser.resolveOperands(op.results, resultTypes);
                }
            }
            if (parser.match('{')) {
                const region = op.addRegion();
                parser.parseRegion(region);
            }
            if (parser.accept('id', 'else') || parser.accept('id', 'do')) {
                if (parser.match('{')) {
                    const secondRegion = {};
                    parser.parseRegion(secondRegion);
                    op.regions.push(secondRegion);
                }
            }
            return true;
        }
        if (this._customOps.has(opName) && this.hasCustomAssemblyFormat(opName) && !this.hasAssemblyFormat(opName)) {
            op.operands = parser.parseArguments();
            if (parser.match('{')) {
                // Parse attribute dict but check if any are actually inputs
                const opInfo = this.getOperation(opName);
                const inputNames = new Set((opInfo && opInfo.metadata && opInfo.metadata.operands || []).map((i) => i.name));
                const tempAttrs = new Map();
                parser.parseAttributeDict(tempAttrs);
                for (const [name, value] of tempAttrs) {
                    // If this is an input (like input_zp, output_zp), add as operand
                    if (inputNames.has(name) && value && typeof value === 'string' && value.startsWith('%')) {
                        op.operands.push({ name, value: value.substring(1) });
                    } else if (inputNames.has(name) && value && value.value && typeof value.value === 'string' && value.value.startsWith('%')) {
                        op.operands.push({ name, value: value.value.substring(1) });
                    } else {
                        op.attributes.set(name, value);
                    }
                }
            }
            if (parser.accept(':')) {
                const type = parser.parseType();
                if (type instanceof mlir.FunctionType) {
                    parser.resolveOperands(op.operands, type.inputs);
                    parser.resolveOperands(op.results, type.results);
                } else {
                    for (const operand of op.operands) {
                        if (!operand.type) {
                            operand.type = type;
                        }
                    }
                    if (parser.accept('->')) {
                        const resultTypes = parser.parseFunctionResultTypes();
                        parser.resolveOperands(op.results, resultTypes);
                    }
                }
            }
            return true;
        }
        return super.parseOperation(parser, opName, op);
    }

    _parseVariableOpTypeOrInitialValue(parser, op /*, args */) {
        if (parser.accept('=')) {
            const initialValue = parser.parseAttribute();
            op.addAttribute('initial_value', initialValue);
        } else if (parser.accept(':')) {
            const type = parser.parseType();
            op.addAttribute('type', type);
        }
    }
};

mlir.IRDLDialect = class extends mlir.Dialect {

    constructor(operations) {
        super(operations, 'irdl');
        this.registerCustomDirective('SingleBlockRegion', this._parseSingleBlockRegion.bind(this));
        this.registerCustomDirective('NamedValueList', this._parseNamedValueList.bind(this));
        this.registerCustomDirective('NamedValueListWithVariadicity', this._parseNamedValueListWithVariadicity.bind(this));
        this.registerCustomDirective('AttributesOp', this._parseAttributesOp.bind(this));
    }

    parseOperation(parser, opName, op) {
        // Only use custom parsing for operations that don't have assemblyFormat
        // Operations with assemblyFormat should be handled by the base class
        if ((opName === 'irdl.operands' || opName === 'irdl.results' ||
            opName === 'irdl.parameters' || opName === 'irdl.attributes' ||
            opName === 'irdl.regions') && !this.hasAssemblyFormat(opName)) {
            if (parser.accept('(')) {
                while (!parser.accept(')')) {
                    if (parser.match('id') || parser.match('string')) {
                        const paramName = parser.expect();
                        parser.expect(':');
                        const paramValue = parser.expect(); // Read the SSA value like %tensor
                        op.addAttribute(paramName, paramValue);
                    }
                    parser.accept(',');
                }
            }
            op.loc = parser.parseLocation();
            return true;
        }
        return super.parseOperation(parser, opName, op);
    }

    _parseSingleBlockRegion(parser, op, /* args */) {
        if (parser.match('{')) {
            const region = op.addRegion();
            parser.parseRegion(region);
        }
    }

    _parseNamedValueList(parser, op, args) {
        const argValues = [];
        const nameValues = [];
        const parseOne = () => {
            const name = parser.expect();
            nameValues.push(name);
            parser.expect(':');
            const value = parser.expect();
            argValues.push(value);
            return value;
        };
        parser.parseCommaSeparatedList('paren', parseOne);
        if (args && args.length >= 2) {
            const argsAttrName = args[0].replace(/^\$/, '');
            const namesAttrName = args[1].replace(/^\$/, '');
            // args contains SSA values (%0, %1, ...) - add as operands
            for (const value of argValues) {
                op.operands.push({ name: argsAttrName, value });
            }
            op.addAttribute(namesAttrName, nameValues);
        }
    }

    _parseNamedValueListWithVariadicity(parser, op, args) {
        const argValues = [];
        const nameValues = [];
        const variadicityValues = [];
        const parseOne = () => {
            let variadicity = null;
            if (parser.match('id')) {
                const peekValue = parser.getToken().value;
                if (peekValue === 'single' || peekValue === 'optional' || peekValue === 'variadic') {
                    variadicity = parser.expect('id');
                }
            }
            const name = parser.expect();
            nameValues.push(name);
            parser.expect(':');
            const value = parser.expect();
            argValues.push(value);
            variadicityValues.push(variadicity || 'single');
            return value;
        };
        parser.parseCommaSeparatedList('paren', parseOne);
        if (args && args.length >= 2) {
            const argsAttrName = args[0].replace(/^\$/, '');
            const namesAttrName = args[1].replace(/^\$/, '');
            // args contains SSA values (%0, %1, ...) - add as operands
            for (const value of argValues) {
                op.operands.push({ name: argsAttrName, value });
            }
            op.addAttribute(namesAttrName, nameValues);
            if (args.length >= 3) {
                const variadicityAttrName = args[2].replace(/^\$/, '');
                op.addAttribute(variadicityAttrName, variadicityValues);
            }
        }
    }

    _parseAttributesOp(parser, op, args) {
        // Format: { "attr1" = %0, "attr2" = %1 }
        const argValues = [];
        const nameValues = [];
        if (parser.accept('{')) {
            while (!parser.match('}')) {
                const name = parser.expect('string');
                nameValues.push(name);
                parser.expect('=');
                const value = parser.expect();
                argValues.push(value);
                parser.accept(',');
            }
            parser.expect('}');
        }
        if (args && args.length >= 2) {
            const argsAttrName = args[0].replace(/^\$/, '');
            const namesAttrName = args[1].replace(/^\$/, '');
            for (const value of argValues) {
                op.operands.push({ name: argsAttrName, value });
            }
            op.addAttribute(namesAttrName, nameValues);
        }
    }
};

mlir.XeGPUDialect = class extends mlir.Dialect {

    constructor(operations) {
        super(operations, 'xegpu');
        this.registerCustomDirective('OptionalDynamicIndexList', this._parseOptionalDynamicIndexList.bind(this));
    }

    _parseOptionalDynamicIndexList(parser, op, args) {
        const indices = [];
        const dynamicValues = [];

        if (parser.accept('[')) {
            while (!parser.match(']')) {
                if (parser.match('int') || parser.match('number')) {
                    indices.push(parseInt(parser.expect(), 10));
                } else if (parser.match('%')) {
                    const value = parser.expect('%');
                    dynamicValues.push(value);
                    indices.push(-9223372036854775808);
                } else {
                    break;
                }
                parser.accept(',');
            }
            parser.accept(']');

            if (args && args.length >= 2) {
                const dynamicAttrName = args[0].replace(/^\$/, '');
                const staticAttrName = args[1].replace(/^\$/, '');
                // Dynamic values are SSA operands (%0, %1, ...) - add as operands
                for (const value of dynamicValues) {
                    op.operands.push({ name: dynamicAttrName, value });
                }
                // Static indices are compile-time constants - add as attribute
                if (indices.length > 0) {
                    op.addAttribute(staticAttrName, indices);
                }
            }
        }
    }
};

mlir.ShardDialect = class extends mlir.Dialect {

    constructor(operations) {
        super(operations, 'shard');
        this.registerCustomDirective('DimensionList', this._parseDimensionList.bind(this));
    }

    _parseDimensionList(parser, op, args) {
        const dimensions = [];

        while (true) {
            if (parser.match('?')) {
                parser.expect('?');
                dimensions.push(-1);
            } else if (parser.match('int')) {
                dimensions.push(parser.parseInteger());
            } else {
                break;
            }

            if (parser.match('id')) {
                const token = parser.getToken().value;
                if (token === 'x') {
                    parser.expect('id');
                    continue;
                } else if (token.startsWith('x')) {
                    parser.expect('id');
                    const remaining = token.substring(1);
                    const parts = remaining.split('x');
                    for (const part of parts) {
                        if (part === '?') {
                            dimensions.push(-1);
                        } else if (part !== '') {
                            const num = parseInt(part, 10);
                            if (!isNaN(num)) {
                                dimensions.push(num);
                            }
                        }
                    }
                    break;
                }
                break;
            }

            if (!parser.match('id') && !parser.match('?')) {
                break;
            }
        }

        if (args && args.length > 0) {
            const attrName = args[0].replace(/^\$/, '');
            op.addAttribute(attrName, dimensions);
        }
    }
};

mlir.SPIRVDialect = class extends mlir.Dialect {

    constructor(operations) {
        super(operations, 'spirv');
        this.typesWithOptionalParams = new Set(['sampler', 'sampled_image', 'matrix', 'image', 'rtarray', 'ptr', 'array', 'struct', 'coopmatrix']);
        this.registerCustomDirective('ImageOperands', this._parseImageOperands.bind(this));
        this.registerCustomDirective('SwitchOpCases', this._parseSwitchOpCases.bind(this));
        this.registerCustomAttribute('SPIRV_ScopeAttr', this._parseEnumFlagsAngleBracketPipe.bind(this));
        this.registerCustomAttribute('SPIRV_MemorySemanticsAttr', this._parseEnumFlagsAngleBracketPipe.bind(this));
        this.registerCustomAttribute('SPIRV_MemoryAccessAttr', this._parseEnumFlagsAngleBracketPipe.bind(this));
        this.registerCustomAttribute('SPIRV_GroupOperationAttr', this._parseEnumFlagsAngleBracketPipe.bind(this));
        this.registerCustomAttribute('SPIRV_KHR_CooperativeMatrixLayoutAttr', this._parseEnumFlagsAngleBracketPipe.bind(this));
        this.registerCustomAttribute('SPIRV_KHR_CooperativeMatrixOperandsAttr', this._parseEnumFlagsAngleBracketPipe.bind(this));
    }

    _parseSwitchOpCases(parser, op) {
        if (!parser.accept('id', 'default')) {
            return;
        }
        if (!parser.accept(':')) {
            return;
        }
        if (!parser.match('^')) {
            return;
        }
        const defaultDestination = parser.expect('^');
        const defaultDest = { label: defaultDestination, arguments: [] };
        if (parser.accept('(')) {
            while (!parser.match(')') && !parser.match(':')) {
                const value = parser.expect('%');
                defaultDest.arguments.push({ value });
                if (!parser.accept(',')) {
                    break;
                }
            }
            if (parser.accept(':')) {
                let idx = 0;
                while (idx < defaultDest.arguments.length && !parser.match(')')) {
                    const type = parser.parseType();
                    if (defaultDest.arguments[idx]) {
                        defaultDest.arguments[idx].type = type;
                    }
                    idx++;
                    parser.accept(',');
                }
            }
            parser.accept(')');
        }
        op.successors = op.successors || [];
        op.successors.push(defaultDest);
        const caseValues = [];
        while (parser.accept(',')) {
            if (!parser.match('int') && !parser.match('-')) {
                break;
            }
            const value = parser.parseInteger();
            caseValues.push(value);
            if (!parser.accept(':')) {
                break;
            }
            if (!parser.match('^')) {
                break;
            }
            const caseDestination = parser.expect('^');
            const caseDest = { label: caseDestination, arguments: [] };
            if (parser.accept('(')) {
                while (!parser.match(')') && !parser.match(':')) {
                    const argValue = parser.expect('%');
                    caseDest.arguments.push({ value: argValue });
                    if (!parser.accept(',')) {
                        break;
                    }
                }
                if (parser.accept(':')) {
                    let idx = 0;
                    while (idx < caseDest.arguments.length && !parser.match(')')) {
                        const type = parser.parseType();
                        if (caseDest.arguments[idx]) {
                            caseDest.arguments[idx].type = type;
                        }
                        idx++;
                        parser.accept(',');
                    }
                }
                parser.accept(')');
            }
            op.successors.push(caseDest);
        }
        if (caseValues.length > 0) {
            op.addAttribute('literals', caseValues);
        }
    }

    _parseImageOperands(parser /*, op, args */) {
        if (parser.match('[')) {
            parser.skip('[');
        }
    }

    parseType(parser, dialectName) {
        let typeName = parser.parseKeyword();
        if (!typeName) {
            return null;
        }
        // Handle sub-dialect types like arm.tensor, KHR.CooperativeMatrix, etc.
        while (parser.accept('.')) {
            const subType = parser.parseKeyword();
            if (subType) {
                typeName += `.${subType}`;
            } else {
                break;
            }
        }
        // Build the full type string
        let type = `!${dialectName}.${typeName}`;
        if (parser.match('<')) {
            const content = parser.skip('<');
            type += content;
        }
        return new mlir.Type(type);
    }

    parseOperation(parser, opName, op) {
        // Operations with '->' in their assembly format should use assembly format parsing
        const arrowFormatOps = new Set([
            'spirv.GL.Distance', 'spirv.GL.FMix', 'spirv.GL.FrexpStruct', 'spirv.GL.Ldexp',
            'spirv.GL.Length', 'spirv.GL.PackHalf2x16', 'spirv.GL.UnpackHalf2x16',
            'spirv.GLSL.Distance', 'spirv.GLSL.FMix', 'spirv.GLSL.FrexpStruct', 'spirv.GLSL.Ldexp',
            'spirv.GLSL.Length', 'spirv.GLSL.PackHalf2x16', 'spirv.GLSL.UnpackHalf2x16',
            'spv.GL.Distance', 'spv.GL.FMix', 'spv.GL.FrexpStruct', 'spv.GL.Ldexp',
            'spv.GL.Length', 'spv.GL.PackHalf2x16', 'spv.GL.UnpackHalf2x16',
            'spv.GLSL.Distance', 'spv.GLSL.FMix', 'spv.GLSL.FrexpStruct', 'spv.GLSL.Ldexp',
            'spv.GLSL.Length', 'spv.GLSL.PackHalf2x16', 'spv.GLSL.UnpackHalf2x16'
        ]);
        if ((opName.startsWith('spirv.GLSL.') || opName.startsWith('spv.GLSL.') || opName.startsWith('spirv.GL.') || opName.startsWith('spv.GL.')) && !arrowFormatOps.has(opName)) {
            while (!parser.match(':')) {
                const operand = parser.parseAttribute();
                op.operands.push(operand);
                if (!parser.accept(',')) {
                    break;
                }
            }
            if (parser.accept(':')) {
                const type = parser.parseType();
                op.results.push({ type });
            }
            return true;
        }
        if (opName === 'spirv.SpecConstantComposite' || opName === 'spv.SpecConstantComposite') {
            parser.parseSymbolName('sym_name', op.attributes);
            parser.expect('(');
            const constituents = [];
            while (!parser.match(')')) {
                if (parser.match('@')) {
                    constituents.push(parser.expect('@'));
                }
                if (!parser.accept(',')) {
                    break;
                }
            }
            parser.expect(')');
            op.addAttribute('constituents', constituents);
            if (parser.accept(':')) {
                const type = parser.parseType();
                op.addAttribute('type', type.toString());
            }
            return true;
        }
        if (opName.endsWith('.SpecConstantCompositeReplicate')) {
            parser.parseSymbolName('sym_name', op.attributes);
            parser.expect('(');
            if (parser.match('@')) {
                const constituent = parser.expect('@');
                op.addAttribute('constituent', constituent);
            }
            parser.expect(')');
            if (parser.accept(':')) {
                const type = parser.parseType();
                op.addAttribute('type', type.toString());
            }
            return true;
        }
        if (opName === 'spirv.SpecConstantOperation' || opName === 'spv.SpecConstantOperation') {
            parser.expect('id', 'wraps');
            const wrappedOp = parser.parseGenericOperation();
            if (wrappedOp) {
                const region = { blocks: [{ operations: [wrappedOp] }] };
                op.regions.push(region);
                if (wrappedOp.results && wrappedOp.results.length > 0) {
                    op.results.push({ type: wrappedOp.results[0].type });
                }
            }
            parser.parseOptionalAttrDict(op.attributes);
            return true;
        }
        if (opName === 'spirv.Constant' || opName === 'spv.Constant') {
            const value = parser.parseAttribute();
            if (parser.accept(':')) {
                const valueType = parser.parseType();
                op.addAttribute('value', { ...value, valueType });
            } else {
                op.addAttribute('value', value);
            }
            if (parser.accept(':')) {
                const type = parser.parseType();
                op.results.push({ type });
            }
            return true;
        }
        if (opName === 'spirv.Load' || opName === 'spv.Load') {
            const storageClass = parser.expect('string');
            op.addAttribute('storage_class', storageClass);
            const ptr = parser.parseAttribute();
            op.operands.push(ptr);
            if (parser.accept('[')) {
                const memoryAccess = [];
                while (!parser.match(']')) {
                    if (parser.match('string')) {
                        memoryAccess.push(parser.expect('string'));
                    } else if (parser.match('int')) {
                        memoryAccess.push(parser.expect('int'));
                    } else {
                        break;
                    }
                    parser.accept(',');
                }
                parser.expect(']');
                if (memoryAccess.length > 0) {
                    op.addAttribute('memory_access', memoryAccess.join(', '));
                }
            }
            if (parser.accept(':')) {
                const type = parser.parseType();
                op.results.push({ type });
            }
            return true;
        }
        if (opName === 'spirv.CompositeExtract' || opName === 'spv.CompositeExtract') {
            const composite = parser.parseAttribute();
            op.operands.push(composite);
            if (parser.accept('[')) {
                const indices = [];
                while (!parser.match(']')) {
                    if (parser.match('int')) {
                        indices.push(parser.parseInteger());
                    }
                    if (parser.accept(':')) {
                        parser.parseType();
                    }
                    if (!parser.accept(',')) {
                        break;
                    }
                }
                parser.expect(']');
                op.addAttribute('indices', indices);
            }
            if (parser.accept(':')) {
                const type = parser.parseType();
                op.results.push({ type });
            }
            return true;
        }
        // Handle AccessChain with old syntax (no -> for result type)
        // Format: base_ptr[indices] : base_type, index_types (without -> result_type)
        if (opName === 'spirv.AccessChain' || opName === 'spv.AccessChain') {
            this._operations.get('spirv.AccessChain').hasParseOperation = false; // compatibility
            const basePtr = parser.parseAttribute();
            op.operands.push(basePtr);
            if (parser.accept('[')) {
                while (!parser.match(']')) {
                    const index = parser.parseAttribute();
                    op.operands.push(index);
                    if (!parser.accept(',')) {
                        break;
                    }
                }
                parser.expect(']');
            }
            parser.parseOptionalAttrDict(op.attributes);
            if (parser.accept(':')) {
                // Parse base pointer type
                const basePtrType = parser.parseType();
                op.operands[0].type = basePtrType;
                // Parse index types
                let idx = 1;
                while (parser.accept(',') && idx < op.operands.length) {
                    const indexType = parser.parseType();
                    if (op.operands[idx]) {
                        op.operands[idx].type = indexType;
                    }
                    idx++;
                }
                // Check for optional -> result_type (newer syntax)
                if (parser.accept('->')) {
                    const resultType = parser.parseType();
                    op.results.push({ type: resultType });
                }
            }
            return true;
        }
        if (opName === 'spirv.Variable' || opName === 'spv.Variable') {
            if (parser.accept('id', 'init')) {
                parser.expect('(');
                const init = parser.parseAttribute();
                op.operands.push(init);
                parser.expect(')');
            }
            if (parser.match('{')) {
                parser.parseAttributeDict(op.attributes);
            }
            if (parser.accept(':')) {
                const type = parser.parseType();
                op.results.push({ type });
            }
            return true;
        }
        if (opName === 'spirv.Store' || opName === 'spv.Store') {
            const storageClass = parser.expect('string');
            op.addAttribute('storage_class', storageClass);
            const ptr = parser.parseAttribute();
            op.operands.push(ptr);
            parser.expect(',');
            const value = parser.parseAttribute();
            op.operands.push(value);
            if (parser.accept('[')) {
                const memoryAccess = [];
                while (!parser.match(']')) {
                    if (parser.match('string')) {
                        memoryAccess.push(parser.expect('string'));
                    } else if (parser.match('int')) {
                        memoryAccess.push(parser.expect('int'));
                    } else {
                        break;
                    }
                    parser.accept(',');
                }
                parser.expect(']');
                if (memoryAccess.length > 0) {
                    op.addAttribute('memory_access', memoryAccess.join(', '));
                }
            }
            if (parser.accept(':')) {
                parser.parseType();
            }
            return true;
        }
        if (opName === 'spirv.CompositeInsert' || opName === 'spv.CompositeInsert') {
            const object = parser.parseAttribute();
            op.operands.push(object);
            parser.expect(',');
            const composite = parser.parseAttribute();
            op.operands.push(composite);
            if (parser.accept('[')) {
                const indices = [];
                while (!parser.match(']')) {
                    if (parser.match('int')) {
                        indices.push(parser.parseInteger());
                    }
                    if (parser.accept(':')) {
                        parser.parseType();
                    }
                    if (!parser.accept(',')) {
                        break;
                    }
                }
                parser.expect(']');
                op.addAttribute('indices', indices);
            }
            if (parser.accept(':')) {
                parser.parseType();
                if (parser.accept('id', 'into')) {
                    const type = parser.parseType();
                    op.results.push({ type });
                }
            }
            return true;
        }
        // Reference: ControlFlowOps.cpp BranchConditionalOp::parse
        // Format: spirv.BranchConditional %cond [trueWeight, falseWeight]?, ^trueTarget(args)?, ^falseTarget(args)?
        if (opName === 'spirv.BranchConditional' || opName === 'spv.BranchConditional') {
            const condition = parser.parseAttribute();
            op.operands.push(condition);
            // Parse optional branch weights [trueWeight, falseWeight]
            if (parser.accept('[')) {
                const weights = [];
                while (!parser.match(']')) {
                    if (parser.match('int')) {
                        weights.push(parser.expect('int'));
                    }
                    parser.accept(',');
                }
                parser.expect(']');
                if (weights.length > 0) {
                    op.addAttribute('branch_weights', weights);
                }
            }
            parser.expect(',');
            if (!op.successors) {
                op.successors = [];
            }
            // Parse true branch successor
            const trueLabel = parser.expect('^');
            const trueSucc = { label: trueLabel };
            if (parser.accept('(')) {
                trueSucc.arguments = [];
                while (!parser.match(')') && !parser.match(':')) {
                    if (parser.match('%')) {
                        trueSucc.arguments.push({ value: parser.expect('%') });
                        parser.accept(',');
                    } else {
                        break;
                    }
                }
                if (parser.accept(':')) {
                    let idx = 0;
                    while (!parser.match(')') && idx < trueSucc.arguments.length) {
                        trueSucc.arguments[idx].type = parser.parseType().toString();
                        idx++;
                        parser.accept(',');
                    }
                }
                parser.expect(')');
            }
            op.successors.push(trueSucc);
            parser.expect(',');
            // Parse false branch successor
            const falseLabel = parser.expect('^');
            const falseSucc = { label: falseLabel };
            if (parser.accept('(')) {
                falseSucc.arguments = [];
                while (!parser.match(')') && !parser.match(':')) {
                    if (parser.match('%')) {
                        falseSucc.arguments.push({ value: parser.expect('%') });
                        parser.accept(',');
                    } else {
                        break;
                    }
                }
                if (parser.accept(':')) {
                    let idx = 0;
                    while (!parser.match(')') && idx < falseSucc.arguments.length) {
                        falseSucc.arguments[idx].type = parser.parseType().toString();
                        idx++;
                        parser.accept(',');
                    }
                }
                parser.expect(')');
            }
            op.successors.push(falseSucc);
            return true;
        }
        if (opName === 'spirv.CompositeConstruct' || opName === 'spv.CompositeConstruct') {
            this.getOperation(opName).hasParseOperation = false; // compatibility?
            while (!parser.match(':')) {
                const constituent = parser.parseAttribute();
                op.operands.push(constituent);
                if (!parser.accept(',')) {
                    break;
                }
            }
            if (parser.accept(':')) {
                if (parser.accept('(')) {
                    parser.parseTypeList();
                    parser.expect(')');
                    parser.expect('->');
                }
                const type = parser.parseType();
                op.results.push({ type });
            }
            return true;
        }
        if (opName === 'spirv.SpecConstant' || opName === 'spv.SpecConstant') {
            parser.parseSymbolName('sym_name', op.attributes);
            if (parser.match('id', 'spec_id')) {
                parser.expect('id', 'spec_id');
                parser.expect('(');
                const specId = parser.parseAttribute();
                op.addAttribute('spec_id', specId);
                parser.expect(')');
            }
            if (parser.accept('=')) {
                const defaultValue = parser.parseAttribute();
                op.addAttribute('default_value', defaultValue);
            }
            if (parser.accept(':')) {
                const type = parser.parseType();
                op.addAttribute('type', type);
            }
            return true;
        }
        if (opName === 'spirv.module' || opName === 'spv.module') {
            // Optional symbol name: spirv.module @Name ...
            if (parser.match('@')) {
                parser.parseSymbolName('sym_name', op.attributes);
            }
            if (parser.match('id')) {
                const addressingModel = parser.expect('id');
                op.addAttribute('addressing_model', addressingModel);
            }
            if (parser.match('id')) {
                const memoryModel = parser.expect('id');
                op.addAttribute('memory_model', memoryModel);
            }
            if (parser.accept('id', 'requires')) {
                const vce = parser.parseAttribute();
                op.addAttribute('vce_triple', vce);
            }
            if (parser.accept('id', 'attributes')) {
                parser.parseAttributeDict(op.attributes);
            }
            if (parser.match('{')) {
                const region = op.addRegion();
                parser.parseRegion(region);
            }
            return true;
        }
        if (opName === 'spirv.ARM.Graph') {
            // Reference: ArmGraphOps.cpp GraphARMOp::parse
            // Format: @name (args) -> (results) [attributes] { body }
            parser.parseFunctionOp(op, false);
            return true;
        }
        if (opName === 'spirv.ARM.GraphEntryPoint') {
            // Reference: ArmGraphOps.cpp GraphEntryPointARMOp::parse
            // Format: @fn_name, @interface1, @interface2, ...
            const fn = parser.expect('@');
            op.addAttribute('fn', fn);
            const interfaceVars = [];
            while (parser.accept(',')) {
                const varSymbol = parser.expect('@');
                interfaceVars.push(varSymbol);
            }
            op.addAttribute('interface', interfaceVars);
            return true;
        }
        if (opName === 'spirv.func' || opName === 'spv.func') {
            if (parser.match('@')) {
                parser.parseSymbolName('sym_name', op.attributes);
            }
            let inputs = [];
            const results = [];
            const resultAttrs = [];
            if (parser.match('(')) {
                const argResult = parser.parseFunctionArgumentList();
                inputs = argResult.arguments.map((a) => a.type);
            }
            if (parser.accept('->')) {
                parser.parseFunctionResultList(results, resultAttrs);
            }
            op.addAttribute('function_type', new mlir.TypeAttrOf(new mlir.FunctionType(inputs, results)));
            if (parser.match('string')) {
                const control = parser.expect('string');
                op.addAttribute('function_control', control);
            }
            if (parser.accept('id', 'attributes')) {
                parser.parseAttributeDict(op.attributes);
            }
            if (parser.match('{')) {
                const region = op.addRegion();
                parser.parseRegion(region);
            }
            return true;
        }
        if (opName === 'spirv.GlobalVariable' || opName === 'spv.GlobalVariable') {
            if (parser.match('@')) {
                parser.parseSymbolName('sym_name', op.attributes);
            }
            if (parser.accept('id', 'initializer')) {
                parser.expect('(');
                const initSymbol = parser.expect('@');
                parser.expect(')');
                op.addAttribute('initializer', initSymbol);
            }
            if (parser.accept('id', 'built_in')) {
                parser.expect('(');
                const builtIn = parser.expect('string');
                parser.expect(')');
                op.addAttribute('built_in', builtIn);
            }
            if (parser.accept('id', 'bind')) {
                parser.expect('(');
                const binding = parser.expect();
                parser.accept(',');
                const set = parser.expect();
                parser.expect(')');
                op.addAttribute('descriptor_set', set);
                op.addAttribute('binding', binding);
            }
            if (parser.match('{')) {
                parser.parseAttributeDict(op.attributes);
            }
            if (parser.accept(':')) {
                const type = parser.parseType();
                op.results = [{ type }];
            }
            return true;
        }
        if (opName === 'spirv.EntryPoint' || opName === 'spv.EntryPoint') {
            // Parse execution model string ("GLCompute", "Vertex", "Fragment", etc.)
            if (parser.match('string')) {
                const executionModel = parser.expect('string');
                op.addAttribute('execution_model', executionModel);
            }
            op.operands = [];
            while (parser.match('@')) {
                const symbol = parser.expect('@');
                op.addAttribute('fn', new mlir.SymbolRefAttr(symbol));
                parser.accept(',');
            }
            return true;
        }
        if (opName === 'spirv.ExecutionMode' || opName === 'spv.ExecutionMode') {
            if (parser.match('@')) {
                const symbol = parser.expect('@');
                op.addAttribute('fn', new mlir.SymbolRefAttr(symbol));
            }
            if (parser.match('string')) {
                const mode = parser.expect('string');
                op.addAttribute('execution_mode', mode);
            }
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
        if (opName === 'spirv.mlir.loop' || opName === 'spv.mlir.loop' || opName === 'spirv.mlir.selection' || opName === 'spv.mlir.selection') {
            // Parse optional control(EnumValue) attribute
            if (parser.accept('id', 'control')) {
                parser.expect('(');
                const controlValue = parser.parseKeyword();
                op.addAttribute('selection_control', controlValue);
                parser.expect(')');
            }
            parser.parseOptionalArrowResultTypes(op);
            if (parser.match('{')) {
                const region = op.addRegion();
                parser.parseRegion(region);
            }
            return true;
        }
        // spirv.CompositeInsert with 'into' keyword
        // Format: spirv.CompositeInsert %object, %composite[indices] : object-type into composite-type
        if (opName === 'spirv.CompositeInsert' || opName === 'spv.CompositeInsert') {
            // Parse operands (object and composite)
            op.operands = parser.parseArguments();
            // Parse indices as attributes
            if (parser.accept('[')) {
                const indices = [];
                while (!parser.accept(']')) {
                    const index = parser.expect();
                    if (parser.accept(':')) {
                        parser.expect(); // Skip type (e.g., i32)
                    }
                    indices.push(index);
                    parser.accept(',');
                }
                op.addAttribute('indices', indices);
            }
            parser.resolveOperands(op.operands, parser.parseOptionalColonTypeList());
            if (parser.accept('id', 'into')) {
                const resultType = parser.parseType();
                op.results = [{ type: resultType }];
            }
            return true;
        }
        // Reference: SPIRVOps.cpp parseArithmeticExtendedBinaryOp
        // Format: spirv.IAddCarry %op1, %op2 : !spirv.struct<(i32, i32)>
        const arithmeticExtendedOps = new Set([
            'spirv.IAddCarry', 'spv.IAddCarry',
            'spirv.ISubBorrow', 'spv.ISubBorrow',
            'spirv.SMulExtended', 'spv.SMulExtended',
            'spirv.UMulExtended', 'spv.UMulExtended'
        ]);
        if (arithmeticExtendedOps.has(opName)) {
            parser.parseOptionalAttrDict(op.attributes);
            op.operands = parser.parseArguments();
            if (parser.accept(':')) {
                const resultType = parser.parseType();
                op.results.push({ type: resultType });
            }
            return true;
        }
        if (opName === 'spirv.INTEL.SubgroupBlockWrite' || opName === 'spv.INTEL.SubgroupBlockWrite') {
            const storageClass = parser.expect('string');
            op.addAttribute('storage_class', storageClass);
            const ptr = parser.parseAttribute();
            op.operands.push(ptr);
            parser.expect(',');
            const value = parser.parseAttribute();
            op.operands.push(value);
            if (parser.accept(':')) {
                const elementType = parser.parseType();
                value.type = elementType.toString();
                ptr.type = `!spirv.ptr<${elementType}, ${storageClass}>`;
            }
            return true;
        }
        if ((opName === 'spirv.CopyMemory' || opName === 'spv.CopyMemory') && this.hasCustomAssemblyFormat(opName) && !this.hasAssemblyFormat(opName)) {
            const targetStorageClass = parser.expect('string');
            op.addAttribute('target_storage_class', targetStorageClass);
            const target = parser.parseAttribute();
            op.operands.push(target);
            parser.expect(',');
            const sourceStorageClass = parser.expect('string');
            op.addAttribute('source_storage_class', sourceStorageClass);
            const source = parser.parseAttribute();
            op.operands.push(source);
            if (parser.accept('[')) {
                const memoryAccess = [];
                while (!parser.match(']')) {
                    if (parser.match('string')) {
                        memoryAccess.push(parser.expect('string'));
                    } else if (parser.match('int')) {
                        memoryAccess.push(parser.expect('int'));
                    } else {
                        break;
                    }
                    parser.accept(',');
                }
                parser.expect(']');
                if (memoryAccess.length > 0) {
                    op.addAttribute('memory_access', memoryAccess.join(', '));
                }
            }
            if (parser.accept(',')) {
                if (parser.accept('[')) {
                    const sourceMemoryAccess = [];
                    while (!parser.match(']')) {
                        if (parser.match('string')) {
                            sourceMemoryAccess.push(parser.expect('string'));
                        } else if (parser.match('int')) {
                            sourceMemoryAccess.push(parser.expect('int'));
                        } else {
                            break;
                        }
                        parser.accept(',');
                    }
                    parser.expect(']');
                    if (sourceMemoryAccess.length > 0) {
                        op.addAttribute('source_memory_access', sourceMemoryAccess.join(', '));
                    }
                }
            }
            if (parser.accept(':')) {
                const elementType = parser.parseType();
                target.type = `!spirv.ptr<${elementType}, ${targetStorageClass}>`;
                source.type = `!spirv.ptr<${elementType}, ${sourceStorageClass}>`;
            }
            return true;
        }
        return super.parseOperation(parser, opName, op);
    }
};

mlir.WasmSSADialect = class extends mlir.Dialect {

    constructor(operations) {
        super(operations, 'wasmssa');
        this.registerCustomType('WasmSSA_LocalRef', this._parseLocalRefType.bind(this));
        this.registerCustomDirective('ElseRegion', this._parseElseRegion.bind(this));
    }

    parseType(parser) {
        if (parser.match('id', 'local')) {
            parser.expect('id', 'local');
            parser.expect('id', 'ref');
            parser.expect('id', 'to');
            const elementType = parser.parseType();
            return new mlir.Type(`!wasmssa<local ref to ${elementType}>`);
        }
        return null;
    }

    parseOperation(parser, opName, op) {
        if (opName === 'wasmssa.import_global') {
            const importName = parser.expect('string');
            op.addAttribute('importName', importName);
            parser.expect('id', 'from');
            const moduleName = parser.expect('string');
            op.addAttribute('moduleName', moduleName);
            parser.expect('id', 'as');
            parser.parseSymbolName('sym_name', op.attributes);
            if (parser.accept('id', 'mutable')) {
                op.addAttribute('isMutable', new mlir.UnitAttr());
            }
            parser.expect(':');
            const type = parser.parseType();
            op.addAttribute('type', type);
            return true;
        }
        if (opName === 'wasmssa.global') {
            if (parser.accept('id', 'exported')) {
                op.addAttribute('exported', new mlir.UnitAttr());
            }
            parser.parseSymbolName('sym_name', op.attributes);
            const type = parser.parseType();
            op.addAttribute('type', type);
            if (parser.accept('id', 'mutable')) {
                op.addAttribute('isMutable', new mlir.UnitAttr());
            }
            parser.expect(':');
            const region = op.addRegion();
            parser.parseRegion(region);
            return true;
        }
        if (opName === 'wasmssa.func') {
            if (parser.accept('id', 'exported')) {
                op.addAttribute('exported', new mlir.UnitAttr());
            }
            parser.parseFunctionOp(op, false);
            return true;
        }
        return super.parseOperation(parser, opName, op);
    }

    // Parse WasmSSA LocalRef type in bare format: `ref to <elementType>`
    // This is used for parseCustomTypeWithFallback when the type doesn't start with `!`
    // Example: `ref to i32` in `wasmssa.local_get %arg0 : ref to i32`
    _parseLocalRefType(parser) {
        // Parse `ref to <type>` - the bare form used in assembly format
        parser.expect('id', 'ref');
        parser.expect('id', 'to');
        const elementType = parser.parseType();
        return new mlir.Type(`ref to ${elementType}`);
    }

    // Parse ElseRegion directive: `else { ... }` or empty
    _parseElseRegion(parser, op) {
        // The else region is optional
        if (parser.accept('id', 'else')) {
            const region = op.addRegion();
            parser.parseRegion(region);
        }
    }
};

mlir.CFDialect = class extends mlir.Dialect {

    constructor(operations) {
        super(operations, 'cf');
        this.registerCustomDirective('SwitchOpCases', this._parseSwitchOpCases.bind(this));
    }

    _parseSwitchOpCases(parser, op) {
        if (!parser.accept('id', 'default')) {
            return false;
        }
        if (!parser.accept(':')) {
            return false;
        }
        if (!parser.match('^')) {
            return false;
        }
        const defaultDestination = parser.expect('^');
        const defaultDest = { label: defaultDestination, arguments: [] };
        if (parser.accept('(')) {
            while (!parser.match(')') && !parser.match(':')) {
                const value = parser.expect('%');
                defaultDest.arguments.push({ value });
                if (!parser.accept(',')) {
                    break;
                }
            }
            if (parser.accept(':')) {
                let idx = 0;
                while (idx < defaultDest.arguments.length && !parser.match(')')) {
                    const type = parser.parseType();
                    if (defaultDest.arguments[idx]) {
                        defaultDest.arguments[idx].type = type;
                    }
                    idx++;
                    parser.accept(',');
                }
            }
            parser.accept(')');
        }
        op.successors = op.successors || [];
        op.successors.push(defaultDest);
        const caseValues = [];
        const caseOperandSegments = [defaultDest.arguments.length];
        while (parser.accept(',')) {
            if (!parser.match('int')) {
                break;
            }
            const value = parser.parseInteger();
            caseValues.push(value);
            if (!parser.accept(':')) {
                break;
            }
            if (!parser.match('^')) {
                break;
            }
            const caseDestination = parser.expect('^');
            const caseDest = { label: caseDestination, arguments: [] };
            if (parser.accept('(')) {
                while (!parser.match(')') && !parser.match(':')) {
                    const operandValue = parser.expect('%');
                    caseDest.arguments.push({ value: operandValue });
                    if (!parser.accept(',')) {
                        break;
                    }
                }
                if (parser.accept(':')) {
                    let idx = 0;
                    while (idx < caseDest.arguments.length && !parser.match(')')) {
                        const type = parser.parseType();
                        if (caseDest.arguments[idx]) {
                            caseDest.arguments[idx].type = type;
                        }
                        idx++;
                        parser.accept(',');
                    }
                }
                parser.accept(')');
            }
            op.successors.push(caseDest);
            caseOperandSegments.push(caseDest.arguments.length);
        }
        if (caseValues.length > 0) {
            op.addAttribute('case_values', caseValues);
            op.addAttribute('case_operand_segments', caseOperandSegments);
        }
        return true;
    }
};

mlir.PDLDialect = class extends mlir.Dialect {

    constructor(operations) {
        super(operations, 'pdl');
        this.registerCustomDirective('OperationOpAttributes', this._parseOperationOpAttributes.bind(this));
        this.registerCustomDirective('RangeType', this._parseRangeType.bind(this));
        this.registerCustomDirective('ResultsValueType', this._parseResultsValueType.bind(this));
        this._customParse = new Set(['pdl.operation']);
    }

    parseOperation(parser, opName, op) {
        if (opName === 'pdl.operation') {
            this._operations.get(opName).hasParseOperation = false;
            return this._parseOperationOp(parser, op);
        }
        return super.parseOperation(parser, opName, op);
    }

    _parseOperationOp(parser, op) {
        if (parser.match('string')) {
            const opNameValue = parser.expect('string');
            op.addAttribute('opName', opNameValue);
        }
        if (parser.accept('(')) {
            while (!parser.match(')')) {
                const value = parser.expect('%');
                op.operands.push({ value, name: 'operandValue' });
                if (!parser.accept(',')) {
                    break;
                }
            }
            if (parser.accept(':')) {
                let idx = 0;
                while (idx < op.operands.length && !parser.match(')')) {
                    const type = parser.parseType();
                    op.operands[idx].type = type;
                    idx++;
                    parser.accept(',');
                }
            }
            parser.accept(')');
        }
        this._parseOperationOpAttributes(parser, op);
        if (parser.accept('->')) {
            parser.accept('(');
            while (!parser.match(')') && !parser.match('{') && !parser.match('id', 'loc')) {
                const value = parser.expect('%');
                op.operands.push({ value, name: 'typeValue' });
                if (!parser.accept(',')) {
                    break;
                }
            }
            if (parser.accept(':')) {
                const typeValueStartIdx = op.operands.findIndex((o) => o.name === 'typeValue');
                let idx = typeValueStartIdx >= 0 ? typeValueStartIdx : op.operands.length;
                while (idx < op.operands.length && !parser.match(')')) {
                    const type = parser.parseType();
                    if (op.operands[idx]) {
                        op.operands[idx].type = type;
                    }
                    idx++;
                    parser.accept(',');
                }
            }
            parser.accept(')');
        }
        return true;
    }

    _parseOperationOpAttributes(parser, op) {
        if (!parser.accept('{')) {
            return true;
        }
        const attributeNames = [];
        while (!parser.match('}')) {
            const name = parser.parseAttribute();
            if (!parser.accept('=')) {
                break;
            }
            const value = parser.expect('%');
            op.operands.push({ value, name: 'attributeValue' });
            attributeNames.push(name);
            if (!parser.accept(',')) {
                break;
            }
        }
        parser.accept('}');
        if (attributeNames.length > 0) {
            op.addAttribute('attributeValueNames', attributeNames);
        }
        return true;
    }

    _parseRangeType(parser, op) {
        // Reference: PDL.cpp parseRangeType
        // If arguments were provided, infer the result type from the argument list.
        // Otherwise, parse the type as a trailing type.
        const hasArgumentTypes = op.operands.length > 0 && op.operands.some((o) => o.type);
        if (hasArgumentTypes) {
            // Infer result type from first operand's element type
            const firstType = op.operands[0].type;
            if (op.results.length > 0) {
                op.results[0].type = firstType;
            }
        } else {
            // Parse `: type`
            parser.expect(':');
            const type = parser.parseType();
            if (op.results.length > 0) {
                op.results[0].type = type;
            }
        }
        return true;
    }

    _parseResultsValueType(parser, op) {
        // Reference: PDL.cpp parseResultsValueType
        // Parses `-> type` for pdl.results operation
        // Format: ($index^)? `of` $parent custom<ResultsValueType>(ref($index), type($val))
        // If index is present, type can be !pdl.value or !pdl.range<value>
        // If index is absent, type is always !pdl.range<value> (full result range)
        if (parser.accept('->')) {
            const type = parser.parseType();
            op.results.push({ type });
        } else {
            // Default to !pdl.range<value> when no explicit type is given
            op.results.push({ type: '!pdl.range<!pdl.value>' });
        }
        return true;
    }
};

mlir.PDLInterpDialect = class extends mlir.Dialect {

    constructor(operations) {
        super(operations, 'pdl_interp');
        this.registerCustomDirective('CreateOperationOpAttributes', this._parseCreateOperationOpAttributes.bind(this));
        this.registerCustomDirective('CreateOperationOpResults', this._parseCreateOperationOpResults.bind(this));
        this.registerCustomDirective('RangeType', this._parseRangeType.bind(this));
    }

    _parseRangeType(parser, op) {
        if (op.operands.length > 0 && op.operands[0].type) {
            let elementType = op.operands[0].type;
            if (typeof elementType === 'string' && elementType.startsWith('!pdl.range<')) {
                elementType = elementType.replace(/^!pdl\.range</, '').replace(/>$/, '');
            }
            const resultType = `!pdl.range<${elementType}>`;
            op.results.push({ type: resultType });
            return;
        }
        if (parser.accept(':')) {
            const resultType = parser.parseType();
            op.results.push({ type: resultType });
        }
    }

    parseOperation(parser, opName, op) {
        if (opName === 'pdl_interp.func' && this.hasCustomAssemblyFormat(opName) && !this.hasAssemblyFormat(opName)) {
            parser.parseFunctionOp(op, false);
            return true;
        }
        if (opName === 'pdl_interp.foreach' && this.hasCustomAssemblyFormat(opName) && !this.hasAssemblyFormat(opName)) {
            return this._parseForeachOp(parser, op);
        }
        return super.parseOperation(parser, opName, op);
    }

    _parseForeachOp(parser, op) {
        const loopVar = parser.expect('%');
        parser.expect(':');
        const loopVarType = parser.parseType();
        parser.expect('id', 'in');
        const range = parser.expect('%');
        op.operands.push({ value: range });
        if (parser.match('{')) {
            const region = {};
            parser.parseRegion(region);
            if (region.blocks && region.blocks.length > 0) {
                if (!region.blocks[0].arguments) {
                    region.blocks[0].arguments = [];
                }
                region.blocks[0].arguments.push({ value: loopVar, type: loopVarType });
            }
            op.regions.push(region);
        }
        if (parser.accept('->')) {
            parser.expect('^');
        }
        parser.parseOptionalAttrDict(op.attributes);
        return true;
    }

    _parseCreateOperationOpAttributes(parser, op) {
        const attrNames = [];
        if (parser.accept('{')) {
            while (!parser.match('}')) {
                const nameAttr = parser.parseAttribute();
                parser.expect('=');
                const operand = parser.expect('%');
                op.operands.push({ value: operand, name: 'attributeValues' });
                attrNames.push(nameAttr);
                if (!parser.accept(',')) {
                    break;
                }
            }
            parser.expect('}');
        }
        if (attrNames.length > 0) {
            op.addAttribute('inputAttributeNames', attrNames);
        }
    }

    _parseCreateOperationOpResults(parser, op) {
        if (!parser.accept('->')) {
            return;
        }
        if (parser.accept('<')) {
            parser.expect('id', 'inferred');
            parser.expect('>');
            op.addAttribute('inferredResultTypes', true);
            return;
        }
        parser.expect('(');
        const operands = [];
        while (!parser.match(')') && !parser.match(':')) {
            const operand = parser.expect('%');
            operands.push({ value: operand, name: 'inputResultTypes' });
            if (!parser.accept(',')) {
                break;
            }
        }
        if (parser.accept(':')) {
            let idx = 0;
            do {
                const type = parser.parseType();
                if (idx < operands.length) {
                    operands[idx].type = type;
                }
                idx++;
            } while (parser.accept(','));
        }
        for (const operand of operands) {
            op.operands.push(operand);
        }
        parser.expect(')');
    }
};

mlir.PtrDialect = class extends mlir.Dialect {

    constructor(operations) {
        super(operations, 'ptr');
        this.registerCustomAttribute('EnumProp', this._parseEnumProp.bind(this));
        this.registerCustomAttribute('Ptr_PtrDiffFlags', this._parsePtrDiffFlags.bind(this));
        this.registerCustomType('Ptr_PtrType', this._parsePtrTypeShorthand.bind(this));
    }

    _parseEnumProp(parser, type) {
        const [innerType] = type.args;
        return this.parseCustomAttributeWithFallback(parser, innerType);
    }

    _parsePtrDiffFlags(parser, type) {
        if (type.values.includes(parser.getToken().value)) {
            return this._parseEnumFlags(parser, type, '|');
        }
        return null;
    }

    _parsePtrTypeShorthand(parser) {
        if (parser.match('<')) {
            const content = parser.skip('<');
            return new mlir.Type(`!ptr.ptr${content}`);
        }
        return parser.parseType();
    }
};

mlir.EmitCDialect = class extends mlir.Dialect {

    constructor(operations) {
        super(operations, 'emitc');
        this.registerCustomType('EmitC_LValueType', this._parseLValueType.bind(this));
        this.registerCustomDirective('SwitchCases', this._parseSwitchCases.bind(this));
        this.registerCustomDirective('EmitCGlobalOpTypeAndInitialValue', this._parseTypeAndInitialValue.bind(this));
        this.registerCustomDirective('EmitCFieldOpTypeAndInitialValue', this._parseTypeAndInitialValue.bind(this));
    }

    parseOperation(parser, opName, op) {
        if (opName === 'emitc.include') {
            if (parser.accept('<')) {
                const include = parser.expect('string');
                parser.expect('>');
                op.addAttribute('is_standard_include', true);
                op.addAttribute('include', include);
            } else {
                const include = parser.expect('string');
                op.addAttribute('include', include);
            }
            return true;
        }
        if (opName === 'emitc.func') {
            parser.parseFunctionOp(op, false);
            return true;
        }
        if (opName === 'emitc.expression') {
            // Format: emitc.expression %operands [noinline] : (inputs) -> output { region }
            // Parse operands
            while (parser.match('%')) {
                const operand = parser.expect('%');
                op.operands.push({ value: operand });
                if (!parser.accept(',')) {
                    break;
                }
            }
            // Parse optional noinline
            if (parser.accept('id', 'noinline')) {
                op.addAttribute('do_not_inline', true);
            }
            // Parse function type
            if (parser.accept(':')) {
                const type = parser.parseType();
                // Function type is (inputs) -> outputs
                // We extract the result type from the function type
                if (type && type.value) {
                    op.addAttribute('type', type);
                    // Try to extract result type from function type
                    const match = type.value.match(/\) -> (.+)$/);
                    if (match) {
                        op.results.push({ type: new mlir.Type(match[1]) });
                    }
                }
            }
            if (parser.match('{')) {
                const region = op.addRegion();
                parser.parseRegion(region);
            }
            return true;
        }
        if (opName === 'emitc.if') {
            const cond = parser.expect('%');
            op.operands.push({ value: cond });
            const thenRegion = {};
            parser.parseRegion(thenRegion);
            op.regions.push(thenRegion);
            if (parser.accept('id', 'else')) {
                const elseRegion = {};
                parser.parseRegion(elseRegion);
                op.regions.push(elseRegion);
            }
            parser.parseOptionalAttrDict(op.attributes);
            return true;
        }
        if (opName === 'emitc.do') {
            const bodyRegion = {};
            parser.parseRegion(bodyRegion);
            op.regions.push(bodyRegion);
            parser.expect('id', 'while');
            const condRegion = {};
            parser.parseRegion(condRegion);
            op.regions.push(condRegion);
            parser.parseOptionalAttrDictWithKeyword(op.attributes);
            return true;
        }
        if (opName === 'emitc.for') {
            // Format: emitc.for %iter = %lb to %ub step %step [: type] { region }
            const iterVar = parser.expect('%');
            parser.expect('=');
            const lb = parser.expect('%');
            op.operands.push({ value: lb });
            parser.expect('id', 'to');
            const ub = parser.expect('%');
            op.operands.push({ value: ub });
            parser.expect('id', 'step');
            const step = parser.expect('%');
            op.operands.push({ value: step });
            // Parse optional type
            if (parser.accept(':')) {
                const type = parser.parseType();
                op.addAttribute('type', type.toString());
            }
            op.addAttribute('iterVar', { value: iterVar, hidden: true });
            if (parser.match('{')) {
                const region = op.addRegion();
                parser.parseRegion(region);
            }
            return true;
        }
        return super.parseOperation(parser, opName, op);
    }

    _parseLValueType(parser) {
        if (parser.match('<')) {
            const content = parser.skip('<');
            return new mlir.Type(`!emitc.lvalue${content}`);
        }
        return null;
    }

    _parseSwitchCases(parser, op /*, args */) {
        const caseValues = [];
        while (parser.accept('id', 'case')) {
            const value = parser.parseInteger();
            caseValues.push(value);
            if (parser.match('{')) {
                const region = op.addRegion();
                parser.parseRegion(region);
            }
        }
        if (caseValues.length > 0) {
            op.addAttribute('cases', caseValues);
        }
    }

    _parseTypeAndInitialValue(parser, op, args) {
        const typeArg = args && args.length > 0 ? args[0] : 'type';
        const valueArg = args && args.length > 1 ? args[1] : 'initial_value';
        const type = parser.parseType();
        op.addAttribute(typeArg.replace(/^\$/, ''), type);
        if (parser.accept('=')) {
            const initialValue = parser.parseAttribute(type);
            op.addAttribute(valueArg.replace(/^\$/, ''), initialValue);
        }
    }
};

mlir.AsukaDialect = class extends mlir.Dialect {

    constructor(operations) {
        super(operations, 'asuka');
        // https://github.com/monellz/FlashTensor/blob/main/bench/ea.mlir
        // uses batch_dims and reduce_dims not valid given the assemblyFormat spec.
        // Custom parsing preserves compatibility with this file.
        this._customParse = new Set(['asuka.dot', 'asuka.add', 'asuka.split', 'asuka.softmax', 'asuka.reduce']);
    }

    parseOperation(parser, opName, op) {
        if (this._customParse.has(opName)) {
            this._operations.get(opName).hasParseOperation = false;
            op.operands = parser.parseArguments();
            while (parser.match('id') && !parser.match(':') && !parser.match('{')) {
                const attrName = parser.expect('id');
                if (parser.accept('=')) {
                    let attrValue = null;
                    if (parser.match('[')) {
                        attrValue = parser.parseAttribute();
                        if (parser.match('id') && parser.token.value === 'x') {
                            parser.expect('id'); // consume 'x'
                            const secondValue = parser.parseAttribute();
                            attrValue = { kind: 'pair', first: attrValue, second: secondValue };
                        }
                    } else {
                        attrValue = parser.parseAttribute();
                    }
                    op.addAttribute(attrName, attrValue);
                    parser.accept(',');
                }
            }
            if (parser.accept(':')) {
                const funcType = parser.parseFunctionType();
                parser.resolveOperands(op.operands, funcType.inputs);
                for (const resultType of funcType.results) {
                    op.results.push({ type: resultType });
                }
            }
            return true;
        }
        return super.parseOperation(parser, opName, op);
    }
};

mlir.AsyncDialect = class extends mlir.Dialect {

    constructor(operations) {
        super(operations, 'async');
        this.registerCustomDirective('AwaitResultType', this._parseAwaitResultType.bind(this));
        this.registerCustomType('Async_ValueType', this._parseValueTypeShorthand.bind(this));
    }

    parseType(parser, dialectName) {
        const typeName = parser.parseKeyword();
        if (!typeName) {
            return null;
        }
        let type = `!${dialectName}.${typeName}`;
        // Handle coro.* types (coro.id, coro.handle, coro.state)
        if (typeName === 'coro') {
            if (parser.accept('.')) {
                const subType = parser.parseKeyword();
                if (subType) {
                    type += `.${subType}`;
                }
            }
            return new mlir.Type(type);
        }
        const simpleTypes = ['token', 'group'];
        if (simpleTypes.includes(typeName)) {
            return new mlir.Type(type);
        }
        if (typeName === 'value') {
            if (parser.match('<')) {
                const content = parser.skip('<');
                type += content;
            }
            return new mlir.Type(type);
        }
        // Fallback for unknown async types
        if (parser.match('<')) {
            type += parser.skip('<');
        }
        return new mlir.Type(type);
    }

    parseOperation(parser, opName, op) {
        if (opName === 'async.execute') {
            return this._parseExecuteOp(parser, op);
        }
        if (opName === 'async.func') {
            return this._parseFuncOp(parser, op);
        }
        return super.parseOperation(parser, opName, op);
    }

    _parseExecuteOp(parser, op) {
        // Reference: Async.cpp:143 uses parseOperandList
        const tokenArgs = parser.parseOperandList('optionalSquare');
        op.operands.push(...tokenArgs);
        if (parser.accept('(')) {
            while (!parser.match(')')) {
                const operand = { value: parser.expect('%') };
                if (parser.accept('id', 'as')) {
                    parser.expect('%');
                }
                if (parser.accept(':')) {
                    operand.type = parser.parseType().toString();
                }
                op.operands.push(operand);
                if (!parser.accept(',')) {
                    break;
                }
            }
            parser.expect(')');
        }
        if (parser.accept('->')) {
            if (parser.accept('(')) {
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
            const region = op.addRegion();
            parser.parseRegion(region);
        }
        return true;
    }

    _parseFuncOp(parser, op) {
        parser.parseOptionalVisibilityKeyword(op.attributes);
        parser.parseSymbolName('sym_name', op.attributes);
        const argResult = parser.parseFunctionArgumentList();
        const inputs = argResult.arguments.map((a) => a.type);
        parser.parseOptionalAttrDictWithKeyword(op.attributes);
        const results = [];
        const resultAttrs = [];
        if (parser.accept('->')) {
            parser.parseFunctionResultList(results, resultAttrs);
        }
        op.addAttribute('function_type', new mlir.TypeAttrOf(new mlir.FunctionType(inputs, results)));
        parser.parseOptionalAttrDictWithKeyword(op.attributes);
        if (parser.match('{')) {
            const region = op.addRegion();
            parser.parseRegion(region);
        }
        return true;
    }

    _parseAwaitResultType(parser, op, args) {
        // custom<AwaitResultType>(type($operand), type($result))
        // This parses the operand type and derives the result type
        const operandType = parser.parseType();
        if (args && args.length > 0 && op.operands.length > 0) {
            op.operands[0].type = operandType;
        }
        const operandTypeStr = operandType ? operandType.toString() : '';
        if (operandTypeStr && operandTypeStr.startsWith('!async.value')) {
            const match = operandTypeStr.match(/!async\.value<(.+)>/);
            if (match && args && args.length > 1) {
                // Extract the inner type and set it as the result type
                const [, innerType] = match;
                if (op.results.length > 0) {
                    op.results[0].type = new mlir.Type(innerType);
                }
            }
        }
    }

    _parseValueTypeShorthand(parser) {
        if (parser.match('<')) {
            const content = parser.skip('<');
            return new mlir.Type(`!async.value${content}`);
        }
        return parser.parseType();
    }
};

mlir.ArithDialect = class extends mlir.Dialect {

    constructor(operations) {
        super(operations, 'arith');
        this.registerCustomAttribute('Arith_FastMathAttr', this._parseEnumFlagsAngleBracketComma.bind(this));
        this.registerCustomAttribute('Arith_IntegerOverflowAttr', this._parseEnumFlagsAngleBracketComma.bind(this));
    }

    parseOperation(parser, opName, op) {
        if (opName === 'arith.select') {
            return this._parseSelectOp(parser, op);
        }
        return super.parseOperation(parser, opName, op);
    }

    _parseSelectOp(parser, op) {
        op.operands = parser.parseArguments();
        if (parser.match('{')) {
            parser.parseAttributeDict(op.attributes);
        }
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
        super(operations, 'builtin');
    }

    parseOperation(parser, opName, op) {
        if (opName === 'builtin.call' || opName === 'builtin.call_indirect') {
            parser.parseSymbolName('callee', op.attributes);
            op.operands = parser.parseArguments();
            parser.resolveOperands(op.operands, parser.parseOptionalColonTypeList());
            if (parser.accept('->')) {
                const resultTypes = parser.parseFunctionResultTypes();
                parser.resolveOperands(op.results, resultTypes);
            }
            return true;
        }
        return super.parseOperation(parser, opName, op);
    }
};

mlir.BufferizationDialect = class extends mlir.Dialect {

    constructor(operations) {
        super(operations, 'bufferization');
    }

    parseOperation(parser, opName, op) {
        if (opName === 'bufferization.alloc_tensor') {
            if (!parser.accept('(')) {
                return false;
            }
            while (!parser.match(')')) {
                if (parser.match('%')) {
                    const operand = {};
                    operand.value = parser.expect('%');
                    op.operands.push(operand);
                    if (!parser.accept(',')) {
                        break;
                    }
                } else {
                    break;
                }
            }
            parser.expect(')');
            if (parser.accept('id', 'copy')) {
                parser.expect('(');
                const copyOperand = {};
                copyOperand.value = parser.expect('%');
                copyOperand.name = 'copy';
                op.operands.push(copyOperand);
                parser.expect(')');
            }
            if (parser.accept('id', 'size_hint')) {
                parser.expect('=');
                const sizeHintOperand = {};
                sizeHintOperand.value = parser.expect('%');
                sizeHintOperand.name = 'size_hint';
                op.operands.push(sizeHintOperand);
            }
            parser.parseOptionalAttrDict(op.attributes);
            if (parser.accept(':')) {
                const type = parser.parseType();
                if (op.results.length === 0) {
                    op.results.push({ type });
                } else {
                    op.results[0].type = type;
                }
            }
            return true;
        }
        // bufferization.to_memref %tensor read_only : tensor_type to memref_type
        if (opName === 'bufferization.to_memref') {
            if (parser.match('%')) {
                const operand = parser.expect('%');
                op.operands.push({ value: operand });
            }
            if (parser.accept('id', 'read_only')) {
                op.addAttribute('read_only', true);
            }
            parser.parseOptionalAttrDict(op.attributes);
            if (parser.accept(':')) {
                const sourceType = parser.parseType();
                op.addAttribute('source_type', sourceType);
                parser.expect('id', 'to');
                const destType = parser.parseType();
                op.results.push({ type: destType });
            }
            return true;
        }
        return super.parseOperation(parser, opName, op);
    }
};

mlir.SCFDialect = class extends mlir.Dialect {

    constructor(operations) {
        super(operations, 'scf');
        this.registerCustomDirective('SwitchCases', this._parseSwitchCases.bind(this));
    }

    parseOperation(parser, opName, op) {
        if (opName === 'scf.for') {
            return this._parseForOp(parser, op);
        }
        if (opName === 'scf.if') {
            return this._parseIfOp(parser, op);
        }
        if (opName === 'scf.while') {
            return this._parseWhileOp(parser, op);
        }
        if (opName === 'scf.forall') {
            return this._parseForallOp(parser, op);
        }
        if (opName === 'scf.forall.in_parallel') {
            return this._parseInParallelOp(parser, op);
        }
        if (opName === 'scf.parallel') {
            return this._parseParallelOp(parser, op);
        }
        if (opName === 'scf.execute_region') {
            return this._parseExecuteRegionOp(parser, op);
        }
        return super.parseOperation(parser, opName, op);
    }

    _parseForOp(parser, op) {
        if (parser.accept('id', 'unsigned')) {
            op.addAttribute('unsignedCmp', true);
        }
        if (!parser.match('%')) {
            return false;
        }
        const inductionVar = parser.expect('%');
        if (!parser.accept('=')) {
            return false;
        }
        if (parser.match('%')) {
            op.operands.push({ value: parser.expect('%') });
        } else {
            return false;
        }
        if (!parser.accept('id', 'to')) {
            return false;
        }
        if (parser.match('%')) {
            op.operands.push({ value: parser.expect('%') });
        } else {
            return false;
        }
        if (!parser.accept('id', 'step')) {
            return false;
        }
        if (parser.match('%')) {
            op.operands.push({ value: parser.expect('%') });
        } else {
            return false;
        }
        if (parser.accept('id', 'iter_args')) {
            if (parser.accept('(')) {
                while (!parser.accept(')')) {
                    if (parser.match('%')) {
                        parser.expect('%'); // Skip the loop-carried variable name
                    }
                    if (parser.accept('=')) {
                        if (parser.match('%')) {
                            op.operands.push({ value: parser.expect('%') });
                        } else {
                            const value = parser.parseAttribute();
                            if (value) {
                                op.operands.push(value);
                            }
                        }
                    }
                    parser.accept(',');
                }
            }
            parser.parseOptionalArrowResultTypes(op);
        }
        if (parser.accept(':')) {
            parser.parseType();
        }
        if (parser.match('{')) {
            const region = {};
            parser.parseRegion(region);
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
        parser.parseOptionalAttrDict(op.attributes);
        return true;
    }

    _parseIfOp(parser, op) {
        if (parser.match('%')) {
            op.operands.push({ value: parser.expect('%') });
        } else {
            return false;
        }
        parser.parseOptionalArrowResultTypes(op);
        if (parser.match('{')) {
            const region = op.addRegion();
            parser.parseRegion(region);
        } else {
            return false;
        }
        // Parse optional else region
        if (parser.accept('id', 'else')) {
            if (parser.match('{')) {
                const region = op.addRegion();
                parser.parseRegion(region);
            }
        }
        parser.parseOptionalAttrDict(op.attributes);
        return true;
    }

    _parseWhileOp(parser, op) {
        if (parser.accept('(')) {
            while (!parser.accept(')')) {
                if (parser.match('%')) {
                    parser.expect('%'); // Skip variable name
                }
                if (parser.accept('=')) {
                    if (parser.match('%')) {
                        op.operands.push({ value: parser.expect('%') });
                    } else {
                        const value = parser.parseAttribute();
                        if (value) {
                            op.operands.push(value);
                        }
                    }
                }
                parser.accept(',');
            }
        }
        if (parser.accept(':')) {
            if (parser.accept('(')) {
                while (!parser.accept(')')) {
                    parser.parseType();
                    parser.accept(',');
                }
            } else {
                parser.parseType();
            }
            parser.parseOptionalArrowResultTypes(op);
        }
        if (parser.match('{')) {
            const region = op.addRegion();
            parser.parseRegion(region);
        }
        if (parser.accept('id', 'do')) {
            if (parser.match('{')) {
                const region = op.addRegion();
                parser.parseRegion(region);
            }
        }
        parser.parseOptionalAttrDictWithKeyword(op.attributes);
        return true;
    }

    _parseForallOp(parser, op) {
        const inductionVars = [];
        if (!parser.accept('(')) {
            return false;
        }
        while (!parser.accept(')')) {
            if (parser.match('%')) {
                inductionVars.push(parser.expect('%'));
            } else {
                return false;
            }
            if (!parser.accept(',')) {
                if (parser.match(')')) {
                    parser.accept(')');
                    break;
                }
                return false;
            }
        }
        const isNormalized = parser.accept('id', 'in');
        if (!isNormalized && !parser.accept('=')) {
            return false;
        }
        if (isNormalized) {
            if (!parser.accept('(')) {
                return false;
            }
            while (!parser.accept(')')) {
                if (parser.match('%')) {
                    const value = parser.expect('%');
                    op.operands.push({ value });
                } else if (parser.match('int')) {
                    const value = parser.expect('int');
                    op.operands.push({ value, type: new mlir.PrimitiveType('i64') });
                } else {
                    return false;
                }
                parser.accept(',');
            }
        } else {
            if (!parser.accept('(')) {
                return false;
            }
            while (!parser.accept(')')) {
                if (parser.match('%')) {
                    const value = parser.expect('%');
                    op.operands.push({ value });
                } else if (parser.match('int')) {
                    const value = parser.expect('int');
                    op.operands.push({ value, type: new mlir.PrimitiveType('i64') });
                } else {
                    return false;
                }
                parser.accept(',');
            }
            if (!parser.accept('id', 'to')) {
                return false;
            }
            if (!parser.accept('(')) {
                return false;
            }
            while (!parser.accept(')')) {
                if (parser.match('%')) {
                    const value = parser.expect('%');
                    op.operands.push({ value });
                } else if (parser.match('int')) {
                    const value = parser.expect('int');
                    op.operands.push({ value, type: new mlir.PrimitiveType('i64') });
                } else {
                    return false;
                }
                parser.accept(',');
            }
            if (!parser.accept('id', 'step')) {
                return false;
            }
            if (!parser.accept('(')) {
                return false;
            }
            while (!parser.accept(')')) {
                if (parser.match('%')) {
                    const value = parser.expect('%');
                    op.operands.push({ value });
                } else if (parser.match('int')) {
                    const value = parser.expect('int');
                    op.operands.push({ value, type: new mlir.PrimitiveType('i64') });
                } else {
                    return false;
                }
                parser.accept(',');
            }
        }
        if (parser.accept('id', 'shared_outs')) {
            if (!parser.accept('(')) {
                return false;
            }
            while (!parser.accept(')')) {
                if (parser.match('%')) {
                    parser.expect('%'); // Skip arg name
                }
                if (parser.accept('=')) {
                    if (parser.match('%')) {
                        op.operands.push({ value: parser.expect('%') });
                    } else {
                        const value = parser.parseAttribute();
                        if (value) {
                            op.operands.push(value);
                        }
                    }
                }
                parser.accept(',');
            }
        }
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
        if (parser.match('{')) {
            const region = op.addRegion();
            parser.parseRegion(region);
        } else {
            return false;
        }
        parser.parseOptionalAttrDict(op.attributes);
        return true;
    }

    _parseParallelOp(parser, op) {
        const inductionVars = [];
        if (!parser.accept('(')) {
            return false;
        }
        while (!parser.accept(')')) {
            if (parser.match('%')) {
                inductionVars.push(parser.expect('%'));
            } else {
                return false;
            }
            parser.accept(',');
        }
        if (!parser.accept('=')) {
            return false;
        }
        if (!parser.accept('(')) {
            return false;
        }
        while (!parser.accept(')')) {
            if (parser.match('%')) {
                op.operands.push({ value: parser.expect('%') });
            } else {
                return false;
            }
            parser.accept(',');
        }
        if (!parser.accept('id', 'to')) {
            return false;
        }
        if (!parser.accept('(')) {
            return false;
        }
        while (!parser.accept(')')) {
            if (parser.match('%')) {
                op.operands.push({ value: parser.expect('%') });
            } else {
                return false;
            }
            parser.accept(',');
        }
        if (!parser.accept('id', 'step')) {
            return false;
        }
        if (!parser.accept('(')) {
            return false;
        }
        while (!parser.accept(')')) {
            if (parser.match('%')) {
                op.operands.push({ value: parser.expect('%') });
            } else {
                return false;
            }
            parser.accept(',');
        }
        if (parser.accept('id', 'init')) {
            if (!parser.accept('(')) {
                return false;
            }
            while (!parser.accept(')')) {
                if (parser.match('%')) {
                    op.operands.push({ value: parser.expect('%') });
                } else {
                    const value = parser.parseAttribute();
                    if (value) {
                        op.operands.push(value);
                    }
                }
                parser.accept(',');
            }
        }
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
        if (parser.match('{')) {
            const region = {};
            parser.parseRegion(region);
            if (region.blocks && region.blocks.length > 0 && inductionVars.length > 0) {
                if (!region.blocks[0].arguments) {
                    region.blocks[0].arguments = [];
                }
                for (const iv of inductionVars) {
                    region.blocks[0].arguments.push({ value: iv });
                }
            }
            op.regions.push(region);
        } else {
            return false;
        }
        parser.parseOptionalAttrDict(op.attributes);
        return true;
    }

    _parseInParallelOp(parser, op) {
        // scf.forall.in_parallel { region }
        if (parser.match('{')) {
            const region = op.addRegion();
            parser.parseRegion(region);
        } else {
            return false;
        }
        parser.parseOptionalAttrDict(op.attributes);
        return true;
    }

    _parseSwitchCases(parser, op, args) {
        const caseValues = [];
        while (parser.accept('id', 'case')) {
            if (!parser.match('int')) {
                break;
            }
            const value = parser.parseInteger();
            caseValues.push(value);
            if (parser.match('{')) {
                const region = op.addRegion();
                parser.parseRegion(region);
            } else {
                break;
            }
        }
        if (args && args.length > 0) {
            const casesAttrName = args[0].replace(/^\$/, '');
            op.addAttribute(casesAttrName, caseValues);
        }
    }

    _parseExecuteRegionOp(parser, op) {
        parser.parseOptionalArrowResultTypes(op);
        if (parser.accept('id', 'no_inline')) {
            op.addAttribute('no_inline', true);
        }
        if (parser.match('{')) {
            const region = op.addRegion();
            parser.parseRegion(region);
        }
        parser.parseOptionalAttrDict(op.attributes);
        return true;
    }

};

mlir.ShapeDialect = class extends mlir.Dialect {

    constructor(operations) {
        super(operations, 'shape');
    }

    parseType(parser, dialectName) {
        const typeName = parser.parseKeyword();
        if (!typeName) {
            return null;
        }
        let type = `!${dialectName}.${typeName}`;
        if (typeName === 'value' && parser.match('_')) {
            parser.expect('_');
            const subType = parser.expect('id');
            type += `_${subType}`;
        }
        const simpleTypes = ['shape', 'witness', 'size', 'value_shape'];
        if (simpleTypes.includes(type.substring(7))) { // Remove "!shape." prefix
            return new mlir.Type(type);
        }
        return null;
    }

    parseOperation(parser, opName, op) {
        if (opName === 'shape.func') {
            parser.parseFunctionOp(op, false);
            return true;
        }
        if (opName === 'shape.assuming') {
            return this._parseAssumingOp(parser, op);
        }
        if (opName === 'shape.const_shape') {
            return this._parseConstShapeOp(parser, op);
        }
        if (opName === 'shape.reduce') {
            return this._parseReduceOp(parser, op);
        }
        if (opName === 'shape.function_library') {
            return this._parseFunctionLibraryOp(parser, op);
        }
        return super.parseOperation(parser, opName, op);
    }

    _parseAssumingOp(parser, op) {
        if (!parser.match('%')) {
            return false;
        }
        op.operands.push({ value: parser.expect('%') });
        if (parser.accept('->')) {
            const types = parser.parseFunctionResultTypes();
            parser.resolveOperands(op.results, types);
        }
        if (parser.match('{')) {
            const region = op.addRegion();
            parser.parseRegion(region);
        }
        parser.parseOptionalAttrDict(op.attributes);
        return true;
    }

    _parseConstShapeOp(parser, op) {
        parser.parseOptionalAttrDict(op.attributes);
        const extents = parser.parseAttribute();
        op.addAttribute('shape', extents);
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
            const types = parser.parseFunctionResultTypes();
            parser.resolveOperands(op.results, types);
            for (let i = 1; i < op.operands.length && i - 1 < op.results.length; i++) {
                op.operands[i].type = op.results[i - 1].type;
            }
        }
        if (parser.match('{')) {
            const region = op.addRegion();
            parser.parseRegion(region);
        }
        parser.parseOptionalAttrDict(op.attributes);
        return true;
    }

    _parseFunctionLibraryOp(parser, op) {
        parser.parseSymbolName('sym_name', op.attributes);
        parser.parseOptionalAttrDictWithKeyword(op.attributes);
        if (parser.match('{')) {
            const region = op.addRegion();
            parser.parseRegion(region);
        }
        if (parser.accept('id', 'mapping')) {
            const mapping = parser.parseAttribute();
            op.addAttribute('mapping', mapping);
        }
        return true;
    }
};

mlir.SparseTensorDialect = class extends mlir.Dialect {

    constructor(operations) {
        super(operations, 'sparse_tensor');
        this.registerCustomDirective('LevelRange', this._parseLevelRange.bind(this));
    }

    _parseLevelRange(parser, op, args) {
        const loLvl = parser.parseInteger();
        const hiLvl = parser.accept('id', 'to') ? parser.parseInteger() : loLvl + 1;
        if (args && args.length >= 2) {
            const startAttr = args[0].replace('$', '');
            const endAttr = args[1].replace('$', '');
            op.addAttribute(startAttr, loLvl);
            op.addAttribute(endAttr, hiLvl);
        }
    }

    parseOperation(parser, opName, op) {
        if (opName === 'sparse_tensor.iterate') {
            return this._parseIterateOp(parser, op);
        }
        if (opName === 'sparse_tensor.coiterate') {
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
            const types = parser.parseFunctionResultTypes();
            parser.resolveOperands(op.results, types);
        }
        if (parser.match('{')) {
            const region = op.addRegion();
            parser.parseRegion(region);
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
            const types = parser.parseFunctionResultTypes();
            parser.resolveOperands(op.results, types);
        }
        while (parser.accept('id', 'case')) {
            while (parser.match('%') || parser.match('id')) {
                parser.expect();
                if (!parser.accept(',')) {
                    break;
                }
            }
            if (parser.match('{')) {
                const region = op.addRegion();
                parser.parseRegion(region);
            }
        }
        parser.parseOptionalAttrDict(op.attributes);
        return true;
    }
};

mlir.FuncDialect = class extends mlir.Dialect {

    constructor(operations) {
        super(operations, 'func');
    }

    parseOperation(parser, opName, op) {
        if (opName === 'func.func') {
            parser.parseFunctionOp(op, false);
            return true;
        }
        return super.parseOperation(parser, opName, op);
    }
};

mlir.GpuDialect = class extends mlir.Dialect {

    constructor(operations) {
        super(operations, 'gpu');
        this.registerCustomDirective('AllReduceOperation', this._parseAllReduceOperation.bind(this));
        this.registerCustomDirective('LaunchFuncOperands', this._parseLaunchFuncOperands.bind(this));
        this.registerCustomDirective('AsyncDependencies', this._parseAsyncDependencies.bind(this));
        this.registerCustomDirective('LaunchDimType', this._parseLaunchDimType.bind(this));
        this.registerCustomDirective('OffloadingHandler', this._parseOffloadingHandler.bind(this));
    }

    _parseAllReduceOperation(parser, op, args) {
        const validOps = ['add', 'mul', 'minui', 'minsi', 'minnumf', 'maxui', 'maxsi', 'maxnumf', 'and', 'or', 'xor', 'minimumf', 'maximumf'];
        if (parser.match('id')) {
            const opName = parser.getToken().value;
            if (validOps.includes(opName)) {
                parser.expect('id');
                const attrName = args && args.length > 0 ? args[0].replace(/^\$/, '') : 'op';
                op.addAttribute(attrName, opName);
            }
        }
    }

    _parseLaunchDimType(parser, op, args) {
        if (parser.accept(':')) {
            const dimType = parser.parseType();
            if (args && args.length > 0) {
                for (const result of op.results) {
                    if (!result.type) {
                        result.type = dimType;
                        break;
                    }
                }
                if (args.length >= 4) {
                    for (let i = 2; i < 5 && i < args.length; i++) {
                        const clusterTypeArgName = args[i].replace(/^\$/, '');
                        op.addAttribute(clusterTypeArgName, dimType);
                    }
                }
            }
        }
    }

    parseOperation(parser, opName, op) {
        if (opName === 'gpu.func') {
            parser.parseOptionalVisibilityKeyword(op.attributes);
            parser.parseSymbolName('sym_name', op.attributes);
            const sig = parser.parseFunctionSignatureWithArguments(false);
            const argTypes = sig.arguments.map((a) => a.type);
            const type = new mlir.FunctionType(argTypes, sig.resultTypes);
            op.addAttribute('function_type', new mlir.TypeAttrOf(type));
            const allArgs = [...sig.arguments];
            if (parser.accept('id', 'workgroup')) {
                const workgroupResult = parser.parseFunctionArgumentList(false);
                allArgs.push(...workgroupResult.arguments);
            }
            if (parser.accept('id', 'private')) {
                const privateResult = parser.parseFunctionArgumentList(false);
                allArgs.push(...privateResult.arguments);
            }
            if (parser.match('id', 'kernel')) {
                parser.expect();
                op.addAttribute('gpu.kernel', true);
            }
            parser.parseOptionalAttrDictWithKeyword(op.attributes);
            if (parser.match('{')) {
                const region = op.addRegion();
                parser.parseRegion(region, allArgs);
            }
            return true;
        }
        if (opName === 'gpu.launch') {
            if (parser.accept('id', 'async')) {
                if (op.results.length === 0) {
                    throw new mlir.Error(`Operation '${opName}' needs to be named when marked 'async' ${parser.location()}`);
                }
                op.results[0].type = 'gpu.async.token';
            }
            // Reference: GPUDialect.cpp:500 parseAsyncDependencies uses parseOperandList
            const asyncDeps = parser.parseOperandList('optionalSquare');
            op.operands.push(...asyncDeps);
            if (parser.accept('id', 'clusters')) {
                this._parseSizeAssignment(parser, op);
                parser.expect('id', 'in');
                this._parseSizeAssignment(parser, op);
            }
            parser.expect('id', 'blocks');
            this._parseSizeAssignment(parser, op);
            parser.expect('id', 'in');
            this._parseSizeAssignment(parser, op);
            parser.expect('id', 'threads');
            this._parseSizeAssignment(parser, op);
            parser.expect('id', 'in');
            this._parseSizeAssignment(parser, op);
            if (parser.accept('id', 'dynamic_shared_memory_size')) {
                const operand = {};
                operand.value = parser.expect('%');
                operand.name = 'dynamic_shared_memory_size';
                op.operands.push(operand);
            }
            if (parser.accept('id', 'module')) {
                parser.expect('(');
                const moduleSymbol = parser.expect('@');
                op.addAttribute('module', moduleSymbol);
                parser.expect(')');
            }
            if (parser.accept('id', 'function')) {
                parser.expect('(');
                const funcSymbol = parser.expect('@');
                op.addAttribute('function', funcSymbol);
                parser.expect(')');
            }
            if (parser.accept('id', 'workgroup')) {
                parser.expect('(');
                while (!parser.match(')')) {
                    parser.expect('%');
                    parser.expect(':');
                    parser.parseType();
                    if (!parser.accept(',')) {
                        break;
                    }
                }
                parser.expect(')');
            }
            if (parser.accept('id', 'private')) {
                parser.expect('(');
                while (!parser.match(')')) {
                    parser.expect('%');
                    parser.expect(':');
                    parser.parseType();
                    if (!parser.accept(',')) {
                        break;
                    }
                }
                parser.expect(')');
            }
            if (parser.match('{')) {
                const region = op.addRegion();
                parser.parseRegion(region);
            }
            parser.parseOptionalAttrDict(op.attributes);
            return true;
        }
        if (opName === 'gpu.warp_execute_on_lane_0') {
            return this._parseWarpExecuteOnLane0Op(parser, op);
        }
        return super.parseOperation(parser, opName, op);
    }

    _parseWarpExecuteOnLane0Op(parser, op) {
        parser.expect('(');
        const laneId = parser.expect('%');
        op.operands.push(laneId);
        parser.expect(')');
        parser.expect('[');
        const warpSize = parser.expect('int');
        op.addAttribute('warp_size', parseInt(warpSize, 10));
        parser.expect(']');
        if (parser.accept('id', 'args')) {
            parser.expect('(');
            op.operands.push(...parser.parseArguments());
            if (parser.accept(':')) {
                const types = parser.parseTypeListNoParens();
                parser.resolveOperands(op.operands.slice(1), types);
            }
            parser.expect(')');
        }
        // Reference: parseOptionalArrowTypeList uses parseFunctionResultTypes which handles (type, type) and single type
        if (parser.accept('->')) {
            const types = parser.parseFunctionResultTypes();
            if (op.results.length > 0) {
                parser.resolveOperands(op.results, types);
            } else {
                for (const type of types) {
                    op.results.push({ type });
                }
            }
        }
        if (parser.match('{')) {
            const region = op.addRegion();
            parser.parseRegion(region);
        }
        parser.parseOptionalAttrDict(op.attributes);
        return true;
    }

    _parseSizeAssignment(parser, op) {
        // Parse: (%id, %id, %id) or (%id = %val, %id = %val, %id = %val)
        parser.expect('(');
        while (!parser.match(')')) {
            if (parser.match('%')) {
                parser.expect('%');
                if (parser.accept('=')) {
                    const operand = {};
                    operand.value = parser.expect('%');
                    op.operands.push(operand);
                }
                if (!parser.accept(',')) {
                    break;
                }
            } else {
                break;
            }
        }
        parser.expect(')');
    }

    _parseLaunchFuncOperands(parser, op /*, args */) {
        if (parser.match('id', 'args')) {
            parser.expect();
            parser.expect('(');
            while (!parser.match(')')) {
                const value = parser.expect('%');
                parser.expect(':');
                const type = parser.parseType();
                op.operands.push({ value, type });
                if (!parser.match(')')) {
                    parser.expect(',');
                }
            }
            parser.expect(')');
        }
    }

    _parseAsyncDependencies(parser /*, op, args */) {
        parser.accept('id', 'async');
        if (parser.match('[')) {
            parser.skip('[');
        }
    }

    _parseOffloadingHandler(parser /*, op, args */) {
        if (parser.accept('<')) {
            parser.parseAttribute();
            parser.expect('>');
        }
    }
};

mlir.ArmSMEDialect = class extends mlir.Dialect {

    constructor(operations) {
        super(operations, 'arm_sme');
        this.registerCustomAttribute('ArmSME_TypeSizeAttr', this._parseEnumFlagsAngleBracketComma.bind(this));
        this.registerCustomAttribute('ArmSME_TileSliceLayoutAttr', this._parseEnumFlagsAngleBracketComma.bind(this));
        this.registerCustomAttribute('ArmSME_CombiningKindAttr', this._parseEnumFlagsAngleBracketComma.bind(this));
    }
};

mlir.ArmNeonDialect = class extends mlir.Dialect {

    constructor(operations) {
        super(operations, 'arm_neon');
    }
};

mlir.ArmSVEDialect = class extends mlir.Dialect {

    constructor(operations) {
        super(operations, 'arm_sve');
    }
};

mlir.AMDGPUDialect = class extends mlir.Dialect {

    constructor(operations) {
        super(operations, 'amdgpu');
        this.registerCustomDirective('MNKDimensionList', this._parseMNKDimensionList.bind(this));
    }

    _parseMNKDimensionList(parser, op, args) {
        // Reference: TypeParser.cpp parseDimensionListRanked with allowDynamic=false, withTrailingX=false
        const dimInfo = parser.parseDimensionListRanked(false, false);
        const dims = dimInfo.dimensions;
        if (dims.length >= 3 && args && args.length >= 3) {
            const mAttr = args[0].replace('$', '');
            const nAttr = args[1].replace('$', '');
            const kAttr = args[2].replace('$', '');
            op.addAttribute(mAttr, dims[0]);
            op.addAttribute(nAttr, dims[1]);
            op.addAttribute(kAttr, dims[2]);
        }
    }
};

mlir.NVGPUDialect = class extends mlir.Dialect {

    constructor(operations) {
        super(operations, 'nvgpu');
        this.registerCustomType('NVGPU_TensorMapDescriptor', this._parseTensorMapDescriptor.bind(this));
        this.registerCustomType('NVGPU_WarpgroupAccumulator', this._parseWarpgroupAccumulator.bind(this));
        this.registerCustomType('NVGPU_WarpgroupMatrixDescriptor', this._parseWarpgroupMatrixDescriptor.bind(this));
        this.registerCustomType('NVGPU_MBarrierGroup', this._parseMBarrierGroup.bind(this));
    }

    _parseTensorMapDescriptor(parser) {
        if (parser.match('<')) {
            const content = parser.skip('<');
            return new mlir.Type(`!nvgpu.tensormap.descriptor${content}`);
        }
        return null;
    }

    _parseWarpgroupAccumulator(parser) {
        if (parser.match('<')) {
            const content = parser.skip('<');
            return new mlir.Type(`!nvgpu.warpgroup.accumulator${content}`);
        }
        return null;
    }

    _parseWarpgroupMatrixDescriptor(parser) {
        if (parser.match('<')) {
            const content = parser.skip('<');
            return new mlir.Type(`!nvgpu.warpgroup.descriptor${content}`);
        }
        return null;
    }

    _parseMBarrierGroup(parser) {
        if (parser.match('<')) {
            const content = parser.skip('<');
            return new mlir.Type(`!nvgpu.mbarrier.barrier${content}`);
        }
        return null;
    }
};

mlir.NVVMDialect = class extends mlir.Dialect {

    constructor(operations) {
        super(operations, 'nvvm');
    }

    parseOperation(parser, opName, op) {
        // Helper to parse operand fragment like A[%a0, %a1]
        const parseOperandFragment = (name) => {
            parser.expect('id', name);
            parser.expect('[');
            const operands = [];
            while (!parser.match(']')) {
                if (parser.match('%')) {
                    operands.push({ value: parser.expect('%') });
                }
                if (!parser.accept(',')) {
                    break;
                }
            }
            parser.expect(']');
            return operands;
        };
        if (opName === 'nvvm.mma.sync') {
            // Format: nvvm.mma.sync A[...] B[...] C[...] {attrs} : (types) -> result
            op.operands.push(...parseOperandFragment('A'));
            op.operands.push(...parseOperandFragment('B'));
            op.operands.push(...parseOperandFragment('C'));
            parser.parseOptionalAttrDict(op.attributes);
            if (parser.accept(':')) {
                const funcType = parser.parseFunctionType();
                if (funcType instanceof mlir.FunctionType) {
                    parser.resolveOperands(op.results, funcType.results);
                }
            }
            return true;
        }
        if (opName === 'nvvm.mma.sp.sync' || opName === 'nvvm.mma.sp.block_scale') {
            op.operands.push(...parseOperandFragment('A'));
            op.operands.push(...parseOperandFragment('B'));
            op.operands.push(...parseOperandFragment('C'));
            op.operands.push(...parseOperandFragment('sparseMetadata'));
            op.operands.push(...parseOperandFragment('selector'));
            if (opName === 'nvvm.mma.sp.block_scale') {
                op.operands.push(...parseOperandFragment('scaleA'));
                op.operands.push(...parseOperandFragment('scaleB'));
            }
            parser.parseOptionalAttrDict(op.attributes);
            if (parser.accept(':')) {
                const funcType = parser.parseFunctionType();
                if (funcType instanceof mlir.FunctionType) {
                    parser.resolveOperands(op.results, funcType.results);
                }
            }
            return true;
        }
        if (opName === 'nvvm.mma.block_scale' && this.hasCustomAssemblyFormat(opName) && !this.hasAssemblyFormat(opName)) {
            op.operands.push(...parseOperandFragment('A'));
            op.operands.push(...parseOperandFragment('B'));
            op.operands.push(...parseOperandFragment('C'));
            op.operands.push(...parseOperandFragment('scaleA'));
            op.operands.push(...parseOperandFragment('scaleB'));
            parser.parseOptionalAttrDict(op.attributes);
            if (parser.accept(':')) {
                const funcType = parser.parseFunctionType();
                if (funcType instanceof mlir.FunctionType) {
                    parser.resolveOperands(op.results, funcType.results);
                }
            }
            return true;
        }
        return super.parseOperation(parser, opName, op);
    }
};

mlir.NVWSDialect = class extends mlir.Dialect {

    constructor(operations) {
        super(operations, 'nvws');
        this.registerCustomType('NVWS_ArefType', this._parseArefTypeShorthand.bind(this));
    }

    _parseArefTypeShorthand(parser) {
        if (parser.match('<')) {
            const content = parser.skip('<');
            return new mlir.Type(`!nvws.aref${content}`);
        }
        return parser.parseType();
    }

    parseOperation(parser, opName, op) {
        if (opName === 'nvws.warp_group') {
            parser.parseOptionalAttrDictWithKeyword(op.attributes);
            const numWarps = [];
            let partitionIndex = 0;
            while (parser.accept('id', `partition${partitionIndex}`)) {
                parser.expect('id', 'num_warps');
                parser.expect('(');
                const n = parseInt(parser.expect('int'), 10);
                numWarps.push(n);
                parser.expect(')');
                const region = op.addRegion();
                parser.parseRegion(region);
                partitionIndex++;
            }
            op.addAttribute('numWarps', numWarps);
            return true;
        }
        return super.parseOperation(parser, opName, op);
    }
};

mlir.OpenMPDialect = class extends mlir.Dialect {

    constructor(operations) {
        super(operations, 'omp');
        this.registerCustomDirective('MapClause', this._parseMapClause.bind(this));
        this.registerCustomDirective('CaptureType', this._parseCaptureType.bind(this));
        this.registerCustomDirective('MembersIndex', this._parseMembersIndex.bind(this));
        this.registerCustomDirective('PrivateReductionRegion', this._parsePrivateReductionRegion.bind(this));
        this.registerCustomDirective('PrivateRegion', this._parsePrivateReductionRegion.bind(this));
        this.registerCustomDirective('InReductionPrivateRegion', this._parsePrivateReductionRegion.bind(this));
        this.registerCustomDirective('InReductionPrivateReductionRegion', this._parsePrivateReductionRegion.bind(this));
        this.registerCustomDirective('TaskReductionRegion', this._parsePrivateReductionRegion.bind(this));
        this.registerCustomDirective('UseDeviceAddrUseDevicePtrRegion', this._parsePrivateReductionRegion.bind(this));
        this.registerCustomDirective('TargetOpRegion', this._parseTargetOpRegion.bind(this));
        this.registerCustomDirective('ClauseAttr', this._parseClauseAttr.bind(this));
        this.registerCustomDirective('DependVarList', this._parseDependVarList.bind(this));
        this.registerCustomDirective('LoopTransformClis', this._parseLoopTransformClis.bind(this));
        this.registerCustomDirective('SynchronizationHint', this._parseSynchronizationHint.bind(this));
        this.registerCustomDirective('AlignedClause', this._parseAlignedClause.bind(this));
        this.registerCustomDirective('ScheduleClause', this._parseScheduleClause.bind(this));
        this.registerCustomDirective('AllocateAndAllocator', this._parseAllocateAndAllocator.bind(this));
        this.registerCustomDirective('LinearClause', this._parseLinearClause.bind(this));
        this.registerCustomDirective('OrderClause', this._parseOrderClause.bind(this));
        this.registerCustomDirective('Copyprivate', this._parseCopyprivate.bind(this));
        this.registerCustomDirective('GrainsizeClause', this._parseGranularityClause.bind(this));
        this.registerCustomDirective('NumTasksClause', this._parseGranularityClause.bind(this));
        this.registerCustomAttribute('DataSharingClauseTypeAttr', this._parseDataSharingClauseTypeAttr.bind(this));
        this.registerCustomAttribute('ClauseCancelConstructTypeAttr', this._parseParenthesizedEnumAttr.bind(this));
        this.registerCustomAttribute('ClauseDependAttr', this._parseParenthesizedEnumAttr.bind(this));
        this.registerCustomAttribute('ClauseOrderingIncludeTypeAttr', this._parseParenthesizedEnumAttr.bind(this));
        this.registerCustomAttribute('ClauseTypeAttr', this._parseParenthesizedEnumAttr.bind(this));
        this.registerCustomAttribute('ClauseDistScheduleTypeAttr', this._parseParenthesizedEnumAttr.bind(this));
        this.registerCustomAttribute('OrderModifierAttr', this._parseParenthesizedEnumAttr.bind(this));
    }

    parseOperation(parser, opName, op) {
        if (opName === 'omp.loop_nest') {
            return this._parseLoopNestOp(parser, op);
        }
        if (opName === 'omp.canonical_loop') {
            return this._parseCanonicalLoopOp(parser, op);
        }
        if (opName === 'omp.unroll_heuristic') {
            return this._parseUnrollHeuristicOp(parser, op);
        }
        if (opName === 'omp.target_allocmem' && this.hasCustomAssemblyFormat(opName) && !this.hasAssemblyFormat(opName)) {
            const device = parser.parseOperand();
            op.operands.push(device);
            parser.expect(':');
            const deviceType = parser.parseType();
            device.type = deviceType.toString();
            parser.expect(',');
            const inType = parser.parseType();
            op.addAttribute('in_type', { value: inType, type: 'type' });
            const typeparams = [];
            if (parser.accept('(')) {
                while (!parser.match(')')) {
                    typeparams.push(parser.parseOperand());
                    if (!parser.match(')')) {
                        parser.accept(',');
                    }
                }
                parser.expect(':');
                const types = parser.parseTypeList();
                parser.resolveOperands(typeparams, types);
                parser.expect(')');
                op.operands.push(...typeparams);
            }
            const shape = [];
            while (parser.accept(',')) {
                shape.push(parser.parseOperand());
            }
            for (const s of shape) {
                s.type = 'index';
            }
            op.operands.push(...shape);
            parser.parseOptionalAttrDict(op.attributes);
            op.addAttribute('operandSegmentSizes', [1, typeparams.length, shape.length]);
            op.results.push({ type: 'i64' });
            return true;
        }
        if (opName === 'omp.target_freemem' && this.hasCustomAssemblyFormat(opName) && !this.hasAssemblyFormat(opName)) {
            const device = parser.parseOperand();
            op.operands.push(device);
            parser.expect(',');
            const ptr = parser.parseOperand();
            op.operands.push(ptr);
            parser.expect(':');
            const deviceType = parser.parseType();
            device.type = deviceType.toString();
            parser.expect(',');
            const ptrType = parser.parseType();
            ptr.type = ptrType.toString();
            return true;
        }
        return super.parseOperation(parser, opName, op);
    }

    _parseCanonicalLoopOp(parser, op) {
        if (parser.accept('(')) {
            const cliOperand = parser.expect('%');
            op.operands.push(cliOperand);
            parser.expect(')');
        }
        const inductionVar = parser.expect('%');
        parser.expect(':');
        const ivType = parser.parseType();
        parser.expect('id', 'in');
        parser.expect('id', 'range');
        parser.expect('(');
        parser.expect('%');
        parser.expect(')');
        if (parser.match('{')) {
            const region = {};
            parser.parseRegion(region);
            if (region.blocks && region.blocks.length > 0) {
                if (!region.blocks[0].arguments) {
                    region.blocks[0].arguments = [];
                }
                region.blocks[0].arguments.push({ value: inductionVar, type: ivType });
            }
            op.regions.push(region);
        }
        parser.parseOptionalAttrDict(op.attributes);
        return true;
    }

    _parseUnrollHeuristicOp(parser, op) {
        parser.expect('(');
        const applyee = parser.expect('%');
        op.operands.push(applyee);
        parser.expect(')');
        if (parser.accept('->')) {
            parser.expect('(');
            parser.expect(')');
        }
        parser.parseOptionalAttrDict(op.attributes);
        return true;
    }

    _parseLoopNestOp(parser, op) {
        if (parser.accept('(')) {
            while (!parser.match(')')) {
                parser.expect('%');
                if (!parser.accept(',')) {
                    break;
                }
            }
            parser.expect(')');
        }
        if (parser.accept(':')) {
            while (!parser.match('=')) {
                parser.parseType();
                if (!parser.accept(',')) {
                    break;
                }
            }
        }
        if (parser.accept('=')) {
            if (parser.accept('(')) {
                op.operands = parser.parseArguments();
                parser.expect(')');
            }
            if (parser.accept('id', 'to')) {
                if (parser.accept('(')) {
                    while (!parser.match(')')) {
                        parser.expect('%');
                        if (!parser.accept(',')) {
                            break;
                        }
                    }
                    parser.expect(')');
                }
            }
            parser.accept('id', 'inclusive');
            if (parser.accept('id', 'step')) {
                if (parser.accept('(')) {
                    while (!parser.match(')')) {
                        parser.expect('%');
                        if (!parser.accept(',')) {
                            break;
                        }
                    }
                    parser.expect(')');
                }
            }
        }
        // Parse optional 'collapse(N)'
        if (parser.accept('id', 'collapse')) {
            parser.expect('(');
            const value = parser.expect('int');
            op.addAttribute('collapse_num_loops', parseInt(value, 10));
            parser.expect(')');
        }
        // Parse optional 'tiles(N, ...)'
        if (parser.accept('id', 'tiles')) {
            parser.expect('(');
            const tiles = [];
            while (!parser.match(')')) {
                tiles.push(parseInt(parser.expect('int'), 10));
                if (!parser.accept(',')) {
                    break;
                }
            }
            parser.expect(')');
            op.addAttribute('tile_sizes', tiles);
        }
        // Parse region BEFORE attr-dict (matches reference impl)
        if (parser.match('{')) {
            const region = op.addRegion();
            parser.parseRegion(region);
        }
        parser.parseOptionalAttrDict(op.attributes);
        return true;
    }

    _parseParenthesizedEnumAttr(parser) {
        if (parser.accept('(')) {
            const value = parser.parseKeyword();
            parser.expect(')');
            return new mlir.TypedAttr(value, null);
        }
        return null;
    }

    _parseOrderClause(parser, op) {
        const orderModifiers = ['reproducible', 'unconstrained'];
        const orderKinds = ['concurrent'];
        let orderMod = null;
        let orderKind = null;
        const keyword = parser.parseKeyword();
        if (orderModifiers.includes(keyword)) {
            orderMod = keyword;
            parser.expect(':');
            orderKind = parser.parseKeyword();
        } else if (orderKinds.includes(keyword)) {
            orderKind = keyword;
        }
        if (orderKind) {
            op.addAttribute('order_kind', orderKind);
        }
        if (orderMod) {
            op.addAttribute('order_mod', orderMod);
        }
    }

    _parseLinearClause(parser, op) {
        const linearVars = [];
        const linearStepVars = [];
        do {
            if (!parser.match('%')) {
                break;
            }
            const linearVar = parser.expect('%');
            parser.expect('=');
            const stepVar = parser.expect('%');
            parser.expect(':');
            const type = parser.parseType();
            linearVars.push({ value: linearVar, type });
            linearStepVars.push({ value: stepVar });
        } while (parser.accept(','));
        for (const v of linearVars) {
            op.operands.push(v);
        }
        for (const v of linearStepVars) {
            op.operands.push(v);
        }
    }

    _parseCopyprivate(parser, op, args) {
        const copyprivateVars = [];
        const copyprivateSyms = [];
        do {
            const operand = parser.expect('%');
            parser.expect('->');
            const sym = parser.expect('@');
            parser.expect(':');
            const type = parser.parseType();
            copyprivateVars.push({ value: operand, type });
            copyprivateSyms.push(sym);
        } while (parser.accept(','));
        for (const v of copyprivateVars) {
            op.operands.push(v);
        }
        if (args && args.length > 2) {
            const symsAttr = args[2].replace(/^\$/, '');
            op.addAttribute(symsAttr, copyprivateSyms);
        }
    }

    _parseGranularityClause(parser, op, args) {
        let modifier = null;
        if (parser.match('id') && !parser.match('%')) {
            modifier = parser.expect('id');
            parser.expect(',');
        }
        const operand = parser.expect('%');
        parser.expect(':');
        const type = parser.parseType();
        op.operands.push({ value: operand, type });
        if (args && args.length > 0 && modifier) {
            const modAttr = args[0].replace(/^\$/, '');
            op.addAttribute(modAttr, modifier);
        }
    }

    _parseAlignedClause(parser, op) {
        const alignedVars = [];
        const alignments = [];
        do {
            if (!parser.match('%')) {
                break;
            }
            const operand = parser.expect('%');
            parser.expect(':');
            const type = parser.parseType();
            parser.expect('->');
            const alignment = parser.parseAttribute();
            alignedVars.push({ value: operand, type });
            alignments.push(alignment);
        } while (parser.accept(','));
        for (const v of alignedVars) {
            op.operands.push(v);
        }
        if (alignments.length > 0) {
            op.addAttribute('alignments', alignments);
        }
    }

    _parseScheduleClause(parser, op) {
        const scheduleKinds = ['static', 'dynamic', 'guided', 'auto', 'runtime', 'distribute'];
        let scheduleKind = null;
        for (const kind of scheduleKinds) {
            if (parser.accept('id', kind)) {
                scheduleKind = kind;
                break;
            }
        }
        if (scheduleKind) {
            op.addAttribute('schedule_kind', scheduleKind);
        }
        if (parser.accept('=')) {
            if (parser.match('%')) {
                const chunk = parser.expect('%');
                op.operands.push({ value: chunk });
            }
            if (parser.accept(':')) {
                const type = parser.parseType();
                if (op.operands.length > 0) {
                    op.operands[op.operands.length - 1].type = type;
                }
            }
        }
        const modifiers = [];
        while (parser.accept(',')) {
            const mod = parser.parseKeyword();
            if (mod) {
                modifiers.push(mod);
            }
        }
        if (modifiers.length > 0) {
            op.addAttribute('schedule_modifiers', modifiers);
        }
    }

    _parseAllocateAndAllocator(parser, op) {
        const allocatorVars = [];
        const allocateVars = [];
        do {
            if (!parser.match('%')) {
                break;
            }
            const allocator = parser.expect('%');
            parser.expect(':');
            const allocatorType = parser.parseType();
            parser.expect('->');
            const allocate = parser.expect('%');
            parser.expect(':');
            const allocateType = parser.parseType();
            allocatorVars.push({ value: allocator, type: allocatorType });
            allocateVars.push({ value: allocate, type: allocateType });
        } while (parser.accept(','));
        for (const v of allocatorVars) {
            op.operands.push(v);
        }
        for (const v of allocateVars) {
            op.operands.push(v);
        }
    }

    _parseSynchronizationHint(parser, op, args) {
        const hintAttr = args && args.length > 0 ? args[0].replace(/^\$/, '') : 'hint';
        if (parser.accept('id', 'none')) {
            op.addAttribute(hintAttr, 0);
            return;
        }
        let hint = 0;
        const hints = [];
        while (parser.match('id')) {
            const keyword = parser.expect('id');
            hints.push(keyword);
            if (keyword === 'uncontended') {
                hint |= 1;
            } else if (keyword === 'contended') {
                hint |= 2;
            } else if (keyword === 'nonspeculative') {
                hint |= 4;
            } else if (keyword === 'speculative') {
                hint |= 8;
            }
            if (!parser.accept(',')) {
                break;
            }
        }
        op.addAttribute(hintAttr, hint);
    }

    _parseClauseAttr(parser, op, args) {
        // Reference: OpenMPDialect.cpp parseClauseAttr
        // Parses a keyword (enum value) and converts to attribute
        if (parser.match('id')) {
            const enumValue = parser.expect('id');
            if (args && args.length > 0) {
                const attrName = args[0].replace(/^\$/, '');
                op.addAttribute(attrName, enumValue);
            }
        } else if (parser.match('{')) {
            parser.skip('{');
        }
    }

    _parseTargetOpRegion(parser, op, args) {
        const unitAttrKeywords = ['nowait', 'bare'];
        for (const kw of unitAttrKeywords) {
            if (parser.accept('id', kw)) {
                op.addAttribute(kw, true);
            }
        }
        if (parser.accept('id', 'depend')) {
            parser.skip('(');
        }
        const singleValueKeywords = ['device', 'if', 'thread_limit'];
        for (const kw of singleValueKeywords) {
            if (parser.accept('id', kw)) {
                parser.expect('(');
                if (parser.match('%')) {
                    const operand = parser.expect('%');
                    op.operands.push({ name: kw, value: operand });
                }
                if (parser.accept(':')) {
                    parser.parseType();
                }
                parser.expect(')');
            }
        }
        if (parser.accept('id', 'is_device_ptr')) {
            parser.expect('(');
            while (!parser.match(')') && !parser.match(':')) {
                if (parser.match('%')) {
                    parser.expect('%');
                }
                if (!parser.accept(',')) {
                    break;
                }
            }
            if (parser.accept(':')) {
                while (!parser.match(')')) {
                    parser.parseType();
                    if (!parser.accept(',')) {
                        break;
                    }
                }
            }
            parser.expect(')');
        }
        const keywords = ['has_device_addr', 'host_eval', 'in_reduction', 'map_entries', 'private', 'reduction', 'task_reduction', 'use_device_addr', 'use_device_ptr'];
        let progress = true;
        while (progress) {
            progress = false;
            // Handle private_barrier unit attribute
            if (parser.accept('id', 'private_barrier')) {
                op.addAttribute('private_needs_barrier', true);
                progress = true;
                continue;
            }
            // Handle list clauses
            if (keywords.some((kw) => parser.match('id', kw))) {
                parser.expect('id');
                progress = true;
                if (parser.accept('(')) {
                    while (!parser.match(')') && !parser.match(':')) {
                        parser.accept('id', 'byref');
                        if (parser.match('@')) {
                            parser.expect('@');
                        }
                        if (parser.match('%')) {
                            parser.expect('%');
                        }
                        if (parser.accept('->')) {
                            if (parser.match('%')) {
                                parser.expect('%');
                            }
                        }
                        if (parser.accept('[')) {
                            parser.expect('id', 'map_idx');
                            parser.expect('=');
                            parser.expect('int');
                            parser.expect(']');
                        }
                        if (!parser.accept(',') || parser.match(':')) {
                            break;
                        }
                    }
                    if (parser.accept(':')) {
                        while (!parser.match(')')) {
                            parser.parseType();
                            if (!parser.accept(',')) {
                                break;
                            }
                        }
                    }
                    parser.expect(')');
                }
            }
        }
        if (!parser.match('{')) {
            return;
        }
        const region = {};
        parser.parseRegion(region);
        if (args && args.length > 0) {
            const regionArgName = args[0].replace(/^\$/, '');
            if (regionArgName === 'region') {
                op.regions.push(region);
            }
        } else {
            op.regions.push(region);
        }
    }

    _parseMapClause(parser, op, args) {
        const mapFlags = [];
        do {
            if (parser.match('id')) {
                const flag = parser.expect('id');
                mapFlags.push(flag);
            }
        } while (parser.accept(','));

        if (args && args.length > 0) {
            const attrName = args[0].replace(/^\$/, '');
            op.addAttribute(attrName, mapFlags.join(', '));
        }
    }

    _parseCaptureType(parser, op, args) {
        if (parser.match('id')) {
            const captureType = parser.expect('id');
            if (args && args.length > 0) {
                const attrName = args[0].replace(/^\$/, '');
                op.addAttribute(attrName, captureType);
            }
        }
    }

    _parseMembersIndex(parser, op, args) {
        const memberIndices = [];
        do {
            if (parser.accept('[')) {
                const indices = [];
                do {
                    if (parser.match('int')) {
                        const idx = parser.expect('int');
                        indices.push(idx);
                    }
                } while (parser.accept(','));
                parser.expect(']');
                memberIndices.push(indices);
            }
        } while (parser.accept(','));

        if (args && args.length > 0 && memberIndices.length > 0) {
            const attrName = args[0].replace(/^\$/, '');
            op.addAttribute(attrName, memberIndices);
        }
    }

    _parsePrivateReductionRegion(parser, op, args) {
        // Parse optional clauses that appear before the region (oilist in assembly format)
        // Reference: OpenMPOpBase.td clausesAssemblyFormat
        const singleValueClauses = ['if', 'num_threads', 'thread_limit', 'device', 'safelen', 'simdlen', 'priority', 'grainsize', 'num_tasks', 'final', 'filter'];
        const enumClauses = ['proc_bind', 'order', 'schedule', 'dist_schedule', 'memory_order', 'hint'];
        const listClauses = ['private', 'reduction', 'in_reduction', 'task_reduction', 'copyin', 'copyprivate', 'firstprivate', 'lastprivate', 'shared', 'linear', 'aligned', 'nontemporal', 'inclusive', 'exclusive', 'allocate', 'depend'];
        const unitClauses = ['nowait', 'untied', 'mergeable', 'nogroup', 'simd', 'threads', 'seq_cst', 'acq_rel', 'acquire', 'release', 'relaxed', 'private_barrier'];
        let progress = true;
        while (progress) {
            progress = false;
            // Handle single-value clauses: keyword(value : type)
            for (const kw of singleValueClauses) {
                if (parser.accept('id', kw)) {
                    progress = true;
                    parser.expect('(');
                    if (parser.match('%')) {
                        const operand = parser.expect('%');
                        op.operands.push({ name: kw, value: operand });
                    } else if (parser.match('int')) {
                        const value = parser.expect('int');
                        op.addAttribute(kw, value);
                    }
                    if (parser.accept(':')) {
                        parser.parseType();
                    }
                    parser.expect(')');
                }
            }
            // Handle enum clauses: keyword(enum_value)
            for (const kw of enumClauses) {
                if (parser.accept('id', kw)) {
                    progress = true;
                    parser.expect('(');
                    const value = parser.expect('id');
                    op.addAttribute(kw, value);
                    // Handle modifier syntax like schedule(static, value)
                    while (parser.accept(',')) {
                        if (parser.match('%')) {
                            const operand = parser.expect('%');
                            op.operands.push({ name: `${kw}_operand`, value: operand });
                        } else if (parser.match('id')) {
                            parser.expect('id');
                        }
                        if (parser.accept(':')) {
                            parser.parseType();
                        }
                    }
                    parser.expect(')');
                }
            }
            // Handle list clauses: keyword(syms %vals -> %new_vals : types) or keyword(@sym %val : type, ...)
            for (const kw of listClauses) {
                if (parser.accept('id', kw)) {
                    progress = true;
                    if (parser.accept('(')) {
                        if (parser.accept('id', 'mod')) {
                            parser.expect(':');
                            parser.expect('id');
                            parser.expect(',');
                        }
                        while (!parser.match(')') && !parser.match(':')) {
                            parser.accept('id', 'byref');
                            if (parser.match('@')) {
                                parser.expect('@');
                            }
                            if (parser.match('%')) {
                                const operand = parser.expect('%');
                                op.operands.push({ name: kw, value: operand });
                            }
                            if (parser.accept('->')) {
                                if (parser.match('%')) {
                                    parser.expect('%');
                                }
                            }
                            if (parser.accept('[')) {
                                parser.expect('id', 'map_idx');
                                parser.expect('=');
                                parser.expect('int');
                                parser.expect(']');
                            }
                            if (!parser.accept(',') || parser.match(':')) {
                                break;
                            }
                        }
                        if (parser.accept(':')) {
                            while (!parser.match(')')) {
                                parser.parseType();
                                if (!parser.accept(',')) {
                                    break;
                                }
                            }
                        }
                        parser.expect(')');
                    }
                }
            }
            // Handle unit clauses (boolean flags)
            for (const kw of unitClauses) {
                if (parser.accept('id', kw)) {
                    progress = true;
                    // private_barrier maps to private_needs_barrier attribute
                    const attrName = kw === 'private_barrier' ? 'private_needs_barrier' : kw;
                    op.addAttribute(attrName, true);
                }
            }
            // Handle map_entries clause: map_entries(%vars : types)
            if (parser.accept('id', 'map_entries')) {
                progress = true;
                parser.expect('(');
                while (!parser.match(')') && !parser.match(':')) {
                    if (parser.match('%')) {
                        const operand = parser.expect('%');
                        op.operands.push({ name: 'map_vars', value: operand });
                    }
                    if (!parser.accept(',') || parser.match(':')) {
                        break;
                    }
                }
                if (parser.accept(':')) {
                    while (!parser.match(')')) {
                        parser.parseType();
                        if (!parser.accept(',')) {
                            break;
                        }
                    }
                }
                parser.expect(')');
            }
            // Handle num_teams clause: num_teams(lower : type to upper : type) or num_teams(to upper : type)
            if (parser.accept('id', 'num_teams')) {
                progress = true;
                parser.expect('(');
                if (parser.accept('id', 'to')) {
                    if (parser.match('%')) {
                        const upper = parser.expect('%');
                        op.operands.push({ name: 'num_teams_upper', value: upper });
                    }
                    if (parser.accept(':')) {
                        parser.parseType();
                    }
                } else if (parser.match('%')) {
                    const lower = parser.expect('%');
                    op.operands.push({ name: 'num_teams_lower', value: lower });
                    if (parser.accept(':')) {
                        parser.parseType();
                    }
                    parser.expect('id', 'to');
                    if (parser.match('%')) {
                        const upper = parser.expect('%');
                        op.operands.push({ name: 'num_teams_upper', value: upper });
                    }
                    if (parser.accept(':')) {
                        parser.parseType();
                    }
                }
                parser.expect(')');
            }
            // Handle use_device_addr/use_device_ptr clauses: keyword(%var -> %arg : type, ...)
            for (const kw of ['use_device_addr', 'use_device_ptr', 'has_device_addr', 'host_eval']) {
                if (parser.accept('id', kw)) {
                    progress = true;
                    parser.expect('(');
                    while (!parser.match(')') && !parser.match(':')) {
                        if (parser.match('%')) {
                            const operand = parser.expect('%');
                            op.operands.push({ name: kw, value: operand });
                        }
                        if (parser.accept('->')) {
                            if (parser.match('%')) {
                                parser.expect('%');
                            }
                        }
                        if (!parser.accept(',') || parser.match(':')) {
                            break;
                        }
                    }
                    if (parser.accept(':')) {
                        while (!parser.match(')')) {
                            parser.parseType();
                            if (!parser.accept(',')) {
                                break;
                            }
                        }
                    }
                    parser.expect(')');
                }
            }
        }
        if (parser.match('{')) {
            const region = {};
            parser.parseRegion(region);
            if (args && args.length > 0) {
                const regionArgName = args[0].replace(/^\$/, '');
                if (regionArgName === 'region') {
                    op.regions.push(region);
                }
            } else {
                op.regions.push(region);
            }
        }
    }

    _parseDataSharingClauseTypeAttr(parser) {
        if (parser.accept('{')) {
            parser.expect('id', 'type');
            parser.expect('=');
            const value = parser.expect('id');
            parser.expect('}');
            return { value };
        }
        return null;
    }

    // Reference: OpenMPDialect.cpp parseDependVarList
    // Format: depend-kind -> %var : type (, depend-kind -> %var : type)*
    _parseDependVarList(parser, op, args) {
        const dependVars = [];
        const dependKinds = [];
        do {
            const keyword = parser.expect('id');
            dependKinds.push(keyword);
            parser.expect('->');
            const operand = parser.expect('%');
            dependVars.push(operand);
            parser.expect(':');
            parser.parseType();
        } while (parser.accept(','));
        if (args && args.length > 0) {
            const operandAttr = args[0].replace('$', '');
            // depend_vars are SSA operands - add as operands
            for (const value of dependVars) {
                op.operands.push({ name: operandAttr, value });
            }
        }
        if (args && args.length > 2) {
            const kindAttr = args[2].replace('$', '');
            op.addAttribute(kindAttr, dependKinds);
        }
    }

    // Reference: OpenMPDialect.cpp parseLoopTransformClis
    // Syntax 1: (generatees) <- (applyees) - generatees present (no leading <)
    // Syntax 2: <- (applyees) - generatees omitted (starts with <-)
    _parseLoopTransformClis(parser, op, args) {
        const generatees = [];
        const applyees = [];
        // Check if starts with '<' (syntax 2, no generatees) or '(' (syntax 1, has generatees)
        if (!parser.accept('<')) {
            // Syntax 1: generatees present, parse (generatees) first
            parser.expect('(');
            while (!parser.match(')')) {
                if (parser.match('%')) {
                    generatees.push({ value: parser.expect('%') });
                }
                if (!parser.accept(',')) {
                    break;
                }
            }
            parser.expect(')');
            // Now parse '<' for the '<-' arrow
            parser.expect('<');
        }
        // '<' already consumed, now parse '-' to complete '<-'
        parser.expect('keyword', '-');
        // Parse applyees list in parens
        parser.expect('(');
        while (!parser.match(')')) {
            if (parser.match('%')) {
                applyees.push({ value: parser.expect('%') });
            }
            if (!parser.accept(',')) {
                break;
            }
        }
        parser.expect(')');
        if (args && args.length >= 1) {
            for (const g of generatees) {
                op.operands.push(g);
            }
        }
        if (args && args.length >= 2) {
            for (const a of applyees) {
                op.operands.push(a);
            }
        }
    }
};

mlir.LLVMDialect = class extends mlir.Dialect {

    constructor(operations, name = 'llvm') {
        super(operations, name);
        this.registerCustomDirective('GEPIndices', this._parseGEPIndices.bind(this));
        this.registerCustomDirective('IndirectBrOpSucessors', this._parseIndirectBrOpSucessors.bind(this));
        this.registerCustomDirective('InsertExtractValueElementType', this._parseInsertExtractValueElementType.bind(this));
        this.registerCustomDirective('LLVMLinkage', this._parseLLVMLinkage.bind(this));
        this.registerCustomDirective('OpBundles', this._parseOpBundles.bind(this));
        this.registerCustomDirective('ShuffleType', this._parseShuffleType.bind(this));
        this.registerCustomDirective('SwitchOpCases', this._parseSwitchOpCases.bind(this));
        this.registerCustomAttribute('LLVM_IntegerOverflowFlagsProp', this._parseLLVMIntegerOverflowFlagsProp.bind(this));
        this.registerCustomAttribute('GEPNoWrapFlagsProp', this._parseGEPNoWrapFlagsProp.bind(this));
        this.registerCustomAttribute('LLVM_BlockAddressAttr', this._parseLLVMBlockAddressAttr.bind(this));
        this.registerCustomAttribute('LLVM_BlockTagAttr', this._parseLLVMBlockTagAttr.bind(this));
        this.registerCustomType('LLVM_AnyPointer', this._parseLLVMPointerType.bind(this));
        this.registerCustomType('LLVM_PointerInAddressSpace', this._parseLLVMPointerType.bind(this));
    }

    _parseLLVMIntegerOverflowFlagsProp(parser) {
        if (parser.accept('id', 'overflow')) {
            return this._parseEnumFlagsAngleBracketComma(parser, { values: ['wrap', 'nuw', 'nsw'] });
        }
        return null;
    }

    _parseGEPNoWrapFlagsProp(parser, type) {
        if (type.values.includes(parser.getToken().value)) {
            return this._parseEnumFlags(parser, type, '|');
        }
        return null;
    }

    _parseLLVMBlockAddressAttr(parser) {
        // Parse: <function = @fn, tag = <id = N>>
        if (!parser.match('<')) {
            return null;
        }
        const content = parser.skip('<');
        return { blockaddress: content };
    }

    _parseLLVMBlockTagAttr(parser) {
        // Parse: <id = N>
        if (!parser.match('<')) {
            return null;
        }
        const content = parser.skip('<');
        return { blocktag: content };
    }

    _parseLLVMPointerType(parser) {
        // Try parsing shorthand syntax first: <addressSpace>
        if (parser.match('<')) {
            const content = parser.skip('<');
            // content includes delimiters like "<1>", extract the inner value
            const inner = content.startsWith('<') && content.endsWith('>') ? content.slice(1, -1) : content;
            // If inner content is a number, it's the address space
            if (/^\d+$/.test(inner)) {
                return new mlir.Type(`!llvm.ptr<${inner}>`);
            }
            // Otherwise, it might be something else - fall through to standard parsing
        }
        // Fall back to standard type parsing (!llvm.ptr or !llvm.ptr<addressSpace>)
        return parser.parseType();
    }

    _parseInsertExtractValueElementType(/* parser, op, args */) {
    }

    _parseLLVMLinkage(parser, op /*, args */) {
        if (parser.match('id')) {
            const linkage = parser.expect('id');
            op.addAttribute('linkage', linkage);
        }
    }

    _parseOpBundles(parser, op /*, args */) {
        // Parse operation bundles: [] or ["tag"()] or ["tag"(%0, %1 : i32, i32), ...]
        // Returns: null if not present, true if success, throws on failure
        // args[0] = $op_bundle_operands - operands for bundles
        // args[1] = type($op_bundle_operands) - types
        // args[2] = $op_bundle_tags - tags attribute

        // Check if '[' is present - if not, optional is not present
        if (!parser.accept('[')) {
            return null; // Not present (equivalent to std::nullopt)
        }

        // Empty bundle list
        if (parser.accept(']')) {
            return true; // Success
        }

        const opBundles = [];
        do {
            const tag = parser.expect('string');
            parser.expect('(');
            const bundleOperands = [];
            if (!parser.match(')')) {
                do {
                    bundleOperands.push(parser.parseAttribute());
                } while (parser.accept(','));
                parser.expect(':');
                // Parse types for bundle operands
                do {
                    parser.parseType();
                } while (parser.accept(','));
            }
            parser.expect(')');
            opBundles.push({ tag, operands: bundleOperands });
        } while (parser.accept(','));
        parser.expect(']');

        if (opBundles.length > 0) {
            op.addAttribute('op_bundle_tags', opBundles);
        }
        return true; // Success
    }

    _parseShuffleType(/* parser, op, args */) {
        // custom<ShuffleType>(ref(type($v1)), type($res), ref($mask))
        // This directive doesn't parse anything - it computes the result type
        // based on the input vector type and mask length.
        // args[0] = ref(type($v1)) - input type
        // args[1] = type($res) - result type to populate
        // args[2] = ref($mask) - mask attribute

        // The result type has the same element type as input, but length = mask.size
        // Since we can't easily compute this in JS without full type info,
        // we do nothing here and let the standard assembly format handle it.
        // The reference impl computes: getVectorType(v1Type.elementType, mask.size())
    }

    // Parse switch operation cases
    // Format: `[` (case (`,` case )* )? `]`
    // Where case: integer `:` bb-id (`(` ssa-use-and-type-list `)`)?
    _parseSwitchOpCases(parser, op /*, args */) {
        // args[0] is ref(type($value)) - the flag type
        // args[1] is $case_values - attribute to populate
        // args[2] is $caseDestinations - successors array
        // args[3] is $caseOperands - operands for each case
        // args[4] is type($caseOperands) - types for case operands
        if (!parser.accept('[')) {
            return;
        }
        // Check for empty case list
        if (parser.accept(']')) {
            return;
        }
        const caseValues = [];
        const caseDestinations = [];
        const caseOperands = [];
        while (!parser.match(']') && !parser.match('eof')) {
            // Handle negative case values: -1, -2, etc.
            let sign = 1;
            if (parser.accept('keyword', '-')) {
                sign = -1;
            }
            if (!parser.match('int') && !parser.match('number')) {
                throw new mlir.Error(`Expected integer case value at ${parser.location()}`);
            }
            const value = sign * parseInt(parser.expect(), 10);
            caseValues.push(value);
            // Parse colon
            parser.expect(':');
            // Parse successor block (starts with ^)
            const successor = parser.expect('^');
            caseDestinations.push(successor);
            // Parse optional operands with types: (%operand : type, ...)
            if (parser.accept('(')) {
                const operands = [];
                while (!parser.match(')') && !parser.match(':') && !parser.match('eof')) {
                    if (parser.match('%')) {
                        const operand = parser.expect('%');
                        operands.push({ name: operand });
                        if (!parser.accept(',')) {
                            break;
                        }
                    } else {
                        break;
                    }
                }
                // Parse types after colon if present
                if (parser.accept(':')) {
                    let idx = 0;
                    while (!parser.match(')') && idx < operands.length) {
                        const type = parser.parseType();
                        if (operands[idx]) {
                            operands[idx].type = type;
                        }
                        idx++;
                        parser.accept(',');
                    }
                }
                parser.expect(')');
                caseOperands.push(operands);
            } else {
                caseOperands.push([]);
            }
            // Check for comma or end of list
            if (!parser.accept(',')) {
                break;
            }
        }
        parser.expect(']');
        // Populate operation with parsed data
        if (caseValues.length > 0) {
            op.addAttribute('case_values', caseValues);
        }
        if (caseDestinations.length > 0) {
            if (!op.successors) {
                op.successors = [];
            }
            // Add case destinations (default destination is already added)
            for (const dest of caseDestinations) {
                op.successors.push({ name: dest });
            }
        }
        // Note: caseOperands handling would require more complex logic
        // to properly associate operands with their successors
    }

    parseType(parser, dialectName) {
        const typeName = parser.parseKeyword();
        if (!typeName) {
            return null;
        }
        let type = `!${dialectName}.${typeName}`;
        if (parser.match('<')) {
            const content = parser.skip('<');
            type += content;
        }
        return new mlir.Type(type);
    }

    parseOperation(parser, opName, op) {
        if (opName === 'llvm.func') {
            return this._parseLLVMFuncOp(parser, op);
        }
        if (opName === 'llvm.mlir.global') {
            return this._parseLLVMGlobalOp(parser, op);
        }
        if (opName === 'llvm.mlir.alias') {
            return this._parseLLVMAliasOp(parser, op);
        }
        if (opName === 'llvm.alloca') {
            return this._parseLLVMAllocaOp(parser, op);
        }
        if (opName === 'llvm.call') {
            return this._parseLLVMCallOp(parser, op);
        }
        if (opName === 'llvm.call_intrinsic') {
            return this._parseLLVMCallIntrinsicOp(parser, op);
        }
        if (opName === 'llvm.invoke') {
            return this._parseLLVMInvokeOp(parser, op);
        }
        if (opName === 'llvm.landingpad') {
            return this._parseLLVMLandingpadOp(parser, op);
        }
        if (opName === 'llvm.icmp' || opName === 'llvm.fcmp') {
            return this._parseLLVMCmpOp(parser, op);
        }
        if (opName.startsWith('llvm.intr.')) {
            // Check if this intrinsic has assembly format - if so, use standard parsing
            if (this.hasAssemblyFormat(opName)) {
                return super.parseOperation(parser, opName, op);
            }
            return this._parseLLVMIntrinsicOp(parser, op);
        }
        return super.parseOperation(parser, opName, op);
    }

    _parseLLVMGlobalOp(parser, op) {
        const linkageKeywords = ['external', 'available_externally', 'linkonce', 'linkonce_odr', 'weak', 'weak_odr', 'appending', 'internal', 'private', 'extern_weak', 'common'];
        if (parser.match('id') && linkageKeywords.includes(parser.getToken().value)) {
            op.addAttribute('linkage', parser.expect('id'));
        }
        const visibilityKeywords = ['default', 'hidden', 'protected'];
        if (parser.match('id') && visibilityKeywords.includes(parser.getToken().value)) {
            op.addAttribute('visibility_', parser.expect('id'));
        }
        if (parser.accept('id', 'thread_local')) {
            op.addAttribute('thread_local_', true);
        }
        const unnamedAddrKeywords = ['unnamed_addr', 'local_unnamed_addr'];
        if (parser.match('id') && unnamedAddrKeywords.includes(parser.getToken().value)) {
            op.addAttribute('unnamed_addr', parser.expect('id'));
        }
        if (parser.accept('id', 'constant')) {
            op.addAttribute('constant', true);
        }
        if (parser.match('@')) {
            parser.parseSymbolName('sym_name', op.attributes);
        }
        parser.expect('(');
        if (!parser.match(')')) {
            const value = parser.parseAttribute();
            if (parser.accept(':')) {
                parser.parseType();
            }
            op.addAttribute('value', value);
        }
        parser.expect(')');
        if (parser.accept('id', 'comdat')) {
            parser.expect('(');
            const comdat = parser.expect('@');
            parser.expect(')');
            op.addAttribute('comdat', comdat);
        }
        if (parser.match('{')) {
            parser.parseAttributeDict(op.attributes);
        }
        if (parser.accept(':')) {
            const type = parser.parseType();
            op.results = [{ type }];
        }
        if (parser.match('{')) {
            const region = op.addRegion();
            parser.parseRegion(region);
        }
        return true;
    }

    _parseLLVMAliasOp(parser, op) {
        const linkageKeywords = ['external', 'available_externally', 'linkonce', 'linkonce_odr', 'weak', 'weak_odr', 'internal', 'private'];
        if (parser.match('id') && linkageKeywords.includes(parser.getToken().value)) {
            op.addAttribute('linkage', parser.expect('id'));
        }
        const visibilityKeywords = ['default', 'hidden', 'protected'];
        if (parser.match('id') && visibilityKeywords.includes(parser.getToken().value)) {
            op.addAttribute('visibility_', parser.expect('id'));
        }
        if (parser.accept('id', 'thread_local')) {
            op.addAttribute('thread_local_', true);
        }
        const unnamedAddrKeywords = ['unnamed_addr', 'local_unnamed_addr'];
        if (parser.match('id') && unnamedAddrKeywords.includes(parser.getToken().value)) {
            op.addAttribute('unnamed_addr', parser.expect('id'));
        }
        if (parser.match('@')) {
            parser.parseSymbolName('sym_name', op.attributes);
        }
        if (parser.match('{')) {
            parser.parseAttributeDict(op.attributes);
        }
        if (parser.accept(':')) {
            const type = parser.parseType();
            op.addAttribute('alias_type', type);
        }
        if (parser.match('{')) {
            const region = op.addRegion();
            parser.parseRegion(region);
        }
        return true;
    }

    _parseGEPIndices(parser, op) {
        // Reference: LLVMDialect.cpp:750 parseGEPIndices uses parseCommaSeparatedList
        // Note: the reference uses 'none' delimiter with TableGen handling brackets,
        // but mlir.js expects '[' already consumed and needs to handle ']' terminator
        const rawConstantIndices = [];
        while (!parser.match(']')) {
            if (parser.match('int')) {
                const constIndex = parser.expect('int');
                rawConstantIndices.push(constIndex);
            } else {
                const operand = parser.parseAttribute();
                op.operands.push(operand);
                rawConstantIndices.push(-2147483648);
            }
            parser.accept(',');
        }
        if (rawConstantIndices.length > 0) {
            op.addAttribute('rawConstantIndices', rawConstantIndices);
        }
    }

    _parseIndirectBrOpSucessors(parser, op /*, args */) {
        // Format: [ ^block(%arg1, %arg2 : type1, type2), ^block2, ... ]
        // All operands listed first, then colon, then all types
        parser.expect('[');
        const segmentSizes = [];
        if (!parser.match(']')) {
            do {
                const successor = parser.expect('^');
                if (!op.successors) {
                    op.successors = [];
                }
                op.successors.push({ name: successor });
                const operands = [];
                if (parser.accept('(')) {
                    // Parse operands (all operands listed first)
                    while (!parser.match(')') && !parser.match(':')) {
                        const operand = parser.parseOperand();
                        operands.push(operand);
                        if (!parser.accept(',')) {
                            break;
                        }
                    }
                    // Parse types after colon
                    if (parser.accept(':')) {
                        let idx = 0;
                        while (!parser.match(')') && idx < operands.length) {
                            const type = parser.parseType();
                            if (operands[idx]) {
                                operands[idx].type = type;
                            }
                            idx++;
                            parser.accept(',');
                        }
                    }
                    parser.expect(')');
                }
                op.operands.push(...operands);
                segmentSizes.push(operands.length);
            } while (parser.accept(','));
        }
        parser.expect(']');
        if (segmentSizes.length > 0) {
            op.addAttribute('indbr_operand_segments', segmentSizes);
        }
    }

    _parseLLVMAllocaOp(parser, op) {
        if (parser.accept('id', 'inalloca')) {
            op.addAttribute('inalloca', true);
        }
        const arraySize = parser.parseAttribute();
        op.operands.push(arraySize);
        parser.expect('id', 'x');
        const elemType = parser.parseType();
        op.addAttribute('elem_type', elemType);
        if (parser.match('{')) {
            parser.parseAttributeDict(op.attributes);
        }
        // Reference: parseColonTypeList (mandatory)
        const types = parser.parseColonTypeList();
        parser.resolveOperands(op.operands, types);
        if (parser.accept('->')) {
            const resultType = parser.parseType();
            op.results = [{ type: resultType }];
        }
        return true;
    }

    _parseLLVMCallOp(parser, op) {
        const cconvKeywords = ['ccc', 'fastcc', 'coldcc', 'cc', 'webkit_jscc', 'anyregcc', 'preserve_mostcc', 'preserve_allcc', 'preserve_nonecc', 'cxx_fast_tlscc', 'tailcc', 'swiftcc', 'swifttailcc', 'cfguard_checkcc', 'ghccc', 'arm_apcscc', 'arm_aapcscc', 'arm_aapcs_vfpcc', 'aarch64_vector_pcs', 'aarch64_sve_vector_pcs', 'aarch64_sme_preservemost_from_x0', 'aarch64_sme_preservemost_from_x2', 'msp430_intrcc', 'avr_intrcc', 'avr_signalcc', 'ptx_kernelcc', 'ptx_devicecc', 'spir_funccc', 'spir_kernelcc', 'intel_ocl_bicc', 'x86_64_sysvcc', 'win64cc', 'x86_fastcallcc', 'x86_stdcallcc', 'x86_thiscallcc', 'x86_vectorcallcc', 'x86_intrcc', 'amdgpu_vs', 'amdgpu_gs', 'amdgpu_ps', 'amdgpu_cs', 'amdgpu_kernel', 'amdgpu_kernelcc', 'x86_regcallcc', 'amdgpu_hs', 'msp430_builtincc', 'amdgpu_ls', 'amdgpu_es', 'aarch64_vfpcc', 'aarch64_sve_vfpcc', 'wasm_emscripten_invokecc', 'amdgpu_gfx', 'm68k_intrcc'];
        if (parser.match('id')) {
            const value = parser.getToken().value;
            if (cconvKeywords.includes(value) || /^cc_\d+$/.test(value)) {
                op.addAttribute('CConv', parser.expect('id'));
            }
        }
        const tailcallKeywords = ['none', 'tail', 'musttail', 'notail'];
        if (parser.match('id') && tailcallKeywords.includes(parser.getToken().value)) {
            op.addAttribute('TailCallKind', parser.expect('id'));
        }
        let isDirect = false;
        if (parser.match('@')) {
            const callee = parser.expect('@');
            op.addAttribute('callee', callee);
            isDirect = true;
        } else if (parser.match('%')) {
            const calleePtr = parser.parseAttribute();
            op.operands.push(calleePtr);
        }
        parser.expect('(');
        while (!parser.match(')')) {
            const arg = parser.parseAttribute();
            op.operands.push(arg);
            parser.accept(',');
        }
        parser.expect(')');
        if (parser.accept('id', 'vararg')) {
            parser.expect('(');
            const varCalleeType = parser.parseType();
            op.addAttribute('var_callee_type', varCalleeType);
            parser.expect(')');
        }
        if (parser.accept('[')) {
            if (!parser.accept(']')) {
                const opBundles = [];
                do {
                    const tag = parser.expect('string');
                    parser.expect('(');
                    const bundleOperands = [];
                    if (!parser.match(')')) {
                        do {
                            bundleOperands.push(parser.parseAttribute());
                        } while (parser.accept(','));
                        parser.expect(':');
                        // Parse types for bundle operands
                        do {
                            parser.parseType();
                        } while (parser.accept(','));
                    }
                    parser.expect(')');
                    opBundles.push({ tag, operands: bundleOperands });
                } while (parser.accept(','));
                parser.expect(']');
                if (opBundles.length > 0) {
                    op.addAttribute('op_bundle_tags', opBundles);
                }
            }
        }
        if (parser.match('{')) {
            parser.parseAttributeDict(op.attributes);
        }
        parser.expect(':');
        if (!isDirect) {
            parser.parseType();
            parser.expect(',');
        }
        const sig = parser.parseFunctionSignature(op.operands);
        for (let i = 0; i < sig.argTypes.length && i < op.operands.length; i++) {
            op.operands[i].type = sig.argTypes[i].toString();
        }
        if (sig.resultTypes.length > 0) {
            op.results = sig.resultTypes.map((t) => ({ type: t.toString() }));
        }
        return true;
    }

    _parseLLVMCallIntrinsicOp(parser, op) {
        // Format: llvm.call_intrinsic "intrinsic.name"(%args) ["op_bundles"] : (arg_types) -> result_type
        const intrinName = parser.expect('string');
        op.addAttribute('intrin', intrinName);

        parser.expect('(');
        while (!parser.match(')')) {
            const arg = parser.parseAttribute();
            op.operands.push(arg);
            parser.accept(',');
        }
        parser.expect(')');

        // Parse operation bundles: [] or ["tag"()] or ["tag"(%0, %1 : i32, i32), ...]
        if (parser.accept('[')) {
            if (!parser.accept(']')) {
                const opBundles = [];
                do {
                    const tag = parser.expect('string');
                    parser.expect('(');
                    const bundleOperands = [];
                    if (!parser.match(')')) {
                        do {
                            bundleOperands.push(parser.parseAttribute());
                        } while (parser.accept(','));
                        parser.expect(':');
                        // Parse types for bundle operands
                        do {
                            parser.parseType();
                        } while (parser.accept(','));
                    }
                    parser.expect(')');
                    opBundles.push({ tag, operands: bundleOperands });
                } while (parser.accept(','));
                parser.expect(']');
                if (opBundles.length > 0) {
                    op.addAttribute('op_bundle_tags', opBundles);
                }
            }
        }

        if (parser.match('{')) {
            parser.parseAttributeDict(op.attributes);
        }

        parser.expect(':');
        // Parse function signature: (arg_types {attrs}) -> (result_types {attrs})
        const sig = parser.parseFunctionSignature(op.operands);
        for (let i = 0; i < sig.argTypes.length && i < op.operands.length; i++) {
            op.operands[i].type = sig.argTypes[i].toString();
        }
        if (sig.resultTypes.length > 0) {
            op.results = sig.resultTypes.map((t) => ({ type: t.toString() }));
        }
        return true;
    }

    _parseLLVMCmpOp(parser, op) {
        const predicate = parser.expect('string');
        op.addAttribute('predicate', predicate);
        const lhs = parser.parseAttribute();
        op.operands.push(lhs);
        parser.expect(',');
        const rhs = parser.parseAttribute();
        op.operands.push(rhs);
        if (parser.match('{')) {
            parser.parseAttributeDict(op.attributes);
        }
        parser.expect(':');
        const type = parser.parseType();
        for (const operand of op.operands) {
            operand.type = type;
        }
        return true;
    }

    _parseLLVMIntrinsicOp(parser, op) {
        parser.expect('(');
        while (!parser.match(')')) {
            const operand = parser.parseAttribute();
            op.operands.push(operand);
            parser.accept(',');
        }
        parser.expect(')');
        if (parser.match('{')) {
            parser.parseAttributeDict(op.attributes);
        }
        // Reference: parseColonTypeList (mandatory)
        const types = parser.parseColonTypeList();
        parser.resolveOperands(op.operands, types);
        if (parser.accept('->')) {
            const resultType = parser.parseType();
            op.results = [{ type: resultType }];
        }
        return true;
    }

    _parseLLVMInvokeOp(parser, op) {
        // Format: [cconv] (@callee | %funcptr) (args) to ^normalDest unwind ^unwindDest : [callee_type,] (arg_types) -> result_type
        const cconvKeywords = ['ccc', 'fastcc', 'coldcc', 'cc', 'webkit_jscc', 'anyregcc', 'preserve_mostcc', 'preserve_allcc', 'preserve_nonecc', 'cxx_fast_tlscc', 'tailcc', 'swiftcc', 'swifttailcc', 'cfguard_checkcc', 'ghccc'];
        if (parser.match('id')) {
            const value = parser.getToken().value;
            if (cconvKeywords.includes(value) || /^cc_\d+$/.test(value)) {
                op.addAttribute('CConv', parser.expect('id'));
            }
        }
        let isDirect = false;
        if (parser.match('@')) {
            isDirect = true;
            const callee = parser.expect('@');
            op.addAttribute('callee', callee);
        } else if (parser.match('%')) {
            const funcPtr = parser.parseAttribute();
            op.operands.push(funcPtr);
        }
        parser.expect('(');
        while (!parser.match(')')) {
            const operand = parser.parseAttribute();
            op.operands.push(operand);
            parser.accept(',');
        }
        parser.expect(')');
        parser.expect('id', 'to');
        const normalDest = parser.expect('^');
        op.successors = op.successors || [];
        const normalSucc = { label: normalDest };
        // Parse optional successor operands: ^bb1(%operand : type)
        if (parser.accept('(')) {
            normalSucc.operands = [];
            while (!parser.match(')')) {
                const operand = parser.parseAttribute();
                normalSucc.operands.push(operand);
                if (parser.accept(':')) {
                    parser.parseType();
                }
                parser.accept(',');
            }
            parser.expect(')');
        }
        op.successors.push(normalSucc);
        parser.expect('id', 'unwind');
        const unwindDest = parser.expect('^');
        const unwindSucc = { label: unwindDest };
        if (parser.accept('(')) {
            unwindSucc.operands = [];
            while (!parser.match(')')) {
                const operand = parser.parseAttribute();
                unwindSucc.operands.push(operand);
                if (parser.accept(':')) {
                    parser.parseType();
                }
                parser.accept(',');
            }
            parser.expect(')');
        }
        op.successors.push(unwindSucc);
        if (parser.accept('id', 'vararg')) {
            parser.expect('(');
            const varargType = parser.parseType();
            op.addAttribute('var_callee_type', varargType);
            parser.expect(')');
        }
        if (parser.accept('[')) {
            if (!parser.accept(']')) {
                const opBundles = [];
                do {
                    const tag = parser.expect('string');
                    parser.expect('(');
                    const bundleOperands = [];
                    if (!parser.match(')')) {
                        do {
                            bundleOperands.push(parser.parseAttribute());
                        } while (parser.accept(','));
                        parser.expect(':');
                        // Parse types for bundle operands
                        do {
                            parser.parseType();
                        } while (parser.accept(','));
                    }
                    parser.expect(')');
                    opBundles.push({ tag, operands: bundleOperands });
                } while (parser.accept(','));
                parser.expect(']');
                if (opBundles.length > 0) {
                    op.addAttribute('op_bundle_tags', opBundles);
                }
            }
        }
        if (parser.match('{')) {
            parser.parseAttributeDict(op.attributes);
        }
        parser.expect(':');
        if (!isDirect) {
            const calleeType = parser.parseType();
            if (op.operands.length > 0) {
                op.operands[0].type = calleeType.toString();
            }
            parser.expect(',');
        }
        const callArgOperands = isDirect ? op.operands : op.operands.slice(1);
        const sig = parser.parseFunctionSignature(callArgOperands);
        for (let i = 0; i < sig.argTypes.length && i < callArgOperands.length; i++) {
            callArgOperands[i].type = sig.argTypes[i].toString();
        }
        if (sig.resultTypes.length > 0) {
            op.results = sig.resultTypes.map((t) => ({ type: t.toString() }));
        }
        return true;
    }

    _parseLLVMLandingpadOp(parser, op) {
        // Format: cleanup? (catch|filter operand : type)*  : result_type
        // Parse optional cleanup
        if (parser.accept('id', 'cleanup')) {
            op.addAttribute('cleanup', true);
        }
        // Parse clauses
        while (parser.match('(')) {
            parser.expect('(');
            parser.expect('id'); // 'catch' or 'filter'
            const operand = parser.parseAttribute();
            op.operands.push(operand);
            parser.expect(':');
            const type = parser.parseType();
            operand.type = type.toString();
            parser.expect(')');
        }
        parser.expect(':');
        const resultType = parser.parseType();
        op.results = [{ type: resultType }];
        return true;
    }

    _parseLLVMFuncOp(parser, op) {
        const linkageKeywords = ['external', 'available_externally', 'linkonce', 'linkonce_odr', 'weak', 'weak_odr', 'appending', 'internal', 'private', 'extern_weak', 'common'];
        if (parser.match('id') && linkageKeywords.includes(parser.getToken().value)) {
            op.addAttribute('linkage', parser.expect('id'));
        }
        const visibilityKeywords = ['default', 'hidden', 'protected'];
        if (parser.match('id') && visibilityKeywords.includes(parser.getToken().value)) {
            op.addAttribute('visibility_', parser.expect('id'));
        }
        const unnamedAddrKeywords = ['unnamed_addr', 'local_unnamed_addr'];
        if (parser.match('id') && unnamedAddrKeywords.includes(parser.getToken().value)) {
            op.addAttribute('unnamed_addr', parser.expect('id'));
        }
        const cconvKeywords = ['ccc', 'fastcc', 'coldcc', 'cc', 'webkit_jscc', 'anyregcc', 'preserve_mostcc', 'preserve_allcc', 'preserve_nonecc', 'cxx_fast_tlscc', 'tailcc', 'swiftcc', 'swifttailcc', 'cfguard_checkcc', 'ghccc', 'arm_apcscc', 'arm_aapcscc', 'arm_aapcs_vfpcc', 'aarch64_vector_pcs', 'aarch64_sve_vector_pcs', 'aarch64_sme_preservemost_from_x0', 'aarch64_sme_preservemost_from_x2', 'msp430_intrcc', 'avr_intrcc', 'avr_signalcc', 'ptx_kernelcc', 'ptx_devicecc', 'spir_funccc', 'spir_kernelcc', 'intel_ocl_bicc', 'x86_64_sysvcc', 'win64cc', 'x86_fastcallcc', 'x86_stdcallcc', 'x86_thiscallcc', 'x86_vectorcallcc', 'x86_intrcc', 'amdgpu_vs', 'amdgpu_gs', 'amdgpu_ps', 'amdgpu_cs', 'amdgpu_kernel', 'amdgpu_kernelcc', 'x86_regcallcc', 'amdgpu_hs', 'msp430_builtincc', 'amdgpu_ls', 'amdgpu_es', 'aarch64_vfpcc', 'aarch64_sve_vfpcc', 'wasm_emscripten_invokecc', 'amdgpu_gfx', 'm68k_intrcc'];
        if (parser.match('id')) {
            const value = parser.getToken().value;
            if (cconvKeywords.includes(value) || /^cc_\d+$/.test(value)) {
                op.addAttribute('CConv', parser.expect('id'));
            }
        }
        parser.parseSymbolName('sym_name', op.attributes);
        const argResult = parser.parseFunctionArgumentList(true);
        const params = argResult.arguments.map((a) => a.type);
        const results = [];
        const resultAttrs = [];
        if (parser.accept('->')) {
            parser.parseFunctionResultList(results, resultAttrs);
        }
        const returnType = results.length > 0 ? results[0] : null;
        const type = new mlir.LLVMFunctionType(returnType, params, argResult.isVariadic);
        op.addAttribute('function_type', new mlir.TypeAttrOf(type));
        if (parser.accept('id', 'vscale_range')) {
            parser.expect('(');
            const minRange = parser.expect();
            parser.expect(',');
            const maxRange = parser.expect();
            parser.expect(')');
            op.addAttribute('vscale_range', `(${minRange}, ${maxRange})`);
        }
        if (parser.accept('id', 'comdat')) {
            parser.expect('(');
            const comdat = parser.expect('@');
            parser.expect(')');
            op.addAttribute('comdat', comdat);
        }
        parser.parseOptionalAttrDictWithKeyword(op.attributes);
        if (parser.match('{')) {
            const region = op.addRegion();
            parser.parseRegion(region, argResult.arguments);
        }
        return true;
    }
};

mlir.ROCDLDialect = class extends mlir.LLVMDialect {

    constructor(operations) {
        super(operations, 'rocdl');
    }

    parseOperation(parser, opName, op) {
        if (opName === 'rocdl.raw.buffer.load' && this.hasCustomAssemblyFormat(opName) && !this.hasAssemblyFormat(opName)) {
            return this._parseRawBufferLoadOp(parser, op);
        }
        if (opName === 'rocdl.raw.buffer.store' && this.hasCustomAssemblyFormat(opName) && !this.hasAssemblyFormat(opName)) {
            return this._parseRawBufferStoreOp(parser, op);
        }
        if (opName === 'rocdl.raw.buffer.atomic.fadd' && this.hasCustomAssemblyFormat(opName) && !this.hasAssemblyFormat(opName)) {
            return this._parseRawBufferAtomicOp(parser, op);
        }
        if (opName === 'rocdl.raw.buffer.atomic.fmax' && this.hasCustomAssemblyFormat(opName) && !this.hasAssemblyFormat(opName)) {
            return this._parseRawBufferAtomicOp(parser, op);
        }
        if (opName === 'rocdl.raw.buffer.atomic.smax' && this.hasCustomAssemblyFormat(opName) && !this.hasAssemblyFormat(opName)) {
            return this._parseRawBufferAtomicOp(parser, op);
        }
        if (opName === 'rocdl.raw.buffer.atomic.umin' && this.hasCustomAssemblyFormat(opName) && !this.hasAssemblyFormat(opName)) {
            return this._parseRawBufferAtomicOp(parser, op);
        }
        return super.parseOperation(parser, opName, op);
    }

    _parseRawBufferLoadOp(parser, op) {
        while (parser.match('%')) {
            op.operands.push({ value: parser.expect('%') });
            parser.accept(',');
        }
        parser.expect(':');
        const resultType = parser.parseType();
        op.results.push({ type: resultType });
        return true;
    }

    _parseRawBufferStoreOp(parser, op) {
        while (parser.match('%')) {
            op.operands.push({ value: parser.expect('%') });
            parser.accept(',');
        }
        parser.expect(':');
        parser.parseType();
        return true;
    }

    _parseRawBufferAtomicOp(parser, op) {
        while (parser.match('%')) {
            op.operands.push({ value: parser.expect('%') });
            parser.accept(',');
        }
        parser.expect(':');
        parser.parseType();
        return true;
    }
};

mlir.XSMMDialect = class extends mlir.Dialect {

    constructor(operations) {
        super(operations, 'xsmm');
    }

    parseOperation(parser, opName, op) {
        if (opName === 'xsmm.unary.invoke' && this.hasCustomAssemblyFormat(opName) && !this.hasAssemblyFormat(opName)) {
            return this._parseUnaryInvokeOp(parser, op);
        }
        if (opName.startsWith('xsmm.') && opName.includes('.invoke') && this.hasCustomAssemblyFormat(opName) && !this.hasAssemblyFormat(opName)) {
            return this._parseGemmInvokeOp(parser, op);
        }
        return super.parseOperation(parser, opName, op);
    }

    _parseUnaryInvokeOp(parser, op) {
        op.operands.push({ value: parser.expect('%') });
        if (parser.accept('[')) {
            while (!parser.accept(']')) {
                if (parser.match('%')) {
                    op.operands.push({ value: parser.expect('%') });
                }
                parser.accept(',');
            }
        }
        parser.expect('=');
        op.operands.push({ value: parser.expect('%') });
        parser.expect('(');
        op.operands.push({ value: parser.expect('%') });
        if (parser.accept('[')) {
            while (!parser.accept(']')) {
                if (parser.match('%')) {
                    op.operands.push({ value: parser.expect('%') });
                }
                parser.accept(',');
            }
        }
        parser.expect(')');
        parser.expect(':');
        parser.parseType();
        return true;
    }

    _parseGemmInvokeOp(parser, op) {
        op.operands.push({ value: parser.expect('%') });
        parser.expect(',');
        op.operands.push({ value: parser.expect('%') });
        if (parser.accept('[')) {
            while (!parser.accept(']')) {
                if (parser.match('%')) {
                    op.operands.push({ value: parser.expect('%') });
                }
                parser.accept(',');
            }
        }
        parser.expect('=');
        op.operands.push({ value: parser.expect('%') });
        if (parser.accept('[')) {
            while (!parser.accept(']')) {
                if (parser.match('%')) {
                    op.operands.push({ value: parser.expect('%') });
                }
                parser.accept(',');
            }
        }
        parser.expect(',');
        op.operands.push({ value: parser.expect('%') });
        if (parser.accept('[')) {
            while (!parser.accept(']')) {
                if (parser.match('%')) {
                    op.operands.push({ value: parser.expect('%') });
                }
                parser.accept(',');
            }
        }
        while (parser.accept(',')) {
            if (parser.match('%')) {
                op.operands.push({ value: parser.expect('%') });
                if (parser.accept('[')) {
                    while (!parser.accept(']')) {
                        if (parser.match('%')) {
                            op.operands.push({ value: parser.expect('%') });
                        }
                        parser.accept(',');
                    }
                }
            } else if (parser.match('id')) {
                const keyword = parser.expect('id');
                parser.expect('=');
                const attrValue = parser.parseAttribute(new mlir.PrimitiveType('i64'));
                op.addAttribute(keyword, attrValue);
            } else {
                break;
            }
        }
        parser.expect(':');
        parser.parseType();
        return true;
    }
};

mlir.StdxDialect = class extends mlir.Dialect {

    constructor(operations) {
        super(operations, 'stdx');
    }

    parseOperation(parser, opName, op) {
        if (opName === 'stdx.closure') {
            return this._parseClosureOp(parser, op);
        }
        return super.parseOperation(parser, opName, op);
    }

    _parseClosureOp(parser, op) {
        const sig = parser.parseFunctionSignatureWithArguments(false);
        const argTypes = sig.arguments.map((a) => a.type);
        const type = { inputs: argTypes, results: sig.resultTypes };
        op.addAttribute('type', type);
        parser.parseOptionalAttrDictWithKeyword(op.attributes);
        if (parser.match('{')) {
            const region = op.addRegion();
            parser.parseRegion(region, sig.arguments);
        }
        return true;
    }
};

mlir.VMDialect = class extends mlir.Dialect {

    constructor(operations) {
        super(operations, 'vm');
        this.registerCustomDirective('BranchTableCases', this._parseBranchTableCases.bind(this));
    }

    parseOperation(parser, opName, op) {
        if (opName === 'vm.func') {
            parser.parseFunctionOp(op, false);
            return true;
        }
        if (opName === 'vm.cond_fail') {
            // Format: vm.cond_fail %cond, %status, "message"
            // or: vm.cond_fail %status, "message"
            // or: vm.cond_fail %cond, %status
            // or: vm.cond_fail %status
            const firstOp = parser.expect('%');
            op.operands.push({ value: firstOp });
            if (parser.accept(',')) {
                // Could be second operand or message
                if (parser.match('%')) {
                    const secondOp = parser.expect('%');
                    op.operands.push({ value: secondOp });
                    // Optional message
                    if (parser.accept(',')) {
                        if (parser.match('string')) {
                            const msg = parser.expect('string');
                            op.addAttribute('message', msg);
                        }
                    }
                } else if (parser.match('string')) {
                    const msg = parser.expect('string');
                    op.addAttribute('message', msg);
                }
            }
            return true;
        }
        if (opName === 'vm.import') {
            parser.parseOptionalVisibilityKeyword(op.attributes);
            if (parser.accept('id', 'optional')) {
                op.addAttribute('is_optional', true);
            }
            parser.parseSymbolName('sym_name', op.attributes);
            if (parser.match('(')) {
                parser.skip('(');
            }
            const inputs = [];
            const results = [];
            const resultAttrs = [];
            if (parser.accept('->')) {
                parser.parseFunctionResultList(results, resultAttrs);
            }
            op.addAttribute('function_type', new mlir.TypeAttrOf(new mlir.FunctionType(inputs, results)));
            parser.parseOptionalAttrDictWithKeyword(op.attributes);
            return true;
        }
        if (opName === 'vm.export') {
            const functionRef = parser.expect('@');
            op.addAttribute('function_ref', functionRef);
            if (parser.accept('id', 'as')) {
                parser.expect('(');
                const exportName = parser.expect('string');
                op.addAttribute('export_name', exportName);
                parser.expect(')');
            } else {
                op.addAttribute('export_name', functionRef);
            }
            parser.parseOptionalAttrDictWithKeyword(op.attributes);
            return true;
        }
        // Handle vm.global.* declarations (but not store/load operations)
        if (opName.startsWith('vm.global.') && !opName.startsWith('vm.global.store.') && !opName.startsWith('vm.global.load.')) {
            this.getOperation(opName).hasParseOperation = false; // compatibility?
            parser.parseOptionalVisibilityKeyword(op.attributes);
            if (parser.match('id', 'mutable')) {
                const mutable = parser.expect('id');
                op.addAttribute('is_mutable', mutable);
            }
            parser.parseSymbolName('sym_name', op.attributes);
            parser.parseOptionalAttrDict(op.attributes);
            if (parser.accept(':')) {
                const type = parser.parseType();
                op.addAttribute('type', type);
            }
            if (parser.accept('=')) {
                const initialValue = parser.parseAttribute();
                op.addAttribute('initial_value', initialValue);
            }
            return true;
        }
        if (opName === 'vm.initializer') {
            parser.parseOptionalVisibilityKeyword(op.attributes);
            if (parser.match('@')) {
                parser.parseSymbolName('sym_name', op.attributes);
            }
            parser.parseOptionalAttrDictWithKeyword(op.attributes);
            if (parser.match('{')) {
                const region = op.addRegion();
                parser.parseRegion(region);
            }
            return true;
        }
        if (opName === 'vm.rodata.inline') {
            this.getOperation(opName).hasParseOperation = false; // compatibility?
            // Optional name (string)
            if (parser.match('string')) {
                const name = parser.expect('string');
                op.addAttribute('name', name);
            }
            // Attr-dict
            parser.parseOptionalAttrDict(op.attributes);
            // : type = value
            // Reference: parseOptionalColonTypeList
            parser.resolveOperands(op.results, parser.parseOptionalColonTypeList());
            if (parser.accept('=')) {
                const value = parser.parseAttribute();
                // Handle type annotation after the value (e.g., dense<...> : vector<21xi8>)
                if (parser.accept(':')) {
                    const valueType = parser.parseType();
                    value.type = valueType;
                }
                op.addAttribute('value', value);
            }
            return true;
        }
        // Handle vm.const.* operations (e.g., vm.const.i32.zero : i32, vm.const.i32 1 : i32, vm.const.ref.rodata @symbol : !vm.buffer)
        if (opName.startsWith('vm.const.') &&
            opName !== 'vm.const.i32' &&
            opName !== 'vm.const.ref.rodata' && opName !== 'vm.const.ref.zero') {
            if (opName === 'vm.const.i32.zero') {
                this.getOperation(opName).hasParseOperation = false; // compatibility?
            }
            // Optional value or symbol reference
            if (parser.match('int') || parser.match('float') || parser.match('string')) {
                const value = parser.parseAttribute();
                op.addAttribute('value', value.value === undefined ? value : value.value);
            } else if (parser.match('@')) {
                // Handle symbol reference (e.g., @symbol_name)
                const symbol = parser.expect('@');
                op.addAttribute('rodata', symbol);
            }
            parser.parseOptionalAttrDict(op.attributes);
            // Reference: parseOptionalColonTypeList
            parser.resolveOperands(op.results, parser.parseOptionalColonTypeList());
            return true;
        }
        // Handle vm.switch.ref operation
        // Format: $index `[` $values `]` `else` $default_value attr-dict `:` type($result)
        if (opName === 'vm.switch.ref' && this.hasCustomAssemblyFormat(opName) && !this.hasAssemblyFormat(opName)) {
            const index = parser.expect('%');
            op.operands.push({ value: index, name: 'index' });
            parser.expect('[');
            while (!parser.match(']')) {
                if (parser.match('%')) {
                    const value = parser.expect('%');
                    op.operands.push({ value, name: 'values' });
                }
                if (!parser.accept(',')) {
                    break;
                }
            }
            parser.expect(']');
            parser.expect('id', 'else');
            const defaultValue = parser.expect('%');
            op.operands.push({ value: defaultValue, name: 'default_value' });
            parser.parseOptionalAttrDict(op.attributes);
            if (parser.accept(':')) {
                const resultType = parser.parseType();
                op.results.push({ type: resultType });
            }
            return true;
        }
        // Handle vm.call and vm.call.variadic
        // Format: @callee(operands) {attrs} : (types) -> results
        // Variadic has complex syntax like: @callee(op1, op2, [(tuple1), (tuple2)])
        if (opName === 'vm.call' || opName === 'vm.call.variadic') {
            this.getOperation(opName).hasParseOperation = false; // compatibility?
            if (parser.match('@')) {
                const callee = parser.expect('@');
                op.addAttribute('callee', callee);
            }
            // Parse operands - use skip for complex nested structures
            if (parser.accept('(')) {
                while (!parser.match(')')) {
                    if (parser.match('[')) {
                        // Skip complex nested structures in variadic calls
                        parser.skip('[');
                        parser.accept(','); // consume trailing comma if present
                    } else if (parser.match('%')) {
                        const operand = parser.expect('%');
                        op.operands.push({ value: operand });
                        parser.accept(','); // consume trailing comma if present
                    } else {
                        // Unexpected token, break to avoid infinite loop
                        break;
                    }
                }
                parser.expect(')');
            }
            parser.parseOptionalAttrDict(op.attributes);
            // vm.call.variadic has special syntax with '...' ellipsis
            if (parser.accept(':')) {
                if (opName === 'vm.call.variadic') {
                    parser.skip('(');
                    if (parser.accept('->')) {
                        const resultTypes = parser.parseFunctionResultTypes();
                        parser.resolveOperands(op.results, resultTypes);
                    }
                } else {
                    // Regular vm.call - Reference: uses functional-type(operands, results)
                    const type = parser.parseType();
                    if (type instanceof mlir.FunctionType) {
                        parser.resolveOperands(op.operands, type.inputs);
                        parser.resolveOperands(op.results, type.results);
                    }
                }
            }
            return true;
        }
        return super.parseOperation(parser, opName, op);
    }

    _parseBranchTableCases(parser, op /*, args */) {
        // Format: default: ^bb(%args : types), 0: ^bb2(%args2 : types2), ...
        // Parse default case
        if (parser.accept('id', 'default')) {
            parser.expect(':');
            const defaultDest = parser.expect('^');
            op.successors = op.successors || [];
            const succ = { dest: defaultDest };
            if (parser.match('(')) {
                parser.expect('(');
                const operands = [];
                while (!parser.match(')')) {
                    if (parser.match('%')) {
                        operands.push({ value: parser.expect('%') });
                    }
                    if (parser.accept(':')) {
                        // Parse types for the operands
                        while (!parser.match(')') && !parser.match(',')) {
                            parser.parseType();
                            parser.accept(',');
                        }
                    }
                    if (!parser.accept(',')) {
                        break;
                    }
                }
                parser.expect(')');
                succ.operands = operands;
            }
            op.successors.push(succ);
            parser.accept(',');
        }
        // Parse numbered cases: 0: ^bb(...), 1: ^bb2(...), ...
        const caseValues = [];
        while (parser.match('int')) {
            const caseValue = parser.parseInteger();
            caseValues.push(caseValue);
            parser.expect(':');
            const caseDest = parser.expect('^');
            const caseSucc = { dest: caseDest };
            if (parser.match('(')) {
                parser.expect('(');
                const operands = [];
                while (!parser.match(')')) {
                    if (parser.match('%')) {
                        operands.push({ value: parser.expect('%') });
                    }
                    if (parser.accept(':')) {
                        while (!parser.match(')') && !parser.match(',')) {
                            parser.parseType();
                            parser.accept(',');
                        }
                    }
                    if (!parser.accept(',')) {
                        break;
                    }
                }
                parser.expect(')');
                caseSucc.operands = operands;
            }
            op.successors.push(caseSucc);
            parser.accept(',');
        }
        if (caseValues.length > 0) {
            op.addAttribute('case_values', caseValues);
        }
    }
};

mlir.MathDialect = class extends mlir.Dialect {

    constructor(operations) {
        super(operations, 'math');
        this.registerCustomAttribute('Arith_FastMathAttr', this._parseEnumFlagsAngleBracketComma.bind(this));
    }
};

mlir.TMTensorDialect = class extends mlir.Dialect {

    constructor(operations) {
        super(operations, 'tm_tensor');
    }
};

mlir.MLProgramDialect = class extends mlir.Dialect {

    constructor(operations) {
        super(operations, 'ml_program');
        this.registerCustomDirective('TypedInitialValue', this._parseTypedInitialValue.bind(this));
        this.registerCustomDirective('TokenOrdering', this._parseTokenOrdering.bind(this));
    }

    parseOperation(parser, opName, op) {
        if (opName === 'ml_program.func' || opName === 'ml_program.subgraph') {
            parser.parseFunctionOp(op, false);
            return true;
        }
        return super.parseOperation(parser, opName, op);
    }

    _parseTokenOrdering(parser, op) {
        // Reference: MLProgramOps.cpp parseTokenOrdering
        // Format: ordering(() -> type) or ordering(%tok1, %tok2 -> type)
        if (!parser.accept('id', 'ordering')) {
            return;
        }
        parser.expect('(');
        // Parse consuming token list: either () or %tok1, %tok2, ...
        if (parser.accept('(')) {
            parser.expect(')');
        } else {
            while (parser.match('%')) {
                const tok = parser.expect('%');
                op.operands.push({ name: 'consumeTokens', value: tok });
                if (!parser.accept(',')) {
                    break;
                }
            }
        }
        // Parse producer token type: -> type
        parser.expect('->');
        const produceType = parser.parseType();
        op.addAttribute('produceTokenType', { value: produceType, hidden: true });
        parser.expect(')');
    }

    _parseTypedInitialValue(parser, op, args) {
        const typeArg = args[0].replace('$', '');
        const valueArg = args[1].replace('$', '');
        if (parser.accept('(')) {
            const attr = parser.parseAttribute();
            if (parser.accept(':')) {
                attr.type = parser.parseType();
            }
            parser.expect(')');
            op.addAttribute(valueArg, attr.value === undefined ? attr : attr.value);
        }
        parser.expect(':');
        const type = parser.parseType();
        op.addAttribute(typeArg, type);
    }
};

mlir.IREEGPUDialect = class extends mlir.Dialect {

    constructor(operations) {
        super(operations, 'iree_gpu');
    }
};

mlir.TFDeviceDialect = class extends mlir.Dialect {

    constructor(operations) {
        super(operations, 'tf_device');
    }

    parseOperation(parser, opName, op) {
        if (opName === 'tf_device.replicate') {
            return this._parseReplicateOp(parser, op);
        }
        return super.parseOperation(parser, opName, op);
    }

    _parseReplicateOp(parser, op) {
        if (!parser.accept('(')) {
            parser.parseOptionalAttrDict(op.attributes);
            if (parser.match('{')) {
                const region = op.addRegion();
                parser.parseRegion(region);
            }
            return true;
        }
        if (parser.match(')')) {
            parser.expect(')');
            parser.parseOptionalAttrDict(op.attributes);
            if (parser.match('{')) {
                const region = op.addRegion();
                parser.parseRegion(region);
            }
            return true;
        }
        do {
            if (parser.match('[')) {
                const replicatedInputs = [];
                parser.expect('[');
                while (!parser.accept(']')) {
                    const value = parser.expect('%');
                    replicatedInputs.push({ value });
                    if (!parser.accept(',')) {
                        parser.expect(']');
                        break;
                    }
                }
                parser.expect('id', 'as');
                const blockArg = parser.expect('%');
                parser.expect(':');
                const type = parser.parseType();
                op.operands.push({ name: 'replicated_inputs', value: replicatedInputs, blockArg, type });
            } else if (parser.match('%')) {
                const value = parser.expect('%');
                parser.expect('id', 'as');
                const blockArg = parser.expect('%');
                parser.expect(':');
                const type = parser.parseType();
                op.operands.push({ name: 'packed_inputs', value, blockArg, type });
            } else {
                break;
            }
        } while (parser.accept(','));
        parser.expect(')');
        parser.parseOptionalAttrDict(op.attributes);
        if (parser.match('{')) {
            const region = op.addRegion();
            parser.parseRegion(region);
        }
        return true;
    }
};

mlir.TFGDialect = class extends mlir.Dialect {

    constructor(operations) {
        super(operations, 'tfg');
        this._operations.set('tfg.graph', { metadata: { name: 'tfg.graph', hasCustomAssemblyFormat: true, defaultDialect: 'tfg' } });
        this._operations.set('tfg.func', { metadata: { name: 'tfg.func', hasCustomAssemblyFormat: true, defaultDialect: 'tfg' } });
    }

    getOperation(opName) {
        const op = this._operations.get(opName);
        if (op) {
            return op;
        }
        return { metadata: { name: opName, hasCustomAssemblyFormat: true } };
    }

    parseOperation(parser, opName, op) {
        if (opName === 'tfg.graph') {
            if (parser.match('#')) {
                const attr = parser.parseAttribute();
                op.addAttribute('version', attr);
            }
            if (parser.match('{')) {
                const region = op.addRegion();
                parser.parseRegion(region);
            }
            return true;
        }
        if (opName === 'tfg.func') {
            // Handle optional 'generic' keyword
            if (parser.accept('id', 'generic')) {
                op.addAttribute('generic', true);
            }
            parser.parseFunctionOp(op, false);
            return true;
        }
        // yield: [`(` $args `)`]? [`[` $ctls `]`]? attr-dict [`:` type($args)]?
        if (opName === 'tfg.yield') {
            // Parse optional data operands
            if (parser.match('(')) {
                op.operands.push(...parser.parseArguments());
            }
            // Parse optional control dependencies
            if (parser.accept('[')) {
                while (!parser.match(']')) {
                    if (parser.match('%')) {
                        const ctlDep = parser.expect('%');
                        op.operands.push({ value: ctlDep, isControlDep: true });
                    }
                    if (!parser.accept(',')) {
                        break;
                    }
                }
                parser.expect(']');
            }
            // Parse optional attributes
            parser.parseOptionalAttrDict(op.attributes);
            // Parse optional types
            if (parser.accept(':')) {
                const types = parser.parseTypeListNoParens();
                parser.resolveOperands(op.operands, types);
            }
            return true;
        }
        // return: `(` $operands `)` [`[` $ctl {attr}, ... `]`]? attr-dict `:` types
        if (opName === 'tfg.return') {
            // Parse optional data operands
            if (parser.match('(')) {
                op.operands.push(...parser.parseArguments());
            }
            // Parse optional control operands with attributes
            const controlRetAttrs = [];
            if (parser.accept('[')) {
                while (!parser.match(']')) {
                    if (parser.match('%')) {
                        const ctlDep = parser.expect('%');
                        op.operands.push({ value: ctlDep, isControlDep: true });
                        // Parse optional attributes on control operand
                        if (parser.match('{')) {
                            const attrs = new Map();
                            parser.parseAttributeDict(attrs);
                            controlRetAttrs.push(Object.fromEntries(attrs));
                        } else {
                            controlRetAttrs.push({});
                        }
                    }
                    if (!parser.accept(',')) {
                        break;
                    }
                }
                parser.expect(']');
            }
            if (controlRetAttrs.length > 0) {
                op.addAttribute('control_ret_attrs', controlRetAttrs);
            }
            // Parse optional attributes
            parser.parseOptionalAttrDict(op.attributes);
            // Parse optional types
            if (parser.accept(':')) {
                const types = parser.parseTypeListNoParens();
                parser.resolveOperands(op.operands, types);
            }
            return true;
        }
        // condition: $cond `:` type($cond) [`(` $args `)`]? [`[` $ctls `]`]? attr-dict [`:` type($args)]?
        if (opName === 'tfg.condition') {
            // Parse condition operand
            const cond = parser.expect('%');
            op.operands.push({ value: cond });
            // Parse condition type
            parser.expect(':');
            const condType = parser.parseType();
            op.operands[0].type = condType.toString();
            // Parse optional args
            if (parser.match('(')) {
                op.operands.push(...parser.parseArguments());
            }
            // Parse optional control dependencies
            if (parser.accept('[')) {
                while (!parser.match(']')) {
                    if (parser.match('%')) {
                        const ctlDep = parser.expect('%');
                        op.operands.push({ value: ctlDep, isControlDep: true });
                    }
                    if (!parser.accept(',')) {
                        break;
                    }
                }
                parser.expect(']');
            }
            // Parse optional attributes
            parser.parseOptionalAttrDict(op.attributes);
            // Parse optional arg types
            if (parser.accept(':')) {
                const types = parser.parseTypeListNoParens();
                // Skip first operand (cond) when resolving types
                const argsOperands = op.operands.slice(1).filter((o) => !o.isControlDep);
                parser.resolveOperands(argsOperands, types);
            }
            return true;
        }
        // get_result: `(` $value `)` $name `:` $number attr-dict-with-keyword
        if (opName === 'tfg.get_result') {
            parser.expect('(');
            const value = parser.expect('%');
            op.operands.push({ value, type: '!tfg.opaque_tensor' });
            parser.expect(')');
            const name = parser.expect('string');
            op.addAttribute('name', name);
            parser.expect(':');
            const number = parser.expect('int');
            op.addAttribute('number', parseInt(number, 10));
            parser.parseOptionalAttrDictWithKeyword(op.attributes);
            op.results.push({ type: '!tfg.opaque_tensor' });
            return true;
        }
        // IfRegion: $cond [`[` $ctls `]`]? `then` $then_region `else` $else_region attr-dict `:` functional-type($cond, $outs)
        if (opName === 'tfg.IfRegion' || opName === 'tfg.StatelessIfRegion' || opName === 'tfg.StatefulIfRegion') {
            // Parse condition operand
            const cond = parser.expect('%');
            op.operands.push({ value: cond });
            // Parse optional control dependencies
            if (parser.accept('[')) {
                while (!parser.match(']')) {
                    if (parser.match('%')) {
                        const ctlDep = parser.expect('%');
                        op.operands.push({ value: ctlDep, isControlDep: true });
                    }
                    if (!parser.accept(',')) {
                        break;
                    }
                }
                parser.expect(']');
            }
            // Parse 'then' region
            parser.expect('id', 'then');
            const thenRegion = op.addRegion();
            parser.parseRegion(thenRegion);
            // Parse 'else' region
            parser.expect('id', 'else');
            const elseRegion = op.addRegion();
            parser.parseRegion(elseRegion);
            // Parse optional attributes and type
            parser.parseOptionalAttrDict(op.attributes);
            if (parser.accept(':')) {
                const type = parser.parseType();
                if (type instanceof mlir.FunctionType) {
                    parser.resolveOperands(op.operands, type.inputs);
                    parser.resolveOperands(op.results, type.results);
                }
            }
            return true;
        }
        // CaseRegion: $branch_index [`[` $ctls `]`]? $branches attr-dict `:` functional-type($branch_index, $outs)
        if (opName === 'tfg.CaseRegion' || opName === 'tfg.StatelessCaseRegion' || opName === 'tfg.StatefulCaseRegion') {
            // Parse branch index operand
            const branchIndex = parser.expect('%');
            op.operands.push({ value: branchIndex });
            // Parse optional control dependencies
            if (parser.accept('[')) {
                while (!parser.match(']')) {
                    if (parser.match('%')) {
                        const ctlDep = parser.expect('%');
                        op.operands.push({ value: ctlDep, isControlDep: true });
                    }
                    if (!parser.accept(',')) {
                        break;
                    }
                }
                parser.expect(']');
            }
            // Parse branch regions (comma-separated)
            do {
                const region = op.addRegion();
                parser.parseRegion(region);
            } while (parser.accept(','));
            // Parse optional attributes and type
            parser.parseOptionalAttrDict(op.attributes);
            if (parser.accept(':')) {
                const type = parser.parseType();
                if (type instanceof mlir.FunctionType) {
                    parser.resolveOperands(op.operands, type.inputs);
                    parser.resolveOperands(op.results, type.results);
                }
            }
            return true;
        }
        // WhileRegion: [`(` $init `)`]? [`[` $ctls `]`]? $cond_region `do` $body_region attr-dict [`:` functional-type($init, $outs)]?
        if (opName === 'tfg.WhileRegion' || opName === 'tfg.StatelessWhileRegion' || opName === 'tfg.StatefulWhileRegion') {
            // Parse optional init operands
            if (parser.match('(')) {
                op.operands.push(...parser.parseArguments());
            }
            // Parse optional control dependencies
            if (parser.accept('[')) {
                while (!parser.match(']')) {
                    if (parser.match('%')) {
                        const ctlDep = parser.expect('%');
                        op.operands.push({ value: ctlDep, isControlDep: true });
                    }
                    if (!parser.accept(',')) {
                        break;
                    }
                }
                parser.expect(']');
            }
            // Parse condition region
            const condRegion = op.addRegion();
            parser.parseRegion(condRegion);
            // Parse 'do' and body region
            parser.expect('id', 'do');
            const bodyRegion = op.addRegion();
            parser.parseRegion(bodyRegion);
            // Parse optional attributes and type
            parser.parseOptionalAttrDict(op.attributes);
            if (parser.accept(':')) {
                const type = parser.parseType();
                if (type instanceof mlir.FunctionType) {
                    parser.resolveOperands(op.operands, type.inputs);
                    parser.resolveOperands(op.results, type.results);
                }
            }
            return true;
        }
        // ForRegion: [`(` $init `)`]? [`[` $ctls `]`]? `from` $start `to` $limit `by` $delta $body_region attr-dict `:` ...
        if (opName === 'tfg.ForRegion') {
            // Parse optional init operands
            if (parser.match('(')) {
                op.operands.push(...parser.parseArguments());
            }
            // Parse optional control dependencies
            if (parser.accept('[')) {
                while (!parser.match(']')) {
                    if (parser.match('%')) {
                        const ctlDep = parser.expect('%');
                        op.operands.push({ value: ctlDep, isControlDep: true });
                    }
                    if (!parser.accept(',')) {
                        break;
                    }
                }
                parser.expect(']');
            }
            // Parse 'from' $start 'to' $limit 'by' $delta
            parser.expect('id', 'from');
            const start = parser.expect('%');
            op.operands.push({ value: start });
            parser.expect('id', 'to');
            const limit = parser.expect('%');
            op.operands.push({ value: limit });
            parser.expect('id', 'by');
            const delta = parser.expect('%');
            op.operands.push({ value: delta });
            // Parse body region
            const bodyRegion = op.addRegion();
            parser.parseRegion(bodyRegion);
            // Parse optional attributes
            parser.parseOptionalAttrDict(op.attributes);
            // Parse type: `(` type($start) `,` type($limit) `,` type($delta) [`,` type($init)]? `)` [`->` `(` type($outs) `)`]?
            if (parser.accept(':')) {
                parser.expect('(');
                const types = parser.parseTypeListNoParens();
                parser.expect(')');
                // First 3 types are start, limit, delta
                if (types.length >= 3) {
                    // Assign types to start, limit, delta operands (last 3 before init)
                    const initCount = op.operands.length - 3;
                    for (let i = 0; i < 3; i++) {
                        op.operands[initCount + i].type = types[i].toString();
                    }
                    // Remaining types are for init operands
                    for (let i = 3; i < types.length && i - 3 < initCount; i++) {
                        op.operands[i - 3].type = types[i].toString();
                    }
                }
                // Parse optional result types
                if (parser.accept('->')) {
                    parser.expect('(');
                    const resultTypes = parser.parseTypeListNoParens();
                    parser.expect(')');
                    for (const t of resultTypes) {
                        op.results.push({ type: t.toString() });
                    }
                }
            }
            return true;
        }
        this._parseTFGOperation(parser, op);
        return true;
    }

    _parseTFGOperation(parser, op) {
        if (parser.match('(')) {
            op.operands.push(...parser.parseArguments());
        }
        if (parser.accept('[')) {
            while (!parser.match(']')) {
                if (parser.match('%')) {
                    const ctlDep = parser.expect('%');
                    op.operands.push({ value: ctlDep, isControlDep: true });
                }
                if (!parser.accept(',')) {
                    break;
                }
            }
            parser.expect(']');
        }
        if (parser.accept('id', 'device')) {
            parser.expect('(');
            const device = parser.expect('string');
            parser.expect(')');
            op.addAttribute('device', device);
        }
        if (parser.accept('id', 'name')) {
            parser.expect('(');
            const name = parser.expect('string');
            parser.expect(')');
            op.addAttribute('_mlir_name', name);
        }
        if (parser.match('{')) {
            parser.parseAttributeDict(op.attributes);
        }
        if (parser.accept(':')) {
            const type = parser.parseType();
            if (type instanceof mlir.FunctionType) {
                parser.resolveOperands(op.operands, type.inputs);
                parser.resolveOperands(op.results, type.results);
            } else {
                // Parse remaining types in the comma-separated list (for return-like operations)
                const types = [type];
                while (parser.accept(',')) {
                    types.push(parser.parseType());
                }
                parser.resolveOperands(op.operands, types);
            }
        }
    }
};

mlir.TFExecutorDialect = class extends mlir.Dialect {

    constructor(operations) {
        super(operations, 'tf_executor');
    }

    parseType(parser, dialectName) {
        const typeName = parser.parseKeyword();
        if (!typeName) {
            return null;
        }
        const type = `!${dialectName}.${typeName}`;
        if (typeName === 'control' || typeName === 'token') {
            return new mlir.Type(type);
        }
        return null;
    }

    parseOperation(parser, opName, op) {
        if (opName === 'tf_executor.graph') {
            return this._parseGraphOp(parser, op);
        }
        if (opName === 'tf_executor.island') {
            return this._parseIslandOp(parser, op);
        }
        if (opName === 'tf_executor.Enter') {
            return this._parseEnterOp(parser, op);
        }
        if (opName === 'tf_executor._SwitchN') {
            const data = parser.expect('%');
            op.operands.push({ value: data });
            parser.expect(',');
            const index = parser.expect('%');
            op.operands.push({ value: index });
            parser.expect('id', 'of');
            const numOuts = parseInt(parser.expect('int'), 10);
            op.addAttribute('num_outs', numOuts);
            if (parser.match('(')) {
                const controlInputs = parser.parseArguments();
                op.operands.push(...controlInputs);
            }
            parser.expect(':');
            const type = parser.parseType();
            const typeStr = type.toString();
            op.operands[0].type = typeStr;
            op.operands[1].type = 'tensor<i32>';
            for (let i = 2; i < op.operands.length; i++) {
                op.operands[i].type = '!tf_executor.control';
            }
            for (let i = 0; i < numOuts; i++) {
                op.results.push({ type: typeStr });
            }
            op.results.push({ type: '!tf_executor.control' });
            parser.parseOptionalAttrDict(op.attributes);
            return true;
        }
        if (opName === 'tf_executor.Switch' || opName === 'tf_executor.Merge' ||
            opName === 'tf_executor.LoopCond' || opName === 'tf_executor.Exit') {
            // These ops have hasCustomAssemblyFormat: true but no assemblyFormat in metadata
            op.operands = parser.parseArguments();
            if (parser.accept(':')) {
                const type = parser.parseType();
                if (type instanceof mlir.FunctionType) {
                    for (let i = 0; i < op.operands.length; i++) {
                        const inputType = type.inputs[i];
                        if (inputType) {
                            op.operands[i].type = inputType;
                        }
                    }
                    for (let i = 0; i < type.results.length; i++) {
                        const outputType = type.results[i];
                        if (outputType) {
                            op.results.push({ type: outputType });
                        }
                    }
                } else {
                    const typeStr = type.toString();
                    for (const operand of op.operands) {
                        operand.type = typeStr;
                    }
                    op.results.push({ type: typeStr });
                    op.results.push({ type: '!tf_executor.control' });
                }
            }
            parser.parseOptionalAttrDict(op.attributes);
            return true;
        }
        return super.parseOperation(parser, opName, op);
    }

    _parseEnterOp(parser, op) {
        while (parser.match('%')) {
            const operand = parser.expect('%');
            op.operands.push({ value: operand });
            if (!parser.accept(',')) {
                break;
            }
        }
        parser.expect('id', 'frame');
        const frameName = parser.expect('string');
        op.addAttribute('frame_name', frameName);
        if (parser.accept('id', 'parallel_iterations')) {
            const parallelIterations = parser.expect('int');
            op.addAttribute('parallel_iterations', parseInt(parallelIterations, 10));
        } else {
            op.addAttribute('parallel_iterations', 10);
        }
        const isConstant = parser.accept('id', 'constant');
        op.addAttribute('is_constant', isConstant);
        parser.expect(':');
        const type = parser.parseType();
        if (type instanceof mlir.FunctionType) {
            for (let i = 0; i < op.operands.length; i++) {
                const inputType = type.inputs[i];
                if (inputType) {
                    op.operands[i].type = inputType;
                }
            }
            for (const outputType of type.results) {
                op.results.push({ type: outputType });
            }
        } else {
            const typeStr = type.toString();
            if (op.operands.length > 0) {
                op.operands[0].type = typeStr;
            }
            for (let i = 1; i < op.operands.length; i++) {
                op.operands[i].type = '!tf_executor.control';
            }
            op.results.push({ type: typeStr });
            op.results.push({ type: '!tf_executor.control' });
        }
        parser.parseOptionalAttrDict(op.attributes);
        return true;
    }

    _parseGraphOp(parser, op) {
        if (parser.match('{')) {
            const region = op.addRegion();
            parser.parseRegion(region);
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
        // or: tf_executor.island(%control_inputs) {...}
        if (parser.match('(')) {
            op.operands = parser.parseArguments();
        }
        if (parser.accept('id', 'wraps')) {
            const wrappedOp = parser.parseGenericOperation();
            op.addAttribute('wrappedOp', wrappedOp);
        } else if (parser.match('{')) {
            const region = op.addRegion();
            parser.parseRegion(region);
        }
        parser.parseOptionalAttrDict(op.attributes);
        return true;
    }
};

mlir.TFFrameworkDialect = class extends mlir.Dialect {

    constructor(operations) {
        super(operations, 'tf_framework');
    }
};

mlir.TFRDialect = class extends mlir.Dialect {

    constructor(operations) {
        super(operations, 'tfr');
    }

    parseOperation(parser, opName, op) {
        if (opName === 'tfr.func') {
            parser.parseFunctionOp(op, false);
            return true;
        }
        return super.parseOperation(parser, opName, op);
    }
};

mlir.CoreRTDialect = class extends mlir.Dialect {

    constructor(operations) {
        super(operations, 'corert');
    }

    parseOperation(parser, opName, op) {
        const opInfo = this.getOperation(opName);
        if (!opInfo) {
            return false;
        }
        if (opName === 'corert.executeop' || opName === 'corert.executeop.seq') {
            const opHandler = parser.parseArguments();
            op.operands.push(...opHandler);
            const opNameAttr = parser.expect('string');
            op.addAttribute('op_name', opNameAttr);
            const operands = parser.parseArguments();
            op.operands.push(...operands);
            if (parser.match('{')) {
                parser.parseAttributeDict(op.attributes);
            }
            if (parser.accept(':')) {
                const resultCount = parser.expect();
                op.addAttribute('result_count', parseInt(resultCount, 10));
            }
            return true;
        }
        return super.parseOperation(parser, opName, op);
    }
};

mlir.TFRTDialect = class extends mlir.Dialect {

    constructor(operations) {
        super(operations, 'tfrt');
    }

    parseType(parser, dialectName) {
        const typeName = parser.parseKeyword();
        if (!typeName) {
            return null;
        }
        let type = `!${dialectName}.${typeName}`;
        const simpleTypes = ['chain', 'string', 'dist_context', 'device', 'tensor_type'];
        if (simpleTypes.includes(typeName)) {
            return new mlir.Type(type);
        }
        if (typeName === 'tensor') {
            if (parser.match('<')) {
                const content = parser.skip('<');
                type += content;
            }
            return new mlir.Type(type);
        }
        // Fallback for unknown tfrt types
        if (parser.match('<')) {
            type += parser.skip('<');
        }
        return new mlir.Type(type);
    }

    parseOperation(parser, opName, op) {
        const opInfo = this.getOperation(opName);
        if (!opInfo) {
            return false;
        }
        if (opInfo.metadata?.assemblyFormat === 'operands attr-dict') {
            while (parser.match('%')) {
                op.operands.push({ value: parser.expect() });
                if (!parser.accept(',')) {
                    break;
                }
            }
            if (parser.match('{')) {
                parser.parseAttributeDict(op.attributes);
            }
            return true;
        }
        if (opName === 'tfrt.call') {
            parser.parseSymbolName('callee', op.attributes);
            op.operands = parser.parseArguments();
            parser.parseOptionalAttrDict(op.attributes);
            if (parser.accept(':')) {
                const type = parser.parseFunctionType();
                if (type && type.results) {
                    type.results.forEach((resultType) => {
                        op.results.push({ type: resultType });
                    });
                }
            }
            return true;
        }
        if (opName === 'tfrt.return') {
            if (!parser.match('keyword', 'loc') && !parser.match('eof')) {
                op.operands = parser.parseArguments();
                // Reference: parseOptionalColonTypeList
                parser.resolveOperands(op.operands, parser.parseOptionalColonTypeList());
            }
            return true;
        }
        if (opName === 'tfrt.repeat.i32') {
            if (parser.match('%')) {
                op.operands = parser.parseArguments();
            }
            if (parser.accept(':')) {
                while (!parser.match('{') && !parser.match('eof')) {
                    parser.expect();
                }
            }
            if (parser.match('{')) {
                const region = op.addRegion();
                parser.parseRegion(region);
            }
            return true;
        }
        if (opName === 'tfrt.if') {
            op.operands = parser.parseArguments();
            if (parser.accept('id', 'attributes')) {
                parser.parseOptionalAttrDict(op.attributes);
            }
            if (parser.accept(':')) {
                const funcType = parser.parseFunctionType();
                if (funcType && funcType.results) {
                    for (const resultType of funcType.results) {
                        op.results.push({ type: resultType });
                    }
                }
            }
            if (parser.match('{')) {
                const thenRegion = {};
                parser.parseRegion(thenRegion);
                op.regions.push(thenRegion);
            }
            if (parser.accept('id', 'else')) {
                if (parser.match('{')) {
                    const elseRegion = {};
                    parser.parseRegion(elseRegion);
                    op.regions.push(elseRegion);
                }
            }
            return true;
        }
        if (opName === 'tfrt.parallel_for.i32') {
            const start = parser.expect('%');
            op.operands.push({ value: start, type: 'i32' });
            parser.expect('id', 'to');
            const end = parser.expect('%');
            op.operands.push({ value: end, type: 'i32' });
            parser.expect('id', 'fixed');
            const blockSize = parser.expect('%');
            op.operands.push({ value: blockSize, type: 'i32' });
            if (parser.accept(',')) {
                const additionalOperands = parser.parseArguments();
                op.operands.push(...additionalOperands);
            }
            const types = parser.parseOptionalColonTypeList();
            for (let i = 3; i < op.operands.length && i - 3 < types.length; i++) {
                op.operands[i].type = types[i - 3];
            }
            op.results.push({ type: '!tfrt.chain' });
            if (parser.match('{')) {
                const region = op.addRegion();
                parser.parseRegion(region);
            }
            return true;
        }
        if (opName === 'tfrt.parallel_call.i32') {
            const start = parser.expect('%');
            op.operands.push({ value: start, type: 'i32' });
            parser.expect('id', 'to');
            const end = parser.expect('%');
            op.operands.push({ value: end, type: 'i32' });
            parser.expect('id', 'fixed');
            const blockSize = parser.expect('%');
            op.operands.push({ value: blockSize, type: 'i32' });
            const callee = parser.expect('@');
            op.addAttribute('callee', callee);
            const additionalOperands = parser.parseArguments();
            op.operands.push(...additionalOperands);
            const types = parser.parseOptionalColonTypeList();
            for (let i = 3; i < op.operands.length && i - 3 < types.length; i++) {
                op.operands[i].type = types[i - 3];
            }
            op.results.push({ type: '!tfrt.chain' });
            return true;
        }
        if (opName === 'tfrt.while') {
            // Format: $cond $body_fn `(` $arguments `)` attr-dict? `parallel_iterations` `(` $parallel_iterations `)` `:` `(` type($arguments) `)` `->` `(` type(results) `)`
            const cond = parser.parseOperand();
            op.operands.push(cond);
            const bodyFn = parser.expect('@');
            op.addAttribute('body_fn', bodyFn);
            parser.expect('(');
            const args = [];
            while (!parser.match(')')) {
                const arg = parser.parseOperand();
                args.push(arg);
                if (!parser.accept(',')) {
                    break;
                }
            }
            parser.expect(')');
            op.operands.push(...args);
            // Optional attr-dict
            if (parser.match('{')) {
                parser.parseAttributeDict(op.attributes);
            }
            // Parse parallel_iterations(N)
            if (parser.accept('id', 'parallel_iterations')) {
                parser.expect('(');
                const parallelIterations = parser.expect('int');
                op.addAttribute('parallel_iterations', parseInt(parallelIterations, 10));
                parser.expect(')');
            }
            // Parse : (types) -> (types)
            parser.expect(':');
            const inputTypes = parser.parseTypeListParens();
            for (let i = 0; i < args.length && i < inputTypes.length; i++) {
                args[i].type = inputTypes[i];
            }
            cond.type = 'i1';
            parser.expect('->');
            const resultTypes = parser.parseTypeListParens();
            for (const resultType of resultTypes) {
                op.results.push({ type: resultType.toString() });
            }
            return true;
        }
        return super.parseOperation(parser, opName, op);
    }
};

mlir.TFRTFallbackAsyncDialect = class extends mlir.Dialect {

    constructor(operations) {
        super(operations, 'tfrt_fallback_async');
    }

    parseOperation(parser, opName, op) {
        const opInfo = this.getOperation(opName);
        if (!opInfo) {
            return false;
        }
        if (opName === 'tfrt_fallback_async.batch_function') {
            parser.expect('id', 'device');
            parser.expect('(');
            const device = parser.expect('string');
            parser.expect(')');
            op.addAttribute('device', device);
            const funcName = parser.expect('@');
            op.addAttribute('f', funcName);
            const operands = parser.parseArguments();
            op.operands.push(...operands);
            if (parser.match('{')) {
                parser.parseAttributeDict(op.attributes);
                if (parser.match('{')) {
                    parser.parseAttributeDict(op.attributes);
                }
            }
            if (parser.accept(':')) {
                const resultCount = parseInt(parser.expect(), 10);
                for (let i = 0; i < resultCount; i++) {
                    op.results.push({ type: '!tfrt_fallback.tf_tensor' });
                }
            }
            return true;
        }
        if (opName === 'tfrt_fallback_async.createop' || opName.startsWith('tfrt_fallback_async.executeop')) {
            const isCreateOp = opName === 'tfrt_fallback_async.createop';
            const hasChain = isCreateOp || opName.includes('.seq');
            const hasAllocator = opName.includes('.allocator');
            if ((hasChain || hasAllocator) && parser.match('(')) {
                const chainOperands = parser.parseArguments();
                op.operands.push(...chainOperands);
            }
            while (!parser.match(':') && !parser.match('{')) {
                if (parser.match('id')) {
                    const key = parser.expect('id');
                    if (parser.accept('(')) {
                        const value = parser.expect();
                        parser.expect(')');
                        op.addAttribute(key, value);
                    }
                } else if (parser.match('string')) {
                    const opNameAttr = parser.expect('string');
                    op.addAttribute('op_name', opNameAttr);
                    const operands = parser.parseArguments();
                    op.operands.push(...operands);
                    break;
                } else {
                    break;
                }
            }
            if (parser.match('{')) {
                parser.parseAttributeDict(op.attributes);
                if (parser.match('{')) {
                    parser.parseAttributeDict(op.attributes);
                }
            }
            if (isCreateOp) {
                if (parser.match('id', 'num_args')) {
                    parser.expect('id');
                    parser.expect('(');
                    const numArgs = parser.expect();
                    parser.expect(')');
                    op.addAttribute('num_args', parseInt(numArgs, 10));
                }
            } else if (parser.accept(':')) {
                const resultCount = parser.expect();
                op.addAttribute('result_count', parseInt(resultCount, 10));
            }
            return true;
        }
        return super.parseOperation(parser, opName, op);
    }
};

mlir.TileDialect = class extends mlir.Dialect {

    constructor(operations) {
        super(operations, 'tile');
    }

    parseOperation(parser, opName, op) {
        // tile.contract has format: tile.contract agg, combo, operands... attributes : types -> result
        // Example: %1 = tile.contract add, mul, %0, %arg0, %arg1 {sink = #map0, srcs = [#map1, #map2]} : tensor<f32>, tensor<1x256xf32>, tensor<256x512xf32> -> tensor<1x512xf32>
        if (opName === 'tile.contract') {
            // Parse aggregation kind (add, mul, etc.)
            if (parser.match('id')) {
                const agg = parser.expect('id');
                op.addAttribute('agg', agg);
            }
            parser.accept(',');
            // Parse combination kind (add, mul, etc.)
            if (parser.match('id')) {
                const combo = parser.expect('id');
                op.addAttribute('combo', combo);
            }
            parser.accept(',');
            op.operands = parser.parseArguments();
            if (parser.match('{')) {
                parser.parseAttributeDict(op.attributes);
            }
            parser.resolveOperands(op.operands, parser.parseOptionalColonTypeList());
            if (parser.accept('->')) {
                const resultTypes = parser.parseFunctionResultTypes();
                parser.resolveOperands(op.results, resultTypes);
            }
            return true;
        }
        return super.parseOperation(parser, opName, op);
    }
};

mlir.PXADialect = class extends mlir.Dialect {

    constructor(operations) {
        super(operations, 'pxa');
    }

    parseOperation(parser, opName, op) {
        if (opName === 'pxa.reduce' || opName === 'pxa.vector_reduce') {
            const agg = parser.expect('id');
            op.addAttribute('agg', agg);
            const val = { value: parser.expect('%') };
            op.operands.push(val);
            parser.accept(',');
            const memref = { value: parser.expect('%') };
            op.operands.push(memref);
            parser.skip('[');
            parser.parseOptionalAttrDict(op.attributes);
            if (parser.accept(':')) {
                const type = parser.parseType();
                memref.type = type;
                op.results.push({ type });
                if (opName === 'pxa.vector_reduce' && parser.accept(',')) {
                    const vectorType = parser.parseType();
                    val.type = vectorType;
                }
            }
            return true;
        }
        if (opName === 'pxa.load' || opName === 'pxa.vector_load') {
            const memref = { value: parser.expect('%') };
            op.operands.push(memref);
            parser.skip('[');
            parser.parseOptionalAttrDict(op.attributes);
            if (parser.accept(':')) {
                const type = parser.parseType();
                memref.type = type;
                if (opName === 'pxa.vector_load' && parser.accept(',')) {
                    const vectorType = parser.parseType();
                    op.results.push({ type: vectorType });
                }
            }
            return true;
        }
        if (opName === 'pxa.generic') {
            if (parser.accept('(')) {
                while (!parser.match(')')) {
                    const operand = { value: parser.expect('%') };
                    op.operands.push(operand);
                    if (parser.match('[')) {
                        parser.skip('[');
                    }
                    if (parser.accept(':')) {
                        parser.expect('#');  // Skip affine map reference
                    }
                    if (!parser.accept(',')) {
                        break;
                    }
                }
                parser.expect(')');
            }
            if (parser.accept('<')) {
                const reduction = parser.expect('id');
                op.addAttribute('reduction', reduction);
                parser.expect('>');
            }
            if (parser.match('@')) {
                op.addAttribute('kernel', parser.expect('@'));
            }
            if (parser.accept('(')) {
                while (!parser.match(')')) {
                    const operand = { value: parser.expect('%') };
                    op.operands.push(operand);
                    if (parser.match('[')) {
                        parser.skip('[');
                    }
                    if (parser.accept(':')) {
                        parser.expect('#');  // Skip affine map reference
                    }
                    if (!parser.accept(',')) {
                        break;
                    }
                }
                parser.expect(')');
            }
            if (parser.accept('id', 'tile')) {
                parser.expect(':');
                const tile = parser.skip('[');
                op.addAttribute('tile', tile);
            }
            parser.parseOptionalAttrDict(op.attributes);
            if (parser.accept(':')) {
                const funcType = parser.parseFunctionType();
                if (funcType && funcType.results) {
                    for (const resultType of funcType.results) {
                        op.results.push({ type: resultType });
                    }
                }
            }
            return true;
        }
        return super.parseOperation(parser, opName, op);
    }
};

mlir.ToyDialect = class extends mlir.Dialect {

    constructor(operations) {
        super(operations, 'toy');
    }

    parseOperation(parser, opName, op) {
        if (opName === 'toy.func') {
            parser.parseFunctionOp(op, false);
            return true;
        }
        if (opName === 'toy.constant') {
            const value = parser.parseAttribute();
            op.addAttribute('value', value.value === undefined ? value : value.value);
            // Reference: parseOptionalColonTypeList
            parser.resolveOperands(op.results, parser.parseOptionalColonTypeList());
            return true;
        }
        if (opName === 'toy.mul' || opName === 'toy.add') {
            op.operands = parser.parseArguments();
            parser.parseOptionalAttrDictWithKeyword(op.attributes);
            if (parser.accept(':')) {
                // The type signature can be either:
                // 1. Simple: `: tensor<*xf64>` - just the result type
                // 2. Function-style: `: (tensor<*xf64>, tensor<*xf64>) -> tensor<*xf64>`
                const type = parser.parseType();
                if (type instanceof mlir.FunctionType) {
                    // Function-style type signature
                    for (let i = 0; i < op.operands.length; i++) {
                        const inputType = type.getInput(i);
                        if (inputType) {
                            op.operands[i].type = inputType;
                        }
                    }
                    // Add result type to existing result (which already has SSA name)
                    if (op.results.length > 0) {
                        const resultType = type.getResult(0);
                        if (resultType) {
                            op.results[0].type = resultType;
                        }
                    }
                } else {
                    // Simple type signature - type applies to both operands and result
                    const typeStr = type.toString();
                    for (const operand of op.operands) {
                        operand.type = typeStr;
                    }
                    if (op.results.length > 0) {
                        op.results[0].type = typeStr;
                    }
                }
            }
            return true;
        }
        return super.parseOperation(parser, opName, op);
    }

};

mlir.SdfgDialect = class extends mlir.Dialect {

    constructor(operations) {
        super(operations, 'sdfg');
    }

    parseType(parser, dialectName) {
        const typeName = parser.parseKeyword();
        if (!typeName) {
            return null;
        }
        let type = `!${dialectName}.${typeName}`;
        if (typeName === 'stream' && parser.match('_')) {
            parser.expect('_');
            const suffix = parser.expect('id');
            if (suffix === 'array') {
                type += `_${suffix}`;
            }
        }
        if (typeName === 'array' || typeName === 'stream' || typeName === 'memlet' || type.endsWith('stream_array')) {
            if (parser.match('<')) {
                const content = parser.skip('<');
                type += content;
            }
            return new mlir.Type(type);
        }
        return null;
    }

    parseOperation(parser, opName, op) {
        if (opName === 'sdfg.sdfg' || opName === 'sdfg.nested_sdfg' || opName === 'sdir.sdfg') {
            parser.parseOptionalAttrDict(op.attributes);
            const inputResult = parser.parseFunctionArgumentList();
            const inputs = inputResult.arguments.map((a) => a.type);
            parser.parseOptionalAttrDictWithKeyword(op.attributes);
            let results = [];
            if (parser.accept('->')) {
                const outputResult = parser.parseFunctionArgumentList();
                results = outputResult.arguments.map((a) => a.type);
            }
            op.addAttribute('function_type', new mlir.TypeAttrOf(new mlir.FunctionType(inputs, results)));
            parser.parseOptionalAttrDictWithKeyword(op.attributes);
            if (parser.match('{')) {
                const region = op.addRegion();
                parser.parseRegion(region);
            }
            return true;
        }
        if (opName === 'sdfg.tasklet' || opName === 'sdir.tasklet') {
            parser.parseOptionalAttrDict(op.attributes);
            if (parser.match('@')) {
                parser.parseSymbolName('sym_name', op.attributes);
            }
            const blockArgs = [];
            if (parser.accept('(')) {
                while (!parser.accept(')')) {
                    const operand = parser.expect('%');
                    let blockArgName = operand;
                    let type = null;
                    if (parser.accept('id', 'as')) {
                        blockArgName = parser.expect('%');
                        parser.expect(':');
                        type = parser.parseType();
                    } else {
                        parser.expect(':');
                        type = parser.parseType();
                    }
                    op.operands.push({ value: operand, type });
                    blockArgs.push({ value: blockArgName, type });
                    parser.accept(',');
                }
            }
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
            if (parser.match('{')) {
                const region = op.addRegion();
                parser.parseRegion(region, blockArgs);
            }
            return true;
        }
        if (opName === 'sdfg.consume') {
            parser.parseOptionalAttrDict(op.attributes);
            if (parser.match('(')) {
                op.operands = parser.parseArguments();
            }
            parser.parseOptionalAttrDictWithKeyword(op.attributes);
            if (parser.accept('->')) {
                if (parser.accept('(')) {
                    while (!parser.accept(')')) {
                        if (parser.match('id')) {
                            const name = parser.expect('id');
                            if (parser.accept(':')) {
                                const value = parser.expect('%');
                                op.results.push({ name, value });
                            }
                        } else if (parser.match('%') || parser.match(')')) {
                            break;
                        } else {
                            throw new mlir.Error(`Expected named result in sdfg.consume but got '${parser.getToken().value}' ${parser.location()}`);
                        }
                        parser.accept(',');
                    }
                }
            }
            parser.parseOptionalAttrDictWithKeyword(op.attributes);
            if (parser.match('{')) {
                const region = op.addRegion();
                parser.parseRegion(region);
            }
            return true;
        }
        if (opName === 'sdfg.state' || opName === 'sdir.state') {
            parser.parseOptionalAttrDict(op.attributes);
            if (parser.match('@')) {
                parser.parseSymbolName('sym_name', op.attributes);
            }
            parser.parseOptionalAttrDictWithKeyword(op.attributes);
            const region = op.addRegion();
            parser.parseRegion(region);
            return true;
        }
        if (opName === 'sdfg.alloc' || opName === 'sdir.alloc' || opName === 'sdir.alloc_transient' || opName === 'sdir.alloc_stream') {
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
        if (opName === 'sdfg.store' || opName === 'sdir.store') {
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
        if (opName === 'sdfg.load' || opName === 'sdir.load') {
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
        if (opName === 'sdfg.map' || opName === 'sdir.map') {
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
                parser.skip('(');
            }
            if (parser.match('id', 'to')) {
                parser.accept('id', 'to');
                parser.skip('(');
            }
            if (parser.match('id', 'step')) {
                parser.accept('id', 'step');
                parser.skip('(');
            }
            parser.parseOptionalAttrDictWithKeyword(op.attributes);
            if (parser.match('{')) {
                const region = op.addRegion();
                parser.parseRegion(region);
            }
            return true;
        }
        if (opName === 'sdfg.consume' || opName === 'sdir.consume') {
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
                const region = op.addRegion();
                parser.parseRegion(region);
            }
            return true;
        }
        if (opName === 'sdfg.edge' || opName === 'sdir.edge') {
            parser.parseOptionalAttrDict(op.attributes);
            if (parser.match('(')) {
                op.operands = parser.parseArguments();
            }
            parser.parseOptionalAttrDictWithKeyword(op.attributes);
            if (parser.match('@')) {
                const src = parser.expect('@');
                op.addAttribute('src', src);
            }
            parser.accept('->');
            if (parser.match('@')) {
                const dst = parser.expect('@');
                op.addAttribute('dst', dst);
            }
            return true;
        }
        if (opName === 'sdfg.sym' || opName === 'sdir.sym') {
            if (parser.accept('(')) {
                const expr = parser.expect('string');
                op.addAttribute('expr', expr);
                parser.accept(')');
            }
            parser.parseOptionalAttrDictWithKeyword(op.attributes);
            if (parser.accept(':')) {
                const type = parser.parseType();
                op.results.push({ type });
            }
            return true;
        }
        if (opName === 'sdfg.copy' || opName === 'sdir.copy') {
            parser.parseOptionalAttrDict(op.attributes);
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
        if (opName === 'sdfg.libcall' || opName === 'sdir.libcall') {
            parser.parseOptionalAttrDict(op.attributes);
            if (parser.match('string')) {
                const libname = parser.expect('string');
                op.addAttribute('libname', libname);
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
        if (opName === 'sdfg.get_access' || opName === 'sdir.get_access') {
            if (parser.match('%')) {
                const value = parser.expect();
                op.operands.push({ value });
            }
            if (parser.accept(':')) {
                const inputType = parser.parseType();
                if (op.operands.length > 0) {
                    op.operands[0].type = inputType;
                }
                if (parser.accept('->')) {
                    const resultType = parser.parseType();
                    op.results.push({ type: resultType });
                }
            }
            return true;
        }
        if (opName === 'sdir.call') {
            const callee = parser.parseOptionalSymbolName();
            if (callee) {
                op.addAttribute('callee', callee);
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
        if (opName === 'sdfg.alloc_symbol' || opName === 'sdir.alloc_symbol') {
            if (parser.accept('(')) {
                const sym = parser.expect('string');
                op.addAttribute('sym', sym);
                parser.accept(')');
            }
            return true;
        }
        if (opName === 'sdfg.return') {
            if (parser.match('%')) {
                op.operands = parser.parseArguments();
                // Reference: parseOptionalColonTypeList
                parser.resolveOperands(op.operands, parser.parseOptionalColonTypeList());
            }
            return true;
        }
        if (opName === 'sdfg.stream_push' || opName === 'sdir.stream_push') {
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
        if (opName === 'sdfg.stream_pop' || opName === 'sdir.stream_pop') {
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
        if (opName === 'sdfg.stream_length' || opName === 'sdir.stream_length') {
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
        if (opName === 'sdfg.view_cast' || opName === 'sdir.view_cast') {
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
        if (opName === 'sdfg.subview' || opName === 'sdir.subview') {
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

mlir.TFLDialect = class extends mlir.Dialect {

    constructor(operations) {
        super(operations, 'tfl');
        // Operations that use parseOneResultSameOperandTypeOp in tfl_ops.cc
        // Format: operands attr-dict : single-type
        this._binaryOps = new Set([
            'add', 'sub', 'mul', 'div', 'floor_div', 'pow', 'squared_difference',
            'less', 'less_equal', 'greater', 'greater_equal', 'not_equal',
            'logical_and', 'logical_or'
        ]);
    }

    parseOperation(parser, opName, op) {
        const opKind = opName.substring('tfl.'.length);
        if (opKind === 'control_node') {
            if (parser.accept('(')) {
                parser.parseOptionalSSAUseList(op.operands);
                parser.expect(')');
            }
            if (parser.accept('id', 'controls')) {
                const region = { blocks: [{ operations: [] }] };
                const innerOp = parser.parseGenericOperation();
                region.blocks[0].operations.push(innerOp);
                op.regions.push(region);
            } else if (parser.match('{')) {
                const region = op.addRegion();
                parser.parseRegion(region);
            }
            parser.parseOptionalAttrDict(op.attributes);
            return true;
        }
        if (this._binaryOps.has(opKind)) {
            // Parse: operands attr-dict : type (compact form)
            // Or: (operands) <properties> : fn-type (generic form)
            if (parser.match('(')) {
                parser.expect('(');
                parser.parseOptionalSSAUseList(op.operands);
                parser.expect(')');
                parser.parsePropertyDict(op.attributes);
                parser.parseOptionalAttrDict(op.attributes);
                if (parser.accept(':')) {
                    const fnType = parser.parseType();
                    if (fnType instanceof mlir.FunctionType) {
                        for (let i = 0; i < fnType.inputs.length && i < op.operands.length; i++) {
                            op.operands[i].type = fnType.inputs[i].toString();
                        }
                        for (let i = 0; i < fnType.results.length; i++) {
                            op.results.push({ type: fnType.results[i].toString() });
                        }
                    } else {
                        for (const operand of op.operands) {
                            operand.type = fnType;
                        }
                        op.results.push({ type: fnType });
                    }
                }
                return true;
            }
            op.operands = parser.parseArguments();
            parser.parseOptionalAttrDict(op.attributes);
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

        return super.parseOperation(parser, opName, op);
    }
};

mlir.TFDialect = class extends mlir.Dialect {

    constructor(operations) {
        super(operations, 'tf');
    }

    parseType(parser, dialectName) {
        const typeName = parser.parseKeyword();
        if (!typeName) {
            return null;
        }
        let type = `!${dialectName}.${typeName}`;
        if (typeName === 'resource' || typeName === 'variant') {
            if (parser.match('<')) {
                const content = parser.skip('<');
                type += content;
            }
            return new mlir.Type(type);
        }
        if (typeName === 'string' || typeName === 'control') {
            return new mlir.Type(type);
        }
        return null;
    }
};

mlir.TFTypeDialect = class extends mlir.Dialect {

    constructor(operations) {
        super(operations, 'tf_type');
        this.simpleTypes = new Set([
            'string', 'qint8', 'qint16', 'qint32', 'quint8', 'quint16',
            'f32ref', 'f64ref', 'uint4ref', 'int4ref', 'uint8ref', 'int8ref',
            'uint16ref', 'int16ref', 'uint32ref', 'int32ref', 'uint64ref', 'int64ref',
            'stringref', 'boolref', 'quint8ref', 'qint8ref', 'quint16ref', 'qint16ref',
            'qint32ref', 'bfloat16ref', 'complex64ref', 'complex128ref', 'halfref',
            'resourceref', 'variantref',
            'float8e4m3fnref', 'float8e5m2ref', 'float8e4m3fnuzref',
            'float8e4m3b11fnuzref', 'float8e5m2fnuzref'
        ]);
    }

    parseType(parser, dialectName) {
        const typeName = parser.parseKeyword();
        if (!typeName) {
            return null;
        }
        let type = `!${dialectName}.${typeName}`;
        // Handle parametrized types like resource<>, variant<>, resource_handle<>
        if (typeName === 'resource' || typeName === 'variant' || typeName === 'resource_handle') {
            if (parser.accept('<')) {
                const subtypes = [];
                while (!parser.match('>')) {
                    subtypes.push(parser.parseType());
                    parser.accept(',');
                }
                parser.expect('>');
                return new mlir.Type(`${type}<${subtypes.join(', ')}>`);
            }
            return new mlir.Type(type);
        }
        if (this.simpleTypes.has(typeName)) {
            return new mlir.Type(type);
        }
        // Fallback for unknown tf_type types
        if (parser.match('<')) {
            type += parser.skip('<');
        }
        return new mlir.Type(type);
    }
};

mlir.CheckDialect = class extends mlir.Dialect {
    constructor(operations) {
        super(operations, 'check');
        // Workaround: Handle conflicting dialects from stablehlo and iree
        for (const [name] of this._operations.entries()) {
            this._operations.set(name.replace(/<(stablehlo|iree)>\./, ''), { metadata: {} });
        }
    }

    parseOperation(parser, opName, op) {
        // Workaround: Handle conflicting dialects from stablehlo and iree
        let dialect = 'stablehlo';
        if (parser.match('(') || parser.match('<')) {
            dialect = 'iree';
        }
        opName = opName.replace('check.', `check.<${dialect}>.`);
        return super.parseOperation(parser, opName, op);
    }
};

mlir.TransformDialect = class extends mlir.Dialect {

    constructor(operations) {
        super(operations, 'transform');
        this.registerCustomDirective('PackedOrDynamicIndexList', this._parsePackedOrDynamicIndexList.bind(this));
        this.registerCustomDirective('SemiFunctionType', this._parseSemiFunctionType.bind(this));
        this.registerCustomDirective('SequenceOpOperands', this._parseSequenceOpOperands.bind(this));
        this.registerCustomDirective('ForeachMatchSymbols', this._parseForeachMatchSymbols.bind(this));
        this.registerCustomDirective('TransformMatchDims', this._parseTransformMatchDims.bind(this));
        this.registerCustomDirective('ApplyRegisteredPassOptions', this._parseApplyRegisteredPassOptions.bind(this));
        this.registerCustomDirective('AlternativesOpSelectedRegion', this._parseAlternativesOpSelectedRegion.bind(this));
        this.registerCustomDirective('ContinuousTileSizeTypes', this._parseContinuousTileSizeTypes.bind(this));
        this.registerCustomDirective('MultitileSizesTypes', this._parseMultitileSizesTypes.bind(this));
    }

    parseOperation(parser, opName, op) {
        if (opName === 'transform.named_sequence') {
            return this._parseNamedSequenceOp(parser, op);
        }
        // C++-only operation: transform.test_transform_op ["message"]
        // Defined in mlir/test/lib/Dialect/Transform/TestTransformDialectExtension.cpp
        if (opName === 'transform.test_transform_op') {
            if (parser.match('string')) {
                const message = parser.expect('string');
                op.addAttribute('message', message);
            }
            return true;
        }
        // LinalgTransformOps.cpp:3009 SplitOp::parse
        // Format: %target after (%dynamic_chunk | static_int) attr-dict : target_type [, chunk_type]
        if (opName === 'transform.structured.split') {
            const target = parser.expect('%');
            op.operands.push({ value: target });
            parser.expect('id', 'after');
            let hasDynamicChunk = false;
            if (parser.match('%')) {
                const dynamicChunk = parser.expect('%');
                op.operands.push({ value: dynamicChunk });
                hasDynamicChunk = true;
            } else {
                const staticChunkSizes = parser.parseInteger();
                op.addAttribute('static_chunk_sizes', staticChunkSizes);
            }
            parser.parseOptionalAttrDict(op.attributes);
            parser.expect(':');
            const targetType = parser.parseType();
            op.operands[0].type = targetType;
            op.results.push({ type: targetType });
            if (hasDynamicChunk && parser.accept(',')) {
                const chunkType = parser.parseType();
                op.operands[1].type = chunkType;
            }
            return true;
        }
        return super.parseOperation(parser, opName, op);
    }

    _parseSequenceOpOperands(parser, op /*, args */) {
        if (parser.match('%')) {
            const rootOperand = parser.expect('%');
            op.operands.push({ value: rootOperand });
            if (parser.accept(',')) {
                while (parser.match('%')) {
                    const extraOperand = parser.expect('%');
                    op.operands.push({ value: extraOperand });
                    if (!parser.accept(',')) {
                        break;
                    }
                }
            }
        }
        if (parser.accept(':')) {
            parser.accept('(');
            const types = parser.parseTypeListNoParens();
            parser.resolveOperands(op.operands, types);
            parser.accept(')');
        }
    }

    _parseForeachMatchSymbols(parser, op, args) {
        const matchers = [];
        const actions = [];
        do {
            const matcher = parser.expect('@');
            parser.expect('->');
            const action = parser.expect('@');
            matchers.push(matcher);
            actions.push(action);
        } while (parser.accept(','));
        const matchersAttr = args[0].replace('$', '');
        const actionsAttr = args[1].replace('$', '');
        op.addAttribute(matchersAttr, matchers);
        op.addAttribute(actionsAttr, actions);
    }

    _parseTransformMatchDims(parser, op, args) {
        if (parser.accept('id', 'all')) {
            const allAttr = args[2].replace('$', '');
            op.addAttribute(allAttr, true);
            return;
        }
        const isInverted = parser.accept('id', 'except');
        if (isInverted) {
            parser.expect('(');
        }
        const dims = [];
        do {
            if (parser.match('int')) {
                dims.push(parser.parseInteger());
            }
        } while (parser.accept(','));
        if (isInverted) {
            parser.expect(')');
            const invertedAttr = args[1].replace('$', '');
            op.addAttribute(invertedAttr, true);
        }
        const dimsAttr = args[0].replace('$', '');
        op.addAttribute(dimsAttr, dims);
    }

    parseType(parser, dialectName) {
        const typeName = parser.parseKeyword();
        if (!typeName) {
            return null;
        }
        let type = `!${dialectName}.${typeName}`;
        if (typeName === 'any' && parser.match('_')) {
            parser.expect('_');
            const suffix = parser.expect('id');
            type += `_${suffix}`;
        }
        if (parser.match('<')) {
            const content = parser.skip('<');
            type += content;
        }
        return new mlir.Type(type);
    }

    _parseNamedSequenceOp(parser, op) {
        parser.parseOptionalVisibilityKeyword(op.attributes);
        parser.parseSymbolName('sym_name', op.attributes);
        const argResult = parser.parseFunctionArgumentList();
        const inputs = argResult.arguments.map((a) => a.type);
        const results = [];
        const resultAttrs = [];
        if (parser.accept('->')) {
            parser.parseFunctionResultList(results, resultAttrs);
        }
        op.addAttribute('function_type', new mlir.TypeAttrOf(new mlir.FunctionType(inputs, results)));
        parser.parseOptionalAttrDictWithKeyword(op.attributes);
        if (parser.match('{')) {
            const region = op.addRegion();
            parser.parseRegion(region);
        }
        return true;
    }

    _parseSemiFunctionType(parser, op /* , args */) {
        // Reference: Syntax.cpp parseSemiFunctionType
        // Format: type OR (type) -> result_types
        const hasLParen = parser.accept('(');
        // Parse the argument type (first operand type)
        const argType = parser.parseType();
        if (op.operands.length > 0) {
            op.operands[0].type = argType;
        }
        if (!hasLParen) {
            return;
        }
        parser.expect(')');
        parser.expect('->');
        // Handle both single type and parenthesized type list
        if (parser.accept('(')) {
            let idx = 0;
            while (!parser.match(')')) {
                const type = parser.parseType();
                if (idx < op.results.length) {
                    op.results[idx].type = type;
                } else {
                    op.results.push({ type });
                }
                idx++;
                if (!parser.accept(',')) {
                    break;
                }
            }
            parser.expect(')');
        } else {
            const type = parser.parseType();
            if (op.results.length > 0) {
                op.results[0].type = type;
            } else {
                op.results.push({ type });
            }
        }
    }

    _parsePackedOrDynamicIndexList(parser, op, args) {
        const dynamicOperands = [];
        const staticValues = [];
        let packedOperand = null;

        // Check for packed syntax: *(%operand)
        if (parser.accept('keyword', '*')) {
            parser.expect('(');
            if (parser.match('%')) {
                packedOperand = parser.expect('%');
            }
            parser.expect(')');
        } else if (parser.accept('[')) {
            // List syntax: [int, %operand, int, ...]
            while (!parser.match(']')) {
                if (parser.match('%')) {
                    const value = parser.expect('%');
                    dynamicOperands.push(value);
                    staticValues.push(-9223372036854775808); // ShapedType::kDynamic
                    if (parser.accept(':')) {
                        parser.parseType();
                    }
                } else if (parser.match('int') || parser.match('number')) {
                    const intVal = parseInt(parser.expect(), 10);
                    staticValues.push(intVal);
                } else {
                    break;
                }
                parser.accept(',');
            }
            parser.expect(']');
        }
        // args[0] = $packed (packed operand name)
        // args[1] = $dynamic (variadic operands name)
        // args[2] = $static (static attribute name)
        if (packedOperand && args && args.length > 0) {
            const packedName = args[0].replace(/^\$/, '');
            op.operands.push({ name: packedName, value: packedOperand });
        }
        if (args && args.length > 1) {
            const operandName = args[1].replace(/^\$/, '');
            for (const value of dynamicOperands) {
                op.operands.push({ name: operandName, value });
            }
        }
        if (args && args.length > 2) {
            const staticAttrName = args[2].replace(/^\$/, '');
            if (staticValues.length > 0) {
                op.addAttribute(staticAttrName, staticValues);
            }
        }
    }

    _parseContinuousTileSizeTypes(parser, op) {
        const funcType = parser.parseType();
        if (funcType && funcType.value) {
            const match = funcType.value.match(/^\((.*?)\)\s*->\s*(.+)$/);
            if (match) {
                const [, inputType, resultType] = match;
                if (op.operands.length > 0) {
                    op.operands[0].type = new mlir.Type(inputType);
                }
                op.results.push({ type: new mlir.Type(resultType) });
                op.results.push({ type: new mlir.Type(resultType) });
            }
        }
    }

    _parseMultitileSizesTypes(parser, op) {
        const funcType = parser.parseType();
        if (funcType && funcType.value) {
            const match = funcType.value.match(/^\((.*?)\)\s*->\s*(.+)$/);
            if (match) {
                const [, inputType, resultType] = match;
                if (op.operands.length > 0) {
                    op.operands[0].type = new mlir.Type(inputType);
                }
                op.results.push({ type: new mlir.Type(resultType) });
                op.results.push({ type: new mlir.Type(resultType) });
                op.results.push({ type: new mlir.Type(resultType) });
            }
        }
    }

    _parseApplyRegisteredPassOptions(parser, op) {
        if (!parser.accept('{')) {
            return;
        }
        const options = {};
        while (!parser.match('}')) {
            const key = parser.match('string') ? parser.expect('string') : parser.parseKeyword();
            parser.expect('=');
            if (parser.match('%')) {
                const operand = parser.expect('%');
                op.operands.push({ value: operand });
                options[key] = `#transform.param_operand<${op.operands.length - 1}>`;
            } else if (parser.match('[')) {
                parser.accept('[');
                const arr = [];
                while (!parser.match(']')) {
                    if (parser.match('%')) {
                        const operand = parser.expect('%');
                        op.operands.push({ value: operand });
                        arr.push(`#transform.param_operand<${op.operands.length - 1}>`);
                    } else {
                        const val = parser.parseAttribute();
                        arr.push(val);
                    }
                    parser.accept(',');
                }
                parser.expect(']');
                options[key] = arr;
            } else {
                const value = parser.parseAttribute();
                options[key] = value;
            }
            parser.accept(',');
        }
        parser.expect('}');
        op.addAttribute('options', options);
    }

    _parseAlternativesOpSelectedRegion(parser, op) {
        if (parser.match('int')) {
            const value = parser.parseInteger();
            op.addAttribute('selected_region_attr', value);
        } else if (parser.match('%')) {
            const operand = parser.expect('%');
            op.operands.push({ value: operand });
        }
    }
};

mlir.TestDialect = class extends mlir.Dialect {

    constructor(operations) {
        super(operations, 'test');
        // Enum attribute parsers for test dialect
        this.registerCustomDirective('CustomOptionalOperand', this._parseCustomOptionalOperand.bind(this));
        this.registerCustomDirective('CustomDirectiveOperands', this._parseCustomDirectiveOperands.bind(this));
        this.registerCustomDirective('CustomDirectiveOperandsAndTypes', this._parseCustomDirectiveOperandsAndTypes.bind(this));
        this.registerCustomDirective('CustomDirectiveResults', this._parseCustomDirectiveResults.bind(this));
        this.registerCustomDirective('CustomDirectiveWithTypeRefs', this._parseCustomDirectiveWithTypeRefs.bind(this));
        this.registerCustomDirective('CustomDirectiveRegions', this._parseCustomDirectiveRegions.bind(this));
        this.registerCustomDirective('CustomDirectiveSuccessors', this._parseCustomDirectiveSuccessors.bind(this));
        this.registerCustomDirective('CustomDirectiveAttrDict', this._parseCustomDirectiveAttrDict.bind(this));
        this.registerCustomDirective('CustomDirectiveAttributes', this._parseCustomDirectiveAttributes.bind(this));
        this.registerCustomDirective('CustomDirectiveSpacing', this._parseCustomDirectiveSpacing.bind(this));
        this.registerCustomDirective('CustomDirectiveOptionalOperandRef', this._parseCustomDirectiveOptionalOperandRef.bind(this));
        this.registerCustomDirective('UsingPropertyInCustom', this._parseUsingPropertyInCustom.bind(this));
        this.registerCustomDirective('IntProperty', this._parseIntProperty.bind(this));
        this.registerCustomDirective('SumProperty', this._parseSumProperty.bind(this));
        this.registerCustomDirective('SwitchCases', this._parseSwitchCases.bind(this));
        this.registerCustomDirective('DimensionList', this._parseDimensionList.bind(this));
        this.registerCustomDirective('OptionalCustomParser', this._parseOptionalCustomParser.bind(this));
        this.registerCustomDirective('OptionalLoc', this._parseOptionalLoc.bind(this));
        this.registerCustomDirective('DummyRegionRef', this._parseDummyRegionRef.bind(this));
        this.registerCustomDirective('DummySuccessorRef', this._parseDummySuccessorRef.bind(this));
        this.registerCustomType('CompoundNestedOuterType', this._parseCompoundNestedOuterType.bind(this));
        this.registerCustomType('CompoundNestedInnerType', this._parseCompoundNestedInnerType.bind(this));
        this.registerCustomType('CompoundTypeA', this._parseCompoundTypeA.bind(this));
        this.registerCustomAttribute('TestBitEnumAttr', this._parseEnumFlagsAngleBracketComma.bind(this));
        this.registerCustomAttribute('TestBitEnumVerticalBarAttr', this._parseEnumFlagsAngleBracketPipe.bind(this));
        this.registerCustomAttribute('TestEnumAttr', this._parseTestEnumAttr.bind(this));
        this.registerCustomAttribute('TestEnumProp', this._parseTestEnumAttr.bind(this));
        this.registerCustomAttribute('TestEnumPropAttrForm', this._parseTestEnumPropAttrForm.bind(this));
        this.registerCustomAttribute('TestBitEnumProp', this._parseTestBitEnumProp.bind(this));
        this.registerCustomAttribute('TestBitEnumPropNamed', this._parseTestBitEnumPropNamed.bind(this));
    }

    parseOperation(parser, opName, op) {
        // test.conversion_func_op is a function-like operation with FunctionOpInterface
        // Parse it like func.func to handle argument and result attributes properly
        if (opName === 'test.conversion_func_op') {
            parser.parseFunctionOp(op, false);
            return true;
        }
        if (opName === 'test.region_if') {
            while (parser.match('%')) {
                op.operands.push({ value: parser.expect('%') });
                if (!parser.accept(',')) {
                    break;
                }
            }
            parser.expect(':');
            const inputTypes = parser.parseTypeList();
            parser.resolveOperands(op.operands, inputTypes);
            parser.expect('->');
            const outputTypes = parser.parseFunctionResultTypes();
            for (const t of outputTypes) {
                op.results.push({ type: t.toString() });
            }
            parser.expect('id', 'then');
            const thenRegion = {};
            parser.parseRegion(thenRegion);
            op.regions.push(thenRegion);
            parser.expect('id', 'else');
            const elseRegion = {};
            parser.parseRegion(elseRegion);
            op.regions.push(elseRegion);
            parser.expect('id', 'join');
            const joinRegion = {};
            parser.parseRegion(joinRegion);
            op.regions.push(joinRegion);
            return true;
        }
        if (opName === 'test.affine_scope' || opName === 'test.single_no_terminator_custom_asm_op') {
            const region = op.addRegion();
            parser.parseRegion(region);
            return true;
        }
        if (opName === 'test.with_nice_properties') {
            // PropertiesWithCustomPrint is a test-only type in MLIR's test dialect that exists
            // solely to test custom property print/parse. It uses format: "label" is <integer>
            // instead of the standard prop-dict <{...}> format. This is an exception, not a pattern.
            // Reference: llvm-project/mlir/test/lib/Dialect/Test/TestDialect.cpp customParseProperties
            this._operations.get(opName).hasParseOperation = false;
            const label = parser.match('string') ? parser.expect('string') : parser.expect('id');
            parser.expect('id', 'is');
            const negative = parser.accept('keyword', '-');
            const value = parser.parseInteger();
            op.addAttribute('prop', { label, value: negative ? -value : value });
            parser.parseOptionalAttrDict(op.attributes);
            return true;
        }
        // Test operation with default-valued properties and UnitProp
        // Format: <a> <b> <c> (unit|unit_absent) or just "na" for all defaults
        if (opName === 'test.with_default_valued_properties') {
            this._operations.get(opName).hasParseOperation = false;
            if (parser.accept('id', 'na')) {
                // All defaults
            } else {
                const a = parser.parseInteger();
                op.addAttribute('a', a);
                if (parser.match('string')) {
                    op.addAttribute('b', parser.expect('string'));
                }
                if (parser.match('int') || parser.match('keyword', '-')) {
                    const neg = parser.accept('keyword', '-');
                    const c = parser.parseInteger();
                    op.addAttribute('c', neg ? -c : c);
                }
                if (parser.accept('id', 'unit')) {
                    op.addAttribute('unit', true);
                } else if (parser.accept('id', 'unit_absent')) {
                    op.addAttribute('unit', false);
                }
            }
            parser.parseOptionalAttrDict(op.attributes);
            return true;
        }
        // Test operation with optional properties using some<...> syntax
        if (opName === 'test.with_optional_properties') {
            this._operations.get(opName).hasParseOperation = false;
            const parseOptionalValue = () => {
                if (parser.accept('id', 'some')) {
                    parser.expect('<');
                    let value = null;
                    if (parser.accept('id', 'none')) {
                        value = null;
                    } else if (parser.accept('id', 'unit')) {
                        value = true;
                    } else if (parser.match('string')) {
                        value = parser.expect('string');
                    } else {
                        const neg = parser.accept('keyword', '-');
                        value = parser.parseInteger();
                        if (neg) {
                            value = -value;
                        }
                    }
                    parser.expect('>');
                    return { some: value };
                }
                if (parser.match('string')) {
                    return parser.expect('string');
                }
                const neg = parser.accept('keyword', '-');
                const value = parser.parseInteger();
                return neg ? -value : value;
            };
            const knownAttrs = new Set(['anAttr', 'simple', 'simplei8', 'simpleui8', 'nonTrivialStorage', 'hasDefault', 'nested', 'longSyntax', 'hasUnit', 'maybeUnit']);
            while (parser.match('id') && !parser.match('{') && !parser.match('id', 'loc')) {
                const tokenValue = parser.getToken().value;
                // Stop if this looks like an operation name (not a known attribute)
                if (!knownAttrs.has(tokenValue)) {
                    break;
                }
                const name = parser.expect('id');
                if (name === 'hasUnit') {
                    op.addAttribute(name, true);
                } else if (parser.accept('=')) {
                    op.addAttribute(name, parseOptionalValue());
                } else {
                    break;
                }
            }
            parser.parseOptionalAttrDict(op.attributes);
            return true;
        }
        if (opName === 'test.wrapping_region' && this.hasCustomAssemblyFormat(opName) && !this.hasAssemblyFormat(opName)) {
            parser.expect('id', 'wraps');
            const region = op.addRegion();
            const block = { operations: [] };
            region.blocks = [block];
            const wrappedOp = parser.parseGenericOperation();
            block.operations.push(wrappedOp);
            if (wrappedOp.results) {
                for (const result of wrappedOp.results) {
                    op.results.push({ type: result.type });
                }
            }
            return true;
        }
        if (opName === 'test.pretty_printed_region' && this.hasCustomAssemblyFormat(opName) && !this.hasAssemblyFormat(opName)) {
            op.operands = parser.parseOperandList();
            if (parser.accept('id', 'start')) {
                const innerOpName = parser.parseOperationName();
                op.addAttribute('inner_op', innerOpName);
                parser.expect('id', 'end');
                parser.expect(':');
                const fnType = parser.parseFunctionType();
                if (fnType && fnType.inputs) {
                    parser.resolveOperands(op.operands, fnType.inputs);
                }
                if (fnType && fnType.results) {
                    for (let i = 0; i < fnType.results.length; i++) {
                        op.results.push({ type: fnType.results[i].toString() });
                    }
                }
                parser.parseLocation();
            } else {
                parser.expect('(');
                const region = op.addRegion();
                parser.parseRegion(region);
                parser.expect(')');
                parser.expect(':');
                const fnType = parser.parseFunctionType();
                if (fnType && fnType.inputs) {
                    parser.resolveOperands(op.operands, fnType.inputs);
                }
                if (fnType && fnType.results) {
                    for (let i = 0; i < fnType.results.length; i++) {
                        op.results.push({ type: fnType.results[i].toString() });
                    }
                }
            }
            return true;
        }
        if (opName === 'test.isolated_region' && this.hasCustomAssemblyFormat(opName) && !this.hasAssemblyFormat(opName)) {
            const operand = parser.parseOperand();
            op.operands.push(operand);
            operand.type = 'index';
            const region = op.addRegion();
            parser.parseRegion(region, [operand]);
            return true;
        }
        if (opName === 'test.string_attr_pretty_name' && this.hasCustomAssemblyFormat(opName) && !this.hasAssemblyFormat(opName)) {
            parser.parseOptionalAttrDictWithKeyword(op.attributes);
            for (const result of op.results) {
                result.type = 'i32';
            }
            return true;
        }
        if (opName === 'test.with_bounds_region' && this.hasCustomAssemblyFormat(opName) && !this.hasAssemblyFormat(opName)) {
            parser.parseOptionalAttrDict(op.attributes);
            const argName = parser.expect('%');
            parser.expect(':');
            const argType = parser.parseType();
            const region = op.addRegion();
            const arg = { value: argName, type: argType.toString() };
            parser.parseRegion(region, [arg]);
            return true;
        }
        return super.parseOperation(parser, opName, op);
    }

    _parseTestBitEnumProp(parser, type) {
        if (type.values.includes(parser.getToken().value)) {
            return this._parseEnumFlags(parser, type, ',');
        }
        return null;
    }

    _parseTestEnumAttr(parser, type) {
        const token = parser.getToken();
        if (token && type.values && type.values.includes(token.value)) {
            parser.expect();
            return new mlir.TypedAttr(token.value, null);
        }
        return null;
    }

    _parseTestEnumPropAttrForm(parser) {
        return parser.parseOptionalAttribute();
    }

    _parseTestBitEnumPropNamed(parser) {
        if (parser.accept('id', 'bit_enum')) {
            if (parser.accept('<')) {
                const flags = [];
                while (!parser.match('>')) {
                    const value = parser.expect('id');
                    flags.push(value);
                    parser.accept(',');
                }
                parser.expect('>');
                return new mlir.TypedAttr(`bit_enum<${flags.join(', ')}>`, null);
            }
        }
        return null;
    }

    // Parse CompoundNestedOuterType: assemblyFormat = "`<` `i` $inner `>`"
    // Full form: !test.cmpnd_nested_outer<i !test.cmpnd_inner<...>>
    // Elided form: <i <...>>
    _parseCompoundNestedOuterType(parser) {
        parser.expect('<');
        parser.expect('id', 'i');
        // Parse $inner - could be full (!test.cmpnd_inner<...>) or elided (<...>)
        const inner = parser.match('!') ? parser.parseType() : this._parseCompoundNestedInnerType(parser);
        parser.expect('>');
        return new mlir.Type(`!test.cmpnd_nested_outer<i ${inner}>`);
    }

    // Parse CompoundNestedInnerType: assemblyFormat = "`<` $some_int $cmpdA `>`"
    // Full form: !test.cmpnd_inner<42 !test.cmpnd_a<...>>
    // Elided form: <42 <...>>
    _parseCompoundNestedInnerType(parser) {
        parser.expect('<');
        const someInt = parser.parseInteger();
        // Parse $cmpdA - could be full (!test.cmpnd_a<...>) or elided (<...>)
        const cmpdA = parser.match('!') ? parser.parseType() : this._parseCompoundTypeA(parser);
        parser.expect('>');
        return new mlir.Type(`!test.cmpnd_inner<${someInt} ${cmpdA}>`);
    }

    // Parse CompoundTypeA: hasCustomAssemblyFormat = 1
    // Format: <$widthOfSomething, $oneType, [$arrayOfInts]>
    // Example: <1, !test.smpla, [5, 6]>
    _parseCompoundTypeA(parser) {
        parser.expect('<');
        const width = parser.parseInteger();
        parser.expect(',');
        const oneType = parser.parseType();
        parser.expect(',');
        parser.expect('[');
        const arrayOfInts = [];
        while (!parser.match(']')) {
            arrayOfInts.push(parser.parseInteger());
            parser.accept(',');
        }
        parser.expect(']');
        parser.expect('>');
        return new mlir.Type(`!test.cmpnd_a<${width}, ${oneType}, [${arrayOfInts.join(', ')}]>`);
    }

    _parseOptionalLoc(parser, op, args) {
        const attrName = args && args.length > 0 ? args[0].replace(/^\$/, '') : 'loc';
        const loc = parser.parseLocation();
        if (loc) {
            op.addAttribute(attrName, loc);
        } else {
            op.addAttribute(attrName, parser.location());
        }
    }

    _parseDummyRegionRef() {
    }

    _parseDummySuccessorRef() {
    }

    _parseOptionalCustomParser(parser, op, args) {
        if (!parser.accept('id', 'foo')) {
            return null; // Optional group not taken
        }
        const attrName = args && args.length > 0 ? args[0].replace(/^\$/, '') : 'attr';
        const attr = parser.parseAttribute();
        op.addAttribute(attrName, attr.value);
        return true;
    }

    _parseDimensionList(parser, op, args) {
        const dims = [];
        if (parser.accept('[')) {
            parser.accept(']');
            op.addAttribute('dimension_list', []);
            return;
        }
        for (;;) {
            if (parser.accept('?')) {
                dims.push(-1);
            } else if (parser.match('int')) {
                dims.push(parser.parseInteger());
            } else {
                break;
            }
            const token = parser.getToken();
            if (token && token.kind === 'id' && token.value.startsWith('x')) {
                const rest = token.value.slice(1);
                if (rest === '') {
                    parser.expect();
                } else if (/^\d+$/.test(rest)) {
                    parser.expect();
                    dims.push(parseInt(rest, 10));
                } else if (rest === '?') {
                    parser.expect();
                    dims.push(-1);
                } else {
                    break;
                }
            } else {
                break;
            }
        }
        const attrName = args && args.length > 0 ? args[0].replace(/^\$/, '') : 'dimension_list';
        op.addAttribute(attrName, dims);
    }

    _parseCustomOptionalOperand(parser, op) {
        if (parser.accept('(')) {
            const operand = {};
            operand.value = parser.expect('%');
            op.operands.push(operand);
            parser.expect(')');
        }
    }

    // Custom directive: operand [, optOperand] -> (varOperands)
    _parseCustomDirectiveOperands(parser, op) {
        // Parse required operand
        const operand = {};
        operand.value = parser.expect('%');
        op.operands.push(operand);
        // Parse optional operand
        if (parser.accept(',')) {
            const optOperand = {};
            optOperand.value = parser.expect('%');
            op.operands.push(optOperand);
        }
        // Parse -> (varOperands)
        parser.expect('->');
        parser.expect('(');
        while (parser.match('%')) {
            const varOperand = {};
            varOperand.value = parser.expect('%');
            op.operands.push(varOperand);
            if (!parser.accept(',')) {
                break;
            }
        }
        parser.expect(')');
    }

    // Custom directive: operands and types together
    _parseCustomDirectiveOperandsAndTypes(parser, op) {
        this._parseCustomDirectiveOperands(parser, op);
        this._parseCustomDirectiveResults(parser, op);
    }

    // Custom directive: : type [, optType] -> (varTypes)
    _parseCustomDirectiveResults(parser, op) {
        parser.expect(':');
        const type = parser.parseType();
        // Assign type to first operand/result if available
        if (op.operands.length > 0) {
            op.operands[0].type = type.toString();
        }
        if (parser.accept(',')) {
            const optType = parser.parseType();
            if (op.operands.length > 1) {
                op.operands[1].type = optType.toString();
            }
        }
        parser.expect('->');
        parser.expect('(');
        let idx = 2; // Start after first two operands
        while (!parser.match(')')) {
            const varType = parser.parseType();
            if (op.operands.length > idx) {
                op.operands[idx].type = varType.toString();
            }
            idx++;
            if (!parser.accept(',')) {
                break;
            }
        }
        parser.expect(')');
    }

    _parseCustomDirectiveWithTypeRefs(parser, op) {
        // Parses: type_refs_capture : type [, type] -> (types)
        parser.expect('id', 'type_refs_capture');
        this._parseCustomDirectiveResults(parser, op);
    }

    _parseCustomDirectiveRegions(parser, op) {
        // Parse first region
        const region = op.addRegion();
        parser.parseRegion(region);
        // Parse optional variadic regions
        while (parser.accept(',')) {
            const varRegion = op.addRegion();
            parser.parseRegion(varRegion);
        }
    }

    _parseCustomDirectiveSuccessors(parser, op) {
        if (!op.successors) {
            op.successors = [];
        }
        // Parse first successor
        const successor = {};
        successor.label = parser.expect('^');
        op.successors.push(successor);
        // Parse optional variadic successors
        while (parser.accept(',')) {
            const varSuccessor = {};
            varSuccessor.label = parser.expect('^');
            op.successors.push(varSuccessor);
        }
    }

    _parseCustomDirectiveAttrDict(parser, op) {
        parser.parseAttributeDict(op.attributes);
    }

    _parseCustomDirectiveAttributes(parser, op) {
        // Parse: attr [, optAttr]
        const attr = parser.parseAttribute();
        op.addAttribute('attr', attr);
        if (parser.accept(',')) {
            const optAttr = parser.parseAttribute();
            op.addAttribute('optAttr', optAttr);
        }
    }

    _parseCustomDirectiveSpacing(parser, op, args) {
        // Parse attribute for spacing test
        if (args && args.length > 0) {
            const attrName = args[0].name || args[0];
            const attr = parser.parseAttribute();
            op.addAttribute(attrName, attr);
        }
    }

    _parseCustomDirectiveOptionalOperandRef(parser) {
        // This directive parses an integer (1 or 0) indicating if the optional operand was present
        parser.parseInteger();
    }

    _parseSwitchCases(parser, op) {
        const caseValues = [];
        while (parser.match('id', 'case')) {
            parser.expect('id', 'case');
            const value = parser.parseInteger();
            caseValues.push(value);
            const region = op.addRegion();
            parser.parseRegion(region);
        }
        op.addAttribute('cases', `array<i64: ${caseValues.join(', ')}>`);
    }

    _parseUsingPropertyInCustom(parser, op, args) {
        // Parse [int, int, ...] format for property values
        const values = [];
        parser.expect('[');
        while (!parser.match(']')) {
            const value = parser.parseInteger();
            values.push(value);
            if (!parser.accept(',')) {
                break;
            }
        }
        parser.expect(']');
        if (args && args.length > 0) {
            const propName = typeof args[0] === 'string' ? args[0].replace(/^\$/, '') : args[0].name;
            op.addAttribute(propName, `array<i64: ${values.join(', ')}>`);
        }
    }

    _parseIntProperty(parser, op, args) {
        const value = parser.parseInteger();
        if (args && args.length > 0) {
            const propName = typeof args[0] === 'string' ? args[0].replace(/^\$/, '') : args[0].name;
            op.addAttribute(propName, value);
        }
    }

    // Reference: TestFormatUtils.cpp parseSumProperty
    // Format: <second> = <sum> where sum should equal first + second
    _parseSumProperty(parser, op, args) {
        const second = parser.parseInteger();
        parser.expect('=');
        parser.parseInteger(); // sum value (validation skipped)
        if (args && args.length > 0) {
            const propName = typeof args[0] === 'string' ? args[0].replace(/^\$/, '') : args[0].name;
            op.addAttribute(propName, second);
        }
    }
};

mlir.TritonDialect = class extends mlir.Dialect {

    constructor(operations) {
        super(operations, 'tt');
        this.registerCustomType('TT_Ptr', this._parsePtr.bind(this));
        this.registerCustomType('TT_TensorDescType', this._parseTensorDescType.bind(this));
        this.registerCustomType('TT_TensorPtr', this._parseTensorPtr.bind(this));
    }

    parseType(parser, dialectName) {
        const typeName = parser.parseKeyword();
        if (!typeName) {
            return null;
        }
        let type = `!${dialectName}.${typeName}`;
        if (parser.match('<')) {
            const content = parser.skip('<');
            type += content;
        }
        return new mlir.Type(type);
    }

    parseOperation(parser, opName, op) {
        if (opName === 'tt.func') {
            parser.parseFunctionOp(op, false);
            return true;
        }
        return super.parseOperation(parser, opName, op);
    }

    _parseTensorPtr(parser) {
        if (parser.match('<')) {
            const content = parser.skip('<');
            return new mlir.Type(`!tt.ptr${content}`);
        }
        return null;
    }

    _parsePtr(parser) {
        if (parser.match('<')) {
            const content = parser.skip('<');
            return new mlir.Type(`!tt.ptr${content}`);
        }
        return null;
    }

    _parseTensorDescType(parser) {
        if (parser.match('<')) {
            const content = parser.skip('<');
            return new mlir.Type(`!tt.tensor_desc${content}`);
        }
        return null;
    }
};

mlir.TritonGPUDialect = class extends mlir.Dialect {

    constructor(operations) {
        super(operations, 'ttg');
        this.registerCustomType('TTG_MemDescType', this._parseMemDescType.bind(this));
    }

    _parseMemDescType(parser) {
        // Handle shorthand MemDescType notation: <dims x elementType, attributes...>
        // Full notation would be: !ttg.memdesc<dims x elementType, attributes...>
        if (!parser.match('<')) {
            return null;
        }
        const content = parser.skip('<');
        return new mlir.Type(`!ttg.memdesc<${content}>`);
    }

    parseOperation(parser, opName, op) {
        if (opName === 'ttg.warp_specialize') {
            parser.expect('(');
            while (!parser.match(')')) {
                const operand = {};
                operand.name = parser.expect('%');
                op.operands.push(operand);
                if (!parser.match(')')) {
                    parser.expect(',');
                }
            }
            parser.expect(')');
            parser.parseOptionalAttrDictWithKeyword(op.attributes);
            parser.expect('id', 'default');
            const defaultRegion = {};
            parser.parseRegion(defaultRegion);
            op.regions.push(defaultRegion);
            const partitionNumWarps = [];
            let partitionIndex = 0;
            while (parser.match('id', `partition${partitionIndex}`)) {
                parser.expect('id', `partition${partitionIndex}`);
                const argResult = parser.parseFunctionArgumentList();
                parser.expect('id', 'num_warps');
                parser.expect('(');
                const numWarps = parser.expect();
                partitionNumWarps.push(parseInt(numWarps, 10));
                parser.expect(')');
                const partitionRegion = {};
                partitionRegion.arguments = argResult.arguments;
                parser.parseRegion(partitionRegion);
                if (!op.regions[1]) {
                    op.regions[1] = { blocks: [{ operations: [] }] };
                }
                partitionIndex++;
            }
            parser.expect(':');
            const fnType = parser.parseType();
            if (fnType instanceof mlir.FunctionType) {
                op.addAttribute('function_type', new mlir.TypeAttrOf(fnType));
            } else {
                op.addAttribute('function_type', new mlir.TypeAttrOf(new mlir.FunctionType([], [fnType])));
            }
            if (partitionNumWarps.length > 0) {
                op.addAttribute('partitionNumWarps', { type: 'array', element_type: 'i32', value: partitionNumWarps });
            }
            return true;
        }
        return super.parseOperation(parser, opName, op);
    }

    parseType(parser, dialectName) {
        const typeName = parser.parseKeyword();
        if (!typeName) {
            return null;
        }
        let type = `!${dialectName}.${typeName}`;
        if (parser.match('<')) {
            const content = parser.skip('<');
            type += content;
        }
        return new mlir.Type(type);
    }
};

mlir.GluonDialect = class extends mlir.Dialect {

    constructor(operations) {
        super(operations, 'gluon');
    }
};

mlir.TritonNvidiaGPUDialect = class extends mlir.Dialect {

    constructor(operations) {
        super(operations, 'ttng');
        this.registerCustomDirective('Token', this._parseToken.bind(this));
        this.registerCustomDirective('BarriersAndPreds', this._parseBarriersAndPreds.bind(this));
    }

    _parseToken(parser, op, args) {
        if (!parser.accept('[')) {
            return;
        }
        if (args && args.length >= 2) {
            op.addAttribute(args[1].replace(/^\$/, ''), '!ttng.async.token');
        }
        if (parser.match(']')) {
            parser.expect(']');
            return;
        }
        if (parser.match('%')) {
            const dep = parser.expect('%');
            if (args && args.length > 0) {
                op.operands.push({ value: dep });
            }
        }
        parser.expect(']');
    }

    _parseBarriersAndPreds(parser, op, args) {
        while (parser.accept(',')) {
            if (parser.match('%')) {
                const barrier = parser.expect('%');
                if (args && args.length > 0) {
                    op.operands.push({ value: barrier });
                }
                if (parser.accept('[')) {
                    if (parser.match('%')) {
                        const pred = parser.expect('%');
                        if (args && args.length > 1) {
                            op.operands.push({ value: pred });
                        }
                    }
                    parser.expect(']');
                }
            }
        }
    }

    parseType(parser, dialectName) {
        const typeName = parser.parseKeyword();
        if (!typeName) {
            return null;
        }
        let type = `!${dialectName}.${typeName}`;
        if (parser.match('<')) {
            const content = parser.skip('<');
            type += content;
        }
        return new mlir.Type(type);
    }
};

mlir.TritonAMDGPUDialect = class extends mlir.Dialect {

    constructor(operations) {
        super(operations, 'amdg');
        this.registerCustomType('TT_Ptr', 'tt.ptr');
        this.registerCustomType('TT_TensorPtr', 'tt.ptr');
        this.registerCustomType('TTG_MemDescType', 'ttg.memdesc');
    }

    parseType(parser, dialectName) {
        const typeName = parser.parseKeyword();
        if (!typeName) {
            return null;
        }
        let type = `!${dialectName}.${typeName}`;
        if (parser.match('<')) {
            const content = parser.skip('<');
            type += content;
        }
        return new mlir.Type(type);
    }
};

mlir.ProtonDialect = class extends mlir.Dialect {

    constructor(operations) {
        super(operations, 'proton');
    }
};

mlir.MichelsonDialect = class extends mlir.Dialect {

    constructor(operations) {
        super(operations, 'michelson');
    }

    parseType(parser, dialectName) {
        const typeName = parser.parseKeyword();
        if (!typeName) {
            return null;
        }
        let type = `!${dialectName}.${typeName}`;
        if ((typeName === 'big' || typeName === 'chain' || typeName === 'key') && parser.match('_')) {
            parser.expect('_');
            const suffix = parser.expect('id');
            type += `_${suffix}`;
        }
        const simpleTypes = ['int', 'bytes', 'operation', 'nat', 'string', 'unit', 'bool', 'mutez', 'timestamp', 'address', 'key', 'signature', 'chain_id', 'key_hash'];
        if (simpleTypes.includes(type.substring(11))) { // Remove "!michelson." prefix
            return new mlir.Type(type);
        }
        const typesWithParams = ['pair', 'list', 'option', 'or', 'map', 'big_map', 'set', 'contract', 'lambda'];
        if (typesWithParams.includes(type.substring(11))) {
            if (parser.match('<')) {
                const content = parser.skip('<');
                type += content;
            }
            return new mlir.Type(type);
        }
        return null;
    }
};

mlir.PlanDialect = class extends mlir.Dialect {

    constructor(operations) {
        super(operations, 'plan');
        this.registerCustomDirective('WithValuesTypes', this._parseWithValuesTypes.bind(this));
    }

    // Reference: PlanOps.cpp parseWithValuesTypes
    // Parse: type($result) - just a single type, element types are inferred
    _parseWithValuesTypes(parser, op /*, args */) {
        const resultType = parser.parseType();
        if (op.results.length === 0) {
            op.results.push({ type: resultType });
        } else {
            op.results[0].type = resultType;
        }
    }
};

mlir.KernelDialect = class extends mlir.Dialect {

    constructor(operations) {
        super(operations, 'kernel');
        this.registerCustomDirective('KernelFunctionalType', this._parseKernelFunctionalType.bind(this));
    }

    // Parse: (types) -> (types)
    _parseKernelFunctionalType(parser /*, op, args */) {
        parser.expect('(');
        if (!parser.match(')')) {
            do {
                parser.parseType();
            } while (parser.accept(','));
        }
        parser.expect(')');
        parser.expect('->');
        if (parser.accept('(')) {
            if (!parser.match(')')) {
                do {
                    parser.parseType();
                } while (parser.accept(','));
            }
            parser.expect(')');
        } else {
            parser.parseType();
        }
    }
};

mlir.TensorRTDialect = class extends mlir.Dialect {

    constructor(operations) {
        super(operations, 'tensorrt');
        this.registerCustomAttribute('TensorRT_TopKOperationAttr', this._parseEnumAttrBracket.bind(this));
        this.registerCustomAttribute('TensorRT_ScatterModeAttr', this._parseEnumAttrBracket.bind(this));
        this.registerCustomAttribute('TensorRT_ResizeSelectorAttr', this._parseEnumAttrBracket.bind(this));
        this.registerCustomAttribute('TensorRT_ResizeRoundModeAttr', this._parseEnumAttrBracket.bind(this));
        this.registerCustomAttribute('TensorRT_ResizeModeAttr', this._parseEnumAttrBracket.bind(this));
        this.registerCustomAttribute('TensorRT_ResizeCoordinateTransformationAttr', this._parseEnumAttrBracket.bind(this));
        this.registerCustomAttribute('TensorRT_ReduceOperationAttr', this._parseEnumAttrBracket.bind(this));
        this.registerCustomAttribute('TensorRT_PaddingModeAttr', this._parseEnumAttrBracket.bind(this));
        this.registerCustomAttribute('TensorRT_MatrixOperationAttr', this._parseEnumAttrBracket.bind(this));
        this.registerCustomAttribute('TensorRT_LoopOutputAttr', this._parseEnumAttrBracket.bind(this));
        this.registerCustomAttribute('TensorRT_GatherModeAttr', this._parseEnumAttrBracket.bind(this));
        this.registerCustomAttribute('TensorRT_FillOperationAttr', this._parseEnumAttrBracket.bind(this));
        this.registerCustomAttribute('TensorRT_ElementWiseOperationAttr', this._parseEnumAttrBracket.bind(this));
        this.registerCustomAttribute('TensorRT_ActivationTypeAttr', this._parseEnumAttrBracket.bind(this));
        this.registerCustomAttribute('TensorRT_UnaryOperationAttr', this._parseEnumAttrBracket.bind(this));
        this.registerCustomAttribute('TensorRT_TripLimitAttr', this._parseEnumAttrBracket.bind(this));
        this.registerCustomAttribute('TensorRT_PoolingTypeAttr', this._parseEnumAttrBracket.bind(this));
        this.registerCustomDirective('StaticIndexI64Array', this._parseStaticIndexI64Array.bind(this));
        this.registerCustomDirective('StaticIndexI32Array', this._parseStaticIndexI32Array.bind(this));
    }

    _parseEnumAttrBracket(parser) {
        if (parser.match('<')) {
            parser.expect('<');
            const value = parser.expect('id');
            parser.expect('>');
            return { value };
        }
        return null;
    }

    _parseStaticIndexI64Array(parser, op, args) {
        const values = [];
        do {
            if (parser.match('int')) {
                const value = parser.expect('int');
                values.push(parseInt(value, 10));
            } else if (parser.match('-')) {
                parser.expect('-');
                const value = parser.expect('int');
                values.push(-parseInt(value, 10));
            } else {
                break;
            }
        } while (parser.accept(','));
        const name = args && args[0] ? args[0].replace(/^\$/, '') : 'broadcast_dims';
        op.addAttribute(name, values);
    }

    _parseStaticIndexI32Array(parser, op, args) {
        const values = [];
        do {
            if (parser.match('int')) {
                const value = parser.expect('int');
                values.push(parseInt(value, 10));
            } else if (parser.match('-')) {
                parser.expect('-');
                const value = parser.expect('int');
                values.push(-parseInt(value, 10));
            } else {
                break;
            }
        } while (parser.accept(','));
        const name = args && args[0] ? args[0].replace(/^\$/, '') : 'static_values';
        op.addAttribute(name, values);
    }

    parseOperation(parser, opName, op) {
        if (opName === 'tensorrt.for' && this.hasCustomAssemblyFormat(opName) && !this.hasAssemblyFormat(opName)) {
            return this._parseForOp(parser, op);
        }
        return super.parseOperation(parser, opName, op);
    }

    _parseForOp(parser, op) {
        const inductionVar = parser.expect('%');
        parser.expect('=');
        op.operands.push({ value: parser.expect('%') });
        parser.expect('id', 'to');
        op.operands.push({ value: parser.expect('%') });
        parser.expect('id', 'step');
        op.operands.push({ value: parser.expect('%') });
        parser.expect('id', 'init');
        if (parser.accept('(')) {
            while (!parser.accept(')')) {
                if (parser.match('%')) {
                    parser.expect('%');
                }
                if (parser.accept('=')) {
                    if (parser.match('%')) {
                        op.operands.push({ value: parser.expect('%') });
                    }
                }
                parser.accept(',');
            }
        }
        parser.parseOptionalArrowResultTypes(op);
        if (parser.match('{')) {
            const region = {};
            parser.parseRegion(region);
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
        parser.parseOptionalAttrDict(op.attributes);
        return true;
    }
};

mlir.ExecutorDialect = class extends mlir.Dialect {

    constructor(operations) {
        super(operations, 'executor');
        this.registerCustomType('Executor_Table', this._parseTable.bind(this));
        this.registerCustomDirective('ExecutorMixedIndices', this._parseExecutorMixedIndices.bind(this));
    }

    parseOperation(parser, opName, op) {
        if (opName === 'executor.func') {
            parser.parseFunctionOp(op, true);
            return true;
        }
        return super.parseOperation(parser, opName, op);
    }

    _parseTable(parser) {
        if (parser.match('<')) {
            const content = parser.skip('<');
            return new mlir.Type(`!executor.table${content}`);
        }
        return null;
    }

    // Parse: [dynamicIndices, staticIndices] mixed format
    _parseExecutorMixedIndices(parser, op /*, args */) {
        do {
            if (parser.match('%')) {
                const operand = parser.expect('%');
                op.operands.push({ value: operand });
            } else {
                parser.parseAttribute();
            }
        } while (parser.accept(','));
    }
};

mlir.TFRTTestDialect = class extends mlir.Dialect {

    constructor(operations) {
        super(operations, 'tfrt_test');
        this.registerCustomDirective('OptionalLoc', this._parseOptionalLoc.bind(this));
        this.registerCustomDirective('DummyRegionRef', this._parseDummyRegionRef.bind(this));
        this.registerCustomDirective('DummySuccessorRef', this._parseDummySuccessorRef.bind(this));
    }

    _parseOptionalLoc(parser, op, args) {
        const attrName = args && args.length > 0 ? args[0].replace(/^\$/, '') : 'loc';
        const loc = parser.parseLocation();
        if (loc) {
            op.addAttribute(attrName, loc);
        } else {
            op.addAttribute(attrName, parser.location());
        }
    }

    _parseDummyRegionRef() {
    }

    _parseDummySuccessorRef() {
    }

    parseOperation(parser, opName, op) {
        const opInfo = this.getOperation(opName);
        if (!opInfo) {
            return false;
        }
        if (opInfo.metadata?.assemblyFormat === 'operands attr-dict') {
            while (parser.match('%')) {
                op.operands.push({ value: parser.expect() });
                if (!parser.accept(',')) {
                    break;
                }
            }
            if (parser.match('{')) {
                parser.parseAttributeDict(op.attributes);
            }
            return true;
        }
        if (opName === 'tfrt_test.do.async') {
            if (parser.match('%')) {
                op.operands = parser.parseArguments();
            }
            if (parser.accept(':')) {
                const type = parser.parseFunctionType();
                if (type && type.results) {
                    type.results.forEach((resultType) => {
                        op.results.push({ type: resultType });
                    });
                }
            }
            if (parser.match('{')) {
                const region = op.addRegion();
                parser.parseRegion(region);
            }
            return true;
        }
        if (opName === 'tfrt_test.benchmark') {
            if (parser.match('string')) {
                const name = parser.expect('string');
                op.addAttribute('name', name);
            }
            parser.expect('(');
            while (parser.match('%')) {
                const operand = { value: parser.expect('%') };
                if (parser.accept(':')) {
                    operand.type = parser.parseType().toString();
                }
                op.operands.push(operand);
                if (!parser.accept(',')) {
                    break;
                }
            }
            parser.expect(')');
            while (parser.match('id') && !parser.match('{')) {
                const name = parser.expect('id');
                parser.expect('=');
                let value = null;
                if (parser.match('int')) {
                    value = parser.parseInteger();
                } else if (parser.match('string')) {
                    value = parser.expect('string');
                } else {
                    value = parser.expect('id');
                }
                op.addAttribute(name, value);
                parser.accept(',');
            }
            if (parser.match('{')) {
                const region = op.addRegion();
                parser.parseRegion(region);
            }
            return true;
        }
        return super.parseOperation(parser, opName, op);
    }
};

mlir.XeVMDialect = class extends mlir.Dialect {

    constructor(operations) {
        super(operations, 'xevm');
    }
};

mlir.VMVXDialect = class extends mlir.Dialect {

    constructor(operations) {
        super(operations, 'vmvx');
    }
};

mlir.MLRTDialect = class extends mlir.Dialect {

    constructor(operations) {
        super(operations, 'mlrt');
    }
};

mlir.TFRTTensorDialect = class extends mlir.Dialect {

    constructor(operations) {
        super(operations, 'tfrt_tensor');
    }
};

mlir.TFRTDHTDialect = class extends mlir.Dialect {

    constructor(operations) {
        super(operations, 'tfrt_dht');
    }
};

mlir.TFDDialect = class extends mlir.Dialect {

    constructor(operations) {
        super(operations, 'tfd');
    }
};

mlir.ACCDialect = class extends mlir.Dialect {

    constructor(operations) {
        super(operations, 'acc');
        this.registerCustomDirective('Var', this._parseVar.bind(this));
        this.registerCustomDirective('AccVar', this._parseAccVar.bind(this));
        this.registerCustomDirective('VarPtrType', this._parseVarPtrType.bind(this));
        this.registerCustomDirective('DeviceTypeOperandsWithKeywordOnly', this._parseDeviceTypeOperandsWithKeywordOnly.bind(this));
        this.registerCustomDirective('LoopControl', this._parseLoopControl.bind(this));
        this.registerCustomDirective('WaitClause', this._parseWaitClause.bind(this));
        this.registerCustomDirective('NumGangs', this._parseNumGangs.bind(this));
        this.registerCustomDirective('DeviceTypeOperands', this._parseDeviceTypeOperands.bind(this));
        this.registerCustomDirective('GangClause', this._parseGangClause.bind(this));
        this.registerCustomDirective('CombinedConstructsLoop', this._parseCombinedConstructsLoop.bind(this));
        this.registerCustomDirective('RecipeSym', this._parseRecipeSym.bind(this));
        this.registerCustomDirective('OperandWithKeywordOnly', this._parseOperandWithKeywordOnly.bind(this));
        this.registerCustomDirective('OperandsWithKeywordOnly', this._parseOperandsWithKeywordOnly.bind(this));
        this.registerCustomDirective('DeviceTypeOperandsWithSegment', this._parseDeviceTypeOperandsWithSegment.bind(this));
        this.registerCustomDirective('BindName', this._parseBindName.bind(this));
        this.registerCustomDirective('RoutineGangClause', this._parseRoutineGangClause.bind(this));
        this.registerCustomDirective('DeviceTypeArrayAttr', this._parseDeviceTypeArrayAttr.bind(this));
    }

    _parseVar(parser, op) {
        if (!parser.accept('id', 'varPtr')) {
            parser.expect('id', 'var');
        }
        parser.expect('(');
        const operand = parser.parseOperand();
        op.operands.push(operand);
    }

    _parseAccVar(parser, op) {
        if (!parser.accept('id', 'accPtr')) {
            parser.expect('id', 'accVar');
        }
        parser.expect('(');
        const operand = parser.parseOperand();
        op.operands.push(operand);
        parser.expect(':');
        operand.type = parser.parseType();
        parser.expect(')');
    }

    _parseVarPtrType(parser, op) {
        const type = parser.parseType();
        if (op.operands.length > 0) {
            op.operands[op.operands.length - 1].type = type;
        }
        parser.expect(')');
        if (parser.accept('id', 'varType')) {
            parser.expect('(');
            const varType = parser.parseType();
            op.addAttribute('varType', varType);
            parser.expect(')');
        }
    }

    _parseDeviceTypeOperandsWithKeywordOnly(parser, op, args) {
        const [, , deviceTypesVar, keywordOnlyVar] = args.map((a) => a.replace('$', '').replace(/^type\(/, '').replace(/\)$/, ''));
        if (!parser.accept('(')) {
            op.addAttribute(keywordOnlyVar, [{ value: 'none' }]);
            return;
        }
        const keywordOnlyAttrs = [];
        let needComma = false;
        if (parser.accept('[')) {
            while (!parser.match(']')) {
                const attr = parser.parseAttribute();
                keywordOnlyAttrs.push(attr);
                parser.accept(',');
            }
            parser.expect(']');
            needComma = true;
        }
        if (keywordOnlyAttrs.length > 0) {
            op.addAttribute(keywordOnlyVar, keywordOnlyAttrs);
        }
        if (needComma) {
            parser.accept(',');
        }
        const operands = [];
        const deviceTypes = [];
        while (!parser.match(')')) {
            const operand = parser.parseOperand();
            parser.expect(':');
            operand.type = parser.parseType();
            operands.push(operand);
            if (parser.accept('[')) {
                const deviceType = parser.parseAttribute();
                deviceTypes.push(deviceType);
                parser.expect(']');
            } else {
                deviceTypes.push({ value: 'none' });
            }
            if (!parser.accept(',')) {
                break;
            }
        }
        parser.expect(')');
        op.operands.push(...operands);
        if (deviceTypes.length > 0) {
            op.addAttribute(deviceTypesVar, deviceTypes);
        }
    }

    _parseLoopControl(parser, op) {
        const inductionVars = [];
        if (parser.accept('id', 'control')) {
            parser.expect('(');
            while (!parser.match(')')) {
                const value = parser.expect('%');
                parser.expect(':');
                const type = parser.parseType();
                inductionVars.push({ value, type });
                parser.accept(',');
            }
            parser.expect(')');
            parser.expect('=');
            parser.expect('(');
            const lowerbound = parser.parseOperandList();
            const lowerboundTypes = parser.parseColonTypeList();
            parser.resolveOperands(lowerbound, lowerboundTypes);
            op.operands.push(...lowerbound);
            parser.expect(')');
            parser.expect('id', 'to');
            parser.expect('(');
            const upperbound = parser.parseOperandList();
            const upperboundTypes = parser.parseColonTypeList();
            parser.resolveOperands(upperbound, upperboundTypes);
            op.operands.push(...upperbound);
            parser.expect(')');
            parser.expect('id', 'step');
            parser.expect('(');
            const step = parser.parseOperandList();
            const stepTypes = parser.parseColonTypeList();
            parser.resolveOperands(step, stepTypes);
            op.operands.push(...step);
            parser.expect(')');
        }
        const region = op.addRegion();
        parser.parseRegion(region, inductionVars);
    }

    _parseWaitClause(parser, op) {
        if (!parser.match('(')) {
            return;
        }
        parser.expect('(');
        if (parser.accept('[')) {
            while (!parser.accept(']')) {
                parser.parseAttribute();
                parser.accept(',');
            }
            if (!parser.match(')')) {
                parser.accept(',');
            }
        }
        while (!parser.match(')')) {
            if (parser.accept('{')) {
                parser.accept('id', 'devnum');
                parser.accept(':');
                while (!parser.accept('}')) {
                    const operand = parser.parseOperand();
                    parser.expect(':');
                    operand.type = parser.parseType();
                    op.operands.push(operand);
                    parser.accept(',');
                }
                if (parser.accept('[')) {
                    parser.parseAttribute();
                    parser.expect(']');
                }
            }
            parser.accept(',');
        }
        parser.expect(')');
    }

    _parseNumGangs(parser, op) {
        while (parser.accept('{')) {
            while (!parser.accept('}')) {
                const operand = parser.parseOperand();
                parser.expect(':');
                operand.type = parser.parseType();
                op.operands.push(operand);
                parser.accept(',');
            }
            if (parser.accept('[')) {
                parser.parseAttribute();
                parser.expect(']');
            }
            parser.accept(',');
        }
    }

    _parseDeviceTypeOperands(parser, op) {
        while (parser.match('%')) {
            const operand = parser.parseOperand();
            parser.expect(':');
            operand.type = parser.parseType();
            op.operands.push(operand);
            if (parser.accept('[')) {
                parser.parseAttribute();
                parser.expect(']');
            }
            parser.accept(',');
        }
    }

    _parseGangClause(parser, op, args) {
        const [, , , , , gangOnlyVar] = args.map((a) => a.replace('$', '').replace(/^type\(/, '').replace(/\)$/, ''));
        if (!parser.accept('(')) {
            op.addAttribute(gangOnlyVar, [{ value: 'none' }]);
            return;
        }
        const gangOnlyAttrs = [];
        let needComma = false;
        if (parser.accept('[')) {
            while (!parser.match(']')) {
                const attr = parser.parseAttribute();
                gangOnlyAttrs.push(attr);
                parser.accept(',');
            }
            parser.expect(']');
            needComma = true;
        }
        if (gangOnlyAttrs.length > 0) {
            op.addAttribute(gangOnlyVar, gangOnlyAttrs);
        }
        if (needComma) {
            parser.accept(',');
        }
        const gangArgTypes = [];
        const deviceTypes = [];
        const segments = [];
        while (parser.accept('{')) {
            let segmentCount = 0;
            while (!parser.match('}')) {
                let argType = 'Num';
                if (parser.accept('id', 'num')) {
                    parser.expect('=');
                    argType = 'Num';
                } else if (parser.accept('id', 'dim')) {
                    parser.expect('=');
                    argType = 'Dim';
                } else if (parser.accept('id', 'static')) {
                    parser.expect('=');
                    argType = 'Static';
                }
                gangArgTypes.push({ value: argType });
                const operand = parser.parseOperand();
                parser.expect(':');
                operand.type = parser.parseType();
                op.operands.push(operand);
                segmentCount++;
                if (!parser.accept(',')) {
                    break;
                }
            }
            parser.expect('}');
            segments.push(segmentCount);
            if (parser.accept('[')) {
                const deviceType = parser.parseAttribute();
                deviceTypes.push(deviceType);
                parser.expect(']');
            } else {
                deviceTypes.push({ value: 'none' });
            }
            if (!parser.accept(',')) {
                break;
            }
        }
        parser.expect(')');
        if (gangArgTypes.length > 0) {
            op.addAttribute('gangOperandsArgType', gangArgTypes);
        }
        if (deviceTypes.length > 0) {
            op.addAttribute('gangOperandsDeviceType', deviceTypes);
        }
        if (segments.length > 0) {
            op.addAttribute('gangOperandsSegments', segments);
        }
    }

    _parseCombinedConstructsLoop(parser, op) {
        const attr = parser.parseAttribute();
        op.addAttribute('combined', attr);
    }

    _parseRecipeSym(parser, op) {
        const attr = parser.parseAttribute();
        op.addAttribute('recipe', attr);
    }

    _parseOperandWithKeywordOnly(parser, op) {
        if (!parser.match('(')) {
            return;
        }
        parser.expect('(');
        if (parser.accept('[')) {
            while (!parser.accept(']')) {
                parser.parseAttribute();
                parser.accept(',');
            }
            if (!parser.match(')')) {
                parser.accept(',');
            }
        }
        while (!parser.match(')')) {
            const operand = parser.parseOperand();
            parser.expect(':');
            operand.type = parser.parseType();
            op.operands.push(operand);
            if (parser.accept('[')) {
                parser.parseAttribute();
                parser.expect(']');
            }
            parser.accept(',');
        }
        parser.expect(')');
    }

    _parseOperandsWithKeywordOnly(parser, op) {
        // Handles format: (%v1, %v2 : t1, t2) where all operands are listed before colon
        // and all types are listed after colon
        if (!parser.match('(')) {
            return;
        }
        parser.expect('(');
        if (parser.match(')')) {
            parser.expect(')');
            return;
        }
        // Parse all operands (comma-separated) until we hit ':'
        const operands = [];
        do {
            operands.push(parser.parseOperand());
        } while (parser.accept(',') && !parser.match(':'));
        // Parse the colon and types
        parser.expect(':');
        // Parse all types (comma-separated)
        for (let i = 0; i < operands.length; i++) {
            if (i > 0) {
                parser.expect(',');
            }
            operands[i].type = parser.parseType();
            op.operands.push(operands[i]);
        }
        parser.expect(')');
    }

    _parseDeviceTypeOperandsWithSegment(parser, op) {
        this._parseNumGangs(parser, op);
    }

    _parseBindName(parser, op) {
        while (!parser.match(')')) {
            const attr = parser.parseAttribute();
            if (parser.accept('[')) {
                parser.parseAttribute();
                parser.expect(']');
            }
            op.addAttribute('bind', attr);
            parser.accept(',');
        }
    }

    _parseRoutineGangClause(parser, op) {
        if (parser.accept('(')) {
            while (!parser.match(')')) {
                if (parser.accept('id', 'dim')) {
                    parser.expect(':');
                }
                const value = parser.parseAttribute();
                if (parser.accept('[')) {
                    parser.parseAttribute();
                    parser.expect(']');
                }
                op.addAttribute('gangDim', value);
                parser.accept(',');
            }
            parser.expect(')');
        } else if (parser.accept('[')) {
            while (!parser.accept(']')) {
                parser.parseAttribute();
                parser.accept(',');
            }
        }
    }

    _parseDeviceTypeArrayAttr(parser) {
        if (parser.accept('[')) {
            while (!parser.accept(']')) {
                parser.parseAttribute();
                parser.accept(',');
            }
        }
    }
};

mlir.SMTDialect = class extends mlir.Dialect {

    constructor(operations) {
        super(operations, 'smt');
    }

    parseOperation(parser, opName, op) {
        if (opName === 'smt.eq' || opName === 'smt.distinct') {
            return this._parseSameOperandTypeVariadicToBoolOp(parser, op);
        }
        if (opName === 'smt.bv.repeat') {
            return this._parseRepeatOp(parser, op);
        }
        if (opName === 'smt.int.constant') {
            return this._parseIntConstantOp(parser, op);
        }
        return super.parseOperation(parser, opName, op);
    }

    _parseSameOperandTypeVariadicToBoolOp(parser, op) {
        const operands = parser.parseOperandList();
        op.operands = operands;
        parser.parseOptionalAttrDict(op.attributes);
        parser.expect(':');
        const type = parser.parseType();
        for (const operand of op.operands) {
            operand.type = type;
        }
        return true;
    }

    _parseRepeatOp(parser, op) {
        const count = parser.parseInteger();
        op.addAttribute('count', count);
        parser.expect('id', 'times');
        const operand = {};
        operand.value = parser.expect('%');
        op.operands.push(operand);
        parser.parseOptionalAttrDict(op.attributes);
        parser.expect(':');
        const inputType = parser.parseType();
        op.operands[0].type = inputType;
        return true;
    }

    _parseIntConstantOp(parser, op) {
        const value = parser.parseInteger();
        op.addAttribute('value', value);
        parser.parseOptionalAttrDict(op.attributes);
        return true;
    }
};

mlir.MPMDDialect = class extends mlir.Dialect {

    constructor(operations) {
        super(operations, 'mpmd');
        this.registerCustomType('mesh_tensor', this._parseMeshTensorType.bind(this));
    }

    parseOperation(parser, opName, op) {
        if (opName === 'mpmd.named_computation' && !this.hasAssemblyFormat(opName)) {
            return this._parseNamedComputationOp(parser, op);
        }
        if (opName === 'mpmd.fragment' && !this.hasAssemblyFormat(opName)) {
            return this._parseFragmentOp(parser, op);
        }
        if (opName === 'mpmd.fragment_call' && !this.hasAssemblyFormat(opName)) {
            return this._parseFragmentCallOp(parser, op);
        }
        if (opName === 'mpmd.for' && !this.hasAssemblyFormat(opName)) {
            return this._parseForOp(parser, op);
        }
        return super.parseOperation(parser, opName, op);
    }

    _parseNamedComputationOp(parser, op) {
        // mpmd.named_computation<"name"(count)> (%inputs) (%block_args) { region } : (types) -> types
        parser.expect('<');
        // Parse single UserOriginAttr in short format: "name"(count)
        const origin = this._parseUserOriginAttr(parser);
        op.addAttribute('origin', origin);
        parser.expect('>');
        const inputs = parser.parseArguments();
        op.operands.push(...inputs);
        const entryArguments = this.parseBlockArguments(parser);
        if (parser.match('{')) {
            const region = op.addRegion();
            parser.parseRegion(region, entryArguments);
        }
        if (parser.accept(':')) {
            const type = parser.parseType();
            if (type instanceof mlir.FunctionType) {
                parser.resolveOperands(op.operands, type.inputs);
                parser.resolveOperands(op.results, type.results);
            }
        }
        return true;
    }

    parseBlockArguments(parser) {
        const args = [];
        if (parser.accept('(')) {
            while (!parser.accept(')')) {
                const value = parser.expect('%');
                parser.expect(':');
                const type = parser.parseType();
                args.push({ value, type });
                parser.accept(',');
            }
        }
        return args;
    }

    _parseFragmentOp(parser, op) {
        // mpmd.fragment<mesh="m1", origin=["f1"], stage_id=N> (%inputs) {attrs} (%block_args) { region } : (types) -> type
        parser.expect('<');
        // Parse attributes inside <>
        while (!parser.match('>')) {
            const attrName = parser.expect('id');
            parser.expect('=');
            // Use custom parser for origin attribute (array of UserOriginAttr)
            const attrValue = attrName === 'origin' ? this._parseOriginArray(parser) : parser.parseAttribute();
            op.addAttribute(attrName, attrValue);
            parser.accept(',');
        }
        parser.expect('>');
        const inputs = parser.parseArguments();
        op.operands.push(...inputs);
        parser.parseOptionalAttrDict(op.attributes);
        const entryArguments = this.parseBlockArguments(parser);
        if (parser.match('{')) {
            const region = op.addRegion();
            parser.parseRegion(region, entryArguments);
        }
        if (parser.accept(':')) {
            const type = parser.parseType();
            if (type instanceof mlir.FunctionType) {
                parser.resolveOperands(op.operands, type.inputs);
                parser.resolveOperands(op.results, type.results);
            }
        }
        return true;
    }

    _parseFragmentCallOp(parser, op) {
        // mpmd.fragment_call<mesh="m1", origin=["f1"]> @callee(%args) {attrs} : (types) -> type
        parser.expect('<');
        // Parse attributes inside <>
        while (!parser.match('>')) {
            const attrName = parser.expect('id');
            parser.expect('=');
            // Use custom parser for origin attribute (array of UserOriginAttr)
            const attrValue = attrName === 'origin' ? this._parseOriginArray(parser) : parser.parseAttribute();
            op.addAttribute(attrName, attrValue);
            parser.accept(',');
        }
        parser.expect('>');
        const callee = parser.expect('@');
        op.addAttribute('callee', callee);
        const args = parser.parseArguments();
        op.operands.push(...args);
        if (parser.match('{')) {
            parser.parseAttributeDict(op.attributes);
        }
        if (parser.accept(':')) {
            const type = parser.parseType();
            if (type instanceof mlir.FunctionType) {
                parser.resolveOperands(op.operands, type.inputs);
                parser.resolveOperands(op.results, type.results);
            }
        }
        return true;
    }

    _parseForOp(parser, op) {
        // mpmd.for (%inputs) {iterations = N, unroll_factor = M} (%block_args) { region } : type, type, ...
        const inputs = parser.parseArguments();
        op.operands.push(...inputs);
        parser.parseOptionalAttrDict(op.attributes);
        const entryArguments = this.parseBlockArguments(parser);
        if (parser.match('{')) {
            const region = op.addRegion();
            parser.parseRegion(region, entryArguments);
        }
        if (parser.accept(':')) {
            const resultTypes = [];
            do {
                resultTypes.push(parser.parseType());
            } while (parser.accept(','));
            parser.resolveOperands(op.results, resultTypes);
        }
        return true;
    }

    _parseMeshTensorType(parser, prefix) {
        // Parse !mpmd.mesh_tensor<"mesh_name", tensor<shape>, sharding=<...>>
        parser.expect('<');
        const meshName = parser.parseString();
        parser.expect(',');
        const tensorType = parser.parseType();
        const result = { name: prefix, meshName, tensorType };
        if (parser.accept(',')) {
            // Parse optional sharding
            if (parser.accept('id', 'sharding')) {
                parser.expect('=');
                result.sharding = parser.parseAttribute();
            }
        }
        parser.expect('>');
        return result;
    }

    // Parse UserOriginAttr in short format: "name"(count) where (count) is optional
    _parseUserOriginAttr(parser) {
        const name = parser.parseString();
        let transposeCount = 0;
        // Parse optional (transposeCount)
        if (parser.accept('(')) {
            const count = parser.parseInteger();
            transposeCount = count;
            parser.expect(')');
        }
        return { name, transposeCount };
    }

    // Parse array of UserOriginAttr: ["name1"(count1), "name2", ...]
    _parseOriginArray(parser) {
        const origins = [];
        parser.expect('[');
        while (!parser.accept(']')) {
            const origin = this._parseUserOriginAttr(parser);
            origins.push(origin);
            parser.accept(',');
        }
        return origins;
    }
};

mlir.SdyDialect = class extends mlir.Dialect {

    constructor(operations) {
        super(operations, 'sdy');
        this.registerCustomDirective('StrippedTensorShardingPerValueAttr', this._parseStrippedTensorShardingPerValueAttr.bind(this));
        this.registerCustomDirective('SingleBlockRegionNoBlockId', this._parseSingleBlockRegionNoBlockId.bind(this));
        this.registerCustomAttribute('Sdy_ListOfAxisRefLists', this._parseListOfAxisRefLists.bind(this));
        this.registerCustomAttribute('Sdy_ManualAxes', this._parseManualAxes.bind(this));
        this.registerCustomAttribute('Sdy_AllToAllParamList', this._parseAllToAllParamList.bind(this));
        this.registerCustomAttribute('Sdy_TensorSharding', this._parseTensorShardingAttrWrap.bind(this));
        this.registerCustomAttribute('Sdy_AxisRefList', this._parseAxisRefListWrap.bind(this));
    }

    parseOperation(parser, opName, op) {
        if (opName === 'sdy.constant' && !this.hasAssemblyFormat(opName)) {
            return this._parseConstantOp(parser, op);
        }
        return super.parseOperation(parser, opName, op);
    }

    _parseStrippedTensorShardingPerValueAttr(parser, op, args) {
        const shardings = [];
        parser.expect('[');
        while (!parser.accept(']')) {
            const sharding = this._parseTensorShardingAttr(parser);
            shardings.push(sharding);
            parser.accept(',');
        }
        if (args && args.length >= 1) {
            const attrName = args[0].replace('$', '');
            op.addAttribute(attrName, shardings);
        }
        return shardings;
    }

    _parseSingleBlockRegionNoBlockId(parser, op /*, args */) {
        const entryArguments = [];
        if (parser.accept('(')) {
            while (!parser.accept(')')) {
                const value = parser.expect('%');
                let type = null;
                const attrs = [];
                if (parser.accept(':')) {
                    type = parser.parseType();
                }
                if (parser.match('{')) {
                    parser.parseAttributeDict(attrs);
                }
                entryArguments.push({ value, type, attributes: attrs.length > 0 ? attrs : undefined });
                parser.accept(',');
            }
        }
        const region = op.addRegion();
        parser.parseRegion(region, entryArguments);
        return region;
    }

    _parseTensorShardingAttr(parser) {
        parser.expect('<');
        // Parse mesh or reference: either @mesh_name or mesh<["x"=N, ...]>
        let meshOrRef = null;
        if (parser.match('@')) {
            meshOrRef = parser.expect('@');
        } else if (parser.accept('id', 'mesh')) {
            meshOrRef = this._parseMeshAttr(parser);
        } else {
            throw new mlir.Error(`Expected '@' or 'mesh', but got '${parser.token.value}' ${parser.location()}`);
        }
        parser.expect(',');
        const dimShardings = this._parseDimensionShardings(parser);
        let replicatedAxes = null;
        let unreducedAxes = null;
        while (parser.accept(',')) {
            if (parser.accept('id', 'replicated')) {
                parser.expect('=');
                replicatedAxes = this._parseAxisRefList(parser);
            } else if (parser.accept('id', 'unreduced')) {
                parser.expect('=');
                unreducedAxes = this._parseAxisRefList(parser);
            }
        }
        parser.expect('>');
        return { meshOrRef, dimShardings, replicatedAxes, unreducedAxes };
    }

    _parseMeshAttr(parser) {
        parser.expect('<');
        const axes = [];
        parser.expect('[');
        while (!parser.accept(']')) {
            const name = parser.parseString();
            parser.expect('=');
            const size = parser.parseInteger();
            axes.push({ name, size });
            parser.accept(',');
        }
        let deviceIds = null;
        if (parser.accept(',')) {
            if (parser.accept('id', 'device_ids')) {
                parser.expect('=');
                deviceIds = parser.parseAttribute().value;
            }
        }
        parser.expect('>');
        return { axes, deviceIds };
    }

    _parseDimensionShardings(parser) {
        const dimShardings = [];
        parser.expect('[');
        while (!parser.accept(']')) {
            const dimSharding = this._parseDimensionShardingAttr(parser);
            dimShardings.push(dimSharding);
            parser.accept(',');
        }
        return dimShardings;
    }

    _parseDimensionShardingAttr(parser) {
        const axes = [];
        parser.expect('{');
        let isClosed = true;
        while (!parser.accept('}')) {
            if (parser.match('?')) {
                parser.expect('?');
                isClosed = false;
                parser.expect('}');
                break;
            }
            const axis = this._parseAxisRefAttr(parser);
            axes.push(axis);
            if (parser.accept(',')) {
                if (parser.match('?')) {
                    parser.expect('?');
                    isClosed = false;
                    parser.expect('}');
                    break;
                }
            }
        }
        // Parse optional priority suffix like p0, p1, etc.
        let priority = null;
        if (parser.match('id')) {
            const tokenValue = parser.token.value;
            if (typeof tokenValue === 'string' && tokenValue.startsWith('p') && /^p\d+$/.test(tokenValue)) {
                parser.expect('id');
                priority = parseInt(tokenValue.substring(1), 10);
            }
        }
        return { axes, isClosed, priority };
    }

    _parseAxisRefAttr(parser) {
        const name = parser.parseString();
        let subAxisInfo = null;
        if (parser.accept(':')) {
            subAxisInfo = this._parseSubAxisInfo(parser);
        }
        return { name, subAxisInfo };
    }

    _parseSubAxisInfo(parser) {
        parser.expect('(');
        const preSize = parser.parseInteger();
        parser.expect(')');
        const size = parser.parseInteger();
        return { preSize, size };
    }

    _parseAxisRefList(parser) {
        const axes = [];
        parser.expect('{');
        while (!parser.accept('}')) {
            const axis = this._parseAxisRefAttr(parser);
            axes.push(axis);
            parser.accept(',');
        }
        return axes;
    }

    _parseListOfAxisRefLists(parser) {
        const lists = [];
        parser.expect('[');
        while (!parser.accept(']')) {
            const list = this._parseAxisRefList(parser);
            lists.push(list);
            parser.accept(',');
        }
        return { value: lists };
    }

    _parseManualAxes(parser) {
        const axes = [];
        parser.expect('{');
        while (!parser.accept('}')) {
            const axis = parser.parseString();
            axes.push(axis);
            parser.accept(',');
        }
        return { value: axes };
    }

    _parseAllToAllParamList(parser) {
        const params = [];
        parser.expect('[');
        while (!parser.accept(']')) {
            const param = this._parseAllToAllParam(parser);
            params.push(param);
            parser.accept(',');
        }
        return { value: params };
    }

    _parseAllToAllParam(parser) {
        const axes = this._parseAxisRefList(parser);
        parser.expect(':');
        const splitDim = parser.parseInteger();
        parser.expect('->');
        const concatDim = parser.parseInteger();
        return { axes, splitDim, concatDim };
    }

    _parseTensorShardingAttrWrap(parser) {
        return { value: this._parseTensorShardingAttr(parser) };
    }

    _parseAxisRefListWrap(parser) {
        return { value: this._parseAxisRefList(parser) };
    }

    _parseConstantOp(parser, op) {
        parser.parseOptionalAttrDict(op.attributes);
        const attr = parser.parseDenseElementsAttr();
        op.addAttribute('value', attr.value || attr);
        if (attr.type) {
            if (op.results.length === 0) {
                op.results.push({ type: attr.type });
            } else {
                op.results[0].type = attr.type;
            }
        }
        return true;
    }
};

mlir.XlaDialect = class extends mlir.Dialect {

    constructor(operations) {
        super(operations, 'xla');
    }

    parseOperation(parser, opName, op) {
        // xla.apply_indexing #map (%dims)[%syms]
        // Variants: #map (%dims)[%syms], #map (%dims), #map [%syms]
        if (opName === 'xla.apply_indexing') {
            // Parse the indexing map reference (e.g., #map0)
            const map = parser.parseAttribute();
            op.addAttribute('map', map);
            // Parse optional dimensions in parentheses
            if (parser.accept('(')) {
                while (!parser.match(')')) {
                    if (parser.match('%')) {
                        const value = parser.expect('%');
                        op.operands.push({ value, name: 'dim' });
                    }
                    if (!parser.accept(',')) {
                        break;
                    }
                }
                parser.expect(')');
            }
            // Parse optional symbols in brackets
            if (parser.accept('[')) {
                while (!parser.match(']')) {
                    if (parser.match('%')) {
                        const value = parser.expect('%');
                        op.operands.push({ value, name: 'symbol' });
                    }
                    if (!parser.accept(',')) {
                        break;
                    }
                }
                parser.expect(']');
            }
            // The result types are inferred from the number of results (all index type)
            for (const result of op.results) {
                result.type = new mlir.PrimitiveType('index');
            }
            return true;
        }
        // xla.loop (%dims)[%ivs] -> (%map_results) in #map iter_args(%args = %inits) -> (types) { body }
        if (opName === 'xla.loop') {
            // Parse optional dimensions in parentheses
            if (parser.accept('(')) {
                while (!parser.match(')')) {
                    if (parser.match('%')) {
                        const value = parser.expect('%');
                        op.operands.push({ value, name: 'dim' });
                    }
                    if (!parser.accept(',')) {
                        break;
                    }
                }
                parser.expect(')');
            }
            // Parse [%ivs] -> (%map_results)
            if (parser.accept('[')) {
                while (!parser.match(']')) {
                    if (parser.match('%')) {
                        parser.expect('%');
                    }
                    if (!parser.accept(',')) {
                        break;
                    }
                }
                parser.expect(']');
            }
            if (parser.accept('->')) {
                parser.expect('(');
                while (!parser.match(')')) {
                    if (parser.match('%')) {
                        parser.expect('%');
                    }
                    if (!parser.accept(',')) {
                        break;
                    }
                }
                parser.expect(')');
            }
            // Parse 'in #map'
            parser.expect('id', 'in');
            const map = parser.parseAttribute();
            op.addAttribute('indexing_map_attr', map);
            // Parse 'iter_args(%args = %inits)'
            if (parser.accept('id', 'iter_args')) {
                parser.expect('(');
                while (!parser.match(')')) {
                    if (parser.match('%')) {
                        parser.expect('%');
                    }
                    if (parser.accept('=')) {
                        const initValue = parser.expect('%');
                        op.operands.push({ value: initValue, name: 'init' });
                    }
                    if (!parser.accept(',')) {
                        break;
                    }
                }
                parser.expect(')');
            }
            // Parse optional '-> (types)' or '-> type'
            if (parser.accept('->')) {
                if (parser.accept('(')) {
                    const types = parser.parseTypeListNoParens();
                    parser.expect(')');
                    for (const t of types) {
                        op.results.push({ type: t.toString() });
                    }
                } else {
                    // Single type without parentheses
                    const type = parser.parseType();
                    op.results.push({ type: type.toString() });
                }
            }
            // Parse region body
            if (parser.match('{')) {
                const region = op.addRegion();
                parser.parseRegion(region);
            }
            // Parse optional attributes
            parser.parseOptionalAttrDict(op.attributes);
            return true;
        }
        return super.parseOperation(parser, opName, op);
    }
};

mlir.XlaGpuDialect = class extends mlir.Dialect {

    constructor(operations) {
        super(operations, 'xla_gpu');
    }

    parseOperation(parser, opName, op) {
        // xla_gpu.shuffle_reduce(%ops) to N combiner=@func {attrs} : types
        if (opName === 'xla_gpu.shuffle_reduce') {
            // Parse operands in parentheses
            if (parser.accept('(')) {
                while (!parser.match(')')) {
                    if (parser.match('%')) {
                        const value = parser.expect('%');
                        op.operands.push({ value });
                    }
                    if (!parser.accept(',')) {
                        break;
                    }
                }
                parser.expect(')');
            }
            // Parse 'to N'
            parser.expect('id', 'to');
            const maxDistance = parser.expect('int');
            op.addAttribute('max_distance', parseInt(maxDistance, 10));
            // Parse 'combiner=@func'
            parser.expect('id', 'combiner');
            parser.expect('=');
            const combiner = parser.expect('@');
            op.addAttribute('combiner', new mlir.SymbolRefAttr(`@${combiner}`));
            // Parse optional attributes
            parser.parseOptionalAttrDict(op.attributes);
            // Parse : types
            if (parser.accept(':')) {
                const types = parser.parseTypeList();
                parser.resolveOperands(op.operands, types);
                parser.resolveOperands(op.results, types);
            }
            return true;
        }
        // xla_gpu.reduce (%inputs) inits(%inits) dimensions=[...] combiner=@func {attrs} : in_types to out_types
        if (opName === 'xla_gpu.reduce') {
            // Parse inputs in parentheses
            if (parser.accept('(')) {
                while (!parser.match(')')) {
                    if (parser.match('%')) {
                        const value = parser.expect('%');
                        op.operands.push({ value, name: 'inputs' });
                    }
                    if (!parser.accept(',')) {
                        break;
                    }
                }
                parser.expect(')');
            }
            // Parse 'inits(%init_values)'
            parser.expect('id', 'inits');
            parser.expect('(');
            while (!parser.match(')')) {
                if (parser.match('%')) {
                    const value = parser.expect('%');
                    op.operands.push({ value, name: 'init_values' });
                }
                if (!parser.accept(',')) {
                    break;
                }
            }
            parser.expect(')');
            // Parse 'dimensions=[0, 2]'
            parser.expect('id', 'dimensions');
            parser.expect('=');
            const dimensions = parser.parseAttribute();
            op.addAttribute('dimensions', dimensions);
            // Parse 'combiner=@func'
            parser.expect('id', 'combiner');
            parser.expect('=');
            const combiner = parser.expect('@');
            op.addAttribute('combiner', new mlir.SymbolRefAttr(`@${combiner}`));
            // Parse optional attributes
            parser.parseOptionalAttrDict(op.attributes);
            // Parse : in_types to out_types
            if (parser.accept(':')) {
                const inputTypes = parser.parseTypeList();
                // Assign types to input operands
                let idx = 0;
                for (const operand of op.operands) {
                    if (operand.name === 'inputs' && idx < inputTypes.length) {
                        operand.type = inputTypes[idx++];
                    }
                }
                // Parse 'to' and output types
                parser.expect('id', 'to');
                const outputTypes = parser.parseTypeList();
                parser.resolveOperands(op.results, outputTypes);
            }
            return true;
        }
        return super.parseOperation(parser, opName, op);
    }
};

mlir.XTileDialect = class extends mlir.Dialect {

    constructor(operations) {
        super(operations, 'xtile');
    }

    parseOperation(parser, opName, op) {
        // xtile.entry_func - function-like op with custom format
        if (opName === 'xtile.entry_func') {
            parser.parseFunctionOp(op, false);
            return true;
        }
        return super.parseOperation(parser, opName, op);
    }
};

mlir.TritonXlaDialect = class extends mlir.Dialect {

    constructor(operations) {
        super(operations, 'triton_xla');
        // Mark operations with custom directives as having custom assembly format
        this._operations.set('triton_xla.extract', { metadata: { name: 'triton_xla.extract', hasCustomAssemblyFormat: true } });
        this._operations.set('triton_xla.insert', { metadata: { name: 'triton_xla.insert', hasCustomAssemblyFormat: true } });
    }

    parseOperation(parser, opName, op) {
        // triton_xla.extract from $src as memref<shape, layout> [offsets] [sizes] [strides] : result_type
        if (opName === 'triton_xla.extract') {
            parser.expect('id', 'from');
            const src = parser.expect('%');
            op.operands.push({ value: src });
            parser.expect('id', 'as');
            // Parse AsMemRefType: memref<shape, layout> - the memref type describes the pointer's layout
            const memrefType = parser.parseType();
            // Extract shape and layout from the memref type
            this._extractMemRefInfo(memrefType, op, 'src_shape', 'src_layout');
            // Parse offsets, sizes, strides using dynamic index lists
            this._parseDynamicIndexList(parser, op, 'offsets', 'static_offsets');
            this._parseDynamicIndexList(parser, op, 'sizes', 'static_sizes');
            this._parseDynamicIndexList(parser, op, 'strides', 'static_strides');
            parser.parseOptionalAttrDict(op.attributes);
            parser.expect(':');
            const resultType = parser.parseType();
            op.results.push({ type: resultType.toString() });
            return true;
        }
        // triton_xla.insert $src into $dst as memref<shape, layout> [offsets] [sizes] [strides] : src_type
        if (opName === 'triton_xla.insert') {
            const src = parser.expect('%');
            op.operands.push({ value: src });
            parser.expect('id', 'into');
            const dst = parser.expect('%');
            op.operands.push({ value: dst });
            parser.expect('id', 'as');
            // Parse AsMemRefType: memref<shape, layout> - the memref type describes the pointer's layout
            const memrefType = parser.parseType();
            // Extract shape and layout from the memref type
            this._extractMemRefInfo(memrefType, op, 'dst_shape', 'dst_layout');
            // Parse offsets, sizes, strides using dynamic index lists
            this._parseDynamicIndexList(parser, op, 'offsets', 'static_offsets');
            this._parseDynamicIndexList(parser, op, 'sizes', 'static_sizes');
            this._parseDynamicIndexList(parser, op, 'strides', 'static_strides');
            parser.parseOptionalAttrDict(op.attributes);
            parser.expect(':');
            const srcType = parser.parseType();
            op.operands[0].type = srcType.toString();
            return true;
        }
        return super.parseOperation(parser, opName, op);
    }

    _extractMemRefInfo(memrefType, op, shapeAttrName, layoutAttrName) {
        // Extract shape and layout from memref type like memref<512x1x128xbf16, #xtile.layout<[2, 1, 0]>>
        const typeStr = memrefType.toString();
        // Extract shape dimensions
        const shapeMatch = typeStr.match(/memref<([\d?x]+)/);
        if (shapeMatch) {
            const dims = shapeMatch[1].split('x').filter((d) => d).map((d) => d === '?' ? -1 : parseInt(d, 10));
            op.addAttribute(shapeAttrName, dims);
        }
        // Extract layout if present
        const layoutMatch = typeStr.match(/#[a-z_.]+\.<\[([^\]]+)\]>/);
        if (layoutMatch) {
            const layout = layoutMatch[1].split(',').map((s) => parseInt(s.trim(), 10));
            op.addAttribute(layoutAttrName, layout);
        }
    }

    _parseDynamicIndexList(parser, op, dynamicName, staticName) {
        // Parse [val1, val2, ...] where vals can be %ssa or integer constants
        parser.expect('[');
        const staticValues = [];
        while (!parser.match(']')) {
            if (parser.match('%')) {
                const value = parser.expect('%');
                op.operands.push({ value, name: dynamicName });
                staticValues.push(-9223372036854775808n); // ShapedType::kDynamic
            } else if (parser.match('int')) {
                const value = parser.expect('int');
                staticValues.push(BigInt(value));
            }
            if (!parser.accept(',')) {
                break;
            }
        }
        parser.expect(']');
        op.addAttribute(staticName, staticValues);
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
                const [dialectName] = op.name.split('.');
                if (!this.operations.has(dialectName)) {
                    this.operations.set(dialectName, []);
                }
                this.operations.get(dialectName).push(op);
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
