/* jshint esversion: 6 */

var paddle = paddle || {};
var protobuf = protobuf || require('./protobuf');

paddle.ModelFactory = class {

    match(context) {
        const identifier = context.identifier;
        const extension = identifier.split('.').pop().toLowerCase();
        if (identifier === '__model__' || extension === 'paddle' || extension === 'pdmodel') {
            return true;
        }
        if (extension === 'pbtxt' || extension === 'txt') {
            const tags = context.tags('pbtxt');
            if (tags.has('blocks')) {
                return true;
            }
        }
        if (paddle.Container.open(context)) {
            return true;
        }
        return false;
    }

    open(context) {
        return paddle.Metadata.open(context).then((metadata) => {
            return context.require('./paddle-proto').then(() => {
                paddle.proto = protobuf.get('paddle').paddle.framework.proto;
                const container = paddle.Container.open(context);
                if (container) {
                    return new paddle.Model(metadata, container.format, null, container.weights);
                }
                else {
                    let programDesc = null;
                    let format = 'PaddlePaddle';
                    const identifier = context.identifier;
                    const parts = identifier.split('.');
                    const extension = parts.pop().toLowerCase();
                    const base = parts.join('.');
                    switch (extension) {
                        case 'pbtxt':
                        case 'txt': {
                            try {
                                const buffer = context.stream.peek();
                                const reader = protobuf.TextReader.create(buffer);
                                programDesc = paddle.proto.ProgramDesc.decodeText(reader);
                            }
                            catch (error) {
                                const message = error && error.message ? error.message : error.toString();
                                throw new paddle.Error('File text format is not paddle.ProgramDesc (' + message.replace(/\.$/, '') + ').');
                            }
                            break;
                        }
                        default: {
                            try {
                                const buffer = context.stream.peek();
                                const reader = protobuf.Reader.create(buffer);
                                programDesc = paddle.proto.ProgramDesc.decode(reader);
                            }
                            catch (error) {
                                const message = error && error.message ? error.message : error.toString();
                                throw new paddle.Error('File format is not paddle.ProgramDesc (' + message.replace(/\.$/, '') + ').');
                            }
                            break;
                        }
                    }
                    if (programDesc.version && programDesc.version.version && programDesc.version.version.toNumber) {
                        const version = programDesc.version.version.toNumber();
                        if (version > 0) {
                            const list = [ Math.floor(version / 1000000) % 1000, Math.floor(version / 1000) % 1000, version % 1000 ];
                            if (list.slice(-1).pop() === 0) {
                                list.pop();
                                if (list.slice(-1).pop() === 0) {
                                    list.pop();
                                }
                            }
                            format += ' v' + list.map((item) => item.toString()).join('.');
                        }
                    }
                    const variables = new Set();
                    for (const block of programDesc.blocks) {
                        const blockVars = new Set();
                        for (const variable of block.vars) {
                            if (variable.persistable && variable.type &&
                                variable.type.type != paddle.proto.VarType.Type.FETCH_LIST &&
                                variable.type.type != paddle.proto.VarType.Type.FEED_MINIBATCH) {
                                blockVars.add(variable.name);
                            }
                        }
                        for (const op of block.ops) {
                            for (const input of op.inputs) {
                                for (const argument of input.arguments) {
                                    if (blockVars.has(argument)) {
                                        variables.add(argument);
                                    }
                                }
                            }
                        }
                    }
                    const vars = Array.from(variables).sort();
                    const tensors = new Map();
                    const load_entries = (programDesc) => {
                        const promises = vars.map((name) => context.request(name, null));
                        return Promise.all(promises).then((streams) => {
                            for (let i = 0; i < vars.length; i++) {
                                tensors.set(vars[i], new paddle.Tensor(null, streams[i]));
                            }
                            return new paddle.Model(metadata, format, programDesc, tensors);
                        }).catch((/* err */) => {
                            return new paddle.Model(metadata, format, programDesc, tensors);
                        });
                    };
                    if (extension === 'pdmodel') {
                        return context.request(base + '.pdiparams', null).then((stream) => {
                            while (stream.position < stream.length) {
                                tensors.set(vars.shift(), new paddle.Tensor(null, stream));
                            }
                            return new paddle.Model(metadata, format, programDesc, tensors);
                        }).catch((/* err */) => {
                            return load_entries(programDesc, null);
                        });
                    }
                    return load_entries(programDesc, null);
                }
            });
        });
    }
};

paddle.Model = class {

    constructor(metadata, format, programDesc, tensors) {
        this._format = format;
        this._graphs = programDesc ?
            programDesc.blocks.map((block) => new paddle.Graph(metadata, block, tensors)) :
            [ new paddle.Graph(metadata, null, tensors) ];
    }

    get format() {
        return this._format;
    }

    get graphs() {
        return this._graphs;
    }
};

paddle.Graph = class {

    constructor(metadata, block, tensors) {
        this._nodes = [];
        this._inputs = [];
        this._outputs = [];
        if (block) {
            this._name = block.idx.toString();

            const args = new Map();
            for (const variable of block.vars) {
                const type = variable.type && variable.type.type && variable.type.lod_tensor && variable.type.lod_tensor.tensor ? paddle.Utility.createTensorType(variable.type.lod_tensor.tensor) : null;
                const tensor = variable.persistable && variable.type && variable.type.type != paddle.proto.VarType.Type.FETCH_LIST && variable.type.type != paddle.proto.VarType.Type.FEED_MINIBATCH ? (tensors.get(variable.name) || new paddle.Tensor(type)) : null;
                args.set(variable.name, new paddle.Argument(variable.name, type, tensor));
            }

            const scope = {};
            for (let i = 0; i < block.ops.length; i++) {
                for (const input of block.ops[i].inputs) {
                    input.arguments = input.arguments.map((argument) => scope[argument] ? scope[argument] : argument);
                }
                for (const output of block.ops[i].outputs) {
                    output.arguments = output.arguments.map((argument) => {
                        if (scope[argument]) {
                            const next = argument + '\n' + i.toString(); // custom argument id
                            scope[argument] = next;
                            return next;
                        }
                        scope[argument] = argument;
                        return argument;
                    });
                }
            }

            for (const op of block.ops) {
                for (const input of op.inputs) {
                    for (const argument of input.arguments) {
                        const name = argument;
                        if (!args.has(name)) {
                            args.set(name, new paddle.Argument(name, null, null));
                        }
                    }
                }
                for (const output of op.outputs) {
                    for (const argument of output.arguments) {
                        const name = argument;
                        if (!args.has(name)) {
                            args.set(name, new paddle.Argument(name, null, null));
                        }
                    }
                }
            }

            let lastNode = null;
            let lastOutput = null;
            for (const op of block.ops) {
                if (op.type == 'feed') {
                    const inputName = op.attrs.filter((attr) => attr.name == 'col')[0].i.toString();
                    this._inputs.push(new paddle.Parameter(inputName, op.outputs[0].arguments.map((id) => args.get(id))));
                }
                else if (op.type == 'fetch') {
                    const outputName = op.attrs.filter((attr) => attr.name == 'col')[0].i.toString();
                    this._outputs.push(new paddle.Parameter(outputName, op.inputs[0].arguments.map((id) => args.get(id))));
                }
                else {
                    const node = new paddle.Node(metadata, op, args);
                    if (op.inputs.length == 1 && op.inputs[0].arguments.length == 1 &&
                        op.outputs.length >= 1 && op.outputs[0].arguments.length == 1 &&
                        op.inputs[0].arguments[0].split('\n').shift() == op.outputs[0].arguments[0].split('\n').shift() &&
                        lastNode &&
                        lastOutput == op.inputs[0].arguments[0].split('\n').shift()) {
                        lastNode.chain.push(node);
                    }
                    else {
                        this._nodes.push(node);
                        lastNode = null;
                        lastOutput = null;
                        if (op.outputs.length == 1 && op.outputs[0].arguments.length == 1) {
                            lastNode = node;
                            lastOutput = op.outputs[0].arguments[0].split('\n').shift();
                        }
                    }
                }
            }
        }
        else {
            const args = new Map();
            const ops = new Map();
            for (const pair of tensors) {
                const name = pair[0];
                const tensor = pair[1];
                args.set(name, new paddle.Argument(name, tensor.type, tensor));
                const separator = [ '.', '_' ].find((separator) => name.split(separator).length > 1);
                const parts = name.split(separator);
                const parameter_name = parts.pop();
                const op_name = parts.join(separator);
                if (!ops.has(op_name)) {
                    ops.set(op_name, { name: op_name, type: 'Weights', inputs: [] });
                }
                const op = ops.get(op_name);
                op.inputs.push({ parameter: parameter_name, arguments: [ name ] });
            }
            for (const pair of ops) {
                const op = pair[1];
                this._nodes.push(new paddle.Node(metadata, op, args));
            }
        }
    }

    get name() {
        return this._name;
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

paddle.Parameter = class {

    constructor(name, args) {
        this._name = name;
        this._arguments = args;
    }

    get name() {
        return this._name;
    }

    get visible() {
        return true;
    }

    get arguments() {
        return this._arguments;
    }
};

paddle.Argument = class {

    constructor(name, type, initializer) {
        if (typeof name !== 'string') {
            throw new paddle.Error("Invalid argument identifier '" + JSON.stringify(name) + "'.");
        }
        this._name = name;
        this._type = type || null;
        this._initializer = initializer || null;
    }

    get name() {
        return this._name;
    }

    get type() {
        if (this._type) {
            return this._type;
        }
        if (this._initializer) {
            return this._initializer.type;
        }
        return null;
    }

    get initializer() {
        return this._initializer;
    }
};

paddle.Node = class {

    constructor(metadata, op, args) {
        this._metadata = metadata;
        this._type = op.type;
        this._name = op.name || '';
        this._attributes = [];
        this._inputs = [];
        this._outputs = [];
        this._chain = [];
        if (op.attrs) {
            for (const attr of op.attrs) {
                const schema = metadata.attribute(this._type, this._name);
                this._attributes.push(new paddle.Attribute(schema, attr));
            }
        }
        if (op.inputs) {
            for (const input of op.inputs) {
                if (input.arguments.length > 0) {
                    this._inputs.push(new paddle.Parameter(input.parameter, input.arguments.map((name) => args.get(name))));
                }
            }
        }
        if (op.outputs) {
            for (const output of op.outputs) {
                if (output.arguments.length > 0) {
                    this._outputs.push(new paddle.Parameter(output.parameter, output.arguments.map((name) => args.get(name))));
                }
            }
        }
        this._update(this._inputs, 'X');
        this._update(this._inputs, 'Input');
        this._update(this._outputs, 'Y');
        this._update(this._outputs, 'Out');
    }

    get type() {
        return this._type;
    }

    get name() {
        return this._name;
    }

    get metadata() {
        return this._metadata.type(this._type);
    }

    get attributes() {
        return this._attributes;
    }

    get inputs() {
        return this._inputs;
    }

    get outputs() {
        return this._outputs;
    }

    get chain() {
        return this._chain;
    }

    _update(list, name) {
        let item = null;
        for (let i = 0; i < list.length; i++) {
            if (list[i].name == name) {
                item = list[i];
                list.splice(i, 1);
                break;
            }
        }
        if (item) {
            list.splice(0, 0, item);
        }
    }
};

paddle.Attribute = class {

    constructor(schema, attr) {
        this._name = attr.name;
        this._value = '?';
        switch (attr.type) {
            case paddle.proto.AttrType.STRING:
                this._type = 'string';
                this._value = attr.s;
                break;
            case paddle.proto.AttrType.STRINGS:
                this._type = 'string[]';
                this._value = attr.strings;
                break;
            case paddle.proto.AttrType.BOOLEAN:
                this._type = 'boolean';
                this._value = attr.b;
                break;
            case paddle.proto.AttrType.BOOLEANS:
                this._type = 'boolean[]';
                this._value = attr.bools;
                break;
            case paddle.proto.AttrType.FLOAT:
                this._type = 'float32';
                this._value = attr.f;
                break;
            case paddle.proto.AttrType.FLOATS:
                this._type = 'float[]';
                this._value = attr.floats;
                break;
            case paddle.proto.AttrType.INT:
                this._type = 'int32';
                this._value = attr.i;
                break;
            case paddle.proto.AttrType.INTS:
                this._type = 'int32[]';
                this._value = attr.ints;
                break;
            case paddle.proto.AttrType.LONG:
                this._type = 'int64';
                break;
            case paddle.proto.AttrType.LONGS:
                this._type = 'int64[]';
                break;
            default:
                break;
        }
        switch (this._name) {
            case 'use_mkldnn':
            case 'use_cudnn':
            case 'op_callstack':
            case 'op_role':
            case 'op_role_var':
            case 'op_namescope':
            case 'is_test':
                this._visible = false;
                break;
        }
        if (schema) {
            if (Object.prototype.hasOwnProperty.call(schema, 'default')) {
                const defaultValue = schema.default;
                const value = this._value;
                if (defaultValue == value) {
                    this._visible = false;
                }
                else if (Array.isArray(value) && Array.isArray(defaultValue) && value.length == defaultValue.length) {
                    if (value.every((item, index) => { return item == defaultValue[index]; })) {
                        this._visible = false;
                    }
                }

            }
        }
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

paddle.Tensor = class {

    constructor(type, data) {
        this._type = type;
        if (data && !Array.isArray(data)) {
            if (data.__module__ === 'numpy' && data.__name__ === 'ndarray') {
                this._type = new paddle.TensorType(data.dtype.name, new paddle.TensorShape(data.shape));
                this._data = data.data;
                this._kind = 'NumPy Array';
            }
            else {
                const stream = data;
                const signature = stream.read(16);
                if (!signature.every((value) => value === 0x00)) {
                    throw new paddle.Error('Invalid paddle.TensorDesc signature.');
                }
                const buffer = stream.read(4);
                const length = new DataView(buffer.buffer, buffer.byteOffset, buffer.byteLength).getUint32(0, true);
                const reader = protobuf.Reader.create(stream.read(length));
                const tensorDesc = paddle.proto.VarType.TensorDesc.decode(reader);
                const size = tensorDesc.dims.reduce((a, b) => a * b.toNumber());
                let itemsize = 0;
                switch (tensorDesc.data_type) {
                    case paddle.proto.VarType.Type.FP32: itemsize = 4; break;
                    default: throw new paddle.Error("Invalid inference params data type '" + tensorDesc.data_type + "'.");
                }
                this._type = paddle.Utility.createTensorType(tensorDesc);
                this._data = stream.read(itemsize * size);
            }
        }
    }

    get kind() {
        return this._kind;
    }

    get type() {
        return this._type;
    }

    get state() {
        return this._context().state || null;
    }

    get value() {
        const context = this._context();
        if (context.state) {
            return null;
        }
        context.limit = Number.MAX_SAFE_INTEGER;
        return this._decode(context, 0);
    }

    toString() {
        const context = this._context();
        if (context.state) {
            return '';
        }
        context.limit = 10000;
        const value = this._decode(context, 0);
        return paddle.Tensor._stringify(value, '', '    ');
    }

    _context() {
        const context = {};
        context.index = 0;
        context.count = 0;
        context.state = null;

        if (!this._data) {
            context.state = 'Tensor data is empty.';
            return context;
        }
        if (!this._type) {
            context.state = 'Tensor has no data type.';
            return context;
        }

        context.dataType = this._type.dataType;
        context.shape = this._type.shape.dimensions;
        context.view = new DataView(this._data.buffer, this._data.byteOffset, this._data.byteLength);

        switch (context.dataType) {
            case 'float32':
            case 'int32':
            case 'int64':
                break;
            default:
                context.state = "Tensor data type '" + context.dataType + "' is not implemented.";
                break;
        }
        return context;
    }

    _decode(context, dimension) {
        const shape = context.shape.length !== 0 ? context.shape : [ 1 ];
        const results = [];
        const size = shape[dimension];
        if (dimension == shape.length - 1) {
            for (let i = 0; i < size; i++) {
                if (context.count > context.limit) {
                    results.push('...');
                    return results;
                }
                switch (context.dataType) {
                    case 'float32':
                        results.push(context.view.getFloat32(context.index, true));
                        context.index += 4;
                        context.count++;
                        break;
                    case 'int32':
                        results.push(context.view.getInt32(context.index, true));
                        context.index += 4;
                        context.count++;
                        break;
                    case 'int64':
                        results.push(context.view.getInt64(context.index, true));
                        context.index += 8;
                        context.count++;
                        break;

                }
            }
        }
        else {
            for (let j = 0; j < size; j++) {
                if (context.count > context.limit) {
                    results.push('...');
                    return results;
                }
                results.push(this._decode(context, dimension + 1));
            }
        }
        if (context.shape.length == 0) {
            return results[0];
        }
        return results;
    }

    static _stringify(value, indentation, indent) {
        if (Array.isArray(value)) {
            const result = [];
            result.push(indentation + '[');
            const items = value.map((item) => paddle.Tensor._stringify(item, indentation + indent, indent));
            if (items.length > 0) {
                result.push(items.join(',\n'));
            }
            result.push(indentation + ']');
            return result.join('\n');
        }
        if (typeof value == 'string') {
            return indentation + value;
        }
        if (value == Infinity) {
            return indentation + 'Infinity';
        }
        if (value == -Infinity) {
            return indentation + '-Infinity';
        }
        if (isNaN(value)) {
            return indentation + 'NaN';
        }
        return indentation + value.toString();
    }
};

paddle.TensorType = class {

    constructor(dataType, shape) {
        this._dataType = dataType;
        this._shape = shape;
    }

    get dataType() {
        return this._dataType;
    }

    get shape() {
        return this._shape;
    }

    get denotation() {
        return this._denotation;
    }

    toString() {
        return this._dataType + this._shape.toString();
    }
};

paddle.TensorShape = class {

    constructor(dimensions) {
        dimensions = dimensions.map((dimension) => Number.isInteger(dimension) ? dimension : dimension.toNumber());
        this._dimensions = dimensions.map((dimension) => {
            return dimension != -1 ? dimension : '?';
        });
    }

    get dimensions() {
        return this._dimensions;
    }

    toString() {
        return (this._dimensions && this._dimensions.length) ? ('[' + this._dimensions.join(',') + ']') : '';
    }
};

paddle.Utility = class {

    static createTensorType(desc) {
        if (!paddle.Utility._dataTypes) {
            const length = Math.max.apply(null, Object.values(paddle.proto.VarType.Type));
            paddle.Utility._dataTypes = new Array(length);
            for (const key of Object.keys(paddle.proto.VarType.Type)) {
                const index = paddle.proto.VarType.Type[key];
                let name = key.toLowerCase();
                switch (name) {
                    case 'bool': name = 'boolean'; break;
                    case 'bf16': name = 'bfloat16'; break;
                    case 'fp16': name = 'float16'; break;
                    case 'fp32': name = 'float32'; break;
                    case 'fp64': name = 'float64'; break;
                }
                paddle.Utility._dataTypes[index] = name;
            }
        }
        const dataType = desc.data_type < paddle.Utility._dataTypes.length ? paddle.Utility._dataTypes[desc.data_type] : '?';
        return new paddle.TensorType(dataType, new paddle.TensorShape(desc.dims));
    }
};

paddle.Container = class {

    static open(context) {
        const extension = [ 'zip', 'tar' ].find((extension) => context.entries(extension).length > 0);
        if (extension) {
            const entries = context.entries(extension).filter((entry) => !entry.name.endsWith('/') && !entry.name.split('/').pop().startsWith('.')).slice();
            if (entries.length > 2 && entries.every((entry) => entry.name.split('_').length > 0 && entry.data.slice(0, 16).every((value) => value === 0x00))) {
                return new paddle.Container('entries', entries);
            }
        }
        const obj = context.open('pkl');
        if (obj && !Array.isArray(obj) && Object(obj) === obj) {
            return new paddle.Container('pdparams', obj);
        }
        return null;
    }

    constructor(format, data) {
        this._format = format;
        this._data = data;
    }

    get format() {
        switch (this._format) {
            case 'entries':
                return 'PaddlePaddle Weights';
            case 'pdparams':
                return 'PaddlePaddle Pickle';
        }
        return null;
    }

    get model() {
        this._initialize();
        return this._model;
    }

    get weights() {
        this._initialize();
        return this._weights;
    }

    _initialize() {
        if (!this._weights) {
            switch (this._format) {
                case 'entries': {
                    let rootFolder = null;
                    for (const entry of this._data) {
                        const name = entry.name;
                        if (name.startsWith('PaxHeader/') || (name.startsWith('.') && !name.startsWith('./'))) {
                            continue;
                        }
                        const parts = name.split('/');
                        const folder = ((parts.length > 2 && parts[0] === '.') ? ('./' + parts[1] + '/') : (parts.length > 1 ? parts[0] + '/' : ''));
                        rootFolder = (rootFolder === null) ? folder : (rootFolder !== '' && folder !== rootFolder) ? '' : folder;
                    }
                    this._weights = new Map();
                    for (const entry of this._data) {
                        if (entry.name.startsWith(rootFolder)) {
                            const name = entry.name.substring(rootFolder.length);
                            this._weights.set(name, new paddle.Tensor(null, entry.stream));
                        }
                    }
                    break;
                }
                case 'pdparams': {
                    const map = null; // this._data['StructuredToParameterName@@'];
                    this._weights = new Map();
                    for (const key of Object.keys(this._data)) {
                        const value = this._data[key];
                        if (value && !Array.isArray(value) && value.__module__ === 'numpy' && value.__name__ === 'ndarray') {
                            const name = map ? map[key] : key;
                            this._weights.set(name, new paddle.Tensor(null, value));
                        }
                    }
                    break;
                }
            }
            delete this._format;
        }
    }
};

paddle.Metadata = class {

    static open(context) {
        if (paddle.Metadata._metadata) {
            return Promise.resolve(paddle.Metadata._metadata);
        }
        return context.request('paddle-metadata.json', 'utf-8', null).then((data) => {
            paddle.Metadata._metadata = new paddle.Metadata(data);
            return paddle.Metadata._metadata;
        }).catch(() => {
            paddle.Metadata._metadata = new paddle.Metadata(null);
            return paddle.Metadata._metadata;
        });
    }

    constructor(data) {
        this._map = {};
        this._attributeCache = {};
        if (data) {
            const metadata = JSON.parse(data);
            this._map = new Map(metadata.map((item) => [ item.name, item ]));
        }
    }

    type(name) {
        return this._map[name] || null;
    }

    attribute(type, name) {
        let map = this._attributeCache[type];
        if (!map) {
            map = {};
            const schema = this.type(type);
            if (schema && schema.attributes && schema.attributes.length > 0) {
                for (const attribute of schema.attributes) {
                    map[attribute.name] = attribute;
                }
            }
            this._attributeCache[type] = map;
        }
        return map[name] || null;
    }
};

paddle.Error = class extends Error {
    constructor(message) {
        super(message);
        this.name = 'Error loading PaddlePaddle model.';
    }
};

if (typeof module !== 'undefined' && typeof module.exports === 'object') {
    module.exports.ModelFactory = paddle.ModelFactory;
}
