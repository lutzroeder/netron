
var rknn = {};
var base = require('./base');
var flatbuffers = require('./flatbuffers');
var json = require('./json');

rknn.ModelFactory = class {

    match(context) {
        return rknn.Container.open(context);
    }

    async open(context, target) {
        await context.require('./rknn-schema');
        rknn.schema = flatbuffers.get('rknn').rknn;
        const metadata = await context.metadata('rknn-metadata.json');
        const container = target;
        const type = container.type;
        switch (type) {
            case 'json': {
                const buffer = container.value;
                const reader = json.TextReader.open(buffer);
                const model = reader.read();
                return new rknn.Model(metadata, type, model, container.next);
            }
            case 'flatbuffers': {
                const buffer = container.value;
                const reader = flatbuffers.BinaryReader.open(buffer);
                const model = rknn.schema.Model.create(reader);
                return new rknn.Model(metadata, type, model, null);
            }
            case 'openvx': {
                const buffer = container.value;
                const model = new openvx.Model(buffer);
                return new rknn.Model(metadata, type, model, null);
            }
            default: {
                throw new rknn.Error("Unsupported RKNN format '" + container.type + "'.");
            }
        }
    }
};

rknn.Model = class {

    constructor(metadata, type, model, next) {
        switch (type) {
            case 'json': {
                this._format = 'RKNN v' + model.version.split('-').shift();
                this._name = model.name || '';
                this._producer = model.ori_network_platform || model.network_platform || '';
                this._runtime = model.target_platform ? model.target_platform.join(',') : '';
                this._graphs = [ new rknn.Graph(metadata, type, model.name || '', model, next) ];
                break;
            }
            case 'flatbuffers': {
                const version = model.compiler.split('-').shift();
                this._format = 'RKNN Lite' + (version ? ' v' + version : '');
                this._runtime = model.runtime;
                this._name = model.name || '';
                this._graphs = model.graphs.map((graph) => new rknn.Graph(metadata, type, '', graph, null));
                this._metadata = [];
                this._metadata.push({ name: 'source', value: model.source });
                break;
            }
            case 'openvx': {
                this._format = 'RKNN OpenVX';
                this._name = model.name || '';
                this._graphs = [ new rknn.Graph(metadata, type, '', model, next) ];
                break;
            }
            default: {
                throw new rknn.Error("Unsupported RKNN model type '" + type + "'.");
            }
        }
    }

    get format() {
        return this._format;
    }

    get name() {
        return this._name;
    }

    get producer() {
        return this._producer;
    }

    get runtime() {
        return this._runtime;
    }

    get metadata() {
        return this._metadata;
    }

    get graphs() {
        return this._graphs;
    }
};

rknn.Graph = class {

    constructor(metadata, type, name, obj, next) {
        this._name = name;
        this._inputs = [];
        this._outputs = [];
        this._nodes = [];
        switch (type) {
            case 'json': {
                const dataType = (value) => {
                    const type = value.vx_type.startsWith('VSI_NN_TYPE_') ? value.vx_type.split('_').pop().toLowerCase() : value.vx_type;
                    switch (type) {
                        case 'uint8':
                        case 'int8':
                        case 'int16':
                        case 'int32':
                        case 'int64':
                        case 'float16':
                        case 'float32':
                        case 'float64':
                        case 'vdata':
                            return type;
                        default:
                            if (value.vx_type !== '') {
                                throw new rknn.Error("Invalid data type '" + JSON.stringify(dataType) + "'.");
                            }
                            return '?';
                    }
                };
                const model = obj;
                const args = new Map();
                for (const const_tensor of model.const_tensor) {
                    const name = 'const_tensor:' + const_tensor.tensor_id.toString();
                    const shape = new rknn.TensorShape(const_tensor.size);
                    const type = new rknn.TensorType(dataType(const_tensor.dtype), shape);
                    const tensor = new rknn.Tensor(type, const_tensor.offset, next.value);
                    const value = new rknn.Value(name, type, tensor);
                    args.set(name, value);
                }
                for (const virtual_tensor of model.virtual_tensor) {
                    const name = virtual_tensor.node_id.toString() + ':' + virtual_tensor.output_port.toString();
                    const value = new rknn.Value(name, null, null);
                    args.set(name, value);
                }
                for (const norm_tensor of model.norm_tensor) {
                    const name = 'norm_tensor:' + norm_tensor.tensor_id.toString();
                    const shape = new rknn.TensorShape(norm_tensor.size);
                    const type = new rknn.TensorType(dataType(norm_tensor.dtype), shape);
                    const value = new rknn.Value(name, type, null);
                    args.set(name, value);
                }
                const arg = (name) => {
                    if (!args.has(name)) {
                        args.set(name, new rknn.Value(name, null, null));
                    }
                    return args.get(name);
                };
                for (const node of model.nodes) {
                    node.input = [];
                    node.output = [];
                }
                for (const connection of model.connection) {
                    switch (connection.left) {
                        case 'input':
                            model.nodes[connection.node_id].input.push(connection);
                            if (connection.right_node) {
                                model.nodes[connection.right_node.node_id].output[connection.right_node.tensor_id] = connection;
                            }
                            break;
                        case 'output':
                            model.nodes[connection.node_id].output.push(connection);
                            break;
                        default:
                            throw new rknn.Error("Unsupported left connection '" + connection.left + "'.");
                    }
                }
                for (const graph of model.graph) {
                    const key = graph.right + ':' + graph.right_tensor_id.toString();
                    const value = arg(key);
                    const name = graph.left + (graph.left_tensor_id === 0 ? '' : graph.left_tensor_id.toString());
                    const argument = new rknn.Argument(name, [ value ]);
                    switch (graph.left) {
                        case 'input':
                            this._inputs.push(argument);
                            break;
                        case 'output':
                            this._outputs.push(argument);
                            break;
                        default:
                            throw new rknn.Error("Unsupported left graph connection '" + graph.left + "'.");
                    }
                }
                this._nodes = model.nodes.map((node) => new rknn.Node(metadata, type, node, arg, next));
                break;
            }
            case 'flatbuffers': {
                const graph = obj;
                const dataTypes = [ 'unk0', 'int32', '?', 'int8', '?', 'int16', 'float32', 'int64', '?', '?', 'float16', '?', '?', 'unk13' ];
                const args = graph.tensors.map((tensor) => {
                    const shape = new rknn.TensorShape(Array.from(tensor.shape));
                    const dataType = tensor.data_type < dataTypes.length ? dataTypes[tensor.data_type] : '?';
                    if (dataType === '?') {
                        throw new rknn.Error("Unsupported tensor data type '" + tensor.data_type + "'.");
                    }
                    const type = new rknn.TensorType(dataType, shape);
                    const initializer = tensor.kind !== 4 && tensor.kind !== 5 ? null : new rknn.Tensor(type, 0, null);
                    return new rknn.Value(tensor.name, type, initializer);
                });
                const arg = (index) => {
                    if (index >= args.length) {
                        throw new rknn.Error("Invalid tensor index '" + index.toString() + "'.");
                    }
                    return args[index];
                };
                this._nodes = graph.nodes.map((node) => new rknn.Node(metadata, type, node, arg, next));
                break;
            }
            case 'openvx': {
                const model = obj;
                this._nodes = model.nodes.map((node) => new rknn.Node(metadata, type, node, null, next));
                break;
            }
            default: {
                throw new rknn.Error("Unsupported RKNN graph type '" + type + "'.");
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

rknn.Argument = class {

    constructor(name, value) {
        this._name = name;
        this._value = value;
    }

    get name() {
        return this._name;
    }

    get value() {
        return this._value;
    }
};

rknn.Value = class {

    constructor(name, type, initializer) {
        if (typeof name !== 'string') {
            throw new rknn.Error("Invalid value identifier '" + JSON.stringify(name) + "'.");
        }
        this._name = name;
        this._type = type || null;
        this._initializer = initializer || null;
    }

    get name() {
        return this._name;
    }

    get type() {
        return this._type;
    }

    get initializer() {
        return this._initializer;
    }
};

rknn.Node = class {

    constructor(metadata, type, node, arg, next) {
        this._inputs = [];
        this._outputs = [];
        this._attributes = [];
        switch (type) {
            case 'json': {
                this._name = node.name || '';
                if (node.op === 'VSI_NN_OP_NBG' && next && next.type === 'openvx') {
                    const buffer = next.value;
                    const model = new openvx.Model(buffer);
                    this._type = new rknn.Graph(metadata, next.type, 'NBG', model, null);
                } else if (node.op === 'RKNN_OP_NNBG' && next && next.type === 'flatbuffers') {
                    const buffer = next.value;
                    const reader = flatbuffers.BinaryReader.open(buffer);
                    const model = rknn.schema.Model.create(reader);
                    this._type = new rknn.Graph(metadata, next.type, 'NNBG', model.graphs[0], null);
                } else {
                    this._type = Object.assign({}, metadata.type(node.op) || { name: node.op });
                    for (const prefix of [ 'VSI_NN_OP_', 'RKNN_OP_' ]) {
                        this._type.name = this._type.name.startsWith(prefix) ? this._type.name.substring(prefix.length) : this._type.name;
                    }
                }
                node.input = node.input || [];
                for (let i = 0; i < node.input.length;) {
                    const input = this._type && this._type.inputs && i < this._type.inputs.length ? this._type.inputs[i] : { name: i === 0 ? 'input' : i.toString() };
                    const count = input.list ? node.input.length - i : 1;
                    const list = node.input.slice(i, i + count).map((input) => {
                        if (input.right_tensor) {
                            return arg(input.right_tensor.type + ':' + input.right_tensor.tensor_id.toString());
                        }
                        if (input.right_node) {
                            return arg(input.right_node.node_id.toString() + ':' + input.right_node.tensor_id.toString());
                        }
                        throw new rknn.Error('Invalid input argument.');
                    });
                    this._inputs.push(new rknn.Argument(input.name, list));
                    i += count;
                }
                node.output = node.output || [];
                for (let i = 0; i < node.output.length;) {
                    const output = this._metadata && this._metadata.outputs && i < this._metadata.outputs.length ? this._metadata.outputs[i] : { name: i === 0 ? 'output' : i.toString() };
                    const count = output.list ? node.output.length - i : 1;
                    const list = node.output.slice(i, i + count).map((output) => {
                        if (output.right_tensor) {
                            return arg(output.right_tensor.type + ':' + output.right_tensor.tensor_id.toString());
                        }
                        if (output.right_node) {
                            return arg(output.right_node.node_id.toString() + ':' + output.right_node.tensor_id.toString());
                        }
                        throw new rknn.Error('Invalid output argument.');
                    });
                    this._outputs.push(new rknn.Argument(output.name, list));
                    i += count;
                }
                if (node.nn) {
                    const nn = node.nn;
                    for (const key of Object.keys(nn)) {
                        const params = nn[key];
                        for (const name of Object.keys(params)) {
                            const value = params[name];
                            this._attributes.push(new rknn.Attribute(name, value));
                        }
                    }
                }
                break;
            }
            case 'flatbuffers': {
                this._name = node.name;
                this._type = metadata.type(node.type);
                if (node.inputs.length > 0) {
                    const inputs = this._type.inputs || (node.inputs.length === 1 ? [ { name: "input" } ] : [ { name: "inputs", list: true } ]);
                    if (Array.isArray(inputs) && inputs.length > 0 && inputs[0].list === true) {
                        this._inputs = [new rknn.Argument(inputs[0].name, Array.from(node.inputs).map((input) => arg(input))) ];
                    } else {
                        this._inputs = Array.from(node.inputs).map((input, index) => {
                            const value = arg(input);
                            return new rknn.Argument(index < inputs.length ? inputs[index].name : index.toString(), [ value ]);
                        });
                    }
                }
                if (node.outputs.length > 0) {
                    const outputs = this._type.outputs || (node.outputs.length === 1 ? [ { name: "output" } ] : [ { name: "outputs", list: true } ]);
                    if (Array.isArray(outputs) && outputs.length > 0 && outputs[0].list === true) {
                        this._outputs = [ new rknn.Argument(outputs[0].name, Array.from(node.outputs).map((output) => arg(output))) ];
                    } else {
                        this._outputs = Array.from(node.outputs).map((output, index) => {
                            const value = arg(output);
                            return new rknn.Argument(index < outputs.length ? outputs[index].name : index.toString(), [ value ]);
                        });
                    }
                }
                break;
            }
            case 'openvx': {
                this._name = '';
                this._type = metadata.type(node.type);
                break;
            }
            default: {
                throw new rknn.Error("Unsupported RKNN node type '" + type + "'.");
            }
        }
    }

    get name() {
        return this._name;
    }

    get type() {
        return this._type;
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

rknn.Attribute = class {

    constructor(name, value) {
        this._name = name;
        this._value = value;
    }

    get name() {
        return this._name;
    }

    get value() {
        return this._value;
    }
};

rknn.Tensor = class {

    constructor(type, offset, weights) {
        this._type = type;
        this._data = null;
        let itemsize = 0;
        switch (this._type.dataType) {
            case 'uint8': itemsize = 1; break;
            case 'int8': itemsize = 1; break;
            case 'int16': itemsize = 2; break;
            case 'int32': itemsize = 4; break;
            case 'int64': itemsize = 8; break;
            case 'float16': itemsize = 2; break;
            case 'float32': itemsize = 4; break;
            case 'float64': itemsize = 8; break;
            case 'vdata': itemsize = 1; break;
            default: throw new rknn.Error("Unsupported tensor data type '" + this._type.dataType + "'.");
        }
        if (weights) {
            const shape = type.shape.dimensions;
            const size = itemsize * shape.reduce((a, b) => a * b, 1);
            if (size > 0) {
                this._data = weights.slice(offset, offset + size);
            }
        }
    }

    get type() {
        return this._type;
    }

    get values() {
        return this._data;
    }
};

rknn.TensorType = class {

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

    toString() {
        return this.dataType + this._shape.toString();
    }
};

rknn.TensorShape = class {

    constructor(shape) {
        this._dimensions = shape;
    }

    get dimensions() {
        return this._dimensions;
    }

    toString() {
        if (!this._dimensions || this._dimensions.length == 0) {
            return '';
        }
        return '[' + this._dimensions.join(',') + ']';
    }
};

rknn.Container = class {

    static open(context) {
        const stream = context.stream;
        if (stream) {
            const container = new rknn.Container(stream, stream.length);
            if (container.type) {
                return container;
            }
        }
        return null;
    }

    constructor(stream, length) {
        this._stream = stream;
        this._length = length;
        this._type = '';
        if ((stream.position + 16) <= this._length) {
            const signature = [ 0x52, 0x4B, 0x4E, 0x4E ]; // RKNN
            if (stream.peek(signature.length).every((value, index) => value === signature[index])) {
                this._type = 'json';
            }
        }
        if ((stream.position + 16) <= this._length) {
            const signature = [ 0x43, 0x59, 0x50, 0x54, 0x52, 0x4B, 0x4E, 0x4E ]; // CYPTRKNN
            if (stream.peek(signature.length).every((value, index) => value === signature[index])) {
                this._type = 'crypt';
            }
        }
        if ((stream.position + 8) <= this._length) {
            const signature = [ 0x52, 0x4B, 0x4E, 0x4E ]; // RKNN
            if (stream.peek(8).subarray(4, 8).every((value, index) => value === signature[index])) {
                this._type = 'flatbuffers';
            }
        }
        if ((stream.position + 8) <= this._length) {
            const signature = [ 0x56, 0x50, 0x4D, 0x4E ]; // VPMN
            if (stream.peek(signature.length).every((value, index) => value === signature[index])) {
                this._type = 'openvx';
            }
        }
    }

    get type() {
        return this._type;
    }

    get value() {
        this.read();
        return this._value;
    }

    get next() {
        this.read();
        return this._next;
    }

    read() {
        if (this._stream) {
            const stream = this._stream;
            delete this._stream;
            switch (this._type) {
                case 'crypt': {
                    throw new rknn.Error('Invalid file content. File contains undocumented encrypted RKNN data.');
                }
                case 'json': {
                    const uint64 = () => {
                        const buffer = stream.read(8);
                        const reader = new base.BinaryReader(buffer);
                        return reader.uint64();
                    };
                    stream.skip(8);
                    const version = uint64();
                    const data_size = uint64();
                    switch (version) {
                        case 0x0001:
                        case 0x1001:
                            break;
                        case 0x0002:
                        case 0x0004:
                            if (data_size > 0) {
                                stream.skip(40);
                            }
                            break;
                        default:
                            throw new rknn.Error("Unsupported RKNN container version '" + version + "'.");
                    }
                    this._next = new rknn.Container(stream, data_size);
                    this._next.read();
                    const value_size = uint64();
                    this._value = stream.read(value_size);
                    break;
                }
                case 'flatbuffers': {
                    this._value = stream.read(this._length);
                    this._next = null;
                    break;
                }
                case 'openvx': {
                    this._value = stream.read(this._length);
                    this._next = null;
                    break;
                }
                case '': {
                    this._value = stream.read(this._length);
                    this._next = null;
                    break;
                }
                default: {
                    throw new rknn.Error("Unsupported container type '" + this._format + "'.");
                }
            }
        }
    }
};

var openvx = openvx || {};

openvx.Model = class {

    constructor(buffer) {
        const reader = new openvx.BinaryReader(buffer);
        reader.skip(4); // signature
        const major = reader.uint16();
        /* const minor = */ reader.uint16();
        reader.skip(4);
        this._name = reader.string(64);
        this._nodes = new Array(reader.uint32());
        if (major > 3) {
            reader.skip(296);
        } else if (major > 1) {
            reader.skip(288);
        } else {
            reader.skip(32);
        }
        /* const inputOffset = */ reader.uint32();
        /* const inputSize = */ reader.uint32();
        /* const outputOffset = */ reader.uint32();
        /* const outputSize = */ reader.uint32();
        const nodeOffset = reader.uint32();
        /* const nodeSize = */ reader.uint32();
        reader.seek(nodeOffset);
        for (let i = 0; i < this._nodes.length; i++) {
            const type = reader.string(64);
            const node = { type: type };
            node.index = reader.uint32();
            node.c = reader.uint32();
            if (major > 3) {
                node.d = reader.uint32();
            }
            this._nodes[i] = node;
        }
    }

    get name() {
        return this._name;
    }

    get nodes() {
        return this._nodes;
    }
};

openvx.BinaryReader = class extends base.BinaryReader {

    string(length) {
        const buffer = this.read(length);
        const index = buffer.indexOf(0);
        const data = index === -1 ? buffer : buffer.subarray(0, index);
        this._decoder = this._decoder || new TextDecoder('ascii');
        return this._decoder.decode(data);
    }
};

rknn.Error = class extends Error {

    constructor(message) {
        super(message);
        this.name = 'Error loading RKNN model.';
    }
};

if (typeof module !== 'undefined' && typeof module.exports === 'object') {
    module.exports.ModelFactory = rknn.ModelFactory;
}
