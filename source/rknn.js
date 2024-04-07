
import * as base from './base.js';
import * as flatbuffers from './flatbuffers.js';
import * as json from './json.js';

const rknn = {};
const openvx = {};

rknn.ModelFactory = class {

    match(context) {
        const container = rknn.Container.open(context);
        if (container) {
            context.type = 'rknn';
            context.target = container;
        }
    }

    async open(context) {
        rknn.schema = await context.require('./rknn-schema');
        rknn.schema = rknn.schema.rknn;
        const metadata = await context.metadata('rknn-metadata.json');
        const target = context.target;
        target.read();
        if (target.has('json')) {
            const buffer = target.get('json');
            const reader = json.TextReader.open(buffer);
            const model = reader.read();
            return new rknn.Model(metadata, 'json', model, target);
        }
        if (target.has('flatbuffers')) {
            const buffer = target.get('flatbuffers');
            const reader = flatbuffers.BinaryReader.open(buffer);
            const model = rknn.schema.Model.create(reader);
            return new rknn.Model(metadata, 'flatbuffers', model, null);
        }
        if (target.has('openvx')) {
            const buffer = target.get('openvx');
            const model = new openvx.Model(buffer);
            return new rknn.Model(metadata, 'openvx', model, null);
        }
        throw new rknn.Error("Unsupported RKNN format.");
    }
};

rknn.Model = class {

    constructor(metadata, type, model, container) {
        switch (type) {
            case 'json': {
                this.format = `RKNN v${model.version.split('-').shift()}`;
                this.name = model.name || '';
                this.producer = model.ori_network_platform || model.network_platform || '';
                this.runtime = model.target_platform ? model.target_platform.join(',') : '';
                this.graphs = [new rknn.Graph(metadata, type, model.name || '', model, container)];
                break;
            }
            case 'flatbuffers': {
                const version = model.compiler.split('-').shift();
                this.format = `RKNN Lite${version ? ` v${version}` : ''}`;
                this.runtime = model.runtime;
                this.name = model.name || '';
                this.graphs = model.graphs.map((graph) => new rknn.Graph(metadata, type, '', graph, null));
                this.source = model.source;
                break;
            }
            case 'openvx': {
                this.format = 'RKNN OpenVX';
                this.name = model.name || '';
                this.graphs = [new rknn.Graph(metadata, type, '', model, container)];
                break;
            }
            default: {
                throw new rknn.Error(`Unsupported RKNN model type '${type}'.`);
            }
        }
    }
};

rknn.Graph = class {

    constructor(metadata, type, name, obj, container) {
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
                                throw new rknn.Error(`Invalid data type '${JSON.stringify(dataType)}'.`);
                            }
                            return '?';
                    }
                };
                const model = obj;
                const values = new Map();
                for (const const_tensor of model.const_tensor) {
                    const name = `const_tensor:${const_tensor.tensor_id}`;
                    const shape = new rknn.TensorShape(const_tensor.size);
                    const type = new rknn.TensorType(dataType(const_tensor.dtype), shape);
                    const tensor = new rknn.Tensor(type, const_tensor.offset, null);
                    const value = new rknn.Value(name, type, tensor);
                    values.set(name, value);
                }
                for (const virtual_tensor of model.virtual_tensor) {
                    const name = `${virtual_tensor.node_id}:${virtual_tensor.output_port}`;
                    const value = new rknn.Value(name, null, null);
                    values.set(name, value);
                }
                for (const norm_tensor of model.norm_tensor) {
                    const name = `norm_tensor:${norm_tensor.tensor_id}`;
                    const shape = new rknn.TensorShape(norm_tensor.size);
                    const type = new rknn.TensorType(dataType(norm_tensor.dtype), shape);
                    const value = new rknn.Value(name, type, null);
                    values.set(name, value);
                }
                const value = (name) => {
                    if (!values.has(name)) {
                        values.set(name, new rknn.Value(name, null, null));
                    }
                    return values.get(name);
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
                            throw new rknn.Error(`Unsupported left connection '${connection.left}'.`);
                    }
                }
                for (const graph of model.graph) {
                    const key = `${graph.right}:${graph.right_tensor_id}`;
                    const name = graph.left + (graph.left_tensor_id === 0 ? '' : graph.left_tensor_id.toString());
                    const argument = new rknn.Argument(name, [value(key)]);
                    switch (graph.left) {
                        case 'input':
                            this._inputs.push(argument);
                            break;
                        case 'output':
                            this._outputs.push(argument);
                            break;
                        default:
                            throw new rknn.Error(`Unsupported left graph connection '${graph.left}'.`);
                    }
                }
                this._nodes = model.nodes.map((node) => new rknn.Node(metadata, type, node, value, container));
                break;
            }
            case 'flatbuffers': {
                const graph = obj;
                const dataTypes = ['unk0', 'int32', '?', 'int8', '?', 'int16', 'float32', 'int64', '?', '?', 'float16', '?', '?', 'unk13'];
                const args = graph.tensors.map((tensor) => {
                    const shape = new rknn.TensorShape(Array.from(tensor.shape));
                    const dataType = tensor.data_type < dataTypes.length ? dataTypes[tensor.data_type] : '?';
                    if (dataType === '?') {
                        throw new rknn.Error(`Unsupported tensor data type '${tensor.data_type}'.`);
                    }
                    const type = new rknn.TensorType(dataType, shape);
                    const initializer = tensor.kind !== 4 && tensor.kind !== 5 ? null : new rknn.Tensor(type, 0, null);
                    return new rknn.Value(tensor.name, type, initializer);
                });
                const arg = (index) => {
                    if (index >= args.length) {
                        throw new rknn.Error(`Invalid tensor index '${index}'.`);
                    }
                    return args[index];
                };
                this._nodes = graph.nodes.map((node) => new rknn.Node(metadata, type, node, arg, container));
                break;
            }
            case 'openvx': {
                const model = obj;
                this._nodes = model.nodes.map((node) => new rknn.Node(metadata, type, node, null, container));
                break;
            }
            default: {
                throw new rknn.Error(`Unsupported RKNN graph type '${type}'.`);
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
            throw new rknn.Error(`Invalid value identifier '${JSON.stringify(name)}'.`);
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

    constructor(metadata, type, node, value, container) {
        this._inputs = [];
        this._outputs = [];
        this._attributes = [];
        switch (type) {
            case 'json': {
                this._name = node.name || '';
                if (node.op === 'VSI_NN_OP_NBG' && container && container.has('openvx')) {
                    const buffer = container.get('openvx');
                    const model = new openvx.Model(buffer);
                    this._type = new rknn.Graph(metadata, 'openvx', 'NBG', model, null);
                } else if (node.op === 'RKNN_OP_NNBG' && container && container.has('flatbuffers')) {
                    const buffer = container.get('flatbuffers');
                    const reader = flatbuffers.BinaryReader.open(buffer);
                    const model = rknn.schema.Model.create(reader);
                    this._type = new rknn.Graph(metadata, 'flatbuffers', 'NNBG', model.graphs[0], null);
                } else {
                    const type = metadata.type(node.op);
                    this._type = type ? { ...type } : { name: node.op };
                    for (const prefix of ['VSI_NN_OP_', 'RKNN_OP_']) {
                        this._type.name = this._type.name.startsWith(prefix) ? this._type.name.substring(prefix.length) : this._type.name;
                    }
                }
                node.input = node.input || [];
                for (let i = 0; i < node.input.length;) {
                    const input = this._type && this._type.inputs && i < this._type.inputs.length ? this._type.inputs[i] : { name: i === 0 ? 'input' : i.toString() };
                    const count = input.list ? node.input.length - i : 1;
                    const list = node.input.slice(i, i + count).map((input) => {
                        if (input.right_tensor) {
                            return value(`${input.right_tensor.type}:${input.right_tensor.tensor_id}`);
                        }
                        if (input.right_node) {
                            return value(`${input.right_node.node_id}:${input.right_node.tensor_id}`);
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
                            return value(`${output.right_tensor.type}:${output.right_tensor.tensor_id}`);
                        }
                        if (output.right_node) {
                            return value(`${output.right_node.node_id}:${output.right_node.tensor_id}`);
                        }
                        throw new rknn.Error('Invalid output argument.');
                    });
                    this._outputs.push(new rknn.Argument(output.name, list));
                    i += count;
                }
                if (node.nn) {
                    for (const params of Object.values(node.nn)) {
                        for (const [name, value] of Object.entries(params)) {
                            const attribute = new rknn.Attribute(name, value);
                            this._attributes.push(attribute);
                        }
                    }
                }
                break;
            }
            case 'flatbuffers': {
                this._name = node.name;
                this._type = metadata.type(node.type);
                if (node.inputs.length > 0) {
                    const inputs = this._type.inputs || (node.inputs.length === 1 ? [{ name: "input" }] : [{ name: "inputs", list: true }]);
                    if (Array.isArray(inputs) && inputs.length > 0 && inputs[0].list === true) {
                        this._inputs = [new rknn.Argument(inputs[0].name, Array.from(node.inputs).map((input) => value(input)))];
                    } else {
                        this._inputs = Array.from(node.inputs).map((input, index) => {
                            return new rknn.Argument(index < inputs.length ? inputs[index].name : index.toString(), [value(input)]);
                        });
                    }
                }
                if (node.outputs.length > 0) {
                    const outputs = this._type.outputs || (node.outputs.length === 1 ? [{ name: "output" }] : [{ name: "outputs", list: true }]);
                    if (Array.isArray(outputs) && outputs.length > 0 && outputs[0].list === true) {
                        const values = Array.from(node.outputs).map((output) => value(output));
                        const argument = new rknn.Argument(outputs[0].name, values);
                        this._outputs = [argument];
                    } else {
                        this._outputs = Array.from(node.outputs).map((output, index) => {
                            return new rknn.Argument(index < outputs.length ? outputs[index].name : index.toString(), [value(output)]);
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
                throw new rknn.Error(`Unsupported RKNN node type '${type}'.`);
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
            default: throw new rknn.Error(`Unsupported tensor data type '${this._type.dataType}'.`);
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
        if (!this._dimensions || this._dimensions.length === 0) {
            return '';
        }
        return `[${this._dimensions.join(',')}]`;
    }
};

rknn.Container = class extends Map {

    static open(context) {
        const stream = context.stream;
        if (stream) {
            const signature = rknn.Container.signature(stream);
            switch (signature) {
                case 'rknn':
                case 'openvx':
                case 'flatbuffers':
                case 'cyptrknn':
                    return new rknn.Container(stream, signature);
                default:
                    break;
            }
            const obj = context.peek('json');
            if (obj && obj.version && Array.isArray(obj.nodes) && obj.network_platform) {
                const entries = new Map();
                entries.set('json', stream);
                return new rknn.Container(null, null, entries);
            }
        }
        return null;
    }

    constructor(stream, signature, entries) {
        super(entries);
        this.stream = stream;
        this.signature = signature;
    }

    read() {
        const stream = this.stream;
        if (stream) {
            switch (this.signature) {
                case 'rknn': {
                    const uint64 = () => {
                        const buffer = stream.read(8);
                        const reader = base.BinaryReader.open(buffer);
                        return reader.uint64().toNumber();
                    };
                    stream.skip(8);
                    const version = uint64();
                    const data_size = uint64();
                    switch (version) {
                        case 0x0001:
                        case 0x1001:
                            break;
                        case 0x0002:
                        case 0x1002:
                        case 0x0003:
                        case 0x1003:
                        case 0x0004:
                        case 0x1004:
                        case 0x0005:
                        case 0x0006:
                            if (data_size > 0) {
                                stream.skip(40);
                            }
                            break;
                        default:
                            throw new rknn.Error(`Unsupported RKNN container version '${version}'.`);
                    }
                    const signature = rknn.Container.signature(stream, data_size);
                    const data = stream.read(data_size);
                    const json_size = uint64();
                    const json = stream.read(json_size);
                    this.set('json', json);
                    if (signature) {
                        this.set(signature, data);
                    }
                    break;
                }
                case 'openvx':
                case 'flatbuffers': {
                    this.set(this.signature, stream.peek());
                    break;
                }
                case 'cyptrknn': {
                    throw new rknn.Error('Invalid file content. File contains undocumented encrypted RKNN data.');
                }
                default: {
                    break;
                }
            }
            delete this.stream;
        }
    }

    static signature(stream, length) {
        length = length || stream.length;
        if (stream && (stream.position + 16) <= length) {
            const signature = [0x52, 0x4B, 0x4E, 0x4E]; // RKNN
            if (stream.peek(signature.length).every((value, index) => value === signature[index])) {
                return 'rknn';
            }
        }
        if (stream && (stream.position + 16) <= length) {
            const signature = [0x43, 0x59, 0x50, 0x54, 0x52, 0x4B, 0x4E, 0x4E]; // CYPTRKNN
            if (stream.peek(signature.length).every((value, index) => value === signature[index])) {
                return 'cyptrknn';
            }
        }
        if (stream && (stream.position + 8) <= length) {
            const signature = [0x52, 0x4B, 0x4E, 0x4E]; // RKNN
            if (stream.peek(8).subarray(4, 8).every((value, index) => value === signature[index])) {
                return 'flatbuffers';
            }
        }
        if (stream && (stream.position + 8) <= length) {
            const signature = [0x56, 0x50, 0x4D, 0x4E]; // VPMN
            if (stream.peek(signature.length).every((value, index) => value === signature[index])) {
                return 'openvx';
            }
        }
        return undefined;
    }
};

openvx.BufferReader = class {

    constructor(buffer) {
        this._reader = base.BinaryReader.open(buffer);
    }

    seek(position) {
        this._reader.seek(position);
    }

    skip(offset) {
        this._reader.skip(offset);
    }

    read(length) {
        return this._reader.read(length);
    }

    uint16() {
        return this._reader.uint16();
    }

    uint32() {
        return this._reader.uint32();
    }

    string(length) {
        const buffer = this.read(length);
        const index = buffer.indexOf(0);
        const data = index === -1 ? buffer : buffer.subarray(0, index);
        this._decoder = this._decoder || new TextDecoder('ascii');
        return this._decoder.decode(data);
    }
};

openvx.Model = class {

    constructor(buffer) {
        const reader = new openvx.BufferReader(buffer);
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
            const node = { type };
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

rknn.Error = class extends Error {

    constructor(message) {
        super(message);
        this.name = 'Error loading RKNN model.';
    }
};

export const ModelFactory = rknn.ModelFactory;
