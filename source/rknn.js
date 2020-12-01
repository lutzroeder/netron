/* jshint esversion: 6 */

var rknn = rknn || {};
var json = json || require('./json');

rknn.ModelFactory = class {

    match(context) {
        const buffer = context.buffer;
        if (buffer && buffer.length > 4 && [ 0x52, 0x4B, 0x4E, 0x4E ].every((value, index) => buffer[index] === value)) {
            return true;
        }
        return false;
    }

    open(context, host) {
        return rknn.Metadata.open(host).then((metadata) => {
            const container = new rknn.Container(context.buffer);
            return new rknn.Model(metadata, container.model);
        });
    }
};

rknn.Model = class {

    constructor(metadata, model) {
        this._version = model.version;
        this._producer = model.ori_network_platform || model.network_platform || '';
        this._runtime = model.target_platform ? model.target_platform.join(',') : '';
        this._graphs = [ new rknn.Graph(metadata, model) ];
    }

    get format() {
        return 'RKNN v' + this._version;
    }

    get producer() {
        return this._producer;
    }

    get runtime() {
        return this._runtime;
    }

    get graphs() {
        return this._graphs;
    }
};

rknn.Graph = class {

    constructor(metadata, model) {
        this._name = model.name || '';
        this._inputs = [];
        this._outputs = [];
        this._nodes = [];

        const args = new Map();
        for (const const_tensor of model.const_tensor) {
            const name = 'const_tensor:' + const_tensor.tensor_id.toString();
            const shape = new rknn.TensorShape(const_tensor.size);
            const type = new rknn.TensorType(const_tensor.dtype, shape);
            const tensor = new rknn.Tensor(type);
            const argument = new rknn.Argument(name, type, tensor);
            args.set(name, argument);
        }
        for (const virtual_tensor of model.virtual_tensor) {
            const name = virtual_tensor.node_id.toString() + ':' + virtual_tensor.output_port.toString();
            const argument = new rknn.Argument(name, null, null);
            args.set(name, argument);
        }
        for (const norm_tensor of model.norm_tensor) {
            const name = 'norm_tensor:' + norm_tensor.tensor_id.toString();
            const shape = new rknn.TensorShape(norm_tensor.size);
            const type = new rknn.TensorType(norm_tensor.dtype, shape);
            const argument = new rknn.Argument(name, type, null);
            args.set(name, argument);
        }

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
            }
        }

        for (const graph of model.graph) {
            const key = graph.right + ':' + graph.right_tensor_id.toString();
            const argument = args.get(key);
            const name = graph.left + ((graph.left_tensor_id === 0) ? '' : graph.left_tensor_id.toString());
            const parameter = new rknn.Parameter(name, [ argument ]);
            switch (graph.left) {
                case 'input': {
                    this._inputs.push(parameter);
                    break;
                }
                case 'output': {
                    this._outputs.push(parameter);
                    break;
                }
            }
        }

        for (const node of model.nodes) {
            this._nodes.push(new rknn.Node(metadata, node, args));
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

rknn.Parameter = class {

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

rknn.Argument = class {

    constructor(name, type, initializer) {
        if (typeof name !== 'string') {
            throw new rknn.Error("Invalid argument identifier '" + JSON.stringify(name) + "'.");
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

    constructor(metadata, node, args) {
        this._metadata = metadata;
        this._name = node.name || '';
        this._type = node.op;
        this._inputs = [];
        this._outputs = [];
        this._attributes = [];

        for (const input of node.input) {
            if (input.right_tensor) {
                const name = input.right_tensor.type + ':' + input.right_tensor.tensor_id.toString();
                const argument = args.get(name);
                this._inputs.push(new rknn.Parameter('', [ argument ]));
            }
            if (input.right_node) {
                const name = input.right_node.node_id.toString() + ':' + input.right_node.tensor_id.toString();
                const argument = args.get(name);
                this._inputs.push(new rknn.Parameter('', [ argument ]));
            }
        }

        for (const output of node.output) {
            if (output.right_tensor) {
                const name = output.right_tensor.type + ':' + output.right_tensor.tensor_id.toString();
                const argument = args.get(name);
                this._outputs.push(new rknn.Parameter('', [ argument ]));
            }
            if (output.right_node) {
                const name = output.right_node.node_id.toString() + ':' + output.right_node.tensor_id.toString();
                const argument = args.get(name);
                this._outputs.push(new rknn.Parameter('', [ argument ]));
            }
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
    }

    get name() {
        return this._name;
    }

    get type() {
        const prefix = 'VSI_NN_OP_';
        return this._type.startsWith(prefix) ? this._type.substring(prefix.length) : this.type;
    }

    get metadata() {
        return this._metadata.type(this._type);
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

    constructor(type) {
        this._type = type;
    }

    get type() {
        return this._type;
    }
};

rknn.TensorType = class {

    constructor(dataType, shape) {
        switch (dataType.vx_type) {
            case 'VSI_NN_TYPE_UINT8': this._dataType = 'uint8'; break;
            case 'VSI_NN_TYPE_INT32': this._dataType = 'int32'; break;
            case 'VSI_NN_TYPE_FLOAT16': this._dataType = 'float16'; break;
            case 'VSI_NN_TYPE_FLOAT32': this._dataType = 'float32'; break;
            default:
                throw new rknn.Error("Invalid data type '" + JSON.stringify(dataType) + "'.");
        }
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

    constructor(buffer) {
        const view = new DataView(buffer.buffer, buffer.byteOffset, buffer.byteLength);
        const signature = view.getUint64(0, true);
        if (signature.low !== 0x4e4e4b52 || signature.high !== 0) {
            throw new rknn.Error('Invalid RKNN signature.');
        }
        this._version = view.getUint64(8, true).toNumber();
        const blocks = [];
        let position = 16;
        while (position < buffer.length) {
            const size = view.getUint64(position, true).toNumber();
            position += 8;
            blocks.push(buffer.subarray(position, position + size));
            position += size;
        }
        const reader = json.TextReader.create(blocks[1]);
        this._model = reader.read();
    }

    get version() {
        return this._version;
    }

    get model() {
        return this._model;
    }
};

rknn.Metadata = class {

    static open(host) {
        if (rknn.Metadata._metadata) {
            return Promise.resolve(rknn.Metadata._metadata);
        }
        return host.request(null, 'rknn-metadata.json', 'utf-8').then((data) => {
            rknn.Metadata._metadata = new rknn.Metadata(data);
            return rknn.Metadata._metadata;
        }).catch(() => {
            rknn.Metadata._metadata = new rknn.Metadata(null);
            return rknn.Metadata._metadata;
        });
    }

    constructor(data) {
        this._map = new Map();
        if (data) {
            const items = JSON.parse(data);
            if (items) {
                for (const item of items) {
                    item.schema.name = item.name;
                    this._map.set(item.name, item.schema);
                }
            }
        }
    }

    type(name) {
        return this._map.has(name) ? this._map.get(name) : null;
    }

    attribute(type, name) {
        const schema = this.type(type);
        if (schema) {
            let attributeMap = schema.attributeMap;
            if (!attributeMap) {
                attributeMap = {};
                if (schema.attributes) {
                    for (const attribute of schema.attributes) {
                        attributeMap[attribute.name] = attribute;
                    }
                }
                schema.attributeMap = attributeMap;
            }
            const attributeSchema = attributeMap[name];
            if (attributeSchema) {
                return attributeSchema;
            }
        }
        return null;
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
