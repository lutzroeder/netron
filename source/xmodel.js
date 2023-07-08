
var xmodel = {};
var protobuf = require('./protobuf');

xmodel.ModelFactory = class {

    match(context) {
        const tags = context.tags('pb');
        if (tags.get(5) === 2) {
            return 'xmodel.pb';
        }
        return undefined;
    }

    async open(context) {
        await context.require('./xmodel-proto');
        let graph = null;
        try {
            xmodel.proto = protobuf.get('xmodel').serial_v2;
            const stream = context.stream;
            const reader = protobuf.BinaryReader.open(stream);
            graph = xmodel.proto.Graph.decode(reader);
        } catch (error) {
            const message = error && error.message ? error.message : error.toString();
            throw new xmodel.Error('File format is not serial_v2.Graph (' + message.replace(/\.$/, '') + ').');
        }
        return new xmodel.Model(graph);
    }
};

xmodel.Model = class {

    constructor(graph) {
        this.name = graph.graph_name || '';
        this.format = 'xmodel';
        this.producer = graph && graph.graph_attr && graph.graph_attr.origin && graph.graph_attr.origin.string_value ? graph.graph_attr.origin.string_value : '';
        this.graphs = [ new xmodel.Graph(graph) ];
    }
};

xmodel.Graph = class {

    constructor(graph) {
        const metadata = new xmodel.Metadata(graph.op_defs);
        this.inputs = [];
        this.outputs = [];
        const counts = new Map();
        for (const op_node of graph.op_node) {
            for (const arg of op_node.args) {
                for (const arg_op of arg.arg_ops) {
                    counts.set(arg_op, counts.has(arg_op) ? counts.get(arg_op) + 1 : 1);
                }
            }
        }
        const args = new Map();
        const arg = (name, node, initializer) => {
            if (!args.has(name)) {
                args.set(name, new xmodel.Value(name, node, initializer));
            }
            return args.get(name);
        };
        const nodes = [];
        for (const node of graph.op_node) {
            if (node.args.length === 0) {
                if (node.op_type === 'data' || node.op_type === 'data-fix') {
                    const value = arg(node.op_name, node);
                    this.inputs.push(new xmodel.Argument(node.op_name, [ value ]));
                    continue;
                }
            }
            if (node.args.length === 0 && counts.get(node.op_name) === 1) {
                if (node.op_type === 'const' || node.op_type === 'const-fix') {
                    arg(node.op_name, node, true);
                    continue;
                }
            }
            arg(node.op_name, node);
            nodes.push(node);
        }
        this.nodes = nodes.map((node) => new xmodel.Node(metadata, node, arg));
    }
};

xmodel.Argument = class {

    constructor(name, value) {
        this.name = name;
        this.value = value;
    }
};

xmodel.Value = class {

    constructor(name, node, initializer) {
        if (typeof name !== 'string') {
            throw new xmodel.Error("Invalid value identifier '" + JSON.stringify(name) + "'.");
        }
        this.name = name;
        if (node) {
            const tensor = node.output_tensor;
            if (tensor && tensor.tensor_attr && tensor.data_type) {
                if (initializer) {
                    this.initializer = new xmodel.Tensor(node);
                } else {
                    this._type = new xmodel.TensorType(tensor);
                }
            }
        }
    }

    get type() {
        if (this.initializer) {
            return this.initializer.type;
        }
        return this._type;
    }
};

xmodel.Node = class {

    constructor(metadata, op_node, arg) {
        this.name = op_node.op_name || '';
        this.type = metadata.type(op_node.op_type);
        this.inputs = [];
        this.outputs = [];
        this.attributes = [];
        this.chain = [];
        if (op_node.op_attr) {
            for (const entry of Object.entries(op_node.op_attr)) {
                const name = entry[0];
                if (name === 'device') {
                    this.device = entry[1].string_value;
                    continue;
                }
                if (name === 'workload') {
                    continue;
                }
                if (name.startsWith('quant_in_') || name.startsWith('quant_out_')) {
                    continue;
                }
                const value = xmodel.Utility.attribute(entry[1]);
                if (name === 'nonlinear' && value.value && value.value !== 'NONE' && value.value !== 0) {
                    let activation = value.value;
                    if (typeof activation === 'string') {
                        activation = activation.toLowerCase();
                    } else if (Number.isInteger(activation) && activation < 5) {
                        activation = [ 'none', 'relu', 'prelu', 'leakyrelu', 'relu6' ][activation];
                    } else {
                        activation = JSON.stringify(activation);
                    }
                    this.chain.push(new xmodel.Node(metadata, { op_type: activation }, arg));
                    continue;
                }
                this.attributes.push(new xmodel.Attribute(metadata.attribute(this._type, name), name, value));
            }
        }
        if (op_node.args) {
            for (const input of op_node.args) {
                const args = input.arg_ops.map((arg_op) => arg(arg_op));
                this.inputs.push(new xmodel.Argument(input.arg_name, args));
            }
        }
        if (op_node.op_name) {
            this.outputs.push(new xmodel.Argument('output', [ arg(op_node.op_name) ]));
        }
    }
};

xmodel.Attribute = class {

    constructor(metadata, name, attribute) {
        this.name = name;
        this.type = attribute.type;
        this.value = attribute.value;
        if (metadata) {
            if (metadata.default !== undefined) {
                if (metadata.default === this._value) {
                    this._visible = false;
                }
                if (Array.isArray(metadata.default) && Array.isArray(this._value) &&
                    metadata.default.length === this._value.length && metadata.default.every((value, index) => value === this._value[index])) {
                    this._visible = false;
                }
            }
        }
    }

    get visible() {
        return this._visible == false ? false : true;
    }
};

xmodel.TensorType = class {

    constructor(tensor) {
        switch (tensor.data_type) {
            case 0: this.dataType = 'int'; break;
            case 1: this.dataType = 'uint'; break;
            case 2: this.dataType = 'xint'; break;
            case 4: this.dataType = 'float'; break;
            case 3: this.dataType = 'xuint'; break;
            default: throw new xmodel.Error("Unsupported data type '" + tensor.data_type + "'.");
        }
        this.dataType += tensor.tensor_bit_width.toString();
        this.shape = new xmodel.TensorShape(tensor.tensor_dim);
        if (tensor.tensor_attr) {
            const attr = {};
            for (const entry of Object.entries(tensor.tensor_attr)) {
                const key = entry[0];
                const value = entry[1][entry[1].value];
                if (key.startsWith('quant_')) {
                    continue;
                }
                attr[key] = value;
                const denotation = [];
                if (attr.fix_point !== undefined) {
                    denotation.push(attr.fix_point.toString() + '.');
                }
                if (attr.round_mode !== undefined) {
                    denotation.push(attr.round_mode.toString());
                }
                if (denotation.length > 0) {
                    this.denotation = denotation.join(' ');
                }
            }
        }
    }

    toString() {
        return (this.dataType || '?') + this.shape.toString();
    }
};

xmodel.TensorShape = class {

    constructor(dimensions) {
        this.dimensions = Array.from(dimensions);
    }

    toString() {
        if (!this.dimensions || this.dimensions.length == 0) {
            return '';
        }
        return '[' + this.dimensions.map((dimension) => dimension.toString()).join(',') + ']';
    }
};

xmodel.Tensor = class {

    constructor(node) {
        this.type = new xmodel.TensorType(node.output_tensor);
        this.category = node.op_type;
        if (node.op_attr && node.op_attr.data) {
            const data = node.op_attr.data;
            if (data.bytes_value && data.bytes_value.value) {
                this.layout = '<';
                this.values = data.bytes_value.value;
            }
        }
    }
};

xmodel.Utility = class {

    static attribute(attr_value) {
        const key = attr_value.value;
        const type = key.replace(/_value$/, '');
        const value = attr_value[attr_value.value];
        switch (type) {
            case 'bool': return { type: 'boolean', value: value };
            case 'int32': return { type: 'int32', value: value };
            case 'int32_vec': return { type: 'int32[]', value: value.value };
            case 'int64': return { type: 'int64', value: value };
            case 'uint64': return { type: 'uint64', value: value };
            case 'float': return { type: 'float32', value: value };
            case 'float_vec': return { type: 'float32[]', value: value.value };
            case 'double': return { type: 'float64', value: value };
            case 'string': return { type: 'string', value: value };
            case 'string_vec':  return { type: 'string[]', value: value.value };
            case 'bytes': return { type: 'byte[]', value: value.value };
            case 'map_string_2_int32': return { type: 'map<string,int32>', value: value.value };
            default: throw new xmodel.Error("Unsupported attribute type '" + type + "'.");
        }
    }
};

xmodel.Metadata = class {

    constructor(op_defs) {
        this._types = new Map();
        this._attributes = new Map();
        const categories = [
            [ 'avgpool2d', 'Pool' ],
            [ 'batchnorm', 'Normalization' ],
            [ 'celu', 'Activation' ],
            [ 'concat-fix', 'Tensor' ],
            [ 'concat', 'Tensor' ],
            [ 'conv2d-fix', 'Layer' ],
            [ 'conv2d', 'Layer' ],
            [ 'depthwise-conv2d-fix', 'Layer' ],
            [ 'depthwise-conv2d', 'Layer' ],
            [ 'elu', 'Activation' ],
            [ 'fix', 'Quantization' ],
            [ 'fix2float', 'Quantization' ],
            [ 'flatten', 'Shape' ],
            [ 'float2fix', 'Quantization' ],
            [ 'gelu', 'Activation' ],
            [ 'hard-sigmoid', 'Activation' ],
            [ 'hard-sigmoid-fix', 'Activation' ],
            [ 'hard-swish', 'Activation' ],
            [ 'hard-tanh', 'Activation' ],
            [ 'identity', 'Control' ],
            [ 'inner-product', 'Layer' ],
            [ 'l2_normalize', 'Normalization' ],
            [ 'leaky-relu', 'Activation' ],
            [ 'leakyrelu', 'Activation' ],
            [ 'maxpool2d', 'Pool' ],
            [ 'pool-fix', 'Pool' ],
            [ 'relu', 'Activation' ],
            [ 'relu6', 'Activation' ],
            [ 'reshape-fix', 'Shape' ],
            [ 'reshape', 'Shape' ],
            [ 'scale', 'Layer' ],
            [ 'selu', 'Activation' ],
            [ 'shape', 'Shape' ],
            [ 'sigmoid', 'Activation' ],
            [ 'softmax', 'Activation' ],
            [ 'squeeze', 'Transform' ],
            [ 'stack', 'Tensor' ],
            [ 'strided_slice', 'Tensor' ],
            [ 'swish', 'Activation' ],
            [ 'tanh', 'Activation' ],
            [ 'threshold', 'Quantization' ],
            [ 'transpose', 'Tensor' ],
            [ 'transposed-conv2d', 'Layer' ],
            [ 'transposed-conv2d-fix', 'Layer' ],
            [ 'transposed-depthwise-conv2d', 'Layer' ],
            [ 'transposed-depthwise-conv2d-fix', 'Layer' ],
            [ 'upsample-fix', 'Data' ],
        ];
        this._types = new Map(categories.map((entry) => [ entry[0], { name: entry[0], category: entry[1] } ]));
        for (const op_def of op_defs) {
            const type = this._types.get(op_def.name) || { name: op_def.name };
            if (op_def.annotation) {
                type.description = op_def.annotation;
            }
            type.inputs = op_def.input_args.map((input_arg) => {
                const input = {};
                input.name = input_arg.name;
                if (input_arg.annotation) {
                    input.description = input_arg.annotation;
                }
                return input;
            });
            type.attributes = op_def.attrs.map((attr) => {
                const attribute = {};
                attribute.name = attr.name;
                attribute.default = xmodel.Utility.attribute(attr.default_value).value;
                if (attr.annotation) {
                    attribute.description = attr.annotation;
                }
                return attribute;
            });
            for (const attribute of type.attributes) {
                this._attributes.set(type.name + ':' + attribute.name, attribute);
            }
            this._types.set(type.name, type);
        }
    }

    type(name) {
        if (!this._types.has(name)) {
            this._types.set(name, { name: name });
        }
        return this._types.get(name);
    }

    attribute(type, name) {
        const key = type + ':' + name;
        return this._attributes.get(key);
    }
};

xmodel.Error = class extends Error {

    constructor(message) {
        super(message);
        this.name = 'Error loading xmodel.';
    }
};

if (typeof module !== 'undefined' && typeof module.exports === 'object') {
    module.exports.ModelFactory = xmodel.ModelFactory;
}
