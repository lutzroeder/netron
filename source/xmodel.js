
const xmodel = {};

xmodel.ModelFactory = class {

    match(context) {
        const tags = context.tags('pb');
        if (tags.get(5) === 2) {
            context.type = 'xmodel.pb';
        }
    }

    async open(context) {
        xmodel.proto = await context.require('./xmodel-proto');
        xmodel.proto = xmodel.proto.serial_v2;
        let graph = null;
        try {
            const reader = context.read('protobuf.binary');
            graph = xmodel.proto.Graph.decode(reader);
        } catch (error) {
            const message = error && error.message ? error.message : error.toString();
            throw new xmodel.Error(`File format is not serial_v2.Graph (${message.replace(/\.$/, '')}).`);
        }
        return new xmodel.Model(graph);
    }
};

xmodel.Model = class {

    constructor(graph) {
        this.name = graph.graph_name || '';
        this.format = 'xmodel';
        this.producer = graph && graph.graph_attr && graph.graph_attr.origin && graph.graph_attr.origin.string_value ? graph.graph_attr.origin.string_value : '';
        this.graphs = [new xmodel.Graph(graph)];
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
        const values = new Map();
        values.map = (name, node, initializer) => {
            if (!values.has(name)) {
                values.set(name, new xmodel.Value(name, node, initializer));
            }
            return values.get(name);
        };
        const nodes = [];
        for (const node of graph.op_node) {
            if (node.args.length === 0) {
                if (node.op_type === 'data' || node.op_type === 'data-fix') {
                    const value = values.map(node.op_name, node);
                    this.inputs.push(new xmodel.Argument(node.op_name, [value]));
                    continue;
                }
            }
            if (node.args.length === 0 && counts.get(node.op_name) === 1) {
                if (node.op_type === 'const' || node.op_type === 'const-fix') {
                    values.map(node.op_name, node, true);
                    continue;
                }
            }
            values.map(node.op_name, node);
            nodes.push(node);
        }
        this.nodes = nodes.map((node) => new xmodel.Node(metadata, node, values));
    }
};

xmodel.Argument = class {

    constructor(name, value, type, visible) {
        this.name = name;
        this.value = value;
        this.type = type || null;
        this.visible = visible !== false;
    }
};

xmodel.Value = class {

    constructor(name, node, initializer) {
        if (typeof name !== 'string') {
            throw new xmodel.Error(`Invalid value identifier '${JSON.stringify(name)}'.`);
        }
        this.name = name;
        if (node) {
            const tensor = node.output_tensor;
            if (tensor && tensor.tensor_attr && tensor.data_type) {
                if (initializer) {
                    this.initializer = new xmodel.Tensor(node);
                    this.type = this.initializer.type;
                } else {
                    this.type = new xmodel.TensorType(tensor);
                }
            }
        }
    }
};

xmodel.Node = class {

    constructor(metadata, op_node, values) {
        this.name = op_node.op_name || '';
        this.type = metadata.type(op_node.op_type);
        this.inputs = [];
        this.outputs = [];
        this.attributes = [];
        this.chain = [];
        if (op_node.op_attr) {
            for (const [name, obj] of Object.entries(op_node.op_attr)) {
                if (name === 'device') {
                    this.device = obj.string_value;
                } else if (name !== 'workload' && !name.startsWith('quant_in_') && !name.startsWith('quant_out_')) {
                    const attr = xmodel.Utility.attribute(obj);
                    if (name === 'nonlinear' && attr.value && attr.value !== 'NONE' && attr.value !== 0) {
                        let activation = attr.value;
                        if (typeof activation === 'string') {
                            activation = activation.toLowerCase();
                        } else if (Number.isInteger(activation) && activation < 5) {
                            activation = ['none', 'relu', 'prelu', 'leakyrelu', 'relu6'][activation];
                        } else {
                            activation = JSON.stringify(activation);
                        }
                        const node = new xmodel.Node(metadata, { op_type: activation }, values);
                        this.chain.push(node);
                    } else {
                        const schema = metadata.attribute(this.type.name, name);
                        const visible = (schema && schema.default !== undefined && schema.default === attr.value) ||
                            (schema && Array.isArray(schema.default) && Array.isArray(this.value) && schema.default.length === attr.value.length && schema.default.every((value, index) => value === attr.value[index])) ? false : true;
                        const attribute = new xmodel.Argument(name, attr.value, attr.type, visible);
                        this.attributes.push(attribute);
                    }
                }
            }
        }
        if (op_node.args) {
            for (const input of op_node.args) {
                const argument = new xmodel.Argument(input.arg_name, input.arg_ops.map((arg_op) => values.map(arg_op)));
                this.inputs.push(argument);
            }
        }
        if (op_node.op_name) {
            const argument = new xmodel.Argument('output', [values.map(op_node.op_name)]);
            this.outputs.push(argument);
        }
    }
};

xmodel.TensorType = class {

    constructor(tensor) {
        let type = '';
        switch (tensor.data_type) {
            case 0: type = 'int'; break;
            case 1: type = 'uint'; break;
            case 2: type = 'xint'; break;
            case 3: type = 'xuint'; break;
            case 4: type = 'float'; break;
            case 5: type = 'bfloat'; break;
            default: throw new xmodel.Error(`Unsupported data type '${tensor.data_type}'.`);
        }
        this.dataType = type + tensor.tensor_bit_width.toString();
        this.shape = new xmodel.TensorShape(tensor.tensor_dim);
        if (tensor.tensor_attr) {
            const attr = {};
            for (const [key, obj] of Object.entries(tensor.tensor_attr)) {
                const value = obj[obj.value];
                if (key.startsWith('quant_')) {
                    continue;
                }
                attr[key] = value;
                const denotation = [];
                if (attr.fix_point !== undefined) {
                    denotation.push(`${attr.fix_point}.`);
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
        if (!this.dimensions || this.dimensions.length === 0) {
            return '';
        }
        return `[${this.dimensions.map((dimension) => dimension.toString()).join(',')}]`;
    }
};

xmodel.Tensor = class {

    constructor(node) {
        this.type = new xmodel.TensorType(node.output_tensor);
        this.category = node.op_type;
        if (node.op_attr && node.op_attr.data) {
            const data = node.op_attr.data;
            if (data.bytes_value && data.bytes_value.value) {
                this.encoding = '<';
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
            case 'bool': return { type: 'boolean', value };
            case 'int32': return { type: 'int32', value };
            case 'int32_vec': return { type: 'int32[]', value: value.value };
            case 'uint32': return { type: 'uint32', value };
            case 'uint32_vec': return { type: 'uint32[]', value: value.value };
            case 'int64': return { type: 'int64', value };
            case 'uint64': return { type: 'uint64', value };
            case 'float': return { type: 'float32', value };
            case 'float_vec': return { type: 'float32[]', value: value.value };
            case 'double': return { type: 'float64', value };
            case 'double_vec': return { type: 'float64[]', value };
            case 'string': return { type: 'string', value };
            case 'string_vec':  return { type: 'string[]', value: value.value };
            case 'bytes': return { type: 'byte[]', value: value.value };
            case 'map_string_2_int32': return { type: 'map<string,int32>', value: value.value };
            default: throw new xmodel.Error(`Unsupported attribute type '${type}'.`);
        }
    }
};

xmodel.Metadata = class {

    constructor(op_defs) {
        this._types = new Map();
        this._attributes = new Map();
        const categories = [
            ['avgpool2d', 'Pool'],
            ['batchnorm', 'Normalization'],
            ['celu', 'Activation'],
            ['concat-fix', 'Tensor'],
            ['concat', 'Tensor'],
            ['conv2d-fix', 'Layer'],
            ['conv2d', 'Layer'],
            ['depthwise-conv2d-fix', 'Layer'],
            ['depthwise-conv2d', 'Layer'],
            ['elu', 'Activation'],
            ['fix', 'Quantization'],
            ['fix2float', 'Quantization'],
            ['flatten', 'Shape'],
            ['float2fix', 'Quantization'],
            ['gelu', 'Activation'],
            ['hard-sigmoid', 'Activation'],
            ['hard-sigmoid-fix', 'Activation'],
            ['hard-swish', 'Activation'],
            ['hard-tanh', 'Activation'],
            ['identity', 'Control'],
            ['inner-product', 'Layer'],
            ['l2_normalize', 'Normalization'],
            ['leaky-relu', 'Activation'],
            ['leakyrelu', 'Activation'],
            ['maxpool2d', 'Pool'],
            ['pool-fix', 'Pool'],
            ['relu', 'Activation'],
            ['relu6', 'Activation'],
            ['reshape-fix', 'Shape'],
            ['reshape', 'Shape'],
            ['scale', 'Layer'],
            ['selu', 'Activation'],
            ['shape', 'Shape'],
            ['sigmoid', 'Activation'],
            ['softmax', 'Activation'],
            ['squeeze', 'Transform'],
            ['stack', 'Tensor'],
            ['strided_slice', 'Tensor'],
            ['swish', 'Activation'],
            ['tanh', 'Activation'],
            ['threshold', 'Quantization'],
            ['transpose', 'Tensor'],
            ['transposed-conv2d', 'Layer'],
            ['transposed-conv2d-fix', 'Layer'],
            ['transposed-depthwise-conv2d', 'Layer'],
            ['transposed-depthwise-conv2d-fix', 'Layer'],
            ['upsample-fix', 'Data'],
        ];
        this._types = new Map(categories.map(([name, category]) => [name, { name, category }]));
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
                this._attributes.set(`${type.name}:${attribute.name}`, attribute);
            }
            this._types.set(type.name, type);
        }
    }

    type(name) {
        if (!this._types.has(name)) {
            this._types.set(name, { name });
        }
        return this._types.get(name);
    }

    attribute(type, name) {
        const key = `${type}:${name}`;
        return this._attributes.get(key);
    }
};

xmodel.Error = class extends Error {

    constructor(message) {
        super(message);
        this.name = 'Error loading xmodel.';
    }
};

export const ModelFactory = xmodel.ModelFactory;

