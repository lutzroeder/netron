/* jshint esversion: 6 */

var xmodel = xmodel || {};
var protobuf = protobuf || require('./protobuf');

xmodel.ModelFactory = class {

    match(context) {
        const tags = context.tags('pb');
        if (tags.get(5) === 2) {
            return true;
        }
        return false;
    }

    open(context) {
        return context.require('./xmodel-proto').then(() => {
            try {
                xmodel.proto = protobuf.get('xmodel').serial_v2;
                const stream = context.stream;
                const reader = protobuf.BinaryReader.open(stream);
                const graph = xmodel.proto.Graph.decode(reader);
                return new xmodel.Model(graph);
            }
            catch (error) {
                const message = error && error.message ? error.message : error.toString();
                throw new xmodel.Error('File format is not serial_v2.Graph (' + message.replace(/\.$/, '') + ').');
            }
        });
    }
};

xmodel.Model = class {

    constructor(graph) {
        this._name = graph.graph_name || '';
        this._format = 'Vitis-AI xmodel';
        this._producer = graph && graph.graph_attr && graph.graph_attr.origin && graph.graph_attr.origin.string_value ? graph.graph_attr.origin.string_value : '';
        this._graphs = [ new xmodel.Graph(graph) ];
    }

    get name() {
        return this._name;
    }

    get format() {
        return this._format;
    }

    get producer() {
        return this._producer;
    }

    get graphs() {
        return this._graphs;
    }
};

xmodel.Graph = class {

    constructor(graph) {
        const metadata = new xmodel.Metadata(graph.op_defs);
        this._inputs = [];
        this._outputs = [];
        this._root_subgraph = new xmodel.Subgraph(graph.subg_root);
        this._handlers = new Map();
        const count = new Map();
        for (const op_node of graph.op_node) {
            for (const arg of op_node.args) {
                for (const arg_op of arg.arg_ops) {
                    count.set(arg_op, count.has(arg_op) ? count.get(arg_op) + 1 : 1);
                }
            }
        }
        const nodes = [];
        for (const op_node of graph.op_node) {
            nodes.push(op_node);
        }

        let tensors = new Map();
        for (const op_node of nodes) {
            tensors.set(op_node.op_name, op_node.output_tensor);
        }

        this._nodes = nodes.map((node) => new xmodel.Node(metadata, node, tensors));

        let idx = 0;
        let subgraphs = new Map();
        (function add_subgraph(parent) {
            subgraphs.set(parent.group_name, parent);
            for (const child of parent.children)
                add_subgraph(child);
        })(this._root_subgraph);

        this._subgraphs = subgraphs;
        for (const subgraph of subgraphs.values()) {
            for (const op of subgraph.ops) {
                for (let node of this._nodes)
                    if (op == node.name) {
                        node.set_group(subgraph.group_name);
                        let atttributes = new Map();
                        node.attributes.map((attr) => atttributes.set(attr.name, attr.value));
                        if (atttributes.has('device') && atttributes.get('device') == 'DPU') {
                            node.set_category('DPU' + (idx % 6).toString());
                        }
                        break;
                    }
            }
            if (subgraph.ops.length > 0) idx++;
        }
    }

    set_handler(name, handler) {
        this._handlers.set(name, handler);
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

    get root() {
        return this._root_subgraph;
    }

    get subgraphs() {
        return this._subgraphs;
    }

    get groups() {
        return true;
    }

    get xmodel() {
        return true;
    }
};

xmodel.Subgraph = class {

    constructor(subgraph, parent_name) {
        this._name = subgraph.subgraph_name;
        this._children = [];
        this._ops = subgraph.op_name.length > 0 ? subgraph.op_name : [];
        this._attributes = [];
        this._group_name = parent_name ? parent_name + '/' + this._name.replace(/\//g, "_") : this._name;
        this._show_cluster_attr = undefined;
        this._level = undefined;

        for (const child of subgraph.subg_child) {
            this._children.push(new xmodel.Subgraph(child, this._group_name));
        }

        if (!parent_name) this._level = 'ROOT Subgraph';
        for (const name of Object.keys(subgraph.subg_attr)) {
            if (name == 'reg_id_to_parameter_value')
                continue;
            if (name == 'device')
                this._level = subgraph.subg_attr[name].string_value + ' Subgraph';
            const attribute = xmodel.Utility.attribute(subgraph.subg_attr[name]);
            this._attributes.push(new xmodel.Attribute(null, name, attribute.type, attribute.value));
        }

    }

    set ShowClusterAttr(func) {
        this._show_cluster_attr = func;
    }

    get ShowClusterAttr() {
        return this._show_cluster_attr;
    }

    get attributes() {
        return this._attributes;
    }

    get children() {
        return this._children;
    }

    get ops() {
        return this._ops;
    }

    get name() {
        return this._name;
    }

    get group_name() {
        return this._group_name;
    }

    get level() {
        return this._level;
    }
};

xmodel.Parameter = class {

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

xmodel.Argument = class {

    constructor(proto_tensor) {
        if (typeof proto_tensor.tensor_name !== 'string') {
            throw new xmodel.Error("Invalid argument identifier '" + JSON.stringify(proto_tensor.tensor_name) + "'.");
        }
        this._name = proto_tensor.tensor_name;
        this._shape = proto_tensor.tensor_dim;
        this._data_type = xmodel.Utility.type2str(proto_tensor.data_type);
        this._bit_width = proto_tensor.tensor_bit_width;
        this._attributes = [];

        for (const key of Object.keys(proto_tensor.tensor_attr)) {
            if (key.startsWith('quant_in_') || key.startsWith('quant_out_'))
                continue;
            const attribute = xmodel.Utility.attribute(proto_tensor.tensor_attr[key]);
            this._attributes.push(new xmodel.Attribute(undefined, key, attribute.type, attribute.value));
        }
    }

    get name() {
        return this._name;
    }
    get shape() {
        return this._shape;
    }
    get data_type() {
        return this._data_type;
    }
    get bit_width() {
        return this._bit_width;
    }
    get attributes() {
        return this._attributes;
    }
    get xmodel() {
        return true;
    }
};

xmodel.Node = class {

    constructor(metadata, op_node, tensors) {
        this._type = op_node.op_type;
        this._name = op_node.op_name || '';
        this._metadata = Object.assign({}, metadata.type(this._type));
        this._inputs = [];
        this._outputs = [];
        this._attributes = [];
        this._group = '';

        for (const key of Object.keys(op_node.op_attr)) {
            if (key.startsWith('quant_in_') || key.startsWith('quant_out_'))
                continue;
            if (key == 'data' && Object.keys(op_node.op_attr).includes('data_type')) {
                let data_type = op_node.op_attr['data_type'].string_value;
                let data =
                    xmodel.Utility.transform((new Uint8Array(op_node.op_attr[key].bytes_value.value)).buffer, data_type);
                this._attributes.push(new xmodel.Attribute(undefined, key, op_node.op_attr['data'].value.replace(/_value$/, ''), data));
            } else {
                const attribute = xmodel.Utility.attribute(op_node.op_attr[key]);
                this._attributes.push(new xmodel.Attribute(metadata.attribute(this._type, key), key, attribute.type, attribute.value));
            }
        }

        for (const arg of op_node.args) {
            const args = arg.arg_ops.map((arg_op) => new xmodel.Argument(tensors.get(arg_op)));
            if (args.length > 0)
                this._inputs.push(new xmodel.Parameter(arg.arg_name, args));
        }
        const args = new xmodel.Argument(tensors.get(op_node.op_name));
        if (args) this._outputs.push(new xmodel.Parameter('output', [args]));
    }

    set_group(group) {
        this._group = group;
    }

    set_category(category) {
        this._metadata.category = category;
    }

    get type() {
        return this._type;
    }

    get metadata() {
        return this._metadata;
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

    get attributes() {
        return this._attributes;
    }

    get group() {
        return this._group;
    }
};

xmodel.Attribute = class {

    constructor(metadata, name, type, value) {
        this._name = name;
        this._type = type;
        this._value = value;
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

xmodel.TensorType = class {

    constructor(dataType, shape) {
        this._dataType = dataType || '?';
        this._shape = shape;
    }

    get dataType() {
        return this._dataType;
    }

    get shape() {
        return this._shape;
    }

    toString() {
        return (this.dataType || '?') + this._shape.toString();
    }
};

xmodel.TensorShape = class {

    constructor(dimensions) {
        this._dimensions = Array.from(dimensions);
    }

    get dimensions() {
        return this._dimensions;
    }

    toString() {
        if (!this._dimensions || this._dimensions.length == 0) {
            return '';
        }
        return '[' + this._dimensions.map((dimension) => dimension.toString()).join(',') + ']';
    }
};

xmodel.Tensor = class {

    constructor(type, kind) {
        this._type = type;
        this._kind = kind;
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
        return JSON.stringify(value, null, 4);
    }

    _context() {
        const context = {};
        context.index = 0;
        context.count = 0;
        context.state = 'Tensor data not implemented.';
        return context;
    }

    _decode(/* context, dimension */) {
        return [];
    }
};

xmodel.Utility = class {

    static attribute(attr) {
        const key = attr.value;
        const value = {
            value: attr[key],
            type: key.replace(/_value$/, '')
        };
        let map_type = ['bool_vec',
            'int32_vec',
            'uint32_vec',
            'int64_vec',
            'uint64_vec',
            'float_vec',
            'double_vec',
            'string_vec',
            'bytes_vec',
            'map_string_2_int32',
            'map_string_2_uint32',
            'map_string_2_int64',
            'map_string_2_uint64',
            'map_string_2_string',
            'map_string_2_bytes',
            'map_string_2_int32_vec',
            'map_string_2_uint32_vec',
            'map_string_2_int64_vec',
            'map_string_2_uint64_vec',
            'map_string_2_string_vec'];
        if (map_type.includes(value.type)) {
            value.value = value.value.value;
        }
        switch (value.type) {
            case 'bool': {
                value.type = 'boolean';
                break;
            }
        }
        return value;
    }

    static transform(buffer, dtype) {
        let value = [];
        let idx = 0;
        let max_num = 512;
        switch (dtype.toUpperCase()) {
            case "XINT8":
                let i8 = new Int8Array(buffer);
                while (idx < i8.length && idx < max_num)
                    value.push(i8[idx++]);
                break;
            case "INT32":
                let i32 = new Int32Array(buffer);
                while (idx < i32.length && idx < max_num)
                    value.push(i32[idx++]);
                break;
            case "INT64":
                let i64 = new BigInt64Array(buffer);
                while (idx < i64.length && idx < max_num)
                    value.push(i64[idx++]);
                break;
            case "FLOAT32":
                let f32 = new Float32Array(buffer);
                while (idx < f32.length && idx < max_num)
                    value.push(f32[idx++]);
                break;
        }
        return value;
    }

    static type2str(index) {
        switch (index) {
            case 0:
                return "INT";
            case 1:
                return "UINT";
            case 2:
                return "XINT";
            case 3:
                return "XUINT";
            case 4:
                return "FLOAT";
            default:
                return "UNKNOWN";
        }
    }

    static quantization(attr) {
        const quant = { in: {}, out: {} };
        for (const name of Object.keys(attr)) {
            const attribute = xmodel.Utility.attribute(attr[name]);
            switch (name) {
                case 'quant_in_bit_width':
                    quant.in.bit_width = attribute.value;
                    break;
                case 'quant_in_quantize_pos':
                    quant.in.pos = attribute.value;
                    break;
                case 'quant_in_signed':
                    quant.in.signed = attribute.value;
                    break;
                case 'quant_in_round_mode':
                    quant.in.round_mode = attribute.value;
                    break;
                case 'quant_out_bit_width':
                    quant.out.bit_width = attribute.value;
                    break;
                case 'quant_out_quantize_pos':
                    quant.out.pos = attribute.value;
                    break;
                case 'quant_out_signed':
                    quant.out.signed = attribute.value;
                    break;
                case 'quant_out_round_mode':
                    quant.out.round_mode = attribute.value;
                    break;
            }
        }
        return quant;
    }
};

xmodel.Metadata = class {

    constructor(op_defs) {
        this._map = new Map();
        this._attributeCache = new Map();
        const categories = new Map([
            [ 'conv2d', 'Layer' ],
            [ 'depthwise-conv2d', 'Layer' ],
            [ 'transposed-conv2d', 'Layer' ],
            [ 'transposed-depthwise-conv2d', 'Layer' ],
            [ 'inner-product', 'Layer' ],
            [ 'matmul', 'Layer' ],
            [ 'scale', 'Layer' ],
            [ 'maxpool2d', 'Pool' ],
            [ 'avgpool2d', 'Pool' ],
            [ 'relu', 'Activation' ],
            [ 'leaky-relu', 'Activation' ],
            [ 'relu6', 'Activation' ],
            [ 'elu', 'Activation' ],
            [ 'celu', 'Activation' ],
            [ 'gelu', 'Activation' ],
            [ 'selu', 'Activation' ],
            [ 'sigmoid', 'Activation' ],
            [ 'swish', 'Activation' ],
            [ 'tanh', 'Activation' ],
            [ 'hard-sigmoid', 'Activation' ],
            [ 'hard-swish', 'Activation' ],
            [ 'hard-tanh', 'Activation' ],
            [ 'fix', 'Quantization' ],
            [ 'fix2float', 'Quantization' ],
            [ 'float2fix', 'Quantization' ],
            [ 'threshold', 'Quantization' ],
            [ 'batchnorm', 'Normalization' ],
            [ 'l2_normalize', 'Normalization' ],
            [ 'concat', 'Tensor' ],
            [ 'identity', 'Tensor' ],
            [ 'upload', 'Tensor' ],
            [ 'download', 'Tensor' ],
            [ 'placeholder', 'Tensor' ],
            [ 'squeeze', 'Tensor' ],
            [ 'transpose', 'Tensor' ],
            [ 'flatten', 'Tensor' ],
            [ 'placeholder', 'Tensor' ],
            [ 'strided_slice', 'Tensor' ],
            [ 'stack', 'Tensor' ],
            [ 'gstiling', 'Tensor' ],
            [ 'reshape', 'Shape' ],
            [ 'shape', 'Shape' ]
        ]);
        for (const op_def of op_defs) {
            const name = op_def.name;
            const schema = {};
            schema.name = name;
            if (op_def.annotation) {
                schema.description = op_def.annotation;
            }
            schema.inputs = op_def.input_args.map((input_arg) => {
                const input = {};
                input.name = input_arg.name;
                if (input_arg.annotation) {
                    input.description = input_arg.annotation;
                }
                return input_arg;
            });
            schema.attributes = op_def.attrs.map((attr) => {
                const attribute = {};
                attribute.name = attr.name;
                const value = xmodel.Utility.attribute(attr.default_value);
                attribute.default = value.value;
                if (attr.annotation) {
                    attr.description = attr.annotation;
                }
                this._attributeCache.set(name + ':' + attr.name, attribute);
                return attr;
            });
            if (categories.has(name)) {
                schema.category = categories.get(name);
            }
            this._map.set(name, schema);
        }
    }

    type(name) {
        return this._map.get(name);
    }

    attribute(type, name) {
        const key = type + ':' + name;
        return this._attributeCache.get(key);
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
