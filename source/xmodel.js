/* jshint esversion: 6 */

var xmodel = xmodel || {};
var protobuf = protobuf || require('./protobuf');

xmodel.ModelFactory = class {

    match(context) {
        const tags = context.tags('pb');
        if (tags.get(5) === 2) {
            return 'xmodel.pb';
        }
        return undefined;
    }

    open(context) {
        return context.require('./xmodel-proto').then(() => {
            let graph = null;
            try {
                xmodel.proto = protobuf.get('xmodel').serial_v2;
                const stream = context.stream;
                const reader = protobuf.BinaryReader.open(stream);
                graph = xmodel.proto.Graph.decode(reader);
            }
            catch (error) {
                const message = error && error.message ? error.message : error.toString();
                throw new xmodel.Error('File format is not serial_v2.Graph (' + message.replace(/\.$/, '') + ').');
            }
            return new xmodel.Model(graph);
        });
    }
};

xmodel.Model = class {

    constructor(graph) {
        this._name = graph.graph_name || '';
        this._format = 'xmodel';
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
        const count = new Map();
        for (const op_node of graph.op_node) {
            for (const arg of op_node.args) {
                for (const arg_op of arg.arg_ops) {
                    count.set(arg_op, count.has(arg_op) ? count.get(arg_op) + 1 : 1);
                }
            }
        }
        const initializers = new Map();
        const nodes = [];
        for (const op_node of graph.op_node) {
            if (op_node.op_type === 'const-fix' && op_node.args.length === 0 && count.get(op_node.op_name) === 1) {
                const type = xmodel.Utility.type(op_node.op_attr);
                initializers.set(op_node.op_name, new xmodel.Tensor(type, op_node.op_type));
                continue;
            }
            if (op_node.op_type === 'data-fix' && op_node.args.length === 0) {
                const type = xmodel.Utility.type(op_node.op_attr);
                const quantization = xmodel.Utility.quantization(op_node.op_attr);
                this._inputs.push(new xmodel.Parameter(op_node.op_name, [
                    new xmodel.Argument(op_node.op_name, type, quantization.out, null)
                ]));
                continue;
            }
            nodes.push(op_node);
        }
        this._nodes = nodes.map((node) => new xmodel.Node(metadata, node, initializers));
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

    constructor(name, type, quantization, initializer) {
        if (typeof name !== 'string') {
            throw new xmodel.Error("Invalid argument identifier '" + JSON.stringify(name) + "'.");
        }
        this._name = name;
        this._type = type || null;
        this._quantization = quantization;
        this._initializer = initializer || null;
    }

    get name() {
        return this._name;
    }

    get type() {
        if (this._initializer) {
            return this._initializer.type;
        }
        return this._type;
    }

    get quantization() {
        if (this._quantization) {
            const list = [];
            if (this._quantization.bit_width !== undefined) {
                list.push('bit_width: ' + this._quantization.bit_width);
            }
            if (this._quantization.pos !== undefined) {
                list.push('pos: ' + this._quantization.pos);
            }
            if (this._quantization.signed !== undefined) {
                list.push('signed: ' + this._quantization.signed);
            }
            if (this._quantization.round_mode !== undefined) {
                list.push('round_mode: ' + this._quantization.round_mode);
            }
            return list.join(', ');
        }
        return null;
    }

    get initializer() {
        return this._initializer;
    }
};

xmodel.Node = class {

    constructor(metadata, op_node, initializers) {
        this._name = op_node.op_name || '';
        this._type = metadata.type(op_node.op_type) || { name: op_node.op_type };
        this._inputs = [];
        this._outputs = [];
        this._attributes = [];
        for (const name of Object.keys(op_node.op_attr)) {
            if (!name.startsWith('quant_in_') && !name.startsWith('quant_out_') && name !== 'workload') {
                const attribute = xmodel.Utility.attribute(op_node.op_attr[name]);
                this._attributes.push(new xmodel.Attribute(metadata.attribute(this._type, name), name, attribute.type, attribute.value));
            }
        }
        const quantization = xmodel.Utility.quantization(op_node.op_attr);
        for (const arg of op_node.args) {
            const args = arg.arg_ops.map((arg_op) => new xmodel.Argument(arg_op, null, quantization.in, initializers.get(arg_op)));
            this._inputs.push(new xmodel.Parameter(arg.arg_name, args));
        }
        this._outputs.push(new xmodel.Parameter('output', [
            new xmodel.Argument(op_node.op_name, null, quantization.out, null)
        ]));
    }

    get type() {
        return this._type;
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
        switch (value.type) {
            case 'bool': {
                value.type = 'boolean';
                break;
            }
            case 'int32_vec': {
                value.type = 'int32[]';
                value.value = value.value.value;
                break;
            }
        }
        return value;
    }

    static type(attr) {
        let dataType = '?';
        const data_type = attr.data_type.string_value;
        switch (data_type) {
            case 'XINT4': dataType = 'int4'; break;
            case 'XINT8': dataType = 'int8'; break;
            default: throw new xmodel.Error("Unknown data_type '" + data_type + "'.");
        }
        const shape = attr.shape.int32_vec_value.value;
        return new xmodel.TensorType(dataType, new xmodel.TensorShape(shape));
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
            [ 'hard-swish', 'Activation' ],
            [ 'hard-tanh', 'Activation' ],
            [ 'identity', 'Control' ],
            [ 'inner-product', 'Layer' ],
            [ 'l2_normalize', 'Normalization' ],
            [ 'leaky-relu', 'Activation' ],
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
            [ 'transposed-depthwise-conv2d', 'Layer' ],
            [ 'upsample-fix', 'Data' ],
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
                return input;
            });
            schema.attributes = op_def.attrs.map((attr) => {
                const attribute = {};
                attribute.name = attr.name;
                const value = xmodel.Utility.attribute(attr.default_value);
                attribute.default = value.value;
                if (attr.annotation) {
                    attribute.description = attr.annotation;
                }
                this._attributeCache.set(name + ':' + attr.name, attribute);
                return attribute;
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
