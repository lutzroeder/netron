/* jshint esversion: 6 */

var xmodel = xmodel || {};
var protobuf = protobuf || require('./protobuf');

xmodel.ModelFactory = class {

    match(context) {
        const tags = context.tags('pb');
        if (tags.get(101) === 2) {
            return true;
        }
        return false;
    }

    open(context) {
        return context.require('./xmodel-proto').then(() => {
            let graph = null;
            try {
                xmodel.proto = protobuf.get('xmodel').serial_v2;
                const reader = protobuf.Reader.create(context.stream.peek());
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
        const metadata = new xmodel.GraphMetadata(graph.op_defs);
        this._inputs = [];
        this._outputs = [];
        this._nodes = [];
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
                // const data_type = op_node.op_attr.data_type.string_value;
                // const shape = op_node.op_attr.shape.int32_vec_value.value
                initializers.set(op_node.op_name, op_node);
                continue;
            }
            nodes.push(op_node);
        }
        for (const op_node of nodes) {
            this._nodes.push(new xmodel.Node(metadata, op_node));
        }
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
            return 'bitwidth: ' + this._quantization.bit_width + ', pos: ' + this._quantization.pos + ', signed: ' + this._quantization.signed;
        }
        return null;
    }

    get initializer() {
        return this._initializer;
    }
};

xmodel.Node = class {

    constructor(metadata, op_node) {
        this._type = op_node.op_type;
        this._name = op_node.op_name || '';
        this._metadata = metadata.type(this._type);
        this._inputs = [];
        this._outputs = [];
        this._attributes = [];

        const in_quant = {};
        const out_quant = {};
        for (const name of Object.keys(op_node.op_attr)) {
            const attribute = xmodel.Utility.attribute(op_node.op_attr[name]);
            switch (name) {
                case 'quant_in_bit_width':
                    in_quant.bit_width = attribute.value;
                    continue;
                case 'quant_in_quantize_pos':
                    in_quant.pos = attribute.value;
                    continue;
                case 'quant_in_signed':
                    in_quant.signed = attribute.value;
                    continue;
                case 'quant_out_bit_width':
                    out_quant.bit_width = attribute.value;
                    continue;
                case 'quant_out_quantize_pos':
                    out_quant.pos = attribute.value;
                    continue;
                case 'quant_out_signed':
                    out_quant.signed = attribute.value;
                    continue;
            }
            this._attributes.push(new xmodel.Attribute(metadata.attribute(this._type, name), name, attribute.type, attribute.value));
        }

        for (const arg of op_node.args) {
            const args = arg.arg_ops.map((arg_op) => new xmodel.Argument(arg_op, null, in_quant, null));
            this._inputs.push(new xmodel.Parameter(arg.arg_name, args));
        }
        this._outputs.push(new xmodel.Parameter('output', [
            new xmodel.Argument(op_node.op_name, null, out_quant, null)
        ]));
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
};

xmodel.GraphMetadata = class {

    constructor(op_defs) {
        this._map = new Map();
        this._attributeCache = new Map();
        const categories = new Map([
            [ 'conv2d-fix', 'Layer' ],
            [ 'upsample-fix', 'Layer' ],
            [ 'pool-fix', 'Pool' ],
            [ 'batchnorm', 'Normalization' ],
            [ 'concat-fix', 'Tensor' ],
            [ 'reshape-fix', 'Shape' ],
            [ 'softmax', 'Activation' ]
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
