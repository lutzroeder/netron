
const kann = {};

kann.ModelFactory = class {

    match(context) {
        const reader = context.peek('flatbuffers.binary');
        if (reader && reader.identifier === 'KaNN') {
            context.type = 'kann.flatbuffers';
            context.target = reader;
        }
    }

    async open(context) {
        kann.schema = await context.require('./kann-schema');
        kann.schema = kann.schema.kann;
        let model = null;
        switch (context.type) {
            case 'kann.flatbuffers': {
                try {
                    const reader = context.target;
                    model = kann.schema.Model.create(reader);
                } catch (error) {
                    const message = error && error.message ? error.message : error.toString();
                    throw new kann.Error(`File format is not kann.Model (${message.replace(/\.$/, '')}).`);
                }
                break;
            }
            default: {
                throw new kann.Error(`Unsupported KaNN format '${context.type}'.`);
            }
        }
        const metadata = await context.metadata('kann-metadata.json');
        return new kann.Model(metadata, model, context.identifier);
    }
};

kann.Model = class {

    constructor(metadata, model, identifier) {
        this.format = 'KaNN';
        this.name = identifier;
        this.graphs = model.graph.map((graph) => new kann.Graph(metadata, graph));
    }
};

kann.Graph = class {

    constructor(metadata, graph) {
        const arcs = new Map();
        for (const arc of graph.arcs) {
            arcs.set(arc.name, new kann.Value(arc.name, null, null));
        }
        this.nodes = graph.nodes.map((node) => new kann.Node(metadata, node, arcs));
        this.inputs = graph.inputs.map((input) => new kann.Argument(input, [arcs.get(input)]));
        this.outputs = graph.outputs.map((output) => new kann.Argument(output, [arcs.get(output)]));
    }
};

kann.Node = class {

    constructor(metadata, node, arcs) {
        this.type = metadata.type(node.type);
        this.name = node.name;
        this.inputs = [];
        this.outputs = [];
        this.attributes = [];
        const extractData = (value) => {
            switch (value.type) {
                case 'int': case 'int8': case 'int16': case 'int32': case 'int64': return value.value_int;
                case 'uint': case 'uint8': case 'uint16': case 'uint32': case 'uint64': return value.value_uint;
                case 'float': case 'float16': case 'float32': case 'float64': return value.value_float;
                case 'string': return value.value_string;
                case 'int[]': case 'int8[]': case 'int16[]': case 'int32[]': case 'int64[]': return Array.from(value.list_int);
                case 'uint[]': case 'uint8[]': case 'uint16[]': case 'uint32[]': case 'uint64[]': return Array.from(value.list_uint);
                case 'float[]': case 'float16[]': case 'float32[]': case 'float64[]': return Array.from(value.list_float);
                case 'string[]': return Array.from(value.list_string);
                default: throw new kann.Error(`Unsupported data type '${value.type}'.`);
            }
        };
        const getAttributeValue = (attribute) => {
            if (attribute.type === 'attributes') {
                const obj = {};
                for (const attr of attribute.attributes) {
                    obj[attr.name] = getAttributeValue(attr);
                }
                return obj;
            }
            if (attribute.value !== null) {
                return extractData(attribute.value);
            }
            throw new kann.Error(`${attribute.name} doesn't have a value.`);
        };
        if (Array.isArray(node.attributes) && node.attributes.length > 0) {
            for (const attr of node.attributes) {
                let value = attr.type ? getAttributeValue(attr) : attr;
                value = Array.isArray(value) ? value : [value];
                const type = value.type === 'attributes' ? null : attr.type || null;
                const attribute = new kann.Argument(attr.name, value, type);
                this.attributes.push(attribute);
            }
        }
        if (Array.isArray(node.inputs) && node.inputs.length > 0) {
            const name = node.inputs.length > 1 ? 'inputs' : 'input';
            const argument = new kann.Argument(name, node.inputs.map((input) => arcs.get(input)));
            this.inputs.push(argument);
        }
        if (Array.isArray(node.outputs) && node.outputs.length > 0) {
            const name = node.outputs.length > 1 ? 'outputs' : 'output';
            const argument = new kann.Argument(name, node.outputs.map((output) => arcs.get(output)));
            this.outputs.push(argument);
        }
        if (Array.isArray(node.params) && node.params.length > 0) {
            for (const param of node.params) {
                const type = new kann.TensorType(param.type, param.shape);
                const data = null;
                const quantization = null;
                const tensor = new kann.Tensor(param.name, type, data, quantization);
                const value = new kann.Value('', type, tensor);
                const argument = new kann.Argument(param.name, [value]);
                this.inputs.push(argument);
            }
        }
        if (node.relu) {
            const relu = { type: 'ReLU', name: `${node.name}/relu`, params: [] };
            this.chain = [new kann.Node(metadata, relu, arcs)];
        }
    }
};

kann.Argument = class {

    constructor(name, value, type) {
        this.name = name;
        this.value = value;
        this.type = type;
    }
};

kann.Value = class {

    constructor(name, type, initializer) {
        this.name = name;
        this.type = type;
        this.initializer = initializer;
        this.quantization = initializer && initializer.quantization ? initializer.quantization : null;
    }
};

kann.Tensor = class {

    constructor(name, type, values, quantization) {
        this.name = name;
        this.type = type;
        this.encoding = Array.isArray(values) ? '|' : '<';
        this.values = values;
        this.quantization = quantization ? quantization : null;
    }
};

kann.TensorType = class {

    constructor(dataType, shape) {
        this.dataType = dataType || '?';
        this.shape = new kann.TensorShape(shape);
    }

    toString() {
        return this.dataType + this.shape.toString();
    }
};

kann.TensorShape = class {

    constructor(dimensions) {
        this.dimensions = Array.from(dimensions);
    }

    toString() {
        if (Array.isArray(this.dimensions) && this.dimensions.length > 0) {
            return `[${this.dimensions.map((dimension) => dimension.toString()).join(',')}]`;
        }
        return '';
    }
};

kann.Error = class extends Error {

    constructor(message) {
        super(message);
        this.name = 'Error loading KaNN model.';
    }
};

export const ModelFactory = kann.ModelFactory;
