import * as flatbuffers from './flatbuffers.js';

const kann = {};

kann.ModelFactory = class {

    // Check whether a KaNN file is processed and initialize the container
    match(context) {
        // context provide necessary information to interact with the processed file
        const reader = context.peek('flatbuffers.binary');
        // reader is the graph following FlatBuffers schema
        if (reader && reader.identifier === 'KaNN') {
            context.type = 'kann.flatbuffers';
            context.target = reader;
            return;
        }
    }

    // Open and initialize the KaNN model from the context
    async open(context) {
        // Load the schema required for decoding KaNN model
        kann.schema = await context.require('./kann-schema');
        kann.schema = kann.schema.kann;
        let model = null;
        switch (context.type) {
            case 'kann.flatbuffers': {
                try {
                    // Attempt to create the model using the FlatBuffers reader
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
        // Load metadata related to the KaNN model
        const metadata = await context.metadata('kann-metadata.json');
        // Create and return a new KaNN model instance
        return new kann.Model(metadata, model, context.identifier);
    }
};

kann.Model = class {

    constructor(metadata, model, identifier) {
        this.format = model.format;
        // Attempt to extract information from the identifier with a regex
        const regex = /^KaNNv(\d+\.\d+\.\d+)-(.+?)(?:__(.*?))?(_)?\.kann$/;
        const match = identifier.match(regex);
        if (match) {
            [, this.version, this.name, this.description = ""] = match;
        } else {
            this.name = identifier;
        }
        // Following FlatBuffers schema a new KaNN graph is created
        this.graphs = model.graph.map((graph) => new kann.Graph(metadata, graph));
    }
};

kann.Graph = class {

    constructor(metadata, graph) {
        const arcs = new Map();
        graph.arcs.forEach((arc) => {
            const shape = arc.attributes.map((attr) =>
                `${attr.name}: ${kann.Data.extractData(attr.value).join(', ')}`
            );
            const type = new kann.TensorType(arc.type, shape);
            arcs.set(arc.name, new kann.Value(arc.name, type, null));
        });
        this.nodes = graph.nodes.map((node) => new kann.Node(metadata, node, arcs));
        this.inputs = graph.inputs.map((input) => new kann.Argument(input, arcs.get(input)));
        this.outputs = graph.outputs.map((output) => new kann.Argument(output, arcs.get(output)));
    }
};

kann.Node = class {

    constructor(metadata, node, arcs) {
        this.type = metadata.type(node.type);
        this.name = node.name;
        if (node.attributes.length > 0) {
            this.attributes = node.attributes.map((attr) => new kann.Argument(attr.name, attr, attr.type));
        }
        if (node.tensor) {
            this.attributes = typeof this.attributes === 'undefined' ? [] : this.attributes;
            const type = new kann.TensorType(node.tensor.type, node.tensor.shape);
            this.description = type.toString();
        }
        this.inputs = [];
        if (node.inputs.length > 0) {
            this.inputs = [new kann.Argument(node.inputs.length > 1 ? "inputs" : "input", node.inputs.map((input) => arcs.get(input)))];
        }
        this.outputs = [];
        if (node.outputs.length > 0) {
            this.outputs = [new kann.Argument(node.outputs.length > 1 ? "outputs" : "output", node.outputs.map((output) => arcs.get(output)))];
        }
        if (node.params.length > 0) {
            this.inputs = typeof this.inputs === 'undefined' ? [] : this.inputs;
            node.params.forEach((param) => {
                const type = new kann.TensorType(param.type, param.shape);
                let quantization = null;
                if (param.scale && param.zero_point) {
                    const scale = Array.from(kann.Data.extractData(param.scale));
                    const zeroPoint = Array.from(kann.Data.extractData(param.zero_point));
                    const paramsMap = {};
                    for (let i = 0; i < scale.length; i++) {
                        let scaleType = param.scale.type;
                        if (scaleType.endsWith('[]')) {
                            scaleType = scaleType.slice(0, -2);
                        }
                        let zeroPointType = param.zero_point.type;
                        if (zeroPointType.endsWith('[]')) {
                            zeroPointType = zeroPointType.slice(0, -2);
                        }
                        const pair = (`[scale(${scaleType}): ${scale[i]},\n zero_point(${zeroPointType}): ${zeroPoint[i]}]`);
                        if (pair in paramsMap) {
                            paramsMap[pair] += 1;
                        } else {
                            paramsMap[pair] = 1;
                        }
                    }
                    const paramsString = `x / scale + zero_point\n\n${Object.entries(paramsMap).map(([key, value]) => `${key} x ${value}`).join('\n')}`;
                    quantization = {
                        type: "annotation",
                        value: [["quantization", paramsString]]
                    };
                }
                let data = null;
                if (param.value) {
                    data = kann.Data.extractData(param.value);
                }
                const value = new kann.Value('', type, new kann.Tensor(param.name, type, data, quantization));
                this.inputs.push(new kann.Argument(param.name, value));
            });
        }
        if (node.relu) {
            const relu = { type: "ReLU", name: `${node.name}/relu`, attributes: [], inputs: [], outputs: [], params: [] };
            this.chain = [new kann.Node(metadata, relu, arcs)];
        }
    }
};

kann.Argument = class {

    constructor(name, value, type) {
        this.name = name;
        const tmpValue = type ? kann.Argument.getAttributeValue(value) : value;
        this.value = Array.isArray(tmpValue) ? tmpValue : [tmpValue];
        this.type = value.type === 'attributes' ? null : type || null;
    }

    static getAttributeValue(attribute) {
        if (attribute.type === 'attributes') {
            const value = {};
            attribute.attributes.forEach((attr) => {
                value[attr.name] = this.getAttributeValue(attr);
            });
            return value;
        } else if (attribute.value !== null) {
            return kann.Data.extractData(attribute.value);
        }
        throw new kann.Error(`${attribute.name} doesn't have a value.`);
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

    get empty() {
        return !this.values || this.values.length === 0;
    }
};

kann.TensorType = class {

    constructor(dataType, shape) {
        this.dataType = dataType || '?';
        this.shape = new kann.TensorShape(shape);
    }

    toString() {
        let result = this.dataType;
        if (this.dataType === 'subview') {
            result += "\n";
        }
        result += this.shape.toString();
        return result;
    }
};

kann.TensorShape = class {

    constructor(dimensions) {
        this.dimensions = Array.from(dimensions);
    }

    toString() {
        if (Array.isArray(this.dimensions) && this.dimensions.length > 0) {
            if (typeof this.dimensions[0] === 'string') {
                return `${this.dimensions.map((dimension) => dimension.toString()).join("\n")}`;
            }
            return `[${this.dimensions.map((dimension) => dimension.toString()).join(',')}]`;
        }
        return '';
    }
};

kann.Data = class {

    static extractData(value) {
        switch (true) {
            case /^int(8|16|32|64)?$/.test(value.type): return value.value_int;
            case /^uint(8|16|32|64)?$/.test(value.type): return value.value_uint;
            case /^float(16|32|64)?$/.test(value.type): return value.value_float;
            case /^string$/.test(value.type): return value.value_string;
            case /^int(8|16|32|64)?\[\]$/.test(value.type): return Array.from(value.list_int);
            case /^uint(8|16|32|64)?\[\]$/.test(value.type): return Array.from(value.list_uint);
            case /^float(16|32|64)?\[\]$/.test(value.type): return Array.from(value.list_float);
            case /^string\[\]$/.test(value.type): return Array.from(value.list_string);
            default: throw new kann.Error(`${value.type} is not supported.`);
        }
    }
};

kann.Error = class extends Error {

    constructor(message) {
        super(message);
        this.name = 'Error loading KaNN model.';
    }
};

export const ModelFactory = kann.ModelFactory; // Export ModelFactory for use in other modules
