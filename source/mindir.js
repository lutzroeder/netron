
const mindir = {};

mindir.ModelFactory = class {

    async match(context) {
        const identifier = context.identifier;
        const extension = identifier.split('.').pop().toLowerCase();
        if (extension === 'mindir') {
            const tags = await context.tags('pb');
            if (tags.size > 0) {
                return context.set('mindir');
            }
        }
        return null;
    }

    async open(context) {
        mindir.proto = await context.require('./mindir-proto');
        mindir.proto = mindir.proto.mind_ir;
        let model = null;
        try {
            const reader = await context.read('protobuf.binary');
            model = mindir.proto.ModelProto.decode(reader);
        } catch (error) {
            const message = error && error.message ? error.message : error.toString();
            throw new mindir.Error(`File format is not mind_ir.ModelProto (${message.replace(/\.$/, '')}).`);
        }
        const metadata = await context.metadata('mindir-metadata.json');
        return new mindir.Model(metadata, model);
    }
};

mindir.Model = class {

    constructor(metadata, model) {
        this.format = 'MindIR';
        if (model.model_version) {
            this.format += ` v${model.model_version}`;
        }
        this.producer = model.producer_name || '';
        const primitives = new Map();
        for (const primitive of model.primitives) {
            primitives.set(primitive.name, primitive);
        }
        this.modules = [];
        if (model.graph) {
            this.modules.push(new mindir.Graph(metadata, model.graph, primitives));
        }
        for (const graph of model.functions) {
            this.modules.push(new mindir.Graph(metadata, graph, primitives));
        }
    }
};

mindir.Graph = class {

    constructor(metadata, graph, primitives) {
        this.name = graph.name || '';
        this.inputs = [];
        this.outputs = [];
        this.nodes = [];
        const values = new Map();
        values.map = (name, type, initializer) => {
            if (!values.has(name)) {
                values.set(name, new mindir.Value(name, type || null, initializer || null));
            }
            return values.get(name);
        };
        const parameters = new Map();
        for (const parameter of graph.parameter) {
            const name = parameter.name || '';
            const type = new mindir.TensorType(parameter);
            const data = parameter.raw_data && parameter.raw_data.length > 0 ? parameter.raw_data : null;
            const initializer = data ? new mindir.Tensor(type, data) : null;
            parameters.set(name, { type, initializer });
            values.map(name, type, initializer);
        }
        for (const input of graph.input) {
            const name = input.name || '';
            let type = null;
            if (input.tensor && input.tensor.length > 0) {
                type = new mindir.TensorType(input.tensor[0]);
            }
            if (!parameters.has(name)) {
                const value = values.map(name, type);
                this.inputs.push(new mindir.Argument(name, [value]));
            }
        }
        for (const output of graph.output) {
            const name = output.name || '';
            let type = null;
            if (output.tensor && output.tensor.length > 0) {
                type = new mindir.TensorType(output.tensor[0]);
            }
            const value = values.map(name, type);
            this.outputs.push(new mindir.Argument(name, [value]));
        }
        for (const node of graph.node) {
            if (node.op_type === 'Constant') {
                const name = node.output && node.output.length > 0 ? node.output[0] : '';
                const attr = node.attribute && node.attribute.length > 0 ? node.attribute[0] : null;
                if (attr && attr.t) {
                    const type = new mindir.TensorType(attr.t);
                    const data = attr.t.raw_data && attr.t.raw_data.length > 0 ? attr.t.raw_data : null;
                    const initializer = data ? new mindir.Tensor(type, data) : null;
                    values.map(name, type, initializer);
                } else {
                    values.map(name);
                }
                continue;
            }
            this.nodes.push(new mindir.Node(metadata, node, primitives, values));
        }
    }
};

mindir.Node = class {

    constructor(metadata, node, primitives, values) {
        this.name = node.name || '';
        const op_type = node.op_type || '';
        const match = op_type.match(/^(?:REF::)?(.+?)(?::\d+)?$/);
        const name = match ? match[1] : op_type;
        this.type = metadata.type(name) || { name };
        this.inputs = [];
        this.outputs = [];
        this.attributes = [];
        const ref = op_type.replace(/^REF::/, '');
        const primitive = primitives.get(ref);
        const names = (key) => {
            const attr = primitive ? primitive.attribute.find((a) => a.name === key) : null;
            if (attr && attr.values && attr.values.length > 0) {
                return attr.values.map((v) => {
                    if (v.s && v.s.length > 0) {
                        return new TextDecoder('utf-8').decode(v.s);
                    }
                    if (v.strings && v.strings.length > 0) {
                        return new TextDecoder('utf-8').decode(v.strings[0]);
                    }
                    return '';
                });
            }
            return null;
        };
        const inputNames = names('input_names');
        const outputNames = names('output_names');
        for (let i = 0; i < node.input.length; i++) {
            const value = values.map(node.input[i]);
            const argName = inputNames && i < inputNames.length ? inputNames[i] : i.toString();
            this.inputs.push(new mindir.Argument(argName, [value]));
        }
        for (let i = 0; i < node.output.length; i++) {
            const value = values.map(node.output[i]);
            const argName = outputNames && i < outputNames.length ? outputNames[i] : i.toString();
            this.outputs.push(new mindir.Argument(argName, [value]));
        }
        if (primitive) {
            for (const attr of primitive.attribute) {
                if (attr.name === 'input_names' || attr.name === 'output_names') {
                    continue;
                }
                let value = null;
                let type = null;
                switch (attr.type) {
                    case 1: value = attr.f; type = 'float32'; break;
                    case 7: value = attr.i; type = 'int64'; break;
                    case 8: {
                        if (attr.s && attr.s.length > 0) {
                            value = new TextDecoder('utf-8').decode(attr.s);
                            type = 'string';
                        }
                        break;
                    }
                    case 9: value = attr.i !== 0n; type = 'boolean'; break;
                    case 11: value = attr.d; type = 'float64'; break;
                    case 20: {
                        if (attr.values && attr.values.length > 0) {
                            const items = attr.values.map((v) => {
                                if (v.type === 7) {
                                    return v.i;
                                }
                                if (v.type === 1) {
                                    return v.f;
                                }
                                return null;
                            });
                            if (items.every((v) => v !== null)) {
                                value = items;
                            }
                        }
                        break;
                    }
                    default: break;
                }
                if (value !== null) {
                    this.attributes.push(new mindir.Argument(attr.name, value, type));
                }
            }
        }
    }
};

mindir.Argument = class {

    constructor(name, value, type = null) {
        this.name = name;
        this.value = value;
        this.type = type;
    }
};

mindir.Value = class {

    constructor(name, type = null, initializer = null) {
        if (typeof name !== 'string') {
            throw new mindir.Error(`Invalid value identifier '${JSON.stringify(name)}'.`);
        }
        this.name = name;
        this.type = initializer ? initializer.type : type;
        this.initializer = initializer;
    }
};

mindir.Tensor = class {

    constructor(type, data) {
        this.type = type;
        this.values = data;
    }
};

mindir.TensorType = class {

    constructor(tensor) {
        switch (tensor.data_type) {
            case 0: this.dataType = '?'; break;
            case 1: this.dataType = 'float32'; break;
            case 2: this.dataType = 'uint8'; break;
            case 3: this.dataType = 'int8'; break;
            case 4: this.dataType = 'uint16'; break;
            case 5: this.dataType = 'int16'; break;
            case 6: this.dataType = 'int32'; break;
            case 7: this.dataType = 'int64'; break;
            case 8: this.dataType = 'string'; break;
            case 9: this.dataType = 'boolean'; break;
            case 10: this.dataType = 'float16'; break;
            case 11: this.dataType = 'float64'; break;
            case 12: this.dataType = 'uint32'; break;
            case 13: this.dataType = 'uint64'; break;
            case 14: this.dataType = 'complex64'; break;
            case 15: this.dataType = 'complex128'; break;
            case 16: this.dataType = 'bfloat16'; break;
            case 17: this.dataType = 'float64'; break;
            case 18: this.dataType = 'qint4x2'; break;
            default: this.dataType = '?'; break;
        }
        this.shape = new mindir.TensorShape(tensor.dims ? tensor.dims.map((dim) => typeof dim === 'bigint' ? Number(dim) : dim) : []);
    }

    toString() {
        return this.dataType + this.shape.toString();
    }
};

mindir.TensorShape = class {

    constructor(dimensions) {
        this.dimensions = dimensions;
    }

    toString() {
        if (this.dimensions && this.dimensions.length > 0) {
            return `[${this.dimensions.map((dim) => dim ? dim.toString() : '?').join(',')}]`;
        }
        return '';
    }
};

mindir.Error = class extends Error {

    constructor(message) {
        super(message);
        this.name = 'Error loading MindIR model.';
    }
};

export const ModelFactory = mindir.ModelFactory;
