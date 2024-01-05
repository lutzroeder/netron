
import * as protobuf from './protobuf.js';
import * as text from './text.js';

const nnabla = {};

nnabla.ModelFactory = class {

    match(context) {
        const identifier = context.identifier;
        if (identifier.endsWith('.nntxt')) {
            const tags = context.tags('pbtxt');
            if (tags.has('network')) {
                return 'nnabla.pbtxt';
            }
        }
        return undefined;
    }

    async open(context, target) {
        await context.require('./nnabla-proto');
        nnabla.proto = protobuf.get('nnabla').nnabla;
        switch (target) {
            case 'nnabla.pbtxt': {
                const stream = context.stream;
                const reader = protobuf.TextReader.open(stream);
                const model = nnabla.proto.NNablaProtoBuf.decodeText(reader);
                const open = async (model, version) => {
                    const metadata = await context.metadata('nnabla-metadata.json');
                    return new nnabla.Model(metadata, model, `NNabla${version ? ` v${version}` : ''}`);
                };
                try {
                    const contexts = await Promise.all([
                        context.fetch('nnp_version.txt'),
                        context.fetch('parameter.protobuf')
                    ]);
                    const version = text.Reader.open(contexts[0].stream).read();
                    const reader = protobuf.BinaryReader.open(contexts[1].stream);
                    const params = nnabla.proto.NNablaProtoBuf.decode(reader);
                    model.parameter = params.parameter;
                    return await open(model, version);
                } catch (error) {
                    return await open(model);
                }
            }
            default: {
                throw new nnabla.Error(`Unsupported nnabla format '${target}'.`);
            }
        }
    }
};

nnabla.Model = class {

    constructor(metadata, model, format) {
        this.format = format;
        this.graphs = [ new nnabla.Graph(metadata, model) ];
    }
};

nnabla.Graph = class {

    constructor (metadata, model) {
        const [executor] = model.executor; // TODO: Multiple executors?
        const network_name = executor.network_name;
        const network = model.network.find((item) => item.name === network_name);
        const dataTypes = new Map(network.variable.map((item) => {
            const shape = new nnabla.TensorShape(item.shape.dim);
            const type = new nnabla.TensorType(item.type, shape);
            return [ item.name, type ];
        }));
        const tensors = new Map(model.parameter.map((item) => {
            const name = item.variable_name;
            return [ name, new nnabla.Tensor(name, dataTypes.get(name), item.data) ];
        }));
        const values = new Map();
        values.map = (name) => {
            if (!values.has(name)) {
                values.set(name, new nnabla.Value(name, dataTypes.get(name), tensors.get(name)));
            }
            return values.get(name);
        };
        this.inputs = executor.data_variable.map((item) => {
            const name = item.variable_name;
            return new nnabla.Argument(name, [ values.map(name) ]);
        });
        this.outputs = executor.output_variable.map((item) => {
            const name = item.variable_name;
            return new nnabla.Argument(name, [ values.map(name) ]);
        });
        const get_parameters = (func) => {
            for (const [key, value] of Object.entries(func)) {
                if (key.endsWith("_param")) {
                    return value;
                }
            }

            return undefined;
        };
        this.nodes = network.function.map((func) => {
            const parameters = get_parameters(func) || [];
            const attributes = Object.entries(parameters).map(([name, value]) => {
                return new nnabla.Attribute(metadata, func.type, name, value);
            });
            const func_type = metadata.type(func.type);
            const inputs = [];
            for (let index = 0; index < func.input.length;) {
                const input = func_type.inputs && index < func_type.inputs.length ? func_type.inputs[index] : { name: index.toString() };
                const count = input.list ? func.input.length - index : 1;
                const args = func.input.slice(index, index + count).map((input) => values.map(input));
                inputs.push(new nnabla.Argument(input.name, args));
                index += count;
            }
            const outputs = [];
            for (let index = 0; index < func.output.length;) {
                const output = func_type.outputs && index < func_type.outputs.length ? func_type.outputs[index] : { name: index.toString() };
                const count = output.list ? func.output.length - index : 1;
                const args = func.output.slice(index, index + count).map((output) => values.map(output));
                outputs.push(new nnabla.Argument(output.name, args));
                index += count;
            }
            return new nnabla.Node(metadata, func, attributes, inputs, outputs);
        });
    }
};

nnabla.Argument = class {

    constructor(name, value) {
        this.name = name;
        this.value = value;
    }
};

nnabla.Value = class {

    constructor(name, type, initializer) {
        this._name = name;
        this._type = type || null;
        this._initializer = initializer || null;
    }

    get name() {
        return this._name;
    }

    get type() {
        if (this._type) {
            return this._type;
        }
        if (this._initializer) {
            return this._initializer.type;
        }
        return null;
    }

    get initializer() {
        return this._initializer;
    }
};

nnabla.Node = class {

    constructor(metadata, func, attributes, inputs, outputs) {
        this.name = func.name;
        this.type = metadata.type(func.type) || { name: func.type, type: func.type };
        this.attributes = attributes || [];
        this.outputs = outputs || [];
        this.chain = [];
        // TODO: "nonlinearity" does not match metadata type
        const get_nonlinearity = (name) => {
            switch (name) {
                case "identity": return "Identity";
                case "relu": return "ReLU";
                case "sigmoid": return "Sigmoid";
                case "tanh": return "Tanh";
                case "leaky_relu": return "LeakyReLU";
                case "elu": return "ELU";
                case "relu6": return "ReLU6";
                default: return name;
            }
        };
        switch (func.type) {
            case "FusedConvolution": {
                this.inputs = inputs.slice(0, 3) || [];
                if (inputs.length > 3) {
                    this.chain.push(new nnabla.Node(metadata, { name: `${func.name}/bn`, type: "BatchNormalization" }, [], inputs.slice(3, 7)));
                }
                if (inputs.length > 7) {
                    this.chain.push(new nnabla.Node(metadata, { name: `${func.name}/add`, type: "Add2" }, [], inputs.slice(7)));
                }
                const type_a = attributes.find((item) => item.name === "nonlinearity").value;
                this.chain.push(new nnabla.Node(metadata, { name: `${func.name}/act`, type: get_nonlinearity(type_a) }));
                break;
            }
            case "FusedBatchNormalization": {
                this.inputs = inputs.slice(0, 5) || [];
                if (inputs.length > 4) {
                    this.chain.push(new nnabla.Node(metadata, { name: `${func.name}/add`, type: "Add2" }, [], inputs.slice(5)));
                }
                const type_b = attributes.find((item) => item.name === "nonlinearity").value;
                this.chain.push(new nnabla.Node(metadata, { name: `${func.name}/act`, type: get_nonlinearity(type_b) }));
                break;
            }
            default: {
                this.inputs = inputs || [];
                break;
            }
        }
    }
};

nnabla.Attribute = class {

    constructor(metadata, type, name, value) {
        this.name = name;
        const attribute = metadata.attribute(type, name);
        this.description = attribute.description;
        switch (attribute.type) {
            case "shape":
                this.type = "int64[]";
                this.value = value.dim;
                break;
            default:
                this.type = attribute.type;
                this.value = value;
                break;
        }
        if (Object.prototype.hasOwnProperty.call(attribute, 'default') && this.value == attribute.default) {
            this.visible = false;
        }
    }
};

nnabla.Tensor = class {

    constructor(name, type, values) {
        this.name = name;
        this.type = type;
        this.encoding = '|';
        this._values = values;
    }

    get values() {
        const dataType = this.type.dataType;
        switch (dataType) {
            case 'float32': return new Float32Array(this._values);
            default: throw new nnabla.Error(`Unsupported data type '${dataType}'.`);
        }
    }
};

nnabla.TensorType = class {

    constructor(dataType, shape) {
        this.dataType = "float32";
        this.shape = shape;
        this.denotation = null; // TODO
    }

    toString() {
        return this.dataType + this.shape.toString();
    }
};

nnabla.TensorShape = class {

    constructor(dimensions) {
        this.dimensions = dimensions;
    }

    toString() {
        return (this.dimensions && this.dimensions.length) ? (`[${this.dimensions.join(',')}]`) : '';
    }
};

nnabla.Error = class extends Error {

    constructor(message) {
        super(message);
        this.name = 'Error loading Neural Network Library model.';
    }
};

export const ModelFactory = nnabla.ModelFactory;
