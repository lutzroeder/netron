
import * as text from './text.js';

const nnabla = {};

nnabla.ModelFactory = class {

    match(context) {
        const identifier = context.identifier;
        if (identifier.endsWith('.nntxt')) {
            const tags = context.tags('pbtxt');
            if (tags.has('network')) {
                context.type = 'nnabla.pbtxt';
            }
        }
    }

    async open(context) {
        nnabla.proto = await context.require('./nnabla-proto');
        nnabla.proto = nnabla.proto.nnabla;
        switch (context.type) {
            case 'nnabla.pbtxt': {
                const reader = context.read('protobuf.text');
                const model = nnabla.proto.NNablaProtoBuf.decodeText(reader);
                const files = ['nnp_version.txt', 'parameter.protobuf', 'parameter.h5'];
                let contexts = await Promise.all(files.map((file) => context.fetch(file).catch(() => null)));
                contexts = contexts.filter((context) => context !== null);
                contexts = new Map(contexts.map((context) => [context.identifier, context]));
                let version = '';
                if (contexts.has('nnp_version.txt')) {
                    const context = contexts.get('nnp_version.txt');
                    const stream = context.stream;
                    const reader = text.Reader.open(stream);
                    version = reader.read();
                    version = version.split('\r').shift();
                }
                if (contexts.has('parameter.protobuf')) {
                    const context = contexts.get('parameter.protobuf');
                    const reader = context.read('protobuf.binary');
                    const params = nnabla.proto.NNablaProtoBuf.decode(reader);
                    model.parameter = params.parameter;
                } else if (contexts.has('parameter.h5')) {
                    const context = contexts.get('parameter.h5');
                    const file = context.read('hdf5');
                    const queue = [['',file]];
                    while (queue.length > 0) {
                        const [name, group] = queue.shift();
                        if (group.value) {
                            const variable = group.value;
                            const data = variable.data.peek();
                            const buffer = new Uint8Array(data.length);
                            buffer.set(data, 0);
                            const parameter = new nnabla.proto.Parameter();
                            parameter.variable_name = name;
                            parameter.shape = new nnabla.proto.Shape();
                            parameter.shape.dim = variable.shape.map((dim) => BigInt(dim));
                            parameter.data = new Float32Array(buffer.buffer, buffer.byteOffset, buffer.byteLength >> 2);
                            model.parameter.push(parameter);
                        } else {
                            for (const [key, value] of group.groups) {
                                queue.push([name ? `${name}/${key}` : key, value]);
                            }
                        }
                    }
                }
                const metadata = await context.metadata('nnabla-metadata.json');
                return new nnabla.Model(metadata, model, version);
            }
            default: {
                throw new nnabla.Error(`Unsupported nnabla format '${context.type}'.`);
            }
        }
    }

    filter(context, type) {
        return context.type !== 'nnabla.pbtxt' || type !== 'hdf5.parameter.h5';
    }
};

nnabla.Model = class {

    constructor(metadata, model, version) {
        this.format = `NNabla${version ? ` v${version}` : ''}`;
        this.graphs = [];
        const tensors = new Map(model.parameter.map((parameter) => {
            const name = parameter.variable_name;
            const shape = new nnabla.TensorShape(parameter.shape.dim);
            const type = new nnabla.TensorType(shape);
            return [name, new nnabla.Tensor(name, type, parameter.data)];
        }));
        const networks = new Map(model.network.map((network) => [network.name, network]));
        for (const executor of model.executor) {
            const network = networks.get(executor.network_name);
            const graph = new nnabla.Graph(metadata, network, executor.data_variable, executor.output_variable, tensors);
            this.graphs.push(graph);
        }
        for (const optimizer of model.optimizer) {
            const network = networks.get(optimizer.network_name);
            const graph = new nnabla.Graph(metadata, network, optimizer.data_variable, optimizer.loss_variable, tensors);
            this.graphs.push(graph);
        }
        for (const monitor of model.monitor) {
            const network = networks.get(monitor.network_name);
            const graph = new nnabla.Graph(metadata, network, monitor.data_variable, monitor.monitor_variable, tensors);
            this.graphs.push(graph);
        }
    }
};

nnabla.Graph = class {

    constructor (metadata, network, inputs, outputs, tensors) {
        this.name = network.name;
        const values = new Map(network.variable.map((variable) => {
            const name = variable.name;
            const shape = new nnabla.TensorShape(variable.shape.dim);
            const type = new nnabla.TensorType(shape);
            return [name, new nnabla.Value(name, type, tensors.get(name))];
        }));
        values.map = (name) => {
            if (!values.has(name)) {
                values.set(name, new nnabla.Value(name, null, tensors.get(name)));
            }
            return values.get(name);
        };
        this.inputs = inputs.map((item) => {
            const name = item.variable_name;
            return new nnabla.Argument(name, [values.map(name)]);
        });
        this.outputs = outputs.map((output) => {
            const name = output.variable_name;
            return new nnabla.Argument(name, [values.map(name)]);
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
                const argument = new nnabla.Argument(input.name, args);
                inputs.push(argument);
                index += count;
            }
            const outputs = [];
            for (let index = 0; index < func.output.length;) {
                const output = func_type.outputs && index < func_type.outputs.length ? func_type.outputs[index] : { name: index.toString() };
                const count = output.list ? func.output.length - index : 1;
                const args = func.output.slice(index, index + count).map((output) => values.map(output));
                const argument = new nnabla.Argument(output.name, args);
                outputs.push(argument);
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
        // "nonlinearity" does not match metadata type
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
        if (Object.prototype.hasOwnProperty.call(attribute, 'default') && this.value === attribute.default) {
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

    constructor(shape) {
        this.dataType = "float32";
        this.shape = shape;
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
