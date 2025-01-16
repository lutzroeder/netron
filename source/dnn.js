
// Experimental

const dnn = {};

dnn.ModelFactory = class {

    match(context) {
        const tags = context.tags('pb');
        if (tags.get(4) === 0 && tags.get(10) === 2) {
            context.type = 'dnn';
        }
    }

    async open(context) {
        dnn.proto = await context.require('./dnn-proto');
        dnn.proto = dnn.proto.dnn;
        let model = null;
        try {
            const reader = context.read('protobuf.binary');
            model = dnn.proto.Model.decode(reader);
        } catch (error) {
            const message = error && error.message ? error.message : error.toString();
            throw new dnn.Error(`File format is not dnn.Graph (${message.replace(/\.$/, '')}).`);
        }
        const metadata = await context.metadata('dnn-metadata.json');
        return new dnn.Model(metadata, model);
    }
};

dnn.Model = class {

    constructor(metadata, model) {
        this.name = model.name || '';
        this.format = `SnapML${model.version ? ` v${model.version}` : ''}`;
        this.graphs = [new dnn.Graph(metadata, model)];
    }
};

dnn.Graph = class {

    constructor(metadata, model) {
        this.inputs = [];
        this.outputs = [];
        this.nodes = [];
        const scope = {};
        for (let i = 0; i < model.node.length; i++) {
            const node = model.node[i];
            node.input = node.input.map((input) => scope[input] ? scope[input] : input);
            node.output = node.output.map((output) => {
                scope[output] = scope[output] ? `${output}\n${i}` : output; // custom argument id
                return scope[output];
            });
        }
        const values = new Map();
        values.map = (name, type) => {
            if (!values.has(name)) {
                values.set(name, new dnn.Value(name, type));
            }
            return values.get(name);
        };
        for (const input of model.input) {
            const shape = input.shape;
            const type = new dnn.TensorType('float32', new dnn.TensorShape([shape.dim0, shape.dim1, shape.dim2, shape.dim3]));
            const argument = new dnn.Argument(input.name, [values.map(input.name, type)]);
            this.inputs.push(argument);
        }
        for (const output of model.output) {
            const shape = output.shape;
            const type = new dnn.TensorType('float32', new dnn.TensorShape([shape.dim0, shape.dim1, shape.dim2, shape.dim3]));
            const argument = new dnn.Argument(output.name, [values.map(output.name, type)]);
            this.outputs.push(argument);
        }
        if (this.inputs.length === 0 && model.input_name && model.input_shape && model.input_shape.length === model.input_name.length * 4) {
            for (let i = 0; i < model.input_name.length; i++) {
                const name = model.input_name[i];
                const shape = model.input_shape.slice(i * 4, (i * 4 + 4));
                const type = new dnn.TensorType('float32', new dnn.TensorShape([shape[1], shape[3], shape[2], shape[0]]));
                const argument = new dnn.Argument(name, [values.map(name, type)]);
                this.inputs.push(argument);
            }
        }
        if (this.inputs.length === 0 &&  model.input_shape && model.input_shape.length === 4 && model.node.length > 0 && model.node[0].input.length > 0) {
            /* eslint-disable prefer-destructuring */
            const name = model.node[0].input[0];
            /* eslint-enable prefer-destructuring */
            const shape = model.input_shape;
            const type = new dnn.TensorType('float32', new dnn.TensorShape([shape[1], shape[3], shape[2], shape[0]]));
            const argument = new dnn.Argument(name, [values.map(name, type)]);
            this.inputs.push(argument);
        }

        for (const node of model.node) {
            this.nodes.push(new dnn.Node(metadata, node, values));
        }
    }
};

dnn.Argument = class {

    constructor(name, value) {
        this.name = name;
        this.value = value;
    }
};

dnn.Value = class {

    constructor(name, type, initializer, quantization) {
        if (typeof name !== 'string') {
            throw new dnn.Error(`Invalid value identifier '${JSON.stringify(name)}'.`);
        }
        this.name = name;
        this.type = type || null;
        this.initializer = initializer || null;
        if (quantization) {
            this.quantization = {
                type: 'lookup',
                value: new Map(quantization.map((value, index) => [index, value]))
            };
        }
    }
};

dnn.Node = class {

    constructor(metadata, node, values) {
        const layer = node.layer;
        this.name = layer.name;
        const type = layer.type;
        this.type = metadata.type(type) || { name: type };
        this.attributes = [];
        this.inputs = [];
        this.outputs = [];
        const inputs = node.input.map((input) => values.map(input));
        for (const weight of layer.weight) {
            let quantization = null;
            if (layer.is_quantized && weight === layer.weight[0] && layer.quantization && layer.quantization.data) {
                const data = layer.quantization.data;
                quantization = new Array(data.length >> 2);
                const view = new DataView(data.buffer, data.byteOffset, data.byteLength);
                for (let i = 0; i < quantization.length; i++) {
                    quantization[i] = view.getFloat32(i << 2, true);
                }
            }
            const initializer = new dnn.Tensor(weight, quantization);
            inputs.push(new dnn.Value('', initializer.type, initializer, quantization));
        }
        const outputs = node.output.map((output) => values.map(output));
        if (inputs && inputs.length > 0) {
            let inputIndex = 0;
            if (this.type && this.type.inputs) {
                for (const inputSchema of this.type.inputs) {
                    if (inputIndex < inputs.length || inputSchema.option !== 'optional') {
                        const inputCount = (inputSchema.option === 'variadic') ? (node.input.length - inputIndex) : 1;
                        const inputArguments = inputs.slice(inputIndex, inputIndex + inputCount);
                        this.inputs.push(new dnn.Argument(inputSchema.name, inputArguments));
                        inputIndex += inputCount;
                    }
                }
            }
            this.inputs.push(...inputs.slice(inputIndex).map((input, index) => {
                const inputName = ((inputIndex + index) === 0) ? 'input' : (inputIndex + index).toString();
                return new dnn.Argument(inputName, [input]);
            }));
        }
        if (outputs.length > 0) {
            this.outputs = outputs.map((output, index) => {
                const inputName = (index === 0) ? 'output' : index.toString();
                return new dnn.Argument(inputName, [output]);
            });
        }
        for (const [key, obj] of Object.entries(layer)) {
            switch (key) {
                case 'name':
                case 'type':
                case 'weight':
                case 'is_quantized':
                case 'quantization':
                    break;
                default: {
                    const attribute = new dnn.Argument(key, obj);
                    this.attributes.push(attribute);
                    break;
                }
            }
        }
    }
};

dnn.Tensor = class {

    constructor(weight, quantization) {
        const shape = new dnn.TensorShape([weight.dim0, weight.dim1, weight.dim2, weight.dim3]);
        this.values = quantization ? weight.quantized_data : weight.data;
        const size = shape.dimensions.reduce((a, b) => a * b, 1);
        const itemsize = Math.floor(this.values.length / size);
        const remainder = this.values.length - (itemsize * size);
        if (remainder < 0 || remainder > itemsize) {
            throw new dnn.Error('Invalid tensor data size.');
        }
        let dataType = '?';
        switch (itemsize) {
            case 1: dataType = 'int8'; break;
            case 2: dataType = 'float16'; break;
            case 4: dataType = 'float32'; break;
            default: dataType = '?'; break;
        }
        this.type = new dnn.TensorType(dataType, shape);
    }
};

dnn.TensorType = class {

    constructor(dataType, shape) {
        this.dataType = dataType;
        this.shape = shape;
    }

    toString() {
        return this.dataType + this.shape.toString();
    }
};

dnn.TensorShape = class {

    constructor(shape) {
        this.dimensions = shape;
    }

    toString() {
        if (!this.dimensions || this.dimensions.length === 0) {
            return '';
        }
        return `[${this.dimensions.join(',')}]`;
    }
};

dnn.Error = class extends Error {

    constructor(message) {
        super(message);
        this.name = 'Error loading SnapML model.';
    }
};

export const ModelFactory = dnn.ModelFactory;

