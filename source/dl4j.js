
// Experimental

const dl4j = {};

dl4j.ModelFactory = class {

    match(context) {
        const identifier = context.identifier;
        if (identifier === 'configuration.json') {
            const obj = context.peek('json');
            if (obj && (obj.confs || obj.vertices)) {
                context.type = 'dl4j.configuration';
                context.target = obj;
            }
        } else if (identifier === 'coefficients.bin') {
            const signature = [0x00, 0x07, 0x4A, 0x41, 0x56, 0x41, 0x43, 0x50, 0x50]; // JAVACPP
            const stream = context.stream;
            if (signature.length <= stream.length && stream.peek(signature.length).every((value, index) => value === signature[index])) {
                context.type = 'dl4j.coefficients';
            }
        }
    }

    filter(context, type) {
        return context.type !== 'dl4j.configuration' || (type !== 'dl4j.coefficients' && type !== 'openvino.bin');
    }

    async open(context) {
        const metadata = await context.metadata('dl4j-metadata.json');
        switch (context.type) {
            case 'dl4j.configuration': {
                const obj = context.target;
                try {
                    const content = await context.fetch('coefficients.bin');
                    const reader = content.read('binary.big-endian');
                    return new dl4j.Model(metadata, obj, reader);
                } catch {
                    return new dl4j.Model(metadata, obj, null);
                }
            }
            case 'dl4j.coefficients': {
                const content = await context.fetch('configuration.json');
                const obj = content.read('json');
                const reader = context.read('binary.big-endian');
                return new dl4j.Model(metadata, obj, reader);
            }
            default: {
                throw new dl4j.Error(`Unsupported Deeplearning4j format '${context.type}'.`);
            }
        }
    }
};

dl4j.Model = class {

    constructor(metadata, configuration, coefficients) {
        this.format = 'Deeplearning4j';
        this.graphs = [new dl4j.Graph(metadata, configuration, coefficients)];
    }
};

dl4j.Graph = class {

    constructor(metadata, configuration, coefficients) {
        this.inputs = [];
        this.outputs = [];
        this.nodes = [];
        coefficients = coefficients ? new dl4j.NDArray(coefficients) : null;
        const dataType = coefficients ? coefficients.dataType : '?';
        const values = new Map();
        values.map = (name, type, tensor) => {
            if (name.length === 0 && tensor) {
                return new dl4j.Value(name, type || null, tensor);
            }
            if (!values.has(name)) {
                values.set(name, new dl4j.Value(name, type || null, tensor || null));
            } else if (type || tensor) {
                throw new dl4j.Error(`Duplicate value '${name}'.`);
            }
            return values.get(name);
        };
        if (configuration.networkInputs) {
            for (const input of configuration.networkInputs) {
                const value = values.map(input);
                const argument = new dl4j.Argument(input, [value]);
                this.inputs.push(argument);
            }
        }
        if (configuration.networkOutputs) {
            for (const output of configuration.networkOutputs) {
                const value = values.map(output);
                const argument = new dl4j.Argument(output, [value]);
                this.outputs.push(argument);
            }
        }
        let inputs = null;
        // Computation Graph
        if (configuration.vertices) {
            for (const [name,obj] of Object.entries(configuration.vertices)) {
                const vertex = dl4j.Node._object(obj);
                inputs = configuration.vertexInputs[name];
                let variables = [];
                let layer = null;
                switch (vertex.__type__) {
                    case 'LayerVertex':
                        layer = dl4j.Node._object(vertex.layerConf.layer);
                        variables = vertex.layerConf.variables;
                        break;
                    case 'MergeVertex':
                        layer = { __type__: 'Merge', layerName: name };
                        break;
                    case 'ElementWiseVertex':
                        layer = { __type__: 'ElementWise', layerName: name, op: vertex.op };
                        break;
                    case 'PreprocessorVertex':
                        layer = { __type__: 'Preprocessor', layerName: name };
                        break;
                    default:
                        throw new dl4j.Error(`Unsupported vertex class '${vertex['@class']}'.`);
                }
                const node = new dl4j.Node(metadata, layer, inputs, dataType, variables, values);
                this.nodes.push(node);
            }
        }
        // Multi Layer Network
        if (configuration.confs) {
            inputs = ['input'];
            this.inputs.push(new dl4j.Argument('input', [values.map('input')]));
            for (const conf of configuration.confs) {
                const layer = dl4j.Node._object(conf.layer);
                const node = new dl4j.Node(metadata, layer, inputs, dataType, conf.variables, values);
                this.nodes.push(node);
                inputs = [layer.layerName];
            }
            this.outputs.push(new dl4j.Argument('output', [values.map(inputs[0])]));
        }
    }
};

dl4j.Argument = class {

    constructor(name, value, visible) {
        this.name = name;
        this.value = value;
        if (visible === false) {
            this.visible = false;
        }
    }
};

dl4j.Value = class {

    constructor(name, type, initializer) {
        if (typeof name !== 'string') {
            throw new dl4j.Error(`Invalid value identifier '${JSON.stringify(name)}'.`);
        }
        this.name = name;
        this.type = initializer ? initializer.type : type;
        this.initializer = initializer;
    }
};

dl4j.Node = class {

    constructor(metadata, layer, inputs, dataType, variables, values) {
        this.name = layer.layerName || '';
        this.inputs = [];
        this.outputs = [];
        this.attributes = [];
        const type = layer.__type__;
        this.type = metadata.type(type) || { name: type };
        if (inputs && inputs.length > 0) {
            const argument = new dl4j.Argument(values.length < 2 ? 'input' : 'inputs', inputs.map((input) => values.map(input)));
            this.inputs.push(argument);
        }
        if (variables) {
            for (const variable of variables) {
                let tensor = null;
                switch (type) {
                    case 'Convolution':
                        switch (variable) {
                            case 'W':
                                tensor = new dl4j.Tensor(dataType, layer.kernelSize.concat([layer.nin, layer.nout]));
                                break;
                            case 'b':
                                tensor = new dl4j.Tensor(dataType, [layer.nout]);
                                break;
                            default:
                                throw new dl4j.Error(`Unsupported '${type}' variable '${variable}'.`);
                        }
                        break;
                    case 'SeparableConvolution2D':
                        switch (variable) {
                            case 'W':
                                tensor = new dl4j.Tensor(dataType, layer.kernelSize.concat([layer.nin, layer.nout]));
                                break;
                            case 'pW':
                                tensor = new dl4j.Tensor(dataType, [layer.nout]);
                                break;
                            default:
                                throw new dl4j.Error(`Unsupported '${type}' variable '${variable}'.`);
                        }
                        break;
                    case 'Output':
                    case 'Dense':
                        switch (variable) {
                            case 'W':
                                tensor = new dl4j.Tensor(dataType, [layer.nout, layer.nin]);
                                break;
                            case 'b':
                                tensor = new dl4j.Tensor(dataType, [layer.nout]);
                                break;
                            default:
                                throw new dl4j.Error(`Unsupported '${this.type}' variable '${variable}'.`);
                        }
                        break;
                    case 'BatchNormalization':
                        tensor = new dl4j.Tensor(dataType, [layer.nin]);
                        break;
                    default:
                        throw new dl4j.Error(`Unsupported '${type}' variable '${variable}'.`);
                }
                const argument = new dl4j.Argument(variable, [values.map('', null, tensor)]);
                this.inputs.push(argument);
            }
        }
        if (this.name) {
            const value = values.map(this.name);
            const argument = new dl4j.Argument('output', [value]);
            this.outputs.push(argument);
        }
        let attributes = layer;
        if (layer.activationFn) {
            const activation = dl4j.Node._object(layer.activationFn);
            if (activation.__type__ !== 'ActivationIdentity' && activation.__type__ !== 'Identity') {
                if (activation.__type__.startsWith('Activation')) {
                    activation.__type__ = activation.__type__.substring('Activation'.length);
                }
                if (this.type === 'Activation') {
                    this.type = activation.__type__;
                    attributes = activation;
                } else {
                    this.chain = this.chain || [];
                    this.chain.push(new dl4j.Node(metadata, activation, [], null, null, values));
                }
            }
        }
        for (const [name, value] of Object.entries(attributes)) {
            switch (name) {
                case '__type__':
                case 'constraints':
                case 'layerName':
                case 'activationFn':
                case 'idropout':
                case 'hasBias':
                    continue;
                default:
                    break;
            }
            const definition = metadata.attribute(type, name);
            const visible = definition && definition.visible === false ? false : true;
            const attribute = new dl4j.Argument(name, value, visible);
            this.attributes.push(attribute);
        }
        if (layer.idropout) {
            const dropout = dl4j.Node._object(layer.idropout);
            if (dropout.p !== 1.0) {
                throw new dl4j.Error("Layer 'idropout' not implemented.");
            }
        }
    }

    static _object(value) {
        let result = {};
        if (value['@class']) {
            result = value;
            let type = value['@class'].split('.').pop();
            if (type.endsWith('Layer')) {
                type = type.substring(0, type.length - 5);
            }
            delete value['@class'];
            result.__type__ = type;
        } else {
            let [key] = Object.keys(value);
            result = value[key];
            if (key.length > 0) {
                key = key[0].toUpperCase() + key.substring(1);
            }
            result.__type__ = key;
        }
        return result;
    }
};

dl4j.Tensor = class {

    constructor(dataType, shape) {
        this.type = new dl4j.TensorType(dataType, new dl4j.TensorShape(shape));
    }
};

dl4j.TensorType = class {

    constructor(dataType, shape) {
        this.dataType = dataType;
        this.shape = shape;
    }

    toString() {
        return (this.dataType || '?') + this.shape.toString();
    }
};

dl4j.TensorShape = class {

    constructor(dimensions) {
        this.dimensions = dimensions;
    }

    toString() {
        if (this.dimensions) {
            if (this.dimensions.length === 0) {
                return '';
            }
            return `[${this.dimensions.map((dimension) => dimension.toString()).join(',')}]`;
        }
        return '';
    }
};

dl4j.NDArray = class {

    constructor(reader) {
        reader = new dl4j.BinaryReader(reader);
        const readHeader = (reader) => {
            const alloc = reader.string();
            let length = 0;
            switch (alloc) {
                case 'DIRECT':
                case 'HEAP':
                case 'JAVACPP':
                    length = reader.int32();
                    break;
                case 'LONG_SHAPE':
                case 'MIXED_DATA_TYPES':
                    length = reader.int64().toNumber();
                    break;
                default:
                    throw new dl4j.Error(`Unsupported header alloc '${alloc}'.`);
            }
            const type = reader.string();
            return [alloc, length, type];
        };
        const headerShape = readHeader(reader);
        if (headerShape[2] !== 'INT') {
            throw new dl4j.Error(`Unsupported header shape type '${headerShape[2]}'.`);
        }
        const shapeInfo = new Array(headerShape[1]);
        for (let i = 0; i < shapeInfo.length; i++) {
            shapeInfo[i] = reader.int32();
        }
        const [rank] = shapeInfo;
        const shapeInfoLength = rank * 2 + 4;
        this.shape = shapeInfo.slice(1, 1 + rank);
        this.strides = shapeInfo.slice(1 + rank, 1 + (rank * 2));
        this.order = shapeInfo[shapeInfoLength - 1];
        const headerData = readHeader(reader);
        const dataTypes = new Map([
            ['INT', ['int32', 4]],
            ['FLOAT', ['float32', 4]],
            ['DOUBLE', ['float64', 8]]
        ]);
        if (!dataTypes.has(headerData[2])) {
            throw new dl4j.Error(`Unsupported header data type '${headerShape[2]}'.`);
        }
        const [dataType, itemSize] = dataTypes.get(headerData[2]);
        this.dataType = dataType;
        const size = headerData[1] * itemSize;
        if ((reader.position + size) <= reader.length) {
            this.data = reader.read(size);
        }
    }
};

dl4j.BinaryReader = class {

    constructor(reader) {
        this._reader = reader;
    }

    get length() {
        return this._reader.length;
    }

    get position() {
        return this._reader.position;
    }

    read(length) {
        return this._reader.read(length);
    }

    int32() {
        return this._reader.int32();
    }

    int64() {
        return this._reader.int64();
    }

    uint16() {
        return this._reader.uint16();
    }

    string() {
        const size = this.uint16();
        const buffer = this.read(size);
        this._decoder = this._decoder || new TextDecoder('ascii');
        return this._decoder.decode(buffer);
    }
};

dl4j.Error = class extends Error {

    constructor(message) {
        super(message);
        this.name = 'Error loading Deeplearning4j model.';
    }
};

export const ModelFactory = dl4j.ModelFactory;
