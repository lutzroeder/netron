
// Experimental

var dl4j = {};
var json = require('./json');

dl4j.ModelFactory = class {

    match(context) {
        const identifier = context.identifier;
        if (identifier === 'configuration.json') {
            const obj = context.open('json');
            if (obj && (obj.confs || obj.vertices)) {
                return 'dl4j.configuration';
            }
        }
        if (identifier === 'coefficients.bin') {
            const signature = [ 0x00, 0x07, 0x4A, 0x41, 0x56, 0x41, 0x43, 0x50, 0x50 ]; // JAVACPP
            const stream = context.stream;
            if (signature.length <= stream.length && stream.peek(signature.length).every((value, index) => value === signature[index])) {
                return 'dl4j.coefficients';
            }
        }
        return undefined;
    }

    async open(context, target) {
        const metadata = await context.metadata('dl4j-metadata.json');
        switch (target) {
            case 'dl4j.configuration': {
                const obj = context.open('json');
                try {
                    const stream = await context.request('coefficients.bin', null);
                    return new dl4j.Model(metadata, obj, stream.peek());
                } catch (error) {
                    return new dl4j.Model(metadata, obj, null);
                }
            }
            case 'dl4j.coefficients': {
                const stream = await context.request('configuration.json', null);
                const reader = json.TextReader.open(stream);
                const obj = reader.read();
                return new dl4j.Model(metadata, obj, context.stream.peek());
            }
            default: {
                throw new dl4j.Error("Unsupported Deeplearning4j format '" + target + "'.");
            }
        }
    }
};

dl4j.Model = class {

    constructor(metadata, configuration, coefficients) {
        this._graphs = [];
        this._graphs.push(new dl4j.Graph(metadata, configuration, coefficients));
    }

    get format() {
        return 'Deeplearning4j';
    }

    get graphs() {
        return this._graphs;
    }
};

dl4j.Graph = class {

    constructor(metadata, configuration, coefficients) {
        this._inputs = [];
        this._outputs =[];
        this._nodes = [];
        coefficients = coefficients ? new dl4j.NDArray(coefficients) : null;
        const dataType = coefficients ? coefficients.dataType : '?';
        const values = new Map();
        const value = (name, type, tensor) => {
            if (name.length === 0 && tensor) {
                return new dl4j.Value(name, type || null, tensor);
            }
            if (!values.has(name)) {
                values.set(name, new dl4j.Value(name, type || null, tensor || null));
            } else if (type || tensor) {
                throw new dl4j.Error("Duplicate value '" + name + "'.");
            }
            return values.get(name);
        };
        if (configuration.networkInputs) {
            for (const input of configuration.networkInputs) {
                this._inputs.push(new dl4j.Argument(input, [ value(input) ]));
            }
        }
        if (configuration.networkOutputs) {
            for (const output of configuration.networkOutputs) {
                this._outputs.push(new dl4j.Argument(output, [ value(output) ]));
            }
        }
        let inputs = null;
        // Computation Graph
        if (configuration.vertices) {
            for (const name in configuration.vertices) {
                const vertex = dl4j.Node._object(configuration.vertices[name]);
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
                        throw new dl4j.Error("Unsupported vertex class '" + vertex['@class'] + "'.");
                }
                this._nodes.push(new dl4j.Node(metadata, layer, inputs, dataType, variables, value));
            }
        }
        // Multi Layer Network
        if (configuration.confs) {
            inputs = [ 'input' ];
            this._inputs.push(new dl4j.Argument('input', [ value('input') ]));
            for (const conf of configuration.confs) {
                const layer = dl4j.Node._object(conf.layer);
                this._nodes.push(new dl4j.Node(metadata, layer, inputs, dataType, conf.variables, value));
                inputs = [ layer.layerName ];
            }
            this._outputs.push(new dl4j.Argument('output', [ value(inputs[0]) ]));
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

dl4j.Argument = class {

    constructor(name, value) {
        this._name = name;
        this._value = value;
    }

    get name() {
        return this._name;
    }

    get value() {
        return this._value;
    }
};

dl4j.Value = class {

    constructor(name, type, initializer) {
        if (typeof name !== 'string') {
            throw new dl4j.Error("Invalid value identifier '" + JSON.stringify(name) + "'.");
        }
        this._name = name;
        this._type = type;
        this._initializer = initializer;
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

    get initializer() {
        return this._initializer;
    }
};

dl4j.Node = class {

    constructor(metadata, layer, inputs, dataType, variables, value) {
        this._name = layer.layerName || '';
        this._inputs = [];
        this._outputs = [];
        this._attributes = [];
        const type = layer.__type__;
        this._type = metadata.type(type) || { name: type };
        if (inputs && inputs.length > 0) {
            const values = inputs.map((input) => value(input));
            const argument = new dl4j.Argument(values.length < 2 ? 'input' : 'inputs', values);
            this._inputs.push(argument);
        }
        if (variables) {
            for (const variable of variables) {
                let tensor = null;
                switch (type) {
                    case 'Convolution':
                        switch (variable) {
                            case 'W':
                                tensor = new dl4j.Tensor(dataType, layer.kernelSize.concat([ layer.nin, layer.nout ]));
                                break;
                            case 'b':
                                tensor = new dl4j.Tensor(dataType, [ layer.nout ]);
                                break;
                            default:
                                throw new dl4j.Error("Unsupported '" + this._type + "' variable '" + variable + "'.");
                        }
                        break;
                    case 'SeparableConvolution2D':
                        switch (variable) {
                            case 'W':
                                tensor = new dl4j.Tensor(dataType, layer.kernelSize.concat([ layer.nin, layer.nout ]));
                                break;
                            case 'pW':
                                tensor = new dl4j.Tensor(dataType, [ layer.nout ]);
                                break;
                            default:
                                throw new dl4j.Error("Unsupported '" + this._type + "' variable '" + variable + "'.");
                        }
                        break;
                    case 'Output':
                    case 'Dense':
                        switch (variable) {
                            case 'W':
                                tensor = new dl4j.Tensor(dataType, [ layer.nout, layer.nin ]);
                                break;
                            case 'b':
                                tensor = new dl4j.Tensor(dataType, [ layer.nout ]);
                                break;
                            default:
                                throw new dl4j.Error("Unsupported '" + this._type + "' variable '" + variable + "'.");
                        }
                        break;
                    case 'BatchNormalization':
                        tensor = new dl4j.Tensor(dataType, [ layer.nin ]);
                        break;
                    default:
                        throw new dl4j.Error("Unsupported '" + this._type + "' variable '" + variable + "'.");
                }
                const argument = new dl4j.Argument(variable, [ value('', null, tensor) ]);
                this._inputs.push(argument);
            }
        }
        if (this._name) {
            this._outputs.push(new dl4j.Argument('output', [ value(this._name) ]));
        }
        let attributes = layer;
        if (layer.activationFn) {
            const activation = dl4j.Node._object(layer.activationFn);
            if (activation.__type__ !== 'ActivationIdentity' && activation.__type__ !== 'Identity') {
                if (activation.__type__.startsWith('Activation')) {
                    activation.__type__ = activation.__type__.substring('Activation'.length);
                }
                if (this._type == 'Activation') {
                    this._type = activation.__type__;
                    attributes = activation;
                } else {
                    this._chain = this._chain || [];
                    this._chain.push(new dl4j.Node(metadata, activation, [], null, null, value));
                }
            }
        }
        for (const key in attributes) {
            switch (key) {
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
            this._attributes.push(new dl4j.Attribute(metadata.attribute(type, key), key, attributes[key]));
        }
        if (layer.idropout) {
            const dropout = dl4j.Node._object(layer.idropout);
            if (dropout.p !== 1.0) {
                throw new dl4j.Error("Layer 'idropout' not implemented.");
            }
        }
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

    get chain() {
        return this._chain;
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
            let key = Object.keys(value)[0];
            result = value[key];
            if (key.length > 0) {
                key = key[0].toUpperCase() + key.substring(1);
            }
            result.__type__ = key;
        }
        return result;
    }
};

dl4j.Attribute = class {

    constructor(metadata, name, value) {
        this._name = name;
        this._value = value;
        if (metadata && metadata.visible === false) {
            this._visible = false;
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
        return this._visible;
    }
};

dl4j.Tensor = class {

    constructor(dataType, shape) {
        this._type = new dl4j.TensorType(dataType, new dl4j.TensorShape(shape));
    }

    get type() {
        return this._type;
    }
};

dl4j.TensorType = class {

    constructor(dataType, shape) {
        this._dataType = dataType;
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

dl4j.TensorShape = class {

    constructor(dimensions) {
        this._dimensions = dimensions;
    }

    get dimensions() {
        return this._dimensions;
    }

    toString() {
        if (this._dimensions) {
            if (this._dimensions.length == 0) {
                return '';
            }
            return '[' + this._dimensions.map((dimension) => dimension.toString()).join(',') + ']';
        }
        return '';
    }
};

dl4j.NDArray = class {

    constructor(buffer) {
        const reader = new dl4j.BinaryReader(buffer);
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
                    length = reader.int64();
                    break;
                default:
                    throw new dl4j.Error("Unsupported header alloc '" + alloc + "'.");
            }
            const type = reader.string();
            return [ alloc, length, type ];
        };
        const headerShape = readHeader(reader);
        if (headerShape[2] !== 'INT') {
            throw new dl4j.Error("Unsupported header shape type '" + headerShape[2] + "'.");
        }
        const shapeInfo = new Array(headerShape[1]);
        for (let i = 0; i < shapeInfo.length; i++) {
            shapeInfo[i] = reader.int32();
        }
        const rank = shapeInfo[0];
        const shapeInfoLength = rank * 2 + 4;
        this.shape = shapeInfo.slice(1, 1 + rank);
        this.strides = shapeInfo.slice(1 + rank, 1 + (rank * 2));
        this.order = shapeInfo[shapeInfoLength - 1];
        const headerData = readHeader(reader);
        const dataTypes = new Map([
            [ 'INT', [ 'int32', 4 ] ],
            [ 'FLOAT', [ 'float32', 4 ] ],
            [ 'DOUBLE', [ 'float64', 8 ] ]
        ]);
        if (!dataTypes.has(headerData[2])) {
            throw new dl4j.Error("Unsupported header data type '" + headerShape[2] + "'.");
        }
        const dataType = dataTypes.get(headerData[2]);
        this.dataType = dataType[0];
        const size = headerData[1] * dataType[1];
        if ((reader.position + size) <= reader.length) {
            this.data = reader.read(size);
        }
    }
};

dl4j.BinaryReader = class {

    constructor(buffer) {
        this._buffer = buffer;
        this._length = buffer.length;
        this._position = 0;
        this._view = new DataView(buffer.buffer, buffer.byteOffset, buffer.byteLength);
    }

    get length() {
        return this._length;
    }

    get position() {
        return this._position;
    }

    read(size) {
        const data = this._buffer.subarray(this._position, this._position + size);
        this._position += size;
        return data;
    }

    string() {
        const size = this._buffer[this._position++] << 8 | this._buffer[this._position++];
        const buffer = this.read(size);
        this._decoder = this._decoder || new TextDecoder('ascii');
        return this._decoder.decode(buffer);
    }

    int32() {
        const position = this._position;
        this._position += 4;
        return this._view.getInt32(position, false);
    }

    int64() {
        const position = this._position;
        this._position += 4;
        return this._view.getInt64(position, false).toNumber();
    }

    float32() {
        const position = this._position;
        this._position += 4;
        return this._view.getFloat32(position, false);
    }
};

dl4j.Error = class extends Error {
    constructor(message) {
        super(message);
        this.name = 'Error loading Deeplearning4j model.';
    }
};

if (typeof module !== 'undefined' && typeof module.exports === 'object') {
    module.exports.ModelFactory = dl4j.ModelFactory;
}