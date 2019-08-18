/* jshint esversion: 6 */
/* eslint "indent": [ "error", 4, { "SwitchCase": 1 } ] */

// Experimental

var dl4j = dl4j || {};
var long = long || { Long: require('long') };

dl4j.ModelFactory = class {

    match(context) {
        var identifier = context.identifier.toLowerCase();
        var extension = identifier.split('.').pop().toLowerCase();
        if (extension === 'zip' && context.entries.length > 0) {
            if (dl4j.ModelFactory._openContainer(context)) {
                return true;
            }
        }
        return false;
    }

    open(context, host) {
        let identifier = context.identifier;
        try {
            let container = dl4j.ModelFactory._openContainer(context); 
            let configuration = JSON.parse(container.configuration);
            return dl4j.Metadata.open(host).then((metadata) => {
                try {
                    return new dl4j.Model(metadata, configuration, container.coefficients);
                }
                catch (error) {
                    host.exception(error, false);
                    var message = error && error.message ? error.message : error.toString();
                    message = message.endsWith('.') ? message.substring(0, message.length - 1) : message;
                    throw new dl4j.Error(message + " in '" + identifier + "'.");
                }
            });
        }
        catch (error) {
            host.exception(error, false);
            var message = error && error.message ? error.message : error.toString();
            message = message.endsWith('.') ? message.substring(0, message.length - 1) : message;
            return Promise.reject(new dl4j.Error(message + " in '" + identifier + "'."));
        }
    }

    static _openContainer(context) {
        let configurationEntries = context.entries.filter((entry) => entry.name === 'configuration.json');
        if (configurationEntries.length != 1) {
            return null;
        }
        var configuration = null;
        try {
            configuration = new TextDecoder('utf-8').decode(configurationEntries[0].data);
        }
        catch (error) {
            return null;
        }
        if (configuration.indexOf('"vertices"') === -1 && configuration.indexOf('"confs"') === -1) {
            return null;
        }
        let coefficientsEntries = context.entries.filter((entry) => entry.name === 'coefficients.bin');
        if (coefficientsEntries.length > 1) {
            return null;
        }
        let coefficients = coefficientsEntries.length == 1 ? coefficientsEntries[0].data : 0;
        var container = {};
        container.configuration = configuration;
        container.coefficients = coefficients;
        return container;
    }
}

dl4j.Model = class {

    constructor(metadata, configuration, coefficients) {
        this._graphs = [];
        this._graphs.push(new dl4j.Graph(metadata, configuration, coefficients))
    }

    get format() {
        return 'Deeplearning4j';
    }

    get graphs() {
        return this._graphs;
    }
}

dl4j.Graph = class {

    constructor(metadata, configuration, coefficients) {

        this._inputs = [];
        this._outputs =[];
        this._nodes = [];

        var reader = new dl4j.NDArrayReader(coefficients);
        var dataType = reader.dataType;

        if (configuration.networkInputs) {
            for (var input of configuration.networkInputs) {
                this._inputs.push(new dl4j.Parameter(input, true, [
                    new dl4j.Argument(input, null, null)
                ]));
            }
        }

        if (configuration.networkOutputs) {
            for (var output of configuration.networkOutputs) {
                this._outputs.push(new dl4j.Parameter(output, true, [
                    new dl4j.Argument(output, null, null)
                ]));
            }
        }

        var inputs = null;

        // Computation Graph
        if (configuration.vertices) {
            for (var name in configuration.vertices) {

                var vertex = dl4j.Node._object(configuration.vertices[name]);
                inputs = configuration.vertexInputs[name];
                var variables = [];
                var layer = null;

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
        
                this._nodes.push(new dl4j.Node(metadata, layer, inputs, dataType, variables));
            }
        }

        // Multi Layer Network
        if (configuration.confs) {
            inputs = [ 'input' ];
            this._inputs.push(new dl4j.Parameter('input', true, [
                new dl4j.Argument('input', null, null)
            ]));
            for (var conf of configuration.confs) {
                layer = dl4j.Node._object(conf.layer);
                this._nodes.push(new dl4j.Node(metadata, layer, inputs, dataType, conf.variables));
                inputs = [ layer.layerName ];
            }
            this._outputs.push(new dl4j.Parameter('output', true, [
                new dl4j.Argument(inputs[0], null, null)
            ]));
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
}

dl4j.Parameter = class {

    constructor(name, visible, args) {
        this._name = name;
        this._visible = visible;
        this._arguments = args;
    }

    get name() {
        return this._name;
    }

    get visible() {
        return this._visible;
    }

    get arguments() {
        return this._arguments;
    }
};

dl4j.Argument = class {

    constructor(id, type, initializer) {
        this._id = id;
        this._type = type;
        this._initializer = initializer;
    }

    get id() {
        return this._id;
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

    constructor(metadata, layer, inputs, dataType, variables) {

        this._metadata = metadata;
        this._operator = layer.__type__;
        this._name = layer.layerName || '';
        this._inputs = [];
        this._outputs = [];
        this._attributes = [];

        if (inputs && inputs.length > 0) {
            var args = inputs.map((input) => new dl4j.Argument(input, null, null));
            this._inputs.push(new dl4j.Parameter(args.length < 2 ? 'input' : 'inputs', true, args));
        }

        if (variables) {
            for (var variable of variables) {
                var tensor = null;
                switch (this._operator) {
                    case 'Convolution':
                        switch (variable) {
                            case 'W':
                                tensor = new dl4j.Tensor(dataType, layer.kernelSize.concat([ layer.nin, layer.nout ]));
                                break;
                            case 'b':
                                tensor = new dl4j.Tensor(dataType, [ layer.nout ]);
                                break;
                            default:
                                throw new dl4j.Error("Unknown '" + this._operator + "' variable '" + variable + "'.");
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
                                throw new dl4j.Error("Unknown '" + this._operator + "' variable '" + variable + "'.");
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
                                throw new dl4j.Error("Unknown '" + this._operator + "' variable '" + variable + "'.");
                        }
                        break;
                    case 'BatchNormalization':
                        tensor = new dl4j.Tensor(dataType, [ layer.nin ]);
                        break;
                    default:
                        throw new dl4j.Error("Unknown '" + this._operator + "' variable '" + variable + "'.");
                }
                this._inputs.push(new dl4j.Parameter(variable, true, [
                    new dl4j.Argument(variable, null, tensor)
                ]));
            }
        }

        if (this._name) {
            this._outputs.push(new dl4j.Parameter('output', true, [
                new dl4j.Argument(this._name, null, null)
            ]));
        }

        var attributes = layer;

        if (layer.activationFn) {
            var activation = dl4j.Node._object(layer.activationFn);
            if (activation.__type__ !== 'ActivationIdentity' && activation.__type__ !== 'Identity') {
                if (activation.__type__.startsWith('Activation')) {
                    activation.__type__ = activation.__type__.substring('Activation'.length);
                }
                if (this._operator == 'Activation') {
                    this._operator = activation.__type__;
                    attributes = activation;
                }
                else {
                    this._chain = this._chain || []; 
                    this._chain.push(new dl4j.Node(metadata, activation, [], null, null));
                }
            }
        }

        for (var key in attributes) {
            switch (key) {
                case '__type__':
                case 'constraints':
                case 'layerName':
                case 'activationFn':
                case 'idropout':
                case 'hasBias':
                    continue;
            }
            this._attributes.push(new dl4j.Attribute(metadata, this._operator, key, attributes[key]));
        }

        if (layer.idropout) {
            var dropout = dl4j.Node._object(layer.idropout);
            if (dropout.p !== 1.0) {
                throw new dl4j.Error("Layer 'idropout' not implemented.");
            }
        }
    }

    get operator() {
        return this._operator;
    }

    get name() {
        return this._name;
    }

    get category() {
        var schema = this._metadata.getSchema(this._operator);
        return (schema && schema.category) ? schema.category : '';
    }

    get documentation() {
        return '';
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
        var result = {};
        if (value['@class']) {
            result = value;
            var type = value['@class'].split('.').pop();
            if (type.endsWith('Layer')) {
                type = type.substring(0, type.length - 5);
            }
            delete value['@class'];
            result.__type__ = type;
        }
        else {
            var key = Object.keys(value)[0];
            result = value[key];
            if (key.length > 0) {
                key = key[0].toUpperCase() + key.substring(1);
            }
            result.__type__ = key;
        }
        return result;
    }
}

dl4j.Attribute = class {

    constructor(metadata, operator, name, value) {
        this._name = name;
        this._value = value;
        this._visible = false;
        var schema = metadata.getAttributeSchema(operator, name);
        if (schema) {
            if (schema.visible) {
                this._visible = true;
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
        return this._visible;
    }
}

dl4j.Tensor = class {

    constructor(dataType, shape) {
        this._type = new dl4j.TensorType(dataType, new dl4j.TensorShape(shape));
    }

    get type() {
        return this._type;
    }

    get state() {
        return 'Not implemented.'
    }
}

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

dl4j.Metadata = class {

    static open(host) {
        dl4j.Metadata.textDecoder = dl4j.Metadata.textDecoder || new TextDecoder('utf-8');
        if (dl4j.Metadata._metadata) {
            return Promise.resolve(dl4j.Metadata._metadata);
        }
        return host.request(null, 'dl4j-metadata.json', 'utf-8').then((data) => {
            dl4j.Metadata._metadata = new dl4j.Metadata(data);
            return dl4j.Metadata._metadata;
        }).catch(() => {
            dl4j.Metadata._metadata = new dl4j.Metadata(null);
            return dl4j.Metadata._metadata;
        });
    }

    constructor(data) {
        this._map = {};
        this._attributeCache = {};
        if (data) {
            if (data) {
                var items = JSON.parse(data);
                if (items) {
                    for (var item of items) {
                        if (item.name && item.schema) {
                            this._map[item.name] = item.schema;
                        }
                    }
                }
            }
        }
    }

    getSchema(operator) {
        return this._map[operator];
    }

    getAttributeSchema(operator, name) {
        var map = this._attributeCache[operator];
        if (!map) {
            map = {};
            var schema = this.getSchema(operator);
            if (schema && schema.attributes && schema.attributes.length > 0) {
                for (var attribute of schema.attributes) {
                    map[attribute.name] = attribute;
                }
            }
            this._attributeCache[operator] = map;
        }
        return map[name] || null;
    }
};

dl4j.NDArrayReader = class {

    constructor(buffer) {
        var reader = new dl4j.BinaryReader(buffer);
        /* var shape = */ dl4j.NDArrayReader._header(reader);
        var data = dl4j.NDArrayReader._header(reader);
        this._dataType = data.type;
    }

    get dataType() {
        return this._dataType;
    }

    static _header(reader) {
        var header = {};
        header.alloc = reader.string();
        header.length = 0;
        switch (header.alloc) {
            case 'DIRECT':
            case 'HEAP':
            case 'JAVACPP':
                header.length = reader.int32();
                break;
            case 'LONG_SHAPE':
            case 'MIXED_DATA_TYPES':
                header.length = reader.int64();
                break;
        }
        header.type = reader.string();
        switch (header.type) {
            case 'INT':
                header.type = 'int32';
                header.itemsize = 4;
                break;
            case 'FLOAT':
                header.type = 'float32';
                header.itemsize = 4;
                break;
        }
        header.data = reader.bytes(header.itemsize * header.length);
        return header;
    }
}

dl4j.BinaryReader = class {

    constructor(buffer) {
        this._buffer = buffer;
        this._position = 0;
    }

    bytes(size) {
        var data = this._buffer.subarray(this._position, this._position + size);
        this._position += size;
        return data;
    }

    string() {
        var size = this._buffer[this._position++] << 8 | this._buffer[this._position++];
        var buffer = this.bytes(size);
        return new TextDecoder('ascii').decode(buffer);
    }

    int32() {
        return this._buffer[this._position++] << 24 | 
            this._buffer[this._position++] << 16 |
            this._buffer[this._position++] << 8 |
            this._buffer[this._position++];
    }

    int64() {
        var hi = this.int32();
        var lo = this.int32();
        return new long.Long(hi, lo, true).toNumber();
    }
}

dl4j.Error = class extends Error {
    constructor(message) {
        super(message);
        this.name = 'Error loading Deeplearning4j model.';
    }
};

if (typeof module !== 'undefined' && typeof module.exports === 'object') {
    module.exports.ModelFactory = dl4j.ModelFactory;
}