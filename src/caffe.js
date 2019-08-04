/* jshint esversion: 6 */
/* eslint "indent": [ "error", 4, { "SwitchCase": 1 } ] */

var caffe = caffe || {};
var long = long || { Long: require('long') };
var protobuf = protobuf || require('protobufjs');
var prototxt = prototxt || require('protobufjs/ext/prototxt');
var marked = marked || require('marked');

caffe.ModelFactory = class {

    match(context) {
        var identifier = context.identifier;
        var extension = identifier.split('.').pop().toLowerCase();
        var tags = null;
        if (extension == 'caffemodel') {
            return true;
        }
        if (extension == 'pbtxt' || extension == 'prototxt') {
            if (identifier == 'saved_model.pbtxt' || identifier == 'saved_model.prototxt' ||
                identifier.endsWith('predict_net.pbtxt') || identifier.endsWith('predict_net.prototxt') ||
                identifier.endsWith('init_net.pbtxt') || identifier.endsWith('init_net.prototxt')) {
                return false;
            }
            tags = context.tags('pbtxt');
            if (tags.has('layer') || tags.has('layers') || tags.has('net') || tags.has('train_net') || tags.has('net_param')) {
                return true;
            }
        }
        if (extension == 'pt') {
            // Reject PyTorch models
            var buffer = context.buffer;
            var torch = [ 0x8a, 0x0a, 0x6c, 0xfc, 0x9c, 0x46, 0xf9, 0x20, 0x6a, 0xa8, 0x50, 0x19 ];
            if (buffer && buffer.length > 14 && buffer[0] == 0x80 && torch.every((v, i) => v == buffer[i + 2])) {
                return false;
            }
            // Reject TorchScript models
            if (buffer && buffer.length > 2 && buffer[0] == 0x50 && buffer[1] == 0x4B) {
                return false;
            }
            tags = context.tags('pbtxt');
            if (tags.has('layer') || tags.has('layers') || tags.has('net') || tags.has('train_net') || tags.has('net_param')) {
                return true;
            }
        }
        return false;
    }

    open(context, host) {
        return host.require('./caffe-proto').then(() => {
            caffe.proto = protobuf.roots.caffe.caffe;
            return caffe.Metadata.open(host).then((metadata) => {
                var extension = context.identifier.split('.').pop();
                if (extension == 'pbtxt' || extension == 'prototxt' || extension == 'pt') {
                    var tags = context.tags('pbtxt');
                    if (tags.has('net') || tags.has('train_net') || tags.has('net_param')) {
                        try {
                            var reader = prototxt.TextReader.create(context.text);
                            reader.field = function(tag, message) {
                                if (message instanceof caffe.proto.SolverParameter) {
                                    message[tag] = this.skip();
                                    return;
                                }
                                throw new Error("Unknown field '" + tag + "'" + this.location());
                            };
                            var solver = caffe.proto.SolverParameter.decodeText(reader);
                            if (solver.net_param) {
                                return this._openNetParameter(metadata, solver.net_param, host);
                            }
                            else if (solver.net || solver.train_net) {
                                var file = solver.net || solver.train_net;
                                file = file.split('/').pop();
                                return context.request(file, 'utf-8').then((text) => {
                                    return this._openNetParameterText(metadata, context.identifier, text, host);
                                }).catch((error) => {
                                    if (error) {
                                        var message = error && error.message ? error.message : error.toString();
                                        message = message.endsWith('.') ? message.substring(0, message.length - 1) : message;
                                        throw new caffe.Error("Failed to load '" + file + "' (" + message + ").");
                                    }
                                });
                            }
                        }
                        catch (error) {
                            // continue regardless of error
                        }
                    }
                    return this._openNetParameterText(metadata, context.identifier, context.text, host);
                }
                else {
                    return this._openNetParameterBuffer(metadata, context.identifier, context.buffer, host);
                }
            });
        });
    }

    _openNetParameterBuffer(metadata, identifier, buffer, host, resolve, reject) {
        try {
            var netParameter = caffe.proto.NetParameter.decode(buffer);
            return this._openNetParameter(metadata, netParameter, host, resolve, reject);
        }
        catch (error) {
            throw new caffe.Error("File format is not caffe.NetParameter (" + error.message + ") in '" + identifier + "'.");
        }
    }

    _openNetParameterText(metadata, identifier, text, host) {
        try {
            var reader = prototxt.TextReader.create(text);
            reader.field = function(tag, message) {
                var type = message.constructor.name;
                if (tag.endsWith('_param') && (type == 'LayerParameter' || type == 'V1LayerParameter' || type == 'V0LayerParameter')) {
                    message[tag] = caffe.ModelFactory._decodeText(reader, true);
                    return;
                }  
                else if (message.constructor.name.endsWith('Parameter')) {
                    if (message[tag]) {
                        if (!Array.isArray(message[tag])) {
                            message[tag] = [ message[tag] ];
                        }
                        message[tag].push(this.skip());
                    }
                    else {
                        message[tag] = this.skip();
                    }
                    return;
                }
                throw new Error("Unknown field '" + tag + "'" + this.location());
            };
            reader.enum = function(type) {
                var token = this.read();
                if (!Object.prototype.hasOwnProperty.call(type, token)) {
                    var value = Number.parseInt(token, 10);
                    if (!Number.isNaN(token - value)) {
                        return value;
                    }
                    return token;
                }
                return type[token];
            };
            var netParameter = caffe.proto.NetParameter.decodeText(reader);
            return this._openNetParameter(metadata, netParameter, host);
        }
        catch (error) {
            throw new caffe.Error("File text format is not caffe.NetParameter (" + error.message + ") in '" + identifier + "'.");
        }
    }

    _openNetParameter(metadata, netParameter, host) {
        try {
            return new caffe.Model(metadata, netParameter);
        }
        catch (error) {
            host.exception(error, false);
            throw new caffe.Error(error.message);
        }
    }

    static _decodeText(reader, block) {
        var message = {};
        reader.start(block);
        while (!reader.end(block)) {
            var tag = reader.tag();
            if (message[tag]) {
                if (!Array.isArray(message[tag])) {
                    message[tag] = [ message[tag] ];
                }
                message[tag].push(reader.skip());
            }
            else {
                message[tag] = reader.skip();
            }
        }
        return message;
    }
};

caffe.Model = class {

    constructor(metadata, net) {

        this._name = net.name;

        if (net.layers && net.layers.length > 0) {
            if (net.layers.every((layer) => Object.prototype.hasOwnProperty.call(layer, 'layer'))) {
                this._version = 0;
                net.layer = net.layers;
            }
            else {
                this._version = 1;
                net.layer = net.layers;
            }
        }
        else if (net.layer && net.layer.length > 0) {
            this._version = 2;
        }

        this._graphs = [];

        var phases = new Set();
        for (var layer of net.layer) {
            for (var include of layer.include) {
                if (include.phase !== undefined) {
                    phases.add(include.phase);
                }
            }
        }
        if (phases.size === 0) {
            phases.add(-1);
        }

        for (var phase of phases) {
            this._graphs.push(new caffe.Graph(metadata, phase, net, this._version));
        }
    }

    get format() {
        return 'Caffe' + (this._version ? ' v' + this._version.toString() : '');
    }

    get graphs() {
        return this._graphs;
    }
};

caffe.Graph = class {

    constructor(metadata, phase, net, version) {

        switch (phase) {
            case 0: this._phase = 'TRAIN'; break;
            case 1: this._phase = 'TEST'; break;
            case -1: this._phase = ''; break;
            default: this._phase = phase.toString(); break;
        }

        this._nodes = [];
        this._inputs = [];
        this._outputs = [];

        var layer;
        for (layer of net.layer) {
            layer.input = layer.bottom.slice(0);
            layer.output = layer.top.slice(0);
            layer.chain = [];
        }

        var layers = [];
        for (layer of net.layer) {
            if (phase === -1 || layer.include.every((include) => include.phase === phase)) {
                layers.push(layer);
            }
        }

        var scope = {};
        var index = 0;
        for (layer of layers) {
            layer.input = layer.input.map((input) => scope[input] ? scope[input] : input);
            layer.output = layer.output.map((output) => {
                scope[output] = scope[output] ? output + '\n' + index.toString() : output; // custom argument id
                return scope[output];
            });
            index++;
        }

        // Graph Outputs
        var used = new Set();
        for (layer of layers) {
            for (input of layer.input) {
                used.add(input);
            }
        }
        var outputTops = [];
        for (layer of layers) {
            if (layer.input.length > 0) {
                for (var output of layer.output) {
                    if (!used.has(output)) {
                        outputTops.push(output);
                    }
                }
            }
        }
        for (var outputTop of outputTops) {
            this._outputs.push(new caffe.Parameter(outputTop, [ new caffe.Argument(outputTop, null) ]));
        }

        var nodes = [];
        var lastLayer = null;
        var lastTop = null;
        while (layers.length > 0) {
            layer = layers.shift();
            if (layer.output.length == 1 && layer.input.length == 1 && 
                layer.output[0].split('\n').shift() == layer.input[0].split('\n').shift() &&
                lastLayer &&
                lastTop == layer.output[0].split('\n').shift()) {
                lastLayer.chain = lastLayer.chain || [];
                lastLayer.chain.push(layer);
            }
            else {
                if (layer.type == 'Input' || layer.type == 'Data') {
                    if (layer.input.length == 0 && layer.output.length == 1 &&
                        layer.input_param && layer.input_param.shape &&
                        layer.input_param.shape.length == 1 && layer.input_param.shape[0].dim) {
                        var type = new caffe.TensorType(null, new caffe.TensorShape(layer.input_param.shape[0].dim));
                        this._inputs.push(new caffe.Parameter(layer.output[0], [ new caffe.Argument(layer.output[0], type) ]));
                        layer = null;
                    }
                }
                if (layer) {
                    nodes.push(layer);
                    lastLayer = null;
                    lastTop = null;
                    if (layer.output.length == 1) {
                        lastLayer = layer;
                        lastTop = layer.output[0].split('\n').shift();
                    }
                }
            }
        }

        var input;
        if (net.input && net.input.length > 0) {
            index = 0;
            for (input of net.input) {
                var inputType = null;
                if (net.input_shape && index < net.input_shape.length) {
                    var blobShape = net.input_shape[index];
                    if (blobShape && blobShape.dim) {
                        inputType = new caffe.TensorType(null, new caffe.TensorShape(blobShape.dim));
                    }
                }
                if (inputType == null && net.input.length == 1 && net.input_dim && net.input_dim.length > 0) {
                    inputType = new caffe.TensorType(null, new caffe.TensorShape(net.input_dim));
                }
                this._inputs.push(new caffe.Parameter(input, [ new caffe.Argument(input, inputType, null) ]));
                index++;
            }
        }

        for (layer of nodes) {
            var node = new caffe.Node(metadata, layer, version);
            if (layer.chain && layer.chain.length > 0) {
                for (var chain of layer.chain) {
                    node.chain.push(new caffe.Node(metadata, chain, version));
                }
            }
            this._nodes.push(node);
        }
    }

    get name() {
        return this._phase;
    }

    get type() {
        return '';
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

caffe.Parameter = class {

    constructor(name, args) {
        this._name = name;
        this._arguments = args;
    }

    get name() {
        return this._name;
    }

    get visible() {
        return true;
    }

    get arguments() {
        return this._arguments;
    }
};

caffe.Argument = class {
    constructor(id, type, initializer) {
        this._id = id;
        this._type = type || null;
        this._initializer = initializer || null;
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

caffe.Node = class {

    constructor(metadata, layer, version) {
        this._metadata = metadata;
        this._chain = [];

        switch (version) {
            case 0:
                this._name = layer.layer.name;
                this._type = layer.layer.type;
                break;
            case 1:
                this._name = layer.name;
                var typeIndex = layer.type;
                if (typeIndex === undefined) {
                    this._type = '?';
                }
                else {
                    if (!caffe.Node._operatorMap) {
                        caffe.Node._operatorMap = {};
                        var known = { 'BNLL': 'BNLL', 'HDF5': 'HDF5', 'LRN': 'LRN', 'RELU': 'ReLU', 'TANH': 'TanH', 'ARGMAX': 'ArgMax', 'MVN': 'MVN', 'ABSVAL': 'AbsVal' };
                        for (var key of Object.keys(caffe.proto.V1LayerParameter.LayerType)) {
                            var index = caffe.proto.V1LayerParameter.LayerType[key];
                            caffe.Node._operatorMap[index] = key.split('_').map((item) => {
                                return known[item] || item.substring(0, 1) + item.substring(1).toLowerCase();
                            }).join('');
                        }
                    }
                    this._type = caffe.Node._operatorMap[typeIndex] || typeIndex.toString();
                }
                break;
            case 2:
                this._name = layer.name;
                this._type = layer.type;
                break;
        }

        this._initializers = [];
        this._attributes = [];

        switch (version) {
            case 0:
                for (var attributeName of Object.keys(layer.layer)) {
                    if (attributeName != 'type' && attributeName != 'name' && attributeName != 'blobs' && attributeName != 'blobs_lr') {
                        this._attributes.push(new caffe.Attribute(this._metadata, this.operator, attributeName, layer.layer[attributeName]));
                    }
                }
                this._initializers = layer.layer.blobs.map((blob) => new caffe.Tensor(blob));
                break;
            case 1:
            case 2:
                for (var layer_kind of Object.keys(layer)) {
                    if (layer_kind.endsWith('_param') || layer_kind == 'transform_param') {
                        var param = layer[layer_kind];
                        var type = this._type;
                        if (type == 'Deconvolution') {
                            type = 'Convolution';
                        }
                        var prototype = Object.getPrototypeOf(param);
                        for (var name of Object.keys(param)) {
                            var defaultValue = prototype[name];
                            var value = param[name];
                            if (value != defaultValue && (!Array.isArray(value) || !Array.isArray(defaultValue) || value.length != 0 || defaultValue.length != 0)) {
                                this._attributes.push(new caffe.Attribute(this._metadata, this.operator, name, value));
                            }
                        }
                    }
                }
                if (layer.include && layer.include.length > 0) {
                    this._attributes.push(new caffe.Attribute(this._metadata, this.operator, 'include', layer.include));
                }
                if (layer.exclude && layer.exclude.length > 0) {
                    this._attributes.push(new caffe.Attribute(this._metadata, this.operator, 'exclude', layer.exclude));
                }
                if (this._type == 'Data' && layer.input_param && layer.input_param.shape) {
                    this._attributes.push(new caffe.Attribute(this._metadata, this.operator, 'shape', layer.input_param.shape));
                }
                var initializers = layer.blobs.map((blob) => new caffe.Tensor(blob));
                break;
        }

        var schema = this._metadata.getSchema(this.operator);

        this._inputs = [];
        var inputs = layer.input.concat(initializers);
        var inputIndex = 0;
        if (schema && schema.inputs) {
            for (var inputDef of schema.inputs) {
                if (inputIndex < inputs.length || inputDef.option != 'optional') {
                    var inputCount = (inputDef.option == 'variadic') ? (inputs.length - inputIndex) : 1;
                    var inputArguments = [];
                    for (var input of inputs.slice(inputIndex, inputIndex + inputCount)) {
                        if (input != '' || inputDef.option != 'optional') {
                            if (input instanceof caffe.Tensor) {
                                inputArguments.push(new caffe.Argument('', null, input));
                            }
                            else {
                                inputArguments.push(new caffe.Argument(input, null, null));
                            }
                        }
                    }
                    this._inputs.push(new caffe.Parameter(inputDef.name, inputArguments));
                    inputIndex += inputCount;
                }
            }
        }
        this._inputs = this._inputs.concat(inputs.slice(inputIndex).map((input) => {
            return new caffe.Parameter(inputIndex.toString(), [ 
                (input instanceof caffe.Tensor) ?
                    new caffe.Argument('', null, input) :
                    new caffe.Argument(input, null, null)
            ]);
        }));

        this._outputs = [];
        var outputs = layer.output;
        var outputIndex = 0;
        if (schema && schema.outputs) {
            for (var outputDef of schema.outputs) {
                if (outputIndex < outputs.length) {
                    var outputCount = (outputDef.option == 'variadic') ? (outputs.length - outputIndex) : 1;
                    this._outputs.push(new caffe.Parameter(outputDef.name, outputs.slice(outputIndex, outputIndex + outputCount).map((output) => {
                        return new caffe.Argument(output, null, null);
                    })));
                    outputIndex += outputCount;
                }
            }
        }
        this._outputs = this._outputs.concat(outputs.slice(outputIndex).map((output) => {
            return new caffe.Parameter(outputIndex.toString(), [
                new caffe.Argument(output, null, null)
            ]);
        }));
    }

    get operator() {
        return this._type;
    }

    get category() {
        var schema = this._metadata.getSchema(this._type);
        return (schema && schema.category) ? schema.category : '';
    }

    get documentation() {
        return '';
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
};

caffe.Attribute = class {

    constructor(metadata, operator, name, value) {
        this._name = name;
        this._value = value;

        if (value instanceof caffe.proto.BlobShape) {
            this._value = new caffe.TensorShape(value.dim);
        }

        var schema = metadata.getAttributeSchema(operator, this._name);
        if (schema) {
            if (Object.prototype.hasOwnProperty.call(schema, 'visible') && !schema.visible) {
                this._visible = false;
            }
            else if (Object.prototype.hasOwnProperty.call(schema, 'default')) {
                var defaultValue = schema.default;
                if (this._value == defaultValue) {
                    this._visible = false;
                }
                else if (Array.isArray(this._value) && Array.isArray(defaultValue)) {
                    if (this._value.length == defaultValue.length &&
                        this._value.every((item, index) => { return item == defaultValue[index]; })) {
                        this._visible = false;
                    }
                }
            }
        }
    }

    get name() {
        return this._name;
    }

    get value() {
        return this._value;
    }

    get visible() {
        return this._visible == false ? false : true;
    }
};

caffe.Tensor = class {

    constructor(blob) {
        this._blob = blob;

        var shape = [];
        if (Object.prototype.hasOwnProperty.call(blob, 'num') && 
            Object.prototype.hasOwnProperty.call(blob, 'channels') &&
            Object.prototype.hasOwnProperty.call(blob, 'width') &&
            Object.prototype.hasOwnProperty.call(blob, 'height')) {
            if (blob.num != 1) {
                shape.push(blob.num);
            }
            if (blob.channels != 1) {
                shape.push(blob.channels);
            }
            if (blob.width != 1) {
                shape.push(blob.width);
            }
            if (blob.height != 1) {
                shape.push(blob.height);
            }
        }
        else if (Object.prototype.hasOwnProperty.call(blob, 'shape')) {
            shape = blob.shape.dim;
        }

        var dataType = '?';
        if (blob.data.length > 0) {
            dataType = 'float32';
            this._data = blob.data;
        }
        else if (blob.double_data.length > 0) {
            dataType = 'float64';
            this._data = blob.double_data;
        }

        this._type = new caffe.TensorType(dataType, new caffe.TensorShape(shape));
    }

    get kind() {
        return 'Blob';
    }

    get type() {
        return this._type;
    }

    get state() {
        return this._context().state;
    }

    get value() {
        var context = this._context();
        if (context.state) {
            return null;
        }
        context.limit = Number.MAX_SAFE_INTEGER;
        return this._decode(context, 0);
    }

    toString() {
        var context = this._context();
        if (context.state) {
            return '';
        }
        context.limit = 10000;
        var value = this._decode(context, 0);
        return JSON.stringify(value, null, 4);
    }

    _context() {
        var context = {};
        context.state = null;
        context.index = 0;
        context.count = 0;
        context.data = this._data;
        context.dimensions = this.type.shape.dimensions;
        if (!this._data) {
            context.state = 'Tensor data is empty.';
        }
        return context;
    }

    _decode(context, dimension) {
        var results = [];
        var size = context.dimensions[dimension];
        if (dimension == context.dimensions.length - 1) {
            for (var i = 0; i < size; i++) {
                if (context.count > context.limit) {
                    results.push('...');
                    return results;
                }
                results.push(context.data[context.index]);
                context.index++;
                context.count++;
            }
        }
        else {
            for (var j = 0; j < size; j++) {
                if (context.count > context.limit) {
                    results.push('...');
                    return results;
                }
                results.push(this._decode(context, dimension + 1));
            }
        }
        return results;
    }
};

caffe.TensorType = class {

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

caffe.TensorShape = class {

    constructor(dimensions) {
        this._dimensions = dimensions.map((dimension) => {
            if (dimension && long.Long.isLong(dimension)) {
                return dimension.toNumber();
            }
            return dimension;
        });
    }

    get dimensions() {
        return this._dimensions;
    }

    toString() {
        return this._dimensions ? ('[' + this._dimensions.map((dimension) => dimension.toString()).join(',') + ']') : '';
    }
};

caffe.Metadata = class {

    static open(host) {
        if (caffe.Metadata._metadata) {
            return Promise.resolve(caffe.Metadata._metadata);
        }
        return host.request(null, 'caffe-metadata.json', 'utf-8').then((data) => {
            caffe.Metadata._metadata = new caffe.Metadata(data);
            return caffe.Metadata._metadata;
        }).catch(() => {
            caffe.Metadata._metadata = new caffe.Metadata(null);
            return caffe.Metadata._metadata;
        });
    }

    constructor(data) {
        this._map = {};
        this._attributeCache = {};
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

    getSchema(operator) {
        return this._map[operator] || null;
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

caffe.Error = class extends Error {
    constructor(message) {
        super(message);
        this.name = 'Error loading Caffe model.';
    }
};

if (typeof module !== 'undefined' && typeof module.exports === 'object') {
    module.exports.ModelFactory = caffe.ModelFactory;
}