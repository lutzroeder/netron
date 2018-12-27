/*jshint esversion: 6 */

var caffe = caffe || {};
var protobuf = protobuf || require('protobufjs');
var marked = marked || require('marked');

caffe.ModelFactory = class {

    match(context, host) {
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
            if (tags.layer || tags.layers || tags.net || tags.train_net || tags.net_param) {
                return true;
            }
        }
        return false;
    }

    open(context, host, callback) { 
        host.require('./caffe-proto', (err, module) => {
            if (err) {
                callback(err, null);
                return;
            }
            caffe.proto = protobuf.roots.caffe.caffe;
            caffe.Metadata.open(host, (err, metadata) => {
                var extension = context.identifier.split('.').pop();
                if (extension == 'pbtxt' || extension == 'prototxt') {
                    var tags = context.tags('pbtxt');
                    if (tags.net || tags.train_net || tags.net_param) {
                        try {
                            var reader = new protobuf.TextReader(context.text);
                            reader.handle = function(tag, message) {
                                if (message instanceof caffe.proto.SolverParameter) {
                                    message[tag] = this.skip();
                                    return;
                                }
                                throw new Error("Unknown field '" + tag + "'" + this.location());
                            };
                            var solver = caffe.proto.SolverParameter.decodeText(reader);
                            if (solver.net_param) {
                                this._openNetParameter(metadata, solver.net_param, host, callback);
                                return;
                            }
                            else if (solver.net || solver.train_net) {
                                var file = solver.net || solver.train_net;
                                file = file.split('/').pop();
                                context.request(file, 'utf-8', (err, text) => {
                                    if (err) {
                                        var message = err && err.message ? err.message : err.toString();
                                        message = message.endsWith('.') ? message.substring(0, message.length - 1) : message;
                                        callback(new caffe.Error("Failed to load '" + file + "' (" + message + ")."), null);
                                        return;
                                    }
                                    this._openNetParameterText(metadata, context.identifier, text, host, callback);
                                });
                                return;
                            }
                        }
                        catch (error) {
                        }
                    }
                    this._openNetParameterText(metadata, context.identifier, context.text, host, callback);
                }
                else {
                    this._openNetParameterBuffer(metadata, context.identifier, context.buffer, host, callback);
                }
            });
        });
    }

    _openNetParameterBuffer(metadata, identifier, buffer, host, callback) {
        try {
            var netParameter = caffe.proto.NetParameter.decode(buffer);
            this._openNetParameter(metadata, netParameter, host, callback);
        }
        catch (error) {
            callback(new caffe.Error("File format is not caffe.NetParameter (" + error.message + ") in '" + identifier + "'."), null);
            return;
        }
    }

    _openNetParameterText(metadata, identifier, text, host, callback) {
        try {
            var reader = new protobuf.TextReader(text);
            reader.handle = function(tag, message) {
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
                this.assert(":");
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
            this._openNetParameter(metadata, netParameter, host, callback);
        }
        catch (error) {
            callback(new caffe.Error("File text format is not caffe.NetParameter (" + error.message + ") in '" + identifier + "'."), null);
        }
    }

    _openNetParameter(metadata, netParameter, host, callback) {
        try {
            var model = new caffe.Model(metadata, netParameter);
            callback(null, model);
        }
        catch (error) {
            host.exception(error, false);
            callback(new caffe.Error(error.message), null);
            return;
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

    constructor(metadata, netParameter) {
        this._name = netParameter.name;
        if (netParameter.layers && netParameter.layers.length > 0) {
            if (netParameter.layers.every((layer) => layer.hasOwnProperty('layer'))) {
                this._version = 0;
            }
            else {
                this._version = 1;
            }
        }
        else if (netParameter.layer && netParameter.layer.length > 0) {
            this._version = 2;
        }
        var graph = new caffe.Graph(metadata, netParameter, this._version);
        this._graphs = [ graph ];
    }

    get format() {
        return 'Caffe' + (this.hasOwnProperty('_version') ? ' v' + this._version.toString() : '');
    }

    get graphs() {
        return this._graphs;
    }
};

caffe.Graph = class {

    constructor(metadata, netParameter, version)
    {
        this._nodes = [];
        this._inputs = [];
        this._outputs = [];
        this._operators = {};

        var layers = [];
        switch (version) {
            case 0:
            case 1:
                layers = netParameter.layers;
                break;
            case 2:
                layers = netParameter.layer;
                break;
        }

        var scope = {};
        layers.forEach((layer, index) => {
            layer.bottom = layer.bottom.map((input) => scope[input] ? scope[input] : input);
            layer.top = layer.top.map((output) => {
                if (scope[output]) {
                    var next = output + '\n' + index.toString(); // custom connection id
                    scope[output] = next;
                    return next;
                }
                scope[output] = output;   
                return output;
            });
        });

        var lastNode = null;
        var lastTop = null;
        layers.forEach((layer) => {
            var node = new caffe.Node(metadata, layer, version);
            this._operators[node.operator] = (this._operators[node.operator] || 0) + 1;
            if (layer.top.length == 1 && 
                layer.bottom.length >= 1 && 
                layer.top[0].split('\n').shift() == layer.bottom[0].split('\n').shift() &&
                lastNode &&
                lastTop == layer.top[0].split('\n').shift()) {
                lastNode.chain.push(node);
            }
            else if (!this.translateInput(node)) {
                this._nodes.push(node);
                lastNode = null;
                lastTop = null;
                if (layer.top.length == 1) {
                    lastNode = node;
                    lastTop = layer.top[0].split('\n').shift();
                }
            }
        });

        if (netParameter.input && netParameter.input.length > 0) {
            netParameter.input.forEach((input, index) => {
                var inputType = null;
                if (netParameter.input_shape && index < netParameter.input_shape.length) {
                    var blobShape = netParameter.input_shape[index];
                    if (blobShape && blobShape.dim) {
                        inputType = new caffe.TensorType(null, new caffe.TensorShape(blobShape.dim));
                    }
                }
                if (inputType == null && netParameter.input.length == 1 && netParameter.input_dim && netParameter.input_dim.length > 0) {
                    inputType = new caffe.TensorType(null, new caffe.TensorShape(netParameter.input_dim));
                }
                this._inputs.push(new caffe.Argument(input, [ new caffe.Connection(input, inputType, null) ]));
            });
        }

        if (this._outputs.length == 0) {
            var nodeMap = {};
            var countMap = {};
            var outputs = [];
            this._nodes.forEach((node) => {
                if (node._outputs.length == 0) {
                    outputs.push(node);
                }
                else {
                    node._outputs.forEach((output) => {
                        nodeMap[output] = node;
                    });
                }
                node._inputs.forEach((input) => {
                    if (countMap[input]) {
                        countMap[input]++;
                    }
                    else {
                        countMap[input] = 1;
                    }
                });
            });
            Object.keys(nodeMap).forEach((output) => {
                if (countMap[output]) {
                    delete nodeMap[output];
                }
            });
            var keys = Object.keys(nodeMap);
            if (keys.length == 1) {
                this._outputs.push(new caffe.Argument(keys[0], [ new caffe.Connection(keys[0], null) ]));
            }
            else if (outputs.length == 1) {
                outputs[0]._outputs = [ 'output' ];
                this._outputs.push(new caffe.Argument('output', [ new caffe.Connection('output', null) ]));
            }
        }
    }

    get operators() {
        return this._operators;
    }

    get name() {
        return this._name;
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

    translateInput(node) {
        if (node.operator == 'Input' || node.operator == 'Data') {
            if (node._inputs.length == 0 && node._outputs.length == 1) {
                var attributes = node.attributes;
                if (attributes.length == 1) {
                    var attribute = attributes[0];
                    if (attribute.name == 'shape') {
                        if (attribute._value.length == 1 && attribute._value[0].dim) {
                            var input = node._outputs[0];
                            var type = new caffe.TensorType(null, attribute._value[0].dim);
                            this._inputs.push(new caffe.Argument(input, [ new caffe.Connection(input, type) ]));
                            return true;
                        }
                    }
                }
            }
        }
        return false;
    }
};

caffe.Argument = class {

    constructor(name, connections) {
        this._name = name;
        this._connections = connections;
    }

    get name() {
        return this._name;
    }

    get visible() {
        return true;
    }

    get connections() {
        return this._connections;
    }
};

caffe.Connection = class {
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
                        Object.keys(caffe.proto.V1LayerParameter.LayerType).forEach((key) => {
                            var index = caffe.proto.V1LayerParameter.LayerType[key];
                            caffe.Node._operatorMap[index] = key.split('_').map((item) => {
                                return known[item] || item.substring(0, 1) + item.substring(1).toLowerCase();
                            }).join('');
                        });
                    }
                    this._type = caffe.Node._operatorMap[typeIndex] || typeIndex.toString();
                }
                break;
            case 2:
                this._name = layer.name;
                this._type = layer.type;
                break;
        }

        this._inputs = layer.bottom;
        this._outputs = layer.top;
        this._initializers = [];
        this._attributes = [];

        switch (version) {
            case 0:
                Object.keys(layer.layer).forEach((attributeName) => {
                    if (attributeName != 'type' && attributeName != 'name' && attributeName != 'blobs' && attributeName != 'blobs_lr') {
                        var attributeValue = layer.layer[attributeName];
                        this._attributes.push(new caffe.Attribute(this._metadata, this.operator, attributeName, attributeValue));
                    }
                });
                layer.layer.blobs.forEach((blob) => {
                    this._initializers.push(new caffe.Tensor(blob));
                });
                break;
            case 1:
            case 2:
                Object.keys(layer).forEach((key) => {
                    if (key.endsWith('_param') || key == 'transform_param') {
                        var param = layer[key];
                        var type = this._type;
                        if (type == 'Deconvolution') {
                            type = 'Convolution';
                        }
                        var prototype = Object.getPrototypeOf(param);
                        Object.keys(param).forEach((name) => {
                            var defaultValue = prototype[name];
                            var value = param[name];
                            if (value != defaultValue && (!Array.isArray(value) || !Array.isArray(defaultValue) || value.length != 0 || defaultValue.length != 0)) {
                                this._attributes.push(new caffe.Attribute(this._metadata, this.operator, name, value));
                            }
                        });
                    }
                });
                if (layer.include && layer.include.length > 0) {
                    this._attributes.push(new caffe.Attribute(this._metadata, this.operator, 'include', layer.include));
                }
                if (layer.exclude && layer.exclude.length > 0) {
                    this._attributes.push(new caffe.Attribute(this._metadata, this.operator, 'exclude', layer.exclude));
                }
                if (this._type == 'Data' && layer.input_param && layer.input_param.shape) {
                    this._attributes.push(new caffe.Attribute(this._metadata, this.operator, 'shape', layer.input_param.shape));
                }
                layer.blobs.forEach((blob) => {
                    this._initializers.push(new caffe.Tensor(blob));
                });
                break;
        }
    }

    get operator() {
        return this._type;
    }

    get category() {
        var schema = this._metadata.getSchema(this._type);
        return (schema && schema.category) ? schema.category : null;
    }

    get name() { 
        return this._name;
    }

    get inputs() {
        var inputs = this._inputs.concat(this._initializers);
        var args = [];
        var index = 0;
        var schema = this._metadata.getSchema(this.operator);
        if (schema && schema.inputs) {
            schema.inputs.forEach((inputDef) => {
                if (index < inputs.length || inputDef.option != 'optional') {
                    var connections = [];
                    var count = (inputDef.option == 'variadic') ? (inputs.length - index) : 1;
                    inputs.slice(index, index + count).forEach((input) => {
                        if (input != '' || inputDef.option != 'optional') {
                            if (input instanceof caffe.Tensor) {
                                connections.push(new caffe.Connection('', null, input));
                            }
                            else {
                                connections.push(new caffe.Connection(input, null, null));
                            }
                        }
                    });
                    index += count;
                    args.push(new caffe.Argument(inputDef.name, connections));
                }
            });
        }
        else {
            inputs.slice(index).forEach((input) => {
                var connection = null;
                if (input instanceof caffe.Tensor) {
                    connection = new caffe.Connection('', null, input);
                }
                else {
                    connection = new caffe.Connection(input, null, null);
                }
                args.push(new caffe.Argument(index.toString(), [ connection ]));
                index++;
            });
        }
        return args;
    }

    get outputs() {
        var args = [];
        var index = 0;
        var outputs = this._outputs;
        var schema = this._metadata.getSchema(this.operator);
        if (schema && schema.outputs) {
            schema.outputs.forEach((outputDef) => {
                if (index < outputs.length || outputDef.option != 'optional') {
                    var count = (outputDef.option == 'variadic') ? (outputs.length - index) : 1;
                    var connections = [];
                    outputs.slice(index, index + count).forEach((output) => {
                        connections.push(new caffe.Connection(output, null, null));
                    });
                    index += count;
                    args.push(new caffe.Argument(outputDef.name, connections));
                }
            });
        }
        else {
            outputs.slice(index).forEach((output) => {
                args.push(new caffe.Argument(index.toString(), [
                    new caffe.Connection(output, null, null)
                ]));
                index++;
            });
        }
        return args;
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
            this._value = () => {
                return JSON.stringify(value.dim);
            };
        }

        var schema = metadata.getAttributeSchema(operator, this._name);
        if (schema) {
            if (schema.hasOwnProperty('visible') && !schema.visible) {
                this._visible = false;
            }
            else if (schema.hasOwnProperty('default')) {
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
        if (blob.hasOwnProperty('num') && blob.hasOwnProperty('channels') &&
            blob.hasOwnProperty('width') && blob.hasOwnProperty('height')) {
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
        else if (blob.hasOwnProperty('shape')) {
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
            if (dimension && dimension.__isLong__) {
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

    static open(host, callback) {
        if (caffe.Metadata._metadata) {
            callback(null, caffe.Metadata._metadata);
        }
        else {
            host.request(null, 'caffe-metadata.json', 'utf-8', (err, data) => {
                caffe.Metadata._metadata = new caffe.Metadata(data);
                callback(null, caffe.Metadata._metadata);
            });
        }
    }

    constructor(data) {
        this._map = {};
        if (data) {
            var items = JSON.parse(data);
            if (items) {
                items.forEach((item) => {
                    if (item.name && item.schema) {
                        this._map[item.name] = item.schema;
                    }
                });
            }
        }
    }

    getSchema(operator) {
        return this._map[operator] || null;
    }

    getAttributeSchema(operator, name) {
        var schema = this.getSchema(operator);
        if (schema && schema.attributes && schema.attributes.length > 0) {
            if (!schema.attributesMap) {
                schema.attributesMap = {};
                schema.attributes.forEach((attribute) => {
                    schema.attributesMap[attribute.name] = attribute;
                });
            }
            return schema.attributesMap[name] || null;
        }
        return null;
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