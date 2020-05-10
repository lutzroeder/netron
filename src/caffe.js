/* jshint esversion: 6 */
/* eslint "indent": [ "error", 4, { "SwitchCase": 1 } ] */

var caffe = caffe || {};
var long = long || { Long: require('long') };
var protobuf = protobuf || require('protobufjs');
var prototxt = prototxt || require('protobufjs/ext/prototxt');

caffe.ModelFactory = class {

    match(context) {
        const identifier = context.identifier;
        const extension = identifier.split('.').pop().toLowerCase();
        if (extension == 'caffemodel') {
            return true;
        }
        if (extension == 'pbtxt' || extension == 'prototxt') {
            if (identifier == 'saved_model.pbtxt' || identifier == 'saved_model.prototxt' ||
                identifier.endsWith('predict_net.pbtxt') || identifier.endsWith('predict_net.prototxt') ||
                identifier.endsWith('init_net.pbtxt') || identifier.endsWith('init_net.prototxt')) {
                return false;
            }
            const tags = context.tags('pbtxt');
            if (tags.has('layer') || tags.has('layers') || tags.has('net') || tags.has('train_net') || tags.has('net_param')) {
                return true;
            }
        }
        if (extension == 'pt') {
            // Reject PyTorch models
            const buffer = context.buffer;
            const torch = [ 0x8a, 0x0a, 0x6c, 0xfc, 0x9c, 0x46, 0xf9, 0x20, 0x6a, 0xa8, 0x50, 0x19 ];
            if (buffer && buffer.length > 14 && buffer[0] == 0x80 && torch.every((v, i) => v == buffer[i + 2])) {
                return false;
            }
            // Reject TorchScript models
            if (buffer && buffer.length > 2 && buffer[0] == 0x50 && buffer[1] == 0x4B) {
                return false;
            }
            const tags = context.tags('pbtxt');
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
                const extension = context.identifier.split('.').pop();
                if (extension == 'pbtxt' || extension == 'prototxt' || extension == 'pt') {
                    const tags = context.tags('pbtxt');
                    if (tags.has('net') || tags.has('train_net') || tags.has('net_param')) {
                        try {
                            const reader = prototxt.TextReader.create(context.text);
                            reader.field = function(tag, message) {
                                if (message instanceof caffe.proto.SolverParameter) {
                                    message[tag] = this.skip();
                                    return;
                                }
                                throw new Error("Unknown field '" + tag + "'" + this.location());
                            };
                            const solver = caffe.proto.SolverParameter.decodeText(reader);
                            if (solver.net_param) {
                                return this._openNetParameter(metadata, solver.net_param, host);
                            }
                            else if (solver.net || solver.train_net) {
                                let file = solver.net || solver.train_net;
                                file = file.split('/').pop();
                                return context.request(file, 'utf-8').then((text) => {
                                    return this._openNetParameterText(metadata, context.identifier, text, host);
                                }).catch((error) => {
                                    if (error) {
                                        const message = error && error.message ? error.message : error.toString();
                                        throw new caffe.Error("Failed to load '" + file + "' (" + message.replace(/\.$/, '') + ").");
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
            const netParameter = caffe.proto.NetParameter.decode(buffer);
            return this._openNetParameter(metadata, netParameter, host, resolve, reject);
        }
        catch (error) {
            throw new caffe.Error("File format is not caffe.NetParameter (" + error.message + ") in '" + identifier + "'.");
        }
    }

    _openNetParameterText(metadata, identifier, text, host) {
        try {
            let reader = prototxt.TextReader.create(text);
            reader.field = function(tag, message) {
                const type = message.constructor.name;
                if (tag.endsWith('_param') && (type == 'LayerParameter' || type == 'V1LayerParameter' || type == 'V0LayerParameter')) {
                    message[tag] = caffe.ModelFactory._decodeText(reader);
                    return;
                }
                else if (message.constructor.name.endsWith('Parameter') || message.constructor.name === 'ParamSpec') {
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
                const token = this.read();
                if (!Object.prototype.hasOwnProperty.call(type, token)) {
                    const value = Number.parseInt(token, 10);
                    if (!Number.isNaN(token - value)) {
                        return value;
                    }
                    return token;
                }
                return type[token];
            };
            const netParameter = caffe.proto.NetParameter.decodeText(reader);
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

    static _decodeText(reader) {
        let message = {};
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            const value = reader.skip();
            if (!message[tag]) {
                message[tag] = value;
            }
            else {
                if (!Array.isArray(message[tag])) {
                    message[tag] = [ message[tag] ];
                }
                message[tag].push(value);
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

        let phases = new Set();
        for (const layer of net.layer) {
            for (const include of layer.include) {
                if (include.phase !== undefined) {
                    phases.add(include.phase);
                }
            }
        }
        if (phases.size === 0) {
            phases.add(-1);
        }

        for (const phase of phases) {
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

        for (const layer of net.layer) {
            layer.input = layer.bottom.slice(0);
            layer.output = layer.top.slice(0);
            layer.chain = [];
        }

        let layers = [];
        for (const layer of net.layer) {
            if (phase === -1 || layer.include.every((include) => include.phase === phase)) {
                layers.push(layer);
            }
        }

        let scope = {};
        let index = 0;
        for (const layer of layers) {
            layer.input = layer.input.map((input) => scope[input] ? scope[input] : input);
            layer.output = layer.output.map((output) => {
                scope[output] = scope[output] ? output + '\n' + index.toString() : output; // custom argument id
                return scope[output];
            });
            index++;
        }

        // Graph Inputs
        let usedOutputs = new Set();
        for (const layer of layers) {
            for (const output of layer.output) {
                usedOutputs.add(output);
            }
        }
        let unusedInputs = [];
        for (const layer of layers) {
            for (const input of layer.input) {
                if (!usedOutputs.has(input)) {
                    unusedInputs.push(input);
                }
            }
        }

        let nodes = [];
        let lastLayer = null;
        let lastTop = null;
        while (layers.length > 0) {
            let layer = layers.shift();
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
                        const type = new caffe.TensorType(null, new caffe.TensorShape(layer.input_param.shape[0].dim));
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

        if (net.input && net.input.length > 0) {
            index = 0;
            for (const input of net.input) {
                let inputType = null;
                if (net.input_shape && index < net.input_shape.length) {
                    const blobShape = net.input_shape[index];
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

        for (const layer of nodes) {
            const node = new caffe.Node(metadata, layer, version);
            if (layer.chain && layer.chain.length > 0) {
                for (const chain of layer.chain) {
                    node.chain.push(new caffe.Node(metadata, chain, version));
                }
            }
            this._nodes.push(node);
        }

        if (this._inputs.length === 0 && unusedInputs.length === 1) {
            this._inputs.push(new caffe.Parameter(unusedInputs[0], [
                new caffe.Argument(unusedInputs[0], null)
            ]));
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

    constructor(name, type, initializer) {
        if (typeof name !== 'string') {
            throw new caffe.Error("Invalid argument identifier '" + JSON.stringify(name) + "'.");
        }
        this._name = name;
        this._type = type || null;
        this._initializer = initializer || null;
    }

    get name() {
        return this._name;
    }

    get type() {
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
        this._attributes = [];

        switch (version) {
            case 0: {
                this._name = layer.layer.name;
                this._type = layer.layer.type;
                break;
            }
            case 1: {
                this._name = layer.name;
                const typeIndex = layer.type;
                if (typeIndex === undefined) {
                    this._type = '?';
                }
                else {
                    if (!caffe.Node._operatorMap) {
                        caffe.Node._operatorMap = {};
                        const known = { 'BNLL': 'BNLL', 'HDF5': 'HDF5', 'LRN': 'LRN', 'RELU': 'ReLU', 'TANH': 'TanH', 'ARGMAX': 'ArgMax', 'MVN': 'MVN', 'ABSVAL': 'AbsVal' };
                        for (const key of Object.keys(caffe.proto.V1LayerParameter.LayerType)) {
                            const index = caffe.proto.V1LayerParameter.LayerType[key];
                            caffe.Node._operatorMap[index] = key.split('_').map((item) => {
                                return known[item] || item.substring(0, 1) + item.substring(1).toLowerCase();
                            }).join('');
                        }
                    }
                    this._type = caffe.Node._operatorMap[typeIndex] || typeIndex.toString();
                }
                break;
            }
            case 2:
                this._name = layer.name;
                this._type = layer.type;
                break;
        }

        let initializers = [];

        switch (version) {
            case 0:
                for (const attributeName of Object.keys(layer.layer)) {
                    if (attributeName != 'type' && attributeName != 'name' && attributeName != 'blobs' && attributeName != 'blobs_lr') {
                        this._attributes.push(new caffe.Attribute(this._metadata, this.operator, attributeName, layer.layer[attributeName]));
                    }
                }
                initializers = layer.layer.blobs.map((blob) => new caffe.Tensor(blob));
                break;
            case 1:
            case 2:
                for (const layer_kind of Object.keys(layer)) {
                    if (layer_kind.endsWith('_param') || layer_kind == 'transform_param') {
                        const param = layer[layer_kind];
                        let type = this._type;
                        if (type == 'Deconvolution') {
                            type = 'Convolution';
                        }
                        const prototype = Object.getPrototypeOf(param);
                        for (const name of Object.keys(param)) {
                            const defaultValue = prototype[name];
                            const value = param[name];
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
                initializers = layer.blobs.map((blob) => new caffe.Tensor(blob));
                break;
        }

        const schema = this._metadata.type(this.operator);

        this._inputs = [];
        const inputs = layer.input.concat(initializers);
        let inputIndex = 0;
        if (schema && schema.inputs) {
            for (const inputDef of schema.inputs) {
                if (inputIndex < inputs.length || inputDef.option != 'optional') {
                    const inputCount = inputDef.option == 'variadic' ? inputs.length - inputIndex : 1;
                    this._inputs.push(new caffe.Parameter(inputDef.name, inputs.slice(inputIndex, inputIndex + inputCount).filter((input) => input !== '' || inputDef.option != 'optional').map((input) => {
                        return input instanceof caffe.Tensor ? new caffe.Argument('', input.type, input) : new caffe.Argument(input, null, null);
                    })));
                    inputIndex += inputCount;
                }
            }
        }
        this._inputs = this._inputs.concat(inputs.slice(inputIndex).map((input) => {
            return new caffe.Parameter(inputIndex.toString(), [
                input instanceof caffe.Tensor ? new caffe.Argument('', input.type, input) : new caffe.Argument(input, null, null)
            ]);
        }));

        this._outputs = [];
        const outputs = layer.output;
        let outputIndex = 0;
        if (schema && schema.outputs) {
            for (const outputDef of schema.outputs) {
                if (outputIndex < outputs.length) {
                    let outputCount = (outputDef.option == 'variadic') ? (outputs.length - outputIndex) : 1;
                    this._outputs.push(new caffe.Parameter(outputDef.name, outputs.slice(outputIndex, outputIndex + outputCount).map((output) => {
                        return new caffe.Argument(output, null, null);
                    })));
                    outputIndex += outputCount;
                }
            }
        }
        this._outputs = this._outputs.concat(outputs.slice(outputIndex).map((output, index) => {
            return new caffe.Parameter((outputIndex + index).toString(), [
                new caffe.Argument(output, null, null)
            ]);
        }));
    }

    get operator() {
        return this._type;
    }

    get metadata() {
        return this._metadata.type(this._type);
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

        const schema = metadata.attribute(operator, this._name);
        if (schema) {
            if (Object.prototype.hasOwnProperty.call(schema, 'visible') && !schema.visible) {
                this._visible = false;
            }
            else if (Object.prototype.hasOwnProperty.call(schema, 'default')) {
                const defaultValue = schema.default;
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

        let shape = [];
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

        let dataType = '?';
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
        let context = this._context();
        if (context.state) {
            return null;
        }
        context.limit = Number.MAX_SAFE_INTEGER;
        return this._decode(context, 0);
    }

    toString() {
        let context = this._context();
        if (context.state) {
            return '';
        }
        context.limit = 10000;
        const value = this._decode(context, 0);
        return JSON.stringify(value, null, 4);
    }

    _context() {
        let context = {};
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
        let results = [];
        const size = context.dimensions[dimension];
        if (dimension == context.dimensions.length - 1) {
            for (let i = 0; i < size; i++) {
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
            for (let j = 0; j < size; j++) {
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
            let items = JSON.parse(data);
            if (items) {
                for (const item of items) {
                    if (item.name && item.schema) {
                        item.schema.name = item.name;
                        this._map[item.name] = item.schema;
                    }
                }
            }
        }
    }

    type(operator) {
        return this._map[operator] || null;
    }

    attribute(operator, name) {
        let map = this._attributeCache[operator];
        if (!map) {
            map = {};
            const schema = this.type(operator);
            if (schema && schema.attributes && schema.attributes.length > 0) {
                for (const attribute of schema.attributes) {
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