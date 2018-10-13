/*jshint esversion: 6 */

var caffe = null;

class CaffeModelFactory {

    match(context, host) {
        var identifier = context.identifier;
        var extension = identifier.split('.').pop();
        if (extension == 'caffemodel') {
            return true;
        }
        if (extension == 'pbtxt' || extension == 'prototxt') {
            if (identifier == 'saved_model.pbtxt' || identifier == 'saved_model.prototxt') {
                return false;
            }
            if (identifier.endsWith('predict_net.pbtxt') || identifier.endsWith('predict_net.prototxt')) {
                return false;
            }
            if (context.text) {
                var lines = context.text.split('\n');
                    if (lines.some((line) => 
                        (line.startsWith('ir_version') && line.replace(/\s+/g, '').startsWith('ir_version:')) ||
                        (line.startsWith('graph_def') && line.replace(/\s+/g, '').startsWith('graph_def{')) || 
                        (line.startsWith('op') && line.replace(/\s+/g, '').startsWith('op{')) || 
                        (line.startsWith('node') && line.replace(/\s+/g, '').startsWith('node{')))) {
                    return false;
                }
            }
            return host.environment('PROTOTXT') ? true : false;
        }
        return false;
    }

    open(context, host, callback) { 
        host.require('caffe', (err) => {
            if (err) {
                callback(err, null);
                return;
            }
            caffe = protobuf.roots.caffe.caffe;
            CaffeOperatorMetadata.open(host, (err, metadata) => {
                var extension = context.identifier.split('.').pop();
                if (extension == 'pbtxt' || extension == 'prototxt') {
                    var text = context.text;
                    var lines = context.text.split('\n');
                    if (lines.some((line) => 
                            (line.startsWith('net') && line.replace(/\s+/g, '').startsWith('net:')) ||
                            (line.startsWith('train_net') && line.replace(/\s+/g, '').startsWith('train_net:')) ||
                            (line.startsWith('net_param') && line.replace(/\s+/g, '').startsWith('net_param{')))) {
                        try { 
                            var solver = caffe.SolverParameter.decodeText(context.text);
                            if (solver.net_param) {
                                this._openNetParameter(solver.net_param, callback);
                                return;
                            }
                            else if (solver.net || solver.train_net) {
                                var file = solver.net || solver.train_net;
                                file = file.split('/').pop();
                                context.request(file, 'utf-8', (err, text) => {
                                    if (err) {
                                        var message = err && err.message ? err.message : err.toString();
                                        message = message.endsWith('.') ? message.substring(0, message.length - 1) : message;
                                        callback(new CaffeError("Failed to load '" + file + "' (" + message + ")."), null);
                                        return;
                                    }
                                    this._openNetParameterText(text, callback);
                                });
                                return;
                            }
                        }
                        catch (error) {
                        }
                }
                    this._openNetParameterText(context.text, callback);
                }
                else {
                    this._openNetParameterBuffer(context.buffer, callback);
                }
            });
        });
    }

    _openNetParameterBuffer(buffer, callback) {
        try {
            var netParameter = caffe.NetParameter.decode(buffer);
            this._openNetParameter(netParameter, callback);
        }
        catch (error) {
            host.exception(error, false);
            callback(new CaffeError('File format is not caffe.NetParameter (' + error.message + ').'), null);
            return;    
        }

    }

    _openNetParameterText(text, callback) {
        try {
            var netParameter = caffe.NetParameter.decodeText(text);
            this._openNetParameter(netParameter, callback);
        }
        catch (error) {
            host.exception(error, false);
            callback(new CaffeError('File text format is not caffe.NetParameter (' + error.message + ').'), null);
        }
    }

    _openNetParameter(netParameter, callback) {
        try {
            var model = new CaffeModel(netParameter);
            callback(null, model);
        }
        catch (error) {
            host.exception(error, false);
            callback(new CaffeError(error.message), null);
            return;
        }
    }
}

class CaffeModel {

    constructor(netParameter) {
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
        var graph = new CaffeGraph(netParameter, this._version);
        this._graphs = [ graph ];
    }

    get format() {
        return 'Caffe' + (this.hasOwnProperty('_version') ? ' v' + this._version.toString() : '');
    }

    get graphs() {
        return this._graphs;
    }

}

class CaffeGraph {

    constructor(netParameter, version)
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

        layers.forEach((layer) => {
            var node = new CaffeNode(layer, version);
            this._operators[node.operator] = (this._operators[node.operator] || 0) + 1;
            if (!this.translateInput(node)) {
                this._nodes.push(node);
            }
        });

        if (netParameter.input && netParameter.input.length > 0) {
            netParameter.input.forEach((input, index) => {
                var inputType = null;
                if (netParameter.input_shape && index < netParameter.input_shape.length) {
                    var blobShape = netParameter.input_shape[index];
                    if (blobShape && blobShape.dim) {
                        inputType = new CaffeTensorType(null, blobShape.dim);
                    }
                }
                if (inputType == null && netParameter.input.length == 1 && netParameter.input_dim && netParameter.input_dim.length > 0) {
                    inputType = new CaffeTensorType(null, netParameter.input_dim);
                }
                this._inputs.push(new CaffeArgument(input, [ new CaffeConnection(input, inputType, null) ]));
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
                this._outputs.push(new CaffeArgument(keys[0], [ new CaffeConnection(keys[0], null) ]));
            }
            else if (outputs.length == 1) {
                outputs[0]._outputs = [ 'output' ];
                this._outputs.push(new CaffeArgument('output', [ new CaffeConnection('output', null) ]));
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
                            var type = new CaffeTensorType(null, attribute._value[0].dim);
                            this._inputs.push(new CaffeArgument(input, [ new CaffeConnection(input, type) ]));
                            return true;
                        }
                    }
                }
            }
        }
        return false;
    }
}

class CaffeArgument {
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
}

class CaffeConnection {
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
}

class CaffeNode {

    constructor(layer, version) {

        switch (version) {
            case 0:
                this._type = layer.layer.type;
                this._name = layer.layer.name;
                break;
            case 1:
                this._type = CaffeNode.getOperator(layer.type);
                this._name = layer.name;
                break;
            case 2:
                this._type = layer.type;
                this._name = layer.name;
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
                        this._attributes.push(new CaffeAttribute(this.operator, attributeName, attributeValue));
                    }
                });
                layer.layer.blobs.forEach((blob) => {
                    this._initializers.push(new CaffeTensor(blob));
                });
                break;
            case 1:
            case 2:
                Object.keys(layer).forEach((key) => {
                    if (key.endsWith('_param')) {
                        var param = layer[key];
                        var type = this._type;
                        if (type == 'Deconvolution') {
                            type = 'Convolution';
                        }
                        if (param.constructor.name == type + 'Parameter' || key == 'transform_param') {
                            var prototype = Object.getPrototypeOf(param);
                            Object.keys(param).forEach((name) => {
                                var defaultValue = prototype[name];
                                var value = param[name];
                                if (value != defaultValue && (!Array.isArray(value) || !Array.isArray(defaultValue) || value.length != 0 || defaultValue.length != 0)) {
                                    this._attributes.push(new CaffeAttribute(this.operator, name, value));
                                }
                            });
                        }
                    }
                });
                if (layer.include && layer.include.length > 0) {
                    this._attributes.push(new CaffeAttribute(this.operator, 'include', layer.include));
                }
                if (layer.exclude && layer.exclude.length > 0) {
                    this._attributes.push(new CaffeAttribute(this.operator, 'exclude', layer.exclude));
                }
                if (this._type == 'Data' && layer.input_param && layer.input_param.shape) {
                    this._attributes.push(new CaffeAttribute(this.operator, 'shape', layer.input_param.shape));
                }
                layer.blobs.forEach((blob) => {
                    this._initializers.push(new CaffeTensor(blob));
                });
                break;
        }
    }

    get operator() {
        return this._type;
    }

    get category() {
        var schema = CaffeOperatorMetadata.operatorMetadata.getSchema(this._type);
        return (schema && schema.category) ? schema.category : null;
    }

    get name() { 
        return this._name;
    }

    get inputs() {
        var list = this._inputs.concat(this._initializers);
        var inputs = CaffeOperatorMetadata.operatorMetadata.getInputs(this._type, list);
        return inputs.map((input) => {
            return new CaffeArgument(input.name, input.connections.map((connection) => {
                if (connection.id instanceof CaffeTensor) {
                    return new CaffeConnection('', null, connection.id);
                }
                return new CaffeConnection(connection.id, null, null);
            }));
        });
    }

    get outputs() {
        var outputs = CaffeOperatorMetadata.operatorMetadata.getOutputs(this._type, this._outputs);
        return outputs.map((output) => {
            return new CaffeArgument(output.name, output.connections.map((connection) => {
                return new CaffeConnection(connection.id, null, null);
            }));
        });
    }

    get attributes() {
        return this._attributes;
    }

    static getOperator(index) {
        if (!CaffeNode._operatorMap) {
            CaffeNode._operatorMap = {};
            CaffeNode._operatorMap[caffe.V1LayerParameter.LayerType.NONE] = 'None';
            CaffeNode._operatorMap[caffe.V1LayerParameter.LayerType.ACCURACY] = 'Accuracy';
            CaffeNode._operatorMap[caffe.V1LayerParameter.LayerType.BNLL] = 'BNLL';
            CaffeNode._operatorMap[caffe.V1LayerParameter.LayerType.CONCAT] = 'Concat'; 
            CaffeNode._operatorMap[caffe.V1LayerParameter.LayerType.CONVOLUTION] = 'Convolution';
            CaffeNode._operatorMap[caffe.V1LayerParameter.LayerType.DATA] = 'Data';
            CaffeNode._operatorMap[caffe.V1LayerParameter.LayerType.DROPOUT] = 'Dropout';
            CaffeNode._operatorMap[caffe.V1LayerParameter.LayerType.EUCLIDEAN_LOSS] = 'EuclideanLoss';
            CaffeNode._operatorMap[caffe.V1LayerParameter.LayerType.FLATTEN] = 'Flatten';
            CaffeNode._operatorMap[caffe.V1LayerParameter.LayerType.HDF5_DATA] = 'HDF5Data';
            CaffeNode._operatorMap[caffe.V1LayerParameter.LayerType.HDF5_OUTPUT] = 'HDF5Output';
            CaffeNode._operatorMap[caffe.V1LayerParameter.LayerType.IM2COL] = 'Im2col';
            CaffeNode._operatorMap[caffe.V1LayerParameter.LayerType.IMAGE_DATA] = 'ImageData';
            CaffeNode._operatorMap[caffe.V1LayerParameter.LayerType.INFOGAIN_LOSS] = 'InfogainLoss';
            CaffeNode._operatorMap[caffe.V1LayerParameter.LayerType.INNER_PRODUCT] = 'InnerProduct';
            CaffeNode._operatorMap[caffe.V1LayerParameter.LayerType.LRN] = 'LRN';
            CaffeNode._operatorMap[caffe.V1LayerParameter.LayerType.MULTINOMIAL_LOGISTIC_LOSS] = 'MultinomialLogisticLoss';
            CaffeNode._operatorMap[caffe.V1LayerParameter.LayerType.POOLING] = 'Pooling';
            CaffeNode._operatorMap[caffe.V1LayerParameter.LayerType.RELU] = 'ReLU';
            CaffeNode._operatorMap[caffe.V1LayerParameter.LayerType.SIGMOID] = 'Sigmoid';
            CaffeNode._operatorMap[caffe.V1LayerParameter.LayerType.SOFTMAX] = 'Softmax';
            CaffeNode._operatorMap[caffe.V1LayerParameter.LayerType.SOFTMAX_LOSS] = 'SoftmaxLoss';
            CaffeNode._operatorMap[caffe.V1LayerParameter.LayerType.SPLIT] = 'Split';
            CaffeNode._operatorMap[caffe.V1LayerParameter.LayerType.TANH] = 'TanH';
            CaffeNode._operatorMap[caffe.V1LayerParameter.LayerType.WINDOW_DATA] = 'WindowData';
            CaffeNode._operatorMap[caffe.V1LayerParameter.LayerType.ELTWISE] = 'Eltwise';
            CaffeNode._operatorMap[caffe.V1LayerParameter.LayerType.POWER] = 'Power';
            CaffeNode._operatorMap[caffe.V1LayerParameter.LayerType.SIGMOID_CROSS_ENTROPY_LOSS] = 'SigmoidCrossEntropyLoss';
            CaffeNode._operatorMap[caffe.V1LayerParameter.LayerType.HINGE_LOSS] = 'HingeLoss';
            CaffeNode._operatorMap[caffe.V1LayerParameter.LayerType.MEMORY_DATA] = 'HingeLoss';
            CaffeNode._operatorMap[caffe.V1LayerParameter.LayerType.ARGMAX] = 'ArgMax';
            CaffeNode._operatorMap[caffe.V1LayerParameter.LayerType.THRESHOLD] = 'Threshold';
            CaffeNode._operatorMap[caffe.V1LayerParameter.LayerType.DUMMY_DATA] = 'DummyData';
            CaffeNode._operatorMap[caffe.V1LayerParameter.LayerType.SLICE] = 'Slice';
            CaffeNode._operatorMap[caffe.V1LayerParameter.LayerType.MVN] = 'MVN';
            CaffeNode._operatorMap[caffe.V1LayerParameter.LayerType.ABSVAL] = 'AbsVal';
            CaffeNode._operatorMap[caffe.V1LayerParameter.LayerType.SILENCE] = 'Silence';
            CaffeNode._operatorMap[caffe.V1LayerParameter.LayerType.CONTRASTIVE_LOSS] = 'ContrastiveLoss';
            CaffeNode._operatorMap[caffe.V1LayerParameter.LayerType.EXP] = 'Exp';
            CaffeNode._operatorMap[caffe.V1LayerParameter.LayerType.DECONVOLUTION] = 'Deconvolution';
        }
        if (index === undefined) {
            return '?';
        }
        var type = CaffeNode._operatorMap[index];
        if (type) {
            return type;
        }
        return index.toString();
    }
}

class CaffeAttribute {

    constructor(operator, name, value) {
        this._name = name;
        this._value = value;

        var schema = CaffeOperatorMetadata.operatorMetadata.getAttributeSchema(operator, this._name);
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
                    if (this._value.length != defaultValue.length) {
                        this._value = false;
                    }
                    else if (this._value.every((item, index) => { return item == defaultValue[index]; })) {
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
}

class CaffeTensor {

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

        this._type = new CaffeTensorType(dataType, shape);
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
        context.shape = this.type.shape;
        if (!this._data) {
            context.state = 'Tensor data is empty.';
        }
        return context;
    }

    _decode(context, dimension) {
        var results = [];
        var size = context.shape[dimension];
        if (dimension == context.shape.length - 1) {
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
}

class CaffeTensorType {

    constructor(dataType, shape) {
        this._dataType = dataType;
        this._shape = shape.map((dimension) => {
            if (dimension && dimension.__isLong__) {
                return dimension.toNumber();
            }
            return dimension;
        });
    }

    get dataType() {
        return this._dataType;
    }

    get shape() {
        return this._shape;
    }

    toString() {
        return (this.dataType || '?') + (this._shape ? ('[' + this._shape.map((dimension) => dimension.toString()).join(',') + ']') : '');
    }

}

class CaffeOperatorMetadata 
{

    static open(host, callback) {
        if (CaffeOperatorMetadata.operatorMetadata) {
            callback(null, CaffeOperatorMetadata.operatorMetadata);
        }
        else {
            host.request(null, 'caffe-metadata.json', 'utf-8', (err, data) => {
                CaffeOperatorMetadata.operatorMetadata = new CaffeOperatorMetadata(data);
                callback(null, CaffeOperatorMetadata.operatorMetadata);
            });
        }
    }

    constructor(data) {
        this._map = {};
        if (data) {
            var items = JSON.parse(data);
            if (items) {
                items.forEach((item) => {
                    if (item.name && item.schema)
                    {
                        var name = item.name;
                        var schema = item.schema;
                        this._map[name] = schema;
                    }
                });
            }
        }
    }

    getSchema(operator) {
        return this._map[operator] || null;
    }

    getInputs(operator, inputs) {
        var results = [];
        var index = 0;
        var schema = this.getSchema(operator);
        if (schema && schema.inputs) {
            schema.inputs.forEach((inputDef) => {
                if (index < inputs.length || inputDef.option != 'optional') {
                    var input = {};
                    input.name = inputDef.name;
                    input.type = inputDef.type;
                    var count = (inputDef.option == 'variadic') ? (inputs.length - index) : 1;
                    input.connections = [];
                    inputs.slice(index, index + count).forEach((id) => {
                        if (id != '' || inputDef.option != 'optional') {
                            input.connections.push({ id: id});
                        }
                    });
                    index += count;
                    results.push(input);
                }
            });
        }
        else {
            inputs.slice(index).forEach((input) => {
                results.push({
                    name: index.toString(),
                    connections: [ { id: input } ]
                });
                index++;
            });

        }
        return results;
    }

    getOutputs(operator, outputs) {
        var results = [];
        var index = 0;
        var schema = this.getSchema(operator);
        if (schema && schema.outputs) {
            schema.outputs.forEach((outputDef) => {
                if (index < outputs.length || outputDef.option != 'optional') {
                    var output = {};
                    output.name = outputDef.name;
                    var count = (outputDef.option == 'variadic') ? (outputs.length - index) : 1;
                    output.connections = outputs.slice(index, index + count).map((id) => {
                        return { id: id };
                    });
                    index += count;
                    results.push(output);
                }
            });
        }
        else {
            outputs.slice(index).forEach((output) => {
                results.push({
                    name: index.toString(),
                    connections: [ { id: output } ]
                });
                index++;
            });

        }
        return results;
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
}

class CaffeError extends Error {
    constructor(message) {
        super(message);
        this.name = 'Error loading Caffe model.';
    }
}