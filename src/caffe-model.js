/*jshint esversion: 6 */

// Experimental

var caffe = null;

class CaffeModelFactory {

    match(context) {
        var extension = context.identifier.split('.').pop();
        switch (extension) {
            case 'caffemodel':
                return true;
            // case 'prototxt':
            //    return true;
        }
        return false;
    }

    open(context, host, callback) { 
        host.require('caffe', (err) => {
            if (err) {
                callback(err, null);
                return;
            }
            var netParameter = null;
            try {
                caffe = protobuf.roots.caffe.caffe;
                var extension = context.identifier.split('.').pop();
                if (extension == 'prototxt') {
                    var text = new TextDecoder('utf-8').decode(context.buffer);
                    netParameter = caffe.NetParameter.decodeText(text);
                }
                else {
                    netParameter = caffe.NetParameter.decode(context.buffer);
                }
            }
            catch (error) {
                callback(new CaffeError('File format is not caffe.NetParameter (' + error.message + ').'), null);
                return;
            }
            var model = null;
            try {
                model = new CaffeModel(netParameter);
            }
            catch (error) {
                host.exception(error, false);
                callback(new CaffeError(error.message), null);
                return;
            }
            CaffeOperatorMetadata.open(host, (err, metadata) => {
                callback(null, model);
            });
        });
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
            netParameter.input.forEach((input) => {
                this._inputs.push(new CaffeArgument(input, [ new CaffeConnection(input, null, null) ]));
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
                var input = node._outputs[0];
                var attributes = node.attributes;
                if (attributes.length == 1) {
                    var attribute = attributes[0];
                    if (attribute.name == 'shape') {
                        if (attribute._value.length == 1 && attribute._value[0].dim) {
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
                        this._attributes.push(new CaffeAttribute(this, attributeName, attributeValue));
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
                        if (type == 'Data') {
                            type = 'Input';
                        }
                        if (param.constructor.name == type + 'Parameter') {
                            Object.keys(param).forEach((attributeName) => {
                                var attributeValue = param[attributeName];
                                this._attributes.push(new CaffeAttribute(this, attributeName, attributeValue));
                            });
                        }
                    }
                });
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
        return CaffeOperatorMetadata.operatorMetadata.getOperatorCategory(this._type);
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
        var type = CaffeNode._operatorMap[index];
        return type ? type : index.toString();
    }
}

class CaffeAttribute {

    constructor(owner, name, value) {
        this._owner = owner;
        this._name = name;
        this._value = value;
    }

    get name() {
        return this._name;
    }

    get value() { 
        return JSON.stringify(this._value);
    }

    get visible() {
        return CaffeOperatorMetadata.operatorMetadata.getAttributeVisible(this._owner.operator, this._name, this._value);
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

    getOperatorCategory(operator) {
        var schema = this._map[operator];
        if (schema && schema.category) {
            return schema.category;
        }
        return null;
    }

    getInputs(type, inputs) {
        var results = [];
        var index = 0;
        var schema = this._map[type];
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
                var name = (index == 0) ? 'input' : ('(' + index.toString() + ')');
                results.push({
                    name: name,
                    connections: [ { id: input } ]
                });
                index++;
            });

        }
        return results;
    }

    getOutputs(type, outputs) {
        var results = [];
        var index = 0;
        var schema = this._map[type];
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
                var name = (index == 0) ? 'output' : ('(' + index.toString() + ')');
                results.push({
                    name: name,
                    connections: [ { id: output } ]
                });
                index++;
            });

        }
        return results;
    }

    getAttributeVisible(operator, name, value) {
        var schema = this._map[operator];
        if (schema && schema.attributes && schema.attributes.length > 0) {
            if (!schema.attributesMap) {
                schema.attributesMap = {};
                schema.attributes.forEach((attribute) => {
                    schema.attributesMap[attribute.name] = attribute;
                });
            }
            var attribute = schema.attributesMap[name];
            if (attribute) {
                if (attribute.hasOwnProperty('visible')) {
                    return attribute.visible;
                }
                if (attribute.hasOwnProperty('default')) {
                    return !CaffeOperatorMetadata.isEquivalent(attribute.default, value);
                }
            }
        }
        return true;
    }

    static isEquivalent(a, b) {
        if (a === b) {
            return a !== 0 || 1 / a === 1 / b;
        }
        if (a == null || b == null) {
            return false;
        }
        if (a !== a) {
            return b !== b;
        }
        var type = typeof a;
        if (type !== 'function' && type !== 'object' && typeof b != 'object') {
            return false;
        }
        var className = toString.call(a);
        if (className !== toString.call(b)) {
            return false;
        }
        switch (className) {
            case '[object RegExp]':
            case '[object String]':
                return '' + a === '' + b;
            case '[object Number]':
                if (+a !== +a) {
                    return +b !== +b;
                }
                return +a === 0 ? 1 / +a === 1 / b : +a === +b;
            case '[object Date]':
            case '[object Boolean]':
                return +a === +b;
            case '[object Array]':
                var length = a.length;
                if (length !== b.length) {
                    return false;
                }
                while (length--) {
                    if (!CaffeOperatorMetadata.isEquivalent(a[length], b[length])) {
                        return false;
                    }
                }
                return true;
        }

        var keys = Object.keys(a);
        var size = keys.length;
        if (Object.keys(b).length != size) {
            return false;
        } 
        while (size--) {
            var key = keys[size];
            if (!(b.hasOwnProperty(key) && CaffeOperatorMetadata.isEquivalent(a[key], b[key]))) {
                return false;
            }
        }
        return true;
    }
}

class CaffeError extends Error {
    constructor(message) {
        super(message);
        this.name = 'Error loading Caffe model.';
    }
}