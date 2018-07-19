/*jshint esversion: 6 */

// Experimental

var caffe = null;

class CaffeModelFactory {

    match(buffer, identifier) {
        var extension = identifier.split('.').pop();
        return extension == 'caffemodel';
    }

    open(buffer, identifier, host, callback) { 
        host.import('/caffe.js', (err) => {
            if (err) {
                callback(err, null);
                return;
            }
            var netParameter = null;
            try {
                caffe = protobuf.roots.caffe.caffe;
                netParameter = caffe.NetParameter.decode(buffer);
            }
            catch (error) {
                callback(new CaffeError('Protocol Buffer loader failed to decode caffe.NetParameter input stream (' + error.message + ').'), null);
                return;
            }
            var model = null;
            try {
                model = new CaffeModel(netParameter);
            }
            catch (error) {
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

    get properties() {
        var results = [];
        results.push({ name: 'format', value: 'Caffe' + (this.hasOwnProperty('_version') ? ' v' + this._version.toString() : '') });
        return results;
    }

    get graphs() {
        return this._graphs;
    }

}

class CaffeGraph {

    constructor(netParameter, version)
    {
        this._name = netParameter.name;
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
                this._inputs.push({
                    id: input,
                    name: input,
                    type: 'T'
                });
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
                this._outputs.push({
                    id: keys[0],
                    name: keys[0],
                    type: 'T'
                });
            }
            else if (outputs.length == 1) {
                outputs[0]._outputs = [ 'output' ];
                this._outputs.push({
                    id: 'output',
                    name: 'output',
                    type: 'T'
                });
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
        if (node.operator == 'Input' && node._inputs.length == 0 && node._outputs.length == 1) {
            var input = node._outputs[0];
            var attributes = node.attributes;
            if (attributes.length == 1) {
                var attribute = attributes[0];
                if (attribute.name == 'shape') {
                    if (attribute._value.length == 1 && attribute._value[0].dim) {
                        this._inputs.push({ 
                            id: input,
                            name: input,
                            type: 'T' + JSON.stringify(attribute._value[0].dim)
                        });
                        return true;
                    }
                }
            }
        }
        return false;
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
                    if (attributeName != 'type' && attributeName != 'name' && attributeName != 'blobs' && attributeName != 'blobsLr') {
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
                    if (key.endsWith('Param')) {
                        var param = layer[key];
                        var type = this._type;
                        if (type == 'Deconvolution') {
                            type = 'Convolution';
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
        inputs.forEach((input) => {
            input.connections.forEach((connection) => {
                if (connection.id instanceof CaffeTensor) {
                    connection.initializer = connection.id;
                    connection.type = connection.initializer.type;
                    connection.id = '';
                }
            });
        });

        return inputs;
    }

    get outputs() {
        var outputs = CaffeOperatorMetadata.operatorMetadata.getOutputs(this._type, this._outputs);
        return outputs;
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
        this._name = '';
        for (var i = 0; i < name.length; i++) {
            var character = name[i];
            var lowerCase = character.toLowerCase();
            this._name += (character != lowerCase) ? ('_' + lowerCase) : character;
        }
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

        if (blob.hasOwnProperty('num') && blob.hasOwnProperty('channels') &&
            blob.hasOwnProperty('width') && blob.hasOwnProperty('height')) {
            this._shape = [];
            if (blob.num != 1) {
                this._shape.push(blob.num);
            }
            if (blob.channels != 1) {
                this._shape.push(blob.channels);
            }
            if (blob.width != 1) {
                this._shape.push(blob.width);
            }
            if (blob.height != 1) {
                this._shape.push(blob.height);
            }
        }
        else if (blob.hasOwnProperty('shape')) {
            this._shape = blob.shape.dim;
        }

        this._type = '?';
        if (blob.data.length > 0) {
            this._type = 'float';
            this._data = blob.data;
        }
        else if (blob.doubleData.length > 0) {
            this._type = 'double';
            this._data = blob.doubleData;
        }
    }

    get kind() {
        return 'Blob';
    }

    get type() {
        return this._type + JSON.stringify(this._shape);
    }

    get value() {
        var result = this._decode(Number.MAX_SAFE_INTEGER);
        if (result.error) {
            return null;
        }
        return result.value;
    }

    toString() {
        var result = this._decode(10000);
        if (result.error) {
            return result.error;
        }
        return JSON.stringify(result.value, null, 4);
    }

    _decode(limit) {
        var result = {};
        if (!this._data) {
            result.error = 'Tensor data is empty.';
            return result;
        }
        var context = {};
        context.index = 0;
        context.count = 0;
        context.limit = limit;
        context.data = this._data;
        result.value = this._decodeDimension(context, 0);
        return result;
    }

    _decodeDimension(context, dimension) {
        var results = [];
        var size = this._shape[dimension];
        if (dimension == this._shape.length - 1) {
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
                results.push(this._decodeDimension(context, dimension + 1));
            }
        }
        return results;
    }
}

class CaffeOperatorMetadata 
{

    static open(host, callback) {
        if (CaffeOperatorMetadata.operatorMetadata) {
            callback(null, CaffeOperatorMetadata.operatorMetadata);
        }
        else {
            host.request('/caffe-metadata.json', (err, data) => {
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