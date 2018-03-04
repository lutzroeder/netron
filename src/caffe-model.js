/*jshint esversion: 6 */

// Experimental

var caffe = null;

class CaffeModel {

    static open(buffer, identifier, host, callback) { 
        host.import('/caffe.js', (err) => {
            if (err) {
                callback(err, null);
            }
            else {
                caffe = protobuf.roots.caffe.caffe
                CaffeModel.create(buffer, identifier, host, (err, model) => {
                    callback(err, model);
                });
            }
        });
    }

    static create(buffer, identifier, host, callback) {
        try {
            var netParameter = caffe.NetParameter.decode(buffer);
            var model = new CaffeModel(netParameter);
            CaffeOperatorMetadata.open(host, (err, metadata) => {
                callback(null, model);
            });
        }
        catch (err) {
            callback(err, null);
        }
    }

    constructor(netParameter) {
        if (netParameter.layers && netParameter.layers.length > 0) {
            this._version = 1;
        }
        else if (netParameter.layer && netParameter.layer.length > 0) {
            this._version = 2;
        }
        this._graphs = [ new CaffeGraph(netParameter, this._version) ];
    }

    get properties() {
        var results = [];
        results.push({ name: 'Format', value: 'Caffe' + (this._version ? ' v' + this._version.toString() : '') });
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

        var layers = [];
        switch (version) {
            case 1:
                layers = netParameter.layers;
                break;
            case 2:
                layers = netParameter.layer;
                break;
        }

        var nonInplaceLayers = [];
        var inplaceMap = {};
        layers.forEach((layer) => {
            if (layer.top.length == 1 && layer.bottom.length == 1 && layer.top[0] == layer.bottom[0]) {
                var key = layer.top[0];
                if (!inplaceMap[key]) {
                    inplaceMap[key] = [];
                }
                inplaceMap[key].push(layer);
            }
            else {
                nonInplaceLayers.push(layer);
            }
        });

        Object.keys(inplaceMap).forEach((key) => {
            var nodes = inplaceMap[key];
            nodes.forEach((node, index) => {
                if (index > 0) {
                    node.bottom[0] = node.bottom[0] + ':' + index.toString();
                }
                node.top[0] = node.top[0] + ':' + (index + 1).toString();
            });
        });

        nonInplaceLayers.forEach((layer) => {
            layer.bottom = layer.bottom.map((bottom) => {
                if (inplaceMap[bottom]) {
                    return bottom + ':' + inplaceMap[bottom].length.toString();
                }
                return bottom;
            });
        });

        layers.forEach((layer) => {
            this._nodes.push(new CaffeNode(layer, version));
        });
    }

    get name() {
        return this._name;
    }

    get type() {
        return '';
    }

    get inputs() {
        return [];
    }

    get outputs() {
        return [];
    }

    get nodes() {
        return this._nodes;
    }
}

class CaffeNode {

    constructor(layer, version) {
        this._name = layer.name;

        if (version == 1) {
            if (!CaffeNode._operatorTable) {
                var table = {};
                table[caffe.V1LayerParameter.LayerType.NONE] = 'None';
                table[caffe.V1LayerParameter.LayerType.ACCURACY] = 'Accuracy';
                table[caffe.V1LayerParameter.LayerType.BNLL] = 'BNLL';
                table[caffe.V1LayerParameter.LayerType.CONCAT] = 'Concat'; 
                table[caffe.V1LayerParameter.LayerType.CONVOLUTION] = 'Convolution';
                table[caffe.V1LayerParameter.LayerType.DATA] = 'Data';
                table[caffe.V1LayerParameter.LayerType.DROPOUT] = 'Dropout';
                table[caffe.V1LayerParameter.LayerType.EUCLIDEAN_LOSS] = 'EuclideanLoss';
                table[caffe.V1LayerParameter.LayerType.FLATTEN] = 'Flatten';
                table[caffe.V1LayerParameter.LayerType.HDF5_DATA] = 'HDF5Data';
                table[caffe.V1LayerParameter.LayerType.HDF5_OUTPUT] = 'HDF5Output';
                table[caffe.V1LayerParameter.LayerType.IM2COL] = 'Im2col';
                table[caffe.V1LayerParameter.LayerType.IMAGE_DATA] = 'ImageData';
                table[caffe.V1LayerParameter.LayerType.INFOGAIN_LOSS] = 'InfogainLoss';
                table[caffe.V1LayerParameter.LayerType.INNER_PRODUCT] = 'InnerProduct';
                table[caffe.V1LayerParameter.LayerType.LRN] = 'LRN';
                table[caffe.V1LayerParameter.LayerType.MULTINOMIAL_LOGISTIC_LOSS] = 'MultinomialLogisticLoss';
                table[caffe.V1LayerParameter.LayerType.POOLING] = 'Pooling';
                table[caffe.V1LayerParameter.LayerType.RELU] = 'ReLU';
                table[caffe.V1LayerParameter.LayerType.SIGMOID] = 'Sigmoid';
                table[caffe.V1LayerParameter.LayerType.SOFTMAX] = 'Softmax';
                table[caffe.V1LayerParameter.LayerType.SOFTMAX_LOSS] = 'SoftmaxLoss';
                table[caffe.V1LayerParameter.LayerType.SPLIT] = 'Split';
                table[caffe.V1LayerParameter.LayerType.TANH] = 'TanH';
                table[caffe.V1LayerParameter.LayerType.WINDOW_DATA] = 'WindowData';
                table[caffe.V1LayerParameter.LayerType.ELTWISE] = 'Eltwise';
                table[caffe.V1LayerParameter.LayerType.POWER] = 'Power';
                table[caffe.V1LayerParameter.LayerType.SIGMOID_CROSS_ENTROPY_LOSS] = 'SigmoidCrossEntropyLoss';
                table[caffe.V1LayerParameter.LayerType.HINGE_LOSS] = 'HingeLoss';
                table[caffe.V1LayerParameter.LayerType.MEMORY_DATA] = 'HingeLoss';
                table[caffe.V1LayerParameter.LayerType.ARGMAX] = 'ArgMax';
                table[caffe.V1LayerParameter.LayerType.THRESHOLD] = 'Threshold';
                table[caffe.V1LayerParameter.LayerType.DUMMY_DATA] = 'DummyData';
                table[caffe.V1LayerParameter.LayerType.SLICE] = 'Slice';
                table[caffe.V1LayerParameter.LayerType.MVN] = 'MVN';
                table[caffe.V1LayerParameter.LayerType.ABSVAL] = 'AbsVal';
                table[caffe.V1LayerParameter.LayerType.SILENCE] = 'Silence';
                table[caffe.V1LayerParameter.LayerType.CONTRASTIVE_LOSS] = 'ContrastiveLoss';
                table[caffe.V1LayerParameter.LayerType.EXP] = 'Exp';
                table[caffe.V1LayerParameter.LayerType.DECONVOLUTION] = 'Deconvolution';
                CaffeNode._operatorTable = table;
            }
            this._type = CaffeNode._operatorTable[layer.type];
            if (!this._type) {
                this._type = layer.type.toString();
            }
        }
        else if (version == 2) {
            this._type = layer.type;
        }

        this._inputs = [];
        layer.bottom.forEach((bottom) => {
            this._inputs.push({
                connections: [ {
                    id: bottom
                } ]
            });
        });

        this._outputs = [];
        layer.top.forEach((top) => {
            this._outputs.push({
                connections: [ {
                    id: top
                } ]
            });
        });

        this._attributes = [];
        Object.keys(layer).forEach((key) => {
            if (key.endsWith('Param')) {
                var param = layer[key];
                if (param.constructor.name == this._type + 'Parameter') {
                    Object.keys(param).forEach((attributeName) => {
                        var attributeValue = param[attributeName];
                        this._attributes.push(new CaffeAttribute(attributeName, attributeValue));
                    });
                }
            }
        });
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
        return this._inputs;
    }

    get outputs() {
        return this._outputs;
    }

    get attributes() {
        return this._attributes;
    }
}

class CaffeAttribute {

    constructor(name, value) {
        this._name = name;
        this._value = value;
    }

    get name() {
        return this._name;
    }

    get value() { 
        return JSON.stringify(this._value);
    }
}

class CaffeOperatorMetadata 
{

    static open(host, callback) {
        if (CaffeOperatorMetadata.operatorMetadata) {
            callback(null, CaffeOperatorMetadata.operatorMetadata);
        }
        else {
            host.request('/caffe-operator.json', (err, data) => {
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
}