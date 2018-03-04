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
            // CaffeOperatorMetadata.open(host, (err, metadata) => {
                callback(null, model);
            // });
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

        switch (version) {
            case 1:
                netParameter.layers.forEach((layer) => {
                    this._nodes.push(new CaffeNode(layer, version));
                });
                break;
            case 2:
                netParameter.layer.forEach((layer) => {
                    this._nodes.push(new CaffeNode(layer, version));
                });
                break;
        }


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
                // BNLL = 2;
                table[caffe.V1LayerParameter.LayerType.CONCAT] = 'Concat'; 
                table[caffe.V1LayerParameter.LayerType.CONVOLUTION] = 'Convolution';
                table[caffe.V1LayerParameter.LayerType.DATA] = 'Data';
                table[caffe.V1LayerParameter.LayerType.DROPOUT] = 'Dropout';
                // EUCLIDEAN_LOSS = 7;
                table[caffe.V1LayerParameter.LayerType.FLATTEN] = 'Flatten';
                table[caffe.V1LayerParameter.LayerType.HDF5_DATA] = 'HDF5Data';
                table[caffe.V1LayerParameter.LayerType.HDF5_OUTPUT] = 'HDF5Output';
                // IM2COL = 11;
                table[caffe.V1LayerParameter.LayerType.IMAGE_DATA] = 'ImageData';
                // INFOGAIN_LOSS = 13;
                table[caffe.V1LayerParameter.LayerType.INNER_PRODUCT] = 'InnerProduct';
                table[caffe.V1LayerParameter.LayerType.LRN] = 'LRN';
                // MULTINOMIAL_LOGISTIC_LOSS = 16;
                table[caffe.V1LayerParameter.LayerType.POOLING] = 'Pooling';
                table[caffe.V1LayerParameter.LayerType.RELU] = 'ReLU';
                table[caffe.V1LayerParameter.LayerType.SIGMOID] = 'Sigmoid';
                table[caffe.V1LayerParameter.LayerType.SOFTMAX] = 'Softmax';
                table[caffe.V1LayerParameter.LayerType.SOFTMAX_LOSS] = 'SoftmaxLoss';
                table[caffe.V1LayerParameter.LayerType.SPLIT] = 'Split';
                /*
                    TANH = 23;
                    WINDOW_DATA = 24;
                    ELTWISE = 25;
                    POWER = 26;
                    SIGMOID_CROSS_ENTROPY_LOSS = 27;
                    HINGE_LOSS = 28;
                    MEMORY_DATA = 29;
                    ARGMAX = 30;
                    THRESHOLD = 31;
                    DUMMY_DATA = 32;
                */
                table[caffe.V1LayerParameter.LayerType.SLICE] = 'Slice';
                /*
                    MVN = 34;
                    ABSVAL = 35;
                    SILENCE = 36;
                    CONTRASTIVE_LOSS = 37;
                    EXP = 38;
                    DECONVOLUTION = 39;
                */
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