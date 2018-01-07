/*jshint esversion: 6 */

// Experimental

class KerasModel {

    static open(buffer, identifier, host, callback) { 
        try {
            var version = null;
            var backend = null;
            var model_config = null;

            var extension = identifier.split('.').pop();
            if (extension == 'keras' || extension == 'h5') {
                var h5 = new H5(buffer);
                version = h5.rootGroup.attributes.keras_version;
                backend = h5.rootGroup.attributes.backend;
                model_config = h5.rootGroup.attributes.model_config;
                if (!model_config) {
                    throw new Error('H5 file has no \'model_config\' data.');
                }
            }
            else if (extension == 'json') {
                if (!window.TextDecoder) {
                    throw new Error('TextDecoder not avaialble.');
                }

                var decoder = new TextDecoder('utf-8');
                model_config = decoder.decode(buffer);
            }

            var root = JSON.parse(model_config);
            var model = new KerasModel(root, version, backend);

            KerasOperatorMetadata.open(host, (err, metadata) => {
                callback(null, model);
            });
        }
        catch (err) {
            callback(err, null);
        }
    }

    constructor(root, keras_version, backend) {
        if (!root.class_name) {
            throw new Error('class_name is not present.');
        }
        if (root.class_name != 'Model' && root.class_name != 'Sequential') {
            throw new Error('\'' + root.class_name + '\' is not supported.');
        }
        this._version = keras_version;
        this._backend = backend;
        var graph = new KerasGraph(root);
        this._graphs = [ graph ];
        this._activeGraph = graph; 
    }

    format() {
        var summary = { properties: [], graphs: [] };

        this.graphs.forEach((graph) => {
            summary.graphs.push({
                name: graph.name,
                inputs: graph.inputs,
                outputs: graph.outputs
            });
        });

        summary.properties.push({ 
            name: 'Format', 
            value: 'Keras' + (this._version ? (' ' + this._version) : '')
        });
        if (this._backend) {
            summary.properties.push({ 
                name: 'Backend', 
                value: this._backend
            });
        }

        return summary;
    }

    get graphs() {
        return this._graphs;
    }

    get activeGraph() {
        return this._activeGraph;
    }

    updateActiveGraph(name) {
        this._activeGraph = (name == this._graphs[0].name) ? this._graph : null;
    }
}

class KerasGraph {

    constructor(root) {
        this._name = root.name;
        this._inputs = [];
        this._outputs = [];
        this._nodes = [];

        switch (root.class_name) {
            case 'Sequential':
                this.loadSequential(root.config);
                break;
            case 'Model':
                this.loadModel(root.config);
                break;
        }
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

    get nodes() {
        return this._nodes;
    }

    loadModel(root) {

        if (root.layers) {

            var nodeMap = {};
            root.layers.forEach((layer) => {
                if (layer.name) {
                    if (!nodeMap[layer.name]) {
                        nodeMap[layer.name] = layer;
                        layer._inputs = [];
                        layer._outputs = [];
                    }
                }
            });
            root.layers.forEach((layer) => {
                if (layer.inbound_nodes) {
                    layer.inbound_nodes.forEach((inbound_node) => {
                        var input = { connections: [] };
                        inbound_node.forEach((inbound_connection) => {
                            var inputName = inbound_connection[0];
                            input.connections.push({ id: inputName });
                            var inputNode = nodeMap[inputName];
                            if (inputNode) {
                                inputNode._outputs.push(inputNode.name);
                            }
                        });       
                        layer._inputs.push(input);
                    });
                }
            });
        }

        /*
        if (root.input_layers) {
            root.input_layers.forEach((input_layer) => {
                this._inputs.push({ id: input_layer[0], name: input_layer[0] });
            });    
        }
        */

        if (root.output_layers) {
            root.output_layers.forEach((output_layer) => {
                var inputName = output_layer[0];
                var inputNode = nodeMap[inputName];
                if (inputNode) {
                    inputNode._outputs.push(inputName);
                }
                this._outputs.push({ id: inputName, name: inputName, type: '?' });
            });
        }

        if (root.layers) {
            root.layers.forEach((layer) => {
                var node = new KerasNode(layer.class_name, layer.name, layer.config, layer._inputs, layer._outputs);
                this._nodes.push(node);
            });
        }
    }

    loadSequential(root) {
        var output = 'input';

        this._inputs.push({
            name: output,
            id: output,
            type: '?'
        });

        var id = 0;
        root.forEach((layer) => {
            var inputs = [];
            if (output) {
                inputs.push({
                    name: '(0)',
                    connections: [ { id: output }]
                });
            }

            var name = id.toString();
            if (layer.config || layer.config.name) {
                name = layer.config.name;
            }
            id++;
            output = name;

            var outputs = [ output ];

            var node = new KerasNode(layer.class_name, name, layer.config, inputs, outputs);
            this._nodes.push(node);
        });

        this._outputs.push({
            name: 'output',
            id: output,
            type: '?'
        });
    }
}

class KerasNode {

    constructor(operator, name, config, inputs, outputs) {
        this._operator = operator;
        this._name = name;
        this._config = config;
        this._inputs = inputs;
        this._outputs = outputs;
    }

    get operator() {
        return this._operator;
    }

    get category() {
        return KerasOperatorMetadata.operatorMetadata.getOperatorCategory(this.operator);
    }

    get name() {
        return this._name;
    }

    get inputs() {
        var results = [];
        this._inputs.forEach((input, index) => {
            results.push({ 
                name: '(' + index.toString() + ')', 
                connections: input.connections
            });
        });
        return results;
    }

    get outputs() {
        var results = [];
        this._outputs.forEach((output, index) => {
            results.push({ 
                name: '(' + index.toString() + ')', 
                connections: [ { id: output }]
            });
        });
        return results;
    }

    get attributes() {
        var results = [];
        if (this._config) {
            Object.keys(this._config).forEach((name) => {
                var value = this._config[name];
                if (name != 'name' && value != null) {
                    var hidden = !KerasOperatorMetadata.operatorMetadata.showAttribute(this.operator, name, value);
                    results.push(new KerasAttribute(name, value, hidden));
                }
            });
        }
        return results;
    }

    get dependencies() {
        return [];
    }
}

class KerasAttribute {

    constructor(name, value, hidden) {
        this._name = name;
        this._value = value;
        this._hidden = hidden;
    }

    get name() {
        return this._name;
    }

    get value() {
        if (this._value == true) {
            return 'true';
        }
        if (this._value == false) {
            return 'false';
        }
        if (this._value) {
            return JSON.stringify(this._value);
        }
        return '?';
    }

    get hidden() {
        return this._hidden;
    }
}

class KerasOperatorMetadata {

    static open(host, callback) {
        if (KerasOperatorMetadata.operatorMetadata) {
            callback(null, KerasOperatorMetadata.operatorMetadata);
        }
        else {
            host.request('/keras-operator.json', (err, data) => {
                if (err == null) {
                    KerasOperatorMetadata.operatorMetadata = new KerasOperatorMetadata(data);
                }
                callback(null, KerasOperatorMetadata.operatorMetadata);
            });    
        }
    }

    constructor(data) {
        this._map = {};
        var items = JSON.parse(data);
        if (items) {
            items.forEach((item) => {
                if (item.name && item.schema)
                {
                    this._map[item.name] = item.schema;
                }
            });
        }

        this._categoryMap = {
            'Conv1D': 'Layer',
            'Conv2D': 'Layer',
            'Conv3D': 'Layer',
            'Convolution1D': 'Layer',
            'Convolution2D': 'Layer',
            'Convolution3D': 'Layer',
            'DepthwiseConv2D': 'Layer',
            'Dense': 'Layer',
            'BatchNormalization': 'Normalization',
            'Concatenate': 'Tensor',
            'Activation': 'Activation',
            'GlobalAveragePooling2D': 'Pool',
            'AveragePooling2D': 'Pool',
            'MaxPooling2D': 'Layer',
            'GlobalMaxPooling2D': 'Layer',
            'Flatten': 'Shape',
            'Reshape': 'Shape',
            'Dropout': 'Dropout'
        };    
    }

    showAttribute(operator, attributeName, attributeValue) {
        if (attributeName == 'trainable') {
            return false;
        }
        return !this.defaultAttribute(operator, attributeName, attributeValue);
    }

    defaultAttribute(operator, attributeName, attributeValue) {
        var schema = this._map[operator];
        if (schema && schema.attributes && schema.attributes.length > 0) {
            if (!schema.attributeMap) {
                schema.attributeMap = {};
            }
            schema.attributes.forEach(attribute => {
                schema.attributeMap[attribute.name] = attribute;
            });

            var attribute = schema.attributeMap[attributeName];
            if (attribute) {
                if (attribute && attribute.hasOwnProperty('default')) {
                    return KerasOperatorMetadata.isEquivalent(attribute.default, attributeValue);
                }
            }
        }
        return false;
    }

    getOperatorCategory(operator) {
        var category = this._categoryMap[operator];
        if (category) {
            return category;
        }
        return null;
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
                    if (!KerasOperatorMetadata.isEquivalent(a[length], b[length])) {
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
            if (!(b.hasOwnProperty(key) && KerasOperatorMetadata.isEquivalent(a[key], b[key]))) {
                return false;
            }
        }
        return true;
    }
}