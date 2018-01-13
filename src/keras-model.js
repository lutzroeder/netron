/*jshint esversion: 6 */

// Experimental

class KerasModel {

    static open(buffer, identifier, host, callback) { 
        try {
            var version = null;
            var backend = null;
            var json = null;
            var rootGroup = null;

            var extension = identifier.split('.').pop();
            if (extension == 'keras' || extension == 'h5') {
                var file = new hdf5.File(buffer);
                rootGroup = file.rootGroup;
                json = rootGroup.attributes.model_config;
                if (!json) {
                    throw new KerasError('HDF5 file has no \'model_config\' data.');
                }
            }
            else if (extension == 'json') {
                if (!window.TextDecoder) {
                    throw new KerasError('TextDecoder not avaialble.');
                }

                var decoder = new TextDecoder('utf-8');
                json = decoder.decode(buffer);
            }

            var model = new KerasModel(json, rootGroup);

            KerasOperatorMetadata.open(host, (err, metadata) => {
                callback(null, model);
            });
        }
        catch (err) {
            callback(err, null);
        }
    }

    constructor(json, rootGroup) {
        var model = JSON.parse(json);
        if (!model.class_name) {
            throw new KerasError('class_name is not present.');
        }
        if (rootGroup && rootGroup.attributes.keras_version) {
            this._version = rootGroup.attributes.keras_version;
        }
        if (rootGroup && rootGroup.attributes.backend) {
            this._backend = rootGroup.attributes.backend;
        }

        var graph = new KerasGraph(model, rootGroup);
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

    constructor(model, rootGroup) {
        if (model.name) {
            this._name = model.name;            
        }
        else if (model.config && model.config.name) {
            this._name = model.config.name;
        }
        this._inputs = [];
        this._outputs = [];
        this._nodes = [];

        switch (model.class_name) {
            case 'Sequential':
                this.loadSequential(model.config, rootGroup);
                break;
            case 'Model':
                this.loadModel(model.config, rootGroup);
                break;
            default:
                throw new KerasError('\'' + model.class_name + '\' is not supported.');
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

    loadModel(config, rootGroup) {
        if (config.layers) {
            var nodeMap = {};
            config.layers.forEach((layer) => {
                if (layer.name) {
                    if (!nodeMap[layer.name]) {
                        nodeMap[layer.name] = layer;
                        layer._inputs = [];
                        layer._outputs = [];
                    }
                }
            });
            config.layers.forEach((layer) => {
                if (layer.inbound_nodes) {
                    layer.inbound_nodes.forEach((inbound_node) => {
                        inbound_node.forEach((inbound_connection) => {
                            var input = { connections: [] };
                            var inputName = inbound_connection[0];
                            var inputNode = nodeMap[inputName];
                            if (inputNode) {
                                var inputIndex = inbound_connection[2];
                                if (inputIndex != 0) {
                                    inputName += ':' + inputIndex.toString();
                                }
                                while (inputIndex >= inputNode._outputs.length) {
                                    inputNode._outputs.push({ connections: [ ] });        
                                }     
                                inputNode._outputs[inputIndex].connections.push({ 
                                    id: inputName
                                });
                            }
                            input.connections.push({ id: inputName });
                            layer._inputs.push(input);
                        });       
                    });
                }
            });
        }
        if (config.input_layers) {
            config.input_layers.forEach((input_layer) => {
                var name = input_layer[0];
                var input = {
                    id: name + ':0',
                    name: name
                };
                var node = nodeMap[name];
                if (node && node.class_name == 'InputLayer') {
                    this.translateInput(node, input);
                    delete nodeMap[name];
                }
                this._inputs.push(input); 
            });
        }
        if (config.output_layers) {
            config.output_layers.forEach((output_layer) => {
                var inputName = output_layer[0];
                var inputNode = nodeMap[inputName];
                if (inputNode) {
                    inputNode._outputs.push({
                        connections: [ { id: inputName } ]                        
                    });
                }
                var output = {
                    id: inputName,
                    name: inputName,
                    type: '?'
                };
                this._outputs.push(output);
            });
        }
        if (config.layers) {
            config.layers.forEach((layer) => {
                if (nodeMap[layer.name]) {
                    this.translateNode(layer.name, layer, layer._inputs, layer._outputs, rootGroup).forEach((node) => {
                        this._nodes.push(node);
                    });
                }
            });
        }
    }

    loadSequential(config, rootGroup) {
        var connection = 'input';
        var input = {
            id: connection,
            name: connection
        };
        this._inputs.push(input);
        var id = 0;
        config.forEach((layer) => {
            var inputs = [ {
                connections: [ { id: connection } ]
            } ];
            var name = id.toString();
            if (id == 0) {
                this.translateInput(layer, input);
            }
            id++;
            if (layer.config && layer.config.name) {
                name = layer.config.name;
            }
            connection = name;
            var outputs = [ {
                connections: [ { id: connection } ]
            } ];
            this.translateNode(name, layer, inputs, outputs, rootGroup).forEach((node) => {
                this._nodes.push(node);
            });
        });
        this._outputs.push({
            name: 'output',
            id: connection,
            type: '?'
        });
    }

    translateNode(name, layer, inputs, outputs, rootGroup) {
        var results = [];
        if (layer.class_name == 'Bidirectional' || layer.class_name == 'TimeDistributed') {
            if (layer.config.layer) {
                var subLayer = layer.config.layer;
                var subConnection = name + '|layer';
                inputs.push({
                    name: 'layer',
                    connections: [ { id: subConnection} ]
                });
                var subOutputs = [ {
                    connections: [ { id: subConnection } ]
                } ];
                results.push(new KerasNode(subLayer.class_name, subLayer.config.name, subLayer.config, [], subOutputs, rootGroup));
                delete layer.config.layer;
            }
        }
        var node = new KerasNode(layer.class_name, name, layer.config, inputs, outputs, rootGroup);
        results.push(node);
        return results;
    }

    translateInput(layer, input) {
        input.type = '';
        if (layer && layer.config) {
            var config = layer.config;
            if (config.dtype) {
                input.type = config.dtype;
                delete config.dtype;
            }
            if (config.batch_input_shape) {
                var shape = config.batch_input_shape;
                if (shape.length > 0 && shape[0] == null) {
                    shape.shift();
                }
                input.type = input.type + '[' + shape.toString() + ']';
                delete config.batch_input_shape;
            }
        }
    }
}

class KerasNode {

    constructor(operator, name, config, inputs, outputs, rootGroup) {
        this._operator = operator;
        this._name = name;
        this._config = config;
        this._inputs = inputs;
        this._outputs = outputs;

        if (rootGroup) {
            var model_weights = rootGroup.group('model_weights');
            if (model_weights) {
                var group = model_weights.group(this._name);
                if (group) {
                    var weight_names = group.attributes.weight_names;
                    if (weight_names) {
                        weight_names.forEach((weight_name) => {
                            var weight_variable = group.group(weight_name);
                            if (weight_variable) {
                                var variable = weight_variable.value;
                                if (variable) {
                                    this._inputs.push({
                                        connections: [ {
                                            id: weight_name,
                                            type: variable.type + JSON.stringify(variable.shape),
                                            initializer: variable
                                        } ]
                                    });
                                }
                            }
                        });
                    }
                }
            }
        }
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
                name: input.name ? input.name : KerasOperatorMetadata.operatorMetadata.getInputName(this.operator, index), 
                connections: input.connections
            });
        });
        return results;
    }

    get outputs() {
        var results = [];
        this._outputs.forEach((output, index) => {
            results.push({
                name: output.name ? output.name : KerasOperatorMetadata.operatorMetadata.getOutputName(this.operator, index), 
                connections: output.connections
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
        if (this._value === true) {
            return 'true';
        }
        if (this._value === false) {
            return 'false';
        }
        if (this._value === null) {
            return 'null';
        }
        if (this._value == 0) {
            return 0;
        }
        if (typeof this._value == 'object' && this._value.class_name && this._value.config) {
            return this._value.class_name + '(' + Object.keys(this._value.config).map(key => {
                var value = this._value.config[key];
                return key + '=' + JSON.stringify(value);
            }).join(', ') + ')';
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
                KerasOperatorMetadata.operatorMetadata = new KerasOperatorMetadata(data);
                callback(null, KerasOperatorMetadata.operatorMetadata);
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
                        this._map[item.name] = item.schema;
                    }
                });
            }
        }
    }

    getInputName(operator, index) {
        var schema = this._map[operator];
        if (schema) {
            var inputs = schema.inputs;
            if (inputs && index < inputs.length) {
                var input = inputs[index];
                if (input) {
                    var name = input.name;
                    if (name) {
                        return name;
                    }
                } 
            }
        }
        return "(" + index.toString() + ")";
    }

    getOutputName(operator, index) {
        var schema = this._map[operator];
        if (schema) {
            var outputs = schema.outputs;
            if (outputs && index < outputs.length) {
                var output = outputs[index];
                if (output) {
                    var name = output.name;
                    if (name) {
                        return name;
                    }
                } 
            }
        }
        return "(" + index.toString() + ")";
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
        var schema = this._map[operator];
        if (schema) {
            var category = schema.category;
            if (category) {
                return category;
            }
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

class KerasError extends Error {
    constructor(message) {
        super(message);
        this.name = 'Keras Error';
    }
}