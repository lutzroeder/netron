/*jshint esversion: 6 */

class KerasModel {

    static open(buffer, identifier, host, callback) { 
        host.import('/hdf5.js', (err) => {
            if (err) {
                callback(err, null);
            }
            else {
                KerasModel.create(buffer, identifier, host, (err, model) => {
                    callback(err, model);
                });
            }
        });
    }

    static create(buffer, identifier, host, callback) {
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
                    throw new KerasError('HDF5 file does not contain a \'model_config\' graph. Use \'save()\' instead of \'save_weights()\' to save both the graph and weights.');
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

    get properties() {
        var results = [];

        var format = 'Keras';
        if (this._version) {
            format = format + ' v' + this._version;
        }
        results.push({ name: 'Format', value: format });

        if (this._backend) {
            results.push({ name: 'Backend', value: this._backend });
        }

        return results;
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
                            var inputName = inbound_connection[0];
                            var inputNode = nodeMap[inputName];
                            if (inputNode) {
                                var inputIndex = inbound_connection[2];
                                if (inputIndex != 0) {
                                    inputName += ':' + inputIndex.toString();
                                }
                                while (inputIndex >= inputNode._outputs.length) {
                                    inputNode._outputs.push('');        
                                }     
                                inputNode._outputs[inputIndex] = inputName;
                            }
                            layer._inputs.push(inputName);
                        });       
                    });
                }
            });
        }
        if (config.input_layers) {
            config.input_layers.forEach((input_layer) => {
                var name = input_layer[0];
                var input = {
                    id: name,
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
                    var inputIndex = output_layer[2];
                    if (inputIndex != 0) {
                        inputName += ':' + inputIndex.toString();
                    }
                    while (inputIndex >= inputNode._outputs.length) {
                        inputNode._outputs.push('');
                    }
                    inputNode._outputs[inputIndex] = inputName;
                }
                this._outputs.push({
                    id: inputName,
                    name: inputName
                });
            });
        }
        if (config.layers) {
            config.layers.forEach((layer) => {
                if (nodeMap[layer.name]) {
                    this._nodes.push(new KerasNode(layer.class_name, layer.config, layer._inputs, layer._outputs, rootGroup));
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
            var inputs = [ connection ];
            var name = id.toString();
            if (id == 0) {
                this.translateInput(layer, input);
            }
            id++;
            if (layer.config && layer.config.name) {
                name = layer.config.name;
            }
            connection = name;
            var outputs = [ connection ];
            this._nodes.push(new KerasNode(layer.class_name, layer.config, inputs, outputs, rootGroup));
        });
        this._outputs.push({ 
            id: connection,
            name: connection
        });
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

    constructor(operator, config, inputs, outputs, rootGroup) {
        this._operator = operator;
        this._config = config;
        this._inputs = inputs;
        this._outputs = outputs;

        if (operator == 'Bidirectional' || operator == 'TimeDistributed') {
            if (this._config && this._config.layer) {
                var inner = this._config.layer;
                this._inner = new KerasNode(inner.class_name, inner.config, [], [], null);
            }
        }

        var name = this.name;
        this._initializers = {};
        if (rootGroup) {
            var model_weights = rootGroup.group('model_weights');
            if (model_weights) {
                var group = model_weights.group(name);
                if (group) {
                    var weight_names = group.attributes.weight_names;
                    if (weight_names) {
                        weight_names.forEach((weight_name) => {
                            var weight_variable = group.group(weight_name);
                            if (weight_variable) {
                                var variable = weight_variable.value;
                                if (variable) {
                                    this._inputs.push(weight_name);
                                    this._initializers[weight_name] = new KerasTensor(variable);
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
        if (this._config && this._config.name) {
            return this._config.name;
        }
        debugger;
        return '';
    }

    get inputs() {
        var inputs = KerasOperatorMetadata.operatorMetadata.getInputs(this, this._inputs);
        inputs.forEach((input) => {
            input.connections.forEach((connection) => {
                var initializer = this._initializers[connection.id];
                if (initializer) {
                    connection.type = initializer.type;
                    connection.initializer = initializer;
                }
            });
        });
        return inputs;
    }

    get outputs() {
        var results = [];
        this._outputs.forEach((output, index) => {
            var result = { connections: [] };
            result.name = KerasOperatorMetadata.operatorMetadata.getOutputName(this.operator, index);
            result.connections.push({ id: output });
            results.push(result);
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

    get inner() {
        return this._inner;
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

class KerasTensor {

    constructor(variable) {
        this._variable = variable;
    }

    get title() {
        return 'Initializer';
    }

    get type() {
        return this._variable.type + JSON.stringify(this._variable.shape);
    }

    get value() {
        var rawData = this._variable.rawData;
        if (rawData) {
            switch (this._variable.type) {
                case 'float16':
                    this._precision = 16;
                    break;
                case 'float32':
                    this._precision = 32;
                    break;
                case 'float64':
                    this._precision = 64;
                    break;
                default:
                    return 'Tensor data type is not supported.';
            }
            this._shape = this._variable.shape;
            this._rawData = new DataView(rawData.buffer, rawData.byteOffset, rawData.byteLength);
            this._index = 0;
            this._count = 0;
            var result = this.read(0);
            delete this._index;
            delete this._count;
            delete this._rawData;
            delete this._shape;
            delete this._precision;
            return JSON.stringify(result, null, 4);
        }
        return 'Tensor data is empty.';
    }

    read(dimension) {
        var results = [];
        var size = this._shape[dimension];
        if (dimension == this._shape.length - 1) {
            for (var i = 0; i < size; i++) {
                if (this._count > 10000) {
                    results.push('...');
                    return results;
                }
                if (this._rawData) {
                    switch (this._precision) {
                        case 16:
                            results.push(KerasTensor.decodeNumberFromFloat16(this._rawData.getUint16(this._index, true)));
                            this._index += 2;
                            break;
                        case 32:
                            results.push(this._rawData.getFloat32(this._index, true));
                            this._index += 4;
                            break;
                        case 64:
                            results.push(this._rawData.getFloat64(this._index, true));
                            this._index += 8;
                            break;
                    }
                    this._count++;
                }
            }
        }
        else {
            for (var j = 0; j < size; j++) {
                if (this._count > 10000) {
                    results.push('...');
                    return results;
                }
                results.push(this.read(dimension + 1));
            }
        }
        return results;
    }

    static decodeNumberFromFloat16(value) {
        var s = (value & 0x8000) >> 15;
        var e = (value & 0x7C00) >> 10;
        var f = value & 0x03FF;
        if(e == 0) {
            return (s ? -1 : 1) * Math.pow(2, -14) * (f / Math.pow(2, 10));
        }
        else if (e == 0x1F) {
            return f ? NaN : ((s ? -1 : 1) * Infinity);
        }
        return (s ? -1 : 1) * Math.pow(2, e-15) * (1 + (f / Math.pow(2, 10)));
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

    getInputs(node, inputs) {
        var results = [];
        var operator = node.operator;
        var schema = this._map[operator];
        var inner = node.inner;
        var innerOperator = inner ? inner.operator : null;
        var innerSchema = innerOperator ? this._map[innerOperator] : null;
        var index = 0;
        while (index < inputs.length) {
            var result = { connections: [] };
            var count = 1;
            var name = null;
            if (!innerSchema || index == 0)
            {
                if (schema && schema.inputs && index < schema.inputs.length) {
                    var input = schema.inputs[index];
                    name = input.name;
                    if (schema.inputs[index].option == 'variadic') {
                        count = inputs.length - index;
                    }
                }
            }
            else {
                switch (operator) {
                    case 'Bidirectional':
                        var innerIndex = index;
                        if (innerSchema && innerSchema.inputs) {
                            if (innerIndex < innerSchema.inputs.length) {
                                name = 'forward_' + innerSchema.inputs[innerIndex].name;
                            }
                            else {
                                innerIndex = innerIndex - innerSchema.inputs.length + 1;
                                if (innerIndex < innerSchema.inputs.length) {
                                    name = 'backward_' + innerSchema.inputs[innerIndex].name;
                                }
                            }
                        }
                        result.hidden = true;
                        break;
                    case 'TimeDistributed':
                        if (innerSchema && innerSchema.inputs && index < innerSchema.inputs.length) {
                            name = innerSchema.inputs[index].name;
                        }
                        break;
                }
            }
            result.name = name ? name : '(' + index.toString() + ')';
            var array = inputs.slice(index, index + count);
            for (var j = 0; j < array.length; j++) {
                result.connections.push({ id: array[j] });
            }
            index += count;
            results.push(result);
        }
        return results;
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
        if (operator == 'Bidirectional' || operator == 'TimeDistributed') {
            if (attributeName == 'layer') {
                return false;
            }
        }
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
                    return !KerasOperatorMetadata.isEquivalent(attribute.default, attributeValue);
                }
            }
        }
        return true;
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