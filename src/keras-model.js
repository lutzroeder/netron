/*jshint esversion: 6 */

class KerasModelFactory {

    match(buffer, identifier) {
        var extension = identifier.split('.').pop();
        if (extension == 'keras' || extension == 'h5') {
            return true;
        }
        if (extension == 'json') {
            var decoder = new TextDecoder('utf-8');
            var json = decoder.decode(buffer);
            if (!json.includes('\"mxnet_version\":')) {
                return true;
            }
        }
        return false;
    }

    open(buffer, identifier, host, callback) {
        host.import('/hdf5.js', (err) => {
            if (err) {
                callback(err, null);
                return;
            }
            try {
                var format = 'Keras';
                var rootGroup = null;
                var rootJson = null;
                var model_config = null;
                var extension = identifier.split('.').pop();
                if (extension == 'keras' || extension == 'h5') {
                    var file = new hdf5.File(buffer);
                    rootGroup = file.rootGroup;
                    var modelConfigJson = rootGroup.attributes.model_config;
                    if (!modelConfigJson) {
                        callback(new KerasError('HDF5 file does not contain a \'model_config\' graph. Use \'save()\' instead of \'save_weights()\' to save both the graph and weights.'), null);
                        return;
                    }
                    model_config = JSON.parse(modelConfigJson);
                }
                else if (extension == 'json') {
                    var decoder = new window.TextDecoder('utf-8');
                    var json = decoder.decode(buffer);
                    model_config = JSON.parse(json);
                    if (model_config && model_config.modelTopology && model_config.modelTopology.model_config) {
                        format = 'TensorFlow.js ' + format;
                        rootJson = model_config;
                        model_config = model_config.modelTopology.model_config;
                    }
                }
                if (!model_config) {
                    callback(new KerasError('\'model_config\' is not present.'));
                    return;
                }
                if (!model_config.class_name) {
                    callback(new KerasError('\'class_name\' is not present.'), null);
                    return;
                }
                var model = new KerasModel(format, model_config, rootGroup, rootJson);
                KerasOperatorMetadata.open(host, (err, metadata) => {
                    callback(null, model);
                });
            }
            catch (error) {
                callback(new KerasError(error.message), null);
            }
        });
    }
}

class KerasModel {

    constructor(format, model_config, rootGroup, rootJson) {
        this._format = format;
        this._graphs = [];

        var model_weights = null;
        var weightsManifest = null;
        if (rootGroup) {
            if (rootGroup.attributes.keras_version) {
                this._version = rootGroup.attributes.keras_version;
            }
            if (rootGroup.attributes.backend) {
                this._backend = rootGroup.attributes.backend;
            }
            model_weights = rootGroup.group('model_weights');
        }
        else if (rootJson) {
            if (rootJson.modelTopology && rootJson.modelTopology.keras_version) {
                this._version = rootJson.modelTopology.keras_version;
            }
            if (rootJson.modelTopology && rootJson.modelTopology.backend) {
                this._backend = rootJson.modelTopology.backend;
            }
            if (rootJson.weightsManifest) {
                weightsManifest = {};
                rootJson.weightsManifest.forEach((manifest) => {
                    var match = false;
                    var key = null;
                    manifest.weights.forEach((weights) => {
                        var name = weights.name.split('/').shift();
                        if (key == null) {
                            key = name;
                            match = true;
                        }
                        else if (key != name) {
                            match = false;
                        }
                    });
                    if (match) {
                        weightsManifest[key] = manifest;
                    }
                });
            }
        }

        this._activeGraph = new KerasGraph(model_config, model_weights, weightsManifest);
        this._graphs.push(this._activeGraph);
    }

    get properties() {
        var results = [];

        var format = this._format + (this._version ? (' v' + this._version) : '');
        results.push({ name: 'format', value: format });

        if (this._backend) {
            results.push({ name: 'backend', value: this._backend });
        }

        return results;
    }

    get graphs() {
        return this._graphs;
    }
}

class KerasGraph {

    constructor(model, model_weights, weightsManifest) {
        if (model.name) {
            this._name = model.name;
        }
        else if (model.config && model.config.name) {
            this._name = model.config.name;
        }
        this._inputs = [];
        this._outputs = [];
        this._nodes = [];
        this._groups = false;
        this._operators = {};

        switch (model.class_name) {
            case 'Sequential':
                this._loadSequential(model.config, model_weights, weightsManifest, '', null, null);
                break;
            case 'Model':
                this._loadModel(model.config, model_weights, weightsManifest, '', null, null);
                break;
            default:
                throw new KerasError('\'' + model.class_name + '\' is not supported.');
        }
    }

    get operators() {
        return this._operators;
    }

    get name() {
        return this._name;
    }

    get groups() {
        return this._groups ? true : false;
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

    _loadModel(config, model_weights, weightsManifest, group, inputs, outputs) {
        if (group) {
            this._groups = true;
        }
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
            config.input_layers.forEach((input_layer, index) => {
                var name = input_layer[0];
                var graphInput = {
                    id: name,
                    name: name
                };
                var node = nodeMap[name];
                if (node && node.class_name == 'InputLayer') {
                    this._loadInput(node, graphInput);
                    delete nodeMap[name];
                }
                if (inputs && index < inputs.length) {
                    if (config.layers) {
                        config.layers.forEach((layer) => {
                            if (layer._inputs) {
                                layer._inputs = layer._inputs.map((input) => {
                                    if (input == graphInput.id) {
                                        return inputs[index];
                                    }
                                    return input;
                                });
                            }
                        });
                    }            
                }
                else {
                    this._inputs.push(graphInput); 
                }
            });
        }
        if (config.output_layers) {
            config.output_layers.forEach((output_layer, index) => {
                var inputName = output_layer[0];
                var inputNode = nodeMap[inputName];
                var addGraphOutput = true;
                if (outputs && index < outputs.length) {
                    inputName = outputs[index];
                    addGraphOutput = false;
                }
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
                if (addGraphOutput) {
                    this._outputs.push({
                        id: inputName,
                        type: 'T',
                        name: inputName
                    });
                }
            });
        }
        if (config.layers) {
            config.layers.forEach((layer) => {
                this._operators[layer.class_name] = (this._operators[layer.class_name] || 0) + 1; 
                if (nodeMap[layer.name]) {
                    this._loadNode(layer, layer._inputs, layer._outputs, model_weights, weightsManifest, group);
                }
            });
        }
    }

    _loadSequential(config, model_weights, weightsManifest, group, inputs, outputs) {
        if (group) {
            this._groups = true;
        }
        var connection = 'input';
        if (!inputs) {
            var input = {
                id: connection,
                name: connection
            };
            this._inputs.push(input);
        }
        var index = 0;
        config.forEach((layer) => {
            this._operators[layer.class_name] = (this._operators[layer.class_name] || 0) + 1; 
            var name = index.toString();
            var nodeInputs = [ connection ];
            if (index == 0) {
                if (inputs && inputs.length > 0) {
                    nodeInputs = [ inputs[0] ];
                }
                else {
                    this._loadInput(layer, input);
                }
            }
            index++;
            if (layer.config && layer.config.name) {
                name = layer.config.name;
            }
            connection = name;
            var nodeOutputs = [ connection ];
            if (index == config.length) {
                if (outputs && outputs.length > 0) {
                    nodeOutputs = [ outputs[0] ];                    
                    connection = null;           
                }
            }

            this._loadNode(layer, nodeInputs, nodeOutputs, model_weights, weightsManifest, group);
        });
        if (connection) {
            this._outputs.push({ 
                id: connection,
                type: 'T',
                name: connection
            });
        }
    }

    _loadNode(layer, inputs, outputs, model_weights, weightsManifest, group) {
        var class_name = layer.class_name;
        switch (class_name) {
            case 'Sequential':
                this._loadSequential(layer.config, model_weights, weightsManifest, layer.name, inputs, outputs);
                break;
            case 'Model':
                this._loadModel(layer.config, model_weights, weightsManifest, layer.name, inputs, outputs);
                break;
            default:
                var config = layer.config;
                this._nodes.push(new KerasNode(class_name, config, inputs, outputs, group, model_weights, weightsManifest));
                break;
        }
    }

    _loadInput(layer, input) {
        input.type = '';
        if (layer && layer.config) {
            var config = layer.config;
            if (config.dtype) {
                input.type = config.dtype;
                delete config.dtype;
            }
            if (config.batch_input_shape) {
                var shape = config.batch_input_shape;
                shape = shape.map(s => s == null ? '?' : s).join(',');
                input.type = input.type + '[' + shape + ']';
                delete config.batch_input_shape;
            }
        }
    }
}

class KerasNode {

    constructor(operator, config, inputs, outputs, group, model_weights, weightsManifest) {
        if (group) {
            this._group = group;
        }
        this._operator = operator;
        this._config = config;
        this._inputs = inputs;
        this._outputs = outputs;

        if (operator == 'Bidirectional' || operator == 'TimeDistributed') {
            if (this._config && this._config.layer) {
                var inner = this._config.layer;
                this._inner = new KerasNode(inner.class_name, inner.config, [], [], null, null);
            }
        }

        var name = this.name;
        this._initializers = {};

        if (model_weights) {
            var weights = null;
            if (group) {
                weights = model_weights.group(group);
            }
            else if (config) {
                weights = model_weights.group(config.name);
            }
            if (weights) {
                var weight_names = weights.attributes.weight_names;
                if (weight_names) {
                    if (group) {
                        weight_names = weight_names.filter(weight => weight.startsWith(name + '/'));
                    }
                    weight_names.forEach((weight_name) => {
                        var weight_variable = weights.group(weight_name);
                        if (weight_variable) {
                            var variable = weight_variable.value;
                            if (variable) {
                                this._inputs.push(weight_name);
                                this._initializers[weight_name] = new KerasTensor(variable.type, variable.shape, variable.rawData, '');
                            }
                        }
                    });
                }
            }
        }
        else if (weightsManifest) {
            var manifest = weightsManifest[name];
            if (manifest) {
                manifest.weights.forEach((weights) => {
                    if (weights.name) {
                        this._inputs.push(weights.name);
                        this._initializers[weights.name] = new KerasTensor(weights.dtype, weights.shape, null, manifest.paths.join(';'));
                    }
                });
            } 
        }
    }

    get operator() {
        return this._operator;
    }

    get documentation() {
        return KerasOperatorMetadata.operatorMetadata.getOperatorDocumentation(this.operator);
    }

    get name() {
        if (this._config && this._config.name) {
            return this._config.name;
        }
        debugger;
        return '';
    }

    get group() {
        return this._group ? this._group : '';
    }

    get category() {
        return KerasOperatorMetadata.operatorMetadata.getOperatorCategory(this.operator);
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
                    var visible = KerasOperatorMetadata.operatorMetadata.getAttributeVisible(this.operator, name, value);
                    results.push(new KerasAttribute(name, value, visible));
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

    constructor(name, value, visible) {
        this._name = name;
        this._value = value;
        if (!visible) {
            this._hidden = true;
        }
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

    get visible() {
        return this._hidden ? false : true;
    }
}

class KerasTensor {

    constructor(type, shape, data, reference) {
        this._type = type;
        this._shape = shape;
        this._data = data;
        this._reference = reference;
    }

    get kind() {
        return 'Weights';
    }

    get name() {
        return this._name;
    }

    get type() {
        return this._type + JSON.stringify(this._shape);
    }

    get reference() {
        return this._reference;
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
        if (this._reference) { 
            result.value = 'Tensor reference not implemented.';
            return result;
        }
        if (!this._data) {
            result.error = 'Tensor data is empty.';
            return result;
        }
        var context = {};
        context.index = 0;
        context.count = 0;
        context.limit = limit;
        switch (this._type) {
            case 'float16':
                context.precision = 16;
                break;
            case 'float32':
                context.precision = 32;
                break;
            case 'float64':
                context.precision = 64;
                break;
            default:
                result.error = 'Tensor data type is not supported.';
                return result.error;
        }
        context.rawData = new DataView(this._data.buffer, this._data.byteOffset, this._data.byteLength);
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
                if (context.rawData) {
                    switch (context.precision) {
                        case 16:
                            results.push(KerasTensor._decodeNumberFromFloat16(context.rawData.getUint16(context.index, true)));
                            context.index += 2;
                            break;
                        case 32:
                            results.push(context.rawData.getFloat32(context.index, true));
                            context.index += 4;
                            break;
                        case 64:
                            results.push(context.rawData.getFloat64(context.index, true));
                            context.index += 8;
                            break;
                    }
                    context.count++;
                }
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

    static _decodeNumberFromFloat16(value) {
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
            host.request('/keras-metadata.json', (err, data) => {
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
        return '(' + index.toString() + ')';
    }

    getAttributeVisible(operator, attributeName, attributeValue) {
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
                schema.attributes.forEach(attribute => {
                    schema.attributeMap[attribute.name] = attribute;
                });
            }
            var attribute = schema.attributeMap[attributeName];
            if (attribute) {
                if (attribute.hasOwnProperty('visible')) {
                    return attribute.visible;
                }
                if (attribute.hasOwnProperty('default')) {
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

    getOperatorDocumentation(operator) {
        var schema = this._map[operator];
        if (schema) {
            schema = JSON.parse(JSON.stringify(schema));
            schema.name = operator;
            if (schema.description) {
                schema.description = marked(schema.description);
            }
            if (schema.attributes) {
                schema.attributes.forEach((attribute) => {
                    if (attribute.description) {
                        attribute.description = marked(attribute.description);
                    }
                });
            }
            if (schema.inputs) {
                schema.inputs.forEach((input) => {
                    if (input.description) {
                        input.description = marked(input.description);
                    }
                });
            }
            if (schema.outputs) {
                schema.outputs.forEach((output) => {
                    if (output.description) {
                        output.description = marked(output.description);
                    }
                });
            }
            if (schema.references) {
                schema.references.forEach((reference) => {
                    if (reference) {
                        reference.description = marked(reference.description);
                    }
                });
            }
            return schema;
        }
        return '';
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
        this.name = 'Error loading Keras model.';
    }
}