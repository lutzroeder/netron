/* jshint esversion: 6 */
/* eslint "indent": [ "error", 4, { "SwitchCase": 1 } ] */

var keras = keras || {};
var base = base || require('./base');
var marked = marked || require('marked');

keras.ModelFactory = class {

    match(context) {
        var identifier = context.identifier;
        var extension = identifier.split('.').pop().toLowerCase();
        var buffer = null;
        if (extension == 'keras' || extension == 'h5' || extension == 'hdf5') {
            // Filter PyTorch models published with incorrect .h5 file extension.
            buffer = context.buffer;
            var torch = [ 0x8a, 0x0a, 0x6c, 0xfc, 0x9c, 0x46, 0xf9, 0x20, 0x6a, 0xa8, 0x50, 0x19 ];
            if (buffer && buffer.length > 14 && buffer[0] == 0x80 && torch.every((v, i) => v == buffer[i + 2])) {
                return false;
            }
            return true;
        }
        if (extension == 'model') {
            buffer = context.buffer;
            var hdf5 = [ 0x89, 0x48, 0x44, 0x46 ];
            return (buffer && buffer.length > hdf5.length && hdf5.every((v, i) => v == buffer[i]));
        }
        if (extension == 'json' && !identifier.endsWith('-symbol.json')) {
            var json = context.text;
            if (json.indexOf('"mxnet_version":', 0) == -1) {
                try {
                    var root = JSON.parse(json);
                    if (root && root.nodes && root.arg_nodes && root.heads) {
                        return false;
                    }
                    if (root && root.modelTopology && root.modelTopology.model_config) {
                        root = root.modelTopology.model_config;
                    }
                    if (root && root.class_name) {
                        return true;
                    }
                }
                catch (err) {
                    // continue regardless of error
                }
            }
        }
        return false;
    }

    open(context, host, callback) {
        host.require('./hdf5', (err, hdf5) => {
            if (err) {
                callback(err, null);
                return;
            }
            var format = 'Keras';
            var model_config = null;
            var rootGroup = null;
            var rootJson = null;
            var identifier = context.identifier;
            try {
                switch (identifier.split('.').pop().toLowerCase()) {
                    case 'keras':
                    case 'h5':
                    case 'hdf5':
                    case 'model':
                        var file = new hdf5.File(context.buffer);
                        rootGroup = file.rootGroup;
                        if (!rootGroup.attribute('model_config')) {
                            callback(new keras.Error('HDF5 file does not contain a Keras \'model_config\' graph. Use \'save()\' instead of \'save_weights()\' to save both the graph and weights.'), null);
                            return;
                        }
                        model_config = JSON.parse(rootGroup.attribute('model_config'));
                        break;
                    case 'json':
                        model_config = JSON.parse(context.text);
                        if (model_config && model_config.modelTopology && model_config.modelTopology.model_config) {
                            format = 'TensorFlow.js ' + format;
                            rootJson = model_config;
                            model_config = model_config.modelTopology.model_config;
                        }
                        break;
                }
            }
            catch (error) {
                var message = error && error.message ? error.message : error.toString();
                message = message.endsWith('.') ? message.substring(0, message.length - 1) : message;
                callback(new keras.Error(message + " in '" + identifier + "'."), null);
                return;
            }
    
            if (!model_config) {
                callback(new keras.Error('\'model_config\' is not present.'));
                return;
            }
            if (!model_config.class_name) {
                callback(new keras.Error('\'class_name\' is not present.'), null);
                return;
            }
    
            keras.Metadata.open(host, (err, metadata) => {
                try {
                    var model = new keras.Model(metadata, format, model_config, rootGroup, rootJson);
                    callback(null, model);
                    return;
                }
                catch (error) {
                    var message = error && error.message ? error.message : error.toString();
                    message = message.endsWith('.') ? message.substring(0, message.length - 1) : message;
                    callback(new keras.Error(message + " in '" + identifier + "'."), null);
                    return;
                }
            });
        });
    }
};

keras.Model = class {

    constructor(metadata, format, model_config, rootGroup, rootJson) {
        this._format = format;
        this._graphs = [];

        var initializer;
        var weights = {};
        if (rootGroup) {
            var version = rootGroup.attribute('keras_version');
            if (version) {
                this._version = version;
            }
            var backend = rootGroup.attribute('backend');
            if (backend) {
                this._backend = backend;
            }
            var model_weights_group = rootGroup.group('model_weights');
            if (model_weights_group) {
                model_weights_group = new keras.Group(model_weights_group);
                var layer_names = model_weights_group.attribute('layer_names');
                var layer_names_map = {};
                var layer_name;
                for (layer_name of layer_names) {
                    layer_names_map[layer_name] = true;
                }
                for (layer_name of layer_names) {
                    var layer_weights = model_weights_group.group(layer_name);
                    var weight_names = layer_weights.attribute('weight_names');
                    if (layer_weights && weight_names && weight_names.length > 0) {
                        for (var weight_name of weight_names) {
                            var group = layer_weights.group(weight_name);
                            if (group) {
                                var variable = group.value;
                                if (variable) {
                                    var parts = weight_name.split('/');
                                    parts.pop();
                                    initializer = new keras.Tensor(weight_name, variable.type, variable.shape, variable.rawData, '');
                                    var match = false;
                                    while (parts.length > 0) {
                                        var name = parts.join('/');
                                        if (layer_names_map[name]) {
                                            match = true;
                                        }
                                        weights[name] = weights[name] || [];
                                        weights[name].push(initializer);
                                        parts.shift();
                                    }
                                    if (!match) {
                                        weights[layer_name] = weights[layer_name] || [];
                                        weights[layer_name].push(initializer);
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        else if (rootJson) {
            if (rootJson.modelTopology && rootJson.modelTopology.keras_version) {
                this._version = rootJson.modelTopology.keras_version;
            }
            if (rootJson.modelTopology && rootJson.modelTopology.backend) {
                this._backend = rootJson.modelTopology.backend;
            }
            if (rootJson.weightsManifest) {
                for (var manifest of rootJson.weightsManifest) {
                    for (var weight of manifest.weights) {
                        var p = weight.name.split('/');
                        p.pop();
                        initializer = new keras.Tensor(weight.name, weight.dtype, weight.shape, null, manifest.paths.join(';'));
                        while (p.length > 0) {
                            var weightName = p.join('/');
                            weights[weightName] = weights[weightName] || [];
                            weights[weightName].push(initializer);
                            p.shift();
                        }
                    }
                }
            }
        }
        
        this._activeGraph = new keras.Graph(metadata, model_config, weights);
        this._graphs.push(this._activeGraph);
    }

    get name() {
        return null;
    }

    get description() {
        return null;
    }

    get format() {
        return this._format + (this._version ? (' v' + this._version) : '');
    }

    get runtime() {
        return this._backend;
    }

    get graphs() {
        return this._graphs;
    }
};

keras.Graph = class {

    constructor(metadata, model, weights) {
        if (model.name) {
            this._name = model.name;
        }
        else if (model.config && model.config.name) {
            this._name = model.config.name;
        }
        this._metadata = metadata;
        this._inputs = [];
        this._outputs = [];
        this._nodes = [];
        this._groups = false;

        switch (model.class_name) {
            case 'Sequential':
                this._loadSequential(model.config, weights, '', null, null);
                break;
            case 'Model':
                this._loadModel(model.config, weights, '', null, null);
                break;
            default:
                throw new keras.Error('\'' + model.class_name + '\' is not supported.');
        }
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

    _loadModel(config, weights, group, inputs, outputs) {
        if (group) {
            this._groups = true;
        }
        if (config.layers) {
            var nodeMap = {};
            var layer;
            for (layer of config.layers) {
                if (layer.name) {
                    if (!nodeMap[layer.name]) {
                        nodeMap[layer.name] = layer;
                        layer._inputs = [];
                        layer._outputs = [];
                    }
                }
            }
            for (layer of config.layers) {
                if (layer.inbound_nodes) {
                    for (var inbound_node of layer.inbound_nodes) {
                        for (var inbound_connection of inbound_node) {
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
                        }
                    }
                }
            }
        }
        var input_layers = config.input_layers;
        if (input_layers) {
            for (var i = 0; i < input_layers.length; i++) {
                var input_layer = input_layers[i];
                var name = input_layer[0];
                var type = null;
                var node = nodeMap[name];
                if (node && node.class_name == 'InputLayer') {
                    type = this._getInputType(node);
                    delete nodeMap[name];
                }
                if (inputs && i < inputs.length) {
                    if (config.layers) {
                        for (layer of config.layers) {
                            if (layer._inputs) {
                                layer._inputs = layer._inputs.map((input) => {
                                    if (input == name) {
                                        return inputs[i];
                                    }
                                    return input;
                                });
                            }
                        }
                    }
                }
                else {
                    this._inputs.push(new keras.Argument(name, true, [ new keras.Connection(name, type, null) ])); 
                }
            }
        }
        var output_layers = config.output_layers;
        if (output_layers) {
            for (var j = 0; j < output_layers.length; j++) {
                var output_layer = output_layers[j];
                var outputName = output_layer[0];
                var outputNode = nodeMap[outputName];
                var addGraphOutput = true;
                if (outputs && j < outputs.length) {
                    outputName = outputs[j];
                    addGraphOutput = false;
                }
                if (outputNode) {
                    var outputIndex = output_layer[2];
                    if (outputIndex != 0) {
                        outputName += ':' + outputIndex.toString();
                    }
                    while (outputIndex >= outputNode._outputs.length) {
                        outputNode._outputs.push('');
                    }
                    outputNode._outputs[outputIndex] = outputName;
                }
                if (addGraphOutput) {
                    this._outputs.push(new keras.Argument(outputName, true, [ new keras.Connection(outputName, null, null) ]));
                }
            }
        }

        if (config.layers) {
            for (layer of config.layers) {
                if (nodeMap[layer.name]) {
                    this._loadNode(layer, layer._inputs, layer._outputs, weights, group);
                }
            }
        }
    }

    _loadSequential(config, weights, group, inputs, outputs) {
        if (group) {
            this._groups = true;
        }
        var inputName = 'input';
        var inputType = null;
        var connection = inputName;
        var index = 0;
        var layers = config.layers ? config.layers : config;

        for (var layer of layers) {
            var name = index.toString();
            var nodeInputs = [ connection ];
            if (index == 0) {
                if (inputs && inputs.length > 0) {
                    nodeInputs = [ inputs[0] ];
                }
                else {
                    inputType = this._getInputType(layer);
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

            this._loadNode(layer, nodeInputs, nodeOutputs, weights, group);
        }
        if (!inputs) {
            this._inputs.push(new keras.Argument(inputName, true, [ new keras.Connection(inputName, inputType, null) ]));
        }
        if (connection) {
            this._outputs.push(new keras.Argument(connection, true, [ new keras.Connection(connection, null, null) ]));
        }
    }

    _loadNode(layer, inputs, outputs, weights, group) {
        var class_name = layer.class_name;
        switch (class_name) {
            case 'Sequential':
                this._loadSequential(layer.config, weights, layer.name, inputs, outputs);
                break;
            case 'Model':
                this._loadModel(layer.config, weights, layer.name, inputs, outputs);
                break;
            default:
                var config = layer.config;
                this._nodes.push(new keras.Node(this._metadata, class_name, config, inputs, outputs, group, weights));
                break;
        }
    }

    _getInputType(layer) {
        if (layer && layer.config) {
            var dataType = '?';
            var shape = [];
            var config = layer.config;
            if (config.dtype) {
                dataType = config.dtype;
                delete config.dtype;
            }
            if (config.batch_input_shape) {
                shape = config.batch_input_shape.map(s => s == null ? '?' : s);
                delete config.batch_input_shape;
            }
            return new keras.TensorType(dataType, new keras.TensorShape(shape));
        }
        return null;
    }
};

keras.Argument = class {
    constructor(name, visible, connections) {
        this._name = name;
        this._visible = visible;
        this._connections = connections;
    }

    get name() {
        return this._name;
    }

    get visible() {
        return this._visible;
    }

    get connections() {
        return this._connections;
    }
};

keras.Connection = class {
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
};

keras.Node = class {

    constructor(metadata, operator, config, inputs, outputs, group, weights) {
        if (group) {
            this._group = group;
        }
        this._metadata = metadata;
        this._operator = operator;
        this._name = (config && config.name) ? config.name : '';
        this._inputs = inputs;
        this._outputs = outputs;
        this._attributes = [];
        this._initializers = {};

        var names = [ this._name ];
        if ((operator == 'Bidirectional' || operator == 'TimeDistributed') && (config && config.layer)) {
            var inner = config.layer;
            delete config.layer;
            this._inner = new keras.Node(this._metadata, inner.class_name, inner.config, [], [], null, null);
            if (operator == 'Bidirectional' && inner.config.name) {
                names = [ this._name + '/forward_' + inner.config.name, this._name + '/backward_' + inner.config.name ];
            }
        }

        if (weights) {
            for (var name of names) {
                var initializers = weights[name];
                if (initializers) {
                    for (var initializer of initializers) {
                        this._inputs.push(initializer.name);
                        this._initializers[initializer.name] = initializer;
                    }
                }
            }
        }

        if (config) {
            for (var attributeName of Object.keys(config)) {
                var attributeValue = config[attributeName];
                if (attributeName != 'name' && attributeValue != null) {
                    this._attributes.push(new keras.Attribute(this._metadata, this.operator, attributeName, attributeValue));
                }
            }
        }
    }

    get operator() {
        return this._operator;
    }

    get name() {
        return this._name;
    }

    get group() {
        return this._group ? this._group : '';
    }

    get category() {
        var schema = this._metadata.getSchema(this._operator);
        return (schema && schema.category) ? schema.category : '';
    }

    get documentation() {
        var schema = this._metadata.getSchema(this._operator);
        if (schema) {
            schema = JSON.parse(JSON.stringify(schema));
            schema.name = this._operator;
            if (schema.description) {
                schema.description = marked(schema.description);
            }
            if (schema.attributes) {
                for (var attribute of schema.attributes) {
                    if (attribute.description) {
                        attribute.description = marked(attribute.description);
                    }
                }
            }
            if (schema.inputs) {
                for (var input of schema.inputs) {
                    if (input.description) {
                        input.description = marked(input.description);
                    }
                }
            }
            if (schema.outputs) {
                for (var output of schema.outputs) {
                    if (output.description) {
                        output.description = marked(output.description);
                    }
                }
            }
            if (schema.references) {
                for (var reference of schema.references) {
                    if (reference) {
                        reference.description = marked(reference.description);
                    }
                }
            }
            return schema;
        }
        return '';
    }

    get inputs() {
        var operator = this.operator;
        var schema = this._metadata.getSchema(operator);
        var innerOperator = this.inner ? this.inner.operator : null;
        var innerSchema = innerOperator ? this._metadata.getSchema(innerOperator) : null;
        var args = [];
        var index = 0;
        while (index < this._inputs.length) {
            var count = 1;
            var name = null;
            var visible = true;
            if (!innerSchema || index == 0)
            {
                if (schema && schema.inputs && index < schema.inputs.length) {
                    var input = schema.inputs[index];
                    name = input.name;
                    visible = input.visible == false ? false : true; 
                    if (schema.inputs[index].option == 'variadic') {
                        count = this._inputs.length - index;
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
                        visible = false;
                        break;
                    case 'TimeDistributed':
                        if (innerSchema && innerSchema.inputs && index < innerSchema.inputs.length) {
                            name = innerSchema.inputs[index].name;
                        }
                        break;
                }
            }
            var argumentName = name ? name : index.toString();
            var connections = [];
            var array = this._inputs.slice(index, index + count);
            for (var j = 0; j < array.length; j++) {
                var id = array[j];
                connections.push(new keras.Connection(id, null, this._initializers[id]));
            }
            index += count;
            args.push(new keras.Argument(argumentName, visible, connections));
        }
        return args;
    }

    get outputs() {
        var schema = this._metadata.getSchema(this.operator);
        return this._outputs.map((output, index) => {
            var outputName = index.toString();
            if (schema && schema.outputs && index < schema.outputs.length && 
                schema.outputs[index] && schema.outputs[index].name) {
                outputName = schema.outputs[index].name;
            }
            return new keras.Argument(outputName, true, [ new keras.Connection(output, null, null) ]);
        });
    }

    get attributes() {
        return this._attributes;
    }

    get inner() {
        return this._inner;
    }
};

keras.Attribute = class {

    constructor(metadata, operator, name, value) {
        this._name = name;
        this._value = value;

        if (typeof value == 'object' && value.class_name && value.config) {
            this._value = () => {
                return value.class_name + '(' + Object.keys(value.config).map(key => {
                    return key + '=' + JSON.stringify(value.config[key]);
                }).join(', ') + ')';
            };
        }

        if (name == 'trainable') {
            this._visible = false;
        }
        else {
            var schema = metadata.getAttributeSchema(operator, this._name);
            if (schema) {
                if (schema.hasOwnProperty('visible') && !schema.visible) {
                    this._visible = false;
                }
                else if (schema.hasOwnProperty('default')) {
                    if (keras.Attribute._isEquivalent(schema.default, value)) {
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

    static _isEquivalent(a, b) {
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
                    if (!keras.Attribute._isEquivalent(a[length], b[length])) {
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
            if (!(b.hasOwnProperty(key) && keras.Attribute._isEquivalent(a[key], b[key]))) {
                return false;
            }
        }
        return true;
    }
};

keras.Tensor = class {

    constructor(name, type, shape, data, reference) {
        this._name = name;
        this._type = new keras.TensorType(type, new keras.TensorShape(shape));
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
        return this._type;
    }

    get reference() {
        return this._reference;
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
        return keras.Tensor._stringify(value, '', '    ');
    }

    _context() {
        var context = {};
        context.index = 0;
        context.count = 0;
        context.state = null;
        if (this._reference) { 
            context.state = 'Tensor reference not implemented.';
            return context;
        }
        if (!this._data) {
            context.state = 'Tensor data is empty.';
            return context;
        }
        switch (this._type.dataType) {
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
                context.state = 'Tensor data type is not supported.';
                break;
        }
        context.dimensions = this._type.shape.dimensions;
        context.rawData = new DataView(this._data.buffer, this._data.byteOffset, this._data.byteLength);
        return context;
    }

    _decode(context, dimension) {
        var results = [];
        var size = context.dimensions[dimension];
        if (dimension == context.dimensions.length - 1) {
            for (var i = 0; i < size; i++) {
                if (context.count > context.limit) {
                    results.push('...');
                    return results;
                }
                if (context.rawData) {
                    switch (context.precision) {
                        case 16:
                            results.push(context.rawData.getFloat16(context.index, true));
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
                results.push(this._decode(context, dimension + 1));
            }
        }
        return results;
    }

    static _stringify(value, indentation, indent) {
        if (Array.isArray(value)) {
            var result = [];
            result.push(indentation + '[');
            var items = value.map((item) => keras.Tensor._stringify(item, indentation + indent, indent));
            if (items.length > 0) {
                result.push(items.join(',\n'));
            }
            result.push(indentation + ']');
            return result.join('\n');
        }
        if (typeof value == 'string') {
            return indentation + value;
        }
        if (value == Infinity) {
            return indentation + 'Infinity';
        }
        if (value == -Infinity) {
            return indentation + '-Infinity';
        }
        if (isNaN(value)) {
            return indentation + 'NaN';
        }
        return indentation + value.toString();
    }
};

keras.TensorType = class {

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
        return this._dataType + this._shape.toString();
    }
};

keras.TensorShape = class {

    constructor(dimensions) {
        this._dimensions = dimensions;
    }

    get dimensions() {
        return this._dimensions;
    }

    toString() {
        return this._dimensions ? ('[' + this._dimensions.map((dimension) => dimension.toString()).join(',') + ']') : '';
    }
};

keras.Metadata = class {

    static open(host, callback) {
        if (keras.Metadata._metadata) {
            callback(null, keras.Metadata._metadata);
        }
        else {
            host.request(null, 'keras-metadata.json', 'utf-8', (err, data) => {
                keras.Metadata._metadata = new keras.Metadata(data);
                callback(null, keras.Metadata._metadata);
            });
        }
    }

    
    constructor(data) {
        this._map = {};
        this._attributeCache = {};
        if (data) {
            var items = JSON.parse(data);
            if (items) {
                for (var item of items) {
                    if (item.name && item.schema) {
                        this._map[item.name] = item.schema;
                    }
                }
            }
        }
    }

    getSchema(operator) {
        return this._map[operator] || null;
    }

    getAttributeSchema(operator, name) {
        var map = this._attributeCache[operator];
        if (!map) {
            map = {};
            var schema = this.getSchema(operator);
            if (schema && schema.attributes && schema.attributes.length > 0) {
                for (var attribute of schema.attributes) {
                    map[attribute.name] = attribute;
                }
            }
            this._attributeCache[operator] = map;
        }
        return map[name] || null;
    }
};

keras.Group = class {

    constructor(group) {
        this._group = group;
    }

    attribute(name) {
        var value = this._group.attribute(name);
        if (!value) {
            if (this._group.attribute(name + '0')) {
                var index = 0;
                value = [];
                for (;;) {
                    var chunk = this._group.attribute(name + index.toString());
                    if (!chunk) {
                        break;
                    }
                    value = value.concat(chunk);
                    index++;
                }
            }
        }
        return value;
    }

    group(name) {
        var value = this._group.group(name);
        if (value) {
            return new keras.Group(value);
        }
        return null;
    }

    get value() {
        return this._group.value;
    }
};

keras.Error = class extends Error {

    constructor(message) {
        super(message);
        this.name = 'Error loading Keras model.';
    }
};

if (typeof module !== 'undefined' && typeof module.exports === 'object') {
    module.exports.ModelFactory = keras.ModelFactory;
}