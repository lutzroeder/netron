/*jshint esversion: 6 */

// Experimental

var tf = tf || {};
var protobuf = protobuf || require('protobufjs');
var marked = marked || require('marked');

tf.ModelFactory = class {

    match(context, host) {
        var identifier = context.identifier;
        var extension = identifier.split('.').pop().toLowerCase();
        if (extension == 'meta') {
            return true;
        }
        var tags = null;
        if (extension == 'pb') {
            if (identifier.endsWith('predict_net.pb') || identifier.endsWith('init_net.pb')) {
                return false;
            }
            if (identifier == 'tfhub_module.pb') {
                var buffer = context.buffer;
                if (buffer && buffer.length == 2 && buffer[0] == 0x08 && buffer[1] == 0x03) {
                    return false;
                }
            }
            tags = context.tags('pb');
            if (Object.keys(tags).length == 0) {
                return false;
            }
            // ignore input_0.pb, output_0.pb
            if (Object.keys(tags).length > 0 &&
                tags.hasOwnProperty(1) && tags[1] == 0 && 
                tags.hasOwnProperty(2) && tags[2] == 0 && 
                tags.hasOwnProperty(9) && tags[9] == 2) {
                return false;
            }
            return true;
        }
        if (extension == 'pbtxt' || extension == 'prototxt') {
            if (identifier.endsWith('predict_net.pbtxt') || identifier.endsWith('predict_net.prototxt') ||
                identifier.endsWith('init_net.pbtxt') || identifier.endsWith('init_net.prototxt')) {
                return false;
            }
            tags = context.tags('pbtxt');
            if (tags.node || tags.saved_model_schema_version || tags.meta_graphs || tags.graph_def) {
                return true;
            }
        }
        return false;
    }

    open(context, host, callback) { 
        host.require('./tf-proto', (err, module) => {
            if (err) {
                callback(err, null);
                return;
            }
            tf.proto = protobuf.roots.tf.tensorflow;
            var graph = null;
            var metaGraph = null;
            var savedModel = null;
            var format = null;
            var identifier = context.identifier; 
            var extension = identifier.split('.').pop().toLowerCase();
            if (extension == 'pbtxt' || extension == 'prototxt') {
                var tags = context.tags('pbtxt');
                if (tags.saved_model_schema_version || tags.meta_graphs) {
                    try {
                        if (identifier.endsWith('saved_model.pbtxt') || identifier.endsWith('saved_model.prototxt')) {
                            savedModel = tf.proto.SavedModel.decodeText(context.text);
                            format = 'TensorFlow Saved Model' + (savedModel.saved_model_schema_version ? (' v' + savedModel.saved_model_schema_version.toString()) : '');
                        }
                    }
                    catch (error) {
                        callback(new tf.Error('File text format is not tensorflow.SavedModel (' + error.message + ').'), null);
                        return;
                    }
                }
                else if (tags.graph_def) {
                    try {
                        if (!savedModel) {
                            metaGraph = tf.proto.MetaGraphDef.decodeText(context.text);
                            savedModel = new tf.proto.SavedModel();
                            savedModel.meta_graphs.push(metaGraph);
                            format = 'TensorFlow MetaGraph';
                        }
                    }
                    catch (error) {
                        callback(new tf.Error('File text format is not tensorflow.MetaGraphDef (' + error.message + ').'), null);
                        return;
                    }
                }
                else if (tags.node) {
                    try {
                        graph = tf.proto.GraphDef.decodeText(context.text);
                        metaGraph = new tf.proto.MetaGraphDef();
                        metaGraph.graph_def = graph;
                        savedModel = new tf.proto.SavedModel();
                        savedModel.meta_graphs.push(metaGraph);
                        format = 'TensorFlow Graph';
                    }
                    catch (error) {
                        callback(new tf.Error('File text format is not tensorflow.GraphDef (' + error.message + ').'), null);
                        return;
                    }
                }
            }
            else {
                try {
                    if (identifier.endsWith('saved_model.pb')) {
                        savedModel = tf.proto.SavedModel.decode(context.buffer);
                        format = 'TensorFlow Saved Model' + (savedModel.saved_model_schema_version ? (' v' + savedModel.saved_model_schema_version.toString()) : '');
                    }
                }
                catch (error) {
                    var buffer = context.buffer;
                    if (buffer.length > 3 && buffer[0] == 0x08 && buffer[1] == 0x01 && buffer[2] == 0x12) {
                        callback(new tf.Error("File format is not tensorflow.SavedModel (" + error.message + ") in '" + identifier + "'."), null);
                        return;
                    }
                }
                try {
                    if (!savedModel && extension == 'meta') {
                        metaGraph = tf.proto.MetaGraphDef.decode(context.buffer);
                        savedModel = new tf.proto.SavedModel();
                        savedModel.meta_graphs.push(metaGraph);
                        format = 'TensorFlow MetaGraph';
                    }
                }
                catch (error) {
                    callback(new tf.Error("File format is not tensorflow.MetaGraphDef (" + error.message + ") in '" + identifier + "'."), null);
                    return;
                }
                try {
                    if (!savedModel) {
                        graph = tf.proto.GraphDef.decode(context.buffer);
                        metaGraph = new tf.proto.MetaGraphDef();
                        metaGraph.graph_def = graph;
                        savedModel = new tf.proto.SavedModel();
                        savedModel.meta_graphs.push(metaGraph);
                        format = 'TensorFlow Graph';
                    }
                }
                catch (error) {
                    callback(new tf.Error("File format is not tensorflow.GraphDef (" + error.message + ") in '" + identifier + "'."), null);
                    return;
                }
            }

            tf.Metadata.open(host, (err, metadata) => {
                try {
                    var model = new tf.Model(metadata, savedModel, format);
                    callback(null, model);
                }
                catch (error) {
                    host.exception(error, false);
                    callback(new tf.Error(error.message), null);
                }
            });
        });
    }
};

tf.Model = class {

    constructor(metadata, model, format) {
        this._model = model;
        this._format = format;
        this._graphs = [];
        for (var i = 0; i < model.meta_graphs.length; i++) {
            var metaGraph = model.meta_graphs[i];
            var name = null;
            if (metaGraph.any_info) {
                name = metaGraph.any_info.toString();
            }
            else if (model.meta_graphs.length > 1) {
                name = '(' + i.toString() + ')';
            }
            this._graphs.push(new tf.Graph(metadata, metaGraph, name));
        }
        this._activeGraph = (this._graphs.length > 0) ? this._graphs[0] : null;
    }

    get format() {
        return this._format;
    }

    get description() {
        return null;
    }

    get graphs() {
        return this._graphs;    
    }
};

tf.Graph = class {

    constructor(metadata, metaGraph, name) {
        this._metaGraph = metaGraph;
        this._version = null;
        this._metadata = new tf.GraphMetadata(metadata, metaGraph.meta_info_def);
        this._name = name;
        this._operators = {};
        this._inputMap = {};
        if (metaGraph.graph_def) {
            var graph = metaGraph.graph_def;
            if (graph.versions) {
                this._version = 'v' + graph.versions.producer.toString();
            }
            else if (graph.version) {
                debugger;
            }
            else if (metaGraph.meta_info_def && metaGraph.meta_info_def.tensorflow_version) {
                this._version = metaGraph.meta_info_def.tensorflow_version;
            }
            if (metaGraph.meta_info_def && metaGraph.meta_info_def.tags) {
                this._tags = metaGraph.meta_info_def.tags.join(', ');
            }
                graph.node.forEach((node) => {
                this._operators[node.op] = (this._operators[node.op] || 0) + 1;
            });
        }
    }

    get operators() {
        return this._operators;
    }

    get name() {
        return this._name;
    }

    get version() {
        return this._version;
    }

    get tags() {
        return this._tags;
    }

    get groups() {
        return false;
        // TODO return true;
    }

    get inputs() {
        this._update();
        return Object.keys(this._inputMap).map((key) => {
            return this._inputMap[key];
        });
    }

    get outputs() {
        this._update();
        return [];
    }

    get nodes() {
        this._update();
        var results = [];
        if (this._metaGraph.graph_def) {
            this._metaGraph.graph_def.node.forEach((node) => {
                if (node.output.filter(output => !output.startsWith('^')) != 0 ||
                    node.input.filter(input => !input.startsWith('^')).length > 0) {
                    var id = node.name;
                    if (!this._initializerMap[id] && !this._inputMap[id] /* && node.op != 'NoOp' */) {
                        results.push(new tf.Node(this, node));
                    }
                }
            });
        }
        return results;
    }

    get metadata() {
        return this._metadata;
    }

    get namespaces() {
        return this._namespaces;
    }

    _update() {
        if (!this._nodeMap && this._metaGraph.graph_def.node) {
            this._nodeMap = {};
            this._namespaces = {};
            var nodes = this._metaGraph.graph_def.node;
            nodes.forEach((node) => {
                var name = node.name;
                this._nodeMap[name] = node;   
                        if (node.op != 'Const') {
                    var lastIndex = name.lastIndexOf('/');
                    if (lastIndex != -1) {
                        var namespace = name.substring(0, lastIndex);
                        this._namespaces[namespace] = true;
                    }
                }
                node.output = [];         
            });
            nodes.forEach((node) => {
                for (var i = 0; i < node.input.length; i++)
                {
                    var split = node.input[i].split(':', 2);
                    var inputName = split[0];
                    if (!inputName.startsWith('^')) {
                        var outputIndex = split.length == 1 ? 0 : parseInt(split[1]);
                        var outputName = inputName;
                        var outputNode = this._nodeMap[outputName];
                        node.input[i] = outputIndex == 0 ? inputName : inputName + ':' + outputIndex.toString();
                        if (outputNode) {
                            for (var j = outputNode.output.length; j <= outputIndex; j++) {
                                outputNode.output.push('');
                            }
                            outputNode.output[outputIndex] = node.input[i];
                        }    
                    }
                    else {
                        var sourceName = inputName.substring(1);
                        var sourceNode = this._nodeMap[sourceName];
                        if (sourceNode) {
                            if (!sourceNode.dependency) {
                                sourceNode.dependency = [];
                            }
                            sourceNode.dependency.push({ 
                                id: inputName, 
                                name: node.name, 
                                operator: node.op
                            });
                        }
                    }
                }
            });
            this._nodeOutputCountMap = {};
            nodes.forEach((node) => {
                this._metadata.getInputs(node).forEach((input) => {
                    var multiple = input.connections.length > 1; 
                    input.connections.forEach((connection) => {
                        if (multiple) {
                            this._nodeOutputCountMap[connection.id] = 'N';
                        }
                        else {
                            var id = connection.id.startsWith('^') ? connection.id.substring(1) : connection.id;
                            var count = this._nodeOutputCountMap[id];
                            if (count != 'N') {
                                count = count ? count : 0;
                                this._nodeOutputCountMap[input] = count + 1;
                            }
                        }
                    });
                });

                node.input.forEach((input) => {
                    input = input.startsWith('^') ? input.substring(1) : input;
                    var count = this._nodeOutputCountMap[input];
                    if (!count) {
                        count = 0;
                    }
                    this._nodeOutputCountMap[input] = count + 1;
                });
            });

            this._initializerMap = {};
            this._metaGraph.graph_def.node.forEach((node) => {
                if (node.op == 'Const' && this._checkEmptyInput(node) && this._checkSingleOutput(node)) {
                    var value = node.attr.value;
                    if (value && value.hasOwnProperty('tensor')) {
                        var output = node.output[0];
                        if (output) {
                            this._initializerMap[output] = new tf.Tensor(value.tensor, node.name, 'Constant');
                        }
                    }
                }
            });
            this._metaGraph.graph_def.node.forEach((node) => {
                if (node.op == 'Identity' && node.input.length == 1 && this._checkSingleOutput(node)) {
                    var input = node.input[0];
                    var tensor = this._initializerMap[input];
                    if (tensor) {
                        var output = node.output[0];
                        this._initializerMap[input] = "-";
                        tensor.kind = 'Identity Constant';
                        this._initializerMap[output] = tensor;
                    }
                }
            });

            this._inputMap = {};
            this._metaGraph.graph_def.node.forEach((node) => {
                if (node.op == 'Placeholder' && node.input.length == 0 && node.output.length == 1) {
                    var dtype = node.attr.dtype;
                    var shape = node.attr.shape;
                    if (dtype && dtype.type && shape && shape.shape) {
                        var type = new tf.TensorType(dtype.type, shape.shape);
                        var connection = new tf.Connection(node.output[0], type, null); 
                        this._inputMap[node.output[0]] = new tf.Argument(node.name, [ connection ]);
                    }
                }
            });
        }
    }

    _getInitializer(input) {
        var initializer = this._initializerMap[input];
        return initializer ? initializer : null;
    }

    _checkEmptyInput(node) {
        var inputs = node.input.filter((input) => !input.startsWith('^'));
        return inputs.length == 0;
    }

    _checkSingleOutput(node) { 
        if (node.output.length != 1) {
            return false;
        }
        var output = node.output[0];
        var count = this._nodeOutputCountMap[output];
        if (count != 1) {
            return false;
        }
        return true;
    }
};

tf.Argument = class {
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
};

tf.Connection = class {
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

tf.Node = class {

    constructor(graph, node) {
        this._graph = graph;
        this._node = node;
        var metadata = graph.metadata;
        this._attributes = [];
        if (node.attr) {
            Object.keys(node.attr).forEach((name) => {
                var value = node.attr[name];
                this._attributes.push(new tf.Attribute(name, value, node.op, metadata));
            });
        }
    }

    get graph() {
        return this._graph;
    }

    get operator() {
        return this._node.op;
    }

    get name() {
        return this._node.name;
    }

    get device() {
        return this._node.device;
    }

    get group() {
        var name = this._node.name;
        if (this._graph.namespaces[name]) {
            return name;
        }
        var lastIndex = name.lastIndexOf('/');
        if (lastIndex != -1) {
            var namespace = name.substring(0, lastIndex);
            if (this._graph.namespaces[namespace]) {
                return namespace;
            }
        }
        return '';
    }

    get description() {
        return null;
    }

    get domain() {
        return null;
    }

    get documentation() {
        var schema = this._graph.metadata.getSchema(this.operator);
        if (schema) {
            schema = JSON.parse(JSON.stringify(schema));
            schema.name = this.operator;
            if (schema.summary) {
                schema.summary = marked(schema.summary);
            }
            if (schema.description) {
                schema.description = marked(schema.description);
            }
            if (schema.inputs) {
                schema.inputs.forEach((input) => {
                    if (input.type) {
                        input.type = tf.Tensor.formatDataType(input.type);
                    }
                    else if (input.typeAttr) {
                        input.type = input.typeAttr;
                    }
                    else if (input.typeListAttr) {
                        input.type = input.typeListAttr;
                    }
                    if (input.description) {
                        input.description = marked(input.description);
                    }
                });
            }
            if (schema.outputs) {
                schema.outputs.forEach((output) => {
                    if (output.type) {
                        output.type = tf.Tensor.formatDataType(output.type);
                    }
                    else if (output.typeAttr) {
                        output.type = output.typeAttr;
                    }
                    else if (output.typeListAttr) {
                        output.type = output.typeListAttr;
                    }
                    if (output.description) {
                        output.description = marked(output.description);
                    }
                });
            }
            if (schema.attributes) {
                schema.attributes.forEach((attribute) => {
                    var description = attribute.description;
                    if (attribute.allowedValues) {
                        var allowedValues = tf.GraphMetadata._formatAttributeValue(attribute.allowedValues);
                        allowedValues = Array.isArray(allowedValues) ? allowedValues : [ allowedValues ];
                        allowedValues = allowedValues.map((item) => '`' + item + '`').join(', ');
                        allowedValues = 'Must be one of the following: ' + allowedValues + '.';
                        description = description ? (allowedValues + ' ' + description) : allowedValues;
                    }
                    if (attribute.defaultValue) {
                        var defaultValue = ƒ._formatAttributeValue(attribute.defaultValue);
                        defaultValue = Array.isArray(defaultValue) ? defaultValue : [ defaultValue ];
                        defaultValue = defaultValue.map((item) => '`' + item + '`').join(', ');
                        defaultValue = 'Defaults to ' + defaultValue + '.';
                        description = description ? (defaultValue + ' ' + description) : defaultValue;
                    }
                    if (description) {
                        attribute.description = marked(description);
                    }
                });
            }
            return schema;
        }
        return null;

    }

    get category() {
        var schema = this._graph.metadata.getSchema(this.operator);
        return (schema && schema.category) ? schema.category : null;
    }

    get inputs() {
        if (this._node.input) {
            var inputs = this._graph.metadata.getInputs(this._node);
            return inputs.map((input) => {
                return new tf.Argument(input.name, input.connections.map((connection) => {
                    var initializer = this._graph._getInitializer(connection.id);
                    return new tf.Connection(connection.id, null, initializer);
                }));
            });          
        }
        return [];
    }

    get outputs() {
        return this._graph.metadata.getOutputs(this._node).map((output) => {
            return new tf.Argument(output.name, output.connections.map((connection) => {
                return new tf.Connection(connection.id, null, null);
            }));
        });
    }

    get dependencies() {
        var results = [];
        if (this._node.dependency) {
            this._node.dependency.forEach((dependency) => {
                results.push(dependency);
            });
        }
        return results;
    }

    get attributes() {
        return this._attributes;
    }
};

tf.Attribute = class { 
    constructor(name, value, operator, metadata) {
        this._name = name;
        this._value = null;
        this._type = null;
        var schema = metadata.getAttributeSchema(operator, name);
        if (value.hasOwnProperty('tensor')) {
            this._type = 'tensor';
            this._value = new tf.Tensor(value.tensor);
        }
        else if (schema && schema.type) {
            this._type = schema.type;
        }
        if (value.hasOwnProperty('type')) {
            this._type = 'type';
            this._value = () => tf.Tensor.formatDataType(value.type);
         }
        else if (value.hasOwnProperty('i')) {
            this._value = value.i;
        }
        else if (value.hasOwnProperty('f')) {
            this._value = value.f;
        }
        else if (value.hasOwnProperty('b')) {
            this._value = value.b;
        }
        else if (value.hasOwnProperty('shape')) {
            this._type = 'shape';
            this._value = new tf.TensorShape(value.shape);
        }
        else if (value.hasOwnProperty('s')) {
            if (value.s.filter(c => c <= 32 && c >= 128).length == 0) {
                this._value = tf.Metadata.textDecoder.decode(value.s);
            }
            else {
                this._value = value.s;
            }
        }
        else if (value.hasOwnProperty('list')) {
            var list = value.list;
            this._value = [];
            if (list.s && list.s.length > 0) {
                if (list.s.length > 65536) {
                    this._value = () => '[...]';
                }
                else {
                    this._value = list.s.map((s) => {
                        if (s.filter(c => c <= 32 && c >= 128).length == 0) {
                            return tf.Metadata.textDecoder.decode(value.s);
                        }
                        return s.map(v => v.toString()).join(', ');    
                    });
                }
            }
            else if (list.i && list.i.length > 0) {
                if (list.i.length > 65536) {
                    this._value = () => '[...]';
                }
                else {
                    this._value = list.i;
                }
            }
            else if (list.f && list.f.length > 0) {
                if (list.f.length > 65536) {
                    this._value = () => '[...]';
                }
                else {
                    this._value = list.f;
                }
            }
            else if (list.type && list.type.length > 0) {
                if (list.type.length > 65536) {
                    this._value = () => '[...]';
                }
                else {
                    this._type = 'type[]';
                    this._value = list.type.map((type) => tf.Tensor.formatDataType(type)); 
                }
            }
            else if (list.shape && list.shape.length > 0) {
                if (list.shape.length > 65536) {
                    this._value = () => '[...]';
                }
                else {
                    this._type = 'shape[]';
                    this._value = list.shape.map((shape) => new tf.TensorShape(shape));
                }
            }
        }

        if (schema) {
            if (schema.hasOwnProperty('visible') && !attributeSchema.visible) {
                this._visible = false;
            }
            else if (schema.hasOwnProperty('default')) {
                var valueText = tf.GraphMetadata._formatAttributeValue(this._value);
                var defaultValueText = tf.GraphMetadata._formatAttributeValue(schema.default);
                if (JSON.stringify(valueText) == JSON.stringify(defaultValueText)) {
                    this._visible = false;
                }
            }
        }
        if (name == '_output_shapes') {
            this._visible = false;
            this._type = 'shape[]';
        }
        if (name == '_class') {
            this._visible = false;
        }
        var attributeVisibleMap = metadata.getAttributeVisibleMap(operator);
        if (attributeVisibleMap[name]) {
            this._visible = false;
        }
        if (this._type == 'list(shape)') {
            this._type = 'shape[]';
        }
    }

    get name() {
        return this._name;
    }

    get type() {
        return this._type;
    }

    get value() {
        return this._value;
    }

    get visible() {
        return this._visible == false ? false : true;
    }
};

tf.Tensor = class {

    constructor(tensor, name, kind) {
        this._tensor = tensor;
        this._name = name;
        if (kind) {
            this._kind = kind;
        }
        this._type = new tf.TensorType(this._tensor.dtype, this._tensor.tensor_shape);
    }

    get name() {
        return this._name;
    }

    get type() {
        return this._type;
    }

    get kind() {
        return this._kind;
    }

    set kind(value) {
        this._kind = value;
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
        context.size = 1;

        if (!this._tensor.dtype) {
            context.state = 'Tensor has no data type.';
            return context;
        }
        if (!this._tensor.tensor_shape || !this._tensor.tensor_shape.dim) {
            context.state = 'Tensor has no dimensions.';
            return context;
        }

        this._tensor.tensor_shape.dim.forEach((dim) => {
            context.size = context.size * (dim.size ? dim.size : 0);
        });

        switch (this._tensor.dtype) {
            case tf.proto.DataType.DT_FLOAT:
                if (this._tensor.tensor_content && this._tensor.tensor_content.length > 0) {
                    context.rawData = new DataView(this._tensor.tensor_content.buffer, this._tensor.tensor_content.byteOffset, this._tensor.tensor_content.byteLength);
                }
                else if (this._tensor.float_val && this._tensor.float_val.length == context.size) {
                    context.data = this._tensor.float_val;
                }
                else {
                    context.state = 'Tensor data is empty.';
                }
                break;
            case tf.proto.DataType.DT_QINT8:
            case tf.proto.DataType.DT_QUINT8:
                if (this._tensor.tensor_content && this._tensor.tensor_content.length > 0) {
                    context.rawData = new DataView(this._tensor.tensor_content.buffer, this._tensor.tensor_content.byteOffset, this._tensor.tensor_content.byteLength);
                }
                else {
                    context.state = 'Tensor data is empty.';
                }
                break;
            case tf.proto.DataType.DT_INT32:
            case tf.proto.DataType.DT_UINT32:
                if (this._tensor.tensor_content && this._tensor.tensor_content.length > 0) {
                    context.rawData = new DataView(this._tensor.tensor_content.buffer, this._tensor.tensor_content.byteOffset, this._tensor.tensor_content.byteLength);
                }
                else if (this._tensor.int_val && this._tensor.int_val.length == context.size) {
                    context.data = this._tensor.int_val;
                }
                else {
                    context.state = 'Tensor data is empty.';
                }
                break;
            case tf.proto.DataType.DT_STRING:
                if (this._tensor.tensor_content && this._tensor.tensor_content.length > 0) {
                    result.state = 'Tensor data type is not implemented.';
                }
                else if (this._tensor.string_val && this._tensor.string_val.length == context.size) {
                    context.data = this._tensor.string_val;
                }
                else {
                    context.state = 'Tensor data is empty.';
                }
                break;
            case tf.proto.DataType.DT_BOOL:
                context.state = "Tensor data type 'bool' is not implemented.";
                break;
            default:
                context.state = "Tensor data type '" + this._tensor.dtype + "'is not implemented.";
                break;
        }

        context.shape = this._tensor.tensor_shape.dim.map((dim) => dim.size);
        return context;
    }

    _decode(context, dimension) {
        var shape = context.shape;
        if (shape.length == 0) {
            shape = [ 1 ];
        }
        var results = [];
        var size = shape[dimension];
        if (dimension == shape.length - 1) {
            for (var i = 0; i < size; i++) {
                if (context.count > context.limit) {
                    results.push('...');
                    return results;
                }
                if (context.data) {
                    results.push(this._decodeDataValue(context));
                    context.count++;
                }
                else {
                    if (context.rawData) {
                        switch (this._tensor.dtype)
                        {
                            case tf.proto.DataType.DT_FLOAT:
                                results.push(context.rawData.getFloat32(context.index, true));
                                context.index += 4;
                                context.count++;
                                break;
                            case tf.proto.DataType.DT_INT32:
                                results.push(context.rawData.getInt32(context.index, true));
                                context.index += 4;
                                context.count++;
                                break;
                            case tf.proto.DataType.DT_UINT32:
                                results.push(context.rawData.getUInt32(context.index, true));
                                context.index += 4;
                                context.count++;
                                break;
                            case tf.proto.DataType.DT_QINT8:
                                results.push(context.rawData.getInt8(context.index, true));
                                context.index += 1;
                                context.count++;
                                break;
                            case tf.proto.DataType.DT_QUINT8:
                                results.push(context.rawData.getUint8(context.index, true));
                                context.index += 1;
                                context.count++;
                                break;
                        }
                    }
                }
            }
        }
        else {
            for (var j = 0; j < size; j++) {
                if (context.count > context.limit) {
                    results.push('...');
                    return results;
                }
                results.push(this._decode(context, dimension + 1, shape));
            }
        }
        if (context.shape.length == 0) {
            return results[0];
        }
        return results;
    }

    _decodeDataValue(context) {
        var value = context.data[context.index++];
        if (this._tensor.dtype == tf.proto.DataType.DT_STRING) {
            return tf.Metadata.textDecoder.decode(value);
        }
        return value;
    }

    static formatDataType(type) {
        if (!tf.Tensor.dataType)
        {
            tf.Tensor.dataType = {};
            Object.keys(tf.proto.DataType).forEach((key) => {
                var value = tf.proto.DataType[key];
                key = key.startsWith('DT_') ? key.substring(3) : key;
                tf.Tensor.dataType[value] = key.toLowerCase();
            });
            tf.Tensor.dataType[tf.proto.DataType.DT_HALF] = 'float16';
            tf.Tensor.dataType[tf.proto.DataType.DT_FLOAT] = 'float32';
            tf.Tensor.dataType[tf.proto.DataType.DT_DOUBLE] = 'float64';
        }
        var text = tf.Tensor.dataType[type];
        if (text) { 
            return text;
        }
        return '?';
    }
};

tf.TensorType = class {

    constructor(dtype, shape) {
        this._dtype = dtype;
        this._shape = new tf.TensorShape(shape);
    }

    get dataType() {
        return this._dtype ? tf.Tensor.formatDataType(this._dtype) : '?';
    }

    get shape() {
        return this._shape;
    }

    toString() {
        return this.dataType + this._shape.toString();
    }    
};

tf.TensorShape = class {

    constructor(shape) {
        this._shape = shape;
    }

    get dimensions() {
        if (this._shape && this._shape.dim) {
            if (this._shape.unknown_rank) {
                return null;
            }
            if (this._shape.dim.length == 0) {
                return [];
            }
            if (this._shape.dim.length == 1 && !this._shape.dim[0].size) {
                return [ 0 ];
            }
            return this._shape.dim.map((dim) => (dim.size && dim.size != -1) ? dim.size : '?');
        }
        return null;
    }

    toString() {
        if (this._shape && this._shape.dim) {
            if (this._shape.unknown_rank) {
                return '[-]';
            }
            if (this._shape.dim.length == 0) {
                return '';
            }
            if (this._shape.dim.length == 1 && !this._shape.dim[0].size) {
                return '[0]';
            }
            return '[' + this._shape.dim.map((dim) => (dim.size && dim.size != -1) ? dim.size.toString() : '?').join(',') + ']';
        }
        return '?';
    }
};

tf.GraphMetadata = class {

    constructor(metadata, meta_info_def) {
        this._metadata = metadata;
        this._map = {};
        if (meta_info_def && meta_info_def.strippedOpList && meta_info_def.strippedOpList.op) {
            meta_info_def.strippedOpList.op.forEach((opDef) => {
            });
        }
    }

    getSchema(operator) {
        var schema = this._metadata.getSchema(operator);
        if (!schema) {
            schema = this._map[operator];
        }
        return schema;
    }

    getAttributeSchema(operator, name, value) {
        var schema = this.getSchema(operator);
        if (schema) {
            var attributeMap = schema.attributeMap;
            if (!attributeMap) {
                attributeMap = {};
                if (schema.attributes) {
                    schema.attributes.forEach((attribute) => {
                        attributeMap[attribute.name] = attribute;
                    });
                }
                schema.attributeMap = attributeMap;
            }
            return attributeMap[name] || null;
        }
        return null;        
    }

    getAttributeVisibleMap(operator) {
        var schema = this.getSchema(operator);
        if (schema) {
            var map = schema.__visisbleAttributeMap__;
            if (!map) {
                map = {};
                if (schema.inputs) {
                    schema.inputs.forEach((input) => {
                        if (input.typeAttr) {
                            map[input.typeAttr] = true;
                        }
                        else if (input.typeListAttr) {
                            map[input.typeListAttr] = true;
                        }
                        if (input.numberAttr) {
                            map[input.numberAttr] = true;
                        }
                    });
                }
                if (schema.outputs) {
                    schema.outputs.forEach((output) => {
                        if (output.typeAttr) {
                            map[output.typeAttr] = true;
                        }
                        else if (output.typeListAttr) {
                            map[output.typeListAttr] = true;
                        }
                        if (output.numberAttr) {
                            map[output.numberAttr] = true;
                        }
                    });
                }
                schema.__visisbleAttributeMap__ = map;
            }
            return map;
        }
        return {};
    }

    getInputs(node) {
        var results = [];
        var index = 0;
        var inputs = node.input.filter(input => !input.startsWith('^'));
        var schema = this.getSchema(node.op);
        if (schema && schema.inputs) {
            schema.inputs.forEach((input) => {
                var count = 1;
                if (input.numberAttr) {
                    var number = node.attr[input.numberAttr];
                    if (number && number.i) {
                        count = number.i;
                    }
                }
                var result = {};
                result.name = input.name;
                if (input.type) {
                    result.type = tf.Tensor.formatDataType(input.type);
                }
                else if (input.typeAttr) {
                    result.type = input.typeAttr;
                }
                else if (input.typeListAttr) {
                    result.type = input.typeListAttr;
                }
                result.connections = inputs.slice(index, index + count).map((id) => {
                    return { 
                        id: id
                    };
                });
                results.push(result);
                index += count;
            });
        }
        else {
            inputs.slice(index).forEach((input) => {
                results.push({
                    name: '(' + index.toString() + ')',
                    connections: [ { id: input } ]
                });
                index++;
            });
        }
        return results;
    }

    getOutputs(node) {
        var results = [];
        var index = 0;
        var outputs = node.output;
        var schema = this.getSchema(node.op);
        if (schema && schema.outputs) {
            schema.outputs.forEach((output) => {
                var count = 1;
                if (output.numberAttr) {
                    var number = node.attr[output.numberAttr];
                    if (number && number.i) {
                        count = number.i;
                    }
                }
                var result = {};
                result.name = output.name;
                if (output.type) {
                    result.type = tf.Tensor.formatDataType(output.type);
                }
                else if (output.typeAttr) {
                    result.type = output.typeAttr;
                }
                else if (output.typeListAttr) {
                    result.type = output.typeListAttr;
                }
                result.connections = outputs.slice(index, index + count).map((id) => {
                    return { id: id };
                });
                results.push(result);
                index += count;
            });
        }
        else {
            outputs.slice(index).forEach((output) => {
                results.push({
                    name: '(' + index.toString() + ')',
                    connections: [ { id: output } ]
                });
                index++;
            });
        }
        return results;
    }

    static _formatAttributeValue(value) {
        if (value == null) {
            return null;
        }
        if (value && value.__isLong__) {
            value = value.toNumber();
        }
        if (Array.isArray(value)) {
            return value.map((item) => tf.GraphMetadata._formatAttributeValue(item));
        }
        if (value === Object(value)) {
            switch (value.type) {
                case 'type':
                    return tf.Tensor.formatDataType(value.value);
                case 'shape':
                    return value.value;
                case 'tensor':
                    return value.value;
            }
        }
        if (typeof value === 'string') {
            return '"' + value + '"';
        }
        return value.toString();
    }
};

tf.Metadata = class {

    static open(host, callback) {

        tf.Metadata.textDecoder = tf.Metadata.textDecoder || new TextDecoder('utf-8');

        if (tf.Metadata._metadata) {
            callback(null, tf.Metadata._metadata);
        }
        else {
            host.request(null, 'tf-metadata.json', 'utf-8', (err, data) => {
                tf.Metadata._metadata = new tf.Metadata(data);
                callback(null, tf.Metadata._metadata);
            });
        }
    }

    constructor(data) {
        this._map = {};
        if (data) {
            if (data) {
                var items = JSON.parse(data);
                if (items) {
                    items.forEach((item) => {
                        if (item.name && item.schema) {
                            this._map[item.name] = item.schema;
                        }
                    });
                }
            }
        }
    }

    getSchema(operator) {
        return this._map[operator];
    }
};

tf.Error = class extends Error {
    constructor(message) {
        super(message);
        this.name = 'Error loading TensorFlow model.';
    }
};

if (typeof module !== 'undefined' && typeof module.exports === 'object') {
    module.exports.ModelFactory = tf.ModelFactory;
}