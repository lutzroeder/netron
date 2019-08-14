/* jshint esversion: 6 */
/* eslint "indent": [ "error", 4, { "SwitchCase": 1 } ] */

// Experimental

var tf = tf || {};
var long = long || { Long: require('long') };
var protobuf = protobuf || require('protobufjs');
var prototxt = prototxt || require('protobufjs/ext/prototxt');
var marked = marked || require('marked');

tf.ModelFactory = class {

    match(context) {
        var identifier = context.identifier;
        var extension = identifier.split('.').pop().toLowerCase();
        var tags = null;
        if (extension == 'meta') {
            tags = context.tags('pb');
            if (tags.size === 0) {
                return false;
            }
            return true;
        }
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
            if (tags.size === 0) {
                tags = context.tags('pbtxt');
                if (tags.has('node') || tags.has('saved_model_schema_version') || tags.has('meta_graphs') || tags.has('graph_def')) {
                    return true;
                }
                return false;
            }
            // ignore input_0.pb, output_0.pb
            if (tags.size > 0 &&
                tags.has(1) && tags.get(1) === 0 && 
                tags.has(2) && tags.get(2) === 0 && 
                tags.has(9) && tags.get(9) === 2) {
                return false;
            }
            if (tags.size > 0 &&
                Array.from(tags.values()).some((v) => v === 5)) {
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
            if (tags.has('node') || tags.has('saved_model_schema_version') || tags.has('meta_graphs') || tags.has('graph_def')) {
                return true;
            }
        }
        if (extension == 'json') {
            try {
                var root = JSON.parse(context.text);
                if (root && root.format && root.format === 'graph-model' && root.modelTopology) {
                    return true;
                }
            }
            catch (err) {
                // continue regardless of error
            }
        }
        return false;
    }

    open(context, host) { 
        return host.require('./tf-proto').then(() => {
            tf.proto = protobuf.roots.tf.tensorflow;
            var graph_def = null;
            var meta_graph = null;
            var saved_model = null;
            var format = null;
            var producer = null;
            var identifier = context.identifier; 
            var extension = identifier.split('.').pop().toLowerCase();
            if (extension !== 'json') {
                var tags = context.tags('pbtxt');
                if (tags.has('node') || tags.has('saved_model_schema_version') || tags.has('meta_graphs') || tags.has('graph_def')) {
                    if (tags.has('saved_model_schema_version') || tags.has('meta_graphs')) {
                        try {
                            if (identifier.endsWith('saved_model.pbtxt') || identifier.endsWith('saved_model.prototxt')) {
                                saved_model = tf.proto.SavedModel.decodeText(prototxt.TextReader.create(context.text));
                                format = 'TensorFlow Saved Model';
                                if (saved_model && Object.prototype.hasOwnProperty.call(saved_model, 'saved_model_schema_version')) {
                                    format = format + ' v' + saved_model.saved_model_schema_version.toString();
                                }
                            }
                        }
                        catch (error) {
                            throw new tf.Error("File text format is not tensorflow.SavedModel (" + error.message + ") in '" + identifier + "'.");
                        }
                    }
                    else if (tags.has('graph_def')) {
                        try {
                            if (!saved_model) {
                                meta_graph = tf.proto.MetaGraphDef.decodeText(prototxt.TextReader.create(context.text));
                                saved_model = new tf.proto.SavedModel();
                                saved_model.meta_graphs.push(meta_graph);
                                format = 'TensorFlow MetaGraph';
                            }
                        }
                        catch (error) {
                            throw new tf.Error("File text format is not tensorflow.MetaGraphDef (" + error.message + ") in '" + identifier + "'.");
                        }
                    }
                    else if (tags.has('node')) {
                        try {
                            graph_def = tf.proto.GraphDef.decodeText(prototxt.TextReader.create(context.text));
                            meta_graph = new tf.proto.MetaGraphDef();
                            meta_graph.graph_def = graph_def;
                            saved_model = new tf.proto.SavedModel();
                            saved_model.meta_graphs.push(meta_graph);
                            format = 'TensorFlow Graph';
                        }
                        catch (error) {
                            throw new tf.Error("File text format is not tensorflow.GraphDef (" + error.message + ") in '" + identifier + "'.");
                        }
                    }
                }
                else {
                    try {
                        if (identifier.endsWith('saved_model.pb')) {
                            saved_model = tf.proto.SavedModel.decode(context.buffer);
                            format = 'TensorFlow Saved Model';
                            if (saved_model && Object.prototype.hasOwnProperty.call(saved_model, 'saved_model_schema_version')) {
                                format = format + ' v' + saved_model.saved_model_schema_version.toString();
                            }
                        }
                    }
                    catch (error) {
                        var buffer = context.buffer;
                        if (buffer.length > 3 && buffer[0] == 0x08 && buffer[1] == 0x01 && buffer[2] == 0x12) {
                            throw new tf.Error("File format is not tensorflow.SavedModel (" + error.message + ") in '" + identifier + "'.");
                        }
                    }
                    try {
                        if (!saved_model && extension == 'meta') {
                            meta_graph = tf.proto.MetaGraphDef.decode(context.buffer);
                            saved_model = new tf.proto.SavedModel();
                            saved_model.meta_graphs.push(meta_graph);
                            format = 'TensorFlow MetaGraph';
                        }
                    }
                    catch (error) {
                        throw new tf.Error("File format is not tensorflow.MetaGraphDef (" + error.message + ") in '" + identifier + "'.");
                    }
                    try {
                        if (!saved_model) {
                            graph_def = tf.proto.GraphDef.decode(context.buffer);
                            meta_graph = new tf.proto.MetaGraphDef();
                            meta_graph.graph_def = graph_def;
                            saved_model = new tf.proto.SavedModel();
                            saved_model.meta_graphs.push(meta_graph);
                            format = 'TensorFlow Graph';
                        }
                    }
                    catch (error) {
                        throw new tf.Error("File format is not tensorflow.GraphDef (" + error.message + ") in '" + identifier + "'.");
                    }
                }

                if (saved_model && saved_model.meta_graphs && saved_model.meta_graphs.length > 0 &&
                    saved_model.meta_graphs[0].meta_info_def && 
                    Object.prototype.hasOwnProperty.call(saved_model.meta_graphs[0].meta_info_def, 'tensorflow_version')) {
                    producer = 'TensorFlow v' + saved_model.meta_graphs[0].meta_info_def.tensorflow_version;
                }
            }
            else {
                try {
                    var root = JSON.parse(context.text);
                    graph_def = new tf.proto.GraphDef();
                    meta_graph = new tf.proto.MetaGraphDef();
                    meta_graph.graph_def = graph_def;
                    saved_model = new tf.proto.SavedModel();
                    saved_model.meta_graphs.push(meta_graph);
                    for (var node of root.modelTopology.node) {
                        graph_def.node.push(node);
                        node.input = node.input || [];
                    }
                    format = 'TensorFlow.js ' + root.format;
                    producer = root.convertedBy || root.generatedBy || '';
                }
                catch (error) {
                    throw new tf.Error("File text format is not TensorFlow.js graph-model (" + error.message + ") in '" + identifier + "'.");
                }
            }

            return tf.Metadata.open(host).then((metadata) => {
                try {
                    return new tf.Model(metadata, saved_model, format, producer);
                }
                catch (error) {
                    host.exception(error, false);
                    throw new tf.Error(error.message);
                }
            });
        });
    }
};

tf.Model = class {

    constructor(metadata, model, format, producer) {
        this._model = model;
        this._format = format;
        this._producer = producer || '';
        this._graphs = [];
        for (var i = 0; i < model.meta_graphs.length; i++) {
            var metaGraph = model.meta_graphs[i];
            var name = null;
            if (metaGraph.any_info) {
                name = metaGraph.any_info.toString();
            }
            else if (model.meta_graphs.length > 1) {
                name = i.toString();
            }
            else {
                name = '-';
            }
            this._graphs.push(new tf.Graph(metadata, metaGraph, name));
        }

        // Recursively add all subgraphs.
        let visited_graph = [];
        let pending_graphs = [...this._graphs];
        while (pending_graphs.length > 0) {
            let g = pending_graphs.shift();
            visited_graph.push(g);
            for (let f of g.functions)
                pending_graphs.push(f);
        }
        this._graphs = visited_graph;

        this._activeGraph = (this._graphs.length > 0) ? this._graphs[0] : null;
    }

    get format() {
        return this._format;
    }

    get producer() {
        return this._producer;
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
        this._inputs = [];
        this._outputs = [];
        this._nodes = [];
        this._functions = [];

        if (metaGraph.graph_def) {
            var graph = metaGraph.graph_def;
            if (graph.versions) {
                this._version = 'v' + graph.versions.producer.toString();
            }
            else if (graph.version) {
                this._version = graph.version;
            }
            else if (metaGraph.meta_info_def && metaGraph.meta_info_def.tensorflow_version) {
                this._version = metaGraph.meta_info_def.tensorflow_version;
            }
            if (metaGraph.meta_info_def && metaGraph.meta_info_def.tags) {
                this._tags = metaGraph.meta_info_def.tags.join(', ');
            }
            var nodes = graph.node
            if (nodes) {
                var node;
                var input;
                var nodeMap = {};
                this._namespaces = {};
                for (node of nodes) {
                    var nodeName = node.name;
                    nodeMap[nodeName] = node;
                    if (node.op != 'Const') {
                        var lastIndex = nodeName.lastIndexOf('/');
                        if (lastIndex != -1) {
                            var namespace = nodeName.substring(0, lastIndex);
                            this._namespaces[namespace] = true;
                        }
                    }
                    node.output = [];
                }
                for (node of nodes) {
                    var inputs = node.input;
                    node.input = [];
                    node.controlDependencies = [];
                    for (input of inputs) {
                        var split = input.split(':', 2);
                        var inputName = split[0];
                        var outputIndex = split.length == 1 ? 0 : parseInt(split[1]);
                        var outputName = inputName.startsWith('^') ? inputName.substring(1) : inputName;
                        var outputNode = nodeMap[outputName];
                        outputName = outputIndex == 0 ? outputName : outputName + ':' + outputIndex.toString();
                        if (inputName.startsWith('^')) {
                            node.controlDependencies.push(outputName);
                        }
                        else {
                            node.input.push(outputName);
                        }
                        if (outputNode) {
                            for (var j = outputNode.output.length; j <= outputIndex; j++) {
                                outputNode.output.push('');
                            }
                            outputNode.output[outputIndex] = outputName;
                        }
                    }
                }
                this._nodeOutputCountMap = {};
                for (node of nodes) {
                    for (input of node.input) {
                        this._nodeOutputCountMap[input] = (this._nodeOutputCountMap[input] || 0) + 1;
                    }
                    for (var controlDependency of node.controlDependencies) {
                        this._nodeOutputCountMap[controlDependency] = (this._nodeOutputCountMap[controlDependency] || 0) + 1;
                    }
                }
                var initializers = {};
                for (node of nodes) {
                    if (node.op == 'Const' && node.input.length == 0 && node.controlDependencies.length == 0 && this._checkSingleOutput(node)) {
                        var value = node.attr.value;
                        if (value && Object.prototype.hasOwnProperty.call(value, 'tensor')) {
                            var output = node.output[0];
                            if (output) {
                                initializers[output] = new tf.Tensor(value.tensor, node.name, 'Constant');
                            }
                        }
                    }
                }
                for (node of nodes) {
                    if (node.op == 'Identity' && node.input.length == 1 && node.controlDependencies.length == 0 && this._checkSingleOutput(node)) {
                        var initializer_name = node.input[0];
                        var initializer = initializers[initializer_name];
                        if (initializer) {
                            initializers[initializer_name] = "-";
                            initializer.kind = 'Identity Constant';
                            initializers[node.output[0]] = initializer;
                        }
                    }
                }
                var inputMap = {};
                for (node of nodes) {
                    if (node.op == 'Placeholder' && node.input.length == 0 && node.controlDependencies.length == 0 && node.output.length == 1) {
                        var dtype = node.attr.dtype;
                        var shape = node.attr.shape;
                        if (dtype && dtype.type && shape && shape.shape) {
                            var type = new tf.TensorType(dtype.type, shape.shape);
                            var argument = new tf.Argument(node.output[0], type, null);
                            inputMap[node.output[0]] = new tf.Parameter(node.name, [ argument ]);
                        }
                    }
                }
                this._inputs = Object.keys(inputMap).map((key) => {
                    return inputMap[key];
                });
                for (node of nodes) {
                    var id = node.name;
                    if (!initializers[id] && !inputMap[id] /* && node.op != 'NoOp' */) {
                        this._nodes.push(new tf.Node(this, node, initializers));
                    }
                }
            }

            if (graph.library) {
                var funcs = graph.library.function;
                for (var func of funcs) {
                    this._functions.push(new tf.Function(this, func, this._metadata));
                }
            }
        }
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
        return this._inputs;
    }

    get outputs() {
        return this._outputs;
    }

    get nodes() {
        return this._nodes;
    }

    get metadata() {
        return this._metadata;
    }

    get namespaces() {
        return this._namespaces;
    }

    get functions() {
        return this._functions;
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

tf.Parameter = class {
    constructor(name, args) {
        this._name = name;
        this._arguments = args;
    }

    get name() {
        return this._name;
    }

    get visible() {
        return true;
    }

    get arguments() {
        return this._arguments;
    }
};

tf.Argument = class {
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

tf.Function = class {

    constructor(graph, func, metadata) {
        this._name = func.signature.name;
        this._version = null;
        this._tags = null;
        this._inputs = [];
        this._outputs = [];
        this._nodes = [];
        this._metadata = metadata;
        this._namespaces = {};
        this._functions = [];

        var inputs = func.signature.input_arg;
        if (inputs) {
            for (let input of inputs) {
                var inputArgument = new tf.Argument(input.name, new tf.TensorType(input.type, null), null);
                this._inputs.push(new tf.Parameter(input.name, [ inputArgument ]));
            }
        }

        var ret_map = {};
        for (let key of Object.keys(func.ret)) {
            let v = func.ret[key].split(':', 2);
            ret_map[key] = v[0];
        }

        var out_args_reverse_map = {};
        var outputs = func.signature.output_arg;
        if (outputs) {
            for (let output of outputs) {
                let name = ret_map[output.name];
                this._outputs.push(new tf.Parameter(output.name, [ 
                    new tf.Argument(name, new tf.TensorType(output.type, null), null)
                ]));
                out_args_reverse_map[name] = output.name;
            }
        }

        var nodes = func.node_def;
        if (nodes) {
            var nodeMap = {};

            for (let node of nodes) {
                let nodeName = node.name;
                nodeMap[nodeName] = node;
                if (node.op != 'Const') {
                    let lastIndex = nodeName.lastIndexOf('/');
                    if (lastIndex != -1) {
                        let namespace = nodeName.substring(0, lastIndex);
                        this._namespaces[namespace] = true;
                    }
                }
                node.output = [];
            }
            for (let node of nodes) {
                let inputs = node.input;
                node.input = [];
                node.controlDependencies = [];
                for (let input of inputs) {
                    var split = input.split(':', 3);
                    var inputName = split[0];
                    var outputIndex = split.length == 1 ? 0 : parseInt(split[split.length - 1]);
                    var outputName = inputName.startsWith('^') ? inputName.substring(1) : inputName;
                    var outputNode = nodeMap[outputName];
                    outputName = outputIndex == 0 ? outputName : outputName + ':' + outputIndex.toString();
                    if (inputName.startsWith('^')) {
                        node.controlDependencies.push(outputName);
                    }
                    else {
                        node.input.push(outputName);
                    }
                    if (outputNode) {
                        for (var j = outputNode.output.length; j <= outputIndex; j++) {
                            outputNode.output.push('');
                        }
                        outputNode.output[outputIndex] = outputName;
                    }
                }

                if (out_args_reverse_map[node.name]) {
                    node.output.push(node.name);
                }
            }

            let nodeOutputCountMap = {};
            for (let node of nodes) {
                for (let input of node.input) {
                    nodeOutputCountMap[input] = (nodeOutputCountMap[input] || 0) + 1;
                }
                for (let controlDependency of node.controlDependencies) {
                    nodeOutputCountMap[controlDependency] = (nodeOutputCountMap[controlDependency] || 0) + 1;
                }
            }

            var initializers = {};
            for (let node of nodes) {
                if (node.op == 'Const' && node.input.length == 0 && node.controlDependencies.length == 0 && tf.Function._checkSingleOutput(node, nodeOutputCountMap)) {
                    var value = node.attr.value;
                    if (value && Object.prototype.hasOwnProperty.call(value, 'tensor')) {
                        var output = node.output[0];
                        if (output) {
                            initializers[output] = new tf.Tensor(value.tensor, node.name, 'Constant');
                        }
                    }
                }
            }
            for (let node of nodes) {
                if (node.op == 'Identity' && node.input.length == 1 && node.controlDependencies.length == 0 && tf.Function._checkSingleOutput(node, nodeOutputCountMap)) {
                    var initializer_name = node.input[0];
                    var initializer = initializers[initializer_name];
                    if (initializer) {
                        initializers[initializer_name] = "-";
                        initializer.kind = 'Identity Constant';
                        initializers[node.output[0]] = initializer;
                    }
                }
            }

            for (let node of nodes) {
                if (!initializers[node.name])
                    this._nodes.push(new tf.Node(this, node, initializers));
            }
        }
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
        return this._inputs;
    }

    get outputs() {
        return this._outputs;
    }

    get nodes() {
        return this._nodes;
    }

    get metadata() {
        return this._metadata;
    }

    get namespaces() {
        return this._namespaces;
    }

    get functions() {
        return this._functions;
    }

    static _checkSingleOutput(node, nodeOutputCountMap) {
        if (node.output.length != 1) {
            return false;
        }
        var output = node.output[0];
        var count = nodeOutputCountMap[output];
        if (count != 1) {
            return false;
        }
        return true;
    }
}

tf.Node = class {

    constructor(graph, node, initializers) {
        this._graph = graph;
        this._operator = node.op;
        this._name = node.name;
        if (Object.prototype.hasOwnProperty.call(node, 'device')) {
            this._device = node.device;
        }
        var metadata = graph.metadata;
        this._attributes = [];
        if (node.attr) {
            for (var attributeName of Object.keys(node.attr)) {
                this._attributes.push(new tf.Attribute(attributeName, node.attr[attributeName], this._operator, metadata));
            }
        }

        var schema = metadata.getSchema(node.op);

        this._inputs = [];
        var inputIndex = 0;
        var inputs = node.input.filter(input => !input.startsWith('^'));
        if (schema && schema.inputs) {
            for (var input of schema.inputs) {
                var inputCount = 1;
                if (input.numberAttr) {
                    var inputNumber = node.attr[input.numberAttr];
                    if (inputNumber && inputNumber.i) {
                        inputCount = inputNumber.i;
                    }
                }
                else if (input.typeListAttr) {
                    var inputTypeListAttr = node.attr[input.typeListAttr];
                    if (inputTypeListAttr && inputTypeListAttr.list && inputTypeListAttr.list.type) {
                        inputCount = inputTypeListAttr.list.type.length;
                    }
                }
                var inputConnections = inputs.slice(inputIndex, inputIndex + inputCount).map((id) => {
                    return new tf.Argument(id, null, initializers[id]);
                });
                this._inputs.push(new tf.Parameter(input.name, inputConnections));
                inputIndex += inputCount;
            }
        }
        this._inputs = this._inputs.concat(inputs.slice(inputIndex).map((input, index) => {
            return new tf.Parameter((inputIndex + index).toString(), [ 
                new tf.Argument(input, null, initializers[input])
            ]);
        }));

        this._outputs = [];
        var outputIndex = 0;
        var outputs = node.output;
        if (schema && schema.outputs) {
            for (var output of schema.outputs) {
                var outputCount = 1;
                if (output.numberAttr) {
                    var outputNumber = node.attr[output.numberAttr];
                    if (outputNumber && outputNumber.i) {
                        outputCount = outputNumber.i;
                    }
                }
                else if (output.typeListAttr) {
                    var outputTypeListAttr = node.attr[output.typeListAttr];
                    if (outputTypeListAttr && outputTypeListAttr.list && outputTypeListAttr.list.type) {
                        outputCount = outputTypeListAttr.list.type.length;
                    }
                }
                var outputConnections = outputs.slice(outputIndex, outputIndex + outputCount).map((id) => {
                    return new tf.Argument(id, null, null);
                });
                this._outputs.push(new tf.Parameter(output.name, outputConnections));
                outputIndex += outputCount;
            }
        }
        this._outputs = this._outputs.concat(outputs.slice(outputIndex).map((output, index) => {
            return new tf.Parameter((outputIndex + index).toString(), [
                new tf.Argument(output, null, null)
            ]);
        }));

        this._controlDependencies = node.controlDependencies;
    }

    get operator() {
        return this._operator;
    }

    get name() {
        return this._name;
    }

    get device() {
        return this._device || null;
    }

    get group() {
        var name = this._name;
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
        return '';
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
                for (var input of schema.inputs) {
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
                }
            }
            if (schema.outputs) {
                for (var output of schema.outputs) {
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
                }
            }
            if (schema.attributes) {
                for (var attribute of schema.attributes) {
                    var description = attribute.description;
                    if (attribute.allowedValues) {
                        var allowedValues = tf.GraphMetadata._formatAttributeValue(attribute.allowedValues);
                        allowedValues = Array.isArray(allowedValues) ? allowedValues : [ allowedValues ];
                        allowedValues = allowedValues.map((item) => '`' + item + '`').join(', ');
                        allowedValues = 'Must be one of the following: ' + allowedValues + '.';
                        description = description ? (allowedValues + ' ' + description) : allowedValues;
                    }
                    if (attribute.defaultValue) {
                        var defaultValue = tf.GraphMetadata._formatAttributeValue(attribute.defaultValue);
                        defaultValue = Array.isArray(defaultValue) ? defaultValue : [ defaultValue ];
                        defaultValue = defaultValue.map((item) => '`' + item + '`').join(', ');
                        defaultValue = 'Defaults to ' + defaultValue + '.';
                        description = description ? (defaultValue + ' ' + description) : defaultValue;
                    }
                    if (description) {
                        attribute.description = marked(description);
                    }
                }
            }
            return schema;
        }
        return '';
    }

    get category() {
        var schema = this._graph.metadata.getSchema(this.operator);
        return (schema && schema.category) ? schema.category : '';
    }

    get inputs() {
        return this._inputs;
    }

    get outputs() {
        return this._outputs;
    }

    get controlDependencies() {
        return this._controlDependencies;
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
        if (Object.prototype.hasOwnProperty.call(value, 'tensor')) {
            this._type = 'tensor';
            this._value = new tf.Tensor(value.tensor);
        }
        else if (schema && schema.type) {
            this._type = schema.type;
        }
        if (Object.prototype.hasOwnProperty.call(value, 'type')) {
            this._type = 'type';
            this._value = tf.Tensor.formatDataType(value.type);
        }
        else if (Object.prototype.hasOwnProperty.call(value, 'i')) {
            this._value = value.i;
        }
        else if (Object.prototype.hasOwnProperty.call(value, 'f')) {
            this._value = value.f;
        }
        else if (Object.prototype.hasOwnProperty.call(value, 'b')) {
            this._value = value.b;
        }
        else if (Object.prototype.hasOwnProperty.call(value, 'shape')) {
            this._type = 'shape';
            this._value = new tf.TensorShape(value.shape);
        }
        else if (Object.prototype.hasOwnProperty.call(value, 's')) {
            if (typeof value.s === 'string'){
                this._value = value.s;
            }
            else if (value.s.filter(c => c <= 32 && c >= 128).length == 0) {
                this._value = tf.Metadata.textDecoder.decode(value.s);
            }
            else {
                this._value = value.s;
            }
        }
        else if (Object.prototype.hasOwnProperty.call(value, 'list')) {
            var list = value.list;
            this._value = [];
            if (list.s && list.s.length > 0) {
                this._value = list.s.map((s) => {
                    if (s.filter(c => c <= 32 && c >= 128).length == 0) {
                        return tf.Metadata.textDecoder.decode(value.s);
                    }
                    return s.map(v => v.toString()).join(', ');
                });
            }
            else if (list.i && list.i.length > 0) {
                this._value = list.i;
            }
            else if (list.f && list.f.length > 0) {
                this._value = list.f;
            }
            else if (list.type && list.type.length > 0) {
                this._type = 'type[]';
                this._value = list.type.map((type) => tf.Tensor.formatDataType(type)); 
            }
            else if (list.shape && list.shape.length > 0) {
                this._type = 'shape[]';
                this._value = list.shape.map((shape) => new tf.TensorShape(shape));
            }
        }
        else if (Object.prototype.hasOwnProperty.call(value, 'func')) {
            var func = value.func;
            this._type = 'function';
            this._value = func.name;
        }

        if (schema) {
            if (Object.prototype.hasOwnProperty.call(schema, 'visible') && !schema.visible) {
                this._visible = false;
            }
            else if (Object.prototype.hasOwnProperty.call(schema, 'default')) {
                if (!Array.isArray(this._value) || Array.isArray(schema.default) || this._value.length === schema.default.length) {
                    var valueText = tf.GraphMetadata._formatAttributeValue(this._value);
                    var defaultValueText = tf.GraphMetadata._formatAttributeValue(schema.default);
                    if (JSON.stringify(valueText) == JSON.stringify(defaultValueText)) {
                        this._visible = false;
                    }
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
        this._type = new tf.TensorType(this._tensor.dtype, this._tensor.tensor_shape || this._tensor.tensorShape);
    }

    get name() {
        return this._name;
    }

    get type() {
        return this._type;
    }

    get kind() {
        return this._kind || null;
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
        var shape = this._tensor.tensor_shape || this._tensor.tensorShape; 
        if (!shape || !shape.dim) {
            context.state = 'Tensor has no dimensions.';
            return context;
        }

        for (var dim of shape.dim) {
            context.size = context.size * (dim.size ? dim.size : 0);
        }

        switch (this._tensor.dtype) {
            case 'DT_FLOAT':
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
                    context.state = 'Tensor data type is not implemented.';
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
                context.state = "Tensor data type '" + this._tensor.dtype + "' is not implemented.";
                break;
        }

        context.shape = shape.dim.map((dim) => dim.size);
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
            for (var key of Object.keys(tf.proto.DataType)) {
                var value = tf.proto.DataType[key];
                key = key.startsWith('DT_') ? key.substring(3) : key;
                tf.Tensor.dataType[value] = key.toLowerCase();
            }
            tf.Tensor.dataType[tf.proto.DataType.DT_HALF] = 'float16';
            tf.Tensor.dataType[tf.proto.DataType.DT_FLOAT] = 'float32';
            tf.Tensor.dataType[tf.proto.DataType.DT_DOUBLE] = 'float64';
            tf.Tensor.dataType['DT_FLOAT'] = 'float32';
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

    constructor(metadata) {
        this._metadata = metadata;
        this._map = {};
        this._attributeCache = {};
    }

    getSchema(operator) {
        var schema = this._metadata.getSchema(operator);
        if (!schema) {
            schema = this._map[operator];
        }
        return schema;
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

    getAttributeVisibleMap(operator) {
        var schema = this.getSchema(operator);
        if (schema) {
            var map = schema.__visisbleAttributeMap__;
            if (!map) {
                map = {};
                if (schema.inputs) {
                    for (var input of schema.inputs) {
                        if (input.typeAttr) {
                            map[input.typeAttr] = true;
                        }
                        else if (input.typeListAttr) {
                            map[input.typeListAttr] = true;
                        }
                        if (input.numberAttr) {
                            map[input.numberAttr] = true;
                        }
                    }
                }
                if (schema.outputs) {
                    for (var output of schema.outputs) {
                        if (output.typeAttr) {
                            map[output.typeAttr] = true;
                        }
                        else if (output.typeListAttr) {
                            map[output.typeListAttr] = true;
                        }
                        if (output.numberAttr) {
                            map[output.numberAttr] = true;
                        }
                    }
                }
                schema.__visisbleAttributeMap__ = map;
            }
            return map;
        }
        return {};
    }

    static _formatAttributeValue(value) {
        if (value == null) {
            return null;
        }
        if (value && long.Long.isLong(value)) {
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

    static open(host) {
        tf.Metadata.textDecoder = tf.Metadata.textDecoder || new TextDecoder('utf-8');
        if (tf.Metadata._metadata) {
            return Promise.resolve(tf.Metadata._metadata);
        }
        return host.request(null, 'tf-metadata.json', 'utf-8').then((data) => {
            tf.Metadata._metadata = new tf.Metadata(data);
            return tf.Metadata._metadata;
        }).catch(() => {
            tf.Metadata._metadata = new tf.Metadata(null);
            return tf.Metadata._metadata;
        });
    }

    constructor(data) {
        this._map = {};
        if (data) {
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