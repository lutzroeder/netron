/* jshint esversion: 6 */
/* eslint "indent": [ "error", 4, { "SwitchCase": 1 } ] */

// Experimental

var tf = tf || {};
var long = long || { Long: require('long') };
var protobuf = protobuf || require('protobufjs');
var prototxt = prototxt || require('protobufjs/ext/prototxt');

tf.ModelFactory = class {

    match(context) {
        const identifier = context.identifier;
        const extension = identifier.split('.').pop().toLowerCase();
        if (extension === 'meta') {
            const tags = context.tags('pb');
            if (tags.size !== 0) {
                return true;
            }
        }
        if (extension === 'pbtxt' || extension === 'prototxt') {
            if (identifier.endsWith('predict_net.pbtxt') || identifier.endsWith('predict_net.prototxt') ||
                identifier.endsWith('init_net.pbtxt') || identifier.endsWith('init_net.prototxt')) {
                return false;
            }
            const tags = context.tags('pbtxt');
            if (tags.has('input_stream') || tags.has('output_stream')) {
                return false;
            }
            if (tags.has('node') || tags.has('saved_model_schema_version') || tags.has('meta_graphs') || tags.has('graph_def')) {
                return true;
            }
        }
        if (extension === 'pb' || extension === 'pbtxt' || extension === 'prototxt') {
            if (identifier.endsWith('predict_net.pb') || identifier.endsWith('init_net.pb')) {
                return false;
            }
            if (identifier == 'tfhub_module.pb') {
                const buffer = context.buffer;
                if (buffer && buffer.length == 2 && buffer[0] == 0x08 && buffer[1] == 0x03) {
                    return false;
                }
            }
            const tags = context.tags('pb');
            if (tags.size === 0) {
                const tags = context.tags('pbtxt');
                if (tags.has('input_stream') || tags.has('output_stream')) {
                    return false;
                }
                if (tags.has('node') || tags.has('saved_model_schema_version') || tags.has('meta_graphs') || tags.has('graph_def')) {
                    return true;
                }
            }
            else {
                // ignore input_0.pb, output_0.pb
                if (tags.has(1) && tags.get(1) === 0 &&
                    tags.has(2) && tags.get(2) === 0 &&
                    tags.has(9) && tags.get(9) === 2) {
                    return false;
                }
                if (!Array.from(tags.values()).some((v) => v === 5)) {
                    return true;
                }
            }
        }
        if (extension === 'json') {
            try {
                const root = JSON.parse(context.text);
                if (root && root.format && root.format === 'graph-model' && root.modelTopology) {
                    return true;
                }
            }
            catch (err) {
                // continue regardless of error
            }
        }
        if (extension === 'index' || extension === 'ckpt') {
            if (context.buffer.length > 8) {
                const buffer = context.buffer.subarray(context.buffer.length - 8, context.buffer.length);
                const signature = [ 0x57, 0xfb, 0x80, 0x8b, 0x24, 0x75, 0x47, 0xdb ];
                if (buffer.every((value, index) => value === signature[index])) {
                    return true;
                }
            }
        }
        return false;
    }

    open(context, host) {
        return host.require('./tf-proto').then(() => {
            tf.proto = protobuf.roots.tf.tensorflow;
            let saved_model = null;
            let format = null;
            let producer = null;
            const identifier = context.identifier;
            const extension = identifier.split('.').pop().toLowerCase();
            switch (extension) {
                case 'ckpt':
                case 'index': {
                    return tf.ModelFactory._openBundle(context, host);
                }
                case 'json': {
                    try {
                        const root = JSON.parse(context.text);
                        let graph_def = new tf.proto.GraphDef();
                        let meta_graph = new tf.proto.MetaGraphDef();
                        meta_graph.graph_def = graph_def;
                        saved_model = new tf.proto.SavedModel();
                        saved_model.meta_graphs.push(meta_graph);
                        for (const node of root.modelTopology.node) {
                            graph_def.node.push(node);
                            node.input = node.input || [];
                        }
                        format = 'TensorFlow.js ' + root.format;
                        producer = root.convertedBy || root.generatedBy || '';
                    }
                    catch (error) {
                        throw new tf.Error("File text format is not TensorFlow.js graph-model (" + error.message + ") in '" + identifier + "'.");
                    }
                    break;
                }
                default: {
                    const tags = context.tags('pbtxt');
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
                                    const meta_graph = tf.proto.MetaGraphDef.decodeText(prototxt.TextReader.create(context.text));
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
                                const graph_def = tf.proto.GraphDef.decodeText(prototxt.TextReader.create(context.text));
                                let meta_graph = new tf.proto.MetaGraphDef();
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
                            let buffer = context.buffer;
                            if (buffer.length > 3 && buffer[0] == 0x08 && buffer[1] == 0x01 && buffer[2] == 0x12) {
                                throw new tf.Error("File format is not tensorflow.SavedModel (" + error.message + ") in '" + identifier + "'.");
                            }
                        }
                        try {
                            if (!saved_model && extension == 'meta') {
                                const meta_graph = tf.proto.MetaGraphDef.decode(context.buffer);
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
                                const graph_def = tf.proto.GraphDef.decode(context.buffer);
                                let meta_graph = new tf.proto.MetaGraphDef();
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
                    break;
                }
            }

            return tf.Metadata.open(host).then((metadata) => {
                if (saved_model.meta_graphs.length === 1 &&
                    saved_model.meta_graphs[0].object_graph_def &&
                    saved_model.meta_graphs[0].object_graph_def.nodes &&
                    saved_model.meta_graphs[0].object_graph_def.nodes.length > 0) {
                    const identifier = 'variables/variables.index';
                    return context.request(identifier, null).then((buffer) => {
                        return tf.TensorBundle.open(buffer, identifier, context, host).then((bundle) => {
                            return tf.ModelFactory._openModel(identifier, host, metadata, saved_model, format, producer, bundle);
                        });
                    }).catch(() => {
                        return tf.ModelFactory._openModel(identifier, host, metadata, saved_model, format, producer, null);
                    });
                }
                return tf.ModelFactory._openModel(identifier, host, metadata, saved_model, format, producer, null);
            });
        });
    }

    static _openModel(identifier, host, metadata, saved_model, format, producer, bundle) {
        try {
            return new tf.Model(metadata, saved_model, format, producer, bundle);
        }
        catch (error) {
            host.exception(error, false);
            const message = error && error.message ? error.message : error.toString();
            throw new tf.Error(message.replace(/\.$/, '') + " in '" + identifier + "'.");
        }
    }

    static _openBundle(context, host) {
        return tf.Metadata.open(host).then((metadata) => {
            const identifier = context.identifier;
            return tf.TensorBundle.open(context.buffer, identifier, context, host).then((bundle) => {
                return new tf.Model(metadata, null, 'TensorFlow Tensor Bundle v' + bundle.format.toString(), null, bundle);
            }).catch((error) => {
                host.exception(error, false);
                const message = error && error.message ? error.message : error.toString();
                throw new tf.Error(message.replace(/\.$/, '') + " in '" + identifier + "'.");
            });
        });
    }
};

tf.Model = class {

    constructor(metadata, model, format, producer, bundle) {
        this._format = format;
        this._producer = producer || '';
        this._graphs = [];
        if (model) {
            for (let i = 0; i < model.meta_graphs.length; i++) {
                const metaGraph = model.meta_graphs[i];
                let name = null;
                if (metaGraph.any_info) {
                    name = metaGraph.any_info.toString();
                }
                else if (model.meta_graphs.length > 1) {
                    name = i.toString();
                }
                else {
                    name = '-';
                }
                this._graphs.push(new tf.Graph(metadata, metaGraph, name, bundle));
            }
            // Recursively add all subgraphs.
            let visited_graph = [];
            let pending_graphs = [...this._graphs];
            while (pending_graphs.length > 0) {
                let g = pending_graphs.shift();
                visited_graph.push(g);
                for (let f of g.functions) {
                    pending_graphs.push(f);
                }
            }
            this._graphs = visited_graph;
        }
        else {
            this._graphs.push(new tf.Graph(metadata, null, '', bundle));
        }

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

    constructor(metadata, metaGraph, name, bundle) {
        this._metadata = metadata;
        this._version = null;
        this._name = name;
        this._inputs = [];
        this._outputs = [];
        this._nodes = [];
        this._functions = [];

        if (metaGraph && metaGraph.graph_def) {
            this._metadata = new tf.GraphMetadata(metadata, metaGraph.meta_info_def);
            const graph = metaGraph.graph_def;
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
            const nodes = graph.node;
            if (nodes) {
                let nodeMap = {};
                this._namespaces = {};
                for (const node of nodes) {
                    const nodeName = node.name;
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
                for (const node of nodes) {
                    const inputs = node.input;
                    node.input = [];
                    node.controlDependencies = [];
                    for (const input of inputs) {
                        let split = input.split(':', 2);
                        let inputName = split[0];
                        let outputIndex = split.length == 1 ? 0 : parseInt(split[1]);
                        let outputName = inputName.startsWith('^') ? inputName.substring(1) : inputName;
                        let outputNode = nodeMap[outputName];
                        outputName = outputIndex == 0 ? outputName : outputName + ':' + outputIndex.toString();
                        if (inputName.startsWith('^')) {
                            node.controlDependencies.push(outputName);
                        }
                        else {
                            node.input.push(outputName);
                        }
                        if (outputNode) {
                            for (let j = outputNode.output.length; j <= outputIndex; j++) {
                                outputNode.output.push('');
                            }
                            outputNode.output[outputIndex] = outputName;
                        }
                    }
                }
                this._nodeOutputCountMap = {};
                for (const node of nodes) {
                    for (const input of node.input) {
                        this._nodeOutputCountMap[input] = (this._nodeOutputCountMap[input] || 0) + 1;
                    }
                    for (const controlDependency of node.controlDependencies) {
                        this._nodeOutputCountMap[controlDependency] = (this._nodeOutputCountMap[controlDependency] || 0) + 1;
                    }
                }
                let initializers = {};
                for (const node of nodes) {
                    if (node.op == 'Const' && node.input.length == 0 && node.controlDependencies.length == 0 && this._checkSingleOutput(node)) {
                        let value = node.attr.value;
                        if (value && Object.prototype.hasOwnProperty.call(value, 'tensor')) {
                            let output = node.output[0];
                            if (output) {
                                initializers[output] = new tf.Tensor(value.tensor, node.name, 'Constant');
                            }
                        }
                    }
                }
                for (const node of nodes) {
                    if (node.op == 'Identity' && node.input.length == 1 && node.controlDependencies.length == 0 && this._checkSingleOutput(node)) {
                        let initializer_name = node.input[0];
                        let initializer = initializers[initializer_name];
                        if (initializer) {
                            initializers[initializer_name] = "-";
                            initializer.kind = 'Identity Constant';
                            initializers[node.output[0]] = initializer;
                        }
                    }
                }
                let inputMap = {};
                for (const node of nodes) {
                    if (node.op == 'Placeholder' && node.input.length == 0 && node.controlDependencies.length == 0 && node.output.length == 1) {
                        const dtype = node.attr.dtype;
                        const shape = node.attr.shape;
                        if (dtype && dtype.type && shape && shape.shape) {
                            const type = new tf.TensorType(dtype.type, shape.shape);
                            const argument = new tf.Argument(node.output[0], type, null);
                            inputMap[node.output[0]] = new tf.Parameter(node.name, [ argument ]);
                        }
                    }
                }
                this._inputs = Object.keys(inputMap).map((key) => {
                    return inputMap[key];
                });
                for (const node of nodes) {
                    let id = node.name;
                    if (!initializers[id] && !inputMap[id] /* && node.op != 'NoOp' */) {
                        this._nodes.push(new tf.Node(this, node, node.op, node.name, initializers, null));
                    }
                }
            }

            if (graph.library) {
                const funcs = graph.library.function;
                for (const func of funcs) {
                    this._functions.push(new tf.Function(this, func, this._metadata));
                }
            }
        }
        else if (bundle) {
            let nodeNames = [];
            let nodeMap = new Map();
            for (const tensor of bundle.tensors) {
                let parts = tensor.name.split('/');
                if (bundle.format === 2) {
                    if (tensor.name === '_CHECKPOINTABLE_OBJECT_GRAPH' ||
                        tensor.name.startsWith('optimizer/') ||
                        tensor.name.startsWith('keras_api/metrics/') ||
                        tensor.name.endsWith('/ExponentialMovingAverage') ||
                        tensor.name.indexOf('.OPTIMIZER_SLOT') !== -1) {
                        continue;
                    }
                    if (tensor.name.endsWith('/.ATTRIBUTES/VARIABLE_VALUE')) {
                        parts.pop();
                        parts.pop();
                    }
                }
                let tensorName = parts.pop();
                let nodeName = parts.join('/');
                if (!nodeMap.has(nodeName)) {
                    nodeNames.push(nodeName);
                    nodeMap.set(nodeName, []);
                }
                nodeMap.get(nodeName).push({ name: tensorName, value: tensor });
            }
            for (const nodeName of nodeNames) {
                this._nodes.push(new tf.Node(this, null, 'Node', nodeName, null, nodeMap.get(nodeName)));
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
        const output = node.output[0];
        const count = this._nodeOutputCountMap[output];
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

    constructor(name, type, initializer) {
        if (typeof name !== 'string') {
            throw new tf.Error("Invalid argument identifier '" + JSON.stringify(name) + "'.");
        }
        this._name = name;
        this._type = type || null;
        this._initializer = initializer || null;
    }

    get name() {
        return this._name;
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

        let inputs = func.signature.input_arg;
        if (inputs) {
            for (const input of inputs) {
                let inputArgument = new tf.Argument(input.name, new tf.TensorType(input.type, null), null);
                this._inputs.push(new tf.Parameter(input.name, [ inputArgument ]));
            }
        }

        let ret_map = {};
        for (const key of Object.keys(func.ret)) {
            const v = func.ret[key].split(':', 2);
            ret_map[key] = v[0];
        }

        let out_args_reverse_map = {};
        let outputs = func.signature.output_arg;
        if (outputs) {
            for (const output of outputs) {
                let name = ret_map[output.name];
                this._outputs.push(new tf.Parameter(output.name, [
                    new tf.Argument(name, new tf.TensorType(output.type, null), null)
                ]));
                out_args_reverse_map[name] = output.name;
            }
        }

        let nodes = func.node_def;
        if (nodes) {
            let nodeMap = {};

            for (const node of nodes) {
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
            for (const node of nodes) {
                let inputs = node.input;
                node.input = [];
                node.controlDependencies = [];
                for (const input of inputs) {
                    let split = input.split(':', 3);
                    let inputName = split[0];
                    let outputIndex = split.length == 1 ? 0 : parseInt(split[split.length - 1]);
                    let outputName = inputName.startsWith('^') ? inputName.substring(1) : inputName;
                    let outputNode = nodeMap[outputName];
                    outputName = outputIndex == 0 ? outputName : outputName + ':' + outputIndex.toString();
                    if (inputName.startsWith('^')) {
                        node.controlDependencies.push(outputName);
                    }
                    else {
                        node.input.push(outputName);
                    }
                    if (outputNode) {
                        for (let j = outputNode.output.length; j <= outputIndex; j++) {
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
            for (const node of nodes) {
                for (const input of node.input) {
                    nodeOutputCountMap[input] = (nodeOutputCountMap[input] || 0) + 1;
                }
                for (const controlDependency of node.controlDependencies) {
                    nodeOutputCountMap[controlDependency] = (nodeOutputCountMap[controlDependency] || 0) + 1;
                }
            }

            let initializers = {};
            for (const node of nodes) {
                if (node.op == 'Const' && node.input.length == 0 && node.controlDependencies.length == 0 && tf.Function._checkSingleOutput(node, nodeOutputCountMap)) {
                    let value = node.attr.value;
                    if (value && Object.prototype.hasOwnProperty.call(value, 'tensor')) {
                        let output = node.output[0];
                        if (output) {
                            initializers[output] = new tf.Tensor(value.tensor, node.name, 'Constant');
                        }
                    }
                }
            }
            for (const node of nodes) {
                if (node.op == 'Identity' && node.input.length == 1 && node.controlDependencies.length == 0 && tf.Function._checkSingleOutput(node, nodeOutputCountMap)) {
                    let initializer_name = node.input[0];
                    let initializer = initializers[initializer_name];
                    if (initializer) {
                        initializers[initializer_name] = "-";
                        initializer.kind = 'Identity Constant';
                        initializers[node.output[0]] = initializer;
                    }
                }
            }

            for (const node of nodes) {
                if (!initializers[node.name])
                    this._nodes.push(new tf.Node(this, node, node.op, node.name, initializers, null));
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
        let output = node.output[0];
        let count = nodeOutputCountMap[output];
        if (count != 1) {
            return false;
        }
        return true;
    }
};

tf.Node = class {

    constructor(graph, node, op, name, initializers, tensors) {
        this._graph = graph;
        this._operator = op;
        this._name = name;
        this._attributes = [];
        this._inputs = [];
        this._outputs = [];
        if (node) {
            if (Object.prototype.hasOwnProperty.call(node, 'device')) {
                this._device = node.device;
            }
            const metadata = graph.metadata;
            if (node.attr) {
                for (const attributeName of Object.keys(node.attr)) {
                    this._attributes.push(new tf.Attribute(attributeName, node.attr[attributeName], this._operator, metadata));
                }
            }
            const schema = metadata.type(this._operator);
            let inputIndex = 0;
            let inputs = node.input.filter(input => !input.startsWith('^'));
            if (schema && schema.inputs) {
                for (const input of schema.inputs) {
                    let inputCount = 1;
                    if (input.numberAttr) {
                        let inputNumber = node.attr[input.numberAttr];
                        if (inputNumber && inputNumber.i) {
                            inputCount = inputNumber.i;
                        }
                    }
                    else if (input.typeListAttr) {
                        let inputTypeListAttr = node.attr[input.typeListAttr];
                        if (inputTypeListAttr && inputTypeListAttr.list && inputTypeListAttr.list.type) {
                            inputCount = inputTypeListAttr.list.type.length;
                        }
                    }
                    let inputArguments = inputs.slice(inputIndex, inputIndex + inputCount).map((id) => {
                        return new tf.Argument(id, null, initializers[id]);
                    });
                    this._inputs.push(new tf.Parameter(input.name, inputArguments));
                    inputIndex += inputCount;
                }
            }
            this._inputs = this._inputs.concat(inputs.slice(inputIndex).map((input, index) => {
                return new tf.Parameter((inputIndex + index).toString(), [
                    new tf.Argument(input, null, initializers[input])
                ]);
            }));
            let outputIndex = 0;
            let outputs = node.output;
            if (schema && schema.outputs) {
                for (const output of schema.outputs) {
                    let outputCount = 1;
                    if (output.numberAttr) {
                        let outputNumber = node.attr[output.numberAttr];
                        if (outputNumber && outputNumber.i) {
                            outputCount = outputNumber.i;
                        }
                    }
                    else if (output.typeListAttr) {
                        let outputTypeListAttr = node.attr[output.typeListAttr];
                        if (outputTypeListAttr && outputTypeListAttr.list && outputTypeListAttr.list.type) {
                            outputCount = outputTypeListAttr.list.type.length;
                        }
                    }
                    let outputArguments = outputs.slice(outputIndex, outputIndex + outputCount).map((id) => {
                        return new tf.Argument(id, null, null);
                    });
                    this._outputs.push(new tf.Parameter(output.name, outputArguments));
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
        else if (tensors) {
            for (const tensor of tensors) {
                this._inputs.push(new tf.Parameter(tensor.name, [
                    new tf.Argument(tensor.value.name, null, tensor.value)
                ]));
            }
        }
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
        const name = this._name;
        if (this._graph.namespaces[name]) {
            return name;
        }
        let lastIndex = name.lastIndexOf('/');
        if (lastIndex != -1) {
            let namespace = name.substring(0, lastIndex);
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

    get metadata() {
        return this._graph.metadata.type(this.operator);
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
        const schema = metadata.attribute(operator, name);
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
            if (typeof value.s === 'string') {
                this._value = value.s;
            }
            else if (ArrayBuffer.isView(value.s)) {
                this._value = (value.s.length === 0) ? '' : (value.s.filter(c => c <= 32 && c >= 128).length === 0) ? tf.Metadata.textDecoder.decode(value.s) : Array.from(value.s);
            }
            else {
                this._value = value.s;
            }
        }
        else if (Object.prototype.hasOwnProperty.call(value, 'list')) {
            let list = value.list;
            if (list.s && list.s.length > 0) {
                this._value = list.s.map((s) => {
                    if (typeof s === 'string') {
                        return s;
                    }
                    else if (ArrayBuffer.isView(s)) {
                        return (s.length === 0) ? '' : (s.filter(c => c <= 32 && c >= 128).length === 0) ? tf.Metadata.textDecoder.decode(s) : Array.from(s);
                    }
                    else {
                        return s;
                    }
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
            else {
                this._value = [];
            }
        }
        else if (Object.prototype.hasOwnProperty.call(value, 'func')) {
            const func = value.func;
            this._type = 'function';
            this._value = func.name;
        }

        if (schema) {
            if (Object.prototype.hasOwnProperty.call(schema, 'visible') && !schema.visible) {
                this._visible = false;
            }
            else if (Object.prototype.hasOwnProperty.call(schema, 'default')) {
                if (!Array.isArray(this._value) || Array.isArray(schema.default) || this._value.length === schema.default.length) {
                    let value = this._value;
                    let defaultValue = schema.default;
                    if (this._type === 'float32') {
                        let temp = new Float32Array(1);
                        temp[0] = value;
                        value = temp[0];
                        temp[0] = defaultValue;
                        defaultValue = temp[0];
                    }
                    const valueText = tf.GraphMetadata._formatAttributeValue(value);
                    const defaultValueText = tf.GraphMetadata._formatAttributeValue(defaultValue);
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
        const attributeVisibleMap = metadata.getAttributeVisibleMap(operator);
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
        this._type = new tf.TensorType(tensor.dtype, tensor.tensor_shape || tensor.tensorShape);
        this._name = name;
        this._kind = kind || null;
        this._tensor = tensor;
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
        let context = this._context();
        if (context.state) {
            return null;
        }
        context.limit = Number.MAX_SAFE_INTEGER;
        return this._decode(context, 0);
    }

    toString() {
        let context = this._context();
        if (context.state) {
            return '';
        }
        context.limit = 10000;
        const value = this._decode(context, 0);
        return JSON.stringify(value, null, 4);
    }

    _context() {
        let context = {};
        context.state = null;
        context.index = 0;
        context.count = 0;
        context.size = 1;

        if (!this._tensor.dtype) {
            context.state = 'Tensor has no data type.';
            return context;
        }
        const shape = this._tensor.tensor_shape || this._tensor.tensorShape;
        if (!shape || !shape.dim) {
            context.state = 'Tensor has no dimensions.';
            return context;
        }

        for (const dim of shape.dim) {
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
        let shape = context.shape;
        if (shape.length == 0) {
            shape = [ 1 ];
        }
        let results = [];
        const size = shape[dimension];
        if (dimension == shape.length - 1) {
            for (let i = 0; i < size; i++) {
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
                        switch (this._tensor.dtype) {
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
            for (let j = 0; j < size; j++) {
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
        const value = context.data[context.index++];
        if (this._tensor.dtype == tf.proto.DataType.DT_STRING) {
            return tf.Metadata.textDecoder.decode(value);
        }
        return value;
    }

    static formatDataType(type) {
        if (!tf.Tensor.dataType) {
            tf.Tensor.dataType = {};
            for (let key of Object.keys(tf.proto.DataType)) {
                const value = tf.proto.DataType[key];
                key = key.startsWith('DT_') ? key.substring(3) : key;
                tf.Tensor.dataType[value] = key.toLowerCase();
            }
            tf.Tensor.dataType[tf.proto.DataType.DT_HALF] = 'float16';
            tf.Tensor.dataType[tf.proto.DataType.DT_FLOAT] = 'float32';
            tf.Tensor.dataType[tf.proto.DataType.DT_DOUBLE] = 'float64';
            tf.Tensor.dataType['DT_FLOAT'] = 'float32';
        }
        return tf.Tensor.dataType[type] || '?';
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

tf.TensorBundle = class {

    static open(buffer, identifier, context, host) {
        const format = !identifier.toLowerCase().endsWith('.index') ? 1 : 2;
        if (buffer.length <= 48) {
            throw new tf.Error('Invalid index file size.');
        }
        let reader = new tf.TensorBundle.BinaryReader(buffer, host);
        reader.seek(-8);
        const signature = [ 0x57, 0xfb, 0x80, 0x8b, 0x24, 0x75, 0x47, 0xdb ];
        if (!reader.bytes(8).every((value, index) => value === signature[index])) {
            throw new tf.Error('Invalid table signature.');
        }
        reader.seek(-48);
        reader.varint64(); // metaindex offset
        reader.varint64(); // metaindex size
        const indexOffset = reader.varint64();
        const indexSize = reader.varint64();
        reader.seek(indexOffset);
        let indexData = reader.bytes(indexSize);
        let indexCompression = reader.byte();
        if (indexCompression !== 0) { // kNoCompression
            throw new tf.Error("Unsupported block compression '" + indexCompression + "'.");
        }
        let indexReader = new tf.TensorBundle.BinaryReader(indexData);
        indexReader.seek(-4);
        const numRestarts = indexReader.int32();
        indexReader.seek(-4 - (4 * numRestarts));
        let restartOffsets = [];
        for (let i = 0; i < numRestarts; i++) {
            restartOffsets.push(indexReader.int32());
        }
        const textDecoder = new TextDecoder();
        let entries = new Map();
        for (let i = 0; i < numRestarts; i++) {
            indexReader.seek(restartOffsets[i]);
            indexReader.varint32(); // index shared size
            const indexNonSharedSize = indexReader.varint32();
            const indexValueSize = indexReader.varint32();
            indexReader.skip(indexNonSharedSize);
            let indexValueReader = new tf.TensorBundle.BinaryReader(indexReader.bytes(indexValueSize));
            reader.seek(indexValueReader.varint64());
            let blockReader = new tf.TensorBundle.BinaryReader(reader.bytes(indexValueReader.varint64()));
            let key = '';
            while (!blockReader.end()) {
                const sharedSize = blockReader.varint32();
                const nonSharedSize = blockReader.varint32();
                const valueSize = blockReader.varint32();
                if (sharedSize === 0 && nonSharedSize === 0 && valueSize === 0) {
                    break;
                }
                key = key.substring(0, sharedSize);
                key = key + textDecoder.decode(blockReader.bytes(nonSharedSize));
                const value = blockReader.bytes(valueSize);
                entries.set(key, value);
            }
        }
        if (!entries.has('')) {
            throw new tf.Error('Bundle header not available.');
        }
        if (format === 1) {
            return Promise.resolve(new tf.TensorBundle(format, entries, []));
        }
        const header = tf.proto.BundleHeaderProto.decode(entries.get(''));
        const numShards = header.num_shards;
        let promises = [];
        for (let i = 0; i < numShards; i++) {
            const shardIndex = ('0000' + i).slice(-5);
            const shardCount = ('0000' + numShards).slice(-5);
            const filename = identifier.split('.');
            filename.pop();
            const basename = filename.join('.');
            const name = basename + '.data-' + shardIndex + '-of-' + shardCount;
            promises.push(context.request(name, null));
        }
        return Promise.all(promises).then((shards) => {
            return new tf.TensorBundle(format, entries, shards);
        }).catch((error) => {
            host.exception(error, false);
            return new tf.TensorBundle(format, entries, null);
        });
    }

    constructor(format, entries, shards) {
        this._format = format;
        this._tensors = [];
        switch (format) {
            case 1: {
                const header = tf.proto.SavedTensorSlices.decode(entries.get(''));
                let data = new Map();
                for (const pair of entries) {
                    if (pair[0] !== '' && pair[0] !== 'global_step') {
                        const slices = tf.proto.SavedTensorSlices.decode(pair[1]);
                        const name = slices.data.name;
                        const tensor = slices.data.data;
                        if (!data.has(name)) {
                            if (tensor.tensor_content && tensor.tensor_content.length > 0) {
                                data.set(name, { key: 'tensor_content', value: tensor.tensor_content });
                            }
                            else {
                                const keys = Object.keys(tensor).filter((key) => key.endsWith('_val') && tensor[key] && tensor[key].length > 0);
                                data.set(name, keys.length == 1 ? { key: keys[0], value: tensor[keys[0]] } : null);
                            }
                        }
                        else {
                            let item = data.get(name);
                            if (item !== null) {
                                if (tensor[item.key] && tensor[item.key].length > 0) {
                                    item.value = item.value.concat(tensor[item.key]);
                                }
                                else {
                                    data.set(name, null);
                                }
                            }
                        }
                    }
                }
                for (const meta of header.meta.tensor) {
                    if (meta.name !== 'global_step') {
                        let tensor = new tf.proto.TensorProto();
                        tensor.dtype = meta.type;
                        tensor.tensor_shape = meta.shape;
                        const item = data.get(meta.name);
                        if (item) {
                            tensor[item.key] = item.value;
                        }
                        this._tensors.push(new tf.Tensor(tensor, meta.name, null));
                    }
                }
                break;
            }
            case 2: {
                entries.forEach((value, name) => {
                    if (name !== '') {
                        const entry = tf.proto.BundleEntryProto.decode(value);
                        let tensor = new tf.proto.TensorProto();
                        tensor.dtype = entry.dtype;
                        tensor.tensor_shape = entry.shape;
                        const offset = (entry.offset instanceof long.Long) ? entry.offset.toNumber() : entry.offset;
                        const size = (entry.size instanceof long.Long) ? entry.size.toNumber() : entry.size;
                        if (shards) {
                            tensor.tensor_content = shards[entry.shard_id].slice(offset, offset + size);
                        }
                        this._tensors.push(new tf.Tensor(tensor, name, null));
                    }
                });
                break;
            }
        }
    }

    get format() {
        return this._format;
    }

    get tensors() {
        return this._tensors;
    }
};

tf.TensorBundle.BinaryReader = class {

    constructor(buffer) {
        this._buffer = buffer;
        this._dataView = new DataView(buffer.buffer, buffer.byteOffset, buffer.byteLength);
        this._position = 0;
    }

    seek(position) {
        this._position = position >= 0 ? position : this._buffer.length + position;
        if (this._position > this._buffer.length) {
            throw new tf.Error('Expected ' + (this._position - this._buffer.length) + ' more bytes. The file might be corrupted. Unexpected end of file.');
        }
    }

    skip(offset) {
        this._position += offset;
        if (this._position > this._buffer.length) {
            throw new tf.Error('Expected ' + (this._position - this._buffer.length) + ' more bytes. The file might be corrupted. Unexpected end of file.');
        }
    }

    end() {
        return this._position >= this._buffer.length;
    }

    bytes(length) {
        const position = this._position;
        this.skip(length);
        return this._buffer.subarray(position, this._position);
    }

    byte() {
        const position = this._position;
        this.skip(1);
        return this._dataView.getUint8(position);
    }

    int32() {
        const position = this._position;
        this.skip(4);
        return this._dataView.getInt32(position, true);
    }

    varint32() {
        return this.varint64();
    }

    varint64() {
        let result = 0;
        for (let shift = 0; shift <= 63; shift += 7) {
            let byte = this.byte();
            if (byte & 128) {
                result |= (byte & 127) << shift;
            }
            else {
                result |= byte << shift;
                break;
            }
        }
        return result;
    }
};

tf.GraphMetadata = class {

    constructor(metadata) {
        this._metadata = metadata;
        this._map = {};
        this._attributeCache = {};
    }

    type(operator) {
        var schema = this._metadata.type(operator);
        if (!schema) {
            schema = this._map[operator];
        }
        return schema;
    }

    attribute(operator, name) {
        let map = this._attributeCache[operator];
        if (!map) {
            map = {};
            const schema = this.type(operator);
            if (schema && schema.attributes && schema.attributes.length > 0) {
                for (const attribute of schema.attributes) {
                    map[attribute.name] = attribute;
                }
            }
            this._attributeCache[operator] = map;
        }
        return map[name] || null;
    }

    getAttributeVisibleMap(operator) {
        const schema = this.type(operator);
        if (schema) {
            let map = schema.__visisbleAttributeMap__;
            if (!map) {
                map = {};
                if (schema.inputs) {
                    for (const input of schema.inputs) {
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
                    for (const output of schema.outputs) {
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
                const items = JSON.parse(data);
                if (items) {
                    for (const item of items) {
                        if (item.name && item.schema) {
                            item.schema.name = item.name;
                            this._map[item.name] = item.schema;
                        }
                    }
                }
            }
        }
    }

    type(operator) {
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