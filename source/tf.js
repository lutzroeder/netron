/* jshint esversion: 6 */

// Experimental

var tf = tf || {};
var base = base || require('./base');
var json = json || require('./json');
var protobuf = protobuf || require('./protobuf');

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
        if (extension === 'pbtxt' || extension === 'prototxt' || extension === 'pt') {
            if (identifier.endsWith('predict_net.pbtxt') || identifier.endsWith('predict_net.prototxt') ||
                identifier.endsWith('init_net.pbtxt') || identifier.endsWith('init_net.prototxt')) {
                return false;
            }
            const tags = context.tags('pbtxt');
            if (['input_stream', 'output_stream', 'input_side_packet', 'output_side_packet'].some((key) => tags.has(key) || tags.has('node.' + key))) {
                return false;
            }
            if (tags.has('node') || tags.has('saved_model_schema_version') || tags.has('meta_graphs') || tags.has('graph_def')) {
                return true;
            }
        }
        if (extension === 'pb' || extension === 'pbtxt' || extension === 'prototxt' || extension === 'graphdef') {
            if (identifier.endsWith('predict_net.pb') || identifier.endsWith('init_net.pb')) {
                return false;
            }
            if (identifier == 'tfhub_module.pb') {
                const stream = context.stream;
                const signature = [ 0x08, 0x03 ];
                if (signature.length === stream.length && stream.peek(signature.length).every((value, index) => value === signature[index])) {
                    return false;
                }
            }
            const tags = context.tags('pb');
            if (tags.size === 0) {
                const tags = context.tags('pbtxt');
                if (['input_stream', 'output_stream', 'input_side_packet', 'output_side_packet'].some((key) => tags.has(key) || tags.has('node.' + key))) {
                    return false;
                }
                if (tags.has('node') || tags.has('saved_model_schema_version') || tags.has('meta_graphs') || tags.has('graph_def')) {
                    return true;
                }
            }
            else if (!Array.from(tags).some((pair) => pair[0] >= 5 || pair[1] === 5)) {
                if (tags.get(1) !== 2) {
                    return true;
                }
                const decode = (buffer, value) => {
                    const reader = protobuf.Reader.create(buffer);
                    const length = reader.length;
                    while (reader.position < length) {
                        const tag = reader.uint32();
                        const number = tag >>> 3;
                        const type = tag & 7;
                        if (value === number) {
                            return type === 2 ? reader.bytes() : null;
                        }
                        else {
                            reader.skipType(type);
                        }
                    }
                    return null;
                };
                const stream = context.stream;
                const buffer = stream.peek();
                const nodeBuffer = decode(buffer, 1);
                if (nodeBuffer) {
                    const nameBuffer = decode(nodeBuffer, 1);
                    if (nameBuffer) {
                        const decoder = new TextDecoder('utf-8');
                        const name = decoder.decode(nameBuffer);
                        if (Array.from(name).filter((c) => c <= ' ').length < 256) {
                            return true;
                        }
                    }
                }
            }
        }
        if (extension === 'json') {
            const obj = context.open('json');
            if (obj && obj.format && obj.format === 'graph-model' && obj.modelTopology) {
                return true;
            }
        }
        if (extension === 'index' || extension === 'ckpt') {
            const stream = context.stream;
            if (stream.length > 8) {
                stream.seek(-8);
                const buffer = stream.read(8);
                stream.seek(0);
                const signature = [ 0x57, 0xfb, 0x80, 0x8b, 0x24, 0x75, 0x47, 0xdb ];
                if (buffer.every((value, index) => value === signature[index])) {
                    return true;
                }
            }
        }
        if (/.data-[0-9][0-9][0-9][0-9][0-9]-of-[0-9][0-9][0-9][0-9][0-9]$/.exec(identifier)) {
            return true;
        }
        if (/^events.out.tfevents./.exec(identifier)) {
            if (tf.EventFileReader.open(context.stream)) {
                return true;
            }
        }
        return false;
    }

    open(context) {
        return context.require('./tf-proto').then(() => {
            tf.proto = protobuf.get('tf').tensorflow;
            let saved_model = null;
            let format = null;
            let producer = null;
            const openModel = (saved_model, format, producer, bundle) => {
                return tf.Metadata.open(context).then((metadata) => {
                    return new tf.Model(metadata, saved_model, format, producer, bundle);
                });
            };
            const openBundle = (context) => {
                const open = (stream, identifier, context) => {
                    return tf.TensorBundle.open(stream, identifier, context).then((bundle) => {
                        return openModel(null, 'TensorFlow Tensor Bundle v' + bundle.format.toString(), null, bundle);
                    }).catch((error) => {
                        context.exception(error, false);
                        const message = error && error.message ? error.message : error.toString();
                        throw new tf.Error(message.replace(/\.$/, '') + " in '" + identifier + "'.");
                    });
                };
                const identifier = context.identifier;
                if (/.data-[0-9][0-9][0-9][0-9][0-9]-of-[0-9][0-9][0-9][0-9][0-9]$/.exec(identifier.toLowerCase())) {
                    const base = identifier.split('.');
                    base.pop();
                    const index = base.join('.') + '.index';
                    return context.request(index, null).then((stream) => {
                        return open(stream, index, context);
                    }).catch((/* error */) => {
                        const ckpt = base.join('.') + '.ckpt';
                        return context.request(ckpt, null).then((stream) => {
                            open(stream, ckpt, context);
                        });
                    });
                }
                return open(context.stream, identifier, context);
            };
            const identifier = context.identifier;
            const extension = identifier.split('.').pop().toLowerCase();
            if (extension === 'ckpt' || extension === 'index' || /.data-[0-9][0-9][0-9][0-9][0-9]-of-[0-9][0-9][0-9][0-9][0-9]$/.exec(identifier)) {
                return openBundle(context);
            }
            else if (/^events.out.tfevents./.exec(identifier)) {
                format = 'TensorFlow Event File';
                const eventFileReader = tf.EventFileReader.open(context.stream);
                saved_model = new tf.proto.SavedModel();
                for (;;) {
                    const event = eventFileReader.read();
                    if (!event) {
                        break;
                    }
                    switch (event.what) {
                        case 'file_version': {
                            switch (event.file_version) {
                                case 'brain.Event:1': {
                                    format = 'TensorFlow Event File v1';
                                    break;
                                }
                                case 'brain.Event:2': {
                                    format = 'TensorFlow Event File v2';
                                    break;
                                }
                                default: {
                                    throw new tf.Error("Unknown event file version '" + event.file_version + "'.");
                                }
                            }
                            break;
                        }
                        case 'graph_def': {
                            const buffer = event.graph_def;
                            const reader = protobuf.Reader.create(buffer);
                            const graph_def = tf.proto.GraphDef.decode(reader);
                            const meta_graph = new tf.proto.MetaGraphDef();
                            meta_graph.meta_info_def = new tf.proto.MetaGraphDef.MetaInfoDef();
                            meta_graph.meta_info_def.any_info = event.wall_time.toString();
                            meta_graph.graph_def = graph_def;
                            saved_model.meta_graphs.push(meta_graph);
                            break;
                        }
                    }
                }
                if (saved_model.meta_graphs.every((meta_graph) => meta_graph.graph_def.node.every((node) => node.op.startsWith('aten::') || node.op.startsWith('prim::') || node.op === 'IO Node'))) {
                    producer = 'PyTorch';
                    return context.request('pytorch-metadata.json', 'utf-8', null).then((data) => {
                        const metadata = new Map();
                        for (const item of JSON.parse(data)) {
                            const index = item.name.indexOf(':');
                            const key = (index !== -1) ? item.name.substring(0, index) : item.name;
                            const name = key.replace(/^torch\./, 'aten::');
                            if (!metadata.has(name)) {
                                metadata.set(name, []);
                            }
                            metadata.get(name).push(item);
                        }
                        for (const meta_graph of saved_model.meta_graphs) {
                            for (const node of meta_graph.graph_def.node) {
                                node.__metadata__ = Array.from(metadata.get(node.op) || []);
                            }
                        }
                        return openModel(saved_model, format, producer, null);
                    }).catch(() => {
                        return openModel(saved_model, format, producer, null);
                    });
                }
            }
            else if (extension === 'json') {
                try {
                    const root = context.open('json');
                    const graph_def = new tf.proto.GraphDef();
                    const meta_graph = new tf.proto.MetaGraphDef();
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
                    throw new tf.Error('File text format is not TensorFlow.js graph-model (' + error.message + ').');
                }
            }
            else {
                const tags = context.tags('pbtxt');
                if (tags.has('node') || tags.has('saved_model_schema_version') || tags.has('meta_graphs') || tags.has('graph_def')) {
                    if (tags.has('saved_model_schema_version') || tags.has('meta_graphs')) {
                        try {
                            if (identifier.endsWith('saved_model.pbtxt') || identifier.endsWith('saved_model.prototxt')) {
                                const buffer = context.stream.peek();
                                const reader = protobuf.TextReader.create(buffer);
                                saved_model = tf.proto.SavedModel.decodeText(reader);
                                format = 'TensorFlow Saved Model';
                                if (saved_model && Object.prototype.hasOwnProperty.call(saved_model, 'saved_model_schema_version')) {
                                    format = format + ' v' + saved_model.saved_model_schema_version.toString();
                                }
                            }
                        }
                        catch (error) {
                            throw new tf.Error('File text format is not tensorflow.SavedModel (' + error.message + ').');
                        }
                    }
                    else if (tags.has('graph_def')) {
                        try {
                            if (!saved_model) {
                                const buffer = context.stream.peek();
                                const reader = protobuf.TextReader.create(buffer);
                                const meta_graph = tf.proto.MetaGraphDef.decodeText(reader);
                                saved_model = new tf.proto.SavedModel();
                                saved_model.meta_graphs.push(meta_graph);
                                format = 'TensorFlow MetaGraph';
                            }
                        }
                        catch (error) {
                            throw new tf.Error('File text format is not tensorflow.MetaGraphDef (' + error.message + ').');
                        }
                    }
                    else if (tags.has('node')) {
                        try {
                            const buffer = context.stream.peek();
                            const reader = protobuf.TextReader.create(buffer);
                            const graph_def = tf.proto.GraphDef.decodeText(reader);
                            const meta_graph = new tf.proto.MetaGraphDef();
                            meta_graph.graph_def = graph_def;
                            saved_model = new tf.proto.SavedModel();
                            saved_model.meta_graphs.push(meta_graph);
                            format = 'TensorFlow Graph';
                        }
                        catch (error) {
                            const message = error && error.message ? error.message : error.toString();
                            throw new tf.Error('File text format is not tensorflow.GraphDef (' + message.replace(/\.$/, '') + ').');
                        }
                    }
                }
                else {
                    try {
                        if (identifier.endsWith('saved_model.pb')) {
                            const buffer = context.stream.peek();
                            const reader = protobuf.Reader.create(buffer);
                            saved_model = tf.proto.SavedModel.decode(reader);
                            format = 'TensorFlow Saved Model';
                            if (saved_model && Object.prototype.hasOwnProperty.call(saved_model, 'saved_model_schema_version')) {
                                format = format + ' v' + saved_model.saved_model_schema_version.toString();
                            }
                        }
                    }
                    catch (error) {
                        const stream = context.stream;
                        const signature = [ 0x08, 0x01, 0x12 ];
                        if (signature.length < stream.length && stream.peek(3).every((value, index) => value === signature[index])) {
                            const message = error && error.message ? error.message : error.toString();
                            throw new tf.Error('File format is not tensorflow.SavedModel (' + message.replace(/\.$/, '') + ').');
                        }
                    }
                    try {
                        if (!saved_model && extension == 'meta') {
                            const buffer = context.stream.peek();
                            const reader = protobuf.Reader.create(buffer);
                            const meta_graph = tf.proto.MetaGraphDef.decode(reader);
                            saved_model = new tf.proto.SavedModel();
                            saved_model.meta_graphs.push(meta_graph);
                            format = 'TensorFlow MetaGraph';
                        }
                    }
                    catch (error) {
                        const message = error && error.message ? error.message : error.toString();
                        throw new tf.Error('File format is not tensorflow.MetaGraphDef (' + message.replace(/\.$/, '') + ').');
                    }
                    try {
                        if (!saved_model) {
                            const buffer = context.stream.peek();
                            const reader = protobuf.Reader.create(buffer);
                            const graph_def = tf.proto.GraphDef.decode(reader);
                            const meta_graph = new tf.proto.MetaGraphDef();
                            meta_graph.graph_def = graph_def;
                            saved_model = new tf.proto.SavedModel();
                            saved_model.meta_graphs.push(meta_graph);
                            format = 'TensorFlow Graph';
                        }
                    }
                    catch (error) {
                        const message = error && error.message ? error.message : error.toString();
                        throw new tf.Error('File format is not tensorflow.GraphDef (' + message.replace(/\.$/, '') + ').');
                    }
                }
                if (saved_model && saved_model.meta_graphs && saved_model.meta_graphs.length > 0 &&
                    saved_model.meta_graphs[0].meta_info_def &&
                    Object.prototype.hasOwnProperty.call(saved_model.meta_graphs[0].meta_info_def, 'tensorflow_version')) {
                    producer = 'TensorFlow v' + saved_model.meta_graphs[0].meta_info_def.tensorflow_version;
                }
            }
            if (saved_model.meta_graphs.length === 1 &&
                saved_model.meta_graphs[0].object_graph_def &&
                saved_model.meta_graphs[0].object_graph_def.nodes &&
                saved_model.meta_graphs[0].object_graph_def.nodes.length > 0) {
                const identifier = 'variables/variables.index';
                return context.request(identifier, null).then((stream) => {
                    return tf.TensorBundle.open(stream, identifier, context).then((bundle) => {
                        return openModel(saved_model, format, producer, bundle);
                    });
                }).catch(() => {
                    return openModel(saved_model, format, producer, null);

                });
            }
            return openModel(saved_model, format, producer, null);
        });
    }
};

tf.Model = class {

    constructor(metadata, model, format, producer, bundle) {
        this._format = format;
        this._producer = producer || '';
        this._graphs = [];
        if (model) {
            const graphs = [];
            for (let i = 0; i < model.meta_graphs.length; i++) {
                const meta_graph = model.meta_graphs[i];
                const name = (meta_graph.meta_info_def && meta_graph.meta_info_def.any_info) ? meta_graph.meta_info_def.any_info.toString() : ((model.meta_graphs.length > 1) ? i.toString() : '-');
                graphs.push(new tf.Graph(metadata, meta_graph, name, bundle));
            }
            // Recursively add all subgraphs.
            while (graphs.length > 0) {
                const graph = graphs.shift();
                this._graphs.push(graph);
                for (const func of graph.functions) {
                    graphs.push(func);
                }
            }
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

    constructor(metadata, meta_graph, name, bundle) {
        this._version = null;
        this._name = name;
        this._inputs = [];
        this._outputs = [];
        this._nodes = [];
        this._functions = [];

        if (meta_graph && meta_graph.graph_def) {
            metadata = new tf.GraphMetadata(metadata, meta_graph.meta_info_def);
            const graph = meta_graph.graph_def;
            if (graph.versions) {
                this._version = 'v' + graph.versions.producer.toString();
            }
            else if (graph.version) {
                this._version = graph.version;
            }
            else if (meta_graph.meta_info_def && meta_graph.meta_info_def.tensorflow_version) {
                this._version = meta_graph.meta_info_def.tensorflow_version;
            }
            if (meta_graph.meta_info_def && meta_graph.meta_info_def.tags) {
                this._tags = meta_graph.meta_info_def.tags.join(', ');
            }

            const nodes = graph.node;
            if (nodes) {
                const node_map = new Map();
                const namespaces = new Set();
                for (const node of nodes) {
                    const nodeName = node.name;
                    node_map.set(nodeName, node);
                    if (node.op != 'Const') {
                        const index = nodeName.lastIndexOf('/');
                        if (index != -1) {
                            const namespace = nodeName.substring(0, index);
                            namespaces.add(namespace);
                        }
                    }
                    node.output = [];
                }
                for (const node of nodes) {
                    const inputs = node.input;
                    node.input = [];
                    node.controlDependencies = [];
                    for (const input of inputs) {
                        const split = input.split(':', 2);
                        const input_name = split[0];
                        const input_index = split.length == 1 ? 0 : parseInt(split[1]);
                        const from_name = input_name.startsWith('^') ? input_name.substring(1) : input_name;
                        const from = node_map.get(from_name);
                        const output_name = input_index == 0 ? from_name : from_name + ':' + input_index.toString();
                        const input_arg = from ? { name: output_name, from: from } : { name: output_name };
                        if (input_name.startsWith('^')) {
                            node.controlDependencies.push(input_arg);
                        }
                        else {
                            node.input.push(input_arg);
                        }
                        if (from) {
                            for (let i = from.output.length; i <= input_index; i++) {
                                from.output.push({ name: i === 0 ? from_name : from_name + ':' + i.toString(), to: [] });
                            }
                            from.output[input_index].to.push(node);
                        }
                    }
                }

                const initializers = new Map();
                const map_tensor = (name, node, kind) => {
                    if (node.op === 'Const' && node.input.length === 0 && node.output.length === 1 && node.output[0].to.length === 1 && node.controlDependencies.length === 0) {
                        const value = node.attr.value;
                        if (value && Object.prototype.hasOwnProperty.call(value, 'tensor')) {
                            const tensor = new tf.Tensor(value.tensor, name, kind);
                            return new tf.Argument(name, tensor.type, tensor);
                        }
                    }
                    return null;
                };
                const map_resource = (name, node, tensor) => {
                    if (node.op === 'Placeholder' && node.input.length === 0 && node.output.length === 1 && node.output[0].to.length === 1 && node.controlDependencies.length === 0) {
                        const dtype = node.attr.dtype.type;
                        if (dtype === tf.proto.DataType.DT_RESOURCE) {
                            return new tf.Argument(name, null, tensor);
                        }
                    }
                    return null;
                };
                for (const node of node_map.values()) {
                    if (node.op === 'Identity' && node.input.length === 1 && node.output.length === 1 && node.output[0].to.length === 1 && node.controlDependencies.length === 0) {
                        const initializer = map_tensor(node.name, node.input[0].from, 'Identity Constant');
                        if (initializer) {
                            initializers.set(initializer.name, initializer);
                            node_map.delete(initializer.name);
                            node_map.delete(node.input[0].name);
                        }
                        const identity = node.input[0].from;
                        if (identity.op === 'Identity' && identity.input.length === 1 && identity.output.length === 1 && node.output[0].to.length === 1 && node.controlDependencies.length === 0) {
                            const initializer = map_tensor(node.name, identity.input[0].from, 'Identity Constant');
                            if (initializer) {
                                initializers.set(initializer.name, initializer);
                                node_map.delete(initializer.name);
                                node_map.delete(initializer.name);
                                node_map.delete(identity.name);
                                node_map.delete(node.name);
                            }
                        }
                    }
                }
                for (const node of node_map.values()) {
                    const initializer = map_tensor(node.name, node, 'Constant');
                    if (initializer) {
                        initializers.set(initializer.name, initializer);
                        node_map.delete(node.name);
                        node_map.delete(initializer.name);
                    }
                }
                for (const node of node_map.values()) {
                    if (node.op === 'ReadVariableOp' && node.input.length === 1 && node.output.length === 1 && node.output[0].to.length === 1 && node.controlDependencies.length === 0) {
                        if (node.attr && node.attr.dtype && node.attr._output_shapes && node.attr._output_shapes.list && node.attr._output_shapes.list.shape) {
                            const tensor = new tf.proto.TensorProto();
                            tensor.dtype = node.attr.dtype.type;
                            tensor.tensor_shape = node.attr._output_shapes.list.shape[0];
                            const initializer = map_resource(node.name, node.input[0].from, new tf.Tensor(tensor, name, 'Resource Variable'));
                            if (initializer) {
                                initializers.set(initializer.name, initializer);
                                node_map.delete(initializer.name);
                                node_map.delete(node.input[0].name);
                            }
                        }
                    }
                }
                const input_map = new Map();
                for (const node of node_map.values()) {
                    if (node.op == 'Placeholder' && node.input.length === 0 && node.output.length === 1 && node.output[0].to.length === 1 && node.controlDependencies.length === 0) {
                        const dtype = node.attr.dtype;
                        const shape = node.attr.shape;
                        if (dtype && dtype.type && shape && shape.shape) {
                            const name = node.name;
                            const type = new tf.TensorType(dtype.type, shape.shape);
                            const argument = new tf.Argument(name, type, null);
                            input_map.set(name, new tf.Parameter(name, [ argument ]));
                            node_map.delete(name);
                        }
                    }
                }
                const updatePyTorch = (node_map) => {
                    for (const node of node_map.values()) {
                        if (node.op === 'prim::Constant' && node.input.length === 0 && node.controlDependencies.length === 0 && node.attr && Object.keys(node.attr).length === 1 && node.attr.attr && node.attr.attr.s) {
                            const value = tf.Utility.decodeText(node.attr.attr.s);
                            const match = /{\s*value\s*:\s*(.*)\s*}/.exec(value);
                            if (match) {
                                node.value = match[1].trim();
                            }
                            const empty = /{\s*}/.exec(value);
                            if (empty) {
                                node.value = null;
                            }
                        }
                        if (node.op === 'prim::GetAttr' && node.input.length === 1 && node.controlDependencies.length === 0 && node.attr && Object.keys(node.attr).length === 1 && node.attr.attr && node.attr.attr.s) {
                            const value = tf.Utility.decodeText(node.attr.attr.s);
                            const match = /{\s*name\s*:\s*([A-za-z0-9_]*)\s*}/.exec(value);
                            if (match) {
                                node.value = match[1].trim();
                            }
                        }
                        if (node.op === 'IO Node' && node.controlDependencies.length === 0) {
                            const shape = node.attr && node.attr._output_shapes && node.attr._output_shapes.list && node.attr._output_shapes.list.shape ? node.attr._output_shapes.list.shape[0] : null;
                            const type = shape ? new tf.TensorType('?', shape) : null;
                            if (node.input.length === 0 && node.output.length === 1) {
                                this._inputs.push(new tf.Parameter(node.name, [
                                    new tf.Argument(node.output[0].name, type, null)
                                ]));
                                node_map.delete(node.name);
                            }
                            if (node.input.length === 1 && node.output.length === 0) {
                                this._outputs.push(new tf.Parameter(node.name, [
                                    new tf.Argument(node.input[0].name, type, null)
                                ]));
                                node_map.delete(node.name);
                            }
                        }
                        if (Object.keys(node.attr).length === 2 &&
                            node.attr.attr && node.attr.attr.s && node.attr._output_shapes) {
                            const value = tf.Utility.decodeText(node.attr.attr.s);
                            if (/\s*/.exec(value) || /{\s*}/.exec(value)) {
                                node.attr = {};
                                delete node._output_shapes;
                            }
                        }
                    }
                    const remove_input = (input, node) => {
                        const from = input.from;
                        if (from) {
                            for (const output of from.output) {
                                output.to = output.to.filter((to) => to !== node);
                            }
                            if (from.output.every((output) => output.to.length === 0) && from.controlDependencies.length === 0) {
                                from.remove = true;
                            }
                            delete input.from;
                        }
                    };
                    for (const node of node_map.values()) {
                        if (node.op === 'prim::ListConstruct' && node.input.every((input) => input.from.value !== undefined) && node.controlDependencies.length === 0) {
                            node.value = node.input.map((input) => input.from.value);
                            for (const input of node.input) {
                                remove_input(input, node);
                            }
                            node.input = [];
                        }
                    }
                    for (const node of node_map.values()) {
                        const remove = new Set();
                        for (let i = 0; i < node.input.length; i++) {
                            const input = node.input[i];
                            const from = input.from;
                            if (from) {
                                if (from.op === 'prim::GetAttr' && from.input.length === 1 && from.output.length === 1 && from.controlDependencies.length === 0 && from.value !== undefined) {
                                    remove_input(input, node);
                                    input.label = from.value;
                                    const tensor = new tf.Tensor(null, input.name, from.op);
                                    const argument = new tf.Argument(input.name, null, tensor);
                                    initializers.set(input.name, argument);
                                }
                                if (from.op === 'prim::Constant' && from.input.length === 0 && from.controlDependencies.length === 0 && from.value !== undefined) {
                                    input.constant = from.value;
                                    remove_input(input, node);
                                    remove.add(input.name);
                                }
                                if (from.op === 'prim::ListConstruct' && from.output.length === 1 && from.controlDependencies.length === 0 && from.value !== undefined) {
                                    input.list = from.value;
                                    remove_input(input, node);
                                    remove.add(input.name);
                                }
                            }
                        }
                        if (node.__metadata__) {
                            for (const metadata of node.__metadata__) {
                                const parameters = Array.prototype.slice.call(metadata.inputs || []).concat(Array.prototype.slice.call(metadata.attributes || []));
                                let match = true;
                                const inputs = Array.from(node.input);
                                if (inputs.length > parameters.length) {
                                    match = false;
                                }
                                while (inputs.length > 0 && match) {
                                    match = false;
                                    const input = inputs.shift();
                                    delete input.metadata;
                                    const parameter = parameters.shift();
                                    switch (parameter.type) {
                                        case 'Tensor': {
                                            if ((input.constant === undefined && input.list === undefined) || input.constant === null) {
                                                input.metadata = parameter;
                                                match = true;
                                            }
                                            else {
                                                inputs.unshift(input);
                                                match = true;
                                            }
                                            break;
                                        }
                                        case 'int64': {
                                            const value = parseInt(input.constant);
                                            if (input.constant !== undefined && Number.isInteger(value)) {
                                                input.attr = new tf.proto.AttrValue();
                                                input.attr.i = value;
                                                input.attr.metadata = parameter;
                                                match = true;
                                            }
                                            break;
                                        }
                                        case 'float32': {
                                            const value = parseFloat(input.constant);
                                            if (input.constant !== undefined && !isNaN(value)) {
                                                input.attr = new tf.proto.AttrValue();
                                                input.attr.f = value;
                                                input.attr.metadata = parameter;
                                                match = true;
                                            }
                                            break;
                                        }
                                        case 'int64[]': {
                                            if (Array.isArray(input.list)) {
                                                const list = input.list.map((item) => parseInt(item));
                                                if (list.every((value) => Number.isInteger(value))) {
                                                    input.attr = new tf.proto.AttrValue();
                                                    input.attr.list = new tf.proto.ListValue();
                                                    input.attr.list.i = list;
                                                    input.attr.metadata = parameter;
                                                    match = true;
                                                }
                                            }
                                            break;
                                        }
                                        case 'boolean': {
                                            if (input.constant === 'false' || input.constant === '0') {
                                                input.attr = new tf.proto.AttrValue();
                                                input.attr.b = false;
                                                input.attr.metadata = parameter;
                                                match = true;
                                            }
                                            else if (input.constant === 'true' || input.constant === '1') {
                                                input.attr = new tf.proto.AttrValue();
                                                input.attr.b = true;
                                                input.attr.metadata = parameter;
                                                match = true;
                                            }
                                            break;
                                        }
                                        case 'Scalar': {
                                            const value = parseInt(input.constant);
                                            if (input.constant !== undefined && Number.isInteger(value)) {
                                                input.attr = new tf.proto.AttrValue();
                                                input.attr.i = value;
                                                input.attr.metadata = parameter;
                                                match = true;
                                            }
                                            break;
                                        }
                                        default:
                                            break;
                                    }
                                }
                                if (match) {
                                    node.metadata = metadata;
                                    break;
                                }
                                else {
                                    for (const input of node.input) {
                                        delete input.metadata;
                                        delete input.attr;
                                    }
                                }
                            }
                        }
                        node.input = node.input.filter((input, index) => {
                            if (input.attr) {
                                const name = input.attr.metadata ? input.attr.metadata.name : index.toString();
                                node.attr[name] = input.attr;
                            }
                            else if (input.constant !== undefined && input.constant !== null) {
                                const attr = new tf.proto.AttrValue();
                                attr.s = input.constant;
                                node.attr[index.toString()] = attr;
                            }
                            else if (input.list !== undefined) {
                                const attr = new tf.proto.AttrValue();
                                attr.list = new tf.proto.ListValue();
                                attr.list.s = input.list;
                                node.attr[index.toString()] = attr;
                            }
                            return !remove.has(input.name);
                        });
                    }
                    for (const node of node_map.values()) {
                        if (node.op === 'prim::GetAttr' && node.remove) {
                            node_map.delete(node.name);
                        }
                        if (node.op === 'prim::Constant' && node.remove) {
                            node_map.delete(node.name);
                        }
                        if (node.op === 'prim::ListConstruct' && node.remove) {
                            node_map.delete(node.name);
                        }
                    }
                };
                updatePyTorch(node_map);
                for (const input of input_map.values()) {
                    this._inputs.push(input);
                }
                for (const node of node_map.values()) {
                    this._nodes.push(new tf.Node(metadata, namespaces, node, node.op, node.name, initializers, null));
                }
            }

            if (graph.library) {
                const funcs = graph.library.function;
                for (const func of funcs) {
                    this._functions.push(new tf.Function(this, func, metadata));
                }
            }
        }
        else if (bundle) {
            const nodeNames = [];
            const nodeMap = new Map();
            for (const tensor of bundle.tensors) {
                const parts = tensor.name.split('/');
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
                const tensorName = parts.pop();
                const nodeName = parts.join('/');
                if (!nodeMap.has(nodeName)) {
                    nodeNames.push(nodeName);
                    nodeMap.set(nodeName, []);
                }
                nodeMap.get(nodeName).push({ name: tensorName, value: tensor });
            }
            const namespaces = new Set();
            for (const name of nodeNames) {
                this._nodes.push(new tf.Node(metadata, namespaces, null, 'Node', name, null, nodeMap.get(name)));
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

    get functions() {
        return this._functions;
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
        this._functions = [];

        const input_arg = func.signature.input_arg;
        const output_arg = func.signature.output_arg;
        const ret = func.ret;

        if (input_arg) {
            for (const input of input_arg) {
                const argument = new tf.Argument(input.name, new tf.TensorType(input.type, null), null);
                this._inputs.push(new tf.Parameter(input.name, [ argument ]));
            }
        }
        const output_arg_map = new Map();
        if (output_arg) {
            const ret_map = new Map();
            for (const key of Object.keys(ret)) {
                const value = func.ret[key];
                const split = value.split(':', 2);
                ret_map.set(key, split[0]);
            }
            for (const output of output_arg) {
                const name = ret_map.get(output.name);
                this._outputs.push(new tf.Parameter(output.name, [
                    new tf.Argument(name, new tf.TensorType(output.type, null), null)
                ]));
                output_arg_map.set(name, output.name);
            }
        }

        const namespaces = new Set();
        const nodes = func.node_def;
        if (nodes) {
            const node_map = new Map();
            for (const node of nodes) {
                const nodeName = node.name;
                node_map.set(nodeName, node);
                if (node.op != 'Const') {
                    const lastIndex = nodeName.lastIndexOf('/');
                    if (lastIndex != -1) {
                        const namespace = nodeName.substring(0, lastIndex);
                        namespaces.add(namespace);
                    }
                }
                node.output = [];
            }
            for (const node of nodes) {
                const inputs = node.input;
                node.input = [];
                node.controlDependencies = [];
                for (const input of inputs) {
                    const split = input.split(':', 3);
                    const input_name = split[0];
                    const input_index = split.length == 1 ? 0 : parseInt(split[split.length - 1]);
                    const from_name = input_name.startsWith('^') ? input_name.substring(1) : input_name;
                    const from = node_map.get(from_name);
                    const output_name = from_name + (input_index == 0 ? '' : ':' + input_index.toString());
                    const input_arg = from ? { name: output_name, from: from } : { name: output_name };
                    if (input_name.startsWith('^')) {
                        node.controlDependencies.push(input_arg);
                    }
                    else {
                        node.input.push(input_arg);
                    }
                    if (from) {
                        for (let i = from.output.length; i <= input_index; i++) {
                            from.output.push({ name: i === 0 ? from_name : from_name + ':' + i.toString(), to: [] });
                        }
                        from.output[input_index].to.push(node);
                    }
                }
            }

            for (const node of nodes) {
                if (output_arg_map.has(node.name)) {
                    node.output.push({ name: node.name, to: [] });
                }
            }

            const initializers = new Map();
            const map_tensor = (name, node, kind) => {
                if (node && node.op === 'Const' && node.input.length === 0 && node.output.length === 1 && node.output[0].to.length === 1 && node.controlDependencies.length === 0) {
                    const value = node.attr.value;
                    if (value && Object.prototype.hasOwnProperty.call(value, 'tensor')) {
                        const tensor = new tf.Tensor(value.tensor, name, kind);
                        return new tf.Argument(name, tensor.type, tensor);
                    }
                }
                return null;
            };
            const map_resource = (name, node, tensor) => {
                if (node && node.op === 'Placeholder' && node.input.length === 0 && node.output.length === 1 && node.output[0].to.length === 1 && node.controlDependencies.length === 0) {
                    const dtype = node.attr.dtype.type;
                    if (dtype === tf.proto.DataType.DT_RESOURCE) {
                        return new tf.Argument(name, null, tensor);
                    }
                }
                return null;
            };
            for (const node of node_map.values()) {
                if (node.op === 'Identity' && node.input.length === 1 && node.output.length === 1 && node.output[0].to.length === 1 && node.controlDependencies.length === 0) {
                    const initializer = map_tensor(node.name, node.input[0].from, 'Identity Constant');
                    if (initializer) {
                        initializers.set(initializer.name, initializer);
                        node_map.delete(initializer.name);
                        node_map.delete(node.input[0].name);
                    }
                    const identity = node.input[0];
                    if (identity.op === 'Identity' && identity.input.length === 1 && identity.output.length === 1 && node.output[0].to.length === 1 && node.controlDependencies.length === 0) {
                        const initializer = map_tensor(node.name, identity.input[0], 'Identity Constant');
                        if (initializer) {
                            initializers.set(initializer.name, initializer);
                            node_map.delete(initializer.name);
                            node_map.delete(initializer.name);
                            node_map.delete(identity.name);
                            node_map.delete(node.name);
                        }
                    }
                }
            }
            for (const node of node_map.values()) {
                const initializer = map_tensor(node.name, node, 'Constant');
                if (initializer) {
                    initializers.set(initializer.name, initializer);
                    node_map.delete(node.name);
                    node_map.delete(initializer.name);
                }
            }
            for (const node of node_map.values()) {
                if (node.op === 'ReadVariableOp' && node.input.length === 1 && node.output.length === 1 && node.output[0].to.length === 1 && node.controlDependencies.length === 0 &&
                    node.attr && node.attr && node.attr.dtype && node.attr._output_shapes && node.attr._output_shapes.list && node.attr._output_shapes.list.shape) {
                    const tensor = new tf.proto.TensorProto();
                    tensor.dtype = node.attr.dtype.type;
                    tensor.tensor_shape = node.attr._output_shapes.list.shape[0];
                    const initializer = map_resource(node.name, node.input[0], new tf.Tensor(tensor, name, 'Resource Variable'));
                    if (initializer) {
                        initializers.set(initializer.name, initializer);
                        node_map.delete(initializer.name);
                        node_map.delete(node.input[0].name);
                    }
                }
            }

            for (const node of node_map.values()) {
                this._nodes.push(new tf.Node(metadata, namespaces, node, node.op, node.name, initializers, null));
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

    get functions() {
        return this._functions;
    }
};

tf.Node = class {

    constructor(metadata, namespaces, node, op, name, initializers, tensors) {
        this._type = op;
        this._metadata = node && node.metadata ? node.metadata : metadata.type(op);
        this._name = name;
        this._attributes = [];
        this._inputs = [];
        this._outputs = [];

        this._group = '';
        if (namespaces.has(name)) {
            this._group = name;
        }
        else {
            const lastIndex = name.lastIndexOf('/');
            if (lastIndex != -1) {
                const namespace = name.substring(0, lastIndex);
                if (namespaces.has(namespace)) {
                    this._group = namespace;
                }
            }
        }

        if (node) {
            if (node.device !== undefined) {
                this._device = node.device;
            }
            if (node.attr) {
                this._attributes = Object.keys(node.attr).map((name) => {
                    const value = node.attr[name];
                    const schema = value && value.metadata ? value.metadata : metadata.attribute(this._type, name);
                    const visible = metadata.visible(this._type, name);
                    return new tf.Attribute(schema, name, value, visible);
                });
            }
            const schema = this._metadata;
            let inputIndex = 0;
            const inputs = node.input.filter((input) => !input.name.startsWith('^'));
            if (schema && schema.inputs) {
                for (const input of schema.inputs) {
                    let inputCount = 1;
                    if (input.numberAttr) {
                        const inputNumber = node.attr[input.numberAttr];
                        if (inputNumber && inputNumber.i) {
                            inputCount = inputNumber.i;
                        }
                    }
                    else if (input.typeListAttr) {
                        const inputTypeListAttr = node.attr[input.typeListAttr];
                        if (inputTypeListAttr && inputTypeListAttr.list && inputTypeListAttr.list.type) {
                            inputCount = inputTypeListAttr.list.type.length;
                        }
                    }
                    const inputArguments = inputs.slice(inputIndex, inputIndex + inputCount).map((input) => {
                        return initializers.has(input.name) ? initializers.get(input.name) : new tf.Argument(input.name, null, null);
                    });
                    this._inputs.push(new tf.Parameter(input.name, inputArguments));
                    inputIndex += inputCount;
                }
            }
            this._inputs.push(...inputs.slice(inputIndex).map((input, index) => {
                return new tf.Parameter(input.label ? input.label : (inputIndex + index).toString(), [
                    initializers.has(input.name) ? initializers.get(input.name) : new tf.Argument(input.name, null, null)
                ]);
            }));
            let outputIndex = 0;
            const outputs = node.output;
            if (schema && schema.outputs) {
                for (const output of schema.outputs) {
                    let outputCount = 1;
                    if (output.numberAttr) {
                        const outputNumber = node.attr[output.numberAttr];
                        if (outputNumber && outputNumber.i) {
                            outputCount = outputNumber.i;
                        }
                    }
                    else if (output.typeListAttr) {
                        const outputTypeListAttr = node.attr[output.typeListAttr];
                        if (outputTypeListAttr && outputTypeListAttr.list && outputTypeListAttr.list.type) {
                            outputCount = outputTypeListAttr.list.type.length;
                        }
                    }
                    const outputArguments = outputs.slice(outputIndex, outputIndex + outputCount).map((output) => {
                        return new tf.Argument(output.name ? output.name : '-', null, null);
                    });
                    this._outputs.push(new tf.Parameter(output.name, outputArguments));
                    outputIndex += outputCount;
                }
            }
            this._outputs.push(...outputs.slice(outputIndex).map((output, index) => {
                return new tf.Parameter((outputIndex + index).toString(), [
                    new tf.Argument(output.name ? output.name : '-', null, null)
                ]);
            }));
            this._controlDependencies = node.controlDependencies.map((input) => input.name);
        }
        else if (tensors) {
            for (const tensor of tensors) {
                this._inputs.push(new tf.Parameter(tensor.name, [
                    new tf.Argument(tensor.value.name, null, tensor.value)
                ]));
            }
        }
    }

    get type() {
        return this._type;
    }

    get name() {
        return this._name;
    }

    get device() {
        return this._device || null;
    }

    get group() {
        return this._group;
    }

    get description() {
        return '';
    }

    get domain() {
        return null;
    }

    get metadata() {
        return this._metadata;
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

    constructor(schema, name, value, visible) {
        this._name = name;
        this._value = null;
        this._type = null;
        if (Object.prototype.hasOwnProperty.call(value, 'tensor')) {
            this._type = 'tensor';
            this._value = new tf.Tensor(value.tensor);
        }
        else if (schema && schema.type) {
            this._type = schema.type;
        }
        switch (value.value) {
            case 'type':
                this._type = 'type';
                this._value = tf.Tensor.formatDataType(value.type);
                break;
            case 'i':
                this._value = value.i;
                break;
            case 'f':
                this._value = value.f;
                break;
            case 'b':
                this._value = value.b;
                break;
            case 'shape':
                this._type = 'shape';
                this._value = new tf.TensorShape(value.shape);
                break;
            case 's':
                this._value = tf.Utility.decodeText(value.s);
                break;
            case 'func': {
                const func = value.func;
                this._type = 'function';
                this._value = func.name;
                break;
            }
            case 'list': {
                const list = value.list;
                if (list.s && list.s.length > 0) {
                    this._value = list.s.map((s) => tf.Utility.decodeText(s));
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
                break;
            }
        }

        if (schema) {
            if (Object.prototype.hasOwnProperty.call(schema, 'visible') && !schema.visible) {
                this._visible = false;
            }
            else if (Object.prototype.hasOwnProperty.call(schema, 'default')) {
                const equals = (value, defaultValue) => {
                    if (!Array.isArray(defaultValue) && defaultValue === Object(defaultValue)) {
                        switch (defaultValue.type) {
                            case 'type':
                                defaultValue = tf.Tensor.formatDataType(defaultValue.value);
                                break;
                            case 'shape':
                            case 'tensor':
                                defaultValue = defaultValue.value;
                                break;
                            default:
                                throw new tf.Error(JSON.stringify(defaultValue));
                        }
                    }
                    switch (typeof value) {
                        case 'boolean':
                        case 'number':
                        case 'string':
                            return value === defaultValue;
                    }
                    if (value instanceof base.Int64 || value instanceof base.Uint64) {
                        return value.toNumber() === defaultValue;
                    }
                    return false;
                };
                const value = this._value;
                const defaultValue = schema.default;
                if (Array.isArray(value) && Array.isArray(defaultValue)) {
                    if (value.length === defaultValue.length && value.every((item, index) => equals(item, defaultValue[index]))) {
                        this._visible = false;
                    }
                }
                else {
                    if (equals(value, defaultValue)) {
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
        if (visible === false) {
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
        this._name = name;
        this._kind = kind || null;
        if (tensor) {
            this._type = new tf.TensorType(tensor.dtype, tensor.tensor_shape || tensor.tensorShape);
            this._tensor = tensor;
            if (Object.prototype.hasOwnProperty.call(tensor, 'tensor_content')) {
                this._buffer = tensor.tensor_content;
            }
            else {
                switch (tensor.dtype) {
                    case tf.proto.DataType.DT_FLOAT:
                        this._data = tensor.float_val || null;
                        break;
                    case tf.proto.DataType.DT_DOUBLE:
                        this._data = tensor.double_val || null;
                        break;
                    case tf.proto.DataType.DT_INT8:
                    case tf.proto.DataType.DT_UINT8:
                    case tf.proto.DataType.DT_INT32:
                        this._data = tensor.int_val || null;
                        break;
                    case tf.proto.DataType.DT_UINT32:
                        this._data = tensor.uint32_val || null;
                        break;
                    case tf.proto.DataType.DT_INT64:
                        this._data = tensor.int64_val || null;
                        break;
                    case tf.proto.DataType.DT_UINT64:
                        this._data = tensor.uint64_val || null;
                        break;
                    case tf.proto.DataType.DT_BOOL:
                        this._data = tensor.bool_val || null;
                        break;
                    case tf.proto.DataType.DT_STRING:
                        this._data = tensor.string_val || null;
                        break;
                }
            }
        }
        else {
            this._type = new tf.TensorType('?', null);
            this._tensor = null;
        }
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
        const context = this._context();
        if (context.state) {
            return null;
        }
        context.limit = Number.MAX_SAFE_INTEGER;
        return this._decode(context, 0);
    }

    toString() {
        const context = this._context();
        if (context.state) {
            return '';
        }
        context.limit = 10000;
        const value = this._decode(context, 0);
        return tf.Tensor._stringify(value, '', '    ');
    }

    _context() {
        const context = {};
        context.state = null;
        context.index = 0;
        context.count = 0;
        context.size = 1;

        if (!this._tensor) {
            context.state = 'Tensor has content.';
            return context;
        }

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

        if (this._buffer) {
            switch (this._tensor.dtype) {
                case tf.proto.DataType.DT_FLOAT:
                case tf.proto.DataType.DT_DOUBLE:
                case tf.proto.DataType.DT_QINT8:
                case tf.proto.DataType.DT_QUINT8:
                case tf.proto.DataType.DT_INT8:
                case tf.proto.DataType.DT_UINT8:
                case tf.proto.DataType.DT_INT32:
                case tf.proto.DataType.DT_UINT32:
                case tf.proto.DataType.DT_INT64:
                case tf.proto.DataType.DT_UINT64:
                    context.rawData = new DataView(this._buffer.buffer, this._buffer.byteOffset, this._buffer.byteLength);
                    break;
            }
        }
        else if (this._data) {
            if (this._data.length == context.size) {
                context.data = this._data;
            }
            else if (this._data.length === 1) {
                context.data = new Array(context.size).fill(this._data[0]);
            }
            else {
                context.state = "Tensor has no data.";
                return context;
            }
        }
        else {
            context.state = "Tensor has no data.";
            return context;
        }

        if (!context.data && !context.rawData) {
            context.state = "Tensor data type '" + this.type.dataType + "' is not implemented.";
            return context;
        }

        context.shape = shape.dim.map((dim) => dim.size);
        return context;
    }

    _decode(context, dimension) {
        let shape = context.shape;
        if (shape.length == 0) {
            shape = [ 1 ];
        }
        const results = [];
        const size = shape[dimension];
        if (dimension == shape.length - 1) {
            for (let i = 0; i < size; i++) {
                if (context.count > context.limit) {
                    results.push('...');
                    return results;
                }
                if (context.data) {
                    const value = context.data[context.index++];
                    results.push((this._tensor.dtype == tf.proto.DataType.DT_STRING) ? tf.Utility.decodeText(value) : value);
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
                            case tf.proto.DataType.DT_DOUBLE:
                                results.push(context.rawData.getFloat64(context.index, true));
                                context.index += 8;
                                context.count++;
                                break;
                            case tf.proto.DataType.DT_INT8:
                                results.push(context.rawData.getInt8(context.index));
                                context.index += 1;
                                context.count++;
                                break;
                            case tf.proto.DataType.DT_UINT8:
                                results.push(context.rawData.getUint8(context.index));
                                context.index += 4;
                                context.count++;
                                break;
                            case tf.proto.DataType.DT_INT32:
                                results.push(context.rawData.getInt32(context.index, true));
                                context.index += 4;
                                context.count++;
                                break;
                            case tf.proto.DataType.DT_UINT32:
                                results.push(context.rawData.getUint32(context.index, true));
                                context.index += 4;
                                context.count++;
                                break;
                            case tf.proto.DataType.DT_INT64:
                                results.push(context.rawData.getInt64(context.index, true));
                                context.index += 8;
                                context.count++;
                                break;
                            case tf.proto.DataType.DT_UINT64:
                                results.push(context.rawData.getUint64(context.index, true));
                                context.index += 8;
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
                            default:
                                throw new tf.Error("Unsupported data type '" + this._tensor.dtype + "'.");
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

    static _stringify(value, indentation, indent) {
        if (Array.isArray(value)) {
            const result = [];
            result.push(indentation + '[');
            const items = value.map((item) => tf.Tensor._stringify(item, indentation + indent, indent));
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

    static open(stream, identifier, context) {
        const format = !identifier.toLowerCase().endsWith('.index') ? 1 : 2;
        const table = new tf.TensorBundle.Table(stream);
        if (!table.entries.has('')) {
            throw new tf.Error('Bundle header not available.');
        }
        if (format === 1) {
            return Promise.resolve(new tf.TensorBundle(format, table.entries, []));
        }
        const entry = table.entries.get('');
        const reader = protobuf.Reader.create(entry);
        const header = tf.proto.BundleHeaderProto.decode(reader);
        const numShards = header.num_shards;
        const promises = [];
        for (let i = 0; i < numShards; i++) {
            const shardIndex = ('0000' + i).slice(-5);
            const shardCount = ('0000' + numShards).slice(-5);
            const filename = identifier.split('.');
            filename.pop();
            const basename = filename.join('.');
            const name = basename + '.data-' + shardIndex + '-of-' + shardCount;
            promises.push(context.request(name, null));
        }
        return Promise.all(promises).then((streams) => {
            return new tf.TensorBundle(format, table.entries, streams);
        }).catch((error) => {
            context.exception(error, false);
            return new tf.TensorBundle(format, table.entries, null);
        });
    }

    constructor(format, entries, streams) {
        this._format = format;
        this._tensors = [];
        switch (format) {
            case 1: {
                const buffer = entries.get('');
                const reader = protobuf.Reader.create(buffer);
                const header = tf.proto.SavedTensorSlices.decode(reader);
                const data = new Map();
                for (const pair of entries) {
                    if (pair[0] !== '' && pair[0] !== 'global_step') {
                        const reader = protobuf.Reader.create(pair[1]);
                        const slices = tf.proto.SavedTensorSlices.decode(reader);
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
                            const item = data.get(name);
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
                        const tensor = new tf.proto.TensorProto();
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
                        const reader = protobuf.Reader.create(value);
                        const entry = tf.proto.BundleEntryProto.decode(reader);
                        const tensor = new tf.proto.TensorProto();
                        tensor.dtype = entry.dtype;
                        tensor.tensor_shape = entry.shape;
                        const offset = Number.isInteger(entry.offset) ? entry.offset : entry.offset.toNumber();
                        const size = Number.isInteger(entry.size) ? entry.size : entry.size.toNumber();
                        if (streams) {
                            const stream = streams[entry.shard_id];
                            stream.seek(offset);
                            tensor.tensor_content = stream.peek(size);
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

tf.TensorBundle.Table = class {

    constructor(stream) {
        // https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/lib/io/table.cc
        this.entries = new Map();
        if (stream.length <= 54) {
            throw new tf.Error('Invalid index file size.');
        }
        stream.seek(-48);
        const buffer = stream.peek(48);
        const reader = new tf.BinaryReader(buffer);
        reader.seek(-8);
        const signature = [ 0x57, 0xfb, 0x80, 0x8b, 0x24, 0x75, 0x47, 0xdb ];
        if (!reader.read(8).every((value, index) => value === signature[index])) {
            throw new tf.Error('Invalid table signature.');
        }
        reader.seek(-48); // kEncodedLength
        reader.varint64(); // metaindex offset
        reader.varint64(); // metaindex size
        const indexOffset = reader.varint64();
        const indexSize = reader.varint64();
        const indexBlock = new tf.TensorBundle.Table.Block(stream, indexOffset, indexSize);
        for (const entry of indexBlock.entries) {
            const valueReader = new tf.BinaryReader(entry[1]);
            const offset = valueReader.varint64();
            const size = valueReader.varint64();
            const block = new tf.TensorBundle.Table.Block(stream, offset, size);
            for (const pair of block.entries) {
                this.entries.set(pair[0], pair[1]);
            }
        }
        stream.seek(0);
    }
};

tf.TensorBundle.Table.Block = class {

    constructor(stream, offset, size) {
        // https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/lib/io/block.cc
        this.entries = new Map();
        stream.seek(offset);
        const buffer = stream.read(size); // blockContents
        const compression = stream.byte();
        stream.skip(4); // crc32
        let reader = new tf.BinaryReader(buffer);
        switch (compression) {
            case 0: // kNoCompression
                break;
            case 1: // kSnappyCompression
                reader = new tf.BinaryReader(reader.unsnappy());
                break;
            default:
                throw new tf.Error("Unsupported block compression '" + compression + "'.");
        }
        reader.seek(-4);
        const numRestarts = reader.int32();
        reader.seek(-4 - (4 * numRestarts));
        const restartOffsets = [];
        for (let i = 0; i < numRestarts; i++) {
            restartOffsets.push(reader.int32());
        }
        const textDecoder = new TextDecoder();
        for (let i = 0; i < numRestarts; i++) {
            reader.seek(restartOffsets[i]);
            let key = '';
            while (reader.position < reader.length) {
                const sharedSize = reader.varint32(); // index shared size
                const nonSharedSize = reader.varint32(); // index non shared size
                const valueSize = reader.varint32();
                if (sharedSize === 0 && nonSharedSize === 0 && valueSize === 0) {
                    break;
                }
                key = key.substring(0, sharedSize);
                key = key + textDecoder.decode(reader.read(nonSharedSize));
                const value = reader.read(valueSize);
                this.entries.set(key, value);
            }
        }
    }
};

tf.BinaryReader = class {

    constructor(buffer) {
        this._buffer = buffer;
        this._position = 0;
        this._length = this._buffer.length;
        this._dataView = new DataView(buffer.buffer, buffer.byteOffset, buffer.byteLength);
    }

    get position() {
        return this._position;
    }

    get length() {
        return this._length;
    }

    seek(position) {
        this._position = position >= 0 ? position : this._length + position;
        if (this._position > this._length) {
            throw new tf.Error('Expected ' + (this._position - this._length) + ' more bytes. The file might be corrupted. Unexpected end of file.');
        }
    }

    skip(offset) {
        this._position += offset;
        if (this._position > this._length) {
            throw new tf.Error('Expected ' + (this._position - this._length) + ' more bytes. The file might be corrupted. Unexpected end of file.');
        }
    }

    read(size) {
        const position = this._position;
        this.skip(size);
        return this._buffer.subarray(position, this._position);
    }

    byte() {
        const position = this._position;
        this.skip(1);
        return this._dataView.getUint8(position);
    }

    uint16() {
        const position = this._position;
        this.skip(2);
        return this._dataView.getUint16(position, true);
    }

    int32() {
        const position = this._position;
        this.skip(4);
        return this._dataView.getInt32(position, true);
    }

    uint32() {
        const position = this._position;
        this.skip(4);
        return this._dataView.getUint32(position, true);
    }

    uint64() {
        const position = this._position;
        this.skip(4);
        return this._dataView.getUint64(position, true);
    }

    varint32() {
        return this.varint64();
    }

    varint64() {
        let result = 0;
        for (let shift = 0; shift <= 63; shift += 7) {
            const byte = this.byte();
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

    unsnappy() {
        const data = new Uint8Array(this.varint64());
        const mask = [0, 0xff, 0xffff, 0xffffff, 0xffffffff];
        let position = 0;
        while (this._position < this._length) {
            let length = 0;
            const c = this.byte();
            switch (c & 0x03) {
                case 0: {
                    length = (c >>> 2) + 1;
                    if (length > 60) {
                        const short = length - 60;
                        length = (this.uint32() & mask[short]) + 1;
                        this._position += short - 4;
                    }
                    data.set(this.read(length), position);
                    break;
                }
                case 1: {
                    length = ((c >>> 2) & 0x07) + 4;
                    const offset = this.byte() + ((c >>> 5) << 8);
                    data.set(data.subarray(position - offset, position - offset + length), position);
                    break;
                }
                case 2: {
                    length = (c >>> 2) + 1;
                    const offset = this.uint16();
                    data.set(data.subarray(position - offset, position - offset + length), position);
                    break;
                }
                case 3: {
                    length = (c >>> 2) + 1;
                    const offset = this.uint32();
                    data.set(data.subarray(position - offset, position - offset + length), position);
                    break;
                }
            }
            position += length;
        }
        return data;
    }
};

tf.EventFileReader = class {

    static open(stream) {
        if (stream.length < 16) {
            return null;
        }
        const masked_crc32c = (bytes) => {
            const poly = 0x82f63b78;
            let crc = 0xffffffff;
            for (let n = 0; n < bytes.length; n++) {
                crc ^= bytes[n];
                crc = crc & 1 ? (crc >>> 1) ^ poly : crc >>> 1;
                crc = crc & 1 ? (crc >>> 1) ^ poly : crc >>> 1;
                crc = crc & 1 ? (crc >>> 1) ^ poly : crc >>> 1;
                crc = crc & 1 ? (crc >>> 1) ^ poly : crc >>> 1;
                crc = crc & 1 ? (crc >>> 1) ^ poly : crc >>> 1;
                crc = crc & 1 ? (crc >>> 1) ^ poly : crc >>> 1;
                crc = crc & 1 ? (crc >>> 1) ^ poly : crc >>> 1;
                crc = crc & 1 ? (crc >>> 1) ^ poly : crc >>> 1;
                crc = crc >>> 0;
            }
            crc = crc ^ 0xffffffff;
            crc = crc >>> 0;
            crc = ((crc >> 15) | (crc << 17)) + 0xa282ead8;
            crc = crc >>> 0;
            return crc;
        };
        const buffer = stream.peek(12);
        const reader = new tf.BinaryReader(buffer);
        const length_bytes = reader.read(8);
        const length_crc = reader.uint32();
        if (masked_crc32c(length_bytes) !== length_crc) {
            return null;
        }
        // reader.skip(-12);
        // const length = reader.uint64().toNumber();
        // reader.uint32(); // masked crc of length
        // const data = reader.read(length);
        // const data_crc = reader.uint32();
        // if (masked_crc32c(data) !== data_crc) {
        //     return null;
        // }
        return new tf.EventFileReader(stream);
    }

    constructor(stream) {
        this._stream = stream;
    }

    read() {
        if (this._stream.position < this._stream.length) {
            const buffer = this._stream.read(12);
            const reader = new tf.BinaryReader(buffer);
            const length = reader.uint64().toNumber();
            reader.uint32(); // masked crc of length
            const data = this._stream.read(length);
            const event = tf.proto.Event.decode(protobuf.Reader.create(data));
            this._stream.skip(4); // masked crc of data
            return event;
        }
    }
};

tf.GraphMetadata = class {

    constructor(metadata) {
        this._metadata = metadata;
        this._attributeCache = new Map();
        this._visibleCache = new Map();
    }

    type(operator) {
        return this._metadata.type(operator);
    }

    attribute(type, name) {
        const key = type + '::' + name;
        if (!this._attributeCache.has(key)) {
            const schema = this.type(type);
            if (schema && schema.attributes) {
                for (const attribute of schema.attributes) {
                    const key = type + '::'  + attribute.name;
                    this._attributeCache.set(key, attribute);
                }
            }
        }
        return this._attributeCache.get(key);
    }

    visible(type, name) {
        if (!this._visibleCache.has(type)) {
            const set = new Set();
            const schema = this.type(type);
            if (schema && schema.inputs) {
                for (const input of schema.inputs) {
                    if (input.typeAttr) {
                        set.add(input.typeAttr);
                    }
                    else if (input.typeListAttr) {
                        set.add(input.typeListAttr);
                    }
                    if (input.numberAttr) {
                        set.add(input.numberAttr);
                    }
                }
            }
            if (schema && schema.outputs) {
                for (const output of schema.outputs) {
                    if (output.typeAttr) {
                        set.add(output.typeAttr);
                    }
                    else if (output.typeListAttr) {
                        set.add(output.typeListAttr);
                    }
                    if (output.numberAttr) {
                        set.add(output.numberAttr);
                    }
                }
            }
            this._visibleCache.set(type, set);
        }
        return !this._visibleCache.get(type).has(name);
    }
};

tf.Metadata = class {

    static open(context) {
        if (tf.Metadata._metadata) {
            return Promise.resolve(tf.Metadata._metadata);
        }
        return context.request('tf-metadata.json', 'utf-8', null).then((data) => {
            tf.Metadata._metadata = new tf.Metadata(data);
            return tf.Metadata._metadata;
        }).catch(() => {
            tf.Metadata._metadata = new tf.Metadata(null);
            return tf.Metadata._metadata;
        });
    }

    constructor(data) {
        this._map = new Map();
        if (data) {
            const metadata = JSON.parse(data);
            this._map = new Map(metadata.map((item) => [ item.name, item ]));
        }
    }

    type(operator) {
        return this._map.get(operator);
    }
};

tf.Utility = class {

    static decodeText(value) {
        if (typeof value === 'string') {
            return value;
        }
        if (value.length === 0) {
            return '';
        }
        tf.Utility._utf8Decoder = tf.Utility._utf8Decoder || new TextDecoder('utf-8');
        return tf.Utility._utf8Decoder.decode(value);
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