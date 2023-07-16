
// Experimental

var tf = {};
var base = require('./base');
var protobuf = require('./protobuf');
var zip = require('./zip');

tf.ModelFactory = class {

    match(context) {
        const identifier = context.identifier;
        const extension = identifier.split('.').pop().toLowerCase();
        if (extension === 'pbtxt' || extension === 'prototxt' || extension === 'pt' || extension === 'txt') {
            if (identifier.endsWith('predict_net.pbtxt') || identifier.endsWith('predict_net.prototxt') ||
                identifier.endsWith('init_net.pbtxt') || identifier.endsWith('init_net.prototxt')) {
                return undefined;
            }
            const tags = context.tags('pbtxt');
            if (['input_stream', 'output_stream', 'input_side_packet', 'output_side_packet'].some((key) => tags.has(key) || tags.has('node.' + key))) {
                return undefined;
            }
            if (tags.has('saved_model_schema_version') || tags.has('meta_graphs')) {
                return 'tf.pbtxt.SavedModel';
            }
            if (tags.has('graph_def')) {
                return 'tf.pbtxt.MetaGraphDef';
            }
            if (tags.has('node')) {
                return 'tf.pbtxt.GraphDef';
            }
        }
        if (extension === 'pb' || extension === 'pbtxt' || extension === 'prototxt' || extension === 'graphdef' || extension === 'meta') {
            if (identifier.endsWith('predict_net.pb') || identifier.endsWith('init_net.pb')) {
                return undefined;
            }
            if (identifier == 'tfhub_module.pb') {
                const stream = context.stream;
                const signature = [ 0x08, 0x03 ];
                if (signature.length === stream.length && stream.peek(signature.length).every((value, index) => value === signature[index])) {
                    return undefined;
                }
            }
            const tags = context.tags('pb');
            if (tags.size > 0) {
                if (Array.from(tags).every((pair) => pair[0] < 8 && pair[1] !== 5)) {
                    const match = (tags, schema) => {
                        for (const pair of schema) {
                            const key = pair[0];
                            const inner = pair[1];
                            const value = tags[key];
                            if (value === undefined) {
                                continue;
                            }
                            if (inner === false) {
                                return false;
                            }
                            if (Array.isArray(inner)) {
                                if (typeof value !== 'object' || !match(value, inner)) {
                                    return false;
                                }
                            } else if (inner !== value) {
                                if (inner === 2 && !Array.isArray(value) && Object(value) === (value) && Object.keys(value).length === 0) {
                                    return true;
                                }
                                return false;
                            }
                        }
                        return true;
                    };
                    const signatureGraphDef = [
                        [1 /* node */, [
                            [1 /* name */, 2],
                            [2 /* op */, 2],
                            [3 /* input */, 2],
                            [4 /* device */,2],
                            [5 /* attr */, [
                                [1,2],
                                [2,[]]
                            ]],
                            [6 /* experimental_debug_info */, []]
                        ]],
                        [2 /* library */, []],
                        [3 /* version */, 0],
                        [4 /* versions */, [[1,0],[2,0]]]
                    ];
                    const signatureMetaGraphDef = [
                        [1 /* meta_info_def */, [[1,2],[2,[]],[3,[]],/* [4,2], */[6,2],[7,0],[8,[]]]],
                        [2 /* graph_def */, signatureGraphDef],
                        [3 /* saver_def */, [[1,2],[2,2],[3,2],[4,0],[5,0],[6,5],[7,0]]],
                        [4 /* collection_def */,[]],
                        [5 /* signature_def */, []],
                        [6 /* asset_file_def */, []],
                        [7 /* object_graph_def */, []]
                    ];
                    const signatureSavedModel = [[1,0],[2,signatureMetaGraphDef]];
                    if (tags.size === 1 && tags.get(1) === 2) {
                        const tags = context.tags('pb+');
                        // mediapipe.BoxDetectorIndex
                        if (match(tags, [[1,[[1,[[1,[[1,5],[2,5],[3,5],[4,5],[6,0],[7,5],[8,5],[10,5],[11,0],[12,0]]],[2,5],[3,[]]]],[2,false],[3,false],[4,false],[5,false]]],[2,false],[3,false]])) {
                            return undefined;
                        }
                        // third_party.tensorflow.python.keras.protobuf.SavedMetadata
                        if (match(tags, [[1,[[1,[[1,0],[2,0]]],[2,0],[3,2],[4,2],[5,2]]]])) {
                            return 'tf.pb.keras.SavedMetadata';
                        }
                    }
                    if ((!tags.has(1) || tags.get(1) === 0) && tags.get(2) === 2) {
                        const tags = context.tags('pb+');
                        if (match(tags, signatureSavedModel)) {
                            return 'tf.pb.SavedModel';
                        }
                    }
                    if ((!tags.has(1) || tags.get(1) === 2) &&
                        (!tags.has(2) || tags.get(2) === 2) &&
                        (!tags.has(3) || tags.get(3) === 2) &&
                        (!tags.has(4) || tags.get(4) === 2)) {
                        const tags = context.tags('pb+');
                        if (match(tags, signatureMetaGraphDef)) {
                            return 'tf.pb.MetaGraphDef';
                        }
                    }
                    if (tags.get(1) !== 2) {
                        const tags = context.tags('pb+');
                        if (match(tags, signatureGraphDef)) {
                            return 'tf.pb.GraphDef';
                        }
                    }
                    // tensorflow.FingerprintDef
                    if (identifier === 'fingerprint.pb' &&
                        tags.get(1) === 0 && tags.get(2) === 0 &&
                        tags.get(3) === 0 && tags.get(4) === 0 &&
                        tags.get(5) === 0 && tags.get(6) === 2) {
                        return 'tf.pb.FingerprintDef';
                    }
                    const decode = (buffer, value) => {
                        const reader = protobuf.BinaryReader.open(buffer);
                        const length = reader.length;
                        while (reader.position < length) {
                            const tag = reader.uint32();
                            const number = tag >>> 3;
                            const type = tag & 7;
                            if (value === number) {
                                return type === 2 ? reader.bytes() : null;
                            }
                            reader.skipType(type);
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
                                return 'tf.pb.GraphDef';
                            }
                        }
                    }
                }
            } else {
                const tags = context.tags('pbtxt');
                if (['input_stream', 'output_stream', 'input_side_packet', 'output_side_packet'].some((key) => tags.has(key) || tags.has('node.' + key))) {
                    return undefined;
                }
                if (tags.has('node')) {
                    return 'tf.pbtxt.GraphDef';
                }
                if (tags.has('graph_def')) {
                    return 'tf.pbtxt.MetaGraphDef';
                }
                if (tags.has('saved_model_schema_version') || tags.has('meta_graphs')) {
                    return 'tf.pbtxt.SavedModel';
                }
            }
        }
        if (extension === 'json') {
            for (const type of [ 'json', 'json.gz' ]) {
                const obj = context.open(type);
                if (obj && obj.modelTopology && (obj.format === 'graph-model' || Array.isArray(obj.modelTopology.node))) {
                    return 'tf.' + type;
                }
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
                    return 'tf.bundle';
                }
            }
        }
        if (/.data-[0-9][0-9][0-9][0-9][0-9]-of-[0-9][0-9][0-9][0-9][0-9]$/.exec(identifier)) {
            return 'tf.data';
        }
        if (/^events.out.tfevents./.exec(identifier)) {
            const stream = context.stream;
            if (tf.EventFileReader.open(stream)) {
                return 'tf.events';
            }
        }
        if (extension === 'pbmm') {
            const stream = context.stream;
            if (stream.length > 8) {
                stream.seek(-8);
                const buffer = stream.read(8);
                stream.seek(0);
                const reader = new base.BinaryReader(buffer);
                const offset = reader.uint64();
                if (offset < stream.length) {
                    return 'tf.pb.mmap';
                }
            }
        }
        return undefined;
    }

    async open(context, target) {
        await context.require('./tf-proto');
        tf.proto = protobuf.get('tf');
        const openModel = async (saved_model, format, producer, bundle) => {
            const metadata = await context.metadata('tf-metadata.json');
            return new tf.Model(metadata, saved_model, format, producer, bundle);
        };
        const openSavedModel = async (saved_model, format, producer) => {
            if (saved_model.meta_graphs.length === 1 &&
                saved_model.meta_graphs[0].object_graph_def &&
                saved_model.meta_graphs[0].object_graph_def.nodes &&
                saved_model.meta_graphs[0].object_graph_def.nodes.length > 0) {
                const identifier = 'variables/variables.index';
                try {
                    const stream = await context.request(identifier, null);
                    const bundle = await tf.TensorBundle.open(stream, identifier, context);
                    return openModel(saved_model, format, producer, bundle);
                } catch (error) {
                    return openModel(saved_model, format, producer, null);
                }
            }
            if (saved_model && saved_model.meta_graphs && saved_model.meta_graphs.length > 0 &&
                saved_model.meta_graphs[0].meta_info_def &&
                Object.prototype.hasOwnProperty.call(saved_model.meta_graphs[0].meta_info_def, 'tensorflow_version')) {
                producer = 'TensorFlow v' + saved_model.meta_graphs[0].meta_info_def.tensorflow_version;
            }
            return openModel(saved_model, format, producer, null);
        };
        const openBundle = async (context, stream, identifier) => {
            stream = stream || context.stream;
            identifier = identifier || context.identifier;
            try {
                const bundle = await tf.TensorBundle.open(stream, identifier, context);
                return openModel(null, 'TensorFlow Tensor Bundle v' + bundle.format.toString(), null, bundle);
            } catch (error) {
                context.exception(error, false);
                throw error;
            }
        };
        const openData = async (context) => {
            const identifier = context.identifier;
            const base = identifier.split('.');
            base.pop();
            const file = base.join('.') + '.index';
            try {
                const stream = await context.request(file, null);
                return openBundle(context, stream, file);
            } catch (error) {
                const file = base.join('.') + '.ckpt';
                const stream = await context.request(file, null);
                return openBundle(context, stream, file);
            }
        };
        const openEventFile = async (context) => {
            let format = 'TensorFlow Event File';
            let producer = null;
            const stream = context.stream;
            const eventFileReader = tf.EventFileReader.open(stream);
            const saved_model = new tf.proto.tensorflow.SavedModel();
            const run_metadata = [];
            const summaries = [];
            for (;;) {
                const event = eventFileReader.read();
                if (!event) {
                    break;
                }
                switch (event.what) {
                    case 'file_version': {
                        const formats = new Map([
                            [ 'brain.Event:1', 'TensorFlow Event File v1' ],
                            [ 'brain.Event:2', 'TensorFlow Event File v2' ]
                        ]);
                        if (!formats.has(event.file_version)) {
                            throw new tf.Error("Unsupported event file version '" + event.file_version + "'.");
                        }
                        format = formats.get(event.file_version);
                        break;
                    }
                    case 'graph_def': {
                        const buffer = event.graph_def;
                        const reader = protobuf.BinaryReader.open(buffer);
                        const graph_def = tf.proto.tensorflow.GraphDef.decode(reader);
                        const meta_graph_def = new tf.proto.tensorflow.MetaGraphDef();
                        meta_graph_def.meta_info_def = new tf.proto.tensorflow.MetaGraphDef.MetaInfoDef();
                        meta_graph_def.meta_info_def.any_info = event.wall_time.toString();
                        meta_graph_def.graph_def = graph_def;
                        saved_model.meta_graphs.push(meta_graph_def);
                        break;
                    }
                    case 'meta_graph_def': {
                        const buffer = event.meta_graph_def;
                        const reader = protobuf.BinaryReader.open(buffer);
                        const meta_graph_def = tf.proto.tensorflow.MetaGraphDef.decode(reader);
                        saved_model.meta_graphs.push(meta_graph_def);
                        break;
                    }
                    case 'summary': {
                        for (const value of event.summary.value) {
                            summaries.push(value);
                        }
                        break;
                    }
                    case 'tagged_run_metadata': {
                        const entry = event.tagged_run_metadata;
                        const buffer = entry.run_metadata;
                        const reader = protobuf.BinaryReader.open(buffer);
                        const metadata = tf.proto.tensorflow.RunMetadata.decode(reader);
                        run_metadata.push(metadata);
                        break;
                    }
                    default: {
                        throw new tf.Error("Unsupported event type '" + event.what + "'.");
                    }
                }
            }
            if (saved_model.meta_graphs.every((meta_graph) => meta_graph.graph_def.node.every((node) => node.op.startsWith('aten::') || node.op.startsWith('prim::') || node.op.startsWith('quantized::') || node.op === 'IO Node'))) {
                producer = 'PyTorch';
                const openPyTorchMetadata = async (context, saved_model) => {
                    try {
                        const data = await context.request('pytorch-metadata.json', 'utf-8', null);
                        const metadata = new Map();
                        for (const item of JSON.parse(data)) {
                            const name = item.name;
                            if (name.indexOf('::') !== -1) {
                                const index = name.indexOf('.');
                                const key = (index !== -1) ? name.substring(0, index) : name;
                                if (!metadata.has(key)) {
                                    metadata.set(key, []);
                                }
                                metadata.get(key).push(item);
                            }
                        }
                        for (const graph of saved_model.meta_graphs) {
                            for (const node of graph.graph_def.node) {
                                node.__metadata__ = Array.from(metadata.get(node.op) || []);
                            }
                        }
                        return saved_model;
                    } catch (error) {
                        return saved_model;
                    }
                };
                const updated_saved_model = await openPyTorchMetadata(context, saved_model);
                return openModel(updated_saved_model, format, producer, null);
            }
            return openSavedModel(saved_model, format, producer);
        };
        const openJson = async (context, type) => {
            try {
                const obj = context.open(type);
                const format = 'TensorFlow.js ' + (obj.format || 'graph-model');
                const producer = obj.convertedBy || obj.generatedBy || '';
                const meta_graph = new tf.proto.tensorflow.MetaGraphDef();
                meta_graph.graph_def = tf.JsonReader.decodeGraphDef(obj.modelTopology);
                const saved_model = new tf.proto.tensorflow.SavedModel();
                saved_model.meta_graphs.push(meta_graph);
                const nodes = new Map();
                for (const node of meta_graph.graph_def.node) {
                    node.input = node.input || [];
                    if (node.op === 'Const') {
                        nodes.set(node.name, node);
                    }
                }
                const shards = new Map();
                const manifests = Array.isArray(obj.weightsManifest) ? obj.weightsManifest : [];
                for (const manifest of manifests) {
                    for (const path of manifest.paths) {
                        if (!shards.has(path)) {
                            shards.set(path, context.request(path, null));
                        }
                    }
                }
                const openShards = (shards) => {
                    const dtype_size_map = new Map([
                        [ 'float16', 2 ], [ 'float32', 4 ], [ 'float64', 8 ],
                        [ 'int8', 1 ], [ 'int16', 2 ], [ 'int32', 4 ], [ 'int64', 8 ],
                        [ 'uint8', 1 ], [ 'uint16', 2 ], [ 'uint32', 4 ], [ 'uint64', 8 ],
                        [ 'bool', 1 ]
                    ]);
                    for (const manifest of manifests) {
                        let buffer = null;
                        if (Array.isArray(manifest.paths) && manifest.paths.length > 0 && manifest.paths.every((path) => shards.has(path))) {
                            const list = manifest.paths.map((path) => shards.get(path));
                            const size = list.reduce((a, b) => a + b.length, 0);
                            buffer = new Uint8Array(size);
                            let offset = 0;
                            for (const item of list) {
                                buffer.set(item, offset);
                                offset += item.length;
                            }
                        }
                        let offset = 0;
                        for (const weight of manifest.weights) {
                            const dtype = weight.quantization && weight.quantization.dtype ? weight.quantization.dtype : weight.dtype;
                            const size = weight.shape.reduce((a, b) => a * b, 1);
                            switch (dtype) {
                                case 'string': {
                                    const data = [];
                                    if (buffer && size > 0) {
                                        const reader = new tf.BinaryReader(buffer.subarray(offset));
                                        for (let i = 0; i < size; i++) {
                                            data[i] = reader.string();
                                        }
                                        offset += reader.position;
                                    }
                                    if (nodes.has(weight.name)) {
                                        const node = nodes.get(weight.name);
                                        node.attr.value.tensor.dtype = tf.Utility.dataTypeKey(dtype);
                                        node.attr.value.tensor.string_val = data;
                                    }
                                    break;
                                }
                                default: {
                                    if (!dtype_size_map.has(dtype)) {
                                        throw new tf.Error("Unsupported weight data type size '" + dtype + "'.");
                                    }
                                    const itemsize = dtype_size_map.get(dtype);
                                    const length = itemsize * size;
                                    const tensor_content = buffer ? buffer.slice(offset, offset + length) : null;
                                    offset += length;
                                    if (nodes.has(weight.name)) {
                                        const node = nodes.get(weight.name);
                                        node.attr.value.tensor.dtype = tf.Utility.dataTypeKey(dtype);
                                        node.attr.value.tensor.tensor_content = tensor_content;
                                    }
                                    break;
                                }
                            }
                        }
                    }
                    return openSavedModel(saved_model, format, producer);
                };
                try {
                    const streams = await Promise.all(shards.values());
                    for (const key of shards.keys()) {
                        const stream = streams.shift();
                        const buffer = stream.peek();
                        shards.set(key, buffer);
                    }
                    if (type === 'json.gz') {
                        try {
                            for (const key of shards.keys()) {
                                const stream = shards.get(key);
                                const archive = zip.Archive.open(stream, 'gzip');
                                if (archive && archive.entries.size === 1) {
                                    const stream = archive.entries.values().next().value;
                                    const buffer = stream.peek();
                                    shards.set(key, buffer);
                                }
                            }
                        } catch (error) {
                            // continue regardless of error
                        }
                    }
                    return openShards(shards);
                } catch (error) {
                    shards.clear();
                    return openShards(shards);
                }
            } catch (error) {
                throw new tf.Error('File text format is not TensorFlow.js graph-model (' + error.message + ').');
            }
        };
        const openTextGraphDef = (context) => {
            try {
                const stream = context.stream;
                const reader = protobuf.TextReader.open(stream);
                const graph_def = tf.proto.tensorflow.GraphDef.decodeText(reader);
                const meta_graph = new tf.proto.tensorflow.MetaGraphDef();
                meta_graph.graph_def = graph_def;
                const saved_model = new tf.proto.tensorflow.SavedModel();
                saved_model.meta_graphs.push(meta_graph);
                const format = 'TensorFlow Graph';
                return openSavedModel(saved_model, format, null);
            } catch (error) {
                const message = error && error.message ? error.message : error.toString();
                throw new tf.Error('File text format is not tensorflow.GraphDef (' + message.replace(/\.$/, '') + ').');
            }
        };
        const openTextMetaGraphDef = (context) => {
            try {
                const stream = context.stream;
                const reader = protobuf.TextReader.open(stream);
                const meta_graph = tf.proto.tensorflow.MetaGraphDef.decodeText(reader);
                const saved_model = new tf.proto.tensorflow.SavedModel();
                saved_model.meta_graphs.push(meta_graph);
                const format = 'TensorFlow MetaGraph';
                return openSavedModel(saved_model, format, null);
            } catch (error) {
                throw new tf.Error('File text format is not tensorflow.MetaGraphDef (' + error.message + ').');
            }
        };
        const openTextSavedModel = (context) => {
            try {
                const stream = context.stream;
                const reader = protobuf.TextReader.open(stream);
                const saved_model = tf.proto.tensorflow.SavedModel.decodeText(reader);
                let format = 'TensorFlow Saved Model';
                if (saved_model && Object.prototype.hasOwnProperty.call(saved_model, 'saved_model_schema_version')) {
                    format = format + ' v' + saved_model.saved_model_schema_version.toString();
                }
                return openSavedModel(saved_model, format, null);
            } catch (error) {
                throw new tf.Error('File text format is not tensorflow.SavedModel (' + error.message + ').');
            }
        };
        const openBinaryGraphDef = (context) => {
            let saved_model = null;
            const format = 'TensorFlow Graph';
            try {
                const stream = context.stream;
                const reader = protobuf.BinaryReader.open(stream);
                const graph_def = tf.proto.tensorflow.GraphDef.decode(reader);
                const meta_graph = new tf.proto.tensorflow.MetaGraphDef();
                meta_graph.graph_def = graph_def;
                saved_model = new tf.proto.tensorflow.SavedModel();
                saved_model.meta_graphs.push(meta_graph);
            } catch (error) {
                const message = error && error.message ? error.message : error.toString();
                throw new tf.Error('File format is not tensorflow.GraphDef (' + message.replace(/\.$/, '') + ').');
            }
            return openSavedModel(saved_model, format, null);
        };
        const openBinaryMetaGraphDef = (context) => {
            let saved_model = null;
            const format = 'TensorFlow MetaGraph';
            try {
                const stream = context.stream;
                const reader = protobuf.BinaryReader.open(stream);
                const meta_graph = tf.proto.tensorflow.MetaGraphDef.decode(reader);
                saved_model = new tf.proto.tensorflow.SavedModel();
                saved_model.meta_graphs.push(meta_graph);
            } catch (error) {
                const message = error && error.message ? error.message : error.toString();
                throw new tf.Error('File format is not tensorflow.MetaGraphDef (' + message.replace(/\.$/, '') + ').');
            }
            return openSavedModel(saved_model, format, null);
        };
        const openBinarySavedModel = (context) => {
            let saved_model = null;
            let format = 'TensorFlow Saved Model';
            try {
                const stream = context.stream;
                const reader = protobuf.BinaryReader.open(stream);
                saved_model = tf.proto.tensorflow.SavedModel.decode(reader);
                if (saved_model && Object.prototype.hasOwnProperty.call(saved_model, 'saved_model_schema_version')) {
                    format = format + ' v' + saved_model.saved_model_schema_version.toString();
                }
            } catch (error) {
                const message = error && error.message ? error.message : error.toString();
                throw new tf.Error('File format is not tensorflow.SavedModel (' + message.replace(/\.$/, '') + ').');
            }
            return openSavedModel(saved_model, format, null);
        };
        const openSavedMetadata = async (context) => {
            /*
            const stream = context.stream;
            const reader = protobuf.BinaryReader.open(stream);
            const saved_metadata = tf.proto.third_party.tensorflow.python.keras.protobuf.SavedMetadata.decode(reader);
            debugger;
            */
            const identifier = 'saved_model.pb';
            const stream = await context.request(identifier, null);
            return openBinarySavedModel({ stream: stream });
        };
        const openFingerprint = async (context) => {
            const identifier = 'saved_model.pb';
            const stream = await context.request(identifier, null);
            return openBinarySavedModel({ stream: stream });
        };
        const openMemmapped = (context) => {
            const stream = context.stream;
            const readDirectoryOffset = (stream) => {
                stream.seek(-8);
                const buffer = stream.read(8);
                const reader = new base.BinaryReader(buffer);
                return reader.uint64();
            };
            const readDirectory = (stream, offset) => {
                const end = stream.position - 8;
                stream.seek(offset);
                const buffer = stream.read(end - offset);
                const reader = protobuf.BinaryReader.open(buffer);
                return tf.proto.tensorflow.MemmappedFileSystemDirectory.decode(reader);
            };
            const offset = readDirectoryOffset(stream);
            const directory = readDirectory(stream, offset);
            const elements = new Map();
            for (const element of directory.element) {
                const name = element.name;
                if (elements.has(name)) {
                    throw new tf.Error("Memory mapped file directory contains duplicate '" + name + "'.");
                }
                elements.set(name, {
                    offset: element.offset ? element.offset.toNumber() : 0,
                    length: element.length ? element.length.toNumber() : 0
                });
            }
            const offsets = Array.from(elements).map((entry) => entry[1].offset);
            offsets.push(offset);
            for (const value of elements.values()) {
                if (value.length === 0) {
                    const min = Math.min.apply(null, offsets.filter((offset) => offset > value.offset));
                    if (Number.isInteger(min)) {
                        value.length = min - value.offset;
                    }
                }
            }
            for (const entry of elements) {
                const offset = entry[1].offset;
                const length = entry[1].length;
                stream.seek(offset);
                entry[1].buffer = stream.read(length);
            }
            if (!elements.has('memmapped_package://.')) {
                throw new tf.Error('Memory mapped file directory does not contain tensorflow.GraphDef root.');
            }
            const element = elements.get('memmapped_package://.');
            const buffer = element.buffer;
            const reader = protobuf.BinaryReader.open(buffer);
            const graph_def = tf.proto.tensorflow.GraphDef.decode(reader);
            const format = 'TensorFlow GraphDef Memmapped';
            const meta_graph = new tf.proto.tensorflow.MetaGraphDef();
            meta_graph.graph_def = graph_def;
            const saved_model = new tf.proto.tensorflow.SavedModel();
            saved_model.meta_graphs.push(meta_graph);
            return openSavedModel(saved_model, format, null);
        };
        switch (target) {
            case 'tf.bundle':
                return openBundle(context);
            case 'tf.data':
                return openData(context);
            case 'tf.events':
                return openEventFile(context);
            case 'tf.json':
                return openJson(context, 'json');
            case 'tf.json.gz':
                return openJson(context, 'json.gz');
            case 'tf.pbtxt.GraphDef':
                return openTextGraphDef(context);
            case 'tf.pbtxt.MetaGraphDef':
                return openTextMetaGraphDef(context);
            case 'tf.pbtxt.SavedModel':
                return openTextSavedModel(context);
            case 'tf.pb.GraphDef':
                return openBinaryGraphDef(context);
            case 'tf.pb.MetaGraphDef':
                return openBinaryMetaGraphDef(context);
            case 'tf.pb.SavedModel':
                return openBinarySavedModel(context);
            case 'tf.pb.keras.SavedMetadata':
                return openSavedMetadata(context);
            case 'tf.pb.FingerprintDef':
                return openFingerprint(context);
            case 'tf.pb.mmap':
                return openMemmapped(context);
            default:
                throw new tf.Error("Unsupported TensorFlow format '" + target + "'.");
        }
    }
};

tf.Model = class {

    constructor(metadata, model, format, producer, bundle) {
        this._format = format;
        this._producer = producer || '';
        this._graphs = [];
        if (model) {
            for (let i = 0; i < model.meta_graphs.length; i++) {
                const meta_graph = model.meta_graphs[i];
                const name = (meta_graph.meta_info_def && meta_graph.meta_info_def.any_info) ? meta_graph.meta_info_def.any_info.toString() : ((model.meta_graphs.length > 1) ? i.toString() : '-');
                const graph = new tf.Graph(metadata, meta_graph, name, bundle);
                this._graphs.push(graph);
            }
        } else {
            const graph = new tf.Graph(metadata, null, '', bundle);
            this._graphs.push(graph);
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
        this._name = name;
        this._inputs = [];
        this._outputs = [];
        this._nodes = [];
        this._version = null;
        if (meta_graph && meta_graph.graph_def) {
            const graph = meta_graph.graph_def;
            if (graph.versions) {
                this._version = 'v' + graph.versions.producer.toString();
            } else if (graph.version) {
                this._version = graph.version;
            } else if (meta_graph.meta_info_def && meta_graph.meta_info_def.tensorflow_version) {
                this._version = meta_graph.meta_info_def.tensorflow_version;
            }
            if (meta_graph.meta_info_def && meta_graph.meta_info_def.tags) {
                this._tags = meta_graph.meta_info_def.tags.join(', ');
            }
            metadata = new tf.GraphMetadata(metadata, graph.library);
            const nodes = graph.node || [];
            const context = new tf.Context();
            context.graph(metadata, nodes);
            this._nodes = context.nodes;
            this._inputs = context.inputs;
            this._outputs = context.outputs;
        } else if (bundle) {
            const nodes = new Map();
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
                const name = parts.join('/');
                if (!nodes.has(name)) {
                    nodes.set(name, []);
                }
                nodes.get(name).push({ name: tensorName, value: tensor });
            }
            const namespaces = new Set();
            this._nodes = Array.from(nodes).map((entry) => {
                const node = { op: 'Node', name: entry[0] };
                return new tf.Node(metadata, node, namespaces, new tf.Context(), entry[1]);
            });
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
};

tf.Argument = class {

    constructor(name, value) {
        this._name = name;
        this._value = value;
    }

    get name() {
        return this._name;
    }

    get value() {
        return this._value;
    }
};

tf.Value = class {

    constructor(name, type, initializer) {
        if (typeof name !== 'string') {
            throw new tf.Error("Invalid value identifier '" + JSON.stringify(name) + "'.");
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

    constructor(metadata, name, func) {
        this._name = name;
        this._version = null;
        this._tags = null;
        this._inputs = [];
        this._outputs = [];
        this._nodes = [];
        this._description = !func ? 'Function definition not found.' : null;
        const context = new tf.Context();
        const input_arg = func && func.signature ? func.signature.input_arg : [];
        const output_arg = func && func.signature ? func.signature.output_arg : [];
        const ret = func && func.ret ? func.ret : {};
        const nodes = func && func.node_def ? func.node_def : [];
        if (input_arg) {
            for (const input of input_arg) {
                const value = context.value(input.name, new tf.TensorType(input.type, null), null);
                this._inputs.push(new tf.Argument(input.name, [ value ]));
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
                const type = new tf.TensorType(output.type, null);
                const argument = new tf.Argument(output.name, [ context.value(name, type, null) ]);
                this._outputs.push(argument);
                output_arg_map.set(name, output.name);
            }
        }
        context.graph(metadata, nodes, output_arg_map);
        this._nodes = context.nodes;
        this._inputs = this._inputs.concat(context.inputs);
        this._outputs = this._outputs.concat(context.outputs);
    }

    get type() {
        return 'function';
    }

    get name() {
        return this._name;
    }

    get description() {
        return this._description || '';
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
};

tf.Node = class {

    constructor(metadata, node, namespaces, context, tensors) {
        this._type = node.metadata || metadata.type(node.op) || { name: node.op };
        this._name = node.name;
        this._attributes = [];
        this._inputs = [];
        this._outputs = [];
        this._group = '';
        if (node.name) {
            if (namespaces.has(node.name)) {
                this._group = node.name;
            } else {
                const index = node.name.lastIndexOf('/');
                if (index != -1) {
                    const namespace = node.name.substring(0, index);
                    if (namespaces.has(namespace)) {
                        this._group = namespace;
                    }
                }
            }
        }
        if (tensors) {
            for (const tensor of tensors) {
                const value = context.value(tensor.value.name, null, tensor.value);
                const argument = new tf.Argument(tensor.name, [ value ]);
                this._inputs.push(argument);
            }
        } else {
            if (node.device !== undefined) {
                this._device = node.device;
            }
            if (node.attr) {
                this._attributes = Object.entries(node.attr).map((entry) => {
                    return new tf.Attribute(metadata, node.op, entry[0], entry[1]);
                });
            }
            let inputIndex = 0;
            const inputs = (node.input || []).filter((input) => !input.name.startsWith('^'));
            if (this._type && this._type.inputs) {
                for (const input of this._type.inputs) {
                    let count = 1;
                    if (input.numberAttr) {
                        const inputNumber = node.attr[input.numberAttr];
                        if (inputNumber && inputNumber.i) {
                            count = inputNumber.i;
                        }
                    } else if (input.typeListAttr) {
                        const inputTypeListAttr = node.attr[input.typeListAttr];
                        if (inputTypeListAttr && inputTypeListAttr.list && inputTypeListAttr.list.type) {
                            count = inputTypeListAttr.list.type.length;
                        }
                    }
                    const values = inputs.slice(inputIndex, inputIndex + count).map((input) => context.value(input.name, null, null));
                    const argument = new tf.Argument(input.name, values);
                    this._inputs.push(argument);
                    inputIndex += count;
                }
            }
            this._inputs.push(...inputs.slice(inputIndex).map((input, index) => {
                const name = input.label ? input.label : (inputIndex + index).toString();
                return new tf.Argument(name, [ context.value(input.name) ]);
            }));
            let outputIndex = 0;
            const outputs = node.output || [];
            if (this._type && this._type.outputs) {
                for (const output of this._type.outputs) {
                    let count = 1;
                    if (output.numberAttr) {
                        const outputNumber = node.attr[output.numberAttr];
                        if (outputNumber && outputNumber.i) {
                            count = outputNumber.i;
                        }
                    } else if (output.typeListAttr) {
                        const outputTypeListAttr = node.attr[output.typeListAttr];
                        if (outputTypeListAttr && outputTypeListAttr.list && outputTypeListAttr.list.type) {
                            count = outputTypeListAttr.list.type.length;
                        }
                    }
                    const values = outputs.slice(outputIndex, outputIndex + count).map((output) => {
                        return context.value(output.name ? output.name : '-', null, null);
                    });
                    const name = output.name ? output.name : 'output' + (this._outputs.length == 0 ? '' : this._outputs.length.toString());
                    const argument = new tf.Argument(name, values);
                    this._outputs.push(argument);
                    outputIndex += count;
                }
            }
            this._outputs.push(...outputs.slice(outputIndex).map((output, index) => {
                const name = (outputIndex + index).toString();
                const value = context.value(output.name ? output.name : '-', null, null);
                return new tf.Argument(name, [ value ]);
            }));
            const controlDependencies = node.controlDependencies || [];
            this._controlDependencies = controlDependencies.map((input) => context.value(input.name));
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

    constructor(metadata, op, name, value) {
        this._name = name;
        this._value = null;
        this._type = null;
        const schema = value && value.metadata ? value.metadata : metadata.attribute(op, name);
        const visible = metadata.visible(op, name);
        if (schema && schema.type) {
            this._type = schema.type;
        }
        switch (value.value) {
            case undefined:
                this._type = '';
                this._value = null;
                break;
            case 'type':
                this._type = 'type';
                this._value = tf.Utility.dataType(value.type);
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
            case 'tensor': {
                this._type = 'tensor';
                this._value = new tf.Tensor(value.tensor);
                break;
            }
            case 'func': {
                this._type = 'function';
                this._value = new tf.Node(metadata, { op: value.func.name, attr: value.func.attr }, null, new tf.Context());
                break;
            }
            case 'placeholder': {
                this._type = 'placeholder';
                this._value = value;
                break;
            }
            case 'list': {
                const list = value.list;
                if (list.s && list.s.length > 0) {
                    this._value = list.s.map((s) => tf.Utility.decodeText(s));
                } else if (list.i && list.i.length > 0) {
                    this._value = list.i;
                } else if (list.f && list.f.length > 0) {
                    this._value = list.f;
                } else if (list.type && list.type.length > 0) {
                    this._type = 'type[]';
                    this._value = list.type.map((type) => tf.Utility.dataType(type));
                } else if (list.shape && list.shape.length > 0) {
                    this._type = 'shape[]';
                    this._value = list.shape.map((shape) => new tf.TensorShape(shape));
                } else if (list.func && list.func.length > 0) {
                    this._type = 'function[]';
                    this._value = list.func.map((func) => new tf.Node(metadata, { op: func.name, attr: func.attr }));
                } else {
                    this._value = [];
                }
                break;
            }
            default: {
                throw new tf.Error("Unsupported attribute value type '" + JSON.stringify(value).substring(0, 32) + "'.");
            }
        }
        if (schema) {
            if (schema.visible === false) {
                this._visible = false;
            } else if (Object.prototype.hasOwnProperty.call(schema, 'default')) {
                const equals = (value, defaultValue) => {
                    if (!Array.isArray(defaultValue) && defaultValue === Object(defaultValue)) {
                        switch (defaultValue.type) {
                            case 'type':
                                defaultValue = tf.Utility.dataType(defaultValue.value);
                                break;
                            case 'shape':
                            case 'tensor':
                                defaultValue = defaultValue.value;
                                break;
                            default:
                                throw new tf.Error(JSON.stringify(defaultValue));
                        }
                    }
                    if (typeof value === 'boolean' || typeof value === 'number' || typeof value === 'string') {
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
                } else if (equals(value, defaultValue)) {
                    this._visible = false;
                }
            }
        }
        if (name == '_output_shapes') {
            this._visible = false;
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

    constructor(tensor, name, category) {
        this._name = name;
        this._category = category || null;
        if (tensor) {
            this._type = new tf.TensorType(tensor.dtype, tensor.tensor_shape || tensor.tensorShape);
            this._tensor = tensor;
            if (Object.prototype.hasOwnProperty.call(tensor, 'tensor_content')) {
                this._values = tensor.tensor_content;
                this._layout = '<';
            } else {
                const DataType = tf.proto.tensorflow.DataType;
                switch (tensor.dtype) {
                    case DataType.DT_INVALID: {
                        break;
                    }
                    case DataType.DT_BFLOAT16: {
                        const values = tensor.half_val || [];
                        this._values = new Uint8Array(values.length << 2);
                        const view = new DataView(this._values.buffer, this._values.byteOffset, this._values.byteLength);
                        for (let i = 0; i < values.length; i++) {
                            view.setUint32(i << 2, values[i] << 16, true);
                        }
                        this._layout = '<';
                        break;
                    }
                    case DataType.DT_HALF: {
                        const values = tensor.half_val || [];
                        this._values = new Uint8Array(values.length << 1);
                        const view = new DataView(this._values.buffer, this._values.byteOffset, this._values.byteLength);
                        for (let i = 0; i < values.length; i++) {
                            view.setUint16(i << 1, values[i], true);
                        }
                        this._layout = '<';
                        break;
                    }
                    case DataType.DT_FLOAT: {
                        this._values = tensor.float_val || null;
                        this._layout = '|';
                        break;
                    }
                    case DataType.DT_DOUBLE: {
                        this._values = tensor.double_val || null;
                        this._layout = '|';
                        break;
                    }
                    case DataType.DT_UINT8:
                    case DataType.DT_UINT16:
                    case DataType.DT_INT8:
                    case DataType.DT_INT16:
                    case DataType.DT_INT32: {
                        this._values = tensor.int_val || null;
                        this._layout = '|';
                        break;
                    }
                    case DataType.DT_UINT32: {
                        this._values = tensor.uint32_val || null;
                        this._layout = '|';
                        break;
                    }
                    case DataType.DT_INT64: {
                        this._values = tensor.int64_val || null;
                        this._layout = '|';
                        break;
                    }
                    case DataType.DT_UINT64: {
                        this._values = tensor.uint64_val || null;
                        this._layout = '|';
                        break;
                    }
                    case DataType.DT_BOOL: {
                        this._values = tensor.bool_val || null;
                        this._layout = '|';
                        break;
                    }
                    case DataType.DT_STRING: {
                        this._values = tensor.string_val || null;
                        this._layout = '|';
                        break;
                    }
                    case DataType.DT_COMPLEX64: {
                        this._layout = '|';
                        const values = tensor.scomplex_val || null;
                        this._values = new Array(values.length >> 1);
                        for (let i = 0; i < values.length; i += 2) {
                            this._values[i >> 1] = base.Complex64.create(values[i], values[i + 1]);
                        }
                        break;
                    }
                    case DataType.DT_COMPLEX128: {
                        this._layout = '|';
                        const values = tensor.dcomplex_val || null;
                        this._values = new Array(values.length >> 1);
                        for (let i = 0; i < values.length; i += 2) {
                            this._values[i >> 1] = base.Complex128.create(values[i], values[i + 1]);
                        }
                        break;
                    }
                    default: {
                        throw new tf.Error("Unsupported tensor data type '" + tensor.dtype + "'.");
                    }
                }
            }
        } else {
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

    get category() {
        return this._category;
    }

    get layout() {
        return this._layout;
    }

    get values() {
        let values = this._values;
        if (this._layout === '|' && Array.isArray(values)) {
            if (this._type.dataType === 'string') {
                values = values.map((value) => tf.Utility.decodeText(value));
            }
            const shape = (this._tensor.tensor_shape || this._tensor.tensorShape).dim.map((dim) => dim.size);
            const size = shape.reduce((a, b) => a * b, 1);
            if (values.length === 1 && size > 1) {
                values = new Array(size).fill(values[0]);
            }
        }
        return values;
    }
};

tf.TensorType = class {

    constructor(dtype, shape) {
        this._dtype = dtype;
        this._shape = new tf.TensorShape(shape);
    }

    get dataType() {
        return this._dtype ? tf.Utility.dataType(this._dtype) : '?';
    }

    get shape() {
        return this._shape;
    }

    equals(obj) {
        return obj && this.dataType === obj.dataType && this.shape.equals(obj.shape);
    }

    toString() {
        return this.dataType + this._shape.toString();
    }
};

tf.TensorShape = class {

    constructor(shape) {
        this._dimensions = null;
        if (shape) {
            if (shape.unknown_rank) {
                this._dimensions = null;
            } else if (Array.isArray(shape.dim)) {
                if (shape.dim.length == 0) {
                    this._dimensions = [];
                } else if (shape.dim.length == 1 && !shape.dim[0].size) {
                    this._dimensions = [ 0 ];
                } else {
                    this._dimensions =shape.dim.map((dim) => (dim.size && dim.size != -1) ? dim.size : '?');
                }
            }
        }
    }

    get dimensions() {
        return this._unknownRank ? null : this._dimensions;
    }

    equals(obj) {
        return (this.dimensions === null && obj.dimensions === null) || (Array.isArray(this.dimensions) && Array.isArray(obj.dimensions) && this.dimensions.length === obj.dimensions.length && this.dimensions.every((value, index) => obj.dimensions[index] === value));
    }

    toString() {
        if (this._dimensions === null) {
            return '[?]';
        }
        if (this._dimensions.length === 0) {
            return '';
        }
        return '[' + this._dimensions.map((dim) => (dim.size && dim.size != -1) ? dim.size.toString() : '?').join(',') + ']';
    }
};

tf.TensorBundle = class {

    static async open(stream, identifier, context) {
        const format = !identifier.toLowerCase().endsWith('.index') ? 1 : 2;
        const table = new tf.TensorBundle.Table(stream);
        if (!table.entries.has('')) {
            throw new tf.Error('Bundle header not available.');
        }
        if (format === 1) {
            return new tf.TensorBundle(format, table.entries, []);
        }
        const buffer = table.entries.get('');
        const reader = protobuf.BinaryReader.open(buffer);
        const header = tf.proto.tensorflow.BundleHeaderProto.decode(reader);
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
        try {
            const streams = await Promise.all(promises);
            return new tf.TensorBundle(format, table.entries, streams);
        } catch (error) {
            context.exception(error, false);
            return new tf.TensorBundle(format, table.entries, null);
        }
    }

    constructor(format, entries, streams) {
        this._format = format;
        this._tensors = [];
        switch (format) {
            case 1: {
                const buffer = entries.get('');
                const reader = protobuf.BinaryReader.open(buffer);
                const header = tf.proto.tensorflow.SavedTensorSlices.decode(reader);
                const data = new Map();
                for (const pair of entries) {
                    if (pair[0] !== '' && pair[0] !== 'global_step') {
                        const buffer = pair[1];
                        const reader = protobuf.BinaryReader.open(buffer);
                        const slices = tf.proto.tensorflow.SavedTensorSlices.decode(reader);
                        const name = slices.data.name;
                        const tensor = slices.data.data;
                        if (!data.has(name)) {
                            if (tensor.tensor_content && tensor.tensor_content.length > 0) {
                                data.set(name, { key: 'tensor_content', value: tensor.tensor_content });
                            } else {
                                const keys = Object.keys(tensor).filter((key) => key.endsWith('_val') && tensor[key] && tensor[key].length > 0);
                                data.set(name, keys.length == 1 ? { key: keys[0], value: tensor[keys[0]] } : null);
                            }
                        } else {
                            const item = data.get(name);
                            if (item !== null) {
                                if (tensor[item.key] && tensor[item.key].length > 0) {
                                    item.value = item.value.concat(tensor[item.key]);
                                } else {
                                    data.set(name, null);
                                }
                            }
                        }
                    }
                }
                for (const meta of header.meta.tensor) {
                    if (meta.name !== 'global_step') {
                        const tensor = new tf.proto.tensorflow.TensorProto();
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
                entries.forEach((buffer, name) => {
                    if (name !== '') {
                        const reader = protobuf.BinaryReader.open(buffer);
                        const entry = tf.proto.tensorflow.BundleEntryProto.decode(reader);
                        const tensor = new tf.proto.tensorflow.TensorProto();
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
            default: {
                throw new tf.Error("Unsupported Tensor Bundle format '" + format + "'.");
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
        const decoder = new TextDecoder();
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
                key = key + decoder.decode(reader.read(nonSharedSize));
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
        this._decoder = new TextDecoder('utf-8');
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

    string() {
        const size = this.uint32();
        const buffer = this.read(size);
        return this._decoder.decode(buffer);
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
            } else {
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
                default: {
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
        return new tf.EventFileReader(stream);
    }

    constructor(stream) {
        this._stream = stream;
    }

    read() {
        if (this._stream.position < this._stream.length) {
            const uint64 = (stream) => {
                const buffer = stream.read(8);
                const view = new DataView(buffer.buffer, buffer.byteOffset, buffer.byteLength);
                return view.getUint64(0, true).toNumber();
            };
            const length = uint64(this._stream);
            this._stream.skip(4); // masked crc of length
            const buffer = this._stream.read(length);
            const reader = protobuf.BinaryReader.open(buffer);
            const event = tf.proto.tensorflow.Event.decode(reader);
            this._stream.skip(4); // masked crc of data
            return event;
        }
        return null;
    }
};

tf.GraphMetadata = class {

    constructor(metadata, library) {
        this._metadata = metadata;
        this._functions = new Map();
        this._attributes = new Map();
        this._visibleCache = new Map();

        if (library && Array.isArray(library.function)) {
            for (const func of library.function) {
                const name = func.signature.name;
                if (this._functions.has(func.name)) {
                    throw new tf.Error("Duplicate function name '" + func.name + "'.");
                }
                this._functions.set(name, func);
            }
        }
    }

    type(name) {
        if (this._functions.has(name)) {
            const func = this._functions.get(name);
            if (func instanceof tf.Function) {
                return func;
            }
            this._functions.set(name, new tf.Function(this, func.signature.name, func));
            return this._functions.get(name);
        }
        const type = this._metadata.type(name);
        if (!type) {
            this._functions.set(name, new tf.Function(this, name, null));
            return this._functions.get(name);
        }
        return type;
    }

    attribute(type, name) {
        const key = type + '::' + name;
        if (!this._attributes.has(key)) {
            const schema = this.type(type);
            if (schema && schema.attributes) {
                for (const attribute of schema.attributes) {
                    const key = type + '::'  + attribute.name;
                    this._attributes.set(key, attribute);
                }
            }
        }
        return this._attributes.get(key);
    }

    visible(type, name) {
        if (!this._visibleCache.has(type)) {
            const set = new Set();
            const schema = this.type(type);
            if (schema && schema.inputs) {
                for (const input of schema.inputs) {
                    if (input.typeAttr) {
                        set.add(input.typeAttr);
                    } else if (input.typeListAttr) {
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
                    } else if (output.typeListAttr) {
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

tf.Context = class {

    constructor() {
        this._values = new Map();
        this.inputs = [];
        this.outputs = [];
        this.nodes = [];
    }

    value(name, type, tensor) {
        if (name.length === 0 && tensor) {
            return new tf.Value(name, type || null, tensor);
        }
        if (!this._values.has(name)) {
            this._values.set(name, new tf.Value(name, type || null, tensor || null));
        } else if ((type && !type.equals(this._values.get(name).type)) || tensor) {
            throw new tf.Error("Duplicate value '" + name + "'.");
        }
        return this._values.get(name);
    }

    graph(metadata, nodes, output_arg_map) {
        const namespaces = new Set();
        const node_map = new Map();
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
                const split = input.split(':', 3);
                const input_name = split[0];
                const input_index = split.length == 1 ? 0 : parseInt(split[split.length - 1]);
                const from_name = input_name.startsWith('^') ? input_name.substring(1) : input_name;
                const from = node_map.get(from_name);
                const output_name = input_index == 0 ? from_name : from_name + ':' + input_index.toString();
                const input_arg = from ? { name: output_name, from: from } : { name: output_name };
                if (input_name.startsWith('^')) {
                    node.controlDependencies.push(input_arg);
                } else {
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
        if (output_arg_map) {
            for (const node of nodes) {
                if (output_arg_map.has(node.name)) {
                    node.output.push({ name: node.name, to: [] });
                }
            }
        }
        const map_tensor = (name, node, kind) => {
            if (node && node.op === 'Const' && node.input.length === 0 && node.output.length === 1 && node.output[0].to.length === 1 && node.controlDependencies.length === 0) {
                const value = node.attr.value;
                if (value && Object.prototype.hasOwnProperty.call(value, 'tensor')) {
                    const tensor = new tf.Tensor(value.tensor, name, kind);
                    return this.value(name, tensor.type, tensor);
                }
            }
            return null;
        };
        const map_resource = (name, node, tensor) => {
            if (node && node.op === 'Placeholder' && node.input.length === 0 && node.output.length === 1 && node.controlDependencies.length === 0) {
                const dtype = node.attr.dtype.type;
                if (dtype === tf.proto.tensorflow.DataType.DT_RESOURCE) {
                    return this.value(name, null, tensor);
                }
            }
            return null;
        };
        for (const node of node_map.values()) {
            if (node.op === 'Identity' && node.input.length === 1 && node.output.length === 1 && node.output[0].to.length === 1 && node.controlDependencies.length === 0) {
                const initializer = map_tensor(node.name, node.input[0].from, 'Identity Constant');
                if (initializer) {
                    node_map.delete(initializer.name);
                    node_map.delete(node.input[0].name);
                }
                const identity = node.input[0].from;
                if (identity && identity.op === 'Identity' && identity.input.length === 1 && identity.output.length === 1 && node.output[0].to.length === 1 && node.controlDependencies.length === 0) {
                    const initializer = map_tensor(node.name, identity.input[0].from, 'Identity Constant');
                    if (initializer) {
                        node_map.delete(initializer.name);
                        node_map.delete(initializer.name);
                        node_map.delete(identity.name);
                        node_map.delete(node.name);
                    }
                }
            }
        }
        for (const node of node_map.values()) {
            const initializer = map_tensor(node.name, node, 'Const');
            if (initializer) {
                node_map.delete(node.name);
                node_map.delete(initializer.name);
            }
        }
        for (const node of node_map.values()) {
            if (node.op === 'ReadVariableOp' && node.input.length === 1 && node.output.length === 1 && node.output[0].to.length === 1 && node.controlDependencies.length === 0) {
                if (node.attr && node.attr.dtype && node.attr._output_shapes && node.attr._output_shapes.list && node.attr._output_shapes.list.shape) {
                    const tensor = new tf.proto.tensorflow.TensorProto();
                    tensor.dtype = node.attr.dtype.type;
                    tensor.tensor_shape = node.attr._output_shapes.list.shape[0];
                    const name = node.name;
                    const initializer = map_resource(name, node.input[0].from,  new tf.Tensor(tensor, name, 'Resource Variable'));
                    if (initializer) {
                        node_map.delete(initializer.name);
                        node_map.delete(node.input[0].name);
                    }
                }
            }
        }
        const input_map = new Map();
        for (const node of node_map.values()) {
            if (node.op == 'Placeholder' && node.input.length === 0 && node.output.length === 1 && node.controlDependencies.length === 0) {
                const dtype = node.attr.dtype;
                const shape = node.attr.shape;
                if (dtype && dtype.type && shape && shape.shape) {
                    const name = node.name;
                    const type = new tf.TensorType(dtype.type, shape.shape);
                    const value = this.value(name, type, null);
                    input_map.set(name, new tf.Argument(name, [ value ]));
                    node_map.delete(name);
                }
            }
        }
        const updateTorchScript = (node_map) => {
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
                    const match = /{\s*name\s*:\s*([A-Za-z0-9_]*)\s*}/.exec(value);
                    if (match) {
                        node.value = match[1].trim();
                    }
                }
                if (node.op === 'IO Node' && node.controlDependencies.length === 0) {
                    const shape = node.attr && node.attr._output_shapes && node.attr._output_shapes.list && node.attr._output_shapes.list.shape ? node.attr._output_shapes.list.shape[0] : null;
                    const type = shape ? new tf.TensorType('?', shape) : null;
                    if (node.input.length === 0 && node.output.length === 1) {
                        const argument = new tf.Argument(node.name, [ this.value(node.output[0].name, type, null) ]);
                        this.inputs.push(argument);
                        node_map.delete(node.name);
                    }
                    if (node.input.length === 1 && node.output.length === 0) {
                        const argument = new tf.Argument(node.name, [ this.value(node.input[0].name, type, null) ]);
                        this.outputs.push(argument);
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
                            this.value(input.name, null, tensor);
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
                    const match = (node, schema) => {
                        const args = schema.inputs || [];
                        const inputs = node.input || [];
                        if (inputs.length > args.length) {
                            return false;
                        }
                        for (let i = 0; i < inputs.length; i++) {
                            const input = inputs[i];
                            const arg = args[i];
                            switch (arg.type) {
                                case 'Tensor': {
                                    if ((input.constant === undefined && input.list === undefined) || input.constant === null) {
                                        continue;
                                    }
                                    break;
                                }
                                case 'int64': {
                                    if (input.constant !== undefined && Number.isInteger(parseInt(input.constant))) {
                                        continue;
                                    }
                                    break;
                                }
                                case 'float32': {
                                    if (input.constant !== undefined && !isNaN(parseFloat(input.constant))) {
                                        continue;
                                    }
                                    break;
                                }
                                case 'int64[]':
                                case 'int64[2]':
                                case 'SymInt[]':
                                case 'SymInt[2]': {
                                    if (Array.isArray(input.list)) {
                                        const list = input.list.map((item) => parseInt(item));
                                        if (list.every((value) => Number.isInteger(value))) {
                                            continue;
                                        }
                                    }
                                    break;
                                }
                                case 'boolean': {
                                    if (input.constant === 'false' ||
                                        input.constant === 'true' ||
                                        input.constant === '0' ||
                                        input.constant === '1') {
                                        continue;
                                    }
                                    break;
                                }
                                case 'Scalar': {
                                    if (input.constant !== undefined && Number.isInteger(parseInt(input.constant))) {
                                        continue;
                                    }
                                    break;
                                }
                                default: {
                                    break;
                                }
                            }
                            return false;
                        }
                        return true;
                    };
                    const schema = node.__metadata__.find((schema) => match(node, schema));
                    if (schema) {
                        const args = schema.inputs || [];
                        const inputs = node.input || [];
                        for (let i = 0; i < inputs.length; i++) {
                            const input = inputs[i];
                            delete input.metadata;
                            const arg = args[i];
                            switch (arg.type) {
                                case 'Tensor': {
                                    input.metadata = arg;
                                    break;
                                }
                                case 'int64': {
                                    const value = parseInt(input.constant);
                                    input.attr = new tf.proto.tensorflow.AttrValue();
                                    input.attr.i = value;
                                    input.attr.metadata = arg;
                                    break;
                                }
                                case 'float32': {
                                    const value = parseFloat(input.constant);
                                    input.attr = new tf.proto.tensorflow.AttrValue();
                                    input.attr.f = value;
                                    input.attr.metadata = arg;
                                    break;
                                }
                                case 'int64[]':
                                case 'int64[2]':
                                case 'SymInt[]':
                                case 'SymInt[2]': {
                                    const list = input.list.map((item) => parseInt(item));
                                    input.attr = new tf.proto.tensorflow.AttrValue();
                                    input.attr.list = new tf.proto.tensorflow.ListValue();
                                    input.attr.list.i = list;
                                    input.attr.metadata = arg;
                                    break;
                                }
                                case 'boolean': {
                                    input.attr = new tf.proto.tensorflow.AttrValue();
                                    input.attr.b = input.constant === 'true' || input.constant === '1';
                                    input.attr.metadata = arg;
                                    break;
                                }
                                case 'Scalar': {
                                    const value = parseInt(input.constant);
                                    input.attr = new tf.proto.tensorflow.AttrValue();
                                    input.attr.i = value;
                                    input.attr.metadata = arg;
                                    break;
                                }
                                default: {
                                    break;
                                }
                            }
                        }
                        node.metadata = Object.assign({}, schema);
                        node.metadata.name = node.op;
                    }
                }
                node.input = node.input.filter((input, index) => {
                    if (input.attr) {
                        const name = input.attr.metadata ? input.attr.metadata.name : index.toString();
                        node.attr[name] = input.attr;
                    } else if (input.constant !== undefined && input.constant !== null) {
                        const attr = new tf.proto.tensorflow.AttrValue();
                        attr.s = input.constant;
                        node.attr[index.toString()] = attr;
                    } else if (input.list !== undefined) {
                        const attr = new tf.proto.tensorflow.AttrValue();
                        attr.list = new tf.proto.tensorflow.ListValue();
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
        updateTorchScript(node_map);
        for (const input of input_map.values()) {
            this.inputs.push(input);
        }
        for (const node of node_map.values()) {
            this.nodes.push(new tf.Node(metadata, node, namespaces, this));
        }
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

    static dataType(type) {
        if (!tf.Utility._dataTypes) {
            const DataType = tf.proto.tensorflow.DataType;
            const dataTypes = new Map(Object.entries(DataType).map((entry) => {
                const key = entry[0].startsWith('DT_') ? entry[0].substring(3) : entry[0];
                return [ entry[1], key.toLowerCase() ];
            }));
            dataTypes.set(DataType.DT_HALF, 'float16');
            dataTypes.set(DataType.DT_FLOAT, 'float32');
            dataTypes.set(DataType.DT_DOUBLE, 'float64');
            dataTypes.set(DataType.DT_BOOL, 'boolean');
            tf.Utility._dataTypes = dataTypes;
        }
        return tf.Utility._dataTypes.has(type) ? tf.Utility._dataTypes.get(type) : '?';
    }

    static dataTypeKey(type) {
        if (!tf.Utility._dataTypeKeys) {
            tf.Utility.dataType(0);
            tf.Utility._dataTypeKeys = new Map(Array.from(tf.Utility._dataTypes).map((entry) => [ entry[1], entry[0] ]));
        }
        return tf.Utility._dataTypeKeys.get(type);
    }
};

tf.JsonReader = class {

    static decodeGraphDef(json) {
        const message = new tf.proto.tensorflow.GraphDef();
        message.node = json.node.map((node) => tf.JsonReader.decodeNodeDef(node));
        message.library = tf.JsonReader.decodeFunctionDefLibrary(json.library);
        if (message.versions) {
            message.versions = tf.JsonReader.decodeVersionDef(json.versions);
        }
        return message;
    }

    static decodeNodeDef(json) {
        const message = new tf.proto.tensorflow.NodeDef();
        message.name = json.name;
        message.op = json.op;
        message.input = json.input || [];
        if (json.device) {
            message.device = json.device;
        }
        message.attr = {};
        if (json.attr) {
            for (const entry of Object.entries(json.attr)) {
                message.attr[entry[0]] = tf.JsonReader.decodeAttrValue(entry[1]);
            }
        }
        return message;
    }

    static decodeAttrValue(json) {
        const message = new tf.proto.tensorflow.AttrValue();
        const keys = Object.keys(json);
        if (keys.length !== 1) {
            throw new tf.Error("Unsupported JSON tensorflow.AttrValue '" + JSON.stringify(keys) + "'.");
        }
        const key = keys[0];
        const value = json[key];
        switch (key) {
            case 'type':
                message.type = typeof value === 'number' ? value : tf.proto.tensorflow.DataType[value];
                break;
            case 'shape':
                message.shape = tf.JsonReader.decodeTensorShapeProto(value);
                break;
            case 'tensor':
                message.tensor = tf.JsonReader.decodeTensorProto(value);
                break;
            case 'b':
                message[key] = value;
                break;
            case 'f':
                message[key] = parseFloat(value);
                break;
            case 'i':
                message[key] = parseInt(value, 10);
                break;
            case 's':
                message[key] = typeof value === 'string' ? atob(value) : tf.Utility.decodeText(Uint8Array.from(value));
                break;
            case 'list':
                message.list = tf.JsonReader.decodeAttrValueListValue(json.list);
                break;
            case 'func':
                message[key]= value;
                break;
            default:
                throw new tf.Error("Unsupported JSON 'tensorflow.AttrValue." + key + "'.");
        }
        return message;
    }

    static decodeAttrValueListValue(json) {
        const message = new tf.proto.tensorflow.AttrValue.ListValue();
        const properties = Object.keys(json);
        if (properties.length > 0) {
            const keys = properties.filter((key) => Array.isArray(json[key]) && json[key].length > 0);
            if (keys.length !== 1) {
                throw new tf.Error("Unsupported JSON tensorflow.AttrValue.ListValue '" + JSON.stringify(keys) + "'.");
            }
            const key = keys[0];
            const list = json[key];
            switch (key) {
                case 'i':
                    message[key] = list.map((value) => parseInt(value, 10));
                    break;
                case 's':
                    message[key] = list.map((value) => typeof value === 'string' ? atob(value) : tf.Utility.decodeText(Uint8Array.from(value)));
                    break;
                case 'type':
                    message[key] = list.map((value) => tf.proto.tensorflow.DataType[value]);
                    break;
                case 'shape':
                    message[key] = list.map((shape) => tf.JsonReader.decodeTensorShapeProto(shape));
                    break;
                default:
                    throw new tf.Error("Unsupported JSON 'tensorflow.AttrValue.ListValue." + key + "'.");
            }
        }
        return message;
    }

    static decodeTensorProto(json) {
        const message = new tf.proto.tensorflow.TensorProto();
        message.dtype = tf.proto.tensorflow.DataType[json.dtype];
        message.tensor_shape = tf.JsonReader.decodeTensorShapeProto(json.tensorShape);
        return message;
    }

    static decodeTensorShapeProto(json) {
        const message = new tf.proto.tensorflow.TensorShapeProto();
        message.dim = (json.dim || []).map((json) => {
            const message = new tf.proto.tensorflow.TensorShapeProto.Dim();
            message.size = typeof json.size === 'string' ? parseInt(json.size, 10) : json.size;
            message.name = json.name;
            return message;
        });
        return message;
    }

    static decodeVersionDef(json) {
        const message = new tf.proto.tensorflow.VersionDef();
        message.producer = json.producer;
        message.min_consumer = json.min_consumer;
        message.bad_consumers = json.bad_consumers ? json.bad_consumers : [];
        return message;
    }

    static decodeFunctionDefLibrary(json) {
        const message = new tf.proto.tensorflow.FunctionDefLibrary();
        message.function = json ? (json.function || []).map((json) => tf.JsonReader.decodeFunctionDef(json)) : [];
        return message;
    }

    static decodeFunctionDef(json) {
        const message = new tf.proto.tensorflow.FunctionDef();
        message.signature = tf.JsonReader.decodeOpDef(json.signature);
        message.attr = {};
        if (json.attr) {
            for (const entry of Object.entries(json.attr)) {
                message.attr[entry[0]] = tf.JsonReader.decodeAttrValue(entry[1]);
            }
        }
        message.nodeDef = (json.nodeDef || []).map((json) => tf.JsonReader.decodeNodeDef(json));
        message.ret = json.ret;
        message.control_ret = json.control_ret;
        return message;
    }

    static decodeOpDef(json) {
        const message = new tf.proto.tensorflow.OpDef();
        message.name = json.name;
        message.input_arg = json.inputArg.map((json) => tf.JsonReader.decodeArgDef(json));
        message.output_arg = json.outputArg.map((json) => tf.JsonReader.decodeArgDef(json));
        return message;
    }

    static decodeArgDef(json) {
        const message = new tf.proto.tensorflow.OpDef.ArgDef();
        message.name = json.name;
        message.description = json.decscription;
        return message;
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