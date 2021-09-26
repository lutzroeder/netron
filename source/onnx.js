/* jshint esversion: 6 */

var onnx = onnx || {};
var protobuf = protobuf || require('./protobuf');
var flatbuffers = flatbuffers || require('./flatbuffers');

onnx.ModelFactory = class {

    match(context) {
        const identifier = context.identifier;
        const extension = identifier.split('.').pop().toLowerCase();
        if (identifier.endsWith('saved_model.pb') || identifier.endsWith('predict_net.pb') || identifier.endsWith('init_net.pb')) {
            return undefined;
        }
        if (identifier.endsWith('predict_net.pbtxt') || identifier.endsWith('predict_net.prototxt') ||
            identifier.endsWith('init_net.pbtxt') || identifier.endsWith('init_net.prototxt')) {
            return undefined;
        }
        let tags = context.tags('pb');
        if (tags.size > 0) {
            if (tags.size === 1 && tags.get(1) === 2) {
                const tags = context.tags('pb+');
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
                        }
                        else if (inner !== value) {
                            if (inner === 2 && !Array.isArray(value) && Object(value) === (value) && Object.keys(value).length === 0) {
                                return true;
                            }
                            return false;
                        }
                    }
                    return true;
                };
                // mediapipe.BoxDetectorIndex
                if (match(tags, [[1,[[1,[[1,[[1,5],[2,5],[3,5],[4,5],[6,0],[7,5],[8,5],[10,5],[11,0],[12,0]]],[2,5],[3,[]]]],[2,false],[3,false],[4,false],[5,false]]],[2,false],[3,false]] )) {
                    return undefined;
                }
                // third_party.tensorflow.python.keras.protobuf.SavedMetadata
                if (match(tags, [[1,[[1,[[1,0],[2,0]]],[2,0],[3,2],[4,2],[5,2]]]])) {
                    return undefined;
                }
            }
            if (Array.from(tags.keys()).every((tag) => tag <= 100) &&
                Array.from(tags.values()).every((type) => type < 5)) {
                // TensorProto
                if (tags.get(1) === 0 && tags.get(2) === 0 && tags.get(9) === 2) {
                    const schema = [[1,0],[2,0],[4,2],[5,2],[7,2],[8,2],[9,2]];
                    if (schema.every((pair) => !tags.has(pair[0]) || tags.get(pair[0]) === pair[1])) {
                        return 'onnx.pb.TensorProto';
                    }
                }
                // GraphProto
                if (tags.get(1) === 2) {
                    const schema = [[1,2],[2,2],[3,2],[4,2],[5,2],[6,0],[7,0],[8,2],[9,2],[10,2],[11,2],[12,2],[13,2],[14,2]];
                    if (schema.every((pair) => !tags.has(pair[0]) || tags.get(pair[0]) === pair[1])) {
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
                            const nameBuffer = decode(nodeBuffer, 4);
                            if (nameBuffer && nameBuffer.every((c) => c > 0x20 && c < 0x7f)) {
                                return 'onnx.pb.GraphProto';
                            }
                        }
                    }
                }
                // ModelProto
                if (tags.get(7) === 2) {
                    const schema = [[1,0],[2,2],[3,2],[4,2][5,0],[6,2],[7,2],[8,2],[14,2],[20,2]];
                    if (schema.every((pair) => !tags.has(pair[0]) || tags.get(pair[0]) === pair[1])) {
                        return 'onnx.pb.ModelProto';
                    }
                }
            }
        }
        const stream = context.stream;
        if (stream.length > 5) {
            const buffer = stream.peek(Math.min(stream.length, 32));
            if (buffer[0] === 0x08 && buffer[1] < 0x0A && buffer[2] === 0x12) {
                const producers = [
                    'backend-test', 'BrainwaveCompiler',
                    'CNTK',
                    'keras2onnx', 'Kneron', 'kneron_formatter', 'kneron_kl530_test_case',
                    'darknet to ONNX example',
                    'htshinichi',
                    'MATLAB Deep Learning Toolbox Converter for ONNX Model Format', 'ML.NET', 'MVTec Software',
                    'onnx-caffe2', 'onnx-example', 'onnx.quantize', 'onnx.utils.extract_model', 'OnnxMLTools', 'onnx_test', 'onnxruntime-tools', 'onnxruntime.transformers',
                    'PaddlePaddle', 'pytorch',
                    'sclblonnx', 'skl2onnx',
                    'Tencent YouTu', 'tf2onnx', 'tflite2onnx',
                    'WinMLTools'
                ];
                if (producers.some((producer) => Array.from(producer).every((ch, index) => index + 4 < buffer.length && ch.charCodeAt(0) === buffer[index + 4]))) {
                    return 'onnx.pb.ModelProto';
                }
            }
        }
        tags = context.tags('pbtxt');
        if (tags.has('ir_version')) {
            return 'onnx.pbtxt.ModelProto';
        }
        if (tags.has('graph') && extension !== 'model') {
            return 'onnx.pbtxt.ModelProto';
        }
        if (context.tags('flatbuffers').get('file_identifier') === 'ORTM') {
            return 'onnx.flatbuffers';
        }
        return undefined;
    }

    open(context, match) {
        const open = (model, format) => {
            return onnx.Metadata.open(context).then((metadata) => {
                return new onnx.Model(metadata, model, format);
            });
        };
        switch (match) {
            case 'onnx.pbtxt.ModelProto':
                return context.require('./onnx-proto').then(() => {
                    try {
                        onnx.proto = protobuf.get('onnx').onnx;
                        const stream = context.stream;
                        const reader = protobuf.TextReader.open(stream);
                        const model = onnx.proto.ModelProto.decodeText(reader);
                        const format = 'ONNX' + (model.ir_version ? ' v' + model.ir_version.toString() : '');
                        return open(model, format);
                    }
                    catch (error) {
                        const message = error && error.message ? error.message : error.toString();
                        throw new onnx.Error('File text format is not onnx.ModelProto (' + message.replace(/\.$/, '') + ').');
                    }
                });
            case 'onnx.pb.TensorProto':
                return context.require('./onnx-proto').then(() => {
                    // TensorProto
                    // input_0.pb, output_0.pb
                    try {
                        onnx.proto = protobuf.get('onnx').onnx;
                        const stream = context.stream;
                        const reader = protobuf.BinaryReader.open(stream);
                        const tensor = onnx.proto.TensorProto.decode(reader);
                        tensor.name = tensor.name || context.identifier;
                        const model = new onnx.proto.ModelProto();
                        model.graph = new onnx.proto.GraphProto();
                        model.graph.initializer = [ tensor ];
                        model.graph.value_info = [ new onnx.proto.ValueInfoProto() ];
                        model.graph.value_info[0].name = tensor.name;
                        model.graph.node = [ new onnx.proto.NodeProto() ];
                        model.graph.node[0].op_type = 'Constant';
                        model.graph.node[0].attribute = [ new onnx.proto.AttributeProto() ];
                        model.graph.node[0].attribute[0].name = 'value';
                        model.graph.node[0].attribute[0].t = tensor;
                        const format = 'ONNX Tensor';
                        return open(model, format);
                    }
                    catch (error) {
                        const message = error && error.message ? error.message : error.toString();
                        throw new onnx.Error('File format is not onnx.TensorProto (' + message.replace(/\.$/, '') + ').');
                    }
                });
            case 'onnx.pb.GraphProto':
                return context.require('./onnx-proto').then(() => {
                    // GraphProto
                    try {
                        onnx.proto = protobuf.get('onnx').onnx;
                        const stream = context.stream;
                        const reader = protobuf.BinaryReader.open(stream);
                        const model = new onnx.proto.ModelProto();
                        model.graph = onnx.proto.GraphProto.decode(reader);
                        const format = 'ONNX';
                        return open(model, format);
                    }
                    catch (error) {
                        const message = error && error.message ? error.message : error.toString();
                        throw new onnx.Error('File format is not onnx.GraphProto (' + message.replace(/\.$/, '') + ').');
                    }
                });
            case 'onnx.pb.ModelProto':
                return context.require('./onnx-proto').then(() => {
                    // ModelProto
                    try {
                        onnx.proto = protobuf.get('onnx').onnx;
                        const stream = context.stream;
                        const reader = protobuf.BinaryReader.open(stream);
                        const model = onnx.proto.ModelProto.decode(reader);
                        const format = 'ONNX' + (model.ir_version ? ' v' + model.ir_version.toString() : '');
                        return open(model, format);
                    }
                    catch (error) {
                        const message = error && error.message ? error.message : error.toString();
                        throw new onnx.Error('File format is not onnx.ModelProto (' + message.replace(/\.$/, '') + ').');
                    }
                });
            case 'onnx.flatbuffers': {
                return context.require('./ort-schema').then((/* schema */) => {
                    try {
                        onnx.schema = flatbuffers.get('ort').onnxruntime.experimental.fbs;
                        const stream = context.stream;
                        const reader = flatbuffers.BinaryReader.open(stream);
                        const session = onnx.schema.InferenceSession.create(reader);
                        const model = session.model;
                        const graph = model.graph;
                        graph.node = graph.nodes;
                        graph.doc_string = model.graph_doc_string;
                        graph.value_info = graph.node_args;
                        graph.input = graph.inputs.map((input) => {
                            return { name: input };
                        });
                        graph.output = graph.outputs.map((output) => {
                            return { name: output };
                        });
                        graph.initializer = graph.initializers.map((tensor) => {
                            tensor.data_location = onnx.DataLocation.DEFAULT;
                            return tensor;
                        });
                        graph.sparse_initializer = graph.sparse_initializers.map((tensor) => {
                            tensor.values.data_location = onnx.DataLocation.DEFAULT;
                            tensor.indices.data_location = onnx.DataLocation.DEFAULT;
                            return tensor;
                        });
                        delete graph.nodes;
                        delete graph.node_args;
                        delete graph.inputs;
                        delete graph.outputs;
                        delete graph.initializers;
                        delete graph.sparse_initializers;
                        delete model.graph_doc_string;
                        for (const node of graph.node) {
                            node.input = node.inputs;
                            node.output = node.outputs;
                            node.attribute = node.attributes;
                            delete node.inputs;
                            delete node.outputs;
                            delete node.attributes;
                        }
                        const format = 'ONNX Runtime' + (model.ir_version ? ' v' + model.ir_version.toString() : '');
                        return open(model, format);
                    }
                    catch (error) {
                        const message = error && error.message ? error.message : error.toString();
                        throw new onnx.Error('File format is not ort.Model (' + message.replace(/\.$/, '') + ').');
                    }
                });
            }
            default: {
                throw new onnx.Error("Unknown ONNX format '" + match + "'.");
            }
        }
    }
};

onnx.Model = class {

    constructor(metadata, model, format) {
        this._graphs = [];
        this._format = format;
        this._producerName = model.producer_name;
        this._producerVersion = model.producer_version;
        this._domain = model.domain;
        this._modelVersion = model.model_version;
        this._description = model.doc_string;
        this._metadata = [];
        this._imports = null;

        const imports = new Map();
        if (model.opset_import && model.opset_import.length > 0) {
            for (const opset_import of model.opset_import) {
                const domain = opset_import.domain || 'ai.onnx';
                const version = opset_import.version ? opset_import.version.toNumber() : 0;
                if (!imports.has(domain) || imports.get(domain) > version) {
                    imports.set(domain, version);
                }
            }
            this._imports = Array.from(imports).map((pair) => pair[0] + ' v' + pair[1].toString());
        }
        if (imports.size == 0) {
            imports.set('ai.onnx', 1);
            imports.set('ai.onnx.ml', 1);
        }

        let imageFormat = '';
        if (model.metadata_props) {
            const imageMetadata = {};
            for (const metadata_prop of model.metadata_props) {
                switch (metadata_prop.key) {
                    case 'author':
                        this._author = metadata_prop.value;
                        break;
                    case 'company':
                        this._company = metadata_prop.value;
                        break;
                    case 'converted_from':
                        this._converted_from = metadata_prop.value;
                        break;
                    case 'license':
                        this._license = metadata_prop.value;
                        break;
                    case 'license_url':
                        this._licenseUrl = metadata_prop.value;
                        break;
                    case 'Image.BitmapPixelFormat':
                    case 'Image.ColorSpaceGamma':
                    case 'Image.NominalPixelRange':
                        imageMetadata[metadata_prop.key] = metadata_prop.value;
                        break;
                    default:
                        this._metadata.push({ name: metadata_prop.key, value: metadata_prop.value});
                        break;
                }
            }
            imageFormat = [ imageMetadata['Image.BitmapPixelFormat'], imageMetadata['Image.ColorSpaceGamma'], imageMetadata['Image.NominalPixelRange'] ].filter((item) => item);
        }
        this._graphs = [];
        if (model && model.graph) {
            let key = 1000;
            const context = {};
            context.metadata = new onnx.GraphMetadata(metadata, imports);
            context.imageFormat = imageFormat;
            for (const func of model.functions || []) {
                context.metadata.add(new onnx.Function(context, func));
            }
            context.graphs = new Map();
            context.graph = function(graph) {
                graph.key = graph.key || (key++).toString();
                if (!this.graphs.has(graph.key)) {
                    this.graphs.set(graph.key, new onnx.Graph(this, graph));
                }
                return this.graphs.get(graph.key);
            };
            const graphs = [ model.graph ];
            while (graphs.length > 0) {
                const graph = graphs.shift();
                this._graphs.push(context.graph(graph));
                for (const node of graph.node || []) {
                    for (const attribute of node.attribute || []) {
                        if (attribute.g) {
                            graphs.push(attribute.g);
                        }
                        else if (attribute.graphs && attribute.graphs.length > 0) {
                            graphs.push(...attribute.graphs);
                        }
                    }
                }
            }
        }
    }

    get format() {
        return this._format;
    }

    get imports() {
        return this._imports;
    }

    get producer() {
        const producer = [];
        if (this._producerName) {
            producer.push(this._producerName);
        }
        if (this._producerVersion && this._producerVersion.length > 0) {
            producer.push(this._producerVersion);
        }
        if (producer.length > 0) {
            return producer.join(' ');
        }
        return null;
    }

    get domain() {
        return this._domain || null;
    }

    get description() {
        return this._description || null;
    }

    get author() {
        return this._author || null;
    }

    get company() {
        return this._company || null;
    }

    get source() {
        return this._converted_from || null;
    }

    get license() {
        const license = [];
        if (this._license && this._license.length > 0) {
            license.push(this._license);
        }
        if (this._licenseUrl && this._licenseUrl.length > 0) {
            license.push('<a href=\'' + this._licenseUrl + '\'>' + this._licenseUrl + '</a>');
        }
        if (license.length > 0) {
            return license;
        }
        return null;
    }

    get metadata() {
        return this._metadata;
    }

    get graphs() {
        return this._graphs;
    }
};

onnx.Graph = class {

    constructor(context, graph) {
        this._node = '';
        this._description = '';
        this._nodes = [];
        this._inputs = [];
        this._outputs = [];
        this._name = graph.name || null;
        this._description = graph.doc_string || '';

        const tensors = onnx.Utility.createTensors(graph.node);

        for (const initializer of graph.initializer) {
            const tensor = tensors.map(initializer.name);
            tensor.initializer = new onnx.Tensor(initializer, 'Initializer');
        }
        for (const sparse_initializer of graph.sparse_initializer) {
            const tensor = tensors.map(sparse_initializer.values.name);
            tensor.initializer = new onnx.Tensor(sparse_initializer, 'Sparse Initializer');
        }
        for (const tensor_annotation of graph.quantization_annotation || []) {
            const tensor = tensors.map(tensor_annotation.tensor_name);
            const annotation = {};
            for (const pair of tensor_annotation.quant_parameter_tensor_names) {
                annotation[pair.key] = pair.value;
            }
            tensor.annotation = annotation;
        }
        for (const valueInfo of graph.value_info) {
            const tensor = tensors.map(valueInfo.name);
            tensor.type = onnx.Utility.formatType(valueInfo.type, context.imageFormat);
            tensor.description = valueInfo.doc_string;
        }
        graph.input = graph.input.map((valueInfo) => {
            const tensor = tensors.map(valueInfo.name);
            tensor.type = onnx.Utility.formatType(valueInfo.type, context.imageFormat);
            tensor.description = valueInfo.doc_string;
            return tensor;
        });
        graph.output = graph.output.map((valueInfo) => {
            const tensor = tensors.map(valueInfo.name);
            tensor.type = onnx.Utility.formatType(valueInfo.type, context.imageFormat);
            tensor.description = valueInfo.doc_string;
            return tensor;
        });
        new onnx.Inference(graph.node, graph.output);
        const args = new Map();
        args.map = function(name) {
            if (!this.has(name)) {
                const tensor = tensors.map(name);
                const type = tensor.initializer ? tensor.initializer.type : tensor.type || null;
                this.set(name, new onnx.Argument(name, type, tensor.initializer, tensor.annotation, tensor.description));
            }
            return this.get(name);
        };
        this._nodes = onnx.Utility.createNodes(context, graph.node, graph.input, graph.output, tensors, args);
        for (const input of graph.input) {
            const argument = args.map(input.name);
            if (!argument.initializer) {
                this._inputs.push(new onnx.Parameter(input.name, [ argument ]));
            }
        }
        for (const output of graph.output) {
            const argument = args.map(output.name);
            if (!argument.initializer) {
                this._outputs.push(new onnx.Parameter(output.name, [ argument ]));
            }
        }
    }

    get name() {
        return this._name;
    }

    get description() {
        return this._description;
    }

    get groups() {
        return false;
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

    toString() {
        return 'graph(' + this.name + ')';
    }
};

onnx.Parameter = class {

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

onnx.Argument = class {

    constructor(name, type, initializer, annotation, description) {
        if (typeof name !== 'string') {
            throw new onnx.Error("Invalid argument identifier '" + JSON.stringify(name) + "'.");
        }
        this._name = name;
        this._type = type || null;
        this._initializer = initializer || null;
        this._annotation = annotation;
        this._description = description || '';
    }

    get name() {
        return this._name;
    }

    get type() {
        return this._type;
    }

    get description() {
        return this._description;
    }

    get quantization() {
        if (this._annotation) {
            return Object.keys(this._annotation).map((key) => key + ': ' + this._annotation[key]).join(', ');
        }
        return null;
    }

    get initializer() {
        return this._initializer;
    }
};

onnx.Node = class {

    constructor(context, op_type, domain, name, description, attributes, inputs, outputs) {
        this._type = context.metadata.type(op_type, domain) || { name: op_type, module: domain };
        if (this.type.module !== domain && !(this._type instanceof onnx.Function)) {
            this._type = Object.assign({}, this.type);
            this._type.name = op_type;
            this._type.module = domain;
        }
        this._name = name || '';
        this._description = description || '';
        this._inputs = inputs;
        this._outputs = outputs;
        this._attributes = (attributes || []).map((attribute) => new onnx.Attribute(context, op_type, domain, attribute));
    }

    get type() {
        return this._type;
    }

    get name() {
        return this._name;
    }

    get description() {
        return this._description;
    }

    get group() {
        return null;
    }

    get attributes() {
        return this._attributes;
    }

    get inputs() {
        return this._inputs;
    }

    get outputs() {
        return this._outputs;
    }
};

onnx.Attribute = class {

    constructor(context, op_type, domain, attribute) {
        this._name = attribute.name;
        this._description = attribute.doc_string || '';
        this._type = null;
        this._value = null;
        switch (attribute.type) {
            case onnx.AttributeType.FLOAT:
                this._value = attribute.f;
                this._type = 'float32';
                break;
            case onnx.AttributeType.INT:
                this._value = attribute.i;
                this._type = 'int64';
                break;
            case onnx.AttributeType.STRING:
                switch (op_type) {
                    case 'Int8GivenTensorFill':
                        this._value = Array.from(attribute.s);
                        break;
                    default:
                        this._value = onnx.Utility.decodeText(attribute.s);
                        break;
                }
                this._type = 'string';
                break;
            case onnx.AttributeType.TENSOR:
                this._value = new onnx.Tensor(attribute.t);
                this._type = 'tensor';
                break;
            case onnx.AttributeType.GRAPH:
                this._value = context.graph(attribute.g);
                this._type = 'graph';
                break;
            case onnx.AttributeType.FLOATS:
                this._value = attribute.floats;
                this._type = 'float32[]';
                break;
            case onnx.AttributeType.INTS:
                this._value = attribute.ints;
                this._type = 'int64[]';
                break;
            case onnx.AttributeType.STRINGS:
                this._value = attribute.strings.map((s) => onnx.Utility.decodeText(s));
                this._type = 'string[]';
                break;
            case onnx.AttributeType.TENSORS:
                this._value = attribute.tensors.map((tensor) => new onnx.Tensor(tensor));
                this._type = 'tensor[]';
                break;
            case onnx.AttributeType.GRAPHS:
                this._value = attribute.graphs.map((graph) => context.graph(graph));
                this._type = 'graph[]';
                break;
            case onnx.AttributeType.SPARSE_TENSOR:
                this._value = new onnx.Tensor(attribute.sparse_tensor);
                this._type = 'tensor';
                break;
            case onnx.AttributeType.SPARSE_TENSORS:
                this._value = attribute.sparse_tensors.map((tensor) => new onnx.Tensor(tensor));
                this._type = 'tensor[]';
                break;
            case onnx.AttributeType.TYPE_PROTO:
                this._value = onnx.Utility.formatType(attribute.tp, context.imageFormat);
                this._type = 'type';
                break;
            case onnx.AttributeType.TYPE_PROTOS:
                this._value = attribute.type_protos.map((type) => onnx.Utility.formatType(type, context.imageFormat));
                this._type = 'type[]';
                break;
            default:
                throw new onnx.Error("Unknown attribute type '" + attribute.type + "'.");
        }

        const metadata = context.metadata.attribute(op_type, domain, attribute.name);
        if (metadata) {
            if (Object.prototype.hasOwnProperty.call(metadata, 'default') && this._value == metadata.default) {
                this._visible = false;
            }
            if (metadata.type === 'DataType') {
                this._type = metadata.type;
                const value = this._value ? parseInt(this._value.toString(), 10) : this._value;
                this._value = Number.isInteger(value) ? onnx.Utility.formatDataType(value) : value;
            }
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

    get description() {
        return this._description;
    }

    get visible() {
        return this._visible == false ? false : true;
    }
};

onnx.Tensor = class {

    constructor(tensor, kind) {
        this._kind = kind || null;
        const data = (tensor) => {
            let data = undefined;
            if (tensor.data_location === onnx.DataLocation.DEFAULT) {
                switch (tensor.data_type) {
                    case onnx.DataType.FLOAT16:
                        if (tensor.int32_data && tensor.int32_data.length > 0) {
                            const buffer = new Uint8Array(tensor.int32_data.length << 1);
                            const view = new DataView(buffer.buffer, buffer.byteOffset, buffer.byteLength);
                            const array = tensor.int32_data;
                            for (let i = 0; i < array.length; i++) {
                                view.setUint16(i << 1, array[i], true);
                            }
                            data = {
                                type: tensor.data_type,
                                buffer: buffer
                            };
                        }
                        break;
                    case onnx.DataType.FLOAT:
                        data = new Float32Array(tensor.float_data);
                        break;
                    case onnx.DataType.DOUBLE:
                        data = new Float64Array(tensor.double_data);
                        break;
                    case onnx.DataType.BOOL:
                        data = new Array(tensor.int32_data.size);
                        for (let i = 0; i < data.length; i++) {
                            data[i] = data[i] === 0 ? false : true;
                        }
                        break;
                    case onnx.DataType.INT8:
                        data = new Int8Array(tensor.int32_data);
                        break;
                    case onnx.DataType.UINT8:
                        data = new Uint8Array(tensor.int32_data);
                        break;
                    case onnx.DataType.INT16:
                        data = new Int32Array(tensor.int32_data);
                        break;
                    case onnx.DataType.UINT16:
                        data = new Int32Array(tensor.int32_data);
                        break;
                    case onnx.DataType.INT32:
                        data = new Int32Array(tensor.int32_data);
                        break;
                    case onnx.DataType.UINT32:
                    case onnx.DataType.UINT64:
                        data = tensor.uint64_data;
                        break;
                    case onnx.DataType.INT64:
                        data = tensor.int64_data;
                        break;
                }
                if (data && (Array.isArray(data) || ArrayBuffer.isView(data)) && data.length === 0) {
                    data = undefined;
                }
                if (!data && tensor.raw_data && tensor.raw_data.length > 0) {
                    data = {
                        type: tensor.data_type,
                        buffer: tensor.raw_data
                    };
                }
            }
            return data;
        };
        const location = (tensor) => {
            return onnx.Utility.formatLocation(tensor.data_location);
        };
        if ((onnx.proto && tensor instanceof onnx.proto.SparseTensorProto) ||
            (onnx.schema && tensor instanceof onnx.schema.SparseTensor)) {
            this._name = tensor.values.name || '';
            this._type = new onnx.TensorType(tensor.values.data_type, new onnx.TensorShape(tensor.dims.map((dim) => dim)), null);
            this._location = Array.from(new Set([ location(tensor.values), location(tensor.indices) ])).join(':');
            this._values = data(tensor.values);
            this._indices = data(tensor.indices);
        }
        else {
            this._name = tensor.name || '';
            this._type = new onnx.TensorType(tensor.data_type, new onnx.TensorShape(tensor.dims.map((dim) => dim)), null);
            this._location = location(tensor);
            this._values = data(tensor);
        }
    }

    get name() {
        return this._name;
    }

    get kind() {
        return this._kind;
    }

    get type() {
        return this._type;
    }

    get state() {
        return this._context().state || null;
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
        return onnx.Tensor._stringify(value, '', '    ');
    }

    _context() {
        const context = {};
        context.state = null;
        if (this._sparse) {
            context.state = 'Sparse data not implemented.';
            return context;
        }
        if (this._location !== 'default') {
            context.state = "Data '" + this._location + "' location not implemented.";
            return context;
        }
        const decode = (data) => {
            if (!data || Array.isArray(data) || ArrayBuffer.isView(data)) {
                return data;
            }
            const buffer = data.buffer;
            const view = new DataView(buffer.buffer, buffer.byteOffset, buffer.byteLength);
            const type = data.type;
            data = undefined;
            switch (type) {
                case onnx.DataType.BOOL:
                    data = new Array(buffer.length);
                    for (let i = 0; i < buffer.length; i++) {
                        data[i] = view.getUint8(i) === 0 ? false : true;
                    }
                    break;
                case onnx.DataType.FLOAT16:
                    data = new Float32Array(buffer.length >> 1);
                    for (let i = 0; i < data.length; i++) {
                        data[i] = view.getFloat16(i << 1, true);
                    }
                    break;
                case onnx.DataType.FLOAT:
                    data = new Float32Array(buffer.length >> 2);
                    for (let i = 0; i < data.length; i++) {
                        data[i] = view.getFloat32(i << 2, true);
                    }
                    break;
                case onnx.DataType.DOUBLE:
                    data = new Float64Array(buffer.length >> 3);
                    for (let i = 0; i < data.length; i++) {
                        data[i] = view.getFloat64(i << 3, true);
                    }
                    break;
                case onnx.DataType.INT8:
                    data = new Int8Array(buffer.length);
                    for (let i = 0; i < data.length; i++) {
                        data[i] = view.getInt8(i, true);
                    }
                    break;
                case onnx.DataType.UINT8:
                    data = new Uint8Array(buffer.length);
                    for (let i = 0; i < data.length; i++) {
                        data[i] = view.getUint8(i, true);
                    }
                    break;
                case onnx.DataType.INT16:
                    data = new Int16Array(buffer.length >> 1);
                    for (let i = 0; i < data.length; i++) {
                        data[i] = view.getInt16(i << 1, true);
                    }
                    break;
                case onnx.DataType.UINT16:
                    data = new Uint16Array(buffer.length >> 1);
                    for (let i = 0; i < data.length; i++) {
                        data[i] = view.getUint16(i << 1, true);
                    }
                    break;
                case onnx.DataType.INT32:
                    data = new Int32Array(buffer.length >> 2);
                    for (let i = 0; i < data.length; i++) {
                        data[i] = view.getInt32(i << 2, true);
                    }
                    break;
                case onnx.DataType.UINT32:
                    data = new Uint32Array(buffer.length >> 2);
                    for (let i = 0; i < data.length; i++) {
                        data[i] = view.getUint32(i << 2, true);
                    }
                    break;
                case onnx.DataType.INT64:
                    data = new Array(buffer.length >> 3);
                    for (let i = 0; i < data.length; i++) {
                        data[i] = view.getInt64(i << 3, true);
                    }
                    break;
                case onnx.DataType.UINT64:
                    data = new Array(buffer.length >> 3);
                    for (let i = 0; i < data.length; i++) {
                        data[i] = view.getUint64(i << 3, true);
                    }
                    break;
            }
            return data;
        };
        this._values = decode(this._values);
        if (!this._values) {
            context.state = 'Tensor data is empty.';
            return context;
        }
        this._indices = decode(this._indices);
        context.values = this._values;
        context.indices = this._indices;
        context.index = 0;
        context.dataType = this.type.dataType;
        context.shape = this.type.shape.dimensions;
        context.data = function() {
            if (!this._data) {
                if (this.indices && this.values && this.indices.length === this.values.length) {
                    const size = context.shape.reduce((a, b) => a * b, 1);
                    const indices = this.indices;
                    const values = this.values;
                    const array = new values.constructor(size);
                    switch (this.dataType) {
                        case 'boolean':
                            array.fill(false);
                            break;
                        case 'int64':
                        case 'uint64':
                            break;
                    }
                    if (indices.length > 0) {
                        if (Object.prototype.hasOwnProperty.call(indices[0], 'low')) {
                            for (let i = 0; i < indices.length; i++) {
                                const index = indices[i];
                                array[index.high === 0 ? index.low : index.toNumber()] = values[i];
                            }
                        }
                        else {
                            for (let i = 0; i < indices.length; i++) {
                                array[indices[i]] = values[i];
                            }
                        }
                    }
                    this._data = array;
                }
                else {
                    this._data = this.values;
                }
            }
            return this._data;
        };
        return context;
    }

    _decode(context, dimension) {
        const shape = context.shape.length !== 0 ? context.shape : [ 1 ];
        const results = [];
        const size = shape[dimension];
        const data = context.data();
        if (dimension == shape.length - 1) {
            for (let i = 0; i < size; i++) {
                if (context.index > context.limit) {
                    results.push('...');
                    return results;
                }
                results.push(data[context.index++]);
            }
        }
        else {
            for (let j = 0; j < size; j++) {
                if (context.index > context.limit) {
                    results.push('...');
                    return results;
                }
                results.push(this._decode(context, dimension + 1));
            }
        }
        if (context.shape.length == 0) {
            return results[0];
        }
        return results;
    }

    static _stringify(value, indentation, indent) {
        if (Array.isArray(value)) {
            const result = [];
            result.push(indentation + '[');
            const items = value.map((item) => onnx.Tensor._stringify(item, indentation + indent, indent));
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

onnx.TensorType = class {

    constructor(dataType, shape, denotation) {
        this._dataType = onnx.Utility.formatDataType(dataType);
        this._shape = shape;
        this._denotation = denotation || null;
    }

    get dataType() {
        return this._dataType;
    }

    get shape() {
        return this._shape;
    }

    get denotation() {
        return this._denotation;
    }

    toString() {
        return this.dataType + this._shape.toString();
    }
};

onnx.TensorShape = class {

    constructor(dimensions) {
        this._dimensions = dimensions;
    }

    get dimensions() {
        return this._dimensions;
    }

    toString() {
        if (!this._dimensions || this._dimensions.length == 0) {
            return '';
        }
        return '[' + this._dimensions.join(',') + ']';
    }
};

onnx.SequenceType = class {

    constructor(elementType, denotation) {
        this._elementType = elementType;
        this._denotation = denotation;
    }

    get elementType() {
        return this._elementType;
    }

    get dennotation() {
        return this._dennotation;
    }

    toString() {
        return 'sequence<' + this._elementType.toString() + '>';
    }
};

onnx.MapType = class {

    constructor(keyType, valueType, denotation) {
        this._keyType = onnx.Utility.formatDataType(keyType);
        this._valueType = valueType;
        this._denotation = denotation;
    }

    get keyType() {
        return this._keyType;
    }

    get valueType() {
        return this._valueType;
    }

    get denotation() {
        return this._denotation;
    }

    toString() {
        return 'map<' + this._keyType + ',' + this._valueType.toString() + '>';
    }
};

onnx.OpaqueType = class {

    constructor(domain, name) {
        this._domain = domain;
        this._name = name;
    }

    toString() {
        const name = (this._domain ? (this._domain + '.') : '') + this._name;
        return 'opaque<' + name + '>';
    }
};

onnx.Function = class {

    constructor(context, func) {
        this._name = func.name;
        this._domain = func.domain;
        this._description = func.doc_string;
        this._inputs = [];
        this._outputs = [];
        this._attributes = func.attribute.map((attribtue) => { return { name: attribtue }; });
        const tensors = onnx.Utility.createTensors(func.node);
        func.input = func.input.map((input) => tensors.map(input));
        func.output = func.output.map((output) => tensors.map(output));
        const args = new Map();
        args.map = function(name) {
            if (!this.has(name)) {
                const tensor = tensors.map(name);
                const type = tensor.initializer ? tensor.initializer.type : tensor.type || null;
                this.set(name, new onnx.Argument(name, type, tensor.initializer, tensor.annotation, tensor.description));
            }
            return this.get(name);
        };
        this._nodes = onnx.Utility.createNodes(context, func.node, func.input, func.output, tensors, args);
        for (const input of func.input) {
            const argument = args.map(input.name);
            if (!argument.initializer) {
                this._inputs.push(new onnx.Parameter(input.name, [ argument ]));
            }
        }
        for (const output of func.output) {
            const argument = args.map(output.name);
            if (!argument.initializer) {
                this._outputs.push(new onnx.Parameter(output.name, [ argument ]));
            }
        }
    }

    get type() {
        return 'function';
    }

    get name() {
        return this._name;
    }

    get module() {
        return this._domain;
    }

    get description() {
        return this._description;
    }

    get inputs() {
        return this._inputs;
    }

    get outputs() {
        return this._outputs;
    }

    get attributes() {
        return this._attributes;
    }

    get nodes() {
        return this._nodes;
    }
};

onnx.GraphMetadata = class {

    constructor(metadata, imports) {
        this._metadata = metadata;
        this._imports = imports;
        this._cache = new Map();
        this._attributes = new Map();
        this._functions = new Map();
    }

    add(func) {
        if (!this._functions.has(func.module)) {
            this._functions.set(func.module, new Map());
        }
        const map = this._functions.get(func.module);
        if (map.has(func.name)) {
            throw new onnx.Error("Duplicate function identifier '" + func.module + '.' + func.name + "'.");
        }
        map.set(func.name, func);
    }

    type(name, domain) {
        domain = domain || 'ai.onnx';
        const key = domain + ':' + name;
        if (!this._cache.has(key)) {
            let value = this._metadata.type(name, domain, this._imports);
            if (!value) {
                if (this._functions.has(domain)) {
                    const map = this._functions.get(domain);
                    if (map.has(name)) {
                        value = map.get(name);
                    }
                }
            }
            this._cache.set(key, value);
        }
        return this._cache.get(key);
    }

    attribute(type, domain, name) {
        const key = domain + ':' + type + ':' + name;
        if (!this._attributes.has(key)) {
            const schema = this.type(type, domain);
            if (schema && schema.attributes && schema.attributes.length > 0) {
                for (const attribute of schema.attributes) {
                    this._attributes.set(key, attribute);
                }
            }
            if (!this._attributes.has(key)) {
                this._attributes.set(key, null);
            }
        }
        return this._attributes.get(key);
    }
};

onnx.Metadata = class {

    static open(context) {
        if (onnx.Metadata._metadata) {
            return Promise.resolve(onnx.Metadata._metadata);
        }
        return context.request('onnx-metadata.json', 'utf-8', null).then((data) => {
            onnx.Metadata._metadata = new onnx.Metadata(data);
            return onnx.Metadata._metadata;
        }).catch(() => {
            onnx.Metadata._metadata = new onnx.Metadata(null);
            return onnx.Metadata._metadata;
        });
    }

    constructor(data) {
        this._map = new Map();
        if (data) {
            const metadata = JSON.parse(data);
            for (const item of metadata) {
                if (!this._map.has(item.module)) {
                    this._map.set(item.module, new Map());
                }
                const map = this._map.get(item.module);
                if (!map.has(item.name)) {
                    map.set(item.name, []);
                }
                map.get(item.name).push(item);
            }
        }
    }

    type(name, domain, imports) {
        domain = domain || 'ai.onnx';
        let current = null;
        if (this._map.has(domain)) {
            const map = this._map.get(domain);
            if (map.has(name)) {
                for (const metadata of map.get(name)) {
                    const matchVersion = current ? current.version : -1;
                    const importVersion = imports.get(metadata.module) || 0;
                    if (importVersion >= metadata.version && matchVersion < metadata.version) {
                        current = metadata;
                    }
                }
            }
        }
        return current;
    }
};

onnx.Inference = class {

    constructor(nodes, outputs) {
        this._outputs = new Map();

        for (const node of nodes) {
            for (const output of node.output) {
                this._outputs.set(output.name, node);
            }
        }

        for (const output of outputs) {
            this._infer(output.name);
        }
    }

    _infer(output) {
        if (this._outputs.has(output)) {
            let hasInputShapes = true;
            const node = this._outputs.get(output);
            for (const input of node.input) {
                if (!input.type) {
                    this._infer(input);
                    if (!input.type) {
                        hasInputShapes = false;
                        break;
                    }
                }
            }
            if (hasInputShapes) {
                // continue
            }
        }
    }
};

onnx.DataLocation = {
    DEFAULT: 0,
    EXTERNAL: 1
};

onnx.DataType = {
    UNDEFINED: 0,
    FLOAT: 1,
    UINT8: 2,
    INT8: 3,
    UINT16: 4,
    INT16: 5,
    INT32: 6,
    INT64: 7,
    STRING: 8,
    BOOL: 9,
    FLOAT16: 10,
    DOUBLE: 11,
    UINT32: 12,
    UINT64: 13,
    COMPLEX64: 14,
    COMPLEX128: 15,
    BFLOAT16: 16
};

onnx.AttributeType = {
    UNDEFINED: 0,
    FLOAT: 1,
    INT: 2,
    STRING: 3,
    TENSOR: 4,
    GRAPH: 5,
    FLOATS: 6,
    INTS: 7,
    STRINGS: 8,
    TENSORS: 9,
    GRAPHS: 10,
    SPARSE_TENSOR: 11,
    SPARSE_TENSORS: 12,
    TYPE_PROTO: 13,
    TYPE_PROTOS: 14
};

onnx.Utility = class {

    static decodeText(value) {
        if (typeof value === 'string') {
            return value;
        }
        onnx.Utility._utf8Decoder = onnx.Utility._utf8Decoder || new TextDecoder('utf-8');
        return onnx.Utility._utf8Decoder.decode(value);
    }

    static formatDataType(value) {
        if (!onnx.Utility._dataTypes) {
            onnx.Utility._dataTypes = new Map([
                [ onnx.DataType.UNDEFINED, 'UNDEFINED' ],
                [ onnx.DataType.FLOAT, 'float32' ],
                [ onnx.DataType.UINT8, 'uint8' ],
                [ onnx.DataType.INT8, 'int8' ],
                [ onnx.DataType.UINT16, 'uint16' ],
                [ onnx.DataType.INT16, 'int16' ],
                [ onnx.DataType.INT32, 'int32' ],
                [ onnx.DataType.INT64, 'int64' ],
                [ onnx.DataType.STRING, 'string' ],
                [ onnx.DataType.BOOL, 'boolean' ],
                [ onnx.DataType.FLOAT16, 'float16' ],
                [ onnx.DataType.DOUBLE, 'float64' ],
                [ onnx.DataType.UINT32, 'uint32' ],
                [ onnx.DataType.UINT64, 'uint64' ],
                [ onnx.DataType.COMPLEX64, 'complex64' ],
                [ onnx.DataType.COMPLEX128, 'complex128' ],
                [ onnx.DataType.BFLOAT16, 'bfloat16' ]
            ]);
        }
        if (onnx.Utility._dataTypes.has(value)) {
            return onnx.Utility._dataTypes.get(value);
        }
        return onnx.Utility._dataTypes.get(onnx.DataType.UNDEFINED);
    }

    static formatType(type, imageFormat) {
        if (!type) {
            return null;
        }
        let denotation = '';
        switch (type.denotation) {
            case 'TENSOR':
                denotation = 'Tensor';
                break;
            case 'IMAGE':
                denotation = 'Image' + (imageFormat ? '(' + imageFormat.join(',') + ')' : '');
                break;
            case 'AUDIO':
                denotation = 'Audio';
                break;
            case 'TEXT':
                denotation = 'Text';
                break;
        }
        switch (type.value) {
            case 'tensor_type': {
                const tensor_type = type.tensor_type;
                let shape = [];
                if (tensor_type.shape && tensor_type.shape.dim) {
                    shape = tensor_type.shape.dim.map((dim) => dim.dim_param ? dim.dim_param : dim.dim_value);
                }
                return new onnx.TensorType(tensor_type.elem_type, new onnx.TensorShape(shape), denotation);
            }
            case 'sparse_tensor_type': {
                const tensor_type = type.sparse_tensor_type;
                let shape = [];
                if (tensor_type.shape && tensor_type.shape.dim) {
                    shape = tensor_type.shape.dim.map((dim) => dim.dim_param ? dim.dim_param : dim.dim_value);
                }
                return new onnx.TensorType(tensor_type.elem_type, new onnx.TensorShape(shape), denotation);
            }
            case 'map_type': {
                return new onnx.MapType(type.map_type.key_type, onnx.Utility.formatType(type.map_type.value_type, imageFormat), denotation);
            }
            case 'sequence_type': {
                return new onnx.SequenceType(onnx.Utility.formatType(type.sequence_type.elem_type, imageFormat), denotation);
            }
            case 'opaque_type': {
                return new onnx.OpaqueType(type.opaque_type.domain, type.opaque_type.name);
            }
        }
        return null;
    }

    static formatLocation(location) {
        if (!onnx.Utility._dataLocations) {
            onnx.Utility._dataLocations = new Map(Object.keys(onnx.DataLocation).map((key) => [ onnx.DataLocation[key], key.toLowerCase() ]));
        }
        return onnx.Utility._dataLocations.get(location);
    }

    static attributeType(attribute) {
        if (attribute.type) {
            return attribute.type;
        }
        if (attribute.ints && attribute.ints.length > 0) {
            return onnx.AttributeType.INTS;
        }
        else if (attribute.floats && attribute.floats.length > 0) {
            return onnx.AttributeType.FLOATS;
        }
        else if (attribute.strings && attribute.strings.length > 0) {
            return onnx.AttributeType.STRINGS;
        }
        else if (attribute.graphs && attribute.graphs.length > 0) {
            return onnx.AttributeType.GRAPHS;
        }
        else if (attribute.s && attribute.s.length > 0) {
            return onnx.AttributeType.STRING;
        }
        else if (Object.prototype.hasOwnProperty.call(attribute, 'f')) {
            return onnx.AttributeType.FLOAT;
        }
        else if (Object.prototype.hasOwnProperty.call(attribute, 'i')) {
            return onnx.AttributeType.INT;
        }
        else if (Object.prototype.hasOwnProperty.call(attribute, 't')) {
            return onnx.AttributeType.TENSOR;
        }
        else if (Object.prototype.hasOwnProperty.call(attribute, 'g')) {
            return onnx.AttributeType.GRAPH;
        }
        else if (Object.prototype.hasOwnProperty.call(attribute, 'sparse_tensor')) {
            return onnx.AttributeType.SPARSE_TENSOR;
        }
        return onnx.AttributeType.UNDEFINED;
    }

    static createTensors(nodes) {
        const tensors = new Map();
        tensors.map = function(name) {
            if (!this.has(name)) {
                this.set(name, { name: name });
            }
            return this.get(name);
        };
        for (const node of nodes) {
            node.input = node.input.map((name) => tensors.map(name));
            node.output = node.output.map((name) => tensors.map(name));
            node.param = {};
            for (const attribute of node.attribute) {
                attribute.type = onnx.Utility.attributeType(attribute);
            }
        }
        return tensors;
    }

    static createNodes(context, nodes, inputs, outputs, tensors, args) {
        const inputMap = new Map();
        const outputMap = new Map();
        for (const node of nodes) {
            node.input.every((input) => inputMap.set(input.name, (inputMap.get(input) || 0) + 1));
            node.output.every((output) => outputMap.set(output.name, (outputMap.get(output) || 0) + 1));
        }
        inputs.every((input) => inputMap.delete(input.name));
        outputs.every((output) => outputMap.delete(output.name));
        nodes = nodes.filter((node) => {
            const constant = node &&
                node.op_type === 'Constant' &&
                node.attribute.length === 1 && node.attribute[0] &&
                node.input.length === 0 &&
                node.output.length === 1 && node.output[0] && inputMap.get(node.output[0].name) === 1 && outputMap.get(node.output[0].name) === 1;
            const attribute = constant ? node.attribute[0] : null;
            if (attribute && attribute.name === 'value' && attribute.type === onnx.AttributeType.TENSOR && attribute.t) {
                const tensor = tensors.map(node.output[0].name);
                tensor.initializer = new onnx.Tensor(attribute.t, 'Constant');
                return false;
            }
            else if (attribute && attribute.name === 'sparse_value' && attribute.type === onnx.AttributeType.SPARSE_TENSOR && attribute.sparse_tensor) {
                const tensor = tensors.map(node.output[0].name);
                tensor.initializer = new onnx.Tensor(attribute.sparse_tensor, 'Sparse Constant');
                return false;
            }
            return true;
        });
        return nodes.map((node) => {
            const schema = context.metadata.type(node.op_type, node.domain);
            const inputs = [];
            node.input = node.input || [];
            for (let i = 0; i < node.input.length; ) {
                const input = schema && schema.inputs && i < schema.inputs.length ? schema.inputs[i] : { name: i.toString() };
                const count = input.list ? node.input.length - i : 1;
                const list = node.input.slice(i, i + count).map((input) => args.map(input.name));
                inputs.push(new onnx.Parameter(input.name, list));
                i += count;
            }
            const outputs = [];
            node.output = node.output || [];
            for (let i = 0; i < node.output.length; ) {
                const output = schema && schema.outputs && i < schema.outputs.length ? schema.outputs[i] : { name: i.toString() };
                const count = output.list ? node.output.length - i : 1;
                const list = node.output.slice(i, i + count).map((output) => args.map(output.name));
                outputs.push(new onnx.Parameter(output.name, list));
                i += count;
            }
            return new onnx.Node(context, node.op_type, node.domain, node.name, node.doc_string, node.attribute, inputs, outputs);
        });
    }
};

onnx.Error = class extends Error {

    constructor(message) {
        super(message);
        this.name = 'Error loading ONNX model.';
    }
};

if (typeof module !== 'undefined' && typeof module.exports === 'object') {
    module.exports.ModelFactory = onnx.ModelFactory;
}
