/* jshint esversion: 6 */

var onnx = onnx || {};
var protobuf = protobuf || require('./protobuf');

onnx.ModelFactory = class {

    match(context) {
        const identifier = context.identifier;
        const extension = identifier.split('.').pop().toLowerCase();
        if (identifier.endsWith('saved_model.pb') || identifier.endsWith('predict_net.pb') || identifier.endsWith('init_net.pb')) {
            return false;
        }
        if (identifier.endsWith('predict_net.pbtxt') || identifier.endsWith('predict_net.prototxt') ||
            identifier.endsWith('init_net.pbtxt') || identifier.endsWith('init_net.prototxt')) {
            return false;
        }
        let tags = context.tags('pb');
        if (tags.size > 0) {
            if (Array.from(tags.keys()).every((tag) => tag <= 20) &&
                Array.from(tags.values()).every((type) => type < 5)) {
                // TensorProto
                if (tags.get(2) === 0 && tags.get(9) === 2) {
                    const schema = [[2,0],[4,2],[5,2],[7,2],[8,2],[9,2]];
                    if (schema.every((pair) => !tags.has(pair[0]) || tags.get(pair[0]) === pair[1])) {
                        return true;
                    }
                }
                // ModelProto
                if (tags.get(7) === 2) {
                    const schema = [[1,0],[2,2],[3,2],[4,2][5,0],[6,2],[7,2],[8,2],[14,2],[20,2]];
                    if (schema.every((pair) => !tags.has(pair[0]) || tags.get(pair[0]) === pair[1])) {
                        return true;
                    }
                }
            }
        }
        const stream = context.stream;
        if (stream.length > 5) {
            const buffer = stream.peek(Math.min(stream.length, 32));
            if (buffer[0] === 0x08 && buffer[1] < 0x08 && buffer[2] === 0x12) {
                const producers = [ 'keras2onnx', 'tf2onnx', 'pytorch', 'skl2onnx', 'onnx-caffe2', 'OnnxMLTools', 'ML.NET', 'kneron_formatter', 'Kneron' ];
                if (producers.some((producer) => Array.from(producer).every((ch, index) => index + 4 < buffer.length && ch.charCodeAt(0) === buffer[index + 4]))) {
                    return true;
                }
            }
        }
        tags = context.tags('pbtxt');
        if (tags.has('ir_version')) {
            return true;
        }
        if (tags.has('graph') && extension !== 'model') {
            return true;
        }
        return false;
    }

    open(context) {
        return context.require('./onnx-proto').then(() => {
            let model = null;
            let format = null;
            const identifier = context.identifier;
            const extension = identifier.split('.').pop().toLowerCase();
            switch (extension) {
                case 'pbtxt':
                case 'prototxt': {
                    try {
                        onnx.proto = protobuf.get('onnx').onnx;
                        const buffer = context.stream.peek();
                        const reader = protobuf.TextReader.create(buffer);
                        model = onnx.proto.ModelProto.decodeText(reader);
                        format = 'ONNX' + (model.ir_version ? ' v' + model.ir_version.toString() : '');
                    }
                    catch (error) {
                        const message = error && error.message ? error.message : error.toString();
                        throw new onnx.Error('File text format is not onnx.ModelProto (' + message.replace(/\.$/, '') + ').');
                    }
                    break;
                }
                default: {
                    const tags = context.tags('pb');
                    const tensor = (tags.has(1) && tags.get(1) === 0 && tags.has(2) && tags.get(2) === 0 && tags.has(9) && tags.get(9) === 2);
                    if (!tensor) {
                        // input_0.pb, output_0.pb
                        try {
                            onnx.proto = protobuf.get('onnx').onnx;
                            const buffer = context.stream.peek();
                            const reader = protobuf.Reader.create(buffer);
                            model = onnx.proto.ModelProto.decode(reader);
                            format = 'ONNX' + (model.ir_version ? ' v' + model.ir_version.toString() : '');
                        }
                        catch (error) {
                            const message = error && error.message ? error.message : error.toString();
                            throw new onnx.Error('File format is not onnx.ModelProto (' + message.replace(/\.$/, '') + ').');
                        }
                    }
                    else {
                        try {
                            onnx.proto = protobuf.get('onnx').onnx;
                            const buffer = context.stream.peek();
                            const reader = protobuf.Reader.create(buffer);
                            const tensor = onnx.proto.TensorProto.decode(reader);
                            tensor.name = tensor.name || context.identifier;
                            model = new onnx.proto.ModelProto();
                            model.graph = new onnx.proto.GraphProto();
                            model.graph.initializer = [ tensor ];
                            model.graph.value_info = [ new onnx.proto.ValueInfoProto() ];
                            model.graph.value_info[0].name = tensor.name;
                            model.graph.node = [ new onnx.proto.NodeProto() ];
                            model.graph.node[0].op_type = 'Constant';
                            model.graph.node[0].attribute = [ new onnx.proto.AttributeProto() ];
                            model.graph.node[0].attribute[0].name = 'value';
                            model.graph.node[0].attribute[0].t = tensor;
                            format = 'ONNX Tensor';
                        }
                        catch (error) {
                            const message = error && error.message ? error.message : error.toString();
                            throw new onnx.Error('File format is not onnx.TensorProto (' + message.replace(/\.$/, '') + ').');
                        }
                    }
                }
            }
            return onnx.Metadata.open(context).then((metadata) => {
                return new onnx.Model(metadata, model, format);
            });
        });
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

        const imports = {};
        if (model.opset_import && model.opset_import.length > 0) {
            const results = [];
            for (const opset_import of model.opset_import) {
                let domain = opset_import.domain || 'ai.onnx';
                const result = domain + ' v' + opset_import.version;
                if (!results.includes(result)) {
                    results.push(result);
                }
                domain = domain == 'ai.onnx' ? '' : domain;
                if (!imports[domain] || imports[domain] > opset_import.version) {
                    imports[domain] = opset_import.version;
                }
            }
            this._imports = results.join(', ');
        }
        if (Object.keys(imports).length == 0) {
            imports[''] = 1;
            imports['ai.onnx.ml'] = 1;
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

        if (graph) {
            this._name = graph.name || null;
            this._description = graph.doc_string || '';

            const tensors = new Map();
            tensors.map = function(name) {
                if (!this.has(name)) {
                    this.set(name, { name: name });
                }
                return this.get(name);
            };
            for (const node of graph.node) {
                node.input = node.input.map((name) => tensors.map(name));
                node.output = node.output.map((name) => tensors.map(name));
                node.param = {};
                const AttributeType = onnx.proto.AttributeProto.AttributeType;
                for (const attribute of node.attribute) {
                    attribute.type = onnx.Utility.attributeType(attribute);
                    switch (attribute.type) {
                        case AttributeType.INT:
                            node.param[attribute.name] = attribute.i;
                            break;
                        case AttributeType.FLOAT:
                            node.param[attribute.name] = attribute.f;
                            break;
                        case AttributeType.STRING:
                            node.param[attribute.name] = attribute.s;
                            break;
                        case AttributeType.INTS:
                            node.param[attribute.name] = attribute.ints;
                            break;
                        case AttributeType.FLOATS:
                            node.param[attribute.name] = attribute.floats;
                            break;
                        case AttributeType.STRINGS:
                            node.param[attribute.name] = attribute.strings;
                            break;
                        case AttributeType.TENSOR:
                            node.param[attribute.name] = attribute.t;
                            break;
                        case AttributeType.SPARSE_TENSOR:
                            node.param[attribute.name] = attribute.sparse_tensor;
                            break;
                    }
                }
            }
            for (const initializer of graph.initializer) {
                const tensor = tensors.map(initializer.name);
                tensor.initializer = new onnx.Tensor(initializer, 'Initializer');
            }
            for (const sparse_initializer of graph.sparse_initializer) {
                const tensor = tensors.map(sparse_initializer.name);
                tensor.initializer = new onnx.Tensor(sparse_initializer, 'Sparse Initializer');
            }
            for (const tensor_annotation of graph.quantization_annotation) {
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
            const nodes = [];
            const inputCountMap = new Map();
            const outputCountMap = new Map();
            for (const node of graph.node) {
                for (const input of node.input) {
                    inputCountMap.set(input.name, inputCountMap.has(input.name) ? inputCountMap.get(input.name) + 1 : 1);
                }
                for (const output of node.output) {
                    outputCountMap.set(output.name, inputCountMap.has(output.name) ? inputCountMap.get(output.name) + 1 : 1);
                }
            }
            for (const input of graph.input) {
                inputCountMap.delete(input);
            }
            for (const output of graph.output) {
                outputCountMap.delete(output);
            }
            for (const node of graph.node) {
                let initializerNode = false;
                if (node.op_type == 'Constant' && node.input.length == 0 && node.output.length == 1) {
                    const name = node.output[0].name;
                    if (inputCountMap.has(name) && inputCountMap.get(name) == 1 &&
                        outputCountMap.has(name) && outputCountMap.get(name) == 1 &&
                        node.attribute.length == 1) {
                        const attribute = node.attribute[0];
                        if (attribute) {
                            const tensor = tensors.map(name);
                            switch (attribute.name) {
                                case 'value': {
                                    tensor.initializer = new onnx.Tensor(attribute.t, 'Constant');
                                    initializerNode = true;
                                    break;
                                }
                                case 'sparse_value':
                                    tensor.initializer = new onnx.Tensor(attribute.sparse_tensor, 'Sparse Constant');
                                    initializerNode = true;
                                    break;
                            }
                        }
                    }
                }
                if (!initializerNode) {
                    nodes.push(node);
                }
            }

            new onnx.Inference(graph);

            const args = new Map();
            args.map = function(name) {
                if (!this.has(name)) {
                    const tensor = tensors.map(name);
                    const type = tensor.initializer ? tensor.initializer.type : tensor.type || null;
                    this.set(name, new onnx.Argument(name, type, tensor.initializer, tensor.annotation, tensor.description));
                }
                return this.get(name);
            };
            for (const valueInfo of graph.input) {
                const argument = args.map(valueInfo.name);
                if (!argument.initializer) {
                    this._inputs.push(new onnx.Parameter(valueInfo.name, [ argument ]));
                }
            }
            for (const valueInfo of graph.output) {
                const argument = args.map(valueInfo.name);
                if (!argument.initializer) {
                    this._outputs.push(new onnx.Parameter(valueInfo.name, [ argument ]));
                }
            }
            for (const node of nodes) {
                const schema = context.metadata.type(node.op_type);
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
                this._nodes.push(new onnx.Node(context, node.op_type, node.domain, node.name, node.doc_string, node.attribute, inputs, outputs));
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

    constructor(context, type, domain, name, description, attributes, inputs, outputs) {
        this._metadata = context.metadata;
        this._type = type;
        this._domain = domain || '';
        this._name = name || '';
        this._description = description || '';
        this._inputs = inputs;
        this._outputs = outputs;
        this._attributes = (attributes || []).map((attribute) => new onnx.Attribute(context, this.type, attribute));
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

    get metadata() {
        return this._metadata.type(this._type);
    }

    get domain() {
        return this._domain;
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

    constructor(context, operator, attribute) {
        this._name = attribute.name;
        this._description = attribute.doc_string || '';
        this._type = null;
        this._value = null;

        const AttributeType = onnx.proto.AttributeProto.AttributeType;
        switch (attribute.type) {
            case AttributeType.FLOAT:
                this._value = attribute.f;
                this._type = 'float32';
                break;
            case AttributeType.INT:
                this._value = attribute.i;
                this._type = 'int64';
                break;
            case AttributeType.STRING:
                switch (operator) {
                    case 'Int8GivenTensorFill':
                        this._value = Array.from(attribute.s);
                        break;
                    default:
                        this._value = onnx.Utility.decodeText(attribute.s);
                        break;
                }
                this._type = 'string';
                break;
            case AttributeType.TENSOR:
                this._value = new onnx.Tensor(attribute.t);
                this._type = 'tensor';
                break;
            case AttributeType.GRAPH:
                this._value = context.graph(attribute.g);
                this._type = 'graph';
                break;
            case AttributeType.FLOATS:
                this._value = attribute.floats;
                this._type = 'float32[]';
                break;
            case AttributeType.INTS:
                this._value = attribute.ints;
                this._type = 'int64[]';
                break;
            case AttributeType.STRINGS:
                this._value = attribute.strings.map((s) => onnx.Utility.decodeText(s));
                this._type = 'string[]';
                break;
            case AttributeType.TENSORS:
                this._value = attribute.tensors.map((tensor) => new onnx.Tensor(tensor));
                this._type = 'tensor[]';
                break;
            case AttributeType.GRAPHS:
                this._value = attribute.graphs.map((graph) => context.graph(graph));
                this._type = 'graph[]';
                break;
            case AttributeType.SPARSE_TENSOR:
                this._value = new onnx.Tensor(attribute.sparse_tensor);
                this._type = 'tensor';
                break;
            case AttributeType.SPARSE_TENSORS:
                this._value = attribute.sparse_tensors.map((tensor) => new onnx.Tensor(tensor));
                this._type = 'tensor[]';
                break;
            default:
                throw new onnx.Error("Unknown attribute type '" + attribute.type + "'.");
        }

        const metadata = context.metadata.attribute(operator, attribute.name);
        if (metadata) {
            if (metadata.type && metadata.type !== this._type) {
                throw new onnx.Error("Unexpected '" + operator + ":" + attribute.name + "' attribute type '" + this._type + "'.");
            }
            if (Object.prototype.hasOwnProperty.call(metadata, 'default')) {
                if (this._value == metadata.default) {
                    this._visible = false;
                }
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
        if (tensor instanceof onnx.proto.SparseTensorProto) {
            this._type = new onnx.TensorType(tensor.values.data_type, new onnx.TensorShape(tensor.dims.map((dim) => dim)), null);
            this._sparse = true;
        }
        else {
            this._name = tensor.name || '';
            this._type = new onnx.TensorType(tensor.data_type, new onnx.TensorShape(tensor.dims.map((dim) => dim)), null);
            this._external = tensor.data_location === onnx.proto.TensorProto.DataLocation.EXTERNAL;
            switch (tensor.data_type) {
                case onnx.proto.TensorProto.DataType.FLOAT16:
                    this._dataType = tensor.data_type;
                    if (tensor.int32_data && tensor.int32_data.length > 0) {
                        this._rawData = new Uint8Array(tensor.int32_data.length << 1);
                        const view = new DataView(this._rawData.buffer, this._rawData.byteOffset, this._rawData.byteLength);
                        const data = tensor.int32_data;
                        for (let i = 0; i < data.length; i++) {
                            view.setUint16(i << 1, data[i], true);
                        }
                    }
                    break;
                case onnx.proto.TensorProto.DataType.FLOAT:
                    this._dataType = tensor.data_type;
                    this._data = tensor.float_data;
                    break;
                case onnx.proto.TensorProto.DataType.DOUBLE:
                    this._dataType = tensor.data_type;
                    this._data = tensor.double_data;
                    break;
                case onnx.proto.TensorProto.DataType.BOOL:
                case onnx.proto.TensorProto.DataType.INT8:
                case onnx.proto.TensorProto.DataType.UINT8:
                case onnx.proto.TensorProto.DataType.INT16:
                case onnx.proto.TensorProto.DataType.UINT16:
                case onnx.proto.TensorProto.DataType.INT32:
                    this._dataType = tensor.data_type;
                    this._data = tensor.int32_data;
                    break;
                case onnx.proto.TensorProto.DataType.UINT32:
                case onnx.proto.TensorProto.DataType.UINT64:
                    this._dataType = tensor.data_type;
                    this._data = tensor.uint64_data;
                    break;
                case onnx.proto.TensorProto.DataType.INT64:
                    this._dataType = tensor.data_type;
                    this._data = tensor.int64_data;
                    break;
            }
            if (this._data && this._data.length === 0) {
                delete this._data;
            }
            if (!this._data && tensor.raw_data && tensor.raw_data.length > 0) {
                this._rawData = this._rawData || tensor.raw_data;
            }
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
        if (this._external) {
            context.state = 'External data not implemented.';
            return context;
        }
        if (this._dataType === undefined) {
            context.state = 'Tensor data type is not implemented.';
            return context;
        }
        if (!this._data && !this._rawData) {
            context.state = 'Tensor data is empty.';
            return context;
        }
        context.index = 0;
        context.count = 0;
        context.shape = this._type.shape.dimensions;
        context.dataType = this._dataType;
        context.data = this._data;
        context.view = this._rawData ? new DataView(this._rawData.buffer, this._rawData.byteOffset, this._rawData.byteLength) : null;
        return context;
    }

    _decode(context, dimension) {
        const shape = context.shape.length !== 0 ? context.shape : [ 1 ];
        const results = [];
        const size = shape[dimension];
        if (dimension == shape.length - 1) {
            for (let i = 0; i < size; i++) {
                if (context.count > context.limit) {
                    results.push('...');
                    return results;
                }
                if (context.data) {
                    let value = context.data[context.index++];
                    switch (context.dataType) {
                        case onnx.proto.TensorProto.DataType.BOOL:
                            value = value === 0 ? false : true;
                            break;
                    }
                    results.push(value);
                    context.count++;
                }
                else if (context.view) {
                    switch (context.dataType) {
                        case onnx.proto.TensorProto.DataType.FLOAT16:
                            results.push(context.view.getFloat16(context.index, true));
                            context.index += 2;
                            context.count++;
                            break;
                        case onnx.proto.TensorProto.DataType.FLOAT:
                            results.push(context.view.getFloat32(context.index, true));
                            context.index += 4;
                            context.count++;
                            break;
                        case onnx.proto.TensorProto.DataType.DOUBLE:
                            results.push(context.view.getFloat64(context.index, true));
                            context.index += 8;
                            context.count++;
                            break;
                        case onnx.proto.TensorProto.DataType.INT8:
                            results.push(context.view.getInt8(context.index, true));
                            context.index++;
                            context.count++;
                            break;
                        case onnx.proto.TensorProto.DataType.UINT8:
                            results.push(context.view.getUint8(context.index, true));
                            context.index++;
                            context.count++;
                            break;
                        case onnx.proto.TensorProto.DataType.INT16:
                            results.push(context.view.getInt16(context.index, true));
                            context.index += 2;
                            context.count++;
                            break;
                        case onnx.proto.TensorProto.DataType.UINT16:
                            results.push(context.view.getUint16(context.index, true));
                            context.index += 2;
                            context.count++;
                            break;
                        case onnx.proto.TensorProto.DataType.INT32:
                            results.push(context.view.getInt32(context.index, true));
                            context.index += 4;
                            context.count++;
                            break;
                        case onnx.proto.TensorProto.DataType.UINT32:
                            results.push(context.view.getUint32(context.index, true));
                            context.index += 4;
                            context.count++;
                            break;
                        case onnx.proto.TensorProto.DataType.INT64:
                            results.push(context.view.getInt64(context.index, true));
                            context.index += 8;
                            context.count++;
                            break;
                        case onnx.proto.TensorProto.DataType.UINT64:
                            results.push(context.view.getUint64(context.index, true));
                            context.index += 8;
                            context.count++;
                            break;
                        case onnx.proto.TensorProto.DataType.BOOL:
                            results.push(context.view.getInt8(context.index, true) === 0 ? false : true);
                            context.index += 1;
                            context.count++;
                            break;
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
        this._dataType = onnx.Utility.formatElementType(dataType);
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
        this._keyType = onnx.Utility.formatElementType(keyType);
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

onnx.GraphMetadata = class {

    constructor(metadata, imports) {
        this._metadata = metadata;
        this._imports = imports;
        this._cache = new Map();
        this._attributeCache = new Map();
    }

    type(operator) {
        if (!this._cache.has(operator)) {
            this._cache.set(operator, this._metadata.type(operator, this._imports));
        }
        return this._cache.get(operator);
    }

    attribute(operator, name) {
        const key = operator + ':' + name;
        if (!this._attributeCache.has(key)) {
            const schema = this.type(operator);
            if (schema && schema.attributes && schema.attributes.length > 0) {
                for (const attribute of schema.attributes) {
                    this._attributeCache.set(operator + ':' + attribute.name, attribute);
                }
            }
            if (!this._attributeCache.has(key)) {
                this._attributeCache.set(key, null);
            }
        }
        return this._attributeCache.get(key);
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
        this._map = {};
        if (data) {
            const items = JSON.parse(data);
            if (items) {
                for (const item of items) {
                    if (item.name && item.schema) {
                        const name = item.name;
                        item.schema.name = name;
                        this._map[name] = this._map[name] || [];
                        this._map[name].push(item.schema);
                    }
                }
            }
        }
    }

    type(operator, imports) {
        let result = null;
        const schemas = this._map[operator];
        if (schemas) {
            let version = -1;
            for (const schema of schemas) {
                const domain = schema.domain === 'ai.onnx' ? '' : schema.domain;
                const importVersion = imports[domain];
                const sinceVersion = schema.since_version;
                if (importVersion >= sinceVersion && version < sinceVersion) {
                    version = sinceVersion;
                    result = schema;
                }
            }
        }
        return result;
    }
};

onnx.Inference = class {

    constructor(graph) {
        this._outputs = new Map();

        for (const node of graph.node) {
            for (const output of node.output) {
                this._outputs.set(output.name, node);
            }
        }

        for (const output of graph.output) {
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

onnx.Utility = class {

    static decodeText(value) {
        onnx.Utility._utf8Decoder = onnx.Utility._utf8Decoder || new TextDecoder('utf-8');
        return onnx.Utility._utf8Decoder.decode(value);
    }

    static formatElementType(elementType) {
        if (!onnx.Utility._elementTypeMap) {
            const map = {};
            map[onnx.proto.TensorProto.DataType.UNDEFINED] = 'UNDEFINED';
            map[onnx.proto.TensorProto.DataType.FLOAT] = 'float32';
            map[onnx.proto.TensorProto.DataType.UINT8] = 'uint8';
            map[onnx.proto.TensorProto.DataType.INT8] = 'int8';
            map[onnx.proto.TensorProto.DataType.UINT16] = 'uint16';
            map[onnx.proto.TensorProto.DataType.INT16] = 'int16';
            map[onnx.proto.TensorProto.DataType.INT32] = 'int32';
            map[onnx.proto.TensorProto.DataType.INT64] = 'int64';
            map[onnx.proto.TensorProto.DataType.STRING] = 'string';
            map[onnx.proto.TensorProto.DataType.BOOL] = 'boolean';
            map[onnx.proto.TensorProto.DataType.FLOAT16] = 'float16';
            map[onnx.proto.TensorProto.DataType.DOUBLE] = 'float64';
            map[onnx.proto.TensorProto.DataType.UINT32] = 'uint32';
            map[onnx.proto.TensorProto.DataType.UINT64] = 'uint64';
            map[onnx.proto.TensorProto.DataType.COMPLEX64] = 'complex64';
            map[onnx.proto.TensorProto.DataType.COMPLEX128] = 'complex128';
            map[onnx.proto.TensorProto.DataType.BFLOAT16] = 'bfloat16';
            onnx.Utility._elementTypeMap = map;
        }
        const name = onnx.Utility._elementTypeMap[elementType];
        if (name) {
            return name;
        }
        return onnx.Utility._elementTypeMap[onnx.proto.TensorProto.DataType.UNDEFINED];
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
            case 'tensor_type':
            case 'sparse_tensor_type': {
                let shape = [];
                if (type.tensor_type.shape && type.tensor_type.shape.dim) {
                    shape = type.tensor_type.shape.dim.map((dim) => {
                        return dim.dim_param ? dim.dim_param : dim.dim_value;
                    });
                }
                return new onnx.TensorType(type.tensor_type.elem_type, new onnx.TensorShape(shape), denotation);
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

    static attributeType(attribute) {
        if (attribute.type) {
            return attribute.type;
        }
        const AttributeType = onnx.proto.AttributeProto.AttributeType;
        if (attribute.ints && attribute.ints.length > 0) {
            return AttributeType.INTS;
        }
        else if (attribute.floats && attribute.floats.length > 0) {
            return AttributeType.FLOATS;
        }
        else if (attribute.strings && attribute.strings.length > 0) {
            return AttributeType.STRINGS;
        }
        else if (attribute.graphs && attribute.graphs.length > 0) {
            return AttributeType.GRAPHS;
        }
        else if (attribute.s && attribute.s.length > 0) {
            return AttributeType.STRING;
        }
        else if (Object.prototype.hasOwnProperty.call(attribute, 'f')) {
            return AttributeType.FLOAT;
        }
        else if (Object.prototype.hasOwnProperty.call(attribute, 'i')) {
            return AttributeType.INT;
        }
        else if (Object.prototype.hasOwnProperty.call(attribute, 't')) {
            return AttributeType.TENSOR;
        }
        else if (Object.prototype.hasOwnProperty.call(attribute, 'g')) {
            return AttributeType.GRAPH;
        }
        else if (Object.prototype.hasOwnProperty.call(attribute, 'sparse_tensor')) {
            return AttributeType.SPARSE_TENSOR;
        }
        return AttributeType.UNDEFINED;
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
