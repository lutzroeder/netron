
var onnx = {};
var protobuf = require('./protobuf');
var flatbuffers = require('./flatbuffers');
var text = require('./text');

onnx.ModelFactory = class {

    match(context) {
        const identifier = context.identifier;
        const extensions = [
            'saved_model.pb', 'predict_net.pb', 'init_net.pb',
            'predict_net.pbtxt', 'init_net.pbtxt', 'predict_net.prototxt', 'init_net.prototxt'
        ];
        if (extensions.some((extension) => identifier.endsWith(extension))) {
            return undefined;
        }
        const entries = [
            onnx.OrtReader,
            onnx.ProtoReader,
            onnx.TextReader,
            onnx.JsonReader,
            onnx.PickleReader
        ];
        for (const entry of entries) {
            const reader = entry.open(context);
            if (reader) {
                return reader;
            }
        }
        return undefined;
    }

    async open(context, target) {
        await target.read();
        const model = target.model;
        const format = target.format;
        const metadata = await onnx.Metadata.open(context);
        const locations = new Set();
        const location = (tensor) => {
            if ((onnx.proto && tensor instanceof onnx.proto.SparseTensorProto) ||
                (onnx.schema && tensor instanceof onnx.schema.SparseTensor)) {
                location(tensor.indices);
                location(tensor.indices);
            } else if (tensor.data_location === onnx.DataLocation.EXTERNAL && Array.isArray(tensor.external_data)) {
                for (const entry of tensor.external_data) {
                    if (entry.key === 'location') {
                        locations.add(entry.value);
                    }
                }
            }
        };
        const graphs = new Set();
        const queue = [ model.graph ];
        while (queue.length > 0) {
            const graph = queue.shift();
            if (Array.isArray(graph.initializer)) {
                for (const initializer of graph.initializer) {
                    location(initializer);
                }
            }
            if (Array.isArray(graph.sparse_initializer)) {
                for (const sparse_initializer of graph.sparse_initializer) {
                    location(sparse_initializer);
                }
            }
            if (Array.isArray(graph.node)) {
                for (const node of graph.node) {
                    if (Array.isArray(node.attribute)) {
                        for (const attribute of node.attribute) {
                            if (attribute.g) {
                                queue.push(attribute.g);
                            } else if (attribute.t) {
                                location(attribute.t);
                            } else if (attribute.sparse_tensor) {
                                location(attribute.sparse_tensor);
                            } else if (Array.isArray(attribute.graphs) && attribute.graphs.length > 0) {
                                for (const graph of attribute.graphs) {
                                    queue.push(graph);
                                }
                            } else if (Array.isArray(attribute.tensors) && attribute.tensors.length > 0) {
                                for (const tensor of attribute.tensors) {
                                    location(tensor);
                                }
                            } else if (Array.isArray(attribute.sparse_tensors) && attribute.sparse_tensors.length > 0) {
                                for (const tensor of attribute.sparse_tensors) {
                                    location(tensor);
                                }
                            }
                        }
                    }
                }
            }
            graphs.add(graph);
        }
        const weights = new Map();
        const keys = Array.from(locations);
        const promises = keys.map((location) => context.request(location, null));
        const streams = await Promise.all(promises.map((promise) => promise.then((value) => value).catch(() => null)));
        for (let i = 0; i < keys.length; i++) {
            if (streams[i] !== null) {
                weights.set(keys[i], streams[i]);
            }
        }
        return new onnx.Model(metadata, format, model, Array.from(graphs), weights);
    }
};

onnx.Model = class {

    constructor(metadata, format, model, graphs, locations) {
        this._graphs = [];
        this._format = format;
        this._producer = model.producer_name && model.producer_name.length > 0 ? model.producer_name + (model.producer_version && model.producer_version.length > 0 ? ' ' + model.producer_version : '') : null;
        this._domain = model.domain;
        const model_version = model.model_version === undefined || typeof model.model_version === 'number' ? model.model_version : model.model_version.toNumber();
        this._version = model_version ? model_version.toString() : '';
        this._description = model.doc_string;
        this._metadata = [];
        this._imports = null;
        const imports = new Map();
        if (model.opset_import && model.opset_import.length > 0) {
            for (const opset_import of model.opset_import) {
                const domain = opset_import.domain || 'ai.onnx';
                const version = opset_import.version ? typeof opset_import.version === 'number' ? opset_import.version: opset_import.version.toNumber() : 0;
                if (!imports.has(domain) || imports.get(domain) > version) {
                    imports.set(domain, version);
                }
            }
            this._imports = Array.from(imports).map((entry) => entry[0] + ' v' + entry[1].toString());
        }
        if (imports.size == 0) {
            imports.set('ai.onnx', 1);
            imports.set('ai.onnx.ml', 1);
        }
        let imageFormat = '';
        const metadata_props = model.metadata_props;
        if (metadata_props) {
            const metadata = new Map(metadata_props.map((entry) => [ entry.key, entry.value ]));
            const converted_from = metadata.get('converted_from');
            if (converted_from) {
                this._metadata.push({ name: 'source', value: converted_from });
            }
            const author = metadata.get('author');
            if (author) {
                this._metadata.push({ name: 'author', value: author });
            }
            const company = metadata.get('company');
            if (company) {
                this._metadata.push({ name: 'company', value: company });
            }
            let license = metadata.get('license');
            const license_url = metadata.get('license_url');
            if (license_url) {
                license = '<a href=\'' + license_url + '\'>' + (license ? license : license_url) + '</a>';
            }
            if (license) {
                this._metadata.push({ name: 'license', value: license });
            }
            metadata.delete('author');
            metadata.delete('company');
            metadata.delete('converted_from');
            metadata.delete('license');
            metadata.delete('license_url');
            const imageMetadata = {};
            for (const entry of metadata) {
                const name = entry[0];
                const value = entry[1];
                switch (name) {
                    case 'Image.BitmapPixelFormat':
                    case 'Image.ColorSpaceGamma':
                    case 'Image.NominalPixelRange':
                        imageMetadata[name] = value;
                        break;
                    default:
                        this._metadata.push({ name: name, value: value });
                        break;
                }
            }
            imageFormat = [ imageMetadata['Image.BitmapPixelFormat'], imageMetadata['Image.ColorSpaceGamma'], imageMetadata['Image.NominalPixelRange'] ].filter((item) => item);
        }
        metadata = new onnx.GraphMetadata(metadata, imports);
        const context = new onnx.ModelContext(metadata, locations, imageFormat);
        for (const func of model.functions || []) {
            context.metadata.add(new onnx.Function(context, func));
        }
        this._graphs = [];
        for (const graph of graphs) {
            this._graphs.push(context.graph(graph));
        }
    }

    get format() {
        return this._format;
    }

    get version() {
        return this._version;
    }

    get imports() {
        return this._imports;
    }

    get producer() {
        return this._producer;
    }

    get domain() {
        return this._domain || null;
    }

    get description() {
        return this._description || null;
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
        context = new onnx.GraphContext(context, graph.node);
        if (Array.isArray(graph.initializer)) {
            for (const initializer of graph.initializer) {
                const tensor = context.tensor(initializer.name);
                tensor.initializer = new onnx.Tensor(context, initializer, 'Initializer');
            }
        }
        if (Array.isArray(graph.sparse_initializer)) {
            for (const sparse_initializer of graph.sparse_initializer) {
                const tensor = context.tensor(sparse_initializer.values.name);
                tensor.initializer = new onnx.Tensor(context, sparse_initializer, 'Initializer');
            }
        }
        if (Array.isArray(graph.quantization_annotation)) {
            for (const tensor_annotation of graph.quantization_annotation) {
                const tensor = context.tensor(tensor_annotation.tensor_name);
                const annotation = {};
                for (const entry of tensor_annotation.quant_parameter_tensor_names) {
                    annotation[entry.key] = entry.value;
                }
                tensor.annotation = annotation;
            }
        }
        if (Array.isArray(graph.value_info)) {
            for (const valueInfo of graph.value_info) {
                const tensor = context.tensor(valueInfo.name);
                tensor.type = context.createType(valueInfo.type);
                tensor.description = valueInfo.doc_string;
            }
        }
        graph.input = graph.input.map((valueInfo) => {
            const tensor = context.tensor(valueInfo.name);
            tensor.type = context.createType(valueInfo.type);
            tensor.description = valueInfo.doc_string;
            return tensor;
        });
        graph.output = graph.output.map((valueInfo) => {
            const tensor = context.tensor(valueInfo.name);
            tensor.type = context.createType(valueInfo.type);
            tensor.description = valueInfo.doc_string;
            return tensor;
        });
        new onnx.Inference(graph.node, graph.output);
        context.push(graph.node, graph.input, graph.output);
        this._nodes = context.pop();
        for (const input of graph.input) {
            const value = context.value(input.name);
            if (!value.initializer) {
                this._inputs.push(new onnx.Argument(input.name, [ value ]));
            }
        }
        for (const output of graph.output) {
            const value = context.value(output.name);
            if (!value.initializer) {
                this._outputs.push(new onnx.Argument(output.name, [ value ]));
            }
        }
    }

    get name() {
        return this._name;
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

    get nodes() {
        return this._nodes;
    }

    toString() {
        return 'graph(' + this.name + ')';
    }
};

onnx.Argument = class {

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

onnx.Value = class {

    constructor(name, type, initializer, annotation, description) {
        if (typeof name !== 'string') {
            throw new onnx.Error("Invalid value identifier '" + JSON.stringify(name) + "'.");
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
        attributes = attributes || [];
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
        this._attributes = attributes.map((attribute) => new onnx.Attribute(context, op_type, domain, attribute));
        this._chain = [];
        const identifier = domain ? domain + '.' + op_type : op_type;
        if (identifier === 'com.microsoft.FusedConv') {
            const activation = attributes.find((attribute) => attribute.name === 'activation');
            if (activation) {
                const type = context.decodeText(activation.s);
                this._chain.push(new onnx.Node(context, type, '', '', '', [], [], []));
            }
        }
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

    get attributes() {
        return this._attributes;
    }

    get inputs() {
        return this._inputs;
    }

    get outputs() {
        return this._outputs;
    }

    get chain() {
        return this._chain;
    }
};

onnx.Attribute = class {

    constructor(context, op_type, domain, attribute) {
        this._name = attribute.name;
        this._description = attribute.doc_string || '';
        this._type = null;
        this._value = null;
        if (attribute.ref_attr_name) {
            this._value = attribute.ref_attr_name;
            this._type = 'reference';
            return;
        }
        switch (attribute.type) {
            case onnx.AttributeType.UNDEFINED:
                break;
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
                        this._value = context.decodeText(attribute.s);
                        break;
                }
                this._type = 'string';
                break;
            case onnx.AttributeType.TENSOR:
                this._value = new onnx.Tensor(context, attribute.t);
                this._type = 'tensor';
                break;
            case onnx.AttributeType.GRAPH:
                this._value = context.graph(attribute.g);
                this._type = 'graph';
                break;
            case onnx.AttributeType.FLOATS:
                this._value = ArrayBuffer.isView(attribute.floats) ? Array.from(attribute.floats) : attribute.floats;
                this._type = 'float32[]';
                break;
            case onnx.AttributeType.INTS:
                this._value = ArrayBuffer.isView(attribute.ints) ? Array.from(attribute.ints) : attribute.ints;
                this._type = 'int64[]';
                break;
            case onnx.AttributeType.STRINGS:
                this._value = attribute.strings.map((s) => context.decodeText(s));
                this._type = 'string[]';
                break;
            case onnx.AttributeType.TENSORS:
                this._value = attribute.tensors.map((tensor) => new onnx.Tensor(context, tensor));
                this._type = 'tensor[]';
                break;
            case onnx.AttributeType.GRAPHS:
                this._value = attribute.graphs.map((graph) => context.graph(graph));
                this._type = 'graph[]';
                break;
            case onnx.AttributeType.SPARSE_TENSOR:
                this._value = new onnx.Tensor(context, attribute.sparse_tensor);
                this._type = 'tensor';
                break;
            case onnx.AttributeType.SPARSE_TENSORS:
                this._value = attribute.sparse_tensors.map((tensor) => new onnx.Tensor(context, tensor));
                this._type = 'tensor[]';
                break;
            case onnx.AttributeType.TYPE_PROTO:
                this._value = context.createType(attribute.tp);
                this._type = 'type';
                break;
            case onnx.AttributeType.TYPE_PROTOS:
                this._value = attribute.type_protos.map((type) => context.createType(type));
                this._type = 'type[]';
                break;
            default:
                throw new onnx.Error("Unsupported attribute type '" + attribute.type + "'.");
        }
        const metadata = context.metadata.attribute(op_type, domain, attribute.name);
        if (metadata) {
            if (Object.prototype.hasOwnProperty.call(metadata, 'default') && this._value == metadata.default) {
                this._visible = false;
            }
            if (metadata.type === 'DataType') {
                this._type = metadata.type;
                this._value = context.createDataType(this._value);
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

onnx.Group = class {

    constructor(name, groups) {
        this._type = { name: 'Scope' };
        this._name = name;
        this._nodes = [];
        for (const entry of groups) {
            const key = entry[0];
            if (key === '') {
                for (const node of entry[1]) {
                    this._nodes.push(node);
                }
            } else {
                this._nodes.push(new onnx.Group(name === '' ? key : name + '/' + key, entry[1]));
            }
        }
        const set = new Set();
        const inputs = [];
        const outputs = [];
        for (const node of this._nodes) {
            if (node instanceof onnx.Group) {
                node.freeze();
            }
            for (const parameter of node.outputs) {
                for (const value of parameter.value) {
                    if (!value.initializer) {
                        outputs.push(value);
                        set.add(value.name);
                    }
                }
            }
        }
        for (const node of this._nodes) {
            for (const parameter of node.inputs) {
                for (const value of parameter.value) {
                    if (!set.has(value.name) && !value.initializer) {
                        inputs.push(value);
                    }
                }
            }
        }
        this._inputs = [ new onnx.Argument('inputs', inputs) ];
        this._outputs = [ new onnx.Argument('outputs', outputs) ];
        this._attributes = [];
    }

    get name() {
        return this._name;
    }

    get type() {
        return this._type;
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

onnx.Tensor = class {

    constructor(context, tensor, category) {
        this._category = category || null;
        if (tensor.indices && tensor.values) {
            this._name = tensor.values.name || '';
            this._type = context.createTensorType(tensor.values.data_type, tensor.dims.map((dim) => dim), 'sparse');
            this._location = context.createLocation(tensor.values.data_location);
            this._values = new onnx.Tensor(context, tensor.values);
            this._indices = new onnx.Tensor(context, tensor.indices);
        } else {
            this._name = tensor.name || '';
            this._type = context.createTensorType(tensor.data_type, tensor.dims.map((dim) => dim));
            this._location = context.createLocation(tensor.data_location);
            switch (tensor.data_location) {
                case onnx.DataLocation.DEFAULT: {
                    switch (tensor.data_type) {
                        case onnx.DataType.UNDEFINED: {
                            break;
                        }
                        case onnx.DataType.FLOAT:
                            this._data = new Float32Array(tensor.float_data);
                            this._encoding = '|';
                            break;
                        case onnx.DataType.DOUBLE:
                            this._data = new Float64Array(tensor.double_data);
                            this._encoding = '|';
                            break;
                        case onnx.DataType.BOOL:
                            if (tensor.int32_data && tensor.int32_data.length > 0) {
                                const array = tensor.int32_data;
                                this._data = new Array(array.length);
                                for (let i = 0; i < this._data.length; i++) {
                                    this._data[i] = array[i] === 0 ? false : true;
                                }
                                this._encoding = '|';
                            }
                            break;
                        case onnx.DataType.INT8:
                            this._data = new Int8Array(tensor.int32_data);
                            this._encoding = '|';
                            break;
                        case onnx.DataType.UINT8:
                            this._data = new Uint8Array(tensor.int32_data);
                            this._encoding = '|';
                            break;
                        case onnx.DataType.INT16:
                            this._data = new Int32Array(tensor.int32_data);
                            this._encoding = '|';
                            break;
                        case onnx.DataType.UINT16:
                            this._data = new Int32Array(tensor.int32_data);
                            this._encoding = '|';
                            break;
                        case onnx.DataType.INT32:
                            this._data = new Int32Array(tensor.int32_data);
                            this._encoding = '|';
                            break;
                        case onnx.DataType.UINT32:
                        case onnx.DataType.UINT64:
                            this._data = tensor.uint64_data;
                            this._encoding = '|';
                            break;
                        case onnx.DataType.INT64:
                            this._data = tensor.int64_data;
                            this._encoding = '|';
                            break;
                        case onnx.DataType.STRING:
                            this._data = tensor.string_data;
                            this._encoding = '|';
                            break;
                        case onnx.DataType.COMPLEX64:
                        case onnx.DataType.COMPLEX128:
                            break;
                        case onnx.DataType.FLOAT16:
                        case onnx.DataType.BFLOAT16:
                            if (tensor.int32_data && tensor.int32_data.length > 0) {
                                const array = tensor.int32_data;
                                const buffer = new Uint8Array(array.length << 1);
                                const view = new DataView(buffer.buffer, buffer.byteOffset, buffer.byteLength);
                                for (let i = 0; i < array.length; i++) {
                                    view.setUint16(i << 1, array[i], true);
                                }
                                this._data = buffer;
                                this._encoding = '<';
                            }
                            break;
                        case onnx.DataType.FLOAT8E4M3FN:
                        case onnx.DataType.FLOAT8E4M3FNUZ:
                        case onnx.DataType.FLOAT8E5M2:
                        case onnx.DataType.FLOAT8E5M2FNUZ:
                            if (tensor.int32_data && tensor.int32_data.length > 0) {
                                this._data = new Uint8Array(Array.from(tensor.int32_data));
                                this._encoding = '<';
                            }
                            break;
                        default:
                            throw new onnx.Error("Unsupported tensor data type '" + tensor.data_type + "'.");
                    }
                    if (this._data && (Array.isArray(this._data) || ArrayBuffer.isView(this._data)) && this._data.length === 0) {
                        this._data = undefined;
                    }
                    if (!this._data && tensor.raw_data && tensor.raw_data.length > 0) {
                        this._data = tensor.raw_data;
                        this._encoding = '<';
                    }
                    break;
                }
                case onnx.DataLocation.EXTERNAL: {
                    if (Array.isArray(tensor.external_data)) {
                        const external_data = {};
                        for (const entry of tensor.external_data) {
                            external_data[entry.key] = entry.value;
                        }
                        if (external_data.location && external_data.offset && external_data.length) {
                            const offset = parseInt(external_data.offset, 10);
                            const length = parseInt(external_data.length, 10);
                            if (Number.isInteger(offset) && Number.isInteger(length)) {
                                this._data = context.location(external_data.location, offset, length);
                                this._encoding = '<';
                            }
                        }
                    }
                    break;
                }
                default: {
                    break;
                }
            }
        }
    }

    get name() {
        return this._name;
    }

    get category() {
        return this._category;
    }

    get encoding() {
        return this._encoding;
    }

    get type() {
        return this._type;
    }

    get indices() {
        return this._indices;
    }

    get values() {
        switch (this.type.layout) {
            case 'sparse': {
                return this._values;
            }
            default: {
                if (!this._data || this._data instanceof Uint8Array) {
                    return this._data;
                }
                if (Array.isArray(this._data) || ArrayBuffer.isView(this._data)) {
                    return this._data;
                }
                return this._data.peek();
            }
        }
    }
};

onnx.TensorType = class {

    constructor(dataType, shape, layout, denotation) {
        this._dataType = dataType;
        this._shape = shape;
        this._layout = layout || null;
        this._denotation = denotation || null;
    }

    get dataType() {
        return this._dataType;
    }

    get shape() {
        return this._shape;
    }

    get layout() {
        return this._layout;
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
        return '[' + this._dimensions.map((dim) => dim ? dim.toString() : '?').join(',') + ']';
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
        const elementType = this._elementType ? this._elementType.toString() : '';
        return 'sequence<' + elementType + '>';
    }
};

onnx.MapType = class {

    constructor(keyType, valueType, denotation) {
        this._keyType = keyType;
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

onnx.OptionalType = class {

    constructor(type) {
        this._type = type;
    }

    get type() {
        return this._type;
    }

    toString() {
        return 'optional<' + this._type.toString() + '>';
    }
};

onnx.Function = class {

    constructor(context, func) {
        this._name = func.name;
        this._domain = func.domain;
        this._description = func.doc_string;
        this._inputs = [];
        this._outputs = [];
        this._attributes = func.attribute.map((attribtue) => {
            return { name: attribtue };
        });
        context = new onnx.GraphContext(context, func.node);
        func.input = func.input.map((input) => context.tensor(input));
        func.output = func.output.map((output) => context.tensor(output));
        context.push(func.node, func.input, func.output);
        this._nodes = context.pop();
        for (const input of func.input) {
            const value = context.value(input.name);
            if (!value.initializer) {
                this._inputs.push(new onnx.Argument(input.name, [ value ]));
            }
        }
        for (const output of func.output) {
            const value = context.value(output.name);
            if (!value.initializer) {
                this._outputs.push(new onnx.Argument(output.name, [ value ]));
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
            this._attributes.set(key, null);
            const metadata = this.type(type, domain);
            if (metadata && Array.isArray(metadata.attributes) && metadata.attributes.length > 0) {
                for (const attribute of metadata.attributes) {
                    const key = domain + ':' + type + ':' + attribute.name;
                    this._attributes.set(key, attribute);
                }
            }
        }
        return this._attributes.get(key);
    }
};

onnx.Metadata = class {

    static async open(context) {
        if (onnx.Metadata._metadata) {
            return onnx.Metadata._metadata;
        }
        try {
            const data = await context.request('onnx-metadata.json', 'utf-8', null);
            onnx.Metadata._metadata = new onnx.Metadata(data);
            return onnx.Metadata._metadata;
        } catch (error) {
            onnx.Metadata._metadata = new onnx.Metadata(null);
            return onnx.Metadata._metadata;
        }
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
    BFLOAT16: 16,
    FLOAT8E4M3FN: 17,
    FLOAT8E4M3FNUZ: 18,
    FLOAT8E5M2: 19,
    FLOAT8E5M2FNUZ: 20
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

onnx.ModelContext = class {

    constructor(metadata, locations, imageFormat) {
        this._metadata = metadata;
        this._locations = locations;
        this._imageFormat = imageFormat;
        this._graphs = new Map();
    }

    get metadata() {
        return this._metadata;
    }

    get imageFormat()  {
        return this._imageFormat;
    }

    location(name, offset, length) {
        if (this._locations.has(name)) {
            const stream = this._locations.get(name);
            if (offset >= 0 && (offset + length) <= stream.length) {
                try {
                    const position = stream.position;
                    stream.seek(offset);
                    const value = stream.stream(length);
                    stream.seek(position);
                    return value;
                } catch (error) {
                    // continue regardless of error
                }
            }
        }
        return null;
    }

    graph(value) {
        if (!this._graphs.has(value)) {
            this._graphs.set(value, new onnx.Graph(this, value));
        }
        return this._graphs.get(value);
    }
};

onnx.GraphContext = class {

    constructor(context, nodes) {
        this._context = context;
        this._dataTypes = new Map(Object.entries(onnx.DataType).map((entry) => [ entry[1], entry[0].toLowerCase() ]));
        this._dataTypes.set(onnx.DataType.UNDEFINED, 'undefined');
        this._dataTypes.set(onnx.DataType.BOOL, 'boolean');
        this._dataTypes.set(onnx.DataType.FLOAT, 'float32');
        this._dataTypes.set(onnx.DataType.DOUBLE, 'float64');
        this._tensors = new Map();
        this._values = new Map();
        this._groups = new Map();
        this._nodes = [];
        for (const node of nodes) {
            node.input = node.input.map((name) => this.tensor(name));
            node.output = node.output.map((name) => this.tensor(name));
            node.param = {};
            if (Array.isArray(node.attribute)) {
                for (const attribute of node.attribute) {
                    if (attribute.type) {
                        continue;
                    }
                    if (Array.isArray(attribute.ints) && attribute.ints.length > 0) {
                        attribute.type = onnx.AttributeType.INTS;
                    } else if (Array.isArray(attribute.floats) && attribute.floats.length > 0) {
                        attribute.type = onnx.AttributeType.FLOATS;
                    } else if (Array.isArray(attribute.strings) && attribute.strings.length > 0) {
                        attribute.type = onnx.AttributeType.STRINGS;
                    } else if (Array.isArray(attribute.graphs) && attribute.graphs.length > 0) {
                        attribute.type = onnx.AttributeType.GRAPHS;
                    } else if (Array.isArray(attribute.s) && attribute.s.length > 0) {
                        attribute.type = onnx.AttributeType.STRING;
                    } else if (attribute.f !== undefined) {
                        attribute.type = onnx.AttributeType.FLOAT;
                    } else if (attribute.i !== undefined) {
                        attribute.type = onnx.AttributeType.INT;
                    } else if (attribute.t !== undefined) {
                        attribute.type = onnx.AttributeType.TENSOR;
                    } else if (attribute.g !== undefined) {
                        attribute.type = onnx.AttributeType.GRAPH;
                    } else if (attribute.sparse_tensor !== undefined) {
                        attribute.type =onnx.AttributeType.SPARSE_TENSOR;
                    } else {
                        attribute.type = onnx.AttributeType.UNDEFINED;
                    }
                }
            }
        }
    }

    get metadata() {
        return this._context.metadata;
    }

    graph(name) {
        return this._context.graph(name);
    }

    tensor(name) {
        if (!this._tensors.has(name)) {
            this._tensors.set(name, { name: name });
        }
        return this._tensors.get(name);
    }

    location(name, offset, length) {
        return this._context.location(name, offset, length);
    }

    group(name) {
        if (!this._groups.has(name)) {
            const path = name.split('/');
            if (path.length > 1) {
                path.pop();
                return this.group(path.join('/'));
            }
            this._groups.set(name, new Map([ [ '', [] ]]));
        }
        return this._groups.get(name);
    }

    value(name) {
        if (!this._values.has(name)) {
            const tensor = this.tensor(name);
            const type = tensor.initializer ? tensor.initializer.type : tensor.type || null;
            this._values.set(name, new onnx.Value(name, type, tensor.initializer, tensor.annotation, tensor.description));
        }
        return this._values.get(name);
    }

    createType(type) {
        if (!type) {
            return null;
        }
        let denotation = '';
        switch (type.denotation) {
            case undefined:
            case null:
            case '':
                break;
            case 'TENSOR':
                denotation = 'Tensor';
                break;
            case 'IMAGE':
                denotation = 'Image' + (this._context.imageFormat ? '(' + this._context.imageFormat.join(',') + ')' : '');
                break;
            case 'AUDIO':
                denotation = 'Audio';
                break;
            case 'TEXT':
                denotation = 'Text';
                break;
            default:
                throw new onnx.Error("Unsupported tensor type denotation '" + type.denotation + "'.");
        }
        if (type.tensor_type) {
            const tensor_type = type.tensor_type;
            const shape = tensor_type.shape && tensor_type.shape.dim ? tensor_type.shape.dim.map((dim) => dim.dim_param ? dim.dim_param : dim.dim_value ? dim.dim_value : null) : [];
            return this.createTensorType(tensor_type.elem_type, shape, null, denotation);
        } else if (type.sparse_tensor_type) {
            type = type.sparse_tensor_type;
            const shape = type.shape && type.shape.dim ? type.shape.dim.map((dim) => dim.dim_param ? dim.dim_param : dim.dim_value ? dim.dim_value : null) : [];
            return this.createTensorType(type.elem_type, shape, 'sparse', denotation);
        } else if (type.map_type) {
            return this.createMapType(type.map_type.key_type, this.createType(type.map_type.value_type), denotation);
        } else if (type.sequence_type) {
            return new onnx.SequenceType(this.createType(type.sequence_type.elem_type), denotation);
        } else if (type.opaque_type) {
            return new onnx.OpaqueType(type.opaque_type.domain, type.opaque_type.name);
        } else if (type.optional_type) {
            return new onnx.OptionalType(this.createType(type.optional_type.elem_type), denotation);
        } else if (Object.keys(type).length == 0) {
            return null;
        }
        throw new onnx.Error("Unsupported tensor type '" + JSON.stringify(type) + "'.");
    }

    createTensorType(dataType, shape, layout, denotation) {
        dataType = this.createDataType(dataType);
        return new onnx.TensorType(dataType, new onnx.TensorShape(shape), layout, denotation);
    }

    createMapType(keyType, valueType, denotation) {
        keyType = this.createDataType(keyType);
        return new onnx.MapType(keyType, valueType, denotation);
    }

    createDataType(value) {
        if (!Number.isInteger(value)) {
            if (value && value.toNumber) {
                value = value.toNumber();
            } else if (value && typeof value === 'string' && onnx.DataType[value.toUpperCase()] !== undefined) {
                value = onnx.DataType[value.toUpperCase()];
            } else {
                throw new onnx.Error("Unsupported data type '" + JSON.stringify(value) + "'.");
            }
        }
        if (this._dataTypes.has(value)) {
            return this._dataTypes.get(value);
        }
        throw new onnx.Error("Unsupported data type '" + JSON.stringify(value) + "'.");
    }

    createLocation(value) {
        switch (value) {
            case onnx.DataLocation.DEFAULT: return 'default';
            case onnx.DataLocation.EXTERNAL: return 'external';
            default: return 'UNDEFINED';
        }
    }

    decodeText(value) {
        if (typeof value === 'string') {
            return value;
        }
        this._decoder = this._decoder || new TextDecoder('utf-8');
        return this._decoder.decode(value);
    }

    push(nodes, inputs, outputs) {
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
                const tensor = this.tensor(node.output[0].name);
                tensor.initializer = new onnx.Tensor(this, attribute.t, 'Constant');
                return false;
            } else if (attribute && attribute.name === 'sparse_value' && attribute.type === onnx.AttributeType.SPARSE_TENSOR && attribute.sparse_tensor) {
                const tensor = this.tensor(node.output[0].name);
                tensor.initializer = new onnx.Tensor(this, attribute.sparse_tensor, 'Constant');
                return false;
            }
            return true;
        });
        for (let node of nodes) {
            const schema = this._context.metadata.type(node.op_type, node.domain);
            const inputs = [];
            node.input = node.input || [];
            for (let i = 0; i < node.input.length;) {
                const input = schema && schema.inputs && i < schema.inputs.length ? schema.inputs[i] : { name: i.toString() };
                const count = input.list ? node.input.length - i : 1;
                const list = node.input.slice(i, i + count).filter((arg) => arg.name !== '' || arg.initializer);
                const args = list.map((input) => this.value(input.name));
                inputs.push(new onnx.Argument(input.name, args));
                i += count;
            }
            const outputs = [];
            node.output = node.output || [];
            for (let i = 0; i < node.output.length;) {
                const output = schema && schema.outputs && i < schema.outputs.length ? schema.outputs[i] : { name: i.toString() };
                const count = output.list ? node.output.length - i : 1;
                const list = node.output.slice(i, i + count).filter((arg) => arg.name !== '' || arg.initializer);
                const args = list.map((output) => this.value(output.name));
                outputs.push(new onnx.Argument(output.name, args));
                i += count;
            }
            node = new onnx.Node(this, node.op_type, node.domain, node.name, node.doc_string, node.attribute, inputs, outputs);
            this._nodes.push(node);

            // const path = (node.name || '').split('/');
            // path.pop();
            // this.group(path.join('/')).get('').push(node);
        }
    }

    pop() {
        /*
        const nodes = [];
        for (const entry of this._groups) {
            if (entry[0] === '') {
                for (const node of entry[1].get('')) {
                    nodes.push(node);
                }
                continue;
            }
            nodes.push(new onnx.Group(entry[0], entry[1]));
        }
        return nodes;
        */
        return this._nodes;
    }
};

onnx.ProtoReader = class {

    static open(context) {
        const binaryTags = context.tags('pb');
        if (binaryTags.size > 0) {
            const tags = binaryTags;
            if (tags.size === 1 && tags.get(1) === 2) {
                const tags = context.tags('pb+');
                const match = (tags, schema) => {
                    for (const entry of schema) {
                        const key = entry[0];
                        const inner = entry[1];
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
                // mediapipe.BoxDetectorIndex
                if (match(tags, [[1,[[1,[[1,[[1,5],[2,5],[3,5],[4,5],[6,0],[7,5],[8,5],[10,5],[11,0],[12,0]]],[2,5],[3,[]]]],[2,false],[3,false],[4,false],[5,false]]],[2,false],[3,false]])) {
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
                if (tags.get(1) === 0 && tags.get(2) === 0) {
                    const schema = [[1,0],[2,0],[4,2],[5,2],[7,2],[8,2],[9,2]];
                    if (schema.every((entry) => !tags.has(entry[0]) || tags.get(entry[0]) === entry[1])) {
                        return new onnx.ProtoReader(context, 'binary', 'tensor');
                    }
                }
                // GraphProto
                if (tags.get(1) === 2) {
                    const schema = [[1,2],[2,2],[3,2],[4,2],[5,2],[6,0],[7,0],[8,2],[9,2],[10,2],[11,2],[12,2],[13,2],[14,2]];
                    if (schema.every((entry) => !tags.has(entry[0]) || tags.get(entry[0]) === entry[1])) {
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
                            const nameBuffer = decode(nodeBuffer, 4);
                            if (nameBuffer && nameBuffer.every((c) => c > 0x20 && c < 0x7f)) {
                                return new onnx.ProtoReader(context, 'binary', 'graph');
                            }
                        }
                    }
                }
                // ModelProto
                if (tags.get(7) === 2) {
                    const schema = [[1,0],[2,2],[3,2],[4,2],[5,0],[6,2],[7,2],[8,2],[14,2],[20,2]];
                    if (schema.every((entry) => !tags.has(entry[0]) || tags.get(entry[0]) === entry[1])) {
                        return new onnx.ProtoReader(context, 'binary', 'model');
                    }
                }
            }
        }
        const stream = context.stream;
        if (stream && stream.length > 5) {
            const buffer = stream.peek(Math.min(stream.length, 32));
            if (buffer[0] === 0x08 && buffer[1] < 0x0A && buffer[2] === 0x12) {
                const producers = [
                    'backend-test', 'BrainwaveCompiler',
                    'CNTK', 'customvision',
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
                    return new onnx.ProtoReader(context, 'binary', 'model');
                }
            }
        }
        if (stream && stream.length > 8) {
            const buffer = stream.peek(4);
            const length = buffer[0] | (buffer[1] << 8) | (buffer[2] << 16) | (buffer[3] << 24);
            if (length === stream.length - 4) {
                stream.seek(4);
                try {
                    const reader = protobuf.BinaryReader.open(stream);
                    const tags = reader.signature();
                    if (tags.get(7) === 2) {
                        stream.seek(4);
                        return new onnx.ProtoReader(context, 'binary', 'model');
                    }
                } catch (error) {
                    // continue regardless of error
                }
            }
        }
        const textTags = context.tags('pbtxt');
        if (textTags.size > 0) {
            const tags = textTags;
            if (tags.has('ir_version')) {
                return new onnx.ProtoReader(context, 'text', 'model');
            }
            const identifier = context.identifier;
            const extension = identifier.split('.').pop().toLowerCase();
            if (tags.has('graph') && extension !== 'model') {
                return new onnx.ProtoReader(context, 'text', 'model');
            }
        }
        return undefined;
    }

    constructor(context, encoding, type) {
        this._context = context;
        this._encoding = encoding;
        this._type = type;
    }

    async read() {
        await this._context.require('./onnx-proto');
        onnx.proto = protobuf.get('onnx').onnx;
        const stream = this._context.stream;
        switch (this._encoding) {
            case 'text': {
                try {
                    const reader = protobuf.TextReader.open(stream);
                    this.model = onnx.proto.ModelProto.decodeText(reader);
                    this.format = 'ONNX' + (this.model.ir_version ? ' v' + this.model.ir_version.toString() : '');
                } catch (error) {
                    const message = error && error.message ? error.message : error.toString();
                    throw new onnx.Error('File text format is not onnx.ModelProto (' + message.replace(/\.$/, '') + ').');
                }
                break;
            }
            case 'binary': {
                switch (this._type) {
                    case 'tensor': {
                        // TensorProto
                        // input_0.pb, output_0.pb
                        try {
                            const reader = protobuf.BinaryReader.open(stream);
                            const tensor = onnx.proto.TensorProto.decode(reader);
                            tensor.name = tensor.name || this._context.identifier;
                            const attribute = new onnx.proto.AttributeProto();
                            attribute.name = 'value';
                            attribute.type = onnx.AttributeType.TENSOR;
                            attribute.t = tensor;
                            const node = new onnx.proto.NodeProto();
                            node.op_type = 'Constant';
                            node.attribute = [ attribute ];
                            const graph = new onnx.proto.GraphProto();
                            graph.node = [ node ];
                            this.model = new onnx.proto.ModelProto();
                            this.model.graph = graph;
                            this.format = 'ONNX Tensor';
                        } catch (error) {
                            const message = error && error.message ? error.message : error.toString();
                            throw new onnx.Error('File format is not onnx.TensorProto (' + message.replace(/\.$/, '') + ').');
                        }
                        break;
                    }
                    case 'graph': {
                        // GraphProto
                        try {
                            const reader = protobuf.BinaryReader.open(stream);
                            this.model = new onnx.proto.ModelProto();
                            this.model.graph = onnx.proto.GraphProto.decode(reader);
                            this.format = 'ONNX';
                        } catch (error) {
                            const message = error && error.message ? error.message : error.toString();
                            throw new onnx.Error('File format is not onnx.GraphProto (' + message.replace(/\.$/, '') + ').');
                        }
                        break;
                    }
                    case 'model': {
                        // ModelProto
                        try {
                            const reader = protobuf.BinaryReader.open(stream);
                            this.model = onnx.proto.ModelProto.decode(reader);
                            this.format = 'ONNX' + (this.model.ir_version ? ' v' + this.model.ir_version.toString() : '');
                        } catch (error) {
                            const message = error && error.message ? error.message : error.toString();
                            throw new onnx.Error('File format is not onnx.ModelProto (' + message.replace(/\.$/, '') + ').');
                        }
                        break;
                    }
                    default: {
                        throw new onnx.Error('Unsupported ONNX format type.');
                    }
                }
                break;
            }
            default: {
                throw new onnx.Error('Unsupported ONNX format encoding.');
            }
        }
    }
};

onnx.OrtReader = class {

    static open(context) {
        const identifier = context.identifier;
        const extension = identifier.split('.').pop().toLowerCase();
        const stream = context.stream;
        if (stream && stream.length >= 8) {
            const buffer = stream.peek(Math.min(32, stream.length));
            const reader = flatbuffers.BinaryReader.open(buffer);
            const identifier = reader.identifier;
            if (identifier === 'ORTM') {
                return new onnx.OrtReader(context);
            }
            if (extension === 'ort') {
                const signature = [ 0x08, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00 ];
                if (signature.length <= stream.length && stream.peek(signature.length).every((value, index) => value === signature[index])) {
                    return new onnx.OrtReader(context);
                }
            }
        }
        return null;
    }

    constructor(context) {
        this._context = context;
    }

    async read() {
        await this._context.require('./onnx-schema');
        onnx.schema = flatbuffers.get('ort').onnxruntime.fbs;
        try {
            const stream = this._context.stream;
            this._graphs = new Set();
            const reader = flatbuffers.BinaryReader.open(stream);
            const session = onnx.schema.InferenceSession.create(reader);
            this.model = session.model;
        } catch (error) {
            const message = error && error.message ? error.message : error.toString();
            throw new onnx.Error('File format is not ort.Model (' + message.replace(/\.$/, '') + ').');
        }
        const tensor_shape = (value) => {
            if (value && value.dim && Array.isArray(value.dim)) {
                for (const dimension of value.dim) {
                    switch (dimension.value.dim_type) {
                        case 0:
                            return {};
                        case 1:
                            dimension.dim_value = dimension.value.dim_value;
                            delete dimension.value;
                            break;
                        case 2:
                            dimension.dim_param = dimension.value.dim_param;
                            delete dimension.value.dim_param;
                            break;
                        default:
                            throw new onnx.Error("Unknown shape dimension '" + JSON.stringify(dimension.value) + "'.");
                    }
                }
            }
            return value;
        };
        /* eslint-disable no-use-before-define */
        const node = (value) => {
            value.input = value.inputs;
            value.output = value.outputs;
            value.attribute = value.attributes.map((attribute) => {
                const type = attribute.type;
                if (type === onnx.AttributeType.GRAPH) {
                    graph(attribute.g);
                } else if (type === onnx.AttributeType.GRAPHS) {
                    for (const graph of attribute.graphs) {
                        graph(graph);
                    }
                } else if (type === onnx.AttributeType.TYPE_PROTO) {
                    attribute.tp = type(attribute.tp);
                } else if (type === onnx.AttributeType.TYPE_PROTOS) {
                    attribute.type_protos = attribute.type_protos.map((type) => type(type));
                }
                return attribute;
            });
            delete value.inputs;
            delete value.outputs;
            delete value.attributes;
            return value;
        };
        const tensor_type = (value) => {
            value.shape = tensor_shape(value.shape);
            return value;
        };
        const sequence_type = (value) => {
            value.shape = type(value.elem_type);
            return value;
        };
        const map_type = (value) => {
            value.value_type = type(value.value_type);
            return value;
        };
        /* eslint-enable no-use-before-define */
        const type = (value) => {
            if (value) {
                const type = value.value;
                if (type && type instanceof onnx.schema.TensorTypeAndShape) {
                    value.tensor_type = tensor_type(type);
                    return value;
                }
                if (type && type instanceof onnx.schema.SequenceType) {
                    value.sequence_type = sequence_type(type);
                    return value;
                }
                if (type && type instanceof onnx.schema.MapType) {
                    value.map_type = map_type(type);
                    return value;
                }
                throw new onnx.Error("Unsupported type value '" + JSON.stringify(value.value));
            }
            return null;
        };
        const graph = (value) => {
            if (this._graphs.has(value)) {
                return;
            }
            this._graphs.add(value);
            value.name = this._graphs.size.toString();
            value.node = value.nodes.map((value) => node(value));
            delete value.nodes;
            value.value_info = value.node_args.map((valueInfo) => {
                return {
                    name: valueInfo.name,
                    doc_string: valueInfo.doc_string,
                    type: type(valueInfo.type)
                };
            });
            delete value.node_args;
            const value_info = new Map(value.value_info.map((entry) => [ entry.name, entry ]));
            value.input = value.inputs.map((input) => {
                return value_info.has(input) ? value_info.get(input) : { name: input };
            });
            delete value.inputs;
            value.output = value.outputs.map((output) => {
                return value_info.has(output) ? value_info.get(output) : { name: output };
            });
            delete value.outputs;
            value.initializer = value.initializers.map((tensor) => {
                tensor.data_location = onnx.DataLocation.DEFAULT;
                return tensor;
            });
            delete value.initializers;
            value.sparse_initializer = value.sparse_initializers.map((tensor) => {
                tensor.values.data_location = onnx.DataLocation.DEFAULT;
                tensor.indices.data_location = onnx.DataLocation.DEFAULT;
                return tensor;
            });
            delete value.sparse_initializers;
        };
        graph(this.model.graph);
        this.model.graph.doc_string = this.model.graph_doc_string;
        delete this.model.graph_doc_string;
        this.format = 'ONNX Runtime' + (this.model.ir_version ? ' v' + this.model.ir_version.toString() : '');
    }
};

onnx.JsonReader = class {

    static open(context) {
        const obj = context.open('json');
        if (obj && (obj.irVersion !== undefined || (obj.graph && Array.isArray(obj.graph.node)))) {
            return new onnx.JsonReader(obj);
        }
        return null;
    }

    constructor(obj) {
        this.model = obj;
        this._attributeTypes = new Map(Object.entries(onnx.AttributeType));
    }

    async read() {
        const tensor_shape = (value) => {
            if (Array.isArray(value.dim)) {
                for (const dimension of value.dim) {
                    if (dimension.dimValue !== undefined) {
                        dimension.dim_value = parseInt(dimension.dimValue, 10);
                        delete dimension.dimValue;
                    } else if (dimension.dimParam !== undefined) {
                        dimension.dim_param = dimension.dimParam;
                        delete dimension.dimParam;
                    }
                }
            }
            return value;
        };
        const tensor_type = (value) => {
            value.elem_type = value.elemType;
            delete value.elemType;
            if (value.shape) {
                value.shape = tensor_shape(value.shape);
            }
            return value;
        };
        /* eslint-disable no-use-before-define */
        const optional_type = (value) => {
            value.elem_type = type(value.elemType);
            delete value.elemType;
            return value;
        };
        const sequence_type = (value) => {
            value.elem_type = type(value.elemType);
            delete value.elemType;
            return value;
        };
        const map_type = (value) => {
            value.key_type = value.keyType;
            delete value.keyType;
            value.value_type = type(value.valueType);
            delete value.valueType;
            return value;
        };
        const sparse_tensor_type = (value) => {
            value.elem_type = value.elemType;
            delete value.elemType;
            if (value.shape) {
                value.shape = tensor_shape(value.shape);
            }
            return value;
        };
        const type = (value) => {
            if (value.tensorType) {
                value.tensor_type = tensor_type(value.tensorType);
                delete value.tensorType;
            } else if (value.sequenceType) {
                value.sequence_type = sequence_type(value.sequenceType);
                delete value.sequenceType;
            } else if (value.optionalType) {
                value.optional_type = optional_type(value.optionalType);
                delete value.optionalType;
            } else if (value.mapType) {
                value.map_type = map_type(value.mapType);
                delete value.mapType;
            } else if (value.sparseTensorType) {
                value.sparse_tensor_type = sparse_tensor_type(value.sparseTensorType);
                delete value.sparseTensorType;
            } else {
                throw new onnx.Error("Unsupported ONNX JSON type '" + JSON.stringify(Object.keys(value)) + "'.");
            }
            return value;
        };
        const tensor = (value) => {
            value.data_type = value.dataType;
            value.dims = Array.isArray(value.dims) ? value.dims.map((dim) => parseInt(dim, 10)) : [];
            delete value.dataType;
            if (value.rawData !== undefined) {
                value.data_location = onnx.DataLocation.DEFAULT;
                const data = atob(value.rawData);
                const length = data.length;
                const array = new Uint8Array(length);
                for (let i = 0; i < length; i++) {
                    array[i] = data[i].charCodeAt(0);
                }
                value.raw_data = array;
                delete value.rawData;
            } else if (Array.isArray(value.floatData)) {
                value.data_location = onnx.DataLocation.DEFAULT;
                value.float_data = value.floatData;
                delete value.floatData;
            } else if (Array.isArray(value.int32Data)) {
                value.data_location = onnx.DataLocation.DEFAULT;
                value.int32_data = value.int32Data;
                delete value.int32Data;
            } else if (Array.isArray(value.int64Data)) {
                value.data_location = onnx.DataLocation.DEFAULT;
                value.int64_data = value.int64Data.map((value) => parseInt(value, 10));
                delete value.int64Data;
            } else {
                throw new onnx.Error("Unsupported ONNX JSON tensor data '" + JSON.stringify(value.data_type) + ".");
            }
            return value;
        };
        const sparse_tensor = (value) => {
            value.indices = tensor(value.indices);
            value.values = tensor(value.values);
            return value;
        };
        const attribute = (value) => {
            if (value.type && this._attributeTypes.has(value.type)) {
                value.type = this._attributeTypes.get(value.type);
            }
            if (value.refAttrName) {
                value.ref_attr_name = value.refAttrName;
                delete value.refAttrName;
            } else if (value.type === onnx.AttributeType.FLOATS || Array.isArray(value.floats)) {
                value.floats = value.floats.map((value) => parseFloat(value));
            } else if (value.type === onnx.AttributeType.INTS || Array.isArray(value.ints)) {
                value.ints = value.ints.map((value) => parseInt(value, 10));
            } else if (value.type === onnx.AttributeType.STRINGS || Array.isArray(value.strings)) {
                value.strings = value.strings.map((value) => atob(value));
            } else if (value.type === onnx.AttributeType.TENSORS || Array.isArray(value.tensors)) {
                value.tensors = value.tensors.map((value) => tensor(value));
            } else if (value.type === onnx.AttributeType.GRAPHS || Array.isArray(value.graphs)) {
                value.graphs = value.graphs.map((value) => graph(value));
            } else if (value.type === onnx.AttributeType.SPARSE_TENSORS || Array.isArray(value.sparseTensors)) {
                value.sparse_tensors = value.sparseTensors.map((value) => sparse_tensor(value));
                delete value.sparseTensors;
            } else if (value.type === onnx.AttributeType.FLOAT || value.f !== undefined) {
                value.f = parseFloat(value.f);
            } else if (value.type === onnx.AttributeType.INT || value.i !== undefined) {
                value.i = parseInt(value.i, 10);
            } else if (value.type === onnx.AttributeType.STRING || value.s !== undefined) {
                value.s = atob(value.s);
            } else if (value.type === onnx.AttributeType.TENSOR || value.t !== undefined) {
                value.t = tensor(value.t);
            } else if (value.type === onnx.AttributeType.GRAPH || value.g !== undefined) {
                value.g = graph(value.g);
            } else if (value.type === onnx.AttributeType.SPARSE_TENSOR || value.sparseTensor !== undefined) {
                value.sparse_tensor = sparse_tensor(value.sparseTensor);
                delete value.sparseTensor;
            } else {
                throw new onnx.Error("Unsupported ONNX JSON attribute type '" + JSON.stringify(value.type) + "'.");
            }
            return value;
        };
        const node = (value) => {
            value.op_type = value.opType;
            delete value.opType;
            value.input = Array.isArray(value.input) ? value.input : [];
            value.output = Array.isArray(value.output) ? value.output : [];
            value.attribute = Array.isArray(value.attribute) ? value.attribute.map((value) => attribute(value)) : [];
            return value;
        };
        const value_info = (value) => {
            value.type = type(value.type);
            return value;
        };
        const operator_set = (value) => {
            value.version = parseInt(value.version, 10);
            return value;
        };
        const graph = (value) => {
            value.node = value.node.map((value) => node(value));
            value.initializer = Array.isArray(value.initializer) ? value.initializer.map((value) => tensor(value)) : [];
            value.sparse_initializer = Array.isArray(value.sparseInitializer) ? value.sparseInitializer.map((value) => sparse_tensor(value)) : [];
            value.value_info = Array.isArray(value.valueInfo) ? value.valueInfo.map((value) => value_info(value)) : [];
            value.input = Array.isArray(value.input) ? value.input.map((value) => value_info(value)) : [];
            value.output = Array.isArray(value.output) ? value.output.map((value) => value_info(value)) : [];
            return value;
        };
        const func = (value) => {
            value.node = value.node.map((value) => node(value));
            value.input = Array.isArray(value.input) ? value.input : [];
            value.output = Array.isArray(value.output) ? value.output : [];
            value.attribute = Array.isArray(value.attribute) ? value.attribute : [];
            value.attribute_proto = Array.isArray(value.attributeProto) ? value.attributeProto.map((value) => attribute(value)) : [];
            delete value.attributeProto;
            if (value.docString) {
                value.doc_string = value.docString;
                delete value.docString;
            }
            return value;
        };
        /* eslint-enable no-use-before-define */
        this.model.ir_version = parseInt(this.model.irVersion, 10);
        delete this.model.irVersion;
        if (this.model.version !== undefined) {
            this.model.version = parseInt(this.model.version, 10);
        }
        if (this.model.producerName) {
            this.model.producer_name = this.model.producerName;
            delete this.model.producerName;
        }
        if (this.model.producerVersion) {
            this.model.producer_version = this.model.producerVersion;
            delete this.model.producerVersion;
        }
        if (this.model.modelVersion) {
            this.model.model_version = parseInt(this.model.modelVersion, 10);
            delete this.model.modelVersion;
        }
        if (this.model.docString) {
            this.model.doc_string = this.model.docString;
            delete this.model.docString;
        }
        this.model.graph = graph(this.model.graph);
        if (Array.isArray(this.model.opsetImport)) {
            this.model.opset_import = this.model.opsetImport.map((value) => operator_set(value));
            delete this.model.opsetImport;
        }
        if (Array.isArray(this.model.metadataProps)) {
            this.model.metadata_props = this.model.metadataProps;
            delete this.model.metadataProps;
        }
        if (Array.isArray(this.model.functions)) {
            this.model.functions = this.model.functions.map((value) => func(value));
        }
        this.format = 'ONNX JSON' + (this.model.ir_version ? ' v' + this.model.ir_version.toString() : '');
    }
};

onnx.TextReader = class {

    static open(context) {
        try {
            const stream = context.stream;
            if (stream && stream.length > 0 && (stream.peek(1)[0] < 0x80 || stream.peek(1)[0] >= 0xFE)) {
                const reader = text.Reader.open(stream);
                const lines = [];
                for (let i = 0; i < 32; i++) {
                    const line = reader.read();
                    if (line === undefined) {
                        break;
                    }
                    lines.push(line);
                }
                const content = lines.join('\n');
                if (/^\s*<\s*ir_version\s*:/m.exec(content) ||
                    /^\s*[a-zA-Z][a-zA-Z0-9]*\s*\(.*\)\s=>\s\(/m.exec(content)) {
                    return new onnx.TextReader(context);
                }
            }
        } catch (err) {
            // continue regardless of error
        }
        return null;
    }

    constructor(context) {
        this._context = context;
        this._dataTypes = new Map(Object.entries(onnx.DataType).map((entry) => [ entry[0].toLowerCase(), entry[1] ]));
        this._attributeTypes = new Map(Object.entries(onnx.AttributeType).map((entry) => [ entry[0].toLowerCase(), entry[1] ]));
    }

    async read() {
        await this._context.require('./onnx-proto');
        onnx.proto = protobuf.get('onnx').onnx;
        try {
            const stream = this._context.stream;
            this._decoder = text.Decoder.open(stream);
            this._position = 0;
            this._char = this._decoder.decode();
            this.model = this._model();
            this.format = 'ONNX Text' + (this.model.ir_version ? ' v' + this.model.ir_version.toString() : '');
            delete this._decoder;
            delete this._position;
            delete this._char;
        } catch (error) {
            const message = error && error.message ? error.message : error.toString();
            throw new onnx.Error('File format is not onnx.ModelProto (' + message.replace(/\.$/, '') + ').');
        }
    }

    _seek(position) {
        this._decoder.position = position;
        this._char = '';
        this._next();
    }

    _model() {
        this._whitespace();
        const model = new onnx.proto.ModelProto();
        if (this._match('<')) {
            do {
                const keyword = this._identifier();
                this._expect(':');
                switch (keyword) {
                    case 'ir_version':
                    case 'model_version':
                        model[keyword] = this._integer();
                        break;
                    case 'opset_import':
                        model[keyword] = this._operatorSetId();
                        break;
                    case 'producer_name':
                    case 'producer_version':
                    case 'domain':
                    case 'doc_string':
                        model[keyword] = this._string();
                        break;
                    case 'metadata_props':
                        this._expect('[');
                        if (!this._match(']')) {
                            do {
                                const entry = new onnx.proto.StringStringEntryProto();
                                entry.key = this._string();
                                this._expect(':');
                                entry.value = this._string();
                                model.metadata_props.push(entry);
                            } while (this._match(','));
                            this._expect(']');
                        }
                        break;
                    default:
                        this._throw("Unknown keyword '" + keyword + "'.");
                        break;
                }
            } while (this._match(','));
            this._expect('>');
        }
        model.graph = this._graph();
        this._whitespace();
        while (this._char !== undefined) {
            const func = this._function();
            if (func) {
                model.functions.push(func);
            }
            this._whitespace();
        }
        return model;
    }

    _graph() {
        const graph = new onnx.proto.GraphProto();
        graph.name = this._identifier();
        if (this._match('(')) {
            if (!this._match(')')) {
                do {
                    const valueInfo = this._valueInfo();
                    if (this._match('=')) {
                        const tensor = this._tensor(valueInfo.type);
                        tensor.name = valueInfo.name;
                        graph.initializer.push(tensor);
                    }
                    graph.input.push(valueInfo);
                }
                while (this._match(','));
                this._expect(')');
            }
        }
        this._expect('=>');
        graph.output = this._valueInfoList();
        if (this._match('<')) {
            if (!this._match('>')) {
                do {
                    const valueInfo = this._valueInfo();
                    if (this._match('=')) {
                        const tensor = this._tensor(valueInfo.type);
                        tensor.name = valueInfo.name;
                        graph.initializer.push(tensor);
                    } else {
                        graph.value_info.push(valueInfo);
                    }
                }
                while (this._match(','));
                this._expect('>');
            }
        }
        graph.node = this._nodeList();
        return graph;
    }

    _nodeList() {
        const list = [];
        this._expect('{');
        while (!this._match('}')) {
            list.push(this._node());
        }
        return list;
    }

    _node() {
        const node = new onnx.proto.NodeProto();
        node.output = this._identifierList();
        this._expect('=');
        let identifier = this._identifier();
        let domain = '';
        while (this._match('.')) {
            if (domain) {
                domain += '.';
            }
            domain += identifier;
            identifier = this._identifier();
        }
        node.domain = domain;
        node.op_type = identifier;
        node.attribute = this._attributeList();
        this._expect('(');
        node.input = this._identifierList();
        this._expect(')');
        if (!node.attribute || node.attribute.length === 0) {
            node.attribute = this._attributeList();
        }
        return node;
    }

    _attributeList() {
        const list = [];
        if (this._match('<')) {
            do {
                list.push(this._attribute());
            }
            while (this._match(','));
            this._expect('>');
        }
        return list;
    }

    _attribute() {
        const attribute = new onnx.proto.AttributeProto();
        attribute.name = this._identifier();
        if (this._match(':')) {
            const type = this._identifier();
            if (!this._attributeTypes.has(type)) {
                this._throw("Unexpected attribute type '" + type + "'.");
            }
            attribute.type = this._attributeTypes.get(type);
        }
        this._expect('=');
        if (this._match('[')) {
            const list = [];
            if (!this._match(']')) {
                do {
                    list.push(this._literal());
                }
                while (this._match(','));
                this._expect(']');
            }
            if (list.every((value) => typeof value === 'string')) {
                attribute.type = onnx.AttributeType.STRINGS;
                attribute.strings = list;
            } else if (list.every((value) => typeof value === 'number' && Number.isInteger(value))) {
                attribute.type = onnx.AttributeType.INTS;
                attribute.ints = list;
            } else if (list.every((value) => typeof value === 'number')) {
                attribute.type = onnx.AttributeType.FLOATS;
                attribute.floats = list;
            } else {
                this._throw("Unexpected value '" + JSON.stringify(list) + "'.");
            }
        } else if ((this._char >= 'a' && this._char <= 'z') || (this._char >= 'A' && this._char <= 'Z') || this._char === '_') {
            const identifier = this._identifier();
            if (this._dataTypes.has(identifier)) {
                attribute.type = onnx.AttributeType.TENSOR;
                if (!this._dataTypes.has(identifier)) {
                    this._throw("Unexpected type '" + identifier + "'.");
                }
                const type = this._type(this._dataTypes.get(identifier));
                if (!type.tensor_type.elem_type) {
                    this._throw('Expected tensor data type.');
                }
                if (!type.tensor_type.shape || !type.tensor_type.shape.dim) {
                    this._throw('Expected tensor shape.');
                }
                attribute.t = this._tensor(type);
            } else {
                attribute.type = onnx.AttributeType.GRAPH;
                attribute.g = this._graph();
            }
        } else if (this._match('@')) {
            attribute.ref_attr_name = this._identifier();
        } else {
            const value = this._literal();
            switch (typeof value) {
                case 'number':
                    if (Number.isInteger(value)) {
                        attribute.type = onnx.AttributeType.INT;
                        attribute.i = value;
                    } else {
                        attribute.type = onnx.AttributeType.FLOAT;
                        attribute.f = value;
                    }
                    break;
                case 'string':
                    attribute.type = onnx.AttributeType.STRING;
                    attribute.s = value;
                    break;
                default: {
                    this._throw("Unexpected value '" + JSON.stringify(value) + "'.");
                }
            }
        }
        return attribute;
    }

    _valueInfoList() {
        const list = [];
        this._expect('(');
        if (!this._match(')')) {
            do {
                list.push(this._valueInfo());
            } while (this._match(','));
            this._expect(')');
        }
        return list;
    }

    _valueInfo() {
        const valueInfo = new onnx.proto.ValueInfoProto();
        let identifier = this._identifier();
        if (this._dataTypes.has(identifier)) {
            valueInfo.type = this._type(this._dataTypes.get(identifier));
            identifier = this._identifier();
        }
        valueInfo.name = identifier;
        return valueInfo;
    }

    _type(elem_type) {
        const type = new onnx.proto.TypeProto();
        type.tensor_type = new onnx.proto.TypeProto.Tensor();
        type.tensor_type.elem_type = elem_type;
        if (this._match('[')) {
            if (!this._match(']')) {
                type.tensor_type.shape = this._shape();
                this._expect(']');
            }
        } else {
            type.tensor_type.shape = new onnx.proto.TensorShapeProto();
        }
        return type;
    }

    _shape() {
        const shape = new onnx.proto.TensorShapeProto();
        do {
            const dimension = new onnx.proto.TensorShapeProto.Dimension();
            if (!this._match('?')) {
                const identifier = this._identifier(true);
                if (identifier) {
                    dimension.dim_param = identifier;
                } else {
                    dimension.dim_value = this._integer();
                }
            }
            shape.dim.push(dimension);
        }
        while (this._match(','));
        return shape;
    }

    _tensor(type) {
        const tensor = new onnx.proto.TensorProto();
        if (!type.tensor_type || !type.tensor_type.elem_type) {
            this._throw('Expected tensor type.');
        }
        if (!type.tensor_type.shape || !type.tensor_type.shape.dim || !type.tensor_type.shape.dim.every((dim) => dim.dim_value)) {
            this._throw('Expected numeric tensor shape.');
        }
        const elem_type = type.tensor_type.elem_type;
        tensor.data_type = elem_type;
        tensor.dims = type.tensor_type.shape.dim.map((dim) => dim.dim_value);
        this._match('=');
        this._expect('{');
        if (!this._match('}')) {
            do {
                switch (elem_type) {
                    case onnx.DataType.INT8:
                    case onnx.DataType.INT16:
                    case onnx.DataType.INT32:
                    case onnx.DataType.UINT8:
                    case onnx.DataType.UINT16:
                    case onnx.DataType.BOOL:
                        tensor.int32_data.push(this._integer());
                        break;
                    case onnx.DataType.INT64:
                        tensor.int64_data.push(this._integer());
                        break;
                    case onnx.DataType.UINT32:
                    case onnx.DataType.UINT64:
                        tensor.uint64_data.push(this._integer());
                        break;
                    case onnx.DataType.FLOAT:
                        tensor.float_data.push(this._float());
                        break;
                    case onnx.DataType.DOUBLE:
                        tensor.double_data.push(this._float());
                        break;
                    case onnx.DataType.STRING:
                        tensor.string_data.push(this.string());
                        break;
                    default:
                        return this._throw("Unsupported tensor element type '" + elem_type.toString() + "'.");
                }
            } while (this._match(','));
            this._expect('}');
        }
        return tensor;
    }

    _function() {
        const func = new onnx.proto.FunctionProto();
        if (this._match('<')) {
            do {
                const keyword = this._identifier();
                this._expect(':');
                switch (keyword) {
                    case 'opset_import':
                        func[keyword] = this._operatorSetId();
                        break;
                    case 'domain':
                    case 'doc_string':
                        func[keyword] = this._string();
                        break;
                    default:
                        this._throw("Unknown keyword '" + keyword + "'.");
                        break;
                }
            }
            while (this._match(','));
            this._expect('>');
        }
        func.name = this._identifier();
        if (this._match('<')) {
            func.attribute = this._identifierList();
            this._expect('>');
        }
        if (this._match('(')) {
            func.input = this._identifierList();
            this._expect(')');
        }
        this._expect('=>');
        if (this._match('(')) {
            func.output = this._identifierList();
            this._expect(')');
        }
        func.node = this._nodeList();
        return func;
    }

    _identifierList() {
        const list = [];
        const identifier = this._identifier(true);
        if (identifier) {
            list.push(identifier);
            while (this._match(',')) {
                list.push(this._identifier());
            }
        }
        return list;
    }

    _identifier(optional) {
        this._whitespace();
        const value = [];
        if ((this._char >= 'a' && this._char <= 'z') || (this._char >= 'A' && this._char <= 'Z')) {
            value.push(this._char);
            this._next();
            while ((this._char >= 'a' && this._char <= 'z') || (this._char >= 'A' && this._char <= 'Z') || (this._char >= '0' && this._char <= '9') || this._char === '_') {
                value.push(this._char);
                this._next();
            }
        }
        if (optional !== true && value.length == 0) {
            this._throw('Identifier expected.');
        }
        return value.join('');
    }

    _literal() {
        this._whitespace();
        let decimal_point = false;
        if (this._char === '"') {
            const value = [];
            this._next();
            while (this._char !== undefined && this._char !== '"') {
                value.push(this._char);
                this._next();
            }
            if (this._char !== undefined) {
                this._next();
            }
            return value.join('');
        } else if ((this._char >= '0' && this._char <= '9') || this._char === '-') {
            const value = [ this._char ];
            this._next();
            while ((this._char >= '0' && this._char <= '9') || this._char === '.') {
                if (this._char === '.') {
                    if (decimal_point) {
                        this._throw();
                    }
                    decimal_point = true;
                }
                value.push(this._char);
                this._next();
            }
            if (value.length === 0) {
                this._throw('Value expected.');
            }
            if (this._char === 'e' || this._char === 'E') {
                decimal_point = true;
                value.push(this._char);
                this._next();
                if (this._char === '+' || this._char === '-') {
                    value.push(this._char);
                    this._next();
                }
                while ((this._char >= '0' && this._char <= '9')) {
                    value.push(this._char);
                    this._next();
                }
            }
            return decimal_point ? Number.parseFloat(value.join('')) : Number.parseInt(value.join(''), 10);
        }
        return undefined;
    }

    _integer() {
        const value = this._literal();
        if (!Number.isInteger(value)) {
            this._throw('Integer value expected.');
        }
        return value;
    }

    _float() {
        const value = this._literal();
        if (typeof value !== 'number') {
            this._throw('Float value expected.');
        }
        return value;
    }

    _string() {
        const value = this._literal();
        if (typeof value !== 'string') {
            this._throw('String value expected.');
        }
        return value;
    }

    _operatorSetId() {
        const list = [];
        this._expect('[');
        if (!this._match(']')) {
            do {
                const value = new onnx.proto.OperatorSetIdProto();
                value.domain = this._string();
                this._expect(':');
                value.version = this._integer();
                list.push(value);
            }
            while (this._match(','));
            this._expect(']');
        }
        return list;
    }

    _match(value) {
        this._whitespace();
        if (this._char !== value[0]) {
            return false;
        }
        if (value.length === 1) {
            this._next();
            return true;
        }
        const position = this._position;
        for (let i = 0; i < value.length; i++) {
            if (this._char !== value[i]) {
                this._seek(position);
                return false;
            }
            this._next();
        }
        return true;
    }

    _expect(value) {
        if (!this._match(value)) {
            this._unexpected();
        }
        return true;
    }

    _whitespace() {
        for (;;) {
            while (this._char === ' ' || this._char === '\n' || this._char === '\r' || this._char === '\t') {
                this._next();
            }
            if (this._char === undefined || this._char !== '#') {
                break;
            }
            while (this._char !== undefined && this._char !== '\n') {
                this._next();
            }
        }
    }

    _next() {
        if (this._char === undefined) {
            this._unexpected();
        }
        this._position = this._decoder.position;
        this._char = this._decoder.decode();
    }

    _unexpected() {
        let c = this._char;
        if (c === undefined) {
            throw new onnx.Error('Unexpected end of input.');
        } else if (c === '"') {
            c = 'string';
        } else if ((c >= '0' && c <= '9') || c === '-') {
            c = 'number';
        } else {
            if (c < ' ' || c > '\x7F') {
                const name = Object.keys(this._escape).filter((key) => this._escape[key] === c);
                c = (name.length === 1) ? '\\' + name : '\\u' + ('000' + c.charCodeAt(0).toString(16)).slice(-4);
            }
            c = "token '" + c + "'";
        }
        this._throw('Unexpected ' + c);
    }

    _throw(message) {
        throw new onnx.Error(message.replace(/\.$/, '') + this._location());
    }

    _location() {
        let line = 1;
        let column = 1;
        this._decoder.position = 0;
        let c;
        do {
            if (this._decoder.position === this._position) {
                return ' at ' + line.toString() + ':' + column.toString() + '.';
            }
            c = this._decoder.decode();
            if (c === '\n') {
                line++;
                column = 1;
            } else {
                column++;
            }
        }
        while (c !== undefined);
        return ' at ' + line.toString() + ':' + column.toString() + '.';
    }
};

onnx.PickleReader = class {

    static open(context) {
        const identifier = context.identifier;
        const extension = identifier.split('.').pop().toLowerCase();
        const stream = context.stream;
        if (extension === 'onnx' && stream && stream.length > 3) {
            const signature = stream.peek(2);
            if (signature[0] === 0x80 && signature[1] < 7) {
                return new onnx.PickleReader();
            }
        }
        return undefined;
    }

    async read() {
        throw new onnx.Error('Unsupported Pickle content.');
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
