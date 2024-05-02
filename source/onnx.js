
import * as protobuf from './protobuf.js';
import * as text from './text.js';

const onnx = {};

onnx.ModelFactory = class {

    match(context) {
        const identifier = context.identifier;
        const extensions = [
            'saved_model.pb', 'predict_net.pb', 'init_net.pb',
            'predict_net.pbtxt', 'init_net.pbtxt', 'predict_net.prototxt', 'init_net.prototxt'
        ];
        if (!extensions.some((extension) => identifier.endsWith(extension))) {
            const entries = [
                onnx.OrtReader,
                onnx.ProtoReader,
                onnx.TextReader,
                onnx.JsonReader,
                onnx.PickleReader,
                onnx.DataReader
            ];
            for (const entry of entries) {
                const reader = entry.open(context);
                if (reader) {
                    context.type = reader.name;
                    context.target = reader;
                    break;
                }
            }
        }
    }

    async open(context) {
        const target = context.target;
        await target.read();
        const metadata = await onnx.Metadata.open(context);
        return new onnx.Model(metadata, target);
    }

    filter(context, type) {
        return context.type !== 'onnx.proto' || type !== 'onnx.data';
    }
};

onnx.Model = class {

    constructor(metadata, target) {
        const model = target.model;
        this._graphs = [];
        this._format = target.format;
        this._producer = model.producer_name && model.producer_name.length > 0 ? model.producer_name + (model.producer_version && model.producer_version.length > 0 ? ` ${model.producer_version}` : '') : null;
        this._domain = model.domain;
        this._version = typeof model.model_version === 'number' || typeof model.model_version === 'bigint' ? model.model_version.toString() : '';
        this._description = model.doc_string;
        this._metadata = [];
        this._imports = null;
        const imports = new Map();
        if (model.opset_import && model.opset_import.length > 0) {
            for (const opset_import of model.opset_import) {
                const domain = opset_import.domain || 'ai.onnx';
                const version = typeof opset_import.version === 'bigint' ? Number(opset_import.version) : opset_import.version;
                if (!imports.has(domain) || imports.get(domain) > version) {
                    imports.set(domain, version);
                }
            }
            this._imports = Array.from(imports).map(([name, version]) => `${name} v${version}`);
        }
        if (imports.size === 0) {
            imports.set('ai.onnx', 1);
            imports.set('ai.onnx.ml', 1);
        }
        let imageFormat = '';
        const metadata_props = model.metadata_props;
        if (metadata_props) {
            const metadata = new Map(metadata_props.map((entry) => [entry.key, entry.value]));
            const converted_from = metadata.get('converted_from');
            if (converted_from) {
                this.source = converted_from;
            }
            const author = metadata.get('author');
            if (author) {
                this._metadata.push(new onnx.Argument('author', author));
            }
            const company = metadata.get('company');
            if (company) {
                this._metadata.push(new onnx.Argument('company', company));
            }
            let license = metadata.get('license');
            const license_url = metadata.get('license_url');
            if (license_url) {
                license = `<a href='${license_url}'>${license ? license : license_url}</a>`;
            }
            if (license) {
                this._metadata.push(new onnx.Argument('license', license));
            }
            metadata.delete('author');
            metadata.delete('company');
            metadata.delete('converted_from');
            metadata.delete('license');
            metadata.delete('license_url');
            const imageMetadata = {};
            for (const [name, value] of metadata) {
                switch (name) {
                    case 'Image.BitmapPixelFormat':
                    case 'Image.ColorSpaceGamma':
                    case 'Image.NominalPixelRange':
                        imageMetadata[name] = value;
                        break;
                    default:
                        this._metadata.push(new onnx.Argument(name, value));
                        break;
                }
            }
            imageFormat = [imageMetadata['Image.BitmapPixelFormat'], imageMetadata['Image.ColorSpaceGamma'], imageMetadata['Image.NominalPixelRange']].filter((item) => item);
        }
        const context = new onnx.Context.Model(metadata, target.locations, imageFormat, imports, model.functions);
        const graph = new onnx.Graph(context, model.graph);
        this._graphs = [graph];
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

    get source() {
        return this._source;
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
        context = new onnx.Context.Graph(context, graph);
        if (Array.isArray(graph.quantization_annotation)) {
            for (const tensor_annotation of graph.quantization_annotation) {
                const tensor = context.tensor(tensor_annotation.tensor_name);
                tensor.annotation = new Map();
                for (const entry of tensor_annotation.quant_parameter_tensor_names) {
                    tensor.annotation.set(entry.key, entry.value);
                }
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
        const inference = new onnx.Inference(graph.node);
        for (const output of graph.output) {
            inference.infer(output.name);
        }
        context.push(graph.node, graph.input, graph.output);
        this._nodes = context.pop();
        for (const input of graph.input) {
            const value = context.value(input.name);
            if (!value.initializer) {
                this._inputs.push(new onnx.Argument(input.name, [value]));
            }
        }
        for (const output of graph.output) {
            const value = context.value(output.name);
            if (!value.initializer) {
                this._outputs.push(new onnx.Argument(output.name, [value]));
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
        return `graph(${this.name})`;
    }
};

onnx.Argument = class {

    constructor(name, value, type, description, visible) {
        this.name = name;
        this.value = value;
        this.type = type || null;
        this.description = description || null;
        this.visible = visible !== false;
    }
};

onnx.Value = class {

    constructor(name, type, initializer, annotation, description) {
        if (typeof name !== 'string') {
            throw new onnx.Error(`Invalid value identifier '${JSON.stringify(name)}'.`);
        }
        this._name = name;
        this._type = type || null;
        this._initializer = initializer || null;
        this._description = description || '';
        this._quantization = annotation ? { type: 'annotation', value: annotation } : null;
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
        return this._quantization;
    }

    get initializer() {
        return this._initializer;
    }
};

onnx.Node = class {

    constructor(context, node, inputs, outputs) {
        const domain = node.domain || 'ai.onnx';
        const op_type = node.op_type;
        const overload = node.overload || '';
        const attributes = node.attribute || [];
        const metadata_props = node.metadata_props || [];
        this.type = context.type(domain, op_type, overload);
        if (!this.type || (this.type.module !== domain && !(this.type instanceof onnx.Function))) {
            this.type = { ...this.type };
            this.type.name = op_type;
            this.type.module = domain;
            this.type.overload = overload;
            this.type.identifier = overload ? `${op_type}:${overload}` : `${op_type}`;
        }
        this.name = node.name || '';
        this.description = node.doc_string || '';
        this.inputs = inputs || [];
        this.outputs = outputs || [];
        this.attributes = attributes.map((attribute) => {
            const name = attribute.name;
            let type = null;
            let value = null;
            let visible = true;
            if (attribute.ref_attr_name) {
                value = attribute.ref_attr_name;
                type = 'reference';
            } else {
                switch (attribute.type) {
                    case onnx.AttributeType.UNDEFINED:
                        break;
                    case onnx.AttributeType.FLOAT:
                        value = attribute.f;
                        type = 'float32';
                        break;
                    case onnx.AttributeType.INT:
                        value = BigInt(attribute.i);
                        type = 'int64';
                        break;
                    case onnx.AttributeType.STRING:
                        value = op_type === 'Int8GivenTensorFill' ? Array.from(attribute.s) : context.decodeText(attribute.s);
                        type = 'string';
                        break;
                    case onnx.AttributeType.TENSOR:
                        value = new onnx.Tensor(context, attribute.t);
                        type = 'tensor';
                        break;
                    case onnx.AttributeType.GRAPH:
                        value = context.graph(attribute.g);
                        type = 'graph';
                        break;
                    case onnx.AttributeType.FLOATS:
                        value = ArrayBuffer.isView(attribute.floats) ? Array.from(attribute.floats) : attribute.floats;
                        type = 'float32[]';
                        break;
                    case onnx.AttributeType.INTS:
                        value = ArrayBuffer.isView(attribute.ints) ? Array.from(attribute.ints) : attribute.ints.map((value) => BigInt(value));
                        type = 'int64[]';
                        break;
                    case onnx.AttributeType.STRINGS:
                        value = attribute.strings.map((s) => context.decodeText(s));
                        type = 'string[]';
                        break;
                    case onnx.AttributeType.TENSORS:
                        value = attribute.tensors.map((tensor) => new onnx.Tensor(context, tensor));
                        type = 'tensor[]';
                        break;
                    case onnx.AttributeType.GRAPHS:
                        value = attribute.graphs.map((graph) => context.graph(graph));
                        type = 'graph[]';
                        break;
                    case onnx.AttributeType.SPARSE_TENSOR:
                        value = new onnx.Tensor(context, attribute.sparse_tensor);
                        type = 'tensor';
                        break;
                    case onnx.AttributeType.SPARSE_TENSORS:
                        value = attribute.sparse_tensors.map((tensor) => new onnx.Tensor(context, tensor));
                        type = 'tensor[]';
                        break;
                    case onnx.AttributeType.TYPE_PROTO:
                        value = context.createType(attribute.tp);
                        type = 'type';
                        break;
                    case onnx.AttributeType.TYPE_PROTOS:
                        value = attribute.type_protos.map((type) => context.createType(type));
                        type = 'type[]';
                        break;
                    default:
                        throw new onnx.Error(`Unsupported attribute type '${attribute.type}'.`);
                }
                const metadata = context.attribute(domain, op_type, overload, attribute.name);
                if (metadata) {
                    if (metadata.default !== undefined) {
                        const defaultValue = type === 'int64' ? BigInt(metadata.default) : metadata.default;
                        if (value === defaultValue) {
                            visible = false;
                        }
                    }
                    if (metadata.type === 'DataType') {
                        type = metadata.type;
                        value = context.createDataType(value);
                    }
                }
            }
            return new onnx.Argument(name, value, type, attribute.doc_string, visible);
        });
        this.metadata = metadata_props.map((metadata) => {
            return new onnx.Argument(metadata.key, metadata.value);
        });
        this.chain = [];
        const identifier = domain ? `${domain}.${op_type}` : op_type;
        if (identifier === 'com.microsoft.FusedConv') {
            const activation = attributes.find((attribute) => attribute.name === 'activation');
            if (activation) {
                const type = context.decodeText(activation.s);
                const node = new onnx.Node(context, { op_type: type });
                this.chain.push(node);
            }
        }
    }
};

onnx.Group = class {

    constructor(name, groups) {
        this._type = { name: 'Scope' };
        this._name = name;
        this._nodes = [];
        for (const [key, value] of groups) {
            if (key === '') {
                for (const node of value) {
                    this._nodes.push(node);
                }
            } else {
                this._nodes.push(new onnx.Group(name === '' ? key : `${name}/${key}`, value));
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
        this._inputs = [new onnx.Argument('inputs', inputs)];
        this._outputs = [new onnx.Argument('outputs', outputs)];
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
            this._type = context.createTensorType(tensor.values.data_type, tensor.dims, 'sparse');
            this._location = context.createLocation(tensor.values.data_location);
            this._values = new onnx.Tensor(context, tensor.values);
            this._indices = new onnx.Tensor(context, tensor.indices);
        } else {
            this._name = tensor.name || '';
            this._type = context.createTensorType(tensor.data_type, tensor.dims);
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
                        case onnx.DataType.UINT4:
                        case onnx.DataType.INT4:
                            if (tensor.int32_data && tensor.int32_data.length > 0) {
                                this._data = new Uint8Array(Array.from(tensor.int32_data));
                                this._encoding = '<';
                            }
                            break;
                        default:
                            throw new onnx.Error(`Unsupported tensor data type '${tensor.data_type}'.`);
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
        this._dimensions = dimensions.map((dim) => typeof dim === 'bigint' ? dim.toNumber() : dim);
    }

    get dimensions() {
        return this._dimensions;
    }

    toString() {
        if (!this._dimensions || this._dimensions.length === 0) {
            return '';
        }
        return `[${this._dimensions.map((dim) => dim || Number.isInteger(dim) ? dim.toString() : '?').join(',')}]`;
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
        return `sequence<${elementType}>`;
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
        return `map<${this._keyType},${this._valueType}>`;
    }
};

onnx.OpaqueType = class {

    constructor(domain, name) {
        this._domain = domain;
        this._name = name;
    }

    toString() {
        const name = (this._domain ? (`${this._domain}.`) : '') + this._name;
        return `opaque<${name}>`;
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
        return `optional<${this._type}>`;
    }
};

onnx.Function = class {

    constructor(context, func) {
        this.type = 'function';
        this.name = func.name;
        this.module = func.domain;
        this.overload = func.overload || '';
        this.identifier = this.overload ? `${this.name}:${this.overload}` : this.name;
        this.description = func.doc_string;
        this.inputs = [];
        this.outputs = [];
        this.attributes = func.attribute.map((attribtue) => {
            return { name: attribtue };
        });
        context = new onnx.Context.Graph(context, func);
        func.input = func.input.map((input) => context.tensor(input));
        func.output = func.output.map((output) => context.tensor(output));
        context.push(func.node, func.input, func.output);
        this.nodes = context.pop();
        for (const input of func.input) {
            const value = context.value(input.name);
            if (!value.initializer) {
                this.inputs.push(new onnx.Argument(input.name, [value]));
            }
        }
        for (const output of func.output) {
            const value = context.value(output.name);
            if (!value.initializer) {
                this.outputs.push(new onnx.Argument(output.name, [value]));
            }
        }
    }
};

onnx.Context = class {};

onnx.Context.Model = class {

    constructor(metadata, locations, imageFormat, imports, functions) {
        this._metadata = metadata;
        this._locations = locations;
        this._imageFormat = imageFormat;
        this._imports = imports;
        this._types = new Map();
        this._attributes = new Map();
        this._functions = new Map();
        for (const func of functions || []) {
            const domain = func.domain;
            const name = func.name;
            const overload = func.overload;
            const key = overload ? `${domain}:${name}:${overload}` : `${domain}:${name}`;
            if (this._functions.has(key)) {
                throw new onnx.Error(`Duplicate function identifier '${key}'.`);
            }
            this._functions.set(key, func);
        }
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
                } catch {
                    // continue regardless of error
                }
            }
        }
        return null;
    }

    initializer(/* name */) {
        return null;
    }

    type(domain, name, overload) {
        const key = overload ? `${domain}:${name}:${overload}` : `${domain}:${name}`;
        if (!this._types.has(key)) {
            let value = null;
            if (this._functions.has(key)) {
                value = this._functions.get(key);
                if (value.domain !== undefined) {
                    value = new onnx.Function(this, value);
                    this._functions.set(key, value);
                }
            }
            if (!value) {
                value = this._metadata.type(domain, name, this._imports);
            }
            this._types.set(key, value);
        }
        return this._types.get(key);
    }

    attribute(domain, type, overload, name) {
        const key = overload ? `${domain}:${type}:${overload}::${name}` : `${domain}:${type}::${name}`;
        if (!this._attributes.has(key)) {
            this._attributes.set(key, null);
            const metadata = this.type(domain, type);
            if (metadata && Array.isArray(metadata.attributes) && metadata.attributes.length > 0) {
                for (const attribute of metadata.attributes) {
                    const name = attribute.name;
                    const key = overload ? `${domain}:${type}:${overload}::${name}` : `${domain}:${type}::${name}`;
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
            const data = await context.request('onnx-metadata.json');
            onnx.Metadata._metadata = new onnx.Metadata(data);
            return onnx.Metadata._metadata;
        } catch {
            onnx.Metadata._metadata = new onnx.Metadata(null);
            return onnx.Metadata._metadata;
        }
    }

    constructor(data) {
        this._types = new Map();
        if (data) {
            const types = JSON.parse(data);
            for (const type of types) {
                if (!this._types.has(type.module)) {
                    this._types.set(type.module, new Map());
                }
                const types = this._types.get(type.module);
                if (!types.has(type.name)) {
                    types.set(type.name, []);
                }
                types.get(type.name).push(type);
            }
        }
    }

    type(domain, name, imports) {
        domain = domain || 'ai.onnx';
        let current = null;
        if (this._types.has(domain)) {
            const types = this._types.get(domain);
            if (types.has(name)) {
                for (const type of types.get(name)) {
                    const matchVersion = current ? current.version : -1;
                    const importVersion = imports.get(type.module) || 0;
                    if (importVersion >= type.version && matchVersion < type.version) {
                        current = type;
                    }
                }
            }
        }
        return current;
    }
};

onnx.Inference = class {

    constructor(nodes) {
        this._outputs = new Map();
        for (const node of nodes) {
            for (const output of node.output) {
                this._outputs.set(output.name, node);
            }
        }
    }

    infer(output) {
        if (this._outputs.has(output)) {
            let hasInputShapes = true;
            const node = this._outputs.get(output);
            for (const input of node.input) {
                if (!input.type) {
                    this.infer(input);
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
    FLOAT8E5M2FNUZ: 20,
    UINT4: 21,
    INT4: 22
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

onnx.Context.Graph = class {

    constructor(context, graph) {
        this._context = context;
        this._dataTypes = new Map(Object.entries(onnx.DataType).map(([name, value]) => [value, name.toLowerCase()]));
        this._dataTypes.set(onnx.DataType.UNDEFINED, 'undefined');
        this._dataTypes.set(onnx.DataType.BOOL, 'boolean');
        this._dataTypes.set(onnx.DataType.FLOAT, 'float32');
        this._dataTypes.set(onnx.DataType.DOUBLE, 'float64');
        this._graphs = new Map();
        this._initializers = new Map();
        this._tensors = new Map();
        this._values = new Map();
        this._groups = new Map();
        this._nodes = [];
        if (Array.isArray(graph.initializer)) {
            for (const initializer of graph.initializer) {
                const tensor = new onnx.Tensor(this, initializer, 'Initializer');
                this._initializers.set(initializer.name, tensor);
            }
        }
        if (Array.isArray(graph.sparse_initializer)) {
            for (const sparse_initializer of graph.sparse_initializer) {
                const tensor = new onnx.Tensor(this, sparse_initializer, 'Initializer');
                this._initializers.set(sparse_initializer.values.name, tensor);
            }
        }
        for (const node of graph.node) {
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
                    } else if (attribute.sparse_tensor) {
                        attribute.type = onnx.AttributeType.SPARSE_TENSOR;
                    } else {
                        attribute.type = onnx.AttributeType.UNDEFINED;
                    }
                }
            }
        }
    }

    type(domain, name, overload) {
        return this._context.type(domain, name, overload);
    }

    attribute(domain, type, overload, name) {
        return this._context.attribute(domain, type, overload, name);
    }

    graph(value) {
        if (!this._graphs.has(value)) {
            this._graphs.set(value, new onnx.Graph(this, value));
        }
        return this._graphs.get(value);
    }

    initializer(name) {
        if (this._initializers.has(name)) {
            return this._initializers.get(name);
        }
        return this._context.initializer(name);
    }

    tensor(name) {
        if (!this._tensors.has(name)) {
            this._tensors.set(name, { name, initializer: this.initializer(name) });
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
            this._groups.set(name, new Map([['', []]]));
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
                denotation = `Image${this._context.imageFormat ? `(${this._context.imageFormat.join(',')})` : ''}`;
                break;
            case 'AUDIO':
                denotation = 'Audio';
                break;
            case 'TEXT':
                denotation = 'Text';
                break;
            default:
                throw new onnx.Error(`Unsupported tensor type denotation '${type.denotation}'.`);
        }
        if (type.tensor_type) {
            const tensor_type = type.tensor_type;
            const shape = tensor_type.shape && tensor_type.shape.dim ? tensor_type.shape.dim.map((dim) => dim.dim_param ? dim.dim_param : dim.dim_value || null) : [];
            return this.createTensorType(tensor_type.elem_type, shape, null, denotation);
        } else if (type.sparse_tensor_type) {
            type = type.sparse_tensor_type;
            const shape = type.shape && type.shape.dim ? type.shape.dim.map((dim) => dim.dim_param ? dim.dim_param : dim.dim_value || null) : [];
            return this.createTensorType(type.elem_type, shape, 'sparse', denotation);
        } else if (type.map_type) {
            return this.createMapType(type.map_type.key_type, this.createType(type.map_type.value_type), denotation);
        } else if (type.sequence_type) {
            return new onnx.SequenceType(this.createType(type.sequence_type.elem_type), denotation);
        } else if (type.opaque_type) {
            return new onnx.OpaqueType(type.opaque_type.domain, type.opaque_type.name);
        } else if (type.optional_type) {
            return new onnx.OptionalType(this.createType(type.optional_type.elem_type), denotation);
        } else if (Object.keys(type).length === 0) {
            return null;
        }
        throw new onnx.Error(`Unsupported tensor type '${JSON.stringify(type)}'.`);
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
            if (typeof value === 'bigint') {
                value = value.toNumber();
            } else if (value && typeof value === 'string' && onnx.DataType[value.toUpperCase()] !== undefined) {
                value = onnx.DataType[value.toUpperCase()];
            } else {
                throw new onnx.Error(`Unsupported data type '${JSON.stringify(value)}'.`);
            }
        }
        if (this._dataTypes.has(value)) {
            return this._dataTypes.get(value);
        }
        throw new onnx.Error(`Unsupported data type '${JSON.stringify(value)}'.`);
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
            const domain = node.domain || 'ai.onnx';
            const op_type = node.op_type;
            const overload = node.overload || '';
            const type = this._context.type(domain, op_type, overload);
            const inputs = [];
            node.input = node.input || [];
            for (let i = 0; i < node.input.length;) {
                const input = type && type.inputs && i < type.inputs.length ? type.inputs[i] : { name: i.toString() };
                const count = input.list ? node.input.length - i : 1;
                const list = node.input.slice(i, i + count).filter((value) => value.name !== '' || value.initializer);
                const values = list.map((input) => this.value(input.name));
                const argument = new onnx.Argument(input.name, values);
                inputs.push(argument);
                i += count;
            }
            const outputs = [];
            node.output = node.output || [];
            for (let i = 0; i < node.output.length;) {
                const output = type && type.outputs && i < type.outputs.length ? type.outputs[i] : { name: i.toString() };
                const count = output.list ? node.output.length - i : 1;
                const list = node.output.slice(i, i + count).filter((value) => value.name !== '' || value.initializer);
                const values = list.map((output) => this.value(output.name));
                const argument = new onnx.Argument(output.name, values);
                outputs.push(argument);
                i += count;
            }
            node = new onnx.Node(this, node, inputs, outputs);
            this._nodes.push(node);

            // const path = (node.name || '').split('/');
            // path.pop();
            // this.group(path.join('/')).get('').push(node);
        }
    }

    pop() {
        /*
        const nodes = [];
        for (const [name, value] of this._groups) {
            if (name === '') {
                for (const node of value.get('')) {
                    nodes.push(node);
                }
                continue;
            }
            nodes.push(new onnx.Group(name, value));
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
                    for (const [key, inner] of schema) {
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
                    if (schema.every(([key, value]) => !tags.has(key) || tags.get(key) === value)) {
                        return new onnx.ProtoReader(context, 'binary', 'tensor');
                    }
                }
                // GraphProto
                if (tags.get(1) === 2) {
                    const schema = [[1,2],[2,2],[3,2],[4,2],[5,2],[6,0],[7,0],[8,2],[9,2],[10,2],[11,2],[12,2],[13,2],[14,2]];
                    if (schema.every(([key, value]) => !tags.has(key) || tags.get(key) === value)) {
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
                    if (schema.every(([key, value]) => !tags.has(key) || tags.get(key) === value)) {
                        return new onnx.ProtoReader(context, 'binary', 'model');
                    }
                }
            }
        }
        const stream = context.stream;
        if (stream && stream.length > 5) {
            const buffer = stream.peek(Math.min(stream.length, 32));
            if (buffer[0] === 0x08 && buffer[1] < 0x0B && buffer[2] === 0x12) {
                const producers = [
                    'backend-test', 'BrainwaveCompiler',
                    'caffe', 'CNTK', 'customvision', 'Cube.AI', 'cvflowbackend',
                    'dmp2x', 'dx-dl',
                    'eva_tool', 'eytorch',
                    'fuse_model',
                    'GE',
                    'horizon_nn',
                    'keras2onnx', 'Kneron', 'Kneron Piano', 'kneron_formatter', 'kneron_kl530_test_case',
                    'darknet to ONNX example',
                    'Novatek NovaOnnx Converter', 'htshinichi',
                    'MATLAB Deep Learning Toolbox Converter for ONNX Model Format', 'ML.NET', 'MVTec Software', 'Novaic',
                    'onnx-caffe2', 'onnx.compose.merge_models', 'onnx-example', 'onnx-fix-nodes', 'onnx-TIDL', 'onnx-typecast', 'onnx_test', 'onnx_tool', 'onnx_translator', 'onnx.quantize', 'onnx.utils.extract_model', 'OnnxMLTools', 'OnnxRuntime', 'onnxruntime-genai', 'onnxruntime-sample', 'onnxruntime-test', 'onnxruntime-tools', 'onnxruntime.transformers', 'optimum-onnx', 'optimum-onnxruntime',
                    'PaddlePaddle', 'pmml2onnx', 'PPL Quantization Tool', 'prune_model_weight', 'Pulsar2', 'pytorch',
                    'rk',
                    'sclblonnx', 'sequencer_save', 'SNPS MWNN Optimizer', 'SNPS MWNN Quantizer', 'SNPS NNAC Compiler', 'SNPS NNAC Legalizer', 'SNPS NNAC Optimizer', 'SNPS NNAC Quantizer', 'skl2onnx',
                    'Tencent YouTu', 'TensorRT', 'tf2onnx', 'tflite2onnx',
                    'Updated Producer',
                    'vai_q_onnx',
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
                } catch {
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
        this.name = 'onnx.proto';
        this.context = context;
        this.encoding = encoding;
        this.type = type;
        this.locations = new Map();
    }

    async read() {
        onnx.proto = await this.context.require('./onnx-proto');
        onnx.proto = onnx.proto.onnx;
        switch (this.encoding) {
            case 'text': {
                try {
                    const reader = this.context.read('protobuf.text');
                    this.model = onnx.proto.ModelProto.decodeText(reader);
                    this.format = `ONNX${this.model.ir_version ? ` v${this.model.ir_version}` : ''}`;
                } catch (error) {
                    const message = error && error.message ? error.message : error.toString();
                    throw new onnx.Error(`File text format is not onnx.ModelProto (${message.replace(/\.$/, '')}).`);
                }
                break;
            }
            case 'binary': {
                switch (this.type) {
                    case 'tensor': {
                        // TensorProto
                        // input_0.pb, output_0.pb
                        try {
                            const reader = this.context.read('protobuf.binary');
                            const tensor = onnx.proto.TensorProto.decode(reader);
                            tensor.name = tensor.name || this.context.identifier;
                            const attribute = new onnx.proto.AttributeProto();
                            attribute.name = 'value';
                            attribute.type = onnx.AttributeType.TENSOR;
                            attribute.t = tensor;
                            const node = new onnx.proto.NodeProto();
                            node.op_type = 'Constant';
                            node.attribute = [attribute];
                            const graph = new onnx.proto.GraphProto();
                            graph.node = [node];
                            this.model = new onnx.proto.ModelProto();
                            this.model.graph = graph;
                            this.format = 'ONNX Tensor';
                        } catch (error) {
                            const message = error && error.message ? error.message : error.toString();
                            throw new onnx.Error(`File format is not onnx.TensorProto (${message.replace(/\.$/, '')}).`);
                        }
                        break;
                    }
                    case 'graph': {
                        // GraphProto
                        try {
                            const reader = this.context.read('protobuf.binary');
                            this.model = new onnx.proto.ModelProto();
                            this.model.graph = onnx.proto.GraphProto.decode(reader);
                            this.format = 'ONNX';
                        } catch (error) {
                            const message = error && error.message ? error.message : error.toString();
                            throw new onnx.Error(`File format is not onnx.GraphProto (${message.replace(/\.$/, '')}).`);
                        }
                        break;
                    }
                    case 'model': {
                        // ModelProto
                        try {
                            const reader = this.context.read('protobuf.binary');
                            this.model = onnx.proto.ModelProto.decode(reader);
                            this.format = `ONNX${this.model.ir_version ? ` v${this.model.ir_version}` : ''}`;
                        } catch (error) {
                            const message = error && error.message ? error.message : error.toString();
                            throw new onnx.Error(`File format is not onnx.ModelProto (${message.replace(/\.$/, '')}).`);
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
        const locations = new Set();
        const location = (tensor) => {
            if (onnx.proto && tensor instanceof onnx.proto.SparseTensorProto) {
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
        const model = this.model;
        const queue = [model.graph];
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
        }
        for (const key of this.locations.keys()) {
            locations.delete(key);
        }
        const keys = Array.from(locations);
        const promises = keys.map((location) => this.context.fetch(location));
        const streams = await Promise.all(promises.map((promise) => promise.then((context) => context.stream).catch(() => null)));
        for (let i = 0; i < keys.length; i++) {
            if (streams[i] !== null) {
                this.locations.set(keys[i], streams[i]);
            }
        }
    }
};

onnx.OrtReader = class {

    static open(context) {
        const identifier = context.identifier;
        const extension = identifier.split('.').pop().toLowerCase();
        const reader = context.peek('flatbuffers.binary');
        if (reader && reader.identifier === 'ORTM') {
            context.target = reader;
            return new onnx.OrtReader(context);
        }
        const stream = context.stream;
        if (stream && stream.length >= 8 && extension === 'ort') {
            const signature = [0x08, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00];
            if (signature.length <= stream.length && stream.peek(signature.length).every((value, index) => value === signature[index])) {
                return new onnx.OrtReader(context);
            }
        }
        return null;
    }

    constructor(context) {
        this.name = 'onnx.ort';
        this.context = context;
    }

    async read() {
        onnx.schema = await this.context.require('./onnx-schema');
        onnx.schema = onnx.schema.onnxruntime.fbs;
        try {
            this.graphs = new Set();
            const reader = this.context.read('flatbuffers.binary');
            const session = onnx.schema.InferenceSession.create(reader);
            this.model = session.model;
        } catch (error) {
            const message = error && error.message ? error.message : error.toString();
            throw new onnx.Error(`File format is not ort.Model (${message.replace(/\.$/, '')}).`);
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
                            throw new onnx.Error(`Unknown shape dimension '${JSON.stringify(dimension.value)}'.`);
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
                throw new onnx.Error(`Unsupported type value '${JSON.stringify(value.value)}`);
            }
            return null;
        };
        const graph = (value) => {
            if (this.graphs.has(value)) {
                return;
            }
            this.graphs.add(value);
            value.name = this.graphs.size.toString();
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
            const value_info = new Map(value.value_info.map((entry) => [entry.name, entry]));
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
        this.format = `ONNX Runtime${this.model.ir_version ? ` v${this.model.ir_version}` : ''}`;
    }
};

onnx.JsonReader = class {

    static open(context) {
        const obj = context.peek('json');
        if (obj && (obj.irVersion !== undefined || (obj.graph && Array.isArray(obj.graph.node)))) {
            return new onnx.JsonReader(obj);
        }
        return null;
    }

    constructor(obj) {
        this.name = 'onnx.json';
        this.model = obj;
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
                throw new onnx.Error(`Unsupported ONNX JSON type '${JSON.stringify(Object.keys(value))}'.`);
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
                throw new onnx.Error(`Unsupported ONNX JSON tensor data '${JSON.stringify(value.data_type)}.`);
            }
            return value;
        };
        const sparse_tensor = (value) => {
            value.indices = tensor(value.indices);
            value.values = tensor(value.values);
            return value;
        };
        const attributeTypes = new Map(Object.entries(onnx.AttributeType));
        const attribute = (value) => {
            if (value.type && attributeTypes.has(value.type)) {
                value.type = attributeTypes.get(value.type);
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
                throw new onnx.Error(`Unsupported ONNX JSON attribute type '${JSON.stringify(value.type)}'.`);
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
        this.format = `ONNX JSON${this.model.ir_version ? ` v${this.model.ir_version}` : ''}`;
    }
};

onnx.TextReader = class {

    static open(context) {
        try {
            const stream = context.stream;
            if (stream && stream.length > 0) {
                const size = Math.min(0x10000, stream.length);
                const buffer = stream.peek(size);
                if (buffer[0] < 0x80 || buffer[0] >= 0xFE) {
                    const reader = text.Reader.open(buffer);
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
            }
        } catch {
            // continue regardless of error
        }
        return null;
    }

    constructor(context) {
        this.name = 'onnx.text';
        this._context = context;
        this._dataTypes = new Map(Object.entries(onnx.DataType).map(([key, value]) => [key.toLowerCase(), value]));
        this._attributeTypes = new Map(Object.entries(onnx.AttributeType).map(([key, value]) => [key.toLowerCase(), value]));
    }

    async read() {
        onnx.proto = await this._context.require('./onnx-proto');
        onnx.proto = onnx.proto.onnx;
        try {
            const stream = this._context.stream;
            this._decoder = text.Decoder.open(stream);
            this._position = 0;
            this._char = this._decoder.decode();
            this.model = this._parseModel();
            this.format = 'ONNX Text';
            if (this.model.ir_version !== undefined) {
                const version = typeof this.model.ir_version === 'bigint' ? Number(this.model.ir_version) : this.model.ir_version;
                this.format += ` v${version}`;
            }
            delete this._decoder;
            delete this._position;
            delete this._char;
        } catch (error) {
            const message = error && error.message ? error.message : error.toString();
            throw new onnx.Error(`File format is not onnx.ModelProto (${message.replace(/\.$/, '')}).`);
        }
    }

    _seek(position) {
        this._decoder.position = position;
        this._char = '';
        this._next();
    }

    _parseModel() {
        this._skipWhitespace();
        const model = new onnx.proto.ModelProto();
        if (this._match('<')) {
            do {
                const keyword = this._parseIdentifier();
                this._expect(':');
                switch (keyword) {
                    case 'ir_version':
                    case 'model_version':
                        model[keyword] = this._parseInteger();
                        break;
                    case 'opset_import':
                        model[keyword] = this._parseOperatorSetId();
                        break;
                    case 'producer_name':
                    case 'producer_version':
                    case 'domain':
                    case 'doc_string':
                        model[keyword] = this._parseString();
                        break;
                    case 'metadata_props':
                        this._expect('[');
                        if (!this._match(']')) {
                            do {
                                const entry = new onnx.proto.StringStringEntryProto();
                                entry.key = this._parseString();
                                this._expect(':');
                                entry.value = this._parseString();
                                model.metadata_props.push(entry);
                            } while (this._match(','));
                            this._expect(']');
                        }
                        break;
                    default:
                        this._throw(`Unknown keyword '${keyword}'.`);
                        break;
                }
            } while (this._match(','));
            this._expect('>');
        }
        model.graph = this._parseGraph();
        this._skipWhitespace();
        while (this._char !== undefined) {
            const func = this._parseFunction();
            if (func) {
                model.functions.push(func);
            }
            this._skipWhitespace();
        }
        return model;
    }

    _parseGraph() {
        const graph = new onnx.proto.GraphProto();
        graph.name = this._parseIdentifier();
        if (this._match('(')) {
            if (!this._match(')')) {
                do {
                    const valueInfo = this._parseValueInfo();
                    if (this._match('=')) {
                        const tensor = this._parseTensor(valueInfo.type);
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
        graph.output = this._parseValueInfoList();
        if (this._match('<')) {
            if (!this._match('>')) {
                do {
                    const valueInfo = this._parseValueInfo();
                    if (this._match('=')) {
                        const tensor = this._parseTensor(valueInfo.type);
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
        graph.node = this._parseNodeList();
        return graph;
    }

    _parseNodeList() {
        const list = [];
        this._expect('{');
        while (!this._match('}')) {
            list.push(this._parseNode());
        }
        return list;
    }

    _parseNode() {
        const node = new onnx.proto.NodeProto();
        node.output = this._parseIdentifierList();
        this._expect('=');
        let identifier = this._parseIdentifier();
        let domain = '';
        while (this._match('.')) {
            if (domain) {
                domain += '.';
            }
            domain += identifier;
            identifier = this._parseIdentifier();
        }
        node.domain = domain;
        node.op_type = identifier;
        if (this._match(':')) {
            node.overload = this._parseIdentifier();
        }
        node.attribute = this._parseAttributeList();
        this._expect('(');
        node.input = this._parseIdentifierList();
        this._expect(')');
        if (!node.attribute || node.attribute.length === 0) {
            node.attribute = this._parseAttributeList();
        }
        return node;
    }

    _parseAttributeList() {
        const list = [];
        if (this._match('<')) {
            do {
                list.push(this._parseAttribute());
            }
            while (this._match(','));
            this._expect('>');
        }
        return list;
    }

    _parseAttribute() {
        const attribute = new onnx.proto.AttributeProto();
        attribute.name = this._parseIdentifier();
        if (this._match(':')) {
            const type = this._parseIdentifier();
            if (!this._attributeTypes.has(type)) {
                this._throw(`Unexpected attribute type '${type}'.`);
            }
            attribute.type = this._attributeTypes.get(type);
        }
        this._expect('=');
        if (this._match('[')) {
            if (this._match(']')) {

                if (attribute.type === onnx.AttributeType.UNDEFINED) {
                    this._throw('Empty list attribute value requires type annotation.');
                }
                switch (attribute.type) {
                    case onnx.AttributeType.FLOAT:
                    case onnx.AttributeType.INT:
                    case onnx.AttributeType.STRING:
                    case onnx.AttributeType.TENSOR:
                    case onnx.AttributeType.GRAPH:
                    case onnx.AttributeType.SPARSE_TENSOR:
                    case onnx.AttributeType.TYPE_PROTO:
                        this._throw("Singleton attribute value cannot be specified as a list.");
                        break;
                    default:
                        break;
                }
            } else {
                do {
                    const value = new onnx.proto.AttributeProto();
                    let type = onnx.AttributeType.UNDEFINED;
                    switch (attribute.type) {
                        case onnx.AttributeType.FLOATS: type = onnx.AttributeType.FLOAT; break;
                        case onnx.AttributeType.INTS: type = onnx.AttributeType.INT; break;
                        case onnx.AttributeType.STRINGS: type = onnx.AttributeType.STRING; break;
                        case onnx.AttributeType.TENSORS: type = onnx.AttributeType.TENSOR; break;
                        case onnx.AttributeType.GRAPHS: type = onnx.AttributeType.GRAPH; break;
                        case onnx.AttributeType.SPARSE_TENSORS: type = onnx.AttributeType.SPARSE_TENSOR; break;
                        case onnx.AttributeType.TYPE_PROTOS: type = onnx.AttributeType.TYPE_PROTO; break;
                        default: type = attribute.type; break;
                    }
                    this._parseAttributeValue(value, type);
                    switch (value.type) {
                        case onnx.AttributeType.INT:
                            attribute.type = onnx.AttributeType.INTS;
                            attribute.ints.push(value.i);
                            break;
                        case onnx.AttributeType.FLOAT:
                            attribute.type = onnx.AttributeType.FLOATS;
                            attribute.floats.push(value.f);
                            break;
                        case onnx.AttributeType.STRING:
                            attribute.type = onnx.AttributeType.STRINGS;
                            attribute.strings.push(value.s);
                            break;
                        default:
                            break;
                    }
                }
                while (this._match(','));
            }
            this._expect(']');
        } else {
            this._parseAttributeValue(attribute, attribute.type);
        }
        return attribute;
    }

    _parseAttributeValue(attribute, type) {
        if (this._isAlpha(this._char) || this._char === '_') {
            const identifier = this._peekIdentifier();
            if (this._isType(identifier)) {
                const type = this._parseType(this._parseIdentifier());
                if (!type.tensor_type.elem_type) {
                    this._throw('Expected tensor data type.');
                }
                if (!type.tensor_type.shape || !type.tensor_type.shape.dim) {
                    this._throw('Expected tensor shape.');
                }
                this._skipWhitespace();
                if (this._char === '{' || this._char === '=' || this._peekIdentifier()) {
                    attribute.type = onnx.AttributeType.TENSOR;
                    const name = this._parseIdentifier(true);
                    this._match('=');
                    attribute.t = this._parseTensor(type);
                    if (name) {
                        attribute.t.name = name;
                    }
                } else {
                    attribute.type = onnx.AttributeType.TYPE_PROTO;
                    attribute.tp = type;
                }
            } else {
                const value = this._peekIdentifier();
                if (value === 'inf' || value === 'infinity' || value === 'nan') {
                    attribute.type = onnx.AttributeType.FLOAT;
                    attribute.f = this._parseLiteral();
                } else {
                    attribute.type = onnx.AttributeType.GRAPH;
                    attribute.g = this._parseGraph();
                }
            }
        } else if (this._match('@')) {
            attribute.ref_attr_name = this._parseIdentifier();
        } else {
            const value = this._parseLiteral();
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
                    this._throw(`Unexpected value '${JSON.stringify(value)}'.`);
                }
            }
        }
        if (type !== onnx.AttributeType.UNDEFINED && type !== attribute.type) {
            if (type === onnx.AttributeType.FLOAT && attribute.type === onnx.AttributeType.INT) {
                attribute.type = onnx.AttributeType.FLOAT;
                attribute.f = attribute.i;
                delete attribute.i;
            } else {
                this._throw('Attribute type mismatch.');
            }
        }
    }

    _parseValueInfoList() {
        const list = [];
        this._expect('(');
        if (!this._match(')')) {
            do {
                const value = this._parseValueInfo();
                list.push(value);
            } while (this._match(','));
            this._expect(')');
        }
        return list;
    }

    _parseValueInfo() {
        const valueInfo = new onnx.proto.ValueInfoProto();
        let identifier = this._parseIdentifier();
        if (this._isType(identifier)) {
            valueInfo.type = this._parseType(identifier);
            identifier = this._parseIdentifier();
        }
        valueInfo.name = identifier;
        return valueInfo;
    }

    _parseType(elem_type) {
        const type = new onnx.proto.TypeProto();
        type.tensor_type = new onnx.proto.TypeProto.Tensor();
        type.tensor_type.elem_type = this._dataTypes.get(elem_type);
        if (this._match('[')) {
            if (!this._match(']')) {
                type.tensor_type.shape = this._parseTensorShape();
                this._expect(']');
            }
        } else {
            type.tensor_type.shape = new onnx.proto.TensorShapeProto();
        }
        return type;
    }

    _parseTensorShape() {
        const shape = new onnx.proto.TensorShapeProto();
        do {
            const dimension = new onnx.proto.TensorShapeProto.Dimension();
            if (!this._match('?')) {
                const identifier = this._parseIdentifier(true);
                if (identifier) {
                    dimension.dim_param = identifier;
                } else {
                    dimension.dim_value = this._parseInteger();
                }
            }
            shape.dim.push(dimension);
        }
        while (this._match(','));
        return shape;
    }

    _parseTensor(type) {
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
                        tensor.int32_data.push(this._parseInteger());
                        break;
                    case onnx.DataType.INT64:
                        tensor.int64_data.push(this._parseInteger());
                        break;
                    case onnx.DataType.UINT32:
                    case onnx.DataType.UINT64:
                        tensor.uint64_data.push(this._parseInteger());
                        break;
                    case onnx.DataType.FLOAT:
                        tensor.float_data.push(this._parseFloat());
                        break;
                    case onnx.DataType.DOUBLE:
                        tensor.double_data.push(this._parseFloat());
                        break;
                    case onnx.DataType.STRING:
                        tensor.string_data.push(this.string());
                        break;
                    default:
                        return this._throw(`Unsupported tensor element type '${elem_type}'.`);
                }
            } while (this._match(','));
            this._expect('}');
        }
        return tensor;
    }

    _parseFunction() {
        const func = new onnx.proto.FunctionProto();
        if (this._match('<')) {
            do {
                const keyword = this._parseIdentifier();
                this._expect(':');
                switch (keyword) {
                    case 'opset_import':
                        func[keyword] = this._parseOperatorSetId();
                        break;
                    case 'domain':
                    case 'doc_string':
                        func[keyword] = this._parseString();
                        break;
                    case 'overload':
                        func[keyword] = this._parseString();
                        break;
                    default:
                        this._throw(`Unknown keyword '${keyword}'.`);
                        break;
                }
            }
            while (this._match(','));
            this._expect('>');
        }
        func.name = this._parseIdentifier();
        if (this._match('<')) {
            func.attribute = this._parseIdentifierList();
            this._expect('>');
        }
        if (this._match('(')) {
            func.input = this._parseIdentifierList();
            this._expect(')');
        }
        this._expect('=>');
        if (this._match('(')) {
            func.output = this._parseIdentifierList();
            this._expect(')');
        }
        func.node = this._parseNodeList();
        return func;
    }

    _parseIdentifierList() {
        const list = [];
        const identifier = this._parseIdentifier(true);
        if (identifier) {
            list.push(identifier);
            while (this._match(',')) {
                list.push(this._parseIdentifier());
            }
        }
        return list;
    }

    _peekIdentifier() {
        const index = this._decoder.position;
        const position = this._position;
        const char = this._char;
        const value = this._parseIdentifier(true);
        this._char = char;
        this._position = position;
        this._decoder.position = index;
        return value;
    }

    _parseIdentifier(optional) {
        this._skipWhitespace();
        const value = [];
        if (this._isAlpha(this._char) || this._char === '_') {
            value.push(this._char);
            this._next();
            while (this._isAlpha(this._char) || (this._char >= '0' && this._char <= '9') || this._char === '_') {
                value.push(this._char);
                this._next();
            }
        }
        if (!optional && value.length === 0) {
            this._throw('Identifier expected.');
        }
        return value.join('');
    }

    _parseLiteral() {
        this._skipWhitespace();
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
            const value = [this._char];
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

    _parseInteger() {
        const value = this._parseLiteral();
        if (!Number.isInteger(value)) {
            this._throw('Integer value expected.');
        }
        return value;
    }

    _parseFloat() {
        const value = this._parseLiteral();
        if (typeof value !== 'number') {
            this._throw('Float value expected.');
        }
        return value;
    }

    _parseString() {
        const value = this._parseLiteral();
        if (typeof value !== 'string') {
            this._throw('String value expected.');
        }
        return value;
    }

    _parseOperatorSetId() {
        const list = [];
        this._expect('[');
        if (!this._match(']')) {
            do {
                const value = new onnx.proto.OperatorSetIdProto();
                value.domain = this._parseString();
                this._expect(':');
                value.version = this._parseInteger();
                list.push(value);
            }
            while (this._match(','));
            this._expect(']');
        }
        return list;
    }

    _isAlpha(value) {
        return (value >= 'a' && value <= 'z') || (value >= 'A' && value <= 'Z');
    }

    _isType(identifier) {
        return this._dataTypes.has(identifier) ||
            identifier === 'seq' ||
            identifier === 'map' ||
            identifier === 'optional' ||
            identifier === 'sparse_tensor';
    }

    _match(value) {
        this._skipWhitespace();
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

    _skipWhitespace() {
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
                c = (name.length === 1) ? `\\${name}` : `\\u${(`000${c.charCodeAt(0).toString(16)}`).slice(-4)}`;
            }
            c = `token '${c}'`;
        }
        this._throw(`Unexpected ${c}`);
    }

    _throw(message) {
        message = message.replace(/\.$/, '');
        throw new onnx.Error(`${message} ${this._location()}`);
    }

    _location() {
        let line = 1;
        let column = 1;
        this._decoder.position = 0;
        let c = '';
        do {
            if (this._decoder.position === this._position) {
                return `at ${line}:${column}.`;
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
        return `at ${line}:${column}.`;
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
        return null;
    }

    constructor() {
        this.name = 'onnx.pickle';
    }

    async read() {
        throw new onnx.Error('Unsupported Pickle content.');
    }
};

onnx.DataReader = class {

    static open(context) {
        const identifier = context.identifier.toLowerCase();
        if (identifier.endsWith('.onnx_data') || identifier.endsWith('.onnx.data')) {
            return new onnx.DataReader(context, identifier);
        }
        return null;
    }

    constructor(context, identifier) {
        this.name = 'onnx.data';
        this.context = context;
        this.identifier = identifier;
        this.locations = new Map();
        this.locations.set(identifier, context.stream);
    }

    async read() {
        const file = this.identifier.substring(0, this.identifier.length - 5);
        const context = await this.context.fetch(file);
        const reader = new onnx.ProtoReader(context, 'binary', 'model');
        reader.locations = this.locations;
        await reader.read();
        this.format = reader.format;
        this.model = reader.model;
    }
};

onnx.Error = class extends Error {

    constructor(message) {
        super(message);
        this.name = 'Error loading ONNX model.';
    }
};

export const ModelFactory = onnx.ModelFactory;
