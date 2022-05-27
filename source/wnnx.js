
var wnn = wnn || {};
var flatbuffers = flatbuffers || require('./flatbuffers');

wnn.ModelFactory = class {

    match(context) {
        const stream = context.stream;
        if (stream.length >= 4) {
            const extension = context.identifier.split('.').pop().toLowerCase();
            if (extension == 'wnnx') {
                const buffer = stream.peek(4);
                const reader = flatbuffers.BinaryReader.open(buffer);
                if (reader.root === 0x00000018 || reader.root === 0x0000001C || reader.root === 0x00000020) {
                    return 'wnn.flatbuffers';
                }
            }
        }
        return undefined;
    }

    open(context) {
        return context.require('./wnnx-schema').then((/* schema */) => {
            let net = null;
            try {
                wnn.schema = flatbuffers.get('wnnx').wnn;
                const stream = context.stream;
                const reader = flatbuffers.BinaryReader.open(stream);
                net = wnn.schema.Graph.create(reader);
            }
            catch (error) {
                const message = error && error.message ? error.message : error.toString();
                throw new wnn.Error('File format is not wnn.Graph (' + message.replace(/\.$/, '') + ').');
            }
            return wnn.Metadata.open(context).then((metadata) => {
                return new wnn.Model(metadata, net);
            });
        });
    }
};

wnn.Model = class {

    constructor(metadata, net) {
        const NetSource = wnn.schema.ModelSource;
        switch (net.model_source) {
            // case NetSource.CAFFE: this._source = 'Caffe'; break;
            case NetSource.TENSORFLOW: this._source = 'TensorFlow'; break;
            case NetSource.TFLITE: this._source = 'TensorFlow Lite'; break;
            case NetSource.ONNX: this._source = 'ONNX'; break;
            case NetSource.TORCH: this._source = 'Torch'; break;
            default: throw new wnn.Error("Unsupported model source '" + net.sourceType + "'.");
        }
        this._graphs = [ new wnn.Graph(metadata, net) ];
    }

    get format() {
        return 'wnnx v0';
    }

    get source() {
        return this._source || '';
    }

    get graphs() {
        return this._graphs;
    }
};

wnn.Graph = class {

    constructor(metadata, net) {
        this._nodes = [];
        this._inputs = [];
        this._outputs = [];
        for (let i = 0; i < net.tensor_names.length; i++) {
            if (net.tensor_names[i] === '') {
                net.tensor_names[i] = '\n' + i.toString();
            }
        }
        const inputs = new Map();
        for (const op of net.oplists) {
            for (const input of op.input_indexes) {
                inputs.set(input, (inputs.get(input) || 0) + 1);
            }
        }
        const consts = new Map();
        const oplists = net.oplists.filter((op) => {
            if (op.type === wnn.schema.OpType.Const &&
                op.input_indexes.length === 0 &&
                op.output_indexes.length === 1 &&
                op.param instanceof wnn.schema.Blob &&
                inputs.get(op.output_indexes[0]) === 1) {
                consts.set(op.output_indexes[0], op);
                return false;
            }
            return true;
        });
        const args = new Map();
        const arg = (index) => {
            if (!args.has(index)) {
                const name = net.tensor_names[index];
                const op = consts.get(index);
                if (op) {
                    const tensor = op ? wnn.Utility.createTensor(op.param, 'Const') : null;
                    const argument = new wnn.Argument(name, null, tensor);
                    args.set(index, argument);
                }
                else {
                    const extraTensorDescribe = net.extra_tensor_describe[index];
                    const blob = extraTensorDescribe ? extraTensorDescribe.blob : null;
                    const type = blob && blob.dims && blob.dims.length > 0 ? new wnn.TensorType(blob.dtype, new wnn.TensorShape(blob.dims), blob.data_format) : null;
                    const argument = new wnn.Argument(name, type, null);
                    args.set(index, argument);
                }
            }
            return args.get(index);
        };

        for (const op of oplists) {
            if (op.type === wnn.schema.OpType.input) {
                const args = Array.from(op.output_indexes).map((index) => arg(index));
                this._inputs.push(new wnn.Parameter(op.name, true, args));
            }
            else {
                this._nodes.push(new wnn.Node(metadata, op, net, arg));
            }
        }

        for (let i = 0; i < net.tensor_names.length; i++) {
            if (!inputs.has(i)) {
                const argument = arg(i);
                const parameter = new wnn.Parameter(argument.name, true, [ argument ]);
                this._outputs.push(parameter);
            }
        }
    }

    get name() {
        return '';
    }

    get nodes() {
        return this._nodes;
    }

    get outputs() {
        return this._outputs;
    }

    get inputs() {
        return this._inputs;
    }
};

wnn.Node = class {

    constructor(metadata, op, net, arg) {
        const type = wnn.Utility.enum('OpType', op.type) || '(' + op.type.toString() + ')';
        this._type = metadata.type(type) || { name: type };
        this._name = op.name || '';
        this._attributes = [];
        this._inputs = [];
        this._outputs = [];
        this._chains = [];
        if (op.input_indexes && op.input_indexes.length > 0) {
            this._inputs.push(new wnn.Parameter('input', true, Array.from(op.input_indexes).map((index) => arg(index))));
        }
        if (op.output_indexes && op.output_indexes.length > 0) {
            this._outputs.push(new wnn.Parameter('output', true, Array.from(op.output_indexes).map((index) => arg(index))));
        }
        const param = op.param;
        if (param) {
            const parameters = [ param ];
            if (param instanceof wnn.schema.Blob) {
                const tensor = wnn.Utility.createTensor(param, 'Blob');
                const argument = new wnn.Argument('', null, tensor);
                const parameter = new wnn.Parameter('value', true, [ argument ]);
                this._inputs.push(parameter);
                parameters.splice(0, parameters.length);
            }
            else if (param instanceof wnn.schema.Conv2D) {
                const common = param.common;
                const outputCount = common.output_count;
                const inputCount = common.input_count;
                const kernelX = common.kernel_x;
                const kernelY = common.kernel_y;
                this._buildTensor('weight', wnn.schema.DataType.DT_FLOAT, [ outputCount, inputCount, kernelX, kernelY ], param.weight);
                this._buildTensor('bias', wnn.schema.DataType.DT_FLOAT, [ outputCount ], param.bias);
                delete param.weight;
                delete param.bias;
                // delete param.quanParameter;
                // delete param.symmetricQuan;
            }
            // else if (param instanceof wnn.schema.InnerProduct) {
            //     const outputCount = param.outputCount;
            //     const inputCount = param.weightSize / outputCount;
            //     this._buildTensor('weight', wnn.schema.DataType.DT_FLOAT, [ outputCount, inputCount ], param.weight);
            //     this._buildTensor('bias', wnn.schema.DataType.DT_FLOAT, [ outputCount ], param.bias);
            //     delete param.weight;
            //     delete param.bias;
            //     delete param.quanParameter;
            // }
            // else if (param instanceof wnn.schema.Scale) {
            //     const scaleDataCount = param.channels;
            //     this._buildTensor('scale', wnn.schema.DataType.DT_FLOAT, [ scaleDataCount ], param.scaleData);
            //     this._buildTensor('bias', wnn.schema.DataType.DT_FLOAT, [ scaleDataCount ], param.biasData);
            //     delete param.scaleData;
            //     delete param.biasData;
            // }
            else if (param instanceof wnn.schema.BatchNorm) {
                const channels = param.channels;
                this._buildTensor('mean', wnn.schema.DataType.DT_FLOAT, [ channels ], param.meanData);
                this._buildTensor('slope', wnn.schema.DataType.DT_FLOAT, [ channels ], param.slopeData);
                this._buildTensor('variance', wnn.schema.DataType.DT_FLOAT, [ channels ], param.varData);
                this._buildTensor('bias', wnn.schema.DataType.DT_FLOAT, [ channels ], param.biasData);
                delete param.slopeData;
                delete param.meanData;
                delete param.varData;
                delete param.biasData;
            }
            else if (param instanceof wnn.schema.PRelu) {
                this._buildTensor('slope', wnn.schema.DataType.DT_FLOAT, [ param.slopeCount ], param.slope);
                delete param.slopeCount;
            }
            // else if (param instanceof wnn.schema.Normalize) {
            //     this._buildTensor('scale', wnn.schema.DataType.DT_FLOAT, [ param.scale.length ], param.scale);
            //     delete param.scale;
            // }
            while (parameters.length > 0) {
                const parameter = parameters.shift();
                for (const key of Object.keys(parameter)) {
                    if (Object.prototype.hasOwnProperty.call(parameter, key)) {
                        const value = parameter[key];
                        if (Object.keys(wnn.schema).find((key) => wnn.schema[key].prototype && value instanceof wnn.schema[key])) {
                            parameters.push(value);
                            continue;
                        }
                        const schema = metadata.attribute(this.type, key);
                        this._attributes.push(new wnn.Attribute(schema, key, value));
                    }
                }
            }
        }
    }

    _buildTensor(name, dataType, dimensions, value) {
        const shape = new wnn.TensorShape(dimensions);
        const type = new wnn.TensorType(dataType, shape);
        const tensor = new wnn.Tensor('Weight', type, value);
        const argument = new wnn.Argument('', null, tensor);
        const parameter = new wnn.Parameter(name, true, [ argument ]);
        this._inputs.push(parameter);
    }

    get type() {
        return this._type;
    }

    get name() {
        return this._name;
    }

    get inputs() {
        return this._inputs;
    }

    get outputs() {
        return this._outputs;
    }

    get chain() {
        return this._chains;
    }

    get attributes() {
        return this._attributes;
    }
};

wnn.Attribute = class {

    constructor(schema, name, value, visible) {
        this._type = null;
        this._value = ArrayBuffer.isView(value) ? Array.from(value) : value;
        this._name = name;
        this._visible = visible ? true : false;
        if (schema) {
            if (schema.type) {
                this._type = schema.type;
                switch (this._type) {
                    case 'DataType':
                        this._value = wnn.Utility.dataType(this._value);
                        break;
                    default:
                        this._value = wnn.Utility.enum(this._type, this._value);
                        break;
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

    get visible() {
        return this._visible == false ? false : true;
    }
};

wnn.Parameter = class {

    constructor(name, visible, args) {
        this._name = name;
        this._visible = visible;
        this._arguments = args;
    }

    get name() {
        return this._name;
    }

    get visible() {
        return this._visible;
    }

    get arguments() {
        return this._arguments;
    }
};

wnn.Argument = class {

    constructor(name, type, initializer) {
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

wnn.Tensor = class {

    constructor(kind, type, data) {
        this._kind = kind;
        this._type = type;
        this._data = data ? data.slice(0) : null;
    }

    get kind() {
        return this._kind;
    }

    get type() {
        return this._type;
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
        return JSON.stringify(value, null, 4);
    }

    _context() {
        const context = {};
        context.state = null;
        if (!this._data || this._data.length === 0) {
            context.state = 'Tensor data is empty.';
            return context;
        }
        context.index = 0;
        context.count = 0;
        context.dataType = this._type.dataType;
        context.dimensions = this._type.shape.dimensions;
        switch (context.dataType) {
            case 'float16':
                context.view = new DataView(this._data.buffer, this._data.byteOffset, this._data.byteLength);
                break;
            default:
                context.data = this._data;
                break;
        }
        return context;
    }

    _decode(context, dimension) {
        let shape = context.dimensions;
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
                switch (context.dataType) {
                    case 'float16':
                        results.push(context.view.getFloat16(context.index, true));
                        context.index += 2;
                        break;
                    default:
                        results.push(context.data[context.index]);
                        context.index++;
                        break;
                }
                context.count++;
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
        if (context.dimensions.length == 0) {
            return results[0];
        }
        return results;
    }
};

wnn.TensorType = class {

    constructor(dataType, shape, format) {
        this._dataType = wnn.Utility.dataType(dataType);
        this._shape = shape;
        if (format) {
            switch (format) {
                case wnn.schema.WNN_DATA_FORMAT.NCHW: this._denotation = 'NCHW'; break;
                case wnn.schema.WNN_DATA_FORMAT.NHWC: this._denotation = 'NHWC'; break;
                case wnn.schema.WNN_DATA_FORMAT.NC4HW4: this._denotation = 'NC4HW4'; break;
                case wnn.schema.WNN_DATA_FORMAT.NHWC4: this._denotation = 'NHWC4'; break;
                default: throw new wnn.Error("Unsupported tensor type format '" + format + "'.");
            }
        }
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
        return this._dataType + this._shape.toString();
    }
};

wnn.TensorShape = class {

    constructor(dimensions) {
        this._dimensions = Array.from(dimensions);
    }

    get dimensions() {
        return this._dimensions;
    }

    toString() {
        if (this._dimensions && this._dimensions.length > 0) {
            return '[' + this._dimensions.map((dimension) => dimension ? dimension.toString() : '?').join(',') + ']';
        }
        return '';
    }
};

wnn.Metadata = class {

    static open(context) {
        if (wnn.Metadata._metadata) {
            return Promise.resolve(wnn.Metadata._metadata);
        }
        return context.request('wnn-metadata.json', 'utf-8', null).then((data) => {
            wnn.Metadata._metadata = new wnn.Metadata(data);
            return wnn.Metadata._metadata;
        }).catch(() => {
            wnn.Metadata._metadata = new wnn.Metadata(null);
            return wnn.Metadata._metadata;
        });
    }

    constructor(data) {
        this._map = new Map();
        if (data) {
            const metadata = JSON.parse(data);
            this._map = new Map(metadata.map((item) => [ item.name, item ]));
        }
    }

    type(name) {
        return this._map.get(name);
    }

    attribute(type, name) {
        const schema = this.type(type);
        if (schema) {
            let attributeMap = schema.attributeMap;
            if (!attributeMap) {
                attributeMap = {};
                if (schema.attributes) {
                    for (const attribute of schema.attributes) {
                        attributeMap[attribute.name] = attribute;
                    }
                }
                schema.attributeMap = attributeMap;
            }
            const attributeSchema = attributeMap[name];
            if (attributeSchema) {
                return attributeSchema;
            }
        }
        return null;
    }
};

wnn.Utility = class {

    static dataType(type) {
        switch (type) {
            case wnn.schema.DataType.DT_INVALID: return '?';
            case wnn.schema.DataType.DT_FLOAT: return 'float32';
            case wnn.schema.DataType.DT_DOUBLE: return 'float64';
            case wnn.schema.DataType.DT_INT32: return 'int32';
            case wnn.schema.DataType.DT_UINT8: return 'uint8';
            case wnn.schema.DataType.DT_INT16: return 'int16';
            case wnn.schema.DataType.DT_INT8: return 'int8';
            case wnn.schema.DataType.DT_STRING: return 'string';
            case wnn.schema.DataType.DT_COMPLEX64: return 'complex64';
            case wnn.schema.DataType.DT_INT64: return 'int64';
            case wnn.schema.DataType.DT_BOOL: return 'boolean';
            case wnn.schema.DataType.DT_QINT8: return 'qint8';
            case wnn.schema.DataType.DT_QUINT8: return 'quint8';
            case wnn.schema.DataType.DT_QINT32: return 'qint32';
            case wnn.schema.DataType.DT_BFLOAT16: return 'bfloat16';
            case wnn.schema.DataType.DT_QINT16: return 'qint16';
            case wnn.schema.DataType.DT_QUINT16: return 'quint16';
            case wnn.schema.DataType.DT_UINT16: return 'uint16';
            case wnn.schema.DataType.DT_COMPLEX128: return 'complex128';
            case wnn.schema.DataType.DT_HALF: return 'float16';
            case wnn.schema.DataType.DT_RESOURCE: return 'resource';
            case wnn.schema.DataType.DT_VARIANT: return 'variant';
            default: throw new wnn.Error("Unsupported data type '" + JSON.stringify(type) + "'.");
        }
    }

    static enum(name, value) {
        const type = name && wnn.schema ? wnn.schema[name] : undefined;
        if (type) {
            wnn.Utility._enumKeyMap = wnn.Utility._enumKeyMap || new Map();
            if (!wnn.Utility._enumKeyMap.has(name)) {
                const map = new Map();
                for (const key of Object.keys(type)) {
                    map.set(type[key], key);
                }
                wnn.Utility._enumKeyMap.set(name, map);
            }
            const map = wnn.Utility._enumKeyMap.get(name);
            if (map.has(value)) {
                return map.get(value);
            }
        }
        return value.toString();
    }

    static createTensor(param, kind) {
        const type = new wnn.TensorType(param.dataType, new wnn.TensorShape(param.dims), param.dataFormat);
        let data = null;
        switch (type.dataType) {
            case 'uint8': data = param.uint8s; break;
            case 'int8': data = param.int8s; break;
            case 'int32': data = param.int32s; break;
            case 'int64': data = param.int64s; break;
            case 'float16': data = param.uint8s; break;
            case 'float32': data = param.float32s; break;
            default: throw new wnn.Error("Unsupported blob data type '" + JSON.stringify(type.dataType) + "'.");
        }
        return new wnn.Tensor(kind, type, data);
    }
};

wnn.Error = class extends Error {

    constructor(message) {
        super(message);
        this.name = 'Error loading WNNX model.';
    }
};

if (typeof module !== 'undefined' && typeof module.exports === 'object') {
    module.exports.ModelFactory = wnn.ModelFactory;
}
