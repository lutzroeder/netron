
const mnn = {};

mnn.ModelFactory = class {

    match(context) {
        const reader = context.peek('flatbuffers.binary');
        if (reader) {
            context.type = 'mnn.flatbuffers';
            context.target = reader;
        }
    }

    async open(context) {
        mnn.schema = await context.require('./mnn-schema');
        mnn.schema = mnn.schema.MNN;
        let net = null;
        try {
            const reader = context.target;
            net = mnn.schema.Net.create(reader);
        } catch (error) {
            const message = error && error.message ? error.message : error.toString();
            throw new mnn.Error(`File format is not mnn.Net (${message.replace(/\.$/, '')}).`);
        }
        const metadata = await context.metadata('mnn-metadata.json');
        return new mnn.Model(metadata, net);
    }
};

mnn.Model = class {

    constructor(metadata, net) {
        this.format = 'MNN v2';
        const sources = new Map([
            [mnn.schema.NetSource.CAFFE, 'Caffe'],
            [mnn.schema.NetSource.TENSORFLOW, 'TensorFlow'],
            [mnn.schema.NetSource.TFLITE, 'TensorFlow Lite'],
            [mnn.schema.NetSource.ONNX, 'ONNX'],
            [mnn.schema.NetSource.TORCH, 'Torch']
        ]);
        if (!sources.has(net.sourceType)) {
            throw new mnn.Error(`Unsupported model source '${net.sourceType}'.`);
        }
        this.source = sources.get(net.sourceType);
        this.graphs = [new mnn.Graph(metadata, net)];
    }
};

mnn.Graph = class {

    constructor(metadata, net) {
        this.name = '';
        this.nodes = [];
        this.inputs = [];
        this.outputs = [];
        for (let i = 0; i < net.tensorName.length; i++) {
            if (net.tensorName[i] === '') {
                net.tensorName[i] = `\n${i}`;
            }
        }
        const inputs = new Map();
        for (const op of net.oplists) {
            for (const input of op.inputIndexes) {
                inputs.set(input, (inputs.get(input) || 0) + 1);
            }
        }
        const consts = new Map();
        const oplists = net.oplists.filter((op) => {
            if (op.type === mnn.schema.OpType.Const &&
                op.inputIndexes.length === 0 &&
                op.outputIndexes.length === 1 &&
                op.main instanceof mnn.schema.Blob &&
                inputs.get(op.outputIndexes[0]) === 1) {
                consts.set(op.outputIndexes[0], op);
                return false;
            }
            return true;
        });
        const values = new Map();
        values.map = (index) => {
            if (!values.has(index)) {
                const name = net.tensorName[index];
                const op = consts.get(index);
                if (op) {
                    const tensor = op ? mnn.Utility.createTensor(op.main, 'Const') : null;
                    values.set(index, new mnn.Value(name, null, tensor));
                } else {
                    const extraTensorDescribe = net.extraTensorDescribe[index];
                    const blob = extraTensorDescribe ? extraTensorDescribe.blob : null;
                    const type = blob && blob.dims && blob.dims.length > 0 ? new mnn.TensorType(blob.dataType, new mnn.TensorShape(blob.dims), blob.dataFormat) : null;
                    values.set(index, new mnn.Value(name, type, null));
                }
            }
            return values.get(index);
        };

        for (const op of oplists) {
            if (op.type === mnn.schema.OpType.Input) {
                const args = Array.from(op.outputIndexes).map((index) => values.map(index));
                const argument = new mnn.Argument(op.name, args);
                this.inputs.push(argument);
            } else {
                const node = new mnn.Node(metadata, op, net, values);
                this.nodes.push(node);
            }
        }

        for (let i = 0; i < net.tensorName.length; i++) {
            if (!inputs.has(i)) {
                const value = values.map(i);
                const argument = new mnn.Argument(value.name, [value]);
                this.outputs.push(argument);
            }
        }
    }
};

mnn.Node = class {

    constructor(metadata, op, net, values) {
        const type = mnn.Utility.enum('OpType', op.type) || `(${op.type})`;
        this.type = metadata.type(type) || { name: type };
        this.name = op.name || '';
        this.attributes = [];
        this.inputs = [];
        this.outputs = [];
        this.chains = [];
        if (op.inputIndexes && op.inputIndexes.length > 0) {
            const argument = new mnn.Argument('input', Array.from(op.inputIndexes).map((index) => values.map(index)));
            this.inputs.push(argument);
        }
        if (op.outputIndexes && op.outputIndexes.length > 0) {
            const argument = new mnn.Argument('output', Array.from(op.outputIndexes).map((index) => values.map(index)));
            this.outputs.push(argument);
        }
        const param = op.main;
        if (param) {
            const parameters = [param];
            if (param instanceof mnn.schema.Blob) {
                const tensor = mnn.Utility.createTensor(param, 'Blob');
                const value = new mnn.Value('', null, tensor);
                const argument = new mnn.Argument('value', [value]);
                this.inputs.push(argument);
                parameters.splice(0, parameters.length);
            } else if (param instanceof mnn.schema.Convolution2D) {
                const common = param.common;
                const outputCount = common.outputCount;
                const inputCount = common.inputCount;
                const kernelX = common.kernelX;
                const kernelY = common.kernelY;
                this._buildTensor('weight', mnn.schema.DataType.DT_FLOAT, [outputCount, inputCount, kernelX, kernelY], param.weight);
                this._buildTensor('bias', mnn.schema.DataType.DT_FLOAT, [outputCount], param.bias);
                delete param.weight;
                delete param.bias;
                delete param.quanParameter;
                delete param.symmetricQuan;
            } else if (param instanceof mnn.schema.InnerProduct) {
                const outputCount = param.outputCount;
                const inputCount = param.weightSize / outputCount;
                this._buildTensor('weight', mnn.schema.DataType.DT_FLOAT, [outputCount, inputCount], param.weight);
                this._buildTensor('bias', mnn.schema.DataType.DT_FLOAT, [outputCount], param.bias);
                delete param.weight;
                delete param.bias;
                delete param.quanParameter;
            } else if (param instanceof mnn.schema.Scale) {
                const scaleDataCount = param.channels;
                this._buildTensor('scale', mnn.schema.DataType.DT_FLOAT, [scaleDataCount], param.scaleData);
                this._buildTensor('bias', mnn.schema.DataType.DT_FLOAT, [scaleDataCount], param.biasData);
                delete param.scaleData;
                delete param.biasData;
            } else if (param instanceof mnn.schema.BatchNorm) {
                const channels = param.channels;
                this._buildTensor('mean', mnn.schema.DataType.DT_FLOAT, [channels], param.meanData);
                this._buildTensor('slope', mnn.schema.DataType.DT_FLOAT, [channels], param.slopeData);
                this._buildTensor('variance', mnn.schema.DataType.DT_FLOAT, [channels], param.varData);
                this._buildTensor('bias', mnn.schema.DataType.DT_FLOAT, [channels], param.biasData);
                delete param.slopeData;
                delete param.meanData;
                delete param.varData;
                delete param.biasData;
            } else if (param instanceof mnn.schema.PRelu) {
                this._buildTensor('slope', mnn.schema.DataType.DT_FLOAT, [param.slopeCount], param.slope);
                delete param.slopeCount;
            } else if (param instanceof mnn.schema.Normalize) {
                this._buildTensor('scale', mnn.schema.DataType.DT_FLOAT, [param.scale.length], param.scale);
                delete param.scale;
            }
            while (parameters.length > 0) {
                const parameter = parameters.shift();
                for (const [key, value] of Object.entries(parameter)) {
                    if (Object.keys(mnn.schema).find((key) => mnn.schema[key].prototype && value instanceof mnn.schema[key])) {
                        parameters.push(value);
                        continue;
                    }
                    const attribute = new mnn.Attribute(metadata.attribute(type, key), key, value);
                    this.attributes.push(attribute);
                }
            }
        }
    }

    _buildTensor(name, dataType, dimensions, value) {
        const shape = new mnn.TensorShape(dimensions);
        const type = new mnn.TensorType(dataType, shape);
        const tensor = new mnn.Tensor('Weight', type, value);
        const argument = new mnn.Argument(name, [new mnn.Value('', null, tensor)]);
        this.inputs.push(argument);
    }
};

mnn.Attribute = class {

    constructor(metadata, name, value, visible) {
        this.type = null;
        this.value = ArrayBuffer.isView(value) ? Array.from(value) : value;
        this.name = name;
        this.visible = visible ? true : false;
        if (metadata && metadata.type) {
            this.type = metadata.type;
            switch (this.type) {
                case 'DataType':
                    this.value = mnn.Utility.dataType(this.value);
                    break;
                default:
                    this.value = mnn.Utility.enum(this.type, this.value);
                    break;
            }
        }
    }
};

mnn.Argument = class {

    constructor(name, value) {
        this.name = name;
        this.value = value;
    }
};

mnn.Value = class {

    constructor(name, type, initializer) {
        this.name = name;
        this.type = !type && initializer ? initializer.type : type;
        this.initializer = initializer || null;
    }
};

mnn.Tensor = class {

    constructor(category, type, data) {
        this.category = category;
        this.type = type;
        switch (type.dataType) {
            case 'int32':
            case 'float32':
                this.encoding = '|';
                this.values = data ? data.slice(0) : null;
                break;
            case 'uint8':
            case 'float16':
            case 'bfloat16':
                this.encoding = '<';
                this.values = data ? data.slice(0) : null;
                break;
            default:
                throw new mnn.Error(`Unsupported data type '${type.dataType}'.`);
        }
    }
};

mnn.TensorType = class {

    constructor(dataType, shape, format) {
        this.dataType = mnn.Utility.dataType(dataType);
        this.shape = shape;
        if (format) {
            switch (format) {
                case mnn.schema.MNN_DATA_FORMAT.NCHW: this.denotation = 'NCHW'; break;
                case mnn.schema.MNN_DATA_FORMAT.NHWC: this.denotation = 'NHWC'; break;
                case mnn.schema.MNN_DATA_FORMAT.NC4HW4: this.denotation = 'NC4HW4'; break;
                case mnn.schema.MNN_DATA_FORMAT.NHWC4: this.denotation = 'NHWC4'; break;
                default: throw new mnn.Error(`Unsupported tensor type format '${format}'.`);
            }
        }
    }

    toString() {
        return this.dataType + this.shape.toString();
    }
};

mnn.TensorShape = class {

    constructor(dimensions) {
        this.dimensions = Array.from(dimensions);
    }

    toString() {
        if (this.dimensions && this.dimensions.length > 0) {
            return `[${this.dimensions.map((dimension) => dimension ? dimension.toString() : '?').join(',')}]`;
        }
        return '';
    }
};

mnn.Utility = class {

    static dataType(type) {
        switch (type) {
            case mnn.schema.DataType.DT_INVALID: return '?';
            case mnn.schema.DataType.DT_FLOAT: return 'float32';
            case mnn.schema.DataType.DT_DOUBLE: return 'float64';
            case mnn.schema.DataType.DT_INT32: return 'int32';
            case mnn.schema.DataType.DT_UINT8: return 'uint8';
            case mnn.schema.DataType.DT_INT16: return 'int16';
            case mnn.schema.DataType.DT_INT8: return 'int8';
            case mnn.schema.DataType.DT_STRING: return 'string';
            case mnn.schema.DataType.DT_COMPLEX64: return 'complex64';
            case mnn.schema.DataType.DT_INT64: return 'int64';
            case mnn.schema.DataType.DT_BOOL: return 'boolean';
            case mnn.schema.DataType.DT_QINT8: return 'qint8';
            case mnn.schema.DataType.DT_QUINT8: return 'quint8';
            case mnn.schema.DataType.DT_QINT32: return 'qint32';
            case mnn.schema.DataType.DT_BFLOAT16: return 'bfloat16';
            case mnn.schema.DataType.DT_QINT16: return 'qint16';
            case mnn.schema.DataType.DT_QUINT16: return 'quint16';
            case mnn.schema.DataType.DT_UINT16: return 'uint16';
            case mnn.schema.DataType.DT_COMPLEX128: return 'complex128';
            case mnn.schema.DataType.DT_HALF: return 'float16';
            case mnn.schema.DataType.DT_RESOURCE: return 'resource';
            case mnn.schema.DataType.DT_VARIANT: return 'variant';
            default: throw new mnn.Error(`Unsupported data type '${JSON.stringify(type)}'.`);
        }
    }

    static enum(name, value) {
        const type = name && mnn.schema ? mnn.schema[name] : undefined;
        if (type) {
            mnn.Utility._enumKeyMap = mnn.Utility._enumKeyMap || new Map();
            if (!mnn.Utility._enumKeyMap.has(name)) {
                const map = new Map();
                for (const key of Object.keys(type)) {
                    map.set(type[key], key);
                }
                mnn.Utility._enumKeyMap.set(name, map);
            }
            const map = mnn.Utility._enumKeyMap.get(name);
            if (map.has(value)) {
                return map.get(value);
            }
        }
        return value.toString();
    }

    static createTensor(param, category) {
        const shape = new mnn.TensorShape(param.dims);
        const type = new mnn.TensorType(param.dataType, shape, param.dataFormat);
        let data = null;
        switch (type.dataType) {
            case 'uint8': data = param.uint8s; break;
            case 'int8': data = param.int8s; break;
            case 'int32': data = param.int32s; break;
            case 'int64': data = param.int64s; break;
            case 'float16': data = param.uint8s; break;
            case 'float32': data = param.float32s; break;
            case 'bfloat16': data = param.uint8s; break;
            default: throw new mnn.Error(`Unsupported blob data type '${JSON.stringify(type.dataType)}'.`);
        }
        return new mnn.Tensor(category, type, data);
    }
};

mnn.Error = class extends Error {

    constructor(message) {
        super(message);
        this.name = 'Error loading MNN model.';
    }
};

export const ModelFactory = mnn.ModelFactory;

