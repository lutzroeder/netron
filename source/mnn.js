/* jshint esversion: 6 */

var mnn = mnn || {};
var flatbuffers = flatbuffers || require('./flatbuffers');

mnn.ModelFactory = class {

    match(context) {
        const extension = context.identifier.split('.').pop().toLowerCase();
        if (extension == 'mnn') {
            return true;
        }
        return false;
    }

    open(context, host) {
        return host.require('./mnn-schema').then((schema) => {
            let net = null;
            try {
                mnn.schema = flatbuffers.get('mnn').MNN;
                const reader = new flatbuffers.Reader(context.buffer);
                net = mnn.schema.Net.create(reader);
            }
            catch (error) {
                const message = error && error.message ? error.message : error.toString();
                throw new mnn.Error('File format is not mnn.Net (' + message.replace(/\.$/, '') + ').');

            }
            return mnn.Metadata.open(host).then((metadata) => {
                return new mnn.Model(metadata, net);
            });
        });
    }
};

mnn.Model = class {

    constructor(metadata, net) {
        switch (net.sourceType) {
            case mnn.schema.NetSource.CAFFE: this._source = 'Caffe'; break;
            case mnn.schema.NetSource.TENSORFLOW: this._source = 'TensorFlow'; break;
            case mnn.schema.NetSource.TFLITE: this._source = 'TensorFlow Lite'; break;
            case mnn.schema.NetSource.ONNX: this._source = 'ONNX'; break;
        }
        this._graphs = [ new mnn.Graph(metadata, net) ];
    }

    get format() {
        return 'MNN v2';
    }

    get source() {
        return this._source || '';
    }

    get graphs() {
        return this._graphs;
    }
};

mnn.Graph = class {

    constructor(metadata, net) {
        this._nodes = [];
        this._inputs = [];
        this._outputs = [];
        const inputSet = new Set();
        for (let i = 0; i < net.tensorName.length; i++) {
            if (net.tensorName[i] === '') {
                net.tensorName[i] = '\n' + i.toString();
            }
        }
        for (let i = 0; i < net.oplists.length; i++) {
            const op = net.oplists[i];
            if (op.type === mnn.schema.OpType.Input) {
                const args = [];
                for (let j = 0; j < op.outputIndexes.length; j++) {
                    const index = op.outputIndexes[j];
                    const name = net.tensorName[index];
                    const extraTensorDescribe = net.extraTensorDescribe[index];
                    const blob = extraTensorDescribe ? extraTensorDescribe.blob : null;
                    const type = blob ? new mnn.TensorType(blob.dataType, new mnn.TensorShape(blob.dims)) : null;
                    args.push(new mnn.Argument(name, type, null));
                }
                this._inputs.push(new mnn.Parameter(op.name, true, args));
            }
            else {
                this._nodes.push(new mnn.Node(metadata, op, net));
            }
            for (let k = 0; k < op.inputIndexes.length; k++) {
                const index = op.inputIndexes[k];
                inputSet.add(index);
            }
        }

        for (let i = 0; i < net.tensorName.length; i++) {
            if (!inputSet.has(i)) {
                const name = net.tensorName[i];
                const extraTensorDescribe = net.extraTensorDescribe[i];
                const blob = extraTensorDescribe ? extraTensorDescribe.blob : null;
                const type = blob ? new mnn.TensorType(blob.dataType, new mnn.TensorShape(blob.dims)) : null;
                this._outputs.push(new mnn.Parameter(name, true, [
                    new mnn.Argument(name, type, null)
                ]));
            }
        }
    }

    get name() {
        return '';
    }

    get groups() {
        return false;
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

mnn.Node = class {

    constructor(metadata, op, net) {
        this._metadata = metadata;
        this._type = mnn.Utility.enum('OpType', op.type) || '(' + op.type.toString() + ')';
        this._name = op.name || '';
        this._attributes = [];
        this._inputs = [];
        this._outputs = [];
        this._chains = [];
        const inputs = [];
        for (let i = 0; i < op.inputIndexes.length; i++) {
            const index = op.inputIndexes[i];
            const id = net.tensorName[index];
            inputs.push(new mnn.Argument(id, null, null));
        }
        this._inputs.push(new mnn.Parameter('input', true, inputs));
        const outputs = [];
        for (let i = 0; i < op.outputIndexes.length; i++) {
            const index = op.outputIndexes[i];
            const name = net.tensorName[index];
            outputs.push(new mnn.Argument(name, null, null));
        }
        this._outputs.push(new mnn.Parameter('output', true, outputs));

        const parameter = op.main;
        if (parameter) {
            const ignoreAttributes = new Set();
            if (parameter instanceof mnn.schema.Blob) {
                const type = new mnn.TensorType(parameter.dataType, new mnn.TensorShape(parameter.dims));
                let data = null;
                switch (type.dataType) {
                    case 'uint8': data = parameter.uint8s; break;
                    case 'int8': data = parameter.int8s; break;
                    case 'int32': data = parameter.int32s; break;
                    case 'int64': data = parameter.int64s; break;
                    case 'float32': data = parameter.float32s; break;
                    default:
                        throw new mnn.Error("Unknown blob data type '" + JSON.stringify(type.dataType) + "'.");
                }
                this._inputs.push(new mnn.Parameter('value', true, [
                    new mnn.Argument('', null, new mnn.Tensor('Blob', type, data))
                ]));
            }
            else if (parameter instanceof mnn.schema.Convolution2D) {
                const common = parameter.common;
                const outputCount = common.outputCount;
                const inputCount = common.inputCount;
                const kernelX = common.kernelX;
                const kernelY = common.kernelY;
                this._buildTensor(mnn.schema.DataType.DT_FLOAT, 'weight', [ outputCount, inputCount, kernelX, kernelY ], parameter.weight);
                this._buildTensor(mnn.schema.DataType.DT_FLOAT, 'bias', [ outputCount ], parameter.bias);
                ignoreAttributes.add('weight');
                ignoreAttributes.add('bias');
            }
            else if (parameter instanceof mnn.schema.InnerProduct) {
                const outputCount = parameter.outputCount;
                const inputCount = parameter.weightSize / outputCount;
                this._buildTensor(mnn.schema.DataType.DT_FLOAT, 'weight', [ outputCount, inputCount ], parameter.weight);
                this._buildTensor(mnn.schema.DataType.DT_FLOAT, 'bias', [ outputCount ], parameter.bias);
                ignoreAttributes.add('weight');
                ignoreAttributes.add('bias');
            }
            else if (parameter instanceof mnn.schema.Scale) {
                const scaleDataCount = parameter.channels;
                this._buildTensor(mnn.schema.DataType.DT_FLOAT, 'scale', [ scaleDataCount ], parameter.scaleData);
                this._buildTensor(mnn.schema.DataType.DT_FLOAT, 'bias', [ scaleDataCount ], parameter.biasData);
                ignoreAttributes.add('scaleData');
                ignoreAttributes.add('biasData');
            }
            else if (parameter instanceof mnn.schema.BatchNorm) {
                const channels = parameter.channels;
                this._buildTensor(mnn.schema.DataType.DT_FLOAT, 'mean', [ channels ], parameter.meanData);
                this._buildTensor(mnn.schema.DataType.DT_FLOAT, 'slope', [ channels ], parameter.slopeData);
                this._buildTensor(mnn.schema.DataType.DT_FLOAT, 'variance', [ channels ], parameter.varData);
                this._buildTensor(mnn.schema.DataType.DT_FLOAT, 'bias', [ channels ], parameter.biasData);
                ignoreAttributes.add('slopeData');
                ignoreAttributes.add('meanData');
                ignoreAttributes.add('varData');
                ignoreAttributes.add('biasData');
            }
            else if (parameter instanceof mnn.schema.PRelu) {
                this._buildTensor(mnn.schema.DataType.DT_FLOAT, 'slope', [ parameter.slopeCount ], parameter.slope);
                ignoreAttributes.add('slope');
            }
            else if (parameter instanceof mnn.schema.Normalize) {
                this._buildTensor(mnn.schema.DataType.DT_FLOAT, 'scale', [ parameter.scale.length ], parameter.scale);
                ignoreAttributes.add('scale');
            }
            if (ignoreAttributes.size > 0) {
                this._buildAttributes(metadata, parameter, ignoreAttributes, this._attributes);
            }
        }
    }

    _buildTensor(dataType, name, dimensions, value) {
        this._inputs.push(new mnn.Parameter(name, true, [
            new mnn.Argument('', null, new mnn.Tensor('Weight', new mnn.TensorType(dataType, new mnn.TensorShape(dimensions)), value))
        ]));
    }

    _buildAttributes(metadata, parameter, ignoreAttributes, attributes) {
        if (parameter) {
            for (const key of Object.keys(parameter)) {
                if (ignoreAttributes && ignoreAttributes.has(key)) {
                    continue;
                }
                const value = parameter[key];
                if (Object.keys(mnn.schema).find((key) => mnn.schema[key].prototype && value instanceof mnn.schema[key])) {
                    this._buildAttributes(metadata, value, null, attributes);
                    continue;
                }
                if (value) {
                    const schema = metadata.attribute(this.type, key);
                    attributes.push(new mnn.Attribute(schema, key, value));
                }
            }
        }
    }

    get type() {
        return this._type;
    }

    get name() {
        return this._name;
    }

    get domain() {
        return null;
    }

    get metadata() {
        return this._metadata.type(this.type);
    }

    get group() {
        return null;
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

mnn.Attribute = class {

    constructor(schema, name, value, visible) {
        this._type = null;
        this._value = ArrayBuffer.isView(value) ? Array.from(value) : value;
        this._name = name;
        this._visible = visible;
        if (schema) {
            if (schema.type) {
                this._type = schema.type;
                const type = mnn.schema[this._type + 'Name'];
                if (type) {
                    this._value = type[this._value];
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

mnn.Parameter = class {

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

mnn.Argument = class {

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

mnn.Tensor = class {

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
        context.data = this._data;
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
                results.push(context.data[context.index]);
                context.index++;
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

mnn.TensorType = class {

    constructor(dataType, shape) {

        switch (dataType) {
            case mnn.schema.DataType.DT_INVALID: this._dataType = '?'; break;
            case mnn.schema.DataType.DT_FLOAT: this._dataType = 'float32'; break;
            case mnn.schema.DataType.DT_DOUBLE: this._dataType = 'float64'; break;
            case mnn.schema.DataType.DT_INT32: this._dataType = 'int32'; break;
            case mnn.schema.DataType.DT_UINT8: this._dataType = 'uint8'; break;
            case mnn.schema.DataType.DT_INT16: this._dataType = 'int16'; break;
            case mnn.schema.DataType.DT_INT8: this._dataType = 'int8'; break;
            case mnn.schema.DataType.DT_STRING: this._dataType = 'string'; break;
            case mnn.schema.DataType.DT_COMPLEX64: this._dataType = 'complex64'; break;
            case mnn.schema.DataType.DT_INT64: this._dataType = 'int64'; break;
            case mnn.schema.DataType.DT_BOOL: this._dataType = 'boolean'; break;
            case mnn.schema.DataType.DT_QINT8: this._dataType = 'qint8'; break;
            case mnn.schema.DataType.DT_QUINT8: this._dataType = 'quint8'; break;
            case mnn.schema.DataType.DT_QINT32: this._dataType = 'qint32'; break;
            case mnn.schema.DataType.DT_BFLOAT16: this._dataType = 'bfloat16'; break;
            case mnn.schema.DataType.DT_QINT16: this._dataType = 'qint16'; break;
            case mnn.schema.DataType.DT_QUINT16: this._dataType = 'quint16'; break;
            case mnn.schema.DataType.DT_UINT16: this._dataType = 'uint16'; break;
            case mnn.schema.DataType.DT_COMPLEX128: this._dataType = 'complex128'; break;
            case mnn.schema.DataType.DT_HALF: this._dataType = 'float16'; break;
            case mnn.schema.DataType.DT_RESOURCE: this._dataType = 'resource'; break;
            case mnn.schema.DataType.DT_VARIANT: this._dataType = 'variant'; break;
            default: throw new mnn.Error("Unknown data type '" + JSON.stringify(dataType) + "'.");
        }
        this._shape = shape;
    }

    get dataType() {
        return this._dataType;
    }

    get shape() {
        return this._shape;
    }

    toString() {
        return this._dataType + this._shape.toString();
    }
};

mnn.TensorShape = class {

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

mnn.Metadata = class {

    static open(host) {
        if (mnn.Metadata._metadata) {
            return Promise.resolve(mnn.Metadata._metadata);
        }
        return host.request(null, 'mnn-metadata.json', 'utf-8').then((data) => {
            mnn.Metadata._metadata = new mnn.Metadata(data);
            return mnn.Metadata._metadata;
        }).catch(() => {
            mnn.Metadata._metadata = new mnn.Metadata(null);
            return mnn.Metadata._metadata;
        });
    }

    constructor(data) {
        this._map = new Map();
        if (data) {
            const items = JSON.parse(data);
            if (items) {
                for (const item of items) {
                    if (item.name && item.schema) {
                        item.schema.name = item.name;
                        this._map.set(item.name, item.schema);
                    }
                }
            }
        }
    }

    type(name) {
        return this._map.has(name) ? this._map.get(name) : null;
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

mnn.Utility = class {

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
};

mnn.Error = class extends Error {

    constructor(message) {
        super(message);
        this.name = 'Error loading MNN model.';
    }
};

if (typeof module !== 'undefined' && typeof module.exports === 'object') {
    module.exports.ModelFactory = mnn.ModelFactory;
}
