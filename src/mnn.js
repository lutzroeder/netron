/* jshint esversion: 6 */
/* eslint "indent": [ "error", 4, { "SwitchCase": 1 } ] */

var mnn = mnn || {};
var base = base || require('./base');
var flatbuffers = flatbuffers || require('flatbuffers').flatbuffers;

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
            return mnn.Metadata.open(host).then((metadata) => {
                const identifier = context.identifier;
                try {
                    mnn.schema = schema.mnn_schema;
                    const byteBuffer = new flatbuffers.ByteBuffer(context.buffer);
                    const net = mnn.schema.Net.getRootAsNet(byteBuffer);
                    return new mnn.Model(metadata, net);
                }
                catch (error) {
                    host.exception(error, false);
                    const message = error && error.message ? error.message : error.toString();
                    throw new mnn.Error(message.replace(/\.$/, '') + " in '" + identifier + "'.");
                }
            });
        });
    }
};

mnn.Model = class {

    constructor(metadata, net) {
        switch (net.sourceType()) {
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
        let inputSet = new Set();
        for (let i = 0; i < net.oplistsLength(); i++) {
            const op = net.oplists(i);
            if (mnn.schema.OpTypeName[op.type()] === 'Input') {
                let args = [];
                for (let j = 0; j < op.outputIndexesLength(); j++) {
                    const index = op.outputIndexes(j);
                    const name = net.tensorName(index);
                    const extraTensorDescribe = net.extraTensorDescribe(index);
                    const blob = extraTensorDescribe ? extraTensorDescribe.blob() : null;
                    const type = blob ? mnn.Graph._blobTensorType(blob) : null;
                    args.push(new mnn.Argument(name, type, null));
                }
                this._inputs.push(new mnn.Parameter(op.name(), true, args));
            }
            else {
                this._nodes.push(new mnn.Node(metadata, op, net));
            }
            for (let k = 0; k < op.inputIndexesLength(); k++) {
                const index = op.inputIndexes(k);
                inputSet.add(index);
            }
        }

        for (let i = 0; i < net.tensorNameLength(); i++) {
            if (!inputSet.has(i)) {
                const name = net.tensorName(i);
                const extraTensorDescribe = net.extraTensorDescribe(i);
                const blob = extraTensorDescribe ? extraTensorDescribe.blob() : null;
                const type = blob ? mnn.Graph._blobTensorType(blob) : null;
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

    static _blobTensorType(blob) {
        mnn.Graph._blobTensorTypeMap = mnn.Graph._blobTensorTypeMap || new Map([
            [ mnn.schema.DataType.DT_INVALID, '?' ],
            [ mnn.schema.DataType.DT_FLOAT, 'float32' ],
            [ mnn.schema.DataType.DT_DOUBLE, 'float64' ],
            [ mnn.schema.DataType.DT_INT32, 'int32' ],
            [ mnn.schema.DataType.DT_UINT8, 'uint8' ],
            [ mnn.schema.DataType.DT_INT16, 'int16' ],
            [ mnn.schema.DataType.DT_INT8, 'int8' ],
            [ mnn.schema.DataType.DT_STRING, 'string' ],
            [ mnn.schema.DataType.DT_COMPLEX64, 'complex64' ],
            [ mnn.schema.DataType.DT_INT64, 'int64' ],
            [ mnn.schema.DataType.DT_BOOL, 'boolean' ],
            [ mnn.schema.DataType.DT_QINT8, 'qint8' ],
            [ mnn.schema.DataType.DT_QUINT8, 'quint8' ],
            [ mnn.schema.DataType.DT_QINT32, 'qint32' ],
            [ mnn.schema.DataType.DT_BFLOAT16, 'bfloat16' ],
            [ mnn.schema.DataType.DT_QINT16, 'qint16' ],
            [ mnn.schema.DataType.DT_QUINT16, 'quint16' ],
            [ mnn.schema.DataType.DT_UINT16, 'uint16' ],
            [ mnn.schema.DataType.DT_COMPLEX128, 'complex128' ],
            [ mnn.schema.DataType.DT_HALF, 'float16' ],
            [ mnn.schema.DataType.DT_RESOURCE, 'resource' ],
            [ mnn.schema.DataType.DT_VARIANT, 'variant' ],
        ]);
        const dataType = mnn.Graph._blobTensorTypeMap.has(blob.dataType()) ? mnn.Graph._blobTensorTypeMap.get(blob.dataType()) : '?';
        const dimensions = blob.dimsArray() || [];
        return new mnn.TensorType(dataType, new mnn.TensorShape(dimensions));
    }
};

mnn.Node = class {

    constructor(metadata, op, net) {
        this._metadata = metadata;
        this._operator = mnn.schema.OpTypeName[op.type()] || '(' + op.type().toString() + ')';
        this._name = op.name() || '';
        this._attributes = [];
        this._inputs = [];
        this._outputs = [];
        this._chains = [];
        let inputs = [];
        for (let i = 0; i < op.inputIndexesLength(); i++) {
            const index = op.inputIndexes(i);
            const id = net.tensorName(index);
            inputs.push(new mnn.Argument(id, null, null));
        }
        this._inputs.push(new mnn.Parameter('input', true, inputs));
        let outputs = [];
        for (let i = 0; i < op.outputIndexesLength(); i++) {
            const index = op.outputIndexes(i);
            const name = net.tensorName(index);
            outputs.push(new mnn.Argument(name, null, null));
        }
        this._outputs.push(new mnn.Parameter('output', true, outputs));

        const parameterType = mnn.schema.OpParameterName[op.mainType()];
        const parameterConstructor = mnn.schema[parameterType];
        if (typeof parameterConstructor === 'function') {
            const parameter = op.main(Reflect.construct(parameterConstructor, []));
            if (parameter !== null && parameter instanceof mnn.schema.Blob) {
                const type = mnn.Graph._blobTensorType(parameter);
                let data = null;
                switch (type.dataType) {
                    case 'int32': data = parameter.int32sArray(); break;
                    case 'float32': data = parameter.float32sArray(); break;
                }
                this._inputs.push(new mnn.Parameter('value', true, [
                    new mnn.Argument('', null, new mnn.Tensor('Blob', type, data))
                ]));
            }
            else {
                // weights & bias
                let invisibleAttributes = null;
                switch (parameterType) {
                    case 'Convolution2D': {
                        const common = parameter.common();
                        const outputCount = common.outputCount();
                        const inputCount = common.inputCount();
                        const kernelX = common.kernelX();
                        const kernelY = common.kernelY();
                        this._buildTensor('float32', 'weight', [ outputCount, inputCount, kernelX, kernelY ], parameter.weightArray());
                        this._buildTensor('float32', 'bias', [ outputCount ], parameter.biasArray());
                        invisibleAttributes = { "weight": true, "bias": true };
                        break;
                    }
                    case 'InnerProduct': {
                        const outputCount = parameter.outputCount();
                        const inputCount = parameter.weightSize() / outputCount;
                        this._buildTensor('float32', 'weight', [ outputCount, inputCount ], parameter.weightArray());
                        this._buildTensor('float32', 'bias', [ outputCount ], parameter.biasArray());
                        invisibleAttributes = { 'weight': true, 'bias': true };
                        break;
                    }
                    case 'Scale': {
                        const scaleDataCount = parameter.channels();
                        this._buildTensor('float32', 'scale', [ scaleDataCount ], parameter.scaleDataArray());
                        this._buildTensor('float32', 'bias', [ scaleDataCount ], parameter.biasDataArray());
                        invisibleAttributes = { 'scaleData': true, 'biasData': true };
                        break;
                    }
                    case 'BatchNorm': {
                        const channels = parameter.channels();
                        this._buildTensor('float32', 'mean', [ channels ], parameter.meanDataArray());
                        this._buildTensor('float32', 'slope', [ channels ], parameter.slopeDataArray());
                        this._buildTensor('float32', 'variance', [ channels ], parameter.varDataArray());
                        this._buildTensor('float32', 'bias', [ channels ], parameter.biasDataArray());
                        invisibleAttributes = { 'slopeData': true, 'meanData': true, 'varData': true, 'biasData': true };
                        break;
                    }
                    case 'PRelu': {
                        this._buildTensor('float32', 'slope', [ parameter.slopeCount() ], parameter.slopeArray());
                        invisibleAttributes = { 'slope': true };
                        break;
                    }
                    case 'Normalize': {
                        this._buildTensor('float32', 'scale', [ parameter.scaleLength() ], parameter.scaleArray());
                        invisibleAttributes = { 'scale': true };
                        break;
                    }
                }
                this._recursivelyBuildAttributes(metadata, net, parameter, parameterType, invisibleAttributes, this._attributes);
            }
        }
    }

    _buildTensor(dataType, name, dimensions, value) {
        this._inputs.push(new mnn.Parameter(name, true, [
            new mnn.Argument('', null, new mnn.Tensor('Weight', new mnn.TensorType(dataType, new mnn.TensorShape(dimensions)), value))
        ]));
    }

    _recursivelyBuildAttributes(metadata, net, parameter, paramterType, invisibleAttributes, attributeHolders) {
        if (!parameter) return;

        let attributeNames = [];
        let attributeNamesMap = {};
        for (const attributeName of Object.keys(Object.getPrototypeOf(parameter))) {
            if (attributeName != '__init') {
                attributeNames.push(attributeName);
            }
            attributeNamesMap[attributeName] = true;
        }

        let attributeArrayNamesMap = {};
        for (const attributeName of Object.keys(attributeNamesMap)) {
            if (attributeNamesMap[attributeName + 'Length']) { // some bugs without array
                attributeArrayNamesMap[attributeName] = true;
                attributeNames = attributeNames.filter((item) => item != (attributeName + 'Array') && item != (attributeName + 'Length'));
            }
        }

        for (const attributeName of attributeNames) {

            if (invisibleAttributes && invisibleAttributes[attributeName]) {
                continue;
            }

            if (parameter[attributeName] && typeof parameter[attributeName] == 'function') {
                let value = null;
                if (attributeArrayNamesMap[attributeName]) {
                    let array = [];
                    const length = parameter[attributeName + 'Length']();
                    for (let i = 0; i < length; i++) {
                        array.push(parameter[attributeName](i));
                    }
                    value = array;
                }
                else {
                    value = parameter[attributeName]();
                    if (typeof value === 'object') {
                        let name = null;
                        for (const key of Object.getOwnPropertyNames(mnn.schema)) {
                            const type = mnn.schema[key];
                            if (typeof type === "function" && value instanceof type) {
                                name = key;
                                break;
                            }
                        }
                        this._recursivelyBuildAttributes(metadata, net, value, name, null, attributeHolders);
                        value = null;
                    }
                }

                if (value != null) {
                    const schema = metadata.attribute(this.operator, attributeName);
                    attributeHolders.push(new mnn.Attribute(schema, attributeName, value));
                }
            }
        }
    }

    get operator() {
        return this._operator;
    }

    get name() {
        return this._name;
    }

    get domain() {
        return null;
    }

    get metadata() {
        return this._metadata.type(this.operator);
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
        this._value = value;
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
        this._data = data;
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
        if (!this._data) {
            context.state = 'Tensor data is empty.';
            return context;
        }
        context.index = 0;
        context.count = 0;
        context.dataType = this._type.dataType;
        context.dimensions = this._type.shape.dimensions;
        context.data = this._dataType;
        return context;
    }

    _decode(context, dimension) {
        let shape = context.dimensions;
        if (shape.length == 0) {
            shape = [ 1 ];
        }
        let results = [];
        let size = shape[dimension];
        if (dimension == shape.length - 1) {
            for (let i = 0; i < size; i++) {
                if (context.count > context.limit) {
                    results.push('...');
                    return results;
                }
                results.push(this._data[context.index]);
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
        this._dataType = dataType || '?';
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
        this._dimensions = dimensions;
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

    type(operator) {
        return this._map.has(operator) ? this._map.get(operator) : null;
    }

    attribute(operator, name) {
        const schema = this.type(operator);
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

mnn.Error = class extends Error {

    constructor(message) {
        super(message);
        this.name = 'Error loading MNN model.';
    }
};

if (typeof module !== 'undefined' && typeof module.exports === 'object') {
    module.exports.ModelFactory = mnn.ModelFactory;
}
