/* jshint esversion: 6 */
/* eslint "indent": [ "error", 4, { "SwitchCase": 1 } ] */

var tflite = tflite || {};
var base = base || require('./base');
var flatbuffers = flatbuffers || require('flatbuffers').flatbuffers;
var long = long || { Long: require('long') };

tflite.ModelFactory = class {

    match(context) {
        const extension = context.identifier.split('.').pop().toLowerCase();
        if (extension == 'tflite' || extension == 'lite') {
            return true;
        }
        if (extension == 'tfl' || extension == 'bin') {
            const buffer = context.buffer;
            const signature = [ 0x54, 0x46, 0x4c, 0x33 ]; // TFL3
            if (buffer && buffer.length > 8 && signature.every((x, i) => x == buffer[i + 4])) {
                return true;
            }
        }
        return false;
    }

    open(context, host) {
        return host.require('./tflite-schema').then((tflite_schema) => {
            const identifier = context.identifier;
            let model = null;
            try {
                const buffer = context.buffer;
                const byteBuffer = new flatbuffers.ByteBuffer(buffer);
                tflite.schema = tflite_schema;
                if (!tflite.schema.Model.bufferHasIdentifier(byteBuffer)) {
                    const signature = Array.from(buffer.subarray(0, Math.min(8, buffer.length))).map((c) => (c < 16 ? '0' : '') + c.toString(16)).join('');
                    throw new tflite.Error("File format is not tflite.Model (" + signature + ").");
                }
                model = tflite.schema.Model.getRootAsModel(byteBuffer);
            }
            catch (error) {
                host.exception(error, false);
                let message = error && error.message ? error.message : error.toString();
                message = message.endsWith('.') ? message.substring(0, message.length - 1) : message;
                throw new tflite.Error(message + " in '" + identifier + "'.");
            }

            return tflite.Metadata.open(host).then((metadata) => {
                try {
                    return new tflite.Model(metadata, model);
                }
                catch (error) {
                    let message = error && error.message ? error.message : error.toString();
                    message = message.endsWith('.') ? message.substring(0, message.length - 1) : message;
                    throw new tflite.Error(message + " in '" + identifier + "'.");
                }
            });
        });
    }
};

tflite.Model = class {

    constructor(metadata, model) {
        this._graphs = [];
        this._format = 'TensorFlow Lite v' + model.version().toString();
        this._description = model.description() || '';
        let operatorCodeList = [];
        let builtinOperatorMap = {};
        for (let key of Object.keys(tflite.schema.BuiltinOperator)) {
            const upperCase = new Set([ '2D', 'LSH', 'SVDF', 'RNN', 'L2', 'LSTM' ]);
            const builtinOperatorIndex = tflite.schema.BuiltinOperator[key]; 
            builtinOperatorMap[builtinOperatorIndex] = key.split('_').map((s) => {
                return (s.length < 1 || upperCase.has(s)) ? s : s.substring(0, 1) + s.substring(1).toLowerCase();
            }).join('');
        }
        for (let operatorIndex = 0; operatorIndex < model.operatorCodesLength(); operatorIndex++) {
            const operatorCode = model.operatorCodes(operatorIndex);
            const builtinCode = operatorCode.builtinCode();
            operatorCodeList.push(builtinCode === tflite.schema.BuiltinOperator.CUSTOM ?
                { name: operatorCode.customCode(), custom: true } :
                { name: builtinOperatorMap[builtinCode] });
        }
        const subgraphsLength = model.subgraphsLength();
        for (let subgraph = 0; subgraph < subgraphsLength; subgraph++) {
            const name = (subgraphsLength > 1) ? subgraph.toString() : '';
            this._graphs.push(new tflite.Graph(metadata, model.subgraphs(subgraph), name, operatorCodeList, model));
        }
    }

    get format() {
        return this._format;
    }

    get description() {
        return this._description;
    }

    get graphs() {
        return this._graphs;
    }
}; 

tflite.Graph = class {

    constructor(metadata, graph, name, operatorCodeList, model) {
        this._name = graph.name() || name;
        this._nodes = [];
        this._inputs = [];
        this._outputs = [];
        let args = [];
        let names = [];
        for (let i = 0; i < graph.tensorsLength(); i++) {
            const tensor = graph.tensors(i);
            let initializer = null;
            const buffer = model.buffers(tensor.buffer());
            const is_variable = tensor.isVariable();
            if (buffer.dataLength() > 0 || is_variable) {
                initializer = new tflite.Tensor(i, tensor, buffer, is_variable);
            }
            args.push(new tflite.Argument(i, tensor, initializer));
            names.push(tensor.name());
        }
        for (let j = 0; j < graph.operatorsLength(); j++) {
            const node = graph.operators(j);
            const opcodeIndex = node.opcodeIndex();
            const operator = (opcodeIndex < operatorCodeList.length) ? operatorCodeList[opcodeIndex] : { name: '(' + opcodeIndex.toString() + ')' };
            this._nodes.push(new tflite.Node(metadata, node, operator, args));
        }
        for (let k = 0; k < graph.inputsLength(); k++) {
            const inputIndex = graph.inputs(k);
            this._inputs.push(new tflite.Parameter(names[inputIndex], true, [ args[inputIndex] ]));
        }
        for (let l = 0; l < graph.outputsLength(); l++) {
            const outputIndex = graph.outputs(l);
            this._outputs.push(new tflite.Parameter(names[outputIndex], true, [ args[outputIndex] ]));
        }
    }

    get name() {
        return this._name;
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
};

tflite.Node = class {

    constructor(metadata, node, operator, args) {
        this._metadata = metadata;
        this._operator = operator;
        this._inputs = [];
        this._outputs = [];
        if (node) {
            const schema = this._metadata.getSchema(this.operator);
            let inputs = [];
            for (let i = 0; i < node.inputsLength(); i++) {
                inputs.push(node.inputs(i));
            }
            let inputIndex = 0;
            while (inputIndex < inputs.length) {
                let count = 1;
                let inputName = null;
                let inputVisible = true;
                let inputConnections = [];
                if (schema && schema.inputs && inputIndex < schema.inputs.length) {
                    const input = schema.inputs[inputIndex];
                    inputName = input.name;
                    if (input.option == 'variadic') {
                        count = inputs.length - inputIndex;
                    }
                    if (Object.prototype.hasOwnProperty.call(input, 'visible') && !input.visible) {
                        inputVisible = false;
                    }
                }
                const inputArray = inputs.slice(inputIndex, inputIndex + count);
                for (let j = 0; j < inputArray.length; j++) {
                    if (inputArray[j] != -1) {
                        inputConnections.push(args[inputArray[j]]);
                    }
                }
                inputIndex += count;
                inputName = inputName ? inputName : inputIndex.toString();
                this._inputs.push(new tflite.Parameter(inputName, inputVisible, inputConnections));
            }
            this._outputs = [];
            for (let k = 0; k < node.outputsLength(); k++) {
                const outputIndex = node.outputs(k);
                const argument = args[outputIndex];
                let outputName = k.toString();
                if (schema && schema.outputs && k < schema.outputs.length) {
                    const output = schema.outputs[k];
                    if (output && (!output.option || output.opcodeIndex != 'variadic') && output.name) {
                        outputName = output.name;
                    }
                }
                this._outputs.push(new tflite.Parameter(outputName, true, [ argument ]));
            }
            this._attributes = [];
            if (operator.custom) {
                let custom = [];
                for (let m = 0; m < node.customOptionsLength(); m++) {
                    custom.push(node.customOptions(m));
                }
                this._attributes.push(new tflite.Attribute(this._metadata, this.operator, 'custom', custom));
            }
            let optionsTypeName = this.operator + 'Options';
            switch (this.operator) {
                case 'MaxPool2D':
                case 'AveragePool2D':
                    optionsTypeName = 'Pool2DOptions';
                    break;
            }
            const optionsType = tflite.Node._getType(optionsTypeName);
            if (typeof optionsType === 'function') {
                let options = Reflect.construct(optionsType, []);
                options = node.builtinOptions(options);
                if (options) {
                    let attributeName;
                    let attributeNames = [];
                    let attributeNamesMap = {};
                    for (attributeName of Object.keys(Object.getPrototypeOf(options))) {
                        if (attributeName != '__init') {
                            attributeNames.push(attributeName);
                        }
                        attributeNamesMap[attributeName] = true;
                    }
                    let attributeArrayNamesMap = {}; 
                    for (attributeName of Object.keys(attributeNamesMap)) {
                        if (attributeNamesMap[attributeName + 'Array'] && attributeNamesMap[attributeName + 'Length']) {
                            attributeArrayNamesMap[attributeName] = true;
                            attributeNames = attributeNames.filter((item) => item != (attributeName + 'Array') && item != (attributeName + 'Length'));
                        }
                    }
                    for (attributeName of attributeNames) {
                        if (options[attributeName] && typeof options[attributeName] == 'function') {
                            let value = null;
                            if (attributeArrayNamesMap[attributeName]) {
                                let array = [];
                                const length = options[attributeName + 'Length']();
                                const a = options[attributeName + 'Array']();
                                for (let l = 0; l < length; l++) {
                                    array.push(a[l]);
                                }
                                value = array;
                            }
                            else {
                                value = options[attributeName]();
                            }
                            const attribute = new tflite.Attribute(this._metadata, this.operator, attributeName, value);
                            if (attribute.name == 'fused_activation_function') {
                                value = attribute.value;
                                if (attribute.value != 'NONE') {
                                    const activationFunctionMap = { 'RELU': 'Relu', 'RELU_N1_TO_1': "ReluN1To1", "RELU6": "Relu6", "TANH": "Tanh", "SIGN_BIT": "SignBit" };
                                    if (activationFunctionMap[value]) {
                                        value = activationFunctionMap[value];
                                    }
                                    this._chain = [];
                                    this._chain.push(new tflite.Node(metadata, null, { name: value }, []));
                                }
                            }
                            this._attributes.push(attribute);
                        }
                    }
                }
            }
        }
    }

    get operator() {
        return this._operator.name;
    }

    get name() {
        return '';
    }

    get domain() {
        return null;
    }

    get documentation() {
        return '';
    }

    get group() {
        return null;
    }

    get category() {
        if (this._operator.custom) {
            return 'custom';
        }
        const schema = this._metadata.getSchema(this.operator);
        return (schema && schema.category) ? schema.category : '';
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

    get attributes() {
        return this._attributes;
    }

    static _getType(name) {
        const list = name.split('.');
        let type = tflite.schema;
        while (list.length > 0) {
            const item = list.shift();
            type = type[item];
            if (!type) {
                return null;
            }
        }
        if (type == tflite.schema) {
            return null;
        }
        return type;
    }
};

tflite.Attribute = class {

    constructor(metadata, operator, name, value) {
        this._type = null;
        this._value = value;
        this._name = '';
        const lower = name.toLowerCase();
        for (let i = 0; i < name.length; i++) {
            this._name += (name[i] == lower[i]) ? name[i] : ('_' + lower[i]);
        }

        const schema = metadata.getAttributeSchema(operator, this._name);
        if (schema) {
            if (schema.type) {
                this._type = schema.type;
            }
            if (this._type == 'shape') {
                this._value = new tflite.TensorShape(value);
            }
            else if (this._type) {
                switch (this._type) {
                    case 'TensorType': {
                        this._value = tflite.Utility.dataType(this._value);
                        break;
                    }
                    default: {
                        this._value = tflite.Utility.enum(this._type, this._value);
                    }
                }
            }
        }

        if (this._name == 'fused_activation_function') {
            this._visible = false;
        }
        else if (schema) {
            if (Object.prototype.hasOwnProperty.call(schema, 'visible') && !schema.visible) {
                this._visible = false;
            }
            else if (Object.prototype.hasOwnProperty.call(schema, 'default')) {
                value = this._value;
                if (typeof value == 'function') {
                    value = value();
                }
                if (value == schema.default) {
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

    get visible() {
        return this._visible == false ? false : true;
    }
};

tflite.Parameter = class {

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

tflite.Argument = class {

    constructor(index, tensor, initializer) {
        this._id = tensor.name() + ':' + index.toString();
        this._type = initializer ? null : new tflite.TensorType(tensor);
        this._initializer = initializer;
        const quantization = tensor.quantization();
        if (quantization) {
            let value = 'q';
            const scale = (quantization.scaleLength() == 1) ? quantization.scale(0) : 0;
            const zeroPoint = (quantization.zeroPointLength() == 1) ? quantization.zeroPoint(0).toFloat64() : 0;
            if (scale != 0 || zeroPoint != 0) {
                value = scale.toString() + ' * ' + (zeroPoint == 0 ? 'q' : ('(q - ' + zeroPoint.toString() + ')'));
            }
            if (quantization.minLength() == 1) {
                value = quantization.min(0).toString() + ' \u2264 ' + value;
            }
            if (quantization.maxLength() == 1) {
                value = value + ' \u2264 ' + quantization.max(0).toString();
            }
            if (value != 'q') {
                this._quantization = value;
            }
        }
    }

    get id() {
        if (this._initializer) {
            return this._initializer.name;
        }
        return this._id;
    }

    get type() {
        if (this._initializer) {
            return this._initializer.type;
        }
        return this._type;
    }

    get quantization() {
        return this._quantization;
    }

    get initializer() {
        return this._initializer;
    }
};

tflite.Tensor = class {

    constructor(index, tensor, buffer, is_variable) {
        this._name = tensor.name() + ':' + index.toString();
        this._type = new tflite.TensorType(tensor);
        this._data = buffer.dataLength() > 0 ? buffer.dataArray() : null;
        this._is_variable = is_variable;
    }

    get kind() {
        return this._is_variable ? 'Variable' : '';
    }

    get name() {
        return this._name;
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
        context.index = 0;
        context.count = 0;

        if (this._data == null) {
            context.state = 'Tensor data is empty.';
            return context;
        }
 
        context.dataType = this._type.dataType;
        context.shape = this._type.shape.dimensions;
        context.data = new DataView(this._data.buffer, this._data.byteOffset, this._data.byteLength);

        if (this._type.dataType == 'string') {
            const utf8Decoder = new TextDecoder('utf-8');
            let offset = 0;
            const count = context.data.getInt32(0, true);
            offset += 4;
            let offsetTable = [];
            for (let j = 0; j < count; j++) {
                offsetTable.push(context.data.getInt32(offset, true));
                offset += 4;
            }
            offsetTable.push(this._data.length);
            let stringTable = [];
            for (let k = 0; k < count; k++) {
                const textArray = this._data.subarray(offsetTable[k], offsetTable[k + 1]);
                if (utf8Decoder) {
                    stringTable.push(utf8Decoder.decode(textArray));
                }
                else {
                    stringTable.push(String.fromCharCode.apply(null, textArray));
                }
            }
            context.data = stringTable;
        }
        return context;
    }

    _decode(context, dimension) {
        const shape = (context.shape.length == 0) ? [ 1 ] : context.shape;
        const size = shape[dimension];
        let results = [];
        if (dimension == shape.length - 1) {
            for (let i = 0; i < size; i++) {
                if (context.count > context.limit) {
                    results.push('...');
                    return results;
                }
                switch (context.dataType) {
                    case 'uint8':
                        results.push(context.data.getUint8(context.index));
                        context.index += 1;
                        context.count++;
                        break;
                    case 'int8':
                        results.push(context.data.getInt8(context.index));
                        context.index += 1;
                        context.count++;
                        break;
                    case 'int32':
                        results.push(context.data.getInt32(context.index, true));
                        context.index += 4;
                        context.count++;
                        break;
                    case 'int64':
                        results.push(new long.Long(context.data.getUint32(context.index, true), context.data.getUint32(context.index + 4, true), false));
                        context.index += 8;
                        context.count++;
                        break;
                    case 'float16':
                        results.push(context.data.getFloat16(context.index, true));
                        context.index += 2;
                        context.count++;
                        break;
                    case 'float32':
                        results.push(context.data.getFloat32(context.index, true));
                        context.index += 4;
                        context.count++;
                        break;
                    case 'string':
                        results.push(context.data[context.index++]);
                        context.count++;
                        break;
                    default:
                        break;
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
};

tflite.TensorType = class {

    constructor(tensor) {
        this._dataType = tflite.Utility.dataType(tensor.type());
        let dimensions = [];
        const shapeLength = tensor.shapeLength();
        if (shapeLength > 0) {
            for (let i = 0; i < shapeLength; i++) {
                dimensions.push(tensor.shape(i));
            }
        }
        this._shape = new tflite.TensorShape(dimensions);
    }

    get dataType() {
        return this._dataType;
    }

    get shape() {
        return this._shape;
    }

    toString() {
        return this.dataType + this._shape.toString();
    }
};

tflite.TensorShape = class {

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
        return '[' + this._dimensions.map((dimension) => dimension.toString()).join(',') + ']';
    }
};

tflite.Metadata = class {

    static open(host) {
        if (tflite.Metadata._metadata) {
            return Promise.resolve(tflite.Metadata._metadata);
        }
        return host.request(null, 'tflite-metadata.json', 'utf-8').then((data) => {
            tflite.Metadata._metadata = new tflite.Metadata(data);
            return tflite.Metadata._metadata;
        }).catch(() => {
            tflite.Metadata._metadata = new tflite.Metadata(null);
            return tflite.Metadata._metadata;
        });
    }

    constructor(data) {
        this._map = {};
        if (data) {
            const items = JSON.parse(data);
            if (items) {
                for (let item of items) {
                    if (item.name && item.schema) {
                        this._map[item.name] = item.schema;
                    }
                }
            }
        }
    }

    getSchema(operator) {
        return this._map[operator];
    }

    getAttributeSchema(operator, name) {
        const schema = this.getSchema(operator);
        if (schema) {
            let attributeMap = schema.attributeMap;
            if (!attributeMap) {
                attributeMap = {};
                if (schema.attributes) {
                    for (let attribute of schema.attributes) {
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

tflite.Utility = class {

    static dataType(type) {
        if (!tflite.Utility._tensorTypeMap) {
            tflite.Utility._tensorTypeMap = new Map();
            for (let name of Object.keys(tflite.schema.TensorType)) {
                tflite.Utility._tensorTypeMap.set(tflite.schema.TensorType[name], name.toLowerCase());
            }
            tflite.Utility._tensorTypeMap.set(6, 'boolean');
        }
        return tflite.Utility._tensorTypeMap.has(type) ? tflite.Utility._tensorTypeMap.get(type) : '?';
    }

    static enum(type, value) {
        if (type && tflite.schema && tflite.schema[type]) {
            if (!tflite.Utility._enumTypeMap) {
                tflite.Utility._enumTypeMap = new Map();
            }
            let typeMap = tflite.Utility._enumTypeMap.get(type);
            if (!typeMap) {
                typeMap = new Map();
                const enumType = tflite.schema[type];
                if (enumType) {
                    for (let key of Object.keys(enumType)) {
                        typeMap.set(enumType[key], key);
                    }
                }
                tflite.Utility._enumTypeMap.set(type, typeMap);
            }
            if (typeMap.has(value)) {
                return typeMap.get(value);
            }
        }
        return value;
    }
}

tflite.Error = class extends Error {

    constructor(message) {
        super(message);
        this.name = 'Error loading TensorFlow Lite model.';
    }
};

if (typeof module !== 'undefined' && typeof module.exports === 'object') {
    module.exports.ModelFactory = tflite.ModelFactory;
}