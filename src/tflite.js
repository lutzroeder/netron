/* jshint esversion: 6 */
/* eslint "indent": [ "error", 4, { "SwitchCase": 1 } ] */

var tflite = tflite || {};
var base = base || require('./base');
var flatbuffers = flatbuffers || require('flatbuffers').flatbuffers;
var long = long || { Long: require('long') };

tflite.ModelFactory = class {

    match(context) {
        var extension = context.identifier.split('.').pop().toLowerCase();
        if (extension == 'tflite' || extension == 'lite') {
            return true;
        }
        if (extension == 'tfl' || extension == 'bin') {
            var buffer = context.buffer;
            var signature = [ 0x54, 0x46, 0x4c, 0x33 ]; // TFL3
            if (buffer && buffer.length > 8 && signature.every((x, i) => x == buffer[i + 4])) {
                return true;
            }
        }
        return false;
    }

    open(context, host) {
        return host.require('./tflite-schema').then((tflite_schema) => {
            var identifier = context.identifier;
            var model = null;
            try {
                var buffer = context.buffer;
                var byteBuffer = new flatbuffers.ByteBuffer(buffer);
                tflite.schema = tflite_schema;
                if (!tflite.schema.Model.bufferHasIdentifier(byteBuffer))
                {
                    var signature = Array.from(buffer.subarray(0, Math.min(8, buffer.length))).map((c) => (c < 16 ? '0' : '') + c.toString(16)).join('');
                    throw new tflite.Error("File format is not tflite.Model (" + signature + ").");
                }
                model = tflite.schema.Model.getRootAsModel(byteBuffer);
            }
            catch (error) {
                host.exception(error, false);
                var message = error && error.message ? error.message : error.toString();
                message = message.endsWith('.') ? message.substring(0, message.length - 1) : message;
                throw new tflite.Error(message + " in '" + identifier + "'.");
            }

            return tflite.Metadata.open(host).then((metadata) => {
                try {
                    return new tflite.Model(metadata, model);
                }
                catch (error) {
                    var message = error && error.message ? error.message : error.toString();
                    message = message.endsWith('.') ? message.substring(0, message.length - 1) : message;
                    throw new new tflite.Error(message + " in '" + identifier + "'.");
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
        var operatorCodeList = [];
        var builtinOperatorMap = {};
        for (var key of Object.keys(tflite.schema.BuiltinOperator)) {
            var upperCase = new Set([ '2D', 'LSH', 'SVDF', 'RNN', 'L2', 'LSTM' ]);
            var builtinOperatorIndex = tflite.schema.BuiltinOperator[key]; 
            builtinOperatorMap[builtinOperatorIndex] = key.split('_').map((s) => {
                return (s.length < 1 || upperCase.has(s)) ? s : s.substring(0, 1) + s.substring(1).toLowerCase();
            }).join('');
        }
        for (var operatorIndex = 0; operatorIndex < model.operatorCodesLength(); operatorIndex++) {
            var operatorCode = model.operatorCodes(operatorIndex);
            var builtinCode = operatorCode.builtinCode();
            operatorCodeList.push(builtinCode === tflite.schema.BuiltinOperator.CUSTOM ?
                { name: operatorCode.customCode(), custom: true } :
                { name: builtinOperatorMap[builtinCode] });
        }
        var subgraphsLength = model.subgraphsLength();
        for (var subgraph = 0; subgraph < subgraphsLength; subgraph++) {
            var name = (subgraphsLength > 1) ? subgraph.toString() : '';
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
        this._graph = graph;
        this._name = this._graph.name() || name;
        this._nodes = [];
        this._inputs = [];
        this._outputs = [];
        var args = [];
        var names = [];
        for (var i = 0; i < graph.tensorsLength(); i++) {
            var tensor = graph.tensors(i);
            var initializer = null;
            var buffer = model.buffers(tensor.buffer());
            if (buffer.dataLength() > 0) {
                initializer = new tflite.Tensor(tensor, buffer);
            }
            args.push(new tflite.Argument(tensor, i, initializer));
            names.push(tensor.name());
        }
        for (var j = 0; j < this._graph.operatorsLength(); j++) {
            var node = this._graph.operators(j);
            var opcodeIndex = node.opcodeIndex();
            var operator = (opcodeIndex < operatorCodeList.length) ? operatorCodeList[opcodeIndex] : { name: '(' + opcodeIndex.toString() + ')' };
            this._nodes.push(new tflite.Node(metadata, node, operator, j.toString(), args));
        }
        for (var k = 0; k < graph.inputsLength(); k++) {
            var inputIndex = graph.inputs(k);
            this._inputs.push(new tflite.Parameter(names[inputIndex], true, [ args[inputIndex] ]));
        }
        for (var l = 0; l < graph.outputsLength(); l++) {
            var outputIndex = graph.outputs(l);
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

    constructor(metadata, node, operator, name, args) {
        this._metadata = metadata;
        this._operator = operator;
        this._name = name;
        this._inputs = [];
        this._outputs = [];
        if (node) {
            var schema = this._metadata.getSchema(this.operator);
            var inputs = [];
            for (var i = 0; i < node.inputsLength(); i++) {
                inputs.push(node.inputs(i));
            }
            var inputIndex = 0;
            while (inputIndex < inputs.length) {
                var count = 1;
                var inputName = null;
                var inputVisible = true;
                var inputConnections = [];
                if (schema && schema.inputs && inputIndex < schema.inputs.length) {
                    var input = schema.inputs[inputIndex];
                    inputName = input.name;
                    if (input.option == 'variadic') {
                        count = inputs.length - inputIndex;
                    }
                    if (Object.prototype.hasOwnProperty.call(input, 'visible') && !input.visible) {
                        inputVisible = false;
                    }
                }
                var inputArray = inputs.slice(inputIndex, inputIndex + count);
                for (var j = 0; j < inputArray.length; j++) {
                    if (inputArray[j] != -1) {
                        inputConnections.push(args[inputArray[j]]);
                    }
                }
                inputIndex += count;
                inputName = inputName ? inputName : inputIndex.toString();
                this._inputs.push(new tflite.Parameter(inputName, inputVisible, inputConnections));
            }
            this._outputs = [];
            for (var k = 0; k < node.outputsLength(); k++) {
                var outputIndex = node.outputs(k);
                var argument = args[outputIndex];
                var outputName = i.toString();
                if (schema && schema.outputs && k < schema.outputs.length) {
                    var output = schema.outputs[k];
                    if (output && (!output.option || output.opcodeIndex != 'variadic') && output.name) {
                        outputName = output.name;
                    }
                }
                this._outputs.push(new tflite.Parameter(outputName, true, [ argument ]));
            }
            this._attributes = [];
            if (operator.custom) {
                var custom = [];
                for (var m = 0; m < node.customOptionsLength(); m++) {
                    custom.push(node.customOptions(m));
                }
                this._attributes.push(new tflite.Attribute(this._metadata, this.operator, 'custom', custom));
            }
            var optionsTypeName = this.operator + 'Options';
            switch (this.operator) {
                case 'MaxPool2D':
                case 'AveragePool2D':
                    optionsTypeName = 'Pool2DOptions';
                    break;
            }
            var optionsType = tflite.Node._getType(optionsTypeName);
            if (typeof optionsType === 'function') {
                var options = Reflect.construct(optionsType, []);
                options = node.builtinOptions(options);
                if (options) {
                    var attributeName;
                    var attributeNames = [];
                    var attributeNamesMap = {};
                    for (attributeName of Object.keys(Object.getPrototypeOf(options))) {
                        if (attributeName != '__init') {
                            attributeNames.push(attributeName);
                        }
                        attributeNamesMap[attributeName] = true;
                    }
                    var attributeArrayNamesMap = {}; 
                    for (attributeName of Object.keys(attributeNamesMap)) {
                        if (attributeNamesMap[attributeName + 'Array'] && attributeNamesMap[attributeName + 'Length']) {
                            attributeArrayNamesMap[attributeName] = true;
                            attributeNames = attributeNames.filter((item) => item != (attributeName + 'Array') && item != (attributeName + 'Length'));
                        }
                    }
                    for (attributeName of attributeNames) {
                        if (options[attributeName] && typeof options[attributeName] == 'function') {
                            var value = null;
                            if (attributeArrayNamesMap[attributeName]) {
                                var array = [];
                                var length = options[attributeName + 'Length']();
                                var a = options[attributeName + 'Array']();
                                for (var l = 0; l < length; l++) {
                                    array.push(a[l]);
                                }
                                value = array;
                            }
                            else {
                                value = options[attributeName]();
                            }
                            var attribute = new tflite.Attribute(this._metadata, this.operator, attributeName, value);
                            if (attribute.name == 'fused_activation_function') {
                                value = attribute.value;
                                if (attribute.value != 'NONE') {
                                    var activationFunctionMap = { 'RELU': 'Relu', 'RELU_N1_TO_1': "ReluN1To1", "RELU6": "Relu6", "TANH": "Tanh", "SIGN_BIT": "SignBit" };
                                    if (activationFunctionMap[value]) {
                                        value = activationFunctionMap[value];
                                    }
                                    this._chain = [];
                                    this._chain.push(new tflite.Node(metadata, null, { name: value }, '', []));
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
        return this._name;
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
        var schema = this._metadata.getSchema(this.operator);
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
        var list = name.split('.');
        var type = tflite.schema;
        while (list.length > 0) {
            var item = list.shift();
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
        var lower = name.toLowerCase();
        for (var i = 0; i < name.length; i++) {
            this._name += (name[i] == lower[i]) ? name[i] : ('_' + lower[i]);
        }

        var schema = metadata.getAttributeSchema(operator, this._name);
        if (schema) {
            if (schema.type) {
                this._type = schema.type;
            }
            if (this._type == 'shape') {
                this._value = new tflite.TensorShape(value);
            }
            else if (this._type && tflite.schema) {
                var type = tflite.schema[this._type];
                if (type) {
                    tflite.Attribute._reverseMap = tflite.Attribute._reverseMap || {};
                    var reverse = tflite.Attribute._reverseMap[this._type];
                    if (!reverse) {
                        reverse = {};
                        for (var key of Object.keys(type)) {
                            reverse[type[key.toString()]] = key;
                        }
                        tflite.Attribute._reverseMap[this._type] = reverse;
                    }
                    if (Object.prototype.hasOwnProperty.call(reverse, this._value)) {
                        this._value = reverse[this._value];
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

    constructor(tensor, index, initializer) {
        this._id = tensor.name() || index.toString();
        this._type = initializer ? null : new tflite.TensorType(tensor);
        this._initializer = initializer;
        var quantization = tensor.quantization();
        if (quantization) {
            var value = 'q';
            var scale = (quantization.scaleLength() == 1) ? quantization.scale(0) : 0;
            var zeroPoint = (quantization.zeroPointLength() == 1) ? quantization.zeroPoint(0).toFloat64() : 0;
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

    constructor(tensor, buffer) {
        this._name = tensor.name();
        this._type = new tflite.TensorType(tensor);
        this._data = buffer.dataLength() > 0 ? buffer.dataArray() : null;
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
        var context = this._context();
        if (context.state) {
            return null;
        }
        context.limit = Number.MAX_SAFE_INTEGER;
        return this._decode(context, 0);
    }

    toString() {
        var context = this._context();
        if (context.state) {
            return '';
        }
        context.limit = 10000;
        var value = this._decode(context, 0);
        return JSON.stringify(value, null, 4);
    }

    _context() {
        var context = {};
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
            var utf8Decoder = new TextDecoder('utf-8');
            var offset = 0;
            var count = context.data.getInt32(0, true);
            offset += 4;
            var offsetTable = [];
            for (var j = 0; j < count; j++) {
                offsetTable.push(context.data.getInt32(offset, true));
                offset += 4;
            }
            offsetTable.push(this._data.length);
            var stringTable = [];
            for (var k = 0; k < count; k++) {
                var textArray = this._data.subarray(offsetTable[k], offsetTable[k + 1]);
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
        var shape = context.shape;
        if (shape.length == 0) {
            shape = [ 1 ];
        }
        var size = shape[dimension];
        var results = [];
        if (dimension == shape.length - 1) {
            for (var i = 0; i < size; i++) {
                if (context.count > context.limit) {
                    results.push('...');
                    return results;
                }
                switch (context.dataType)
                {
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
            for (var j = 0; j < size; j++) {
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
        if (!tflite.TensorType._tensorTypeMap) {
            tflite.TensorType._tensorTypeMap = tflite.TensorType._tensorTypeMap || {};
            for (var key of Object.keys(tflite.schema.TensorType)) {
                tflite.TensorType._tensorTypeMap[tflite.schema.TensorType[key].toString()] = key.toLowerCase();
            }
        }
        this._dataType = tflite.TensorType._tensorTypeMap[tensor.type().toString()] || '?';
        var dimensions = [];
        var shapeLength = tensor.shapeLength();
        if (shapeLength > 0) {
            for (var i = 0; i < shapeLength; i++) {
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
            var items = JSON.parse(data);
            if (items) {
                for (var item of items) {
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
        var schema = this.getSchema(operator);
        if (schema) {
            var attributeMap = schema.attributeMap;
            if (!attributeMap) {
                attributeMap = {};
                if (schema.attributes) {
                    for (var attribute of schema.attributes) {
                        attributeMap[attribute.name] = attribute;
                    }
                }
                schema.attributeMap = attributeMap;
            }
            var attributeSchema = attributeMap[name];
            if (attributeSchema) {
                return attributeSchema; 
            }
        }
        return null;
    }
};

tflite.Error = class extends Error {

    constructor(message) {
        super(message);
        this.name = 'Error loading TensorFlow Lite model.';
    }
};

if (typeof module !== 'undefined' && typeof module.exports === 'object') {
    module.exports.ModelFactory = tflite.ModelFactory;
}