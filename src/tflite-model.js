/*jshint esversion: 6 */

class TensorFlowLiteModelFactory {


    match(context, host) {
        var extension = context.identifier.split('.').pop();
        return extension == 'tflite' || extension == 'lite';
    }

    open(context, host, callback) {
        host.require('tflite', (err) => {
            if (err) {
                callback(err, null);
                return;
            }

            var model = null;
            try {
                var byteBuffer = new flatbuffers.ByteBuffer(context.buffer);
                if (!tflite.Model.bufferHasIdentifier(byteBuffer))
                {
                    callback(new TensorFlowLiteError('Invalid FlatBuffers identifier.'));
                    return;
                }
                model = tflite.Model.getRootAsModel(byteBuffer);
            }
            catch (error) {
                host.exception(error, false);
                callback(new TensorFlowLiteError(error.message), null);
            }

            TensorFlowLiteOperatorMetadata.open(host, (err, metadata) => {
                try {
                    callback(null, new TensorFlowLiteModel(model));
                }
                catch (error) {
                    host.exception(error, false);
                    callback(new TensorFlowLiteError(error.message), null);
                }
            });
        });
    }
}

class TensorFlowLiteModel {

    constructor(model) {
        this._graphs = [];
        this._format = 'TensorFlow Lite v' + model.version().toString();
        var description = model.description();
        this._description = (description && description.length > 0) ? description : null;
        var operatorCodeList = [];
        var builtinOperatorMap = {};
        Object.keys(tflite.BuiltinOperator).forEach(function (key) {
            var upperCase = { '2D': true, 'LSH': true, 'SVDF': true, 'RNN': true, 'L2': true, 'LSTM': true };
            var builtinOperatorIndex = tflite.BuiltinOperator[key]; 
            builtinOperatorMap[builtinOperatorIndex] = key.split('_').map((s) => {
                return (s.length < 1 || upperCase[s]) ? s : s.substring(0, 1) + s.substring(1).toLowerCase();
            }).join('');
        });
        for (var operatorIndex = 0; operatorIndex < model.operatorCodesLength(); operatorIndex++) {
            var operatorCode = model.operatorCodes(operatorIndex);
            var builtinCode = operatorCode.builtinCode();
            operatorCodeList.push((builtinCode == tflite.BuiltinOperator.CUSTOM) ? operatorCode.customCode() : builtinOperatorMap[builtinCode]);
        }
        var subgraphsLength = model.subgraphsLength();
        for (var subgraph = 0; subgraph < subgraphsLength; subgraph++) {
            var name = (subgraphsLength > 1) ? ('(' + subgraph.toString() + ')') : '';
            this._graphs.push(new TensorFlowLiteGraph(model.subgraphs(subgraph), name, operatorCodeList, model));
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
} 

class TensorFlowLiteGraph {

    constructor(graph, name, operatorCodeList, model) {
        this._graph = graph;
        this._name = this._graph.name() || name;
        this._nodes = [];
        this._inputs = [];
        this._outputs = [];
        this._operators = {};
        var connections = [];
        var names = [];
        for (var i = 0; i < graph.tensorsLength(); i++) {
            var tensor = graph.tensors(i);
            var initializer = null;
            var buffer = model.buffers(tensor.buffer());
            if (buffer.dataLength() > 0) {
                initializer = new TensorFlowLiteTensor(tensor, i, buffer);
            }
            connections.push(new TensorFlowLiteConnection(tensor, i, initializer));
            names.push(tensor.name());
        }
        for (var j = 0; j < this._graph.operatorsLength(); j++) {
            var operator = this._graph.operators(j);
            var opcodeIndex = operator.opcodeIndex();
            var operatorName = (opcodeIndex < operatorCodeList.length) ? operatorCodeList[opcodeIndex] : ('(' + opcodeIndex.toString() + ')');
            var node = new TensorFlowLiteNode(this, operator, operatorName, connections);
            this._operators[node.operator] = (this._operators[node.operator] || 0) + 1;
            this._nodes.push(node);
        }
        for (var k = 0; k < graph.inputsLength(); k++) {
            var inputIndex = graph.inputs(k);
            this._inputs.push(new TensorFlowLiteArgument(names[inputIndex], true, [ connections[inputIndex] ]));
        }
        for (var l = 0; l < graph.outputsLength(); l++) {
            var outputIndex = graph.outputs(l);
            this._outputs.push(new TensorFlowLiteArgument(names[outputIndex], true, [ connections[outputIndex] ]));
        }
    }

    get operators() {
        return this._operators;
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
}

class TensorFlowLiteNode {

    constructor(graph, node, operator, connections) {
        this._graph = graph;
        this._node = node;
        this._operator = operator;

        var inputs = TensorFlowLiteOperatorMetadata.operatorMetadata.getInputs(this._node, this.operator);
        this._inputs = inputs.map((input) => {
            return new TensorFlowLiteArgument(input.name, input.visible != false, input.connections.map((connection) => {
                return connections[connection.id];
            }));
        });
        this._outputs = [];
        for (var i = 0; i < this._node.outputsLength(); i++) {
            var index = this._node.outputs(i);
            var connection = connections[index];
            var name = TensorFlowLiteOperatorMetadata.operatorMetadata.getOutputName(this.operator, i);
            this._outputs.push(new TensorFlowLiteArgument(name, true, [ connection ]));
        }
    }

    get operator() {
        return this._operator;
    }

    get name() {
        return null;
    }

    get domain() {
        return null;
    }

    get primitive() {
        return null;
    }

    get documentation() {
        return null;
    }

    get group() {
        return null;
    }

    get category() {
        return TensorFlowLiteOperatorMetadata.operatorMetadata.getOperatorCategory(this.operator);
    }

    get inputs() {
        return this._inputs;
    }

    get outputs() {
        return this._outputs;
    }

    get dependencies() {
        return [];
    }

    get attributes() {
        if (!this._attributes) {
            this._attributes = [];
            var metadata = TensorFlowLiteOperatorMetadata.operatorMetadata;
            var node = this._node;
            var operator = this._operator;
            var optionsTypeName = 'tflite.' + operator + 'Options';
            var optionsType = eval(optionsTypeName);
            if (typeof optionsType === 'function') {
                var options = eval('new ' + optionsTypeName + '()');
                node.builtinOptions(options);
                var attributeNames = [];
                Object.keys(Object.getPrototypeOf(options)).forEach(function (attributeName) {
                    if (attributeName != '__init') {
                        attributeNames.push(attributeName);
                    }
                });
                attributeNames.forEach((name) => {
                    if (options[name] && typeof options[name] == 'function') {
                        var value = options[name]();
                        value = this.formatAttributeValue(value, name, optionsTypeName);
                        if (value != null) {
                            name = this.formatAttributeName(name);
                            var type = metadata.getAttributeType(operator, name);
                            var visible = metadata.getAttributeVisible(operator, name, value);
                            this._attributes.push(new TensorFlowLiteAttribute(name, type, value, visible));
                        }
                    }
                });
            }
        }
        return this._attributes;
    }

    formatAttributeName(name) {
        var lower = name.toLowerCase();
        var result = '';
        for (var i = 0; i < name.length; i++) {
            result += (name[i] == lower[i]) ? name[i] : ('_' + lower[i]);
        }
        return result;
    }

    formatAttributeValue(attributeValue, attributeName, optionsTypeName) {
        if (!TensorFlowLiteNode._optionsEnumTypeMap) {
            TensorFlowLiteNode._optionsEnumTypeMap = {};
            var optionsEnumTypeMap = TensorFlowLiteNode._optionsEnumTypeMap;
            optionsEnumTypeMap['tflite.Conv2DOptions'] = {
                padding: { type: tflite.Padding },
                fusedActivationFunction: { type: tflite.ActivationFunctionType }
            };
            optionsEnumTypeMap['tflite.Pool2DOptions'] = {
                padding: { type: tflite.Padding },
                fusedActivationFunction: { type: tflite.ActivationFunctionType }
            };
            optionsEnumTypeMap['tflite.DepthwiseConv2DOptions'] = {
                padding: { type: tflite.Padding },
                fusedActivationFunction: { type: tflite.ActivationFunctionType }
            };
            optionsEnumTypeMap['tflite.LSHProjectionOptions'] = {
                type: { type: tflite.LSHProjectionType }
            };
            optionsEnumTypeMap['tflite.SVDFOptions'] = {
                fusedActivationFunction: { type: tflite.ActivationFunctionType }
            };
            optionsEnumTypeMap['tflite.RNNOptions'] = {
                fusedActivationFunction: { type: tflite.ActivationFunctionType }
            };
            optionsEnumTypeMap['tflite.FullyConnectedOptions'] = {
                fusedActivationFunction: { type: tflite.ActivationFunctionType }
            };
            optionsEnumTypeMap['tflite.ConcatenationOptions'] = {
                fusedActivationFunction: { type: tflite.ActivationFunctionType }
            };
            optionsEnumTypeMap['tflite.AddOptions'] = {
                fusedActivationFunction: { type: tflite.ActivationFunctionType }
            };
            optionsEnumTypeMap['tflite.MulOptions'] = {
                fusedActivationFunction: { type: tflite.ActivationFunctionType }
            };
            optionsEnumTypeMap['tflite.L2NormOptions'] = {
                fusedActivationFunction: { type: tflite.ActivationFunctionType }
            };
            optionsEnumTypeMap['tflite.LSTMOptions'] = {
                fusedActivationFunction: { type: tflite.ActivationFunctionType }
            };
            optionsEnumTypeMap['tflite.EmbeddingLookupSparseOptions'] = {
                combiner: { type: tflite.CombinerType }
            };
        }
        var optionsEnumType = TensorFlowLiteNode._optionsEnumTypeMap[optionsTypeName];
        if (optionsEnumType) {
            var attributeType = optionsEnumType[attributeName];
            if (attributeType) {
                var map = attributeType.map;
                if (!map) {
                    map = {};
                    var enumType = attributeType.type;
                    Object.keys(enumType).forEach(function (key) {
                        map[enumType[key]] = key;
                    });
                    attributeType.map = map;
                }
                var enumValue = map[attributeValue];
                if (enumValue) {
                    return enumValue;
                }
            }
        }
        if (typeof attributeValue != 'string') {
            attributeValue = attributeValue.toString();
        }
        return attributeValue;
    }
}

class TensorFlowLiteAttribute {

    constructor(name, type, value, visible) {
        this._name = name;
        this._type = type;
        this._value = value;
        this._visible = visible;
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
        return this._visible;
    }    
}

class TensorFlowLiteArgument {
    constructor(name, visible, connections) {
        this._name = name;
        this._visible = visible;
        this._connections = connections;
    }

    get name() {
        return this._name;
    }

    get visible() {
        return this._visible;
    }

    get connections() {
        return this._connections;
    }
}

class TensorFlowLiteConnection {

    constructor(tensor, index, initializer) {
        this._id = tensor.name() || index.toString();
        this._type = initializer ? null : new TensorFlowLiteTensorType(tensor);
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
}

class TensorFlowLiteTensor {

    constructor(tensor, index, buffer) {
        this._id = index;
        this._name = tensor.name();
        this._type = new TensorFlowLiteTensorType(tensor);
        this._data = buffer.dataLength() > 0 ? buffer.dataArray() : null;
    }

    get id() {
        return this._id.toString();
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
        context.shape = this._type.shape;
        context.data = new DataView(this._data.buffer, this._data.byteOffset, this._data.byteLength);

        if (this._type.dataType == 'string') {
            var utf8Decoder = window.TextDecoder ? new TextDecoder('utf-8') : null;
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
        var size = context.shape[dimension];
        var results = [];
        if (dimension == context.shape.length - 1) {
            for (var i = 0; i < size; i++) {
                if (context.count > context.limit) {
                    results.push('...');
                    return results;
                }
                switch (context.dataType)
                {
                    case 'float32':
                        results.push(context.data.getFloat32(context.index, true));
                        context.index += 4;
                        context.count++;
                        break;
                    case 'float16':
                        results.push(TensorFlowLiteTensor._decodeNumberFromFloat16(context.data.getUint16(context.index, true)));
                        context.index += 2;
                        context.count++;
                        break;
                    case 'byte':
                        results.push(context.data.getUint8(context.index));
                        context.index += 1;
                        context.count++;
                        break;
                    case 'int32':
                        results.push(context.data.getInt32(context.index, true));
                        context.index += 4;
                        context.count++;
                        break;
                    case 'int64':
                        results.push(new Int64(context.data.getInt64(context.index, true)));
                        context.index += 8;
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
        return results;
    }

    static _decodeNumberFromFloat16(value) {
        var s = (value & 0x8000) >> 15;
        var e = (value & 0x7C00) >> 10;
        var f = value & 0x03FF;
        if(e == 0) {
            return (s ? -1 : 1) * Math.pow(2, -14) * (f / Math.pow(2, 10));
        }
        else if (e == 0x1F) {
            return f ? NaN : ((s ? -1 : 1) * Infinity);
        }
        return (s ? -1 : 1) * Math.pow(2, e-15) * (1 + (f / Math.pow(2, 10)));
    }
}

class TensorFlowLiteTensorType {

    constructor(tensor) {
        this._dataType = tensor.type();
        this._shape = [];
        var shapeLength = tensor.shapeLength();
        if (shapeLength > 0) {
            for (var i = 0; i < shapeLength; i++) {
                this._shape.push(tensor.shape(i));
            }
        }
    }

    get dataType() {
        if (!TensorFlowLiteTensorType._typeMap)
        {
            TensorFlowLiteTensorType._typeMap = {};
            TensorFlowLiteTensorType._typeMap[tflite.TensorType.FLOAT32] = 'float32';
            TensorFlowLiteTensorType._typeMap[tflite.TensorType.FLOAT16] = 'float16';
            TensorFlowLiteTensorType._typeMap[tflite.TensorType.INT32] = 'int32';
            TensorFlowLiteTensorType._typeMap[tflite.TensorType.UINT8] = 'byte';
            TensorFlowLiteTensorType._typeMap[tflite.TensorType.INT64] = 'int64';
            TensorFlowLiteTensorType._typeMap[tflite.TensorType.STRING] = 'string';
            TensorFlowLiteTensorType._typeMap[tflite.TensorType.BOOL] = 'bool';
        }
        var result = TensorFlowLiteTensorType._typeMap[this._dataType]; 
        if (result) {
            return result;
        }
        return '?';
    }

    get shape() {
        return this._shape;
    }

    toString() {
        return this.dataType + (this._shape ? ('[' + this._shape.map((dimension) => dimension.toString()).join(',') + ']') : '');
    }

}

class TensorFlowLiteOperatorMetadata {

    static open(host, callback) {
        if (TensorFlowLiteOperatorMetadata.operatorMetadata) {
            callback(null, TensorFlowLiteOperatorMetadata.operatorMetadata);
        }
        else {
            host.request(null, 'tflite-metadata.json', 'utf-8', (err, data) => {
                TensorFlowLiteOperatorMetadata.operatorMetadata = new TensorFlowLiteOperatorMetadata(data);
                callback(null, TensorFlowLiteOperatorMetadata.operatorMetadata);
            });    
        }
    }

    constructor(data) {
        this._map = {};
        if (data) {
            var items = JSON.parse(data);
            if (items) {
                items.forEach((item) => {
                    if (item.name && item.schema)
                    {
                        this._map[item.name] = item.schema;
                    }
                });
            }
        }
    }

    getSchema(operator) {
        return this._map[operator];
    }

    getInputs(node, operator) {
        var results = [];
        var connections = [];
        for (var i = 0; i < node.inputsLength(); i++) {
            connections.push(node.inputs(i));
        }
        var schema = this.getSchema(operator);
        var index = 0;
        while (index < connections.length) {
            var result = { connections: [] };
            var count = 1;
            var name = null;
            if (schema && schema.inputs && index < schema.inputs.length) {
                var input = schema.inputs[index];
                name = input.name;
                if (input.option == 'variadic') {
                    count = connections.length - index;
                }
                if (input.hasOwnProperty('visible') && !input.visible) {
                    result.visible = false;
                }
            }
            result.name = name ? name : '(' + index.toString() + ')';
            var array = connections.slice(index, index + count);
            for (var j = 0; j < array.length; j++) {
                if (array[j] != -1) {
                    result.connections.push({ id: array[j] });
                }
            }
            index += count;
            results.push(result);
        }
        return results;
    }

    getOutputName(operator, index) {
        var schema = this.getSchema(operator);
        if (schema) {
            var outputs = schema.outputs;
            if (outputs && index < outputs.length) {
                var output = outputs[index];
                if (output) {
                    if (!output.option || output.option != 'variadic') {
                        var name = output.name;
                        if (name) {
                            return name;
                        }
                    }
                } 
            }
        }
        return '(' + index.toString() + ')';
    }

    getAttributeSchema(operator, name) {
        var schema = this.getSchema(operator);
        if (schema) {
            var attributeMap = schema.attributeMap;
            if (!attributeMap) {
                attributeMap = {};
                if (schema.attributes) {
                    schema.attributes.forEach((attribute) => {
                        attributeMap[attribute.name] = attribute;
                    });
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

    getAttributeType(operator, name) {
        var attributeSchema = this.getAttributeSchema(operator, name);
        if (attributeSchema) {
            return attributeSchema.type;
        }
        return '';
    }

    getAttributeVisible(operator, name, value) {
        var attributeSchema = this.getAttributeSchema(operator, name);
        if (attributeSchema) {
            if (attributeSchema.hasOwnProperty('visible')) {
                return attributeSchema.visible;
            }
            if (attributeSchema.hasOwnProperty('default')) {
                return value != attributeSchema.default;
            }
        }
        return true;
    }

    getOperatorCategory(operator) {
        var schema = this.getSchema(operator);
        if (schema && schema.category) {
            return schema.category;
        }
        return null;
    }
}

class TensorFlowLiteError extends Error {
    constructor(message) {
        super(message);
        this.name = 'Error loading TensorFlow Lite model.';
    }
}
