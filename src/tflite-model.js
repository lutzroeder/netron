/*jshint esversion: 6 */

class TensorFlowLiteModelFactory {


    match(buffer, identifier) {
        var extension = identifier.split('.').pop();
        return extension == 'tflite';
    }

    open(buffer, identifier, host, callback) {
        host.import('/tflite.js', (err) => {
            if (err) {
                callback(err, null);
            }
            else {
                try {
                    var byteBuffer = new flatbuffers.ByteBuffer(buffer);
                    if (!tflite.Model.bufferHasIdentifier(byteBuffer))
                    {
                        callback(new TensorFlowLiteError('Invalid FlatBuffers identifier.'));
                    }
                    else {
                        var model = tflite.Model.getRootAsModel(byteBuffer);
                        model = new TensorFlowLiteModel(model);
                        TensorFlowLiteOperatorMetadata.open(host, (err, metadata) => {
                            callback(null, model);
                        });
                    }
                }
                catch (error) {
                    callback(new TensorFlowLiteError(error.message), null);
                }
            }
        });
    }
}

class TensorFlowLiteModel {

    constructor(model) {
        this._model = model;
        this._graphs = [];
        var subgraphsLength = this._model.subgraphsLength();
        for (var subgraph = 0; subgraph < subgraphsLength; subgraph++) {
            var name = (subgraphsLength > 1) ? ('(' + subgraph.toString() + ')') : '';
            this._graphs.push(new TensorFlowLiteGraph(this, this._model.subgraphs(subgraph), name));
        }
        this._activeGraph = this._graphs.length > 0 ? this._graphs[0] : null;
        this._operatorCodeList = [];
        var builtinOperatorMap = {};
        Object.keys(tflite.BuiltinOperator).forEach(function (key) {
            var upperCase = { '2D': true, 'LSH': true, 'SVDF': true, 'RNN': true, 'L2': true, 'LSTM': true };
            var builtinOperatorIndex = tflite.BuiltinOperator[key]; 
            builtinOperatorMap[builtinOperatorIndex] = key.split('_').map((s) => {
                return (s.length < 1 || upperCase[s]) ? s : s.substring(0, 1) + s.substring(1).toLowerCase();
            }).join('');
        });
        for (var operatorIndex = 0; operatorIndex < this._model.operatorCodesLength(); operatorIndex++) {
            var operatorCode = this._model.operatorCodes(operatorIndex);
            var builtinCode = operatorCode.builtinCode();
            this._operatorCodeList.push((builtinCode == tflite.BuiltinOperator.CUSTOM) ? operatorCode.customCode() : builtinOperatorMap[builtinCode]);
        }
    }

    get properties() {
        var results = [];

        var format = 'TensorFlow Lite v' + this._model.version().toString();
        results.push({ name: 'Format', value: format });

        var description = this._model.description();
        if (description && description.length > 0) {
            results.push({ name: 'Description', value: description });
        }

        return results;
    }

    get graphs() {
        return this._graphs;
    }
} 

class TensorFlowLiteGraph {

    constructor(model, graph, name) {
        this._model = model;
        this._graph = graph;
        this._name = this._graph.name();
        if (!this._name) {
            this._name = name;
        }
        this._initializerMap = {};
        for (var i = 0; i < graph.tensorsLength(); i++) {
            var tensor = graph.tensors(i);
            var buffer = model._model.buffers(tensor.buffer());
            if (buffer.dataLength() > 0) {
                this._initializerMap[i] = new TensorFlowLiteTensor(tensor, buffer, i);
            }
        }
    }

    get model() {
        return this._model;
    }

    get name() {
        return this._name;
    }

    get groups() {
        return false;
    }

    get inputs() {
        var results = [];
        var graph = this._graph;
        for (var i = 0; i < graph.inputsLength(); i++) {
            var tensorIndex = graph.inputs(i);
            var tensor = graph.tensors(tensorIndex);
            results.push({ 
                id: tensorIndex.toString(),
                name: tensor.name(),
                type: TensorFlowLiteTensor.formatTensorType(tensor) 
            });
        }
        return results;
    }

    get outputs() {
        var results = [];
        var graph = this._graph;
        for (var i = 0; i < graph.outputsLength(); i++) {
            var tensorIndex = graph.outputs(i);
            var tensor = graph.tensors(tensorIndex);
            results.push({ 
                id: tensorIndex.toString(),
                name: tensor.name(),
                type: TensorFlowLiteTensor.formatTensorType(tensor) 
            });
        }
        return results;
    }

    get nodes() {
        var results = [];
        for (var i = 0; i < this._graph.operatorsLength(); i++) {
            var node = this._graph.operators(i);
            results.push(new TensorFlowLiteNode(this, node));
        } 
        return results;
    }

    getInitializer(tensorIndex) {
        var initializer = this._initializerMap[tensorIndex];
        return initializer ? initializer : null;
    }
}

class TensorFlowLiteNode {

    constructor(graph, node) {
        this._graph = graph;
        this._node = node;
    }

    get operator() {
        if (!this._operator) {
            var operatorCodeList = this._graph.model._operatorCodeList;
            var opcodeIndex = this._node.opcodeIndex();
            this._operator = (opcodeIndex < operatorCodeList.length) ?
                operatorCodeList[opcodeIndex] :
                ('(' + opcodeIndex.toString() + ')');
        }
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
        var inputs = TensorFlowLiteOperatorMetadata.operatorMetadata.getInputs(this._node, this.operator);
        inputs.forEach((input) => {
            input.connections.forEach((connection) => {
                var tensorIndex = connection.id;
                var tensor = this._graph._graph.tensors(tensorIndex);
                connection.type = TensorFlowLiteTensor.formatTensorType(tensor);
                var initializer = this._graph.getInitializer(tensorIndex);
                if (initializer) {
                    connection.initializer = initializer;
                }
                connection.id = connection.id.toString();
            });
        });
        return inputs;
    }

    get outputs() {
        var results = [];
        var graph = this._graph._graph;
        var node = this._node;
        for (var i = 0; i < node.outputsLength(); i++) {
            var tensorIndex = node.outputs(i);
            var tensor = graph.tensors(tensorIndex);
            var output = {
                name: TensorFlowLiteOperatorMetadata.operatorMetadata.getOutputName(this.operator, i),
                connections: []
            };
            var connection = {};
            connection.id = tensorIndex.toString();
            connection.type = TensorFlowLiteTensor.formatTensorType(tensor);
            var initializer = this._graph.getInitializer(tensorIndex);
            if (initializer) {
                connection.initializer = initializer;
            }
            output.connections.push(connection);
            results.push(output);
        }
        return results;
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

class TensorFlowLiteTensor {

    constructor(tensor, buffer, index) {
        this._id = index;
        this._tensor = tensor;
        this._buffer = buffer;
    }

    get id() {
        return this._id.toString();
    }

    get name() {
        return this._tensor.name();
    }

    get type() {
        return TensorFlowLiteTensor.formatTensorType(this._tensor);
    }

    get quantization() {
        var quantization = this._tensor.quantization();
        if (quantization) {
            var scale = (quantization.scaleLength() == 1) ? quantization.scale(0) : 0;
            var zeroPoint = (quantization.zeroPointLength() == 1) ? quantization.zeroPoint(0).toFloat64() : 0;
            if (scale != 0 || zeroPoint != 0) {
                return 'f = ' + scale.toString() + ' * ' + (zeroPoint == 0 ? 'q' : ('(q - ' + zeroPoint.toString() + ')'));
            }
        }
        return null;
    }

    get value() {

        if (this._buffer.dataLength() == 0) {
            return 'Tensor data is empty.';
        }

        var array = this._buffer.dataArray();
        this._data = new DataView(array.buffer, array.byteOffset, array.byteLength);

        if (this._tensor.type() == tflite.TensorType.STRING) {
            var utf8Decoder = window.TextDecoder ? new TextDecoder('utf-8') : null;
            var offset = 0;
            var count = this._data.getInt32(0, true);
            offset += 4;
            var offsetTable = [];
            for (var j = 0; j < count; j++) {
                offsetTable.push(this._data.getInt32(offset, true));
                offset += 4;
            }
            offsetTable.push(array.length);
            var stringTable = [];
            for (var k = 0; k < count; k++) {
                var textArray = array.subarray(offsetTable[k], offsetTable[k + 1]);
                if (utf8Decoder) {
                    stringTable.push(utf8Decoder.decode(textArray));
                }
                else {
                    stringTable.push(String.fromCharCode.apply(null, textArray));
                }
            }
            this._data = stringTable;
        }

        this._index = 0;                
        this._count = 0;
        var result = this.read(0);

        delete this._index;
        delete this._count;  
        delete this._data;
        delete this._utf8Decoder;

        return JSON.stringify(result, null, 4);
    }

    read(dimension) {
        var size = this._tensor.shape(dimension);
        var results = [];
        if (dimension == this._tensor.shapeLength() - 1) {
            for (var i = 0; i < size; i++) {
                if (this._count > 10000) {
                    results.push('...');
                    return results;
                }
                switch (this._tensor.type())
                {
                    case tflite.TensorType.FLOAT32:
                        results.push(this._data.getFloat32(this._index, true));
                        this._index += 4;
                        this._count++;
                        break;
                    case tflite.TensorType.FLOAT16:
                        results.push(TensorFlowLiteTensor.decodeNumberFromFloat16(this._data.getUint16(this._index, true)));
                        this._index += 2;
                        this._count++;
                        break;
                    case tflite.TensorType.UINT8:
                        results.push(this._data.getUint8(this._index));
                        this._index += 1;
                        this._count++;
                        break;
                    case tflite.TensorType.INT32:
                        results.push(this._data.getInt32(this._index, true));
                        this._index += 4;
                        this._count++;
                        break;
                    case tflite.TensorType.INT64:
                        results.push(new Int64(this._data.getInt64(this._index, true)));
                        this._index += 8;
                        this._count++;
                        break;
                    case tflite.TensorType.STRING:
                        results.push(this._data[this._index++]);
                        this._count++;
                        break;
                    default:
                        break;
                }
            }
        }
        else {
            for (var j = 0; j < size; j++) {
                if (this._count > 10000) {
                    results.push('...');
                    return results;
                }
                results.push(this.read(dimension + 1));
            }
        }
        return results;
    }

    static decodeNumberFromFloat16(value) {
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

    static formatTensorType(tensor) {
        if (!TensorFlowLiteTensor.tensorTypeMap)
        {
            var map = {};
            map[tflite.TensorType.FLOAT32] = 'float';
            map[tflite.TensorType.FLOAT16] = 'float16';
            map[tflite.TensorType.INT32] = 'int32';
            map[tflite.TensorType.UINT8] = 'byte';
            map[tflite.TensorType.INT64] = 'int64';
            map[tflite.TensorType.STRING] = 'string';
            map[tflite.TensorType.BOOL] = 'bool';
            TensorFlowLiteTensor.tensorTypeMap = map;
        }
        var result = TensorFlowLiteTensor.tensorTypeMap[tensor.type()]; 
        if (!result) {
            debugger;
            result = '?';
        }
        var shapeLength = tensor.shapeLength();
        if (shapeLength > 0) {
            var dimensions = [];
            for (var i = 0; i < shapeLength; i++) {
                dimensions.push(tensor.shape(i).toString());
            }
            result += '[' + dimensions.join(',') + ']';
        }
        return result;
    }

}

class TensorFlowLiteOperatorMetadata {

    static open(host, callback) {
        if (TensorFlowLiteOperatorMetadata.operatorMetadata) {
            callback(null, TensorFlowLiteOperatorMetadata.operatorMetadata);
        }
        else {
            host.request('/tflite-metadata.json', (err, data) => {
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
                    result.hidden = true;
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
        this.name = 'TensorFlow Lite Error';
    }
}
