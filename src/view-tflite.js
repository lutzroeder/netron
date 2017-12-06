
// Experimental

function TensorFlowLiteModel(hostService) {
    this.operatorService = null;
}

TensorFlowLiteModel.prototype.openBuffer = function(buffer, identifier) { 
    try {
        var byteBuffer = new flatbuffers.ByteBuffer(buffer);
        if (!tflite.Model.bufferHasIdentifier(byteBuffer))
        {
            throw 'Invalid identifier';
        }
        this.model = tflite.Model.getRootAsModel(byteBuffer);
        this.initialize();
    }
    catch (err) {
        return err;
    }
    return null;
}

TensorFlowLiteModel.prototype.initialize = function() {
    if (!this.model) {
        return;
    }
    var builtinOperatorMap = {}
    this.operatorCodeList = []
    Object.keys(tflite.BuiltinOperator).forEach(function (key) {
        var upperCase = { '2D': true, 'LSH': true, 'SVDF': true, 'RNN': true, 'L2': true, 'LSTM': true };
        var operatorName = key.split('_').map(function (s) {
            if (s.length < 1 || upperCase[s]) {
                return s;
            }
            return s.substring(0, 1) + s.substring(1).toLowerCase();
        }).join('');
        builtinOperatorMap[tflite.BuiltinOperator[key]] = operatorName;
    });
    for (var i = 0; i < this.model.operatorCodesLength(); i++) {
        var operatorCode = this.model.operatorCodes(i);
        var builtinCode = operatorCode.builtinCode();
        this.operatorCodeList.push((builtinCode == tflite.BuiltinOperator.CUSTOM) ?
            operatorCode.customCode() :
            builtinOperatorMap[builtinCode]);
    }
}

TensorFlowLiteModel.prototype.formatModelProperties = function() {
    
    var result = { 'groups': [] };

    var modelProperties = { 'name': 'Model', 'properties': [] };
    result['groups'].push(modelProperties);

    var format = 'TensorFlow Lite v' + this.model.version().toString();
    modelProperties['properties'].push({ 'name': 'Format', 'value': format });

    var description = this.model.description();
    if (description && description.length > 0) {
        modelProperties['properties'].push({ 'name': 'Description', 'value': description });
    }

    return result;
}

TensorFlowLiteModel.prototype.getGraph = function(index) {
    if (index < this.model.subgraphsLength()) {
        return this.model.subgraphs(0);
    }
    return null;
}

TensorFlowLiteModel.prototype.getGraphInputs = function(graph) {
    var results = [];
    for (var i = 0; i < graph.inputsLength(); i++) {
        var tensorIndex = graph.inputs(i);
        var tensor = graph.tensors(tensorIndex);
        results.push({ 
                'id': tensorIndex.toString(),
                'name': tensor.name(),
                'type': this.formatTensorType(tensor) 
            });
    }
    return results;
}

TensorFlowLiteModel.prototype.getGraphOutputs = function(graph) {
    var results = [];
    for (var i = 0; i < graph.outputsLength(); i++) {
        var tensorIndex = graph.outputs(i);
        var tensor = graph.tensors(tensorIndex);
        results.push({ 
                'id': tensorIndex.toString(),
                'name': tensor.name(),
                'type': this.formatTensorType(tensor) 
            });
    }
    return results;
}

TensorFlowLiteModel.prototype.getGraphInitializers = function(graph) {
    var results = []
    for (var i = 0; i < graph.tensorsLength(); i++) {
        var tensor = graph.tensors(i);
        var buffer = this.model.buffers(tensor.buffer());
        if (buffer.dataLength() > 0) {
            tensor = this.formatTensor(tensor, buffer);
            tensor['id'] = i.toString();
            results.push(tensor);
        }
    }
    return results;
}

TensorFlowLiteModel.prototype.getNodes = function(graph) {
    /* for (var i = 0; i < graph.operatorsLength(); i++) {
        var node = graph.operators(i);
        var inputs = [];
        for (var j = 0; j < node.inputsLength(); j++) {
            inputs.push(node.inputs(j));
        }
        var outputs = [];
        for (var j = 0; j < node.outputsLength(); j++) {
            outputs.push(node.outputs(j));
        }
        console.log(this.getNodeOperator(node) + ' [' + inputs.join(',') + '] -> [' + outputs.join(',') + ']');
    } */
    var results = []
    for (var i = 0; i < graph.operatorsLength(); i++) {
        var node = graph.operators(i);
        results.push(node);
    } 
    return results;
}

TensorFlowLiteModel.prototype.getNodeOperator = function(node) {
    var opcodeIndex = node.opcodeIndex();
    if (opcodeIndex < this.operatorCodeList.length) {
        return this.operatorCodeList[opcodeIndex];
    }
    return '(' + opcodeIndex.toString() + ')';
}

TensorFlowLiteModel.prototype.getNodeOperatorDocumentation = function(graph, node) {
    return null;
}

TensorFlowLiteModel.prototype.getNodeInputs = function(graph, node) {
    var result = [];
    for (var i = 0; i < node.inputsLength(); i++) {
        var tensorIndex = node.inputs(i);
        var tensor = graph.tensors(tensorIndex);
        result.push({
            'id': tensorIndex.toString(),
            'name': '(' + i.toString() + ')',
            'type': this.formatTensorType(tensor)
        });
    }
    return result;
}

TensorFlowLiteModel.prototype.getNodeOutputs = function(graph, node) {
    var result = [];
    for (var i = 0; i < node.outputsLength(); i++) {
        var tensorIndex = node.outputs(i);
        var tensor = graph.tensors(tensorIndex);
        result.push({
            'id': tensorIndex.toString(),
            'name': '(' + i.toString() + ')',
            'type': this.formatTensorType(tensor)
        });
    }
    return result;
}

TensorFlowLiteModel.prototype.formatNodeProperties = function(node) {
    return [];
}

TensorFlowLiteModel.prototype.formatNodeAttributes = function(node) {
    var self = this;
    var results = [];
    var operatorName = this.getNodeOperator(node);
    var optionsTypeName = 'tflite.' + operatorName + 'Options';
    var optionsType = eval(optionsTypeName);
    if (typeof optionsType === 'function') {
        var options = eval('new ' + optionsTypeName + '()');
        node.builtinOptions(options);
        var attributeNames = [];
        Object.keys(options.__proto__).forEach(function (attributeName) {
            if (attributeName != '__init') {
                attributeNames.push(attributeName);
            }
        });
        attributeNames.forEach(function (attributeName) {
            if (options[attributeName] && typeof options[attributeName] == 'function') {
                var value = options[attributeName]();
                value = self.formatAttributeValue(value, attributeName, optionsTypeName);
                if (value != null) {
                    results.push({
                        'name': attributeName,
                        'type': '',
                        'value': function() { return value; }, 
                        'value_short': function() { return value; }
                    });
                }
            }
        });
    }
    return results;
}

TensorFlowLiteModel.prototype.formatTensorType = function(tensor) {
    if (!this.tensorTypeMap)
    {
        this.tensorTypeMap = {};
        this.tensorTypeMap[tflite.TensorType.FLOAT32] = 'float';
        this.tensorTypeMap[tflite.TensorType.FLOAT16] = 'float16';
        this.tensorTypeMap[tflite.TensorType.INT32] = 'int32';
        this.tensorTypeMap[tflite.TensorType.UINT8] = 'byte';
        this.tensorTypeMap[tflite.TensorType.INT64] = 'int64';
        this.tensorTypeMap[tflite.TensorType.STRING] = 'string';
    }
    var result = this.tensorTypeMap[tensor.type()]; 
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

TensorFlowLiteModel.prototype.formatTensor = function(tensor, buffer) {
    var result = {};
    result['name'] = tensor.name();
    result['type'] = this.formatTensorType(tensor);
    result['value'] = function () { return new TensorFlowLiteTensorFormatter(tensor, buffer).toString(); };
    return result;
}

TensorFlowLiteModel.prototype.formatAttributeValue = function(attributeValue, attributeName, optionsTypeName) {
    if (!this.optionsEnumTypeMap) {
        this.optionsEnumTypeMap = {};
        this.optionsEnumTypeMap['tflite.Conv2DOptions'] = {
            'padding': { 'type': tflite.Padding },
            'fusedActivationFunction': { 'type': tflite.ActivationFunctionType, 'default': 'NONE' }
        },
        this.optionsEnumTypeMap['tflite.Pool2DOptions'] = {
            'padding': { 'type': tflite.Padding },
            'fusedActivationFunction': { 'type': tflite.ActivationFunctionType, 'default': 'NONE' }
        },
        this.optionsEnumTypeMap['tflite.DepthwiseConv2DOptions'] = {
            'padding': { 'type': tflite.Padding },
            'fusedActivationFunction': { 'type': tflite.ActivationFunctionType, 'default': 'NONE' }
        },
        this.optionsEnumTypeMap['tflite.LSHProjectionOptions'] = {
            'type': { 'type': tflite.LSHProjectionType }
        },
        this.optionsEnumTypeMap['tflite.SVDFOptions'] = {
            'fusedActivationFunction': { 'type': tflite.ActivationFunctionType, 'default': 'NONE' }
        },
        this.optionsEnumTypeMap['tflite.RNNOptions'] = {
            'fusedActivationFunction': { 'type': tflite.ActivationFunctionType, 'default': 'NONE' }
        },
        this.optionsEnumTypeMap['tflite.FullyConnectedOptions'] = {
            'fusedActivationFunction': { 'type': tflite.ActivationFunctionType, 'default': 'NONE' }
        },
        this.optionsEnumTypeMap['tflite.ConcatenationOptions'] = {
            'fusedActivationFunction': { 'type': tflite.ActivationFunctionType, 'default': 'NONE' }
        },
        this.optionsEnumTypeMap['tflite.AddOptions'] = {
            'fusedActivationFunction': { 'type': tflite.ActivationFunctionType, 'default': 'NONE' }
        },
        this.optionsEnumTypeMap['tflite.MulOptions'] = {
            'fusedActivationFunction': { 'type': tflite.ActivationFunctionType, 'default': 'NONE' }
        },
        this.optionsEnumTypeMap['tflite.L2NormOptions'] = {
            'fusedActivationFunction': { 'type': tflite.ActivationFunctionType, 'default': 'NONE' }
        },
        this.optionsEnumTypeMap['tflite.LSTMOptions'] = {
            'fusedActivationFunction': { 'type': tflite.ActivationFunctionType, 'default': 'NONE' }
        },
        this.optionsEnumTypeMap['tflite.EmbeddingLookupSparseOptions'] = {
            'combiner': { 'type': tflite.CombinerType },
        }
    }
    var optionsEnumType = this.optionsEnumTypeMap[optionsTypeName];
    if (optionsEnumType) {
        var attributeType = optionsEnumType[attributeName];
        if (attributeType) {
            var map = attributeType['map'];
            if (!map) {
                map = {};
                var enumType = attributeType['type'];
                Object.keys(enumType).forEach(function (key) {
                    map[enumType[key]] = key;
                });
                attributeType['map'] = map;
            }
            var enumValue = map[attributeValue];
            if (enumValue) {
                var defaultValue = attributeType['default'];
                if (defaultValue && defaultValue == enumValue) {
                    return null;
                }
                return enumValue;
            }
        }
    }
    return attributeValue;
}
    

function TensorFlowLiteTensorFormatter(tensor, buffer) {
    this.tensor = tensor;
    this.buffer = buffer;
    if (window['TextDecoder']) {
        this.utf8Decoder = new TextDecoder('utf-8');
    }
}

TensorFlowLiteTensorFormatter.prototype.toString = function() {
    var size = 1;
    for (var i = 0; i < this.tensor.shapeLength(); i++) {
        size *= this.tensor.shape(i);
    }
    if (size > 65536) {
        return 'Tensor is too large to display.' 
    }

    if (this.buffer.dataLength() == 0) {
        return 'Tensor data is empty.';
    }

    var array = this.buffer.dataArray();
    this.data = new DataView(array.buffer, array.byteOffset, array.byteLength);

    if (this.tensor.type() == tflite.TensorType.STRING) {
        var offset = 0;
        var count = this.data.getInt32(0, true);
        offset += 4;
        var offsetTable = [];
        for (var i = 0; i < count; i++) {
            offsetTable.push(this.data.getInt32(offset, true));
            offset += 4;
        }
        offsetTable.push(array.length);
        var stringTable = [];
        for (var i = 0; i < count; i++) {
            var textArray = array.subarray(offsetTable[i], offsetTable[i + 1]);
            if (this.utf8Decoder) {
                stringTable.push(this.utf8Decoder.decode(textArray));
            }
            else {
                stringTable.push(String.fromCharCode.apply(null, textArray));
            }
        }
        this.data = stringTable;
    }

    this.index = 0;                
    var result = this.read(0);
    this.data = null;

    return JSON.stringify(result, null, 4);
};

TensorFlowLiteTensorFormatter.prototype.read = function(dimension) {
    var size = this.tensor.shape(dimension);
    var results = [];
    if (dimension == this.tensor.shapeLength() - 1) {
        for (var i = 0; i < size; i++) {
            switch (this.tensor.type())
            {
                case tflite.TensorType.FLOAT32:
                    results.push(this.data.getFloat32(this.index, true));
                    this.index += 4;
                    break;
                case tflite.TensorType.FLOAT16:
                    results.push(this.decodeNumberFromFloat16(this.data.getUint16(this.index, true)))
                    this.index += 2;
                    break;
                case tflite.TensorType.UINT8:
                    results.push(this.data.getUint8(this.index));
                    this.index += 4;
                    break;
                case tflite.TensorType.INT32:
                    results.push(this.data.getInt32(this.index, true));
                    this.index += 4;
                    break;
                case tflite.TensorType.INT64:
                    results.push(new Int64(this.data.getInt64(this.index, true)));
                    this.index += 8;
                    break;
                case tflite.TensorType.STRING:
                    results.push(this.data[this.index++]);
                    break;
                default:
                    debugger;
            }
        }
    }
    else {
        for (var i = 0; i < size; i++) {
            results.push(this.read(dimension + 1));
        }
    }
    return results;
};

TensorFlowLiteTensorFormatter.prototype.decodeNumberFromFloat16 = function(value) {
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
