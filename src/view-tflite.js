
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
    var result = [];
    for (var i = 0; i < graph.inputsLength(); i++) {
        var tensorIndex = graph.inputs(i);
        var tensor = graph.tensors(tensorIndex);
        result.push({ 
                'id': tensorIndex.toString(),
                'name': tensor.name(),
                'type': this.formatTensorType(tensor) 
            });
    }
    return result;
}

TensorFlowLiteModel.prototype.getGraphOutputs = function(graph) {
    var result = [];
    for (var i = 0; i < graph.outputsLength(); i++) {
        var tensorIndex = graph.outputs(i);
        var tensor = graph.tensors(tensorIndex);
        result.push({ 
                'id': tensorIndex.toString(),
                'name': tensor.name(),
                'type': this.formatTensorType(tensor) 
            });
    }
    return result;
}

TensorFlowLiteModel.prototype.getGraphInitializers = function(graph) {
    var result = []
    for (var i = 0; i < graph.tensorsLength(); i++) {
        var tensor = graph.tensors(i);
        if (tensor.buffer() != 0) {
            tensor = this.formatTensor(tensor);
            tensor['id'] = i.toString();
            result.push(tensor);
        }
    }
    return result;
}

TensorFlowLiteModel.prototype.getNodes = function(graph) {
    var result = []
    for (var i = 0; i < graph.operatorsLength(); i++) {
        var node = graph.operators(i);
        result.push(node);
    } 
    return result;
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
    var result = [];

    var operatorName = this.getNodeOperator(node);
    var optionsTypeName = 'tflite.' + operatorName + 'Options';
    var optionsType = eval(optionsTypeName);
    if (typeof eval(optionsTypeName) === 'function') {
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
                result.push({
                    'name': attributeName,
                    'type': '',
                    'value': function() { return value; }, 
                    'value_short': function() { return value; }
                });
            }
        });
    }
    return result;
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

TensorFlowLiteModel.prototype.formatTensor = function(tensor) {
    var result = {};
    result['name'] = tensor.name();
    result['type'] = this.formatTensorType(tensor);
    result['value'] = function () { return new TensorFlowLiteTensorFormatter(tensor).toString(); };
    return result;
}

function TensorFlowLiteTensorFormatter(tensor) {
    this.tensor = tensor;
}

TensorFlowLiteTensorFormatter.prototype.toString = function() {
    return "Not implemented (yet).";
}