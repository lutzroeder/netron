/*jshint esversion: 6 */

// Experimental

var tensorflow = null;

class TensorFlowModel {

    static open(buffer, identifier, host, callback) { 
        host.import('/tf.js', (err) => {
            if (err) {
                callback(err, null);
            }
            else {
                tensorflow = protobuf.roots.tf.tensorflow;
                var model = TensorFlowModel.create(buffer, identifier, host, (err, model) => {
                    callback(err, model);
                });
            }
        });
    }

    static create(buffer, identifier, host, callback) {
        try {
            var model = null;
            var format = null;
            if (identifier == 'saved_model.pb') {
                model = tensorflow.SavedModel.decode(buffer);
                format = 'TensorFlow Saved Model';
                if (model.savedModelSchemaVersion) {
                    format = format + ' v' + model.savedModelSchemaVersion.toString();
                }
            }
            else {
                var metaGraphDef = null;
                var extension = identifier.split('.').pop();
                if (extension != 'meta') {
                    try {
                        var graphDef = tensorflow.GraphDef.decode(buffer);
                        metaGraphDef = new tensorflow.MetaGraphDef();
                        metaGraphDef.graphDef = graphDef;
                        metaGraphDef.anyInfo = identifier;
                        format = 'TensorFlow Graph';
                    }
                    catch (err) {
                    }
                }

                if (!metaGraphDef) {
                    metaGraphDef = tensorflow.MetaGraphDef.decode(buffer);
                    format = 'TensorFlow MetaGraph';
                }

                model = new tensorflow.SavedModel();
                model.metaGraphs.push(metaGraphDef);
            }

            var result = new TensorFlowModel(model, format);

            TensorFlowOperatorMetadata.open(host, (err, metadata) => {
                callback(null, result);
            });
        }
        catch (err) {
            callback(err, null);
        }    
    }

    constructor(model, format) {
        this._model = model;
        this._format = format;
        this._graphs = [];
        for (var i = 0; i < this._model.metaGraphs.length; i++) {
            this._graphs.push(new TensorFlowGraph(this, this._model.metaGraphs[i], i));
        }
        this._activeGraph = (this._graphs.length > 0) ? this._graphs[0] : null;
    }

    get properties() {
        var results = [];
        results.push({ name: 'Format', value: this._format });
        return results;
    }

    get graphs() {
        return this._graphs;    
    }
}

class TensorFlowGraph {

    constructor(model, graph, index) {
        this._model = model;
        this._graph = graph;
        this._metadata = new TensorFlowGraphOperatorMetadata(graph.metaInfoDef);
        this._name = this._graph.anyInfo ? this._graph.anyInfo.toString() : ('(' + index.toString() + ')');

        // metaInfoDef.tensorflowGitVersion
        // TODO signature
    }

    get model() {
        return this._model;
    }

    get name() {
        return this._name;
    }

    get version() {
        if (this._graph.metaInfoDef && this._graph.metaInfoDef.tensorflowVersion) {
            return this._graph.metaInfoDef.tensorflowVersion;
        }
        return null;
    }

    get tags() {
        if (this._graph.metaInfoDef && this._graph.metaInfoDef.tags) {
            return this._graph.metaInfoDef.tags.join(', ');
        }
        return null;
    }

    get inputs() {
        this.update();
        var results = [];
        Object.keys(this._inputMap).forEach((key) => {
            results.push(this._inputMap[key]);
        });
        return results;
    }

    get outputs() {
        this.update();
        return [];
    }

    get nodes() {
        this.update();
        var results = [];
        this._graph.graphDef.node.forEach((node) => {
            if (node.output.filter(output => !output.startsWith('^')) != 0 ||
                node.input.filter(input => !input.startsWith('^')).length > 0) {
                var id = node.name + ':0';
                if (!this._initializerMap[id] && !this._inputMap[id] /* && node.op != 'NoOp' */) {
                    results.push(new TensorFlowNode(this, node));
                }
            }
        });
        return results;
    }

    get metadata() {
        return this._metadata;
    }

    update() {
        if (!this._nodeMap) {
            this._nodeMap = {};
            var nodes = this._graph.graphDef.node;
            nodes.forEach((node) => {
                this._nodeMap[node.name] = node;   
                node.output = [];         
            });
            nodes.forEach((node) => {
                for (var i = 0; i < node.input.length; i++)
                {
                    var split = node.input[i].split(':', 1);
                    var inputName = split[0];
                    if (!inputName.startsWith('^')) {
                        var outputIndex = split.length == 1 ? 0 : parseInt(split[1]);
                        var outputName = inputName;
                        var outputNode = this._nodeMap[outputName];
                        node.input[i] = inputName + ':' + outputIndex.toString();
                        if (outputNode) {
                            for (var j = outputNode.output.length; j <= outputIndex; j++) {
                                outputNode.output.push('');
                            }
                            outputNode.output[outputIndex] = node.input[i];
                        }    
                    }
                    else {
                        var sourceName = inputName.substring(1);
                        var sourceNode = this._nodeMap[sourceName];
                        if (sourceNode) {
                            if (!sourceNode.dependency) {
                                sourceNode.dependency = [];
                            }
                            sourceNode.dependency.push({ 
                                id: inputName, 
                                name: node.name, 
                                operator: node.op
                            });
                        }
                    }
                }
            });
            this._nodeOutputCountMap = {};
            nodes.forEach((node) => {
                this._metadata.getInputs(node).forEach((input) => {
                    var multiple = input.connections.length > 1; 
                    input.connections.forEach((connection) => {
                        if (multiple) {
                            this._nodeOutputCountMap[connection.id] = 'N';
                        }
                        else {
                            var id = connection.id.startsWith('^') ? connection.id.substring(1) : connection.id;
                            var count = this._nodeOutputCountMap[id];
                            if (count != 'N') {
                                count = count ? count : 0;
                                this._nodeOutputCountMap[input] = count + 1;
                            }
                        }
                    });
                });

                node.input.forEach((input) => {
                    input = input.startsWith('^') ? input.substring(1) : input;
                    var count = this._nodeOutputCountMap[input];
                    if (!count) {
                        count = 0;
                    }
                    this._nodeOutputCountMap[input] = count + 1;
                });
            });

            this._initializerMap = {};
            this._graph.graphDef.node.forEach((node) => {
                if (node.op == 'Const' && node.input.length == 0 && this.checkSingleOutput(node)) {
                    var value = node.attr['value'];
                    if (value && value.hasOwnProperty('tensor')) {
                        var output = node.output[0];
                        if (output) {
                            this._initializerMap[output] = new TensorFlowTensor(value.tensor, output, node.name, 'Constant');
                        }
                    }
                }
            });
            this._graph.graphDef.node.forEach((node) => {
                if (node.op == 'Identity' && node.input.length == 1 && this.checkSingleOutput(node)) {
                    var input = node.input[0];
                    var tensor = this._initializerMap[input];
                    if (tensor) {
                        var output = node.output[0];
                        this._initializerMap[input] = "-";
                        this._initializerMap[output] = new TensorFlowIdentityTensor(tensor, output, node.name, 'Identity Constant');
                    }
                }
            });

            this._inputMap = {};
            this._graph.graphDef.node.forEach((node) => {
                if (node.op == 'Placeholder' && node.input.length == 0 && node.output.length == 1) {
                    var dtype = node.attr['dtype'];
                    var shape = node.attr['shape'];
                    if (dtype && dtype.type && shape && shape.shape) {
                        this._inputMap[node.output[0]] = {
                            id: node.output[0],
                            name: node.name,
                            type: TensorFlowTensor.formatDataType(dtype.type) + TensorFlowTensor.formatTensorShape(shape.shape)
                        };
                    }
                }
            });
        }
    }

    getInitializer(input) {
        var initializer = this._initializerMap[input];
        return initializer ? initializer : null;
    }

    checkSingleOutput(node) { 
        if (node.output.length != 1) {
            return false;
        }
        var output = node.output[0];
        var count = this._nodeOutputCountMap[output];
        if (count != 1) {
            return false;
        }
        return true;
    }
}

class TensorFlowNode {

    constructor(graph, node) {
        this._graph = graph;
        this._node = node;
    }

    get graph() {
        return this._graph;
    }

    get operator() {
        return this._node.op;
    }

    get name() {
        return this._node.name;
    }

    get description() {
        return null;
    }

    get primitive() {
        /*
        switch (this._node.op) {
            case 'Add': return '+';
            case 'Mul': return '*';
            case 'Sub': return '-';
            case 'Identity': return 'I';
        }
        */
        return null;
    }

    get domain() {
        return null;
    }

    get documentation() {
        return this._graph.metadata.getOperatorDocumentation(this.operator);       
    }

    get category() {
        return this._graph.metadata.getOperatorCategory(this.operator);       
    }

    get inputs() {
        if (this._node.input) {
            var inputs = this._graph.metadata.getInputs(this._node);
            inputs.forEach((input) => {
                input.connections.forEach((connection) => {
                    var initializer = this._graph.getInitializer(connection.id);
                    if (initializer) {
                        connection.type = initializer.type;
                        connection.initializer = initializer;
                    }
                });
            });          
            return inputs;
        }
        return [];
    }

    get outputs() {
        return this._graph.metadata.getOutputs(this._node);
    }

    get dependencies() {
        var results = [];
        if (this._node.dependency) {
            this._node.dependency.forEach((dependency) => {
                results.push(dependency);
            });
        }
        return results;
    }

    get attributes() {
        var graphMetadata = this._graph.metadata;
        var node = this._node;
        var result = [];
        if (node.attr) {
            var hiddenAttributeMap = graphMetadata.getHiddenAttributeMap(node.op);
            Object.keys(node.attr).forEach((name) => {
                var hidden = hiddenAttributeMap[name] == true;
                var value = node.attr[name];
                result.push(new TensorFlowAttribute(this, name, value, hidden));
            });
        }
        return result;
    }
}

class TensorFlowAttribute { 
    constructor(node, name, value, hidden) {
        this._node = node;
        this._name = name;
        this._value = value;
        if (hidden) {
            this._hidden = hidden;
        }
    }

    get name() {
        return this._name;
    }

    get type() {
        if (this._value.hasOwnProperty('tensor')) {
            return TensorFlowTensor.formatTensorType(this._value.tensor);
        }
        var graphMetadata = this._node.graph.metadata;
        if (graphMetadata) {
            return graphMetadata.getAttributeType(this._node.operator, this._name);
        }
        return '';
    }

    get value() {
        var item = TensorFlowAttribute.formatAttributeValue(this._value);
        if (Array.isArray(item)) {
            return item.join(', ');
        }
        return item;
    }

    static formatAttributeValue(value) {
        if (value.hasOwnProperty('type')) {
            return TensorFlowTensor.formatDataType(value.type);
        }
        else if (value.hasOwnProperty('i')) {
            return value.i.toString();
        }
        else if (value.hasOwnProperty('f')) {
            return value.f.toString();
        }
        else if (value.hasOwnProperty('b')) {
            return value.b.toString();
        }
        else if (value.hasOwnProperty('shape')) {
            return TensorFlowTensor.formatTensorShape(value.shape);;
        }
        else if (value.hasOwnProperty('s')) {
            if (value.s.filter(c => c <= 32 && c >= 128).length == 0) {
                return '"' + String.fromCharCode.apply(null, value.s) + '"';
            }
            return value.s.map(v => v.toString()).join(', ');           
        }
        else if (value.hasOwnProperty('tensor')) {
            return new TensorFlowTensor(value.tensor).value;
        }
        else if (value.hasOwnProperty('list')) {
            var list = value.list;
            if (list.s && list.s.length > 0) {
                if (list.s.length > 65536) {
                    return "Too large to render.";
                }
                return list.s.map((s) => {
                    if (s.filter(c => c <= 32 && c >= 128).length == 0) {
                        return '"' + String.fromCharCode.apply(null, s) + '"';
                    }
                    return s.map(v => v.toString()).join(', ');    
                });
            }
            else if (list.i && list.i.length > 0) {
                if (list.i.length > 65536) {
                    return "Too large to render.";
                }
                return list.i.map((v) => v.toString());
            }
            else if (list.f && list.f.length > 0) {
                if (list.f.length > 65536) {
                    return "Too large to render.";
                }
                return list.f.map((v) => v.toString());
            }
            else if (list.type && list.type.length > 0) {
                if (list.type.length > 65536) {
                    return "Too large to render.";
                }
                return list.type.map((type) => TensorFlowTensor.formatDataType(type));
            }
            else if (list.shape && list.shape.length > 0) {
                if (list.shape.length > 65536) {
                    return "Too large to render.";
                }
                return list.shape.map((shape) => TensorFlowTensor.formatTensorShape(shape));
            }
        }
        debugger;
        return '';        
    }

    get hidden() {
        return this._hidden ? this._hidden : false;
    }

    get tensor() {
        if (this._value.hasOwnProperty('tensor')) {
            if (this._value.tensor.tensorShape && this._value.tensor.tensorShape.dim) {
                if (this._value.tensor.tensorShape.dim.length == 0) {
                    return false;
                }
            }
            return true;
        }
        return false;
    }
}

class TensorFlowIdentityTensor {
    
    constructor(tensor, id, name, title) {
        this._tensor = tensor;
        this._id = id;
        this._name = name;
        if (title) {
            this._title = title;
        }
    }

    get id() {
        return this._id;
    }

    get name() {
        return this._name;
    }

    get title() {
        return this._title;
    }

    get type() {
        return this._tensor.type;
    }

    get value() {
        return this._tensor.value;
    }
}

class TensorFlowTensor {

    constructor(tensor, id, name, title) {
        this._tensor = tensor;
        this._id = id;
        this._name = name;
        if (title) {
            this._title = title;
        }
    }

    get id() {
        return this._id;
    }

    get name() {
        return this._name;
    }

    get type() {
        return TensorFlowTensor.formatTensorType(this._tensor);
    }

    get title() {
        return this._title;
    }

    get value() {
        if (!this._tensor.dtype) {
            return 'Tensor has no data type.';
        }
        if (!this._tensor.tensorShape || !this._tensor.tensorShape.dim) {
            return 'Tensor has no dimensions.';
        }

        this._size = 1;
        this._tensor.tensorShape.dim.forEach((dim) => {
            this._size = this._size * (dim.size ? dim.size : 0);
        });

        switch (this._tensor.dtype) {
            case tensorflow.DataType.DT_FLOAT:
                if (this._tensor.tensorContent && this._tensor.tensorContent.length > 0) {
                    this._rawData = new DataView(this._tensor.tensorContent.buffer, this._tensor.tensorContent.byteOffset, this._tensor.tensorContent.byteLength)
                }
                else if (this._tensor.floatVal && this._tensor.floatVal.length == this._size) {
                    this._data = this._tensor.floatVal;
                }
                else {
                    return 'Tensor data is empty.';
                }
                break;
            case tensorflow.DataType.DT_INT32:
                if (this._tensor.tensorContent && this._tensor.tensorContent.length > 0) {
                    this._rawData = new DataView(this._tensor.tensorContent.buffer, this._tensor.tensorContent.byteOffset, this._tensor.tensorContent.byteLength)
                }
                else if (this._tensor.intVal && this._tensor.intVal.length == this._size) {
                    this._data = this._tensor.intVal;
                }
                else {
                    return 'Tensor data is empty.';
                }
                break;
            case tensorflow.DataType.DT_STRING:
                if (this._tensor.tensorContent && this._tensor.tensorContent.length > 0) {
                    return 'Tensor data type is not implemented.';
                }
                else if (this._tensor.stringVal && this._tensor.stringVal.length == this._size) {
                    this._data = this._tensor.stringVal;
                }
                else {
                    return 'Tensor data is empty.';
                }
                break;
            case tensorflow.DataType.DT_BOOL:
                debugger;
                return 'Tensor data type is not implemented.';
            default:
                debugger;
                return 'Tensor data type is not implemented.';
        }

        this._index = 0;
        this._count = 0;
        this._utf8Decoder = window.TextDecoder ? new TextDecoder('utf-8') : null;
        var result = this.read(0);
        delete this._size;
        delete this._index;
        delete this._count;
        delete this._data;
        delete this._rawData;
        delete this._utf8Decoder;

        return JSON.stringify(result, null, 4);
    }

    read(dimension) {
        var results = [];
        var dimensions = this._tensor.tensorShape.dim;
        if (dimensions.length == 0 && this._data.length == 1) {
            return this.readDataValue();
        }
        var dim = dimensions[dimension];
        var size = dim.size;
        if (dimension == dimensions.length - 1) {
            for (var i = 0; i < size; i++) {
                if (this._count > 10000) {
                    results.push('...');
                    return results;
                }
                if (this._data) {
                    results.push(this.readDataValue());
                    this._count++;
                }
                else {
                    if (this._rawData) {
                        switch (this._tensor.dtype)
                        {
                            case tensorflow.DataType.DT_FLOAT:
                                results.push(this._rawData.getFloat32(this._index, true));
                                this._index += 4;
                                this._count++;
                                break;
                            case tensorflow.DataType.DT_INT32:
                                results.push(this._rawData.getInt32(this._index, true));
                                this._index += 4;
                                this._count++;
                                break;
                        }
                    }
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

    readDataValue() {
        var value = this._data[this._index++];
        if (this._tensor.dtype == tensorflow.DataType.DT_STRING) {
            if (this._utf8Decoder) {
                value = this._utf8Decoder.decode(value);
            }
            else {
                value = String.fromCharCode.apply(null, textArray);
            }
        }
        return value;
    }

    static formatTensorType(tensor) {
        if (tensor.dtype) {
            var type = TensorFlowTensor.formatDataType(tensor.dtype);
            if (tensor.tensorShape) {
                type += TensorFlowTensor.formatTensorShape(tensor.tensorShape);
            }
            return type;
        }
        debugger;
        return '?';
    }

    static formatDataType(type) {
        if (!TensorFlowTensor.dataType)
        {
            TensorFlowTensor.dataType = {};
            Object.keys(tensorflow.DataType).forEach((key) => {
                var value = tensorflow.DataType[key];
                key = key.startsWith('DT_') ? key.substring(3) : key;
                TensorFlowTensor.dataType[value] = key.toLowerCase();
            });
        }
        var text = TensorFlowTensor.dataType[type];
        if (text) { 
            return text;
        }
        debugger;
        return '?';
    }

    static formatTensorShape(shape) {
        if (shape.dim) {
            if (shape.unknownRank) {
                return '[-]';
            }
            if (shape.dim.length == 0) {
                return '';
            }
            if (shape.dim.length == 1 && !shape.dim[0].size) {
                return '[0]';
            }
            return '[' + shape.dim.map((dim) => (dim.size && dim.size != -1) ? dim.size.toString() : '?').join(',') + ']';
        }
        debugger;
        return '?';
    }
}

class TensorFlowOperatorMetadata {

    static open(host, callback) {
        if (TensorFlowOperatorMetadata.operatorMetadata) {
            callback(null, TensorFlowOperatorMetadata.operatorMetadata);
        }
        else {
            host.request('/tf-operator.pb', (err, data) => {
                TensorFlowOperatorMetadata.operatorMetadata = new TensorFlowOperatorMetadata(data);
                callback(null, TensorFlowOperatorMetadata.operatorMetadata);
            });
        }
    }

    constructor(data) {
        this._map = {};
        if (data) {
            var operators = tensorflow.OpList.decode(data);
            if (operators.op) {
                operators.op.forEach((opDef) => {
                    this._map[opDef.name] = opDef;
                });
            }
        }
    }

    getOpDef(operator) {
        return this._map[operator];
    }
}

class TensorFlowGraphOperatorMetadata {

    constructor(metaInfoDef) {
        this._map = {};
        if (metaInfoDef && metaInfoDef.strippedOpList && metaInfoDef.strippedOpList.op) {
            metaInfoDef.strippedOpList.op.forEach((opDef) => {
                this._map[opDef.name] = opDef;
            });
        }

        this._categoryMap = {
            'Const': 'Constant',
            'Conv2D': 'Layer',
            'BiasAdd': 'Layer',
            'DepthwiseConv2dNative': 'Layer',
            'Relu': 'Activation',
            'Relu6': 'Activation',
            'Softmax': 'Activation',
            'LRN': 'Normalization',
            'MaxPool': 'Pool',
            'AvgPool': 'Pool',
            'Reshape': 'Shape',
            'Squeeze': 'Shape',
            'ConcatV2': 'Tensor',
            'Dequantize': 'Tensor',
            'Identity': 'Control',
            'BatchNormWithGlobalNormalization': 'Normalization',
            // 'VariableV2':
            // 'Assign':
            // 'BiasAdd':
        };
    }

    getOpDef(operator) {
        var schema = TensorFlowOperatorMetadata.operatorMetadata.getOpDef(operator);
        if (!schema) {
            schema = this._map[operator];
        }
        return schema;
    }

    getInputs(node) {
        var results = [];
        var index = 0;
        var inputs = node.input.filter(input => !input.startsWith('^'));
        var opDef = this.getOpDef(node.op);
        if (opDef && opDef.inputArg) {
            opDef.inputArg.forEach((inputArg) => {
                var count = 1;
                if (inputArg.numberAttr) {
                    var number = node.attr[inputArg.numberAttr];
                    if (number && number.i) {
                        count = number.i;
                    }
                }
                var result = {};
                result.name = inputArg.name;
                if (inputArg.type) {
                    result.type = TensorFlowTensor.formatDataType(inputArg.type);
                }
                else if (inputArg.typeAttr) {
                    result.type = inputArg.typeAttr;
                }
                else if (inputArg.typeListAttr) {
                    result.type = inputArg.typeListAttr;
                }
                result.connections = inputs.slice(index, index + count).map((id) => {
                    return { 
                        id: id
                    };
                });
                results.push(result);
                index += count;
            });
        }
        else {
            inputs.slice(index).forEach((input) => {
                results.push({
                    name: '(' + index.toString() + ')',
                    connections: [ { id: input } ]
                });
                index++;
            });
        }
        return results;
    }

    getOutputs(node) {
        var results = [];
        var index = 0;
        var outputs = node.output;
        var opDef = this.getOpDef(node.op);
        if (opDef && opDef.outputArg) {
            opDef.outputArg.forEach((outputArg) => {
                var count = 1;
                if (outputArg.numberAttr) {
                    var number = node.attr[outputArg.numberAttr];
                    if (number && number.i) {
                        count = number.i;
                    }
                }
                var result = {};
                result.name = outputArg.name;
                if (outputArg.type) {
                    result.type = TensorFlowTensor.formatDataType(outputArg.type);
                }
                else if (outputArg.typeAttr) {
                    result.type = outputArg.typeAttr;
                }
                else if (outputArg.typeListAttr) {
                    result.type = outputArg.typeListAttr;
                }
                result.connections = outputs.slice(index, index + count).map((id) => {
                    return { id: id };
                });
                results.push(result);
                index += count;
            });
        }
        else {
            outputs.slice(index).forEach((output) => {
                results.push({
                    name: '(' + index.toString() + ')',
                    connections: [ { id: output } ]
                });
                index++;
            });
        }
        return results;
    }

    getHiddenAttributeMap(operator) {
        var result = {
            '_output_shapes': true,
            '_class': true
        };
        var opDef = this.getOpDef(operator);
        if (opDef) {
            if (opDef.inputArg) {
                opDef.inputArg.forEach((inputArg) => {
                    if (inputArg.typeAttr) {
                        result[inputArg.typeAttr] = true;
                    }
                    else if (inputArg.typeListAttr) {
                        result[inputArg.typeListAttr] = true;
                    }
                    if (inputArg.numberAttr) {
                        result[inputArg.numberAttr] = true;
                    }
                });
            }
            if (opDef.outputArg) {
                opDef.outputArg.forEach((outputArg) => {
                    if (outputArg.typeAttr) {
                        result[outputArg.typeAttr] = true;
                    }
                    else if (outputArg.typeListAttr) {
                        result[outputArg.typeListAttr] = true;
                    }
                    if (outputArg.numberAttr) {
                        result[outputArg.numberAttr] = true;
                    }
                });
            }
        }   
        return result;
    }

    getAttributeType(operator, name) {
        var opDef = this.getOpDef(operator);
        if (opDef) {
            var attributeMap = opDef.attributeMap;
            if (!attributeMap) {
                attributeMap = {};
                if (opDef.attr) {
                    opDef.attr.forEach((attr) => {
                        attributeMap[attr.name] = attr;
                    });
                }
                opDef.attributeMap = attributeMap;
            }
            var attributeEntry = attributeMap[name];
            if (attributeEntry) { 
                return attributeEntry.type;
            }
        }        
        return '';
    }

    getOperatorCategory(operator) {
        var category = this._categoryMap[operator];
        if (category) {
            return category;
        }
        return null;
    }

    getOperatorDocumentation(operator) {
        var schema = {};
        var opDef = this.getOpDef(operator);
        if (opDef) {
            schema.name = operator;
            if (opDef.summary) {
                schema.summary = marked(opDef.summary);
            }
            if (opDef.description) {
                schema.description = marked(opDef.description);
            }
            if (opDef.inputArg) {
                schema.inputs = [];
                opDef.inputArg.forEach((inputArg) => {
                    var input = {};
                    input.name = inputArg.name;
                    if (inputArg.type) {
                        input.type = TensorFlowTensor.formatDataType(inputArg.type);
                    }
                    else if (inputArg.typeAttr) {
                        input.type = inputArg.typeAttr;
                    }
                    else if (inputArg.typeListAttr) {
                        input.type = inputArg.typeListAttr;
                    }
                    if (inputArg.description) {
                        input.description = marked(inputArg.description);
                    }
                    schema.inputs.push(input);
                });
            }
            if (opDef.outputArg) {
                schema.outputs = [];
                opDef.outputArg.forEach((outputArg) => {
                    var output = {};
                    output.name = outputArg.name;
                    if (outputArg.type) {
                        output.type = TensorFlowTensor.formatDataType(outputArg.type);
                    }
                    else if (outputArg.typeAttr) {
                        output.type = outputArg.typeAttr;
                    }
                    else if (outputArg.typeListAttr) {
                        output.type = outputArg.typeListAttr;
                    }
                    if (outputArg.description) {
                        output.description = marked(outputArg.description);
                    }
                    schema.outputs.push(output);
                });
            }
            if (opDef.attr) {
                schema.attributes = [];
                opDef.attr.forEach((attr) => {
                    var attribute = {};
                    attribute.name = attr.name;
                    if (attr.type) {
                        attribute.type = attr.type;
                    }
                    var description = attr.description;
                    if (attr.allowedValues) {
                        var allowedValues = TensorFlowAttribute.formatAttributeValue(attr.allowedValues);
                        allowedValues = Array.isArray(allowedValues) ? allowedValues : [ allowedValues ];
                        allowedValues = allowedValues.map((item) => '`' + item + '`').join(', ');
                        allowedValues = 'Must be one of the following: ' + allowedValues + '.';
                        description = description ? (allowedValues + ' ' + description) : allowedValues;
                    }
                    if (attr.defaultValue) {
                        var defaultValue = TensorFlowAttribute.formatAttributeValue(attr.defaultValue);
                        defaultValue = Array.isArray(defaultValue) ? defaultValue : [ defaultValue ];
                        defaultValue = defaultValue.map((item) => '`' + item + '`').join(', ');
                        defaultValue = 'Defaults to ' + defaultValue + '.';
                        description = description ? (defaultValue + ' ' + description) : defaultValue;
                    }
                    if (description) {
                        attribute.description = marked(description);
                    }
                    schema.attributes.push(attribute);
                });
            }
            var template = Handlebars.compile(operatorTemplate, 'utf-8');
            return template(schema);
        }
        return null;
    }
}

class TensorFlowError extends Error {
    constructor(message) {
        super(message);
        this.name = 'TensorFlow Error';
    }
}
