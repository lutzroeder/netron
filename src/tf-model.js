/*jshint esversion: 6 */

// Experimental

var tensorflow = null;

class TensorFlowModelFactory {

    match(buffer, identifier) {
        var extension = identifier.split('.').pop();
        return (identifier != 'predict_net.pb') && (extension == 'pb' || extension == 'meta');
    }

    open(buffer, identifier, host, callback) { 
        host.import('/tf.js', (err) => {
            if (err) {
                callback(err, null);
            }
            else {
                tensorflow = protobuf.roots.tf.tensorflow;
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
                            catch (metaError) {
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
                catch (error) {
                    callback(new TensorFlowError(error.message), null);
                }    
            }
        });
    }
}

class TensorFlowModel {

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
        results.push({ name: 'format', value: this._format });
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
        this._operators = {};
        this._graph.graphDef.node.forEach((node) => {
            this._operators[node.op] = (this._operators[node.op] || 0) + 1;
        });
    }

    get operators() {
        return this._operators;
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

    get groups() {
        return false;
        // TODO return true;
    }

    get inputs() {
        this._update();
        var results = [];
        Object.keys(this._inputMap).forEach((key) => {
            results.push(this._inputMap[key]);
        });
        return results;
    }

    get outputs() {
        this._update();
        return [];
    }

    get nodes() {
        this._update();
        var results = [];
        this._graph.graphDef.node.forEach((node) => {
            if (node.output.filter(output => !output.startsWith('^')) != 0 ||
                node.input.filter(input => !input.startsWith('^')).length > 0) {
                var id = node.name;
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

    get namespaces() {
        return this._namespaces;
    }

    _update() {
        if (!this._nodeMap) {
            this._nodeMap = {};
            this._namespaces = {};
            var nodes = this._graph.graphDef.node;
            nodes.forEach((node) => {
                var name = node.name;
                this._nodeMap[name] = node;   
                        if (node.op != 'Const') {
                    var lastIndex = name.lastIndexOf('/');
                    if (lastIndex != -1) {
                        var namespace = name.substring(0, lastIndex);
                        this._namespaces[namespace] = true;
                    }
                }
                node.output = [];         
            });
            nodes.forEach((node) => {
                for (var i = 0; i < node.input.length; i++)
                {
                    var split = node.input[i].split(':', 2);
                    var inputName = split[0];
                    if (!inputName.startsWith('^')) {
                        var outputIndex = split.length == 1 ? 0 : parseInt(split[1]);
                        var outputName = inputName;
                        var outputNode = this._nodeMap[outputName];
                        node.input[i] = outputIndex == 0 ? inputName : inputName + ':' + outputIndex.toString();
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
                if (node.op == 'Const' && this._checkEmptyInput(node) && this._checkSingleOutput(node)) {
                    var value = node.attr.value;
                    if (value && value.hasOwnProperty('tensor')) {
                        var output = node.output[0];
                        if (output) {
                            this._initializerMap[output] = new TensorFlowTensor(value.tensor, output, node.name, 'Constant');
                        }
                    }
                }
            });
            this._graph.graphDef.node.forEach((node) => {
                if (node.op == 'Identity' && node.input.length == 1 && this._checkSingleOutput(node)) {
                    var input = node.input[0];
                    var tensor = this._initializerMap[input];
                    if (tensor) {
                        var output = node.output[0];
                        this._initializerMap[input] = "-";
                        tensor.kind = 'Identity Constant';
                        this._initializerMap[output] = tensor;
                    }
                }
            });

            this._inputMap = {};
            this._graph.graphDef.node.forEach((node) => {
                if (node.op == 'Placeholder' && node.input.length == 0 && node.output.length == 1) {
                    var dtype = node.attr.dtype;
                    var shape = node.attr.shape;
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

    _getInitializer(input) {
        var initializer = this._initializerMap[input];
        return initializer ? initializer : null;
    }

    _checkEmptyInput(node) {
        var inputs = node.input.filter((input) => !input.startsWith('^'));
        return inputs.length == 0;
    }

    _checkSingleOutput(node) { 
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

    get device() {
        return this._node.device;
    }

    get group() {
        var name = this._node.name;
        if (this._graph.namespaces[name]) {
            return name;
        }
        var lastIndex = name.lastIndexOf('/');
        if (lastIndex != -1) {
            var namespace = name.substring(0, lastIndex);
            if (this._graph.namespaces[namespace]) {
                return namespace;
            }
        }
        return '';
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
                    var initializer = this._graph._getInitializer(connection.id);
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
            Object.keys(node.attr).forEach((name) => {
                var value = node.attr[name];
                var visible = graphMetadata.getAttributeVisible(node.op, name, value);
                result.push(new TensorFlowAttribute(this, name, value, visible));
            });
        }
        return result;
    }
}

class TensorFlowAttribute { 
    constructor(node, name, value, visible) {
        this._node = node;
        this._name = name;
        this._value = value;
        if (!visible) {
            this._hidden = true;
        }
    }

    get name() {
        return this._name;
    }

    get type() {
        if (this._value.hasOwnProperty('tensor')) {
            return new TensorFlowTensor(this._value.tensor).type;
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
            return TensorFlowTensor.formatTensorShape(value.shape);
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
        return '';        
    }

    get visible() {
        return this._hidden ? false : true;
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

class TensorFlowTensor {

    constructor(tensor, id, name, kind) {
        this._tensor = tensor;
        this._id = id;
        this._name = name;
        if (kind) {
            this._kind = kind;
        }
    }

    get id() {
        return this._id;
    }

    get name() {
        return this._name;
    }

    get type() {
        if (this._tensor.dtype) {
            var type = TensorFlowTensor.formatDataType(this._tensor.dtype);
            if (this._tensor.tensorShape) {
                type += TensorFlowTensor.formatTensorShape(this._tensor.tensorShape);
            }
            return type;
        }
        return '?';
    }

    get kind() {
        return this._kind;
    }

    set kind(value) {
        this._kind = value;
    }

    get value() {
        var result = this._decode(Number.MAX_SAFE_INTEGER);
        if (result.error) {
            return null;
        }
        return result.value;
    }

    toString() {
        var result = this._decode(10000);
        if (result.error) {
            return result.error;
        }
        return JSON.stringify(result.value, null, 4);
    }

    _decode(limit) {
        var result = {};
        if (!this._tensor.dtype) {
            result.error = 'Tensor has no data type.';
            return result;
        }
        if (!this._tensor.tensorShape || !this._tensor.tensorShape.dim) {
            result.error = 'Tensor has no dimensions.';
            return result;
        }

        var context = {};
        context.index = 0;
        context.count = 0;
        context.limit = limit;
        context.utf8Decoder = window.TextDecoder ? new TextDecoder('utf-8') : null;
        context.size = 1;
        this._tensor.tensorShape.dim.forEach((dim) => {
            context.size = context.size * (dim.size ? dim.size : 0);
        });

        switch (this._tensor.dtype) {
            case tensorflow.DataType.DT_FLOAT:
                if (this._tensor.tensorContent && this._tensor.tensorContent.length > 0) {
                    context.rawData = new DataView(this._tensor.tensorContent.buffer, this._tensor.tensorContent.byteOffset, this._tensor.tensorContent.byteLength);
                }
                else if (this._tensor.floatVal && this._tensor.floatVal.length == context.size) {
                    context.data = this._tensor.floatVal;
                }
                else {
                    result.error = 'Tensor data is empty.';
                    return result;
                }
                break;
            case tensorflow.DataType.DT_INT32:
                if (this._tensor.tensorContent && this._tensor.tensorContent.length > 0) {
                    context.rawData = new DataView(this._tensor.tensorContent.buffer, this._tensor.tensorContent.byteOffset, this._tensor.tensorContent.byteLength);
                }
                else if (this._tensor.intVal && this._tensor.intVal.length == context.size) {
                    context.data = this._tensor.intVal;
                }
                else {
                    result.error = 'Tensor data is empty.';
                    return result;
                }
                break;
            case tensorflow.DataType.DT_STRING:
                if (this._tensor.tensorContent && this._tensor.tensorContent.length > 0) {
                    result.error = 'Tensor data type is not implemented.';
                    return result;
                }
                else if (this._tensor.stringVal && this._tensor.stringVal.length == context.size) {
                    context.data = this._tensor.stringVal;
                }
                else {
                    result.error = 'Tensor data is empty.';
                    return result;
                }
                break;
            case tensorflow.DataType.DT_BOOL:
                result.error = 'Tensor data type is not implemented.';
                return result;
            default:
                result.error = 'Tensor data type is not implemented.';
                return result;
        }

        result.value = this._decodeDimension(context, 0);
        return result;
    }

    _decodeDimension(context, dimension) {
        var results = [];
        var dimensions = this._tensor.tensorShape.dim;
        if (dimensions.length == 0 && context.data.length == 1) {
            return this._decodeDataValue(context);
        }
        var dim = dimensions[dimension];
        var size = dim.size;
        if (dimension == dimensions.length - 1) {
            for (var i = 0; i < size; i++) {
                if (context.count > context.limit) {
                    results.push('...');
                    return results;
                }
                if (context.data) {
                    results.push(this._decodeDataValue(context));
                    context.count++;
                }
                else {
                    if (context.rawData) {
                        switch (this._tensor.dtype)
                        {
                            case tensorflow.DataType.DT_FLOAT:
                                results.push(context.rawData.getFloat32(context.index, true));
                                context.index += 4;
                                context.count++;
                                break;
                            case tensorflow.DataType.DT_INT32:
                                results.push(context.rawData.getInt32(context.index, true));
                                context.index += 4;
                                context.count++;
                                break;
                        }
                    }
                }
            }
        }
        else {
            for (var j = 0; j < size; j++) {
                if (context.count > context.limit) {
                    results.push('...');
                    return results;
                }
                results.push(this._decodeDimension(context, dimension + 1));
            }
        }
        return results;
    }

    _decodeDataValue(context) {
        var value = context.data[context.index++];
        if (this._tensor.dtype == tensorflow.DataType.DT_STRING) {
            if (context.utf8Decoder) {
                value = context.utf8Decoder.decode(value);
            }
            else {
                value = String.fromCharCode.apply(null, textArray);
            }
        }
        return value;
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
        return '?';
    }
}

class TensorFlowGraphOperatorMetadata {

    constructor(metaInfoDef) {
        this._map = {};
        if (metaInfoDef && metaInfoDef.strippedOpList && metaInfoDef.strippedOpList.op) {
            metaInfoDef.strippedOpList.op.forEach((opDef) => {
            });
        }
    }

    getSchema(operator) {
        var schema = TensorFlowOperatorMetadata.operatorMetadata.getSchema(operator);
        if (!schema) {
            schema = this._map[operator];
        }
        return schema;
    }

    getInputs(node) {
        var results = [];
        var index = 0;
        var inputs = node.input.filter(input => !input.startsWith('^'));
        var schema = this.getSchema(node.op);
        if (schema && schema.inputs) {
            schema.inputs.forEach((input) => {
                var count = 1;
                if (input.numberAttr) {
                    var number = node.attr[input.numberAttr];
                    if (number && number.i) {
                        count = number.i;
                    }
                }
                var result = {};
                result.name = input.name;
                if (input.type) {
                    result.type = TensorFlowTensor.formatDataType(input.type);
                }
                else if (input.typeAttr) {
                    result.type = input.typeAttr;
                }
                else if (input.typeListAttr) {
                    result.type = input.typeListAttr;
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
        var schema = this.getSchema(node.op);
        if (schema && schema.outputs) {
            schema.outputs.forEach((output) => {
                var count = 1;
                if (output.numberAttr) {
                    var number = node.attr[output.numberAttr];
                    if (number && number.i) {
                        count = number.i;
                    }
                }
                var result = {};
                result.name = output.name;
                if (output.type) {
                    result.type = TensorFlowTensor.formatDataType(output.type);
                }
                else if (output.typeAttr) {
                    result.type = output.typeAttr;
                }
                else if (output.typeListAttr) {
                    result.type = output.typeListAttr;
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

    getAttributeSchema(operator, name, value) {
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
        if (attributeSchema && attributeSchema.type) {
            return attributeSchema.type;
        }
        return '';
    }

    getAttributeVisible(operator, name, value) {
        var schema = this.getSchema(operator);
        if (schema) {
            var attributeSchema = this.getAttributeSchema(operator, name);
            if (attributeSchema) {
                if (attributeSchema.hasOwnProperty('visible')) {
                    return attributeSchema.visible;
                }
                if (attributeSchema.hasOwnProperty('default')) {
                    var valueText = TensorFlowAttribute.formatAttributeValue(value);
                    var defaultValueText = TensorFlowGraphOperatorMetadata.formatAttributeValue(attributeSchema.default);
                    if (valueText == defaultValueText) {
                        return false;
                    }
                }
            }
            if (name == '_output_shapes' || name == '_class') {
                return false;
            }
            var hiddenAttributeMap = schema.hiddenAttributeMap;
            if (!hiddenAttributeMap) {
                hiddenAttributeMap = {};
                if (schema.inputs) {
                    schema.inputs.forEach((input) => {
                        if (input.typeAttr) {
                            hiddenAttributeMap[input.typeAttr] = true;
                        }
                        else if (input.typeListAttr) {
                            hiddenAttributeMap[input.typeListAttr] = true;
                        }
                        if (input.numberAttr) {
                            hiddenAttributeMap[input.numberAttr] = true;
                        }
                    });
                }
                if (schema.outputs) {
                    schema.outputs.forEach((output) => {
                        if (output.typeAttr) {
                            hiddenAttributeMap[output.typeAttr] = true;
                        }
                        else if (output.typeListAttr) {
                            hiddenAttributeMap[output.typeListAttr] = true;
                        }
                        if (output.numberAttr) {
                            hiddenAttributeMap[output.numberAttr] = true;
                        }
                    });
                }
                schema.hiddenAttributeMap = hiddenAttributeMap;
            }
            if (hiddenAttributeMap[name]) {
                return false;
            }
        }
        return true;
    }

    getOperatorCategory(node) {
        var schema = this.getSchema(node);
        if (schema && schema.category) {
            return schema.category;
        }
        return null;
    }

    getOperatorDocumentation(operator) {
        var schema = this.getSchema(operator);
        if (schema) {
            schema = JSON.parse(JSON.stringify(schema));
            schema.name = operator;
            if (schema.summary) {
                schema.summary = marked(schema.summary);
            }
            if (schema.description) {
                schema.description = marked(schema.description);
            }
            if (schema.inputs) {
                schema.inputs.forEach((input) => {
                    if (input.type) {
                        input.type = TensorFlowTensor.formatDataType(input.type);
                    }
                    else if (input.typeAttr) {
                        input.type = input.typeAttr;
                    }
                    else if (input.typeListAttr) {
                        input.type = input.typeListAttr;
                    }
                    if (input.description) {
                        input.description = marked(input.description);
                    }
                });
            }
            if (schema.outputs) {
                schema.outputs.forEach((output) => {
                    if (output.type) {
                        output.type = TensorFlowTensor.formatDataType(output.type);
                    }
                    else if (output.typeAttr) {
                        output.type = output.typeAttr;
                    }
                    else if (output.typeListAttr) {
                        output.type = output.typeListAttr;
                    }
                    if (output.description) {
                        output.description = marked(output.description);
                    }
                });
            }
            if (schema.attributes) {
                schema.attributes.forEach((attribute) => {
                    var description = attribute.description;
                    if (attribute.allowedValues) {
                        var allowedValues = TensorFlowGraphOperatorMetadata.formatAttributeValue(attribute.allowedValues);
                        allowedValues = Array.isArray(allowedValues) ? allowedValues : [ allowedValues ];
                        allowedValues = allowedValues.map((item) => '`' + item + '`').join(', ');
                        allowedValues = 'Must be one of the following: ' + allowedValues + '.';
                        description = description ? (allowedValues + ' ' + description) : allowedValues;
                    }
                    if (attribute.defaultValue) {
                        var defaultValue = TensorFlowGraphOperatorMetadata.formatAttributeValue(attribute.defaultValue);
                        defaultValue = Array.isArray(defaultValue) ? defaultValue : [ defaultValue ];
                        defaultValue = defaultValue.map((item) => '`' + item + '`').join(', ');
                        defaultValue = 'Defaults to ' + defaultValue + '.';
                        description = description ? (defaultValue + ' ' + description) : defaultValue;
                    }
                    if (description) {
                        attribute.description = marked(description);
                    }
                });
            }
            return schema;
        }
        return null;
    }

    static formatAttributeValue(value) {
        if (Array.isArray(value)) {
            return value.map((item) => TensorFlowGraphOperatorMetadata.formatAttributeValue(item));
        }
        if (value === Object(value)) {
            switch (value.type) {
                case 'type':
                    return TensorFlowTensor.formatDataType(value.value);
                case 'shape':
                    return value.value;
                case 'tensor':
                    return value.value;
            }
        }
        if (typeof value === 'string') {
            return '"' + value + '"';
        }
        return value.toString();
    }
}

class TensorFlowOperatorMetadata {

    static open(host, callback) {
        if (TensorFlowOperatorMetadata.operatorMetadata) {
            callback(null, TensorFlowOperatorMetadata.operatorMetadata);
        }
        else {
            host.request('/tf-metadata.json', (err, data) => {
                TensorFlowOperatorMetadata.operatorMetadata = new TensorFlowOperatorMetadata(data);
                callback(null, TensorFlowOperatorMetadata.operatorMetadata);
            });
        }
    }

    constructor(data) {
        this._map = {};
        if (data) {
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
    }

    getSchema(operator) {
        return this._map[operator];
    }
}

class TensorFlowError extends Error {
    constructor(message) {
        super(message);
        this.name = 'Error loading TensorFlow model.';
    }
}
