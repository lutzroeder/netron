/*jshint esversion: 6 */

var onnx = protobuf.roots.onnx.onnx;

class OnnxModel {

    constructor(hostService) {
        this.hostService = hostService;
    }

    openBuffer(buffer, identifier) { 
        try {

            this._model = onnx.ModelProto.decode(buffer);

            if (this._model.graph) {
                this._graph = new OnnxGraph(this, this._model.graph, 0);
                this._activeGraph = this._graph;
            }

            if (!this._operatorMetadata) {
                this._operatorMetadata = new OnnxOperatorMetadata(this.hostService);
            }
        }
        catch (err) {
            return err;
        }
        return null;
    }

    format() {
        var summary = { properties: [], graphs: [] };

        this.graphs.forEach((graph) => {
            summary.graphs.push({
                name: graph.name,
                inputs: graph.inputs,
                outputs: graph.outputs,
                description: graph.description
            });
        });

        summary.properties.push({ 
            name: 'Format', 
            value: 'ONNX' + (this._model.irVersion ? (' v' + this._model.irVersion) : '') 
        });
        var producer = [];
        if (this._model.producerName) {
            producer.push(this._model.producerName);
        }
        if (this._model.producerVersion && this._model.producerVersion.length > 0) {
            producer.push(this._model.producerVersion);
        }
        if (producer.length > 0) {
            summary.properties.push({ 'name': 'Producer', 'value': producer.join(' ') });
        }
        if (this._model.domain) {
            summary.properties.push({ name: 'Domain', value: this._model.domain });
        }
        if (this._model.modelVersion) {
            summary.properties.push({ name: 'Version', value: this._model.modelVersion });
        }
        if (this._model.docString) {
            summary.properties.push({ name: 'Documentation', value: this._model.docString });
        }

        if (this._model.metadataProps && this._model.metadataProps.length > 0)
        {
            debugger;
        }

        return summary;
    }

    get graphs() {
        return [ this._graph ];
    }

    get activeGraph() {
        return this._activeGraph;
    }

    updateActiveGraph(name) {
        this._activeGraph = (name == this._graph._graph.name) ? this._graph : null;
    }
}

class OnnxGraph {

    constructor(model, graph, index) {
        this._model = model;
        this._graph = graph;
        this._name = this._graph.name ? this._graph.name : ('(' + index.toString() + ')');
    }

    get model() {
        return this._model;
    }

    get name() {
        return this._name;
    }

    get description() {
        return this._graph.docString ? this._graph.docString : '';
    }

    get inputs() {
        if (!this._inputs) {
            this._inputs = [];
            var initializerMap = {};
            this._graph.initializer.forEach((tensor) => {
                initializerMap[tensor.name] = true;
            });
            this._graph.input.forEach((valueInfo, index) => {
                if (!initializerMap[valueInfo.name]) {
                    this._inputs.push({
                        id: valueInfo.name,
                        name: valueInfo.name,
                        type: this.formatType(valueInfo.type)
                    });
                }
            });
        }
        return this._inputs;
    }

    get outputs() {
        if (!this._outputs) {
            this._outputs = this._graph.output.map((valueInfo) => {
                return {
                    id: valueInfo.name,
                    name: valueInfo.name,
                    type: this.formatType(valueInfo.type)
                };
            });
        }
        return this._outputs;
    }

    get initializers() {
        if (!this._initializers) {
            this._initializers = [];
            this._graph.initializer.forEach((tensor) => {
                var result = this.formatTensor(tensor);
                result.id = tensor.name;
                result.title = 'Initializer';
                this._initializers.push(result);
            });
            this._graph.node.forEach((node) => {
                if (node.opType == 'Constant' && node.output && node.output.length == 1) {
                    var result = null;
                    node.attribute.forEach((attribute) => {
                        if (attribute.name == 'value' && attribute.t) {
                            result = this.formatTensor(attribute.t);
                        }                    
                    });
                    if (result) {
                        result.id = node.output[0];
                        if (!result.name) {
                            result.name = result.id;
                        }
                        result.title = 'Constant';
                        this._initializers.push(result);
                    }
                }
            });
        }
        return this._initializers;
    }

    get nodes() {
        var results = [];
        var initializerMap = {};
        this.initializers.forEach((initializer) => {
            initializerMap[initializer.id] = true;
        });
        this._graph.node.forEach((node) => {
            if (node.opType == 'Constant' && node.output.length == 1 && initializerMap[node.output[0]]) {

            }
            else {
                results.push(new OnnxNode(this, node));
            }
        });
        return results;
    }

    formatTensorType(tensor) {
        var result = "";
        if (tensor.hasOwnProperty('dataType')) {
            result = this.formatElementType(tensor.dataType);
            if (tensor.dims) { 
                result += '[' + tensor.dims.map(dimension => dimension.toString()).join(',') + ']';
            }
        }
        return result;
    }

    formatTensor(tensor) {
        return {
            name: tensor.name,
            type: this.formatTensorType(tensor),
            value: () => { return new OnnxTensorFormatter(tensor).toString(); }
        };
    }

    formatElementType(elementType) {
        if (!this._elementTypeMap) {
            this._elementTypeMap = { };
            this._elementTypeMap[onnx.TensorProto.DataType.UNDEFINED] = 'UNDEFINED';
            this._elementTypeMap[onnx.TensorProto.DataType.FLOAT] = 'float';
            this._elementTypeMap[onnx.TensorProto.DataType.UINT8] = 'uint8';
            this._elementTypeMap[onnx.TensorProto.DataType.INT8] = 'int8';
            this._elementTypeMap[onnx.TensorProto.DataType.UINT16] = 'uint16';
            this._elementTypeMap[onnx.TensorProto.DataType.INT16] = 'int16';
            this._elementTypeMap[onnx.TensorProto.DataType.INT32] = 'int32';
            this._elementTypeMap[onnx.TensorProto.DataType.INT64] = 'int64';
            this._elementTypeMap[onnx.TensorProto.DataType.STRING] = 'string';
            this._elementTypeMap[onnx.TensorProto.DataType.BOOL] = 'bool';
            this._elementTypeMap[onnx.TensorProto.DataType.FLOAT16] = 'float16';
            this._elementTypeMap[onnx.TensorProto.DataType.DOUBLE] = 'double';
            this._elementTypeMap[onnx.TensorProto.DataType.UINT32] = 'uint32';
            this._elementTypeMap[onnx.TensorProto.DataType.UINT64] = 'uint64';
            this._elementTypeMap[onnx.TensorProto.DataType.COMPLEX64] = 'complex64';
            this._elementTypeMap[onnx.TensorProto.DataType.COMPLEX128] = 'complex128';    
        }
        var name = this._elementTypeMap[elementType];
        if (name) {
            return name;
        }
        debugger;
        return this._elementTypeMap[onnx.TensorProto.DataType.UNDEFINED];
    }

    formatType(type) {
        if (type.value == 'tensorType') {
            var tensorType = type.tensorType;
            var text = this.formatElementType(tensorType.elemType); 
            if (tensorType.shape && tensorType.shape.dim) {
                text += '[' + tensorType.shape.dim.map(dimension => dimension.dimValue.toString()).join(',') + ']';
            }
            return text;
        }
        else if (type.value == 'mapType') {
            var mapType = type.mapType;
            return '<' + this.formatElementType(mapType.keyType) + ', ' + this.formatType(mapType.valueType) + '>';
        }
        else if (!type.value) {
            return '?';
        }
        debugger;
        return '[UNKNOWN]';
    }
}

class OnnxNode {

    constructor(graph, node) {
        this._graph = graph;
        this._node = node;
    }

    get documentation() {
        return this._graph.model._operatorMetadata.getOperatorDocumentation(this.operator);
    }

    get operator() {
        return this._node.opType;
    }

    get inputs() {
        var operatorMetadata = this._graph.model._operatorMetadata;
        var results = [];
        this._node.input.forEach((input, index) => {
            results.push({
                'id': input,
                'name': operatorMetadata.getInputName(this.operator, index),
                'type': ""
            });
        });
        return results;
    }

    get outputs() {
        var operatorMetadata = this._graph.model._operatorMetadata;
        var results = [];
        this._node.output.forEach((output, index) => {
            results.push({
                id: output,
                name: operatorMetadata.getOutputName(this.operator, index),
                type: ''
            });
        });
        return results;
    }

    get properties() {
        var result = null;
        var node = this._node;
        if (node.name || node.docString || node.domain) {
            result = [];
            if (node.name) {
                result.push({ name: 'name', value: node.name, value_short: () => { return node.name; } });
            }
            if (node.docString) {
                result.push({ name: 'doc', value: node.docString, value_short: () => {
                    var value = node.docString;
                    if (value.length > 50) {
                        return value.substring(0, 25) + '...';
                    }
                    return value;
                } });
            }
            if (node.domain) {
                result.push({ name: 'domain', value: node.domain, value_short: () => { return node.domain; } });
            }        
        }
        return result;
    }

    get attributes() {
        var result = null;
        var node = this._node;
        if (node.attribute && node.attribute.length > 0) {
            result = [];
            node.attribute.forEach((attribute) => { 
                result.push(this.formatNodeAttribute(attribute));
            });
        }
        return result;
    }

    formatNodeAttribute(attribute) {
        var type = "";
        if (attribute.hasOwnProperty('type')) { 
            type = this._graph.formatElementType(attribute.type);
            if ((attribute.ints && attribute.ints.length > 0) ||
                (attribute.floats && attribute.floats.length > 0) ||
                (attribute.strings && attribute.strings.length > 0)) {
                type = type + '[]';
            }
        }
        else if (attribute.hasOwnProperty('t')) {
            type = this._graph.formatTensorType(attribute.t);
        }

        var tensor = false;
        var callback = '';
        if (attribute.ints && attribute.ints.length > 0) {
            callback = () => {
                if (attribute.ints.length > 65536) {
                    return "Too large to render.";
                }
                return attribute.ints.map((v) => { return v.toString(); }).join(', '); 
            };
        }
        else if (attribute.floats && attribute.floats.length > 0) {
            callback = () => {
                if (attribute.floats.length > 65536) {
                    return "Too large to render.";
                }
                return attribute.floats.map(v => v.toString()).join(', ');
            };
        }
        else if (attribute.strings && attribute.strings.length > 0) {
            callback = () => { 
                if (attribute.strings.length > 65536) {
                    return "Too large to render.";
                }
                return attribute.strings.map((s) => {
                    if (s.filter(c => c <= 32 && c >= 128).length == 0) {
                        return '"' + String.fromCharCode.apply(null, s) + '"';
                    }
                    return s.map(v => v.toString()).join(', ');    
                }).join(', ');
            };
        }
        else if (attribute.s && attribute.s.length > 0) {
            callback = () => { 
                if (attribute.s.filter(c => c <= 32 && c >= 128).length == 0) {
                    return '"' + String.fromCharCode.apply(null, attribute.s) + '"';
                }
                return attribute.s.map(v => v.toString()).join(', ');           
            };
        }
        else if (attribute.hasOwnProperty('f')) {
            callback = () => { 
                return attribute.f.toString();
            };
        }
        else if (attribute.hasOwnProperty('i')) {
            callback = () => {
                return attribute.i.toString();
            };
        }
        else if (attribute.hasOwnProperty('t')) {
            tensor = true;
            callback = () => {
                return new OnnxTensorFormatter(attribute.t).toString();
            };
        }
        else {
            debugger;
            callback = () => {
                return "?";
            };
        }

        var result = {};
        result.name = attribute.name;
        if (type) {
            result.type = type;
        }
        result.value = callback;
        result.value_short = () => {
            if (tensor) {
                return "[...]";
            }
            var value = callback();
            if (value.length > 25)
            {
                return value.substring(0, 25) + '...';
            }
            return value;
        };
        if (attribute.docString) {
            result.doc = attribute.docString;
        }

        return result;
    }
}

class OnnxTensorFormatter {

    constructor(tensor) {
        this.tensor = tensor;
    }

    toString() { 
        if (!this.tensor.dataType) {
            return 'Tensor has no data type.';
        }

        if (!this.tensor.dims) {
            return 'Tensor has no dimensions.';
        }

        var size = 1;
        this.tensor.dims.forEach((dimSize) => { size *= dimSize; });
        if (size > 65536) {
            return 'Tensor is too large to display.';
        }

        switch (this.tensor.dataType) {
            case onnx.TensorProto.DataType.FLOAT:
                if (this.tensor.floatData && this.tensor.floatData.length > 0) {
                    this.data = this.tensor.floatData;
                }
                else if (this.tensor.rawData && this.tensor.rawData.length > 0) {
                    this.rawData = new DataView(this.tensor.rawData.buffer, this.tensor.rawData.byteOffset, this.tensor.rawData.byteLength);
                }
                else {
                    return 'Tensor data is empty.';
                }
                break;
            case onnx.TensorProto.DataType.DOUBLE:
                if (this.tensor.doubleData && this.tensor.doubleData.length > 0) {
                    this.data = tensor.doubleData;
                }
                else if (this.tensor.rawData && this.tensor.rawData.length > 0) {
                    this.rawData = new DataView(this.tensor.rawData.buffer, this.tensor.rawData.byteOffset, this.tensor.rawData.byteLength);
                }
                else {
                    return 'Tensor data is empty.';
                }
                break;
            case onnx.TensorProto.DataType.INT32:
                if (this.tensor.int32Data && this.tensor.int32Data.length > 0) {
                    this.data = tensor.int32Data;
                }
                else if (this.tensor.rawData && this.tensor.rawData.length > 0) {
                    this.rawData = new DataView(this.tensor.rawData.buffer, this.tensor.rawData.byteOffset, this.tensor.rawData.byteLength);
                }
                else {
                    return 'Tensor data is empty.';
                }
                break;
            case onnx.TensorProto.DataType.UINT32:
                if (this.tensor.uint64Data && this.tensor.uint64Data.length > 0) {
                    this.data = this.tensor.uint64Data;
                }
                else if (this.tensor.rawData && this.tensor.rawData.length > 0) {
                    this.rawData = this.tensor.rawData;
                }
                else {
                    this.output = 'Tensor data is empty.';
                }
                break;
            case onnx.TensorProto.DataType.INT64:
                if (this.tensor.int64Data && this.tensor.int64Data.length > 0) {
                    this.data = this.tensor.int64Data;
                }
                else if (this.tensor.rawData && this.tensor.rawData.length > 0) {
                    this.rawData = this.tensor.rawData;
                }
                else {
                    this.output = 'Tensor data is empty.';
                }
                break;
            case onnx.TensorProto.DataType.UINT64:
                if (this.tensor.uint64Data && this.tensor.uint64Data.length > 0) {
                    this.data = this.tensor.uint64Data;
                }
                else if (this.tensor.rawData && this.tensor.rawData.length > 0) {
                    this.rawData = this.tensor.rawData;
                }
                else {
                    this.output = 'Tensor data is empty.';
                }
                break;
            default:
                debugger;
                return 'Tensor data type is not implemented.';
        }

        this.index = 0;                
        var result = this.read(0);
        this.data = null;
        this.rawData = null;

        return JSON.stringify(result, null, 4);
    }

    read(dimension) {
        var size = this.tensor.dims[dimension];
        var results = [];
        if (dimension == this.tensor.dims.length - 1) {
            for (var i = 0; i < size; i++) {
                if (this.data) {
                    results.push(this.data[this.index++]);
                }
                else if (this.rawData) {
                    switch (this.tensor.dataType)
                    {
                        case onnx.TensorProto.DataType.FLOAT:
                            results.push(this.rawData.getFloat32(this.index, true));
                            this.index += 4;
                            break;
                        case onnx.TensorProto.DataType.DOUBLE:
                            results.push(this.rawData.getFloat64(this.index, true));
                            this.index += 8;
                            break;
                        case onnx.TensorProto.DataType.INT32:
                            results.push(this.rawData.getInt32(this.index, true));
                            this.index += 4;
                            break;
                        case onnx.TensorProto.DataType.UINT32:
                            results.push(this.rawData.getUint32(this.index, true));
                            this.index += 4;
                            break;
                        case onnx.TensorProto.DataType.INT64:
                            results.push(new Int64(this.rawData.subarray(this.index, 8)));
                            this.index += 8;
                            break;
                        case onnx.TensorProto.DataType.UINT64:
                            results.push(new Uint64(this.rawData.subarray(this.index, 8)));
                            this.index += 8;
                            break;
                    }
                }
            }
        }
        else {
            for (var j = 0; j < size; j++) {
                results.push(this.read(dimension + 1));
            }
        }
        return results;
    }
}

class OnnxOperatorMetadata {

    constructor(hostService) {
        this.map = {};
        hostService.request('/onnx-operator.json', (err, data) => {
            if (err != null) {
                // TODO error
            }
            else {
                var items = JSON.parse(data);
                if (items) {
                    items.forEach((item) => {
                        if (item.name && item.schema)
                        {
                            var name = item.name;
                            var schema = item.schema;
                            this.map[name] = schema;
                        }
                    });
                }
            }
        });
    }

    getInputName(operator, index) {
        var schema = this.map[operator];
        if (schema) {
            var inputs = schema.inputs;
            if (inputs && index < inputs.length) {
                var input = inputs[index];
                if (input) {
                    if (!input.option || input.option != 'variadic') {
                        var name = input.name;
                        if (name) {
                            return name;
                        }
                    }
                } 
            }
        }
        return "(" + index.toString() + ")";
    }

    getOutputName(operator, index) {
        var schema = this.map[operator];
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
        return "(" + index.toString() + ")";
    }

    getOperatorDocumentation(operator) {
        var schema = this.map[operator];
        if (schema) {
            schema = Object.assign({}, schema);
            schema.name = operator;
            if (schema.doc) {
                var input = schema.doc.split('\n');
                var output = [];
                var lines = [];
                var code = true;
                while (input.length > 0) {
                    var line = input.shift();
                    if (line.length > 0)
                    {
                        code = code && line.startsWith('  ');
                        lines.push(line + "\n");
                    }
                    if (line.length == 0 || input.length == 0) {
                        if (lines.length > 0) {
                            if (code) {
                                lines = lines.map((text) => { return text.substring(2); });
                                output.push('<pre>' + lines.join('') + '</pre>');
                            }
                            else {
                                var text = lines.join('');
                                text += text.replace(/\`\`(.*?)\`\`/gm, (match, content) => '<code>' + content + '</code>');
                                text += text.replace(/\`(.*?)\`/gm, (match, content) => '<code>' + content + '</code>');
                                output.push('<p>' + text + '</p>');
                            }
                        }
                        lines = [];
                        code = true;
                    }
                }
                schema.doc = output.join('');
            }
            var formatRange = (value) => {
                return (value == 2147483647) ? '&#8734;' : value.toString();
            };
            if (schema.min_input != schema.max_input) {
                schema.inputs_range = formatRange(schema.min_input) + ' - ' + formatRange(schema.max_input);
            }
            if (schema.min_output != schema.max_output) {
                schema.outputs_range = formatRange(schema.min_output) + ' - ' + formatRange(schema.max_output);
            }
            if (schema.type_constraints) {
                schema.type_constraints.forEach((item) => {
                    if (item.allowed_type_strs) {
                        item.allowed_type_strs_display = item.allowed_type_strs.map((type) => { return type; }).join(', ');
                    }
                });
            }
            var template = Handlebars.compile(operatorTemplate, 'utf-8');
            return template(schema);
        }
        return "";
    }
}
