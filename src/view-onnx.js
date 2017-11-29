
function OnnxModelService(hostService) {
    this.operatorService = new OnnxOperatorService(hostService);

    this.elementTypeMap = { };
    this.elementTypeMap[onnx.TensorProto.DataType.UNDEFINED] = 'UNDEFINED';
    this.elementTypeMap[onnx.TensorProto.DataType.FLOAT] = 'float';
    this.elementTypeMap[onnx.TensorProto.DataType.UINT8] = 'uint8';
    this.elementTypeMap[onnx.TensorProto.DataType.INT8] = 'int8';
    this.elementTypeMap[onnx.TensorProto.DataType.UINT16] = 'uint16';
    this.elementTypeMap[onnx.TensorProto.DataType.INT16] = 'int16';
    this.elementTypeMap[onnx.TensorProto.DataType.INT32] = 'int32';
    this.elementTypeMap[onnx.TensorProto.DataType.INT64] = 'int64';
    this.elementTypeMap[onnx.TensorProto.DataType.STRING] = 'string';
    this.elementTypeMap[onnx.TensorProto.DataType.BOOL] = 'bool';
    this.elementTypeMap[onnx.TensorProto.DataType.FLOAT16] = 'float16';
    this.elementTypeMap[onnx.TensorProto.DataType.DOUBLE] = 'double';
    this.elementTypeMap[onnx.TensorProto.DataType.UINT32] = 'uint32';
    this.elementTypeMap[onnx.TensorProto.DataType.UINT64] = 'uint64';
    this.elementTypeMap[onnx.TensorProto.DataType.COMPLEX64] = 'complex64';
    this.elementTypeMap[onnx.TensorProto.DataType.COMPLEX128] = 'complex128';    
}

OnnxModelService.prototype.openBuffer = function(buffer) { 
    try {
        this.model = onnx.ModelProto.decode(buffer);
    }
    catch (err) {
        return err;
    }
    return null;
}

OnnxModelService.prototype.getOperatorService = function() {
    return this.operatorService;
}

OnnxModelService.prototype.formatModelProperties = function() {
    
    var result = { 'groups': [] };

    var generalProperties = { 'name': 'General', 'properties': [] };
    result['groups'].push(generalProperties);

    var format = 'ONNX';
    if (this.model.irVersion) {
        format += ' v' + this.model.irVersion;
    }
    generalProperties['properties'].push({ 'name': 'Format', 'value': format });
    if (this.model.domain) {
        generalProperties['properties'].push({ 'name': 'Domain', 'value': this.model.domain });
    }
    if (this.model.graph && this.model.graph != null && this.model.graph.name) {
        generalProperties['properties'].push({ 'name': 'Name', 'value': this.model.graph.name });
    }
    if (this.model.modelVersion) {
        generalProperties['properties'].push({ 'name': 'Model Version', 'value': this.model.modelVersion });
    }
    if (this.model.docString) {
        generalProperties['properties'].push({ 'name': 'Documentation', 'value': this.model.docString });
    }
    var producer = [];
    if (this.model.producerName) {
        producer.push(this.model.producerName);
    }
    if (this.model.producerVersion && this.model.producerVersion.length > 0) {
        producer.push(this.model.producerVersion);
    }
    if (producer.length > 0) {
        generalProperties['properties'].push({ 'name': 'Producer', 'value': producer.join(' ') });
    }

    if (this.model.metadataProps && this.model.metadataProps.length > 0)
    {
        var metadataProperties = { 'name': 'Metadata', 'properties': [] };
        result['groups'].push(metadataProperties);
        debugger;
    }

    return result;
}

/*
OnnxModelService.prototype.getGraphs = function() {

}
*/

/*
OnnxModelService.prototype.getNodes = function(graph) {
    
}
*/

/*
OnnxModelService.prototype.formatNode = function(node) {

}
*/

OnnxModelService.prototype.formatNodeProperties = function(node) {
    var result = null;
    if (node.name || node.docString || node.domain) {
        result = [];
        if (node.name) {
            result.push({ 'name': 'name', 'value': node.name, 'value_short': function() { return node.name; } });
        }
        if (node.docString) {
            result.push({ 'name': 'doc', 'value': node.docString, 'value_short': function() {
                var value = node.docString;
                if (value.length > 50) {
                    return value.substring(0, 25) + '...';
                }
                return value;
            } });
        }
        if (node.domain) {
            result.push({ 'name': 'domain', 'value': node.domain, 'value_short': function() { return node.domain } });
        }        
    }
    return result;
}

OnnxModelService.prototype.formatNodeAttributes = function(node) {
    var self = this;
    var result = null;
    if (node.attribute && node.attribute.length > 0) {
        result = [];
        node.attribute.forEach(function (attribute) { 
            result.push(self.formatNodeAttribute(attribute));
        });
    }
    return result;
}

OnnxModelService.prototype.formatNodeAttribute = function(attribute) {

    var type = null;
    if (attribute.type) { 
        type = this.formatElementType(attribute.type);
        if ((attribute.ints && attribute.ints.length > 0) ||
            (attribute.floats && attribute.floats.length > 0) ||
            (attribute.strings && attribute.strings.length > 0)) {
            type = type + '[]';
        }
    }

    var value = "";
    if (attribute.ints && attribute.ints.length > 0) {
        value = attribute.ints.map(v => v.toString()).join(', ');
    }
    else if (attribute.floats && attribute.floats.length > 0) {
        value = attribute.floats.map(v => v.toString()).join(', ');
    }
    else if (attribute.strings && attribute.strings.length > 0) {
        value = attribute.strings.map(function(s) {
            if (s.filter(c => c <= 32 && c >= 128).length == 0) {
                return '"' + String.fromCharCode.apply(null, s) + '"';
            }
            return s.map(v => v.toString()).join(', ');    
        }).join(', ');
    }
    else if (attribute.s && attribute.s.length > 0) {
        if (attribute.s.filter(c => c <= 32 && c >= 128).length == 0) {
            value = '"' + String.fromCharCode.apply(null, attribute.s) + '"';
        }
        else {
            value = attribute.s.map(v => v.toString()).join(', ');           
        }
    }
    else if (attribute.f && attribute.f != 0 || attribute.f == 0 || isNaN(attribute.f)) {
        value = attribute.f.toString();
    }
    else if (attribute.i && (attribute.i != 0 || attribute.i == 0)) {
        value = attribute.i.toString();
    }
    else {
        debugger;
        value = '?';
    }

    var result = {};
    result['name'] = attribute.name;
    if (type) {
        result['type'] = type;
    }
    result['value'] = value;
    result['value_short'] = function() {
        if (value.length > 25)
        {
            return value.substring(0, 25) + '...';
        }
        return value;
    };
    if (attribute.docString) {
        result['doc'] = attribute.docString;
    }

    return result;
}

OnnxModelService.prototype.formatTensor = function(tensor) {
    var result = {};
    result['name'] = tensor.name;
    result['type'] = this.formatElementType(tensor.dataType);
    if (tensor.dims) { 
        result['type'] += '[' + tensor.dims.map(dimension => dimension.toString()).join(',') + ']';
    }
    result['value'] = function() {
        return new OnnxTensorFormatter(tensor).toString();
    }
    return result;
}

OnnxModelService.prototype.formatElementType = function(elementType) {
    var name = this.elementTypeMap[elementType];
    if (name) {
        return name;
    }
    debugger;
    return this.elementTypeMap[onnx.TensorProto.DataType.UNDEFINED];
}

function OnnxTensorFormatter(tensor) {
    this.tensor = tensor;

    if (!this.tensor.dataType) {
        this.output = 'Tensor has no data type.'
        return;
    }

    if (!this.tensor.dims) {
        this.output = 'Tensor has no dimensions.';
        return;
    }

    var size = 1;
    this.tensor.dims.forEach(function (dimSize) {
        size *= dimSize;
    });
    if (size > 65536) {
        this.output = 'Tensor is too large to display.' 
        return;
    }

    switch (this.tensor.dataType) {
        case onnx.TensorProto.DataType.FLOAT:
            if (tensor.floatData && tensor.floatData.length > 0) {
                this.data = tensor.floatData;
            }
            else if (tensor.rawData && tensor.rawData.length > 0) {
                this.rawData = new DataView(tensor.rawData.buffer);
            }
            else {
                this.output = 'Tensor data is empty.';
            }
            break;
        case onnx.TensorProto.DataType.DOUBLE:
            if (tensor.doubleData && tensor.doubleData.length > 0) {
                this.data = tensor.doubleData;
            }
            else if (tensor.rawData && tensor.rawData.length > 0) {
                this.rawData = new DataView(tensor.rawData.buffer);
            }
            else {
                this.output = 'Tensor data is empty.';
            }
            break;
        case onnx.TensorProto.DataType.INT32:
            if (tensor.int32Data && tensor.int32Data.length > 0) {
                this.data = tensor.int32Data;
            }
            else if (tensor.rawData && tensor.rawData.length > 0) {
                this.rawData = new DataView(tensor.rawData.buffer);
            }
            else {
                this.output = 'Tensor data is empty.';
            }
        /* case onnx.TensorProto.DataType.INT64:
            if (tensor.int64Data && tensor.int64Data.length > 0) {
                this.data = tensor.int64Data;
            }
            else if (tensor.rawData && tensor.rawData.length > 0) {
                this.rawData = new DataView(tensor.rawData.buffer);
            }
            else {
                this.output = 'Tensor data is empty.';
            } */
        default:
            this.output = 'Tensor data type is not implemented.';
            break;
    }

    if (!this.output) {
        this.index = 0;
        var result = this.read(0);
        this.output = JSON.stringify(result, null, 4);
    }
}

OnnxTensorFormatter.prototype.read = function(dimension) {
    var size = this.tensor.dims[dimension];
    var result = [];
    if (dimension == this.tensor.dims.length - 1) {
        for (var i = 0; i < size; i++) {
            if (this.data) {
                result.push(this.data[this.index++]);
            }
            else if (this.rawData) {
                switch (this.tensor.dataType)
                {
                    case onnx.TensorProto.DataType.FLOAT:
                        result.push(this.rawData.getFloat32(this.index));
                        this.index += 4;
                        break;
                    case onnx.TensorProto.DataType.DOUBLE:
                        result.push(this.rawData.getFloat64(this.index));
                        this.index += 8;
                        break;
                    case onnx.TensorProto.DataType.INT32:
                        result.push(this.rawData.getInt32(this.index));
                        this.index += 4;
                        break;
                    /* case onnx.TensorProto.DataType.INT64:
                        result.push(this.rawData.getInt64(this.index));
                        this.index += 8;
                        break;*/
                }
            }
        }
    }
    else {
        for (var i = 0; i < size; i++) {
            result.push(this.read(dimension + 1));
        }
    }
    return result;
};

OnnxTensorFormatter.prototype.toString = function() { 
    return this.output;
};

function OnnxOperatorService(hostService) {
    var self = this;
    self.map = {};
    hostService.getResource('onnx-operator.json', function(err, data) {
        if (err != null) {
            // TODO error
        }
        else {
            var items = JSON.parse(data);
            if (items) {
                items.forEach(function (item) {
                    if (item["op_type"] && item["schema"])
                    var op_type = item["op_type"];
                    var schema = item["schema"];
                    self.map[op_type] = schema;
                });
            }
        }
    });
}

OnnxOperatorService.prototype.getInputName = function(operator, index) {
    var schema = this.map[operator];
    if (schema) {
        var inputs = schema["inputs"];
        if (inputs && index < inputs.length) {
            var input = inputs[index];
            if (input) {
                var name = input["name"];
                if (name) {
                    return name;
                }
            } 
        }
    }
    return "(" + index.toString() + ")";
}

OnnxOperatorService.prototype.getOutputName = function(operator, index) {
    var schema = this.map[operator];
    if (schema) {
        var outputs = schema["outputs"];
        if (outputs && index < outputs.length) {
            var output = outputs[index];
            if (output) {
                var name = output["name"];
                if (name) {
                    return name;
                }
            } 
        }
    }
    return "(" + index.toString() + ")";
}

OnnxOperatorService.prototype.getOperatorDocumentation = function(operator) {
    var schema = this.map[operator];
    if (schema) {
        schema = Object.assign({}, schema);
        schema['op_type'] = operator;
        if (schema['doc']) {
            var input = schema['doc'].split('\n');
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
                            lines = lines.map(text => text.substring(2));
                            output.push('<pre>' + lines.join('') + '</pre>');
                        }
                        else {
                            var text = lines.join('');
                            var text = text.replace(/\`\`(.*?)\`\`/gm, (match, content) => '<code>' + content + '</code>');
                            var text = text.replace(/\`(.*?)\`/gm, (match, content) => '<code>' + content + '</code>');
                            output.push('<p>' + text + '</p>')
                        }
                    }
                    lines = [];
                    code = true;
                }
            }
            schema['doc'] = output.join('');
        }
        if (schema['type_constraints']) {
            schema['type_constraints'].forEach(function (item) {
                if (item['allowed_type_strs']) {
                    item['allowed_type_strs_display'] = item['allowed_type_strs'].map(function (type) { return type; }).join(', ');
                }
            });
            var template = Handlebars.compile(operatorTemplate, 'utf-8');
        }
        return template(schema);
    }
    return "";
}
