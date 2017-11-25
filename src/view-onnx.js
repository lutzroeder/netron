
function OnnxModelService(hostService) {
    this.hostService = hostService;
    this.operatorService = new OnnxOperatorService(hostService);
}

OnnxModelService.prototype.openBuffer = function(buffer) { 
    this.model = onnx.ModelProto.decode(buffer);
}

OnnxModelService.prototype.getModelProperties = function() {

    var result = { 'groups': [] };

    var generalProperties = { 'name': 'General', 'properties': [] };
    result['groups'].push(generalProperties);

    generalProperties['properties'].push({ 'name': 'Format', 'value': 'ONNX' });
    if (this.model.irVersion) {
        generalProperties['properties'].push({ 'name': 'Version', 'value': this.model.irVersion.toString() });
    }
    if (this.model.domain) {
        generalProperties['properties'].push({ 'name': 'Domain', 'value': this.model.domain });
    }
    if (this.model.graph && this.model.graph != null && this.model.graph.name) {
        generalProperties['properties'].push({ 'name': 'Name', 'value': this.model.graph.name });
    }
    if (this.model.model_version) {
        generalProperties['properties'].push({ 'name': 'Model Version', 'value': this.model.model_version });
    }
    var producer = [];
    if (this.model.producerName) {
        producer.push(this.model.producerName);
    }
    if (this.model.producerVersion && this.model.producerVersion.length > 0) {
        producer.push(model.producerVersion);
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

OnnxModelService.prototype.getOperatorService = function() {
    return this.operatorService;
}

function OnnxOperatorService(hostService) {
    var self = this;
    self.hostService = hostService;
    self.map = {};
    self.hostService.getResource('onnx-operator.json', function(err, data) {
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

OnnxOperatorService.prototype.getHtmlDocumentation = function(operator) {
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
                if (line.length == 0 || input.length == 0) {
                    if (lines.length > 0) {
                        if (code) {
                            lines = lines.map(text => text.substring(2));
                            output.push('<pre>' + lines.join('') + '</pre>');
                        }
                        else {
                            var text = lines.join('').replace(/\`(.*?)\`/gm, (match, content) => '<code>' + content + '</code>');
                            output.push('<p>' + text + '</p>')
                        }
                    }
                    lines = [];
                    code = true;
                }
                else if (!line.startsWith('  ')) {
                    code = false;
                    lines.push(line + "\n");
                }
                else {
                    lines.push(line + "\n");
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
