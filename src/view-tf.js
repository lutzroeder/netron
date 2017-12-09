
// Experimental

var tensorflow = protobuf.roots.tf.tensorflow;

function TensorFlowModel(hostService) {
    this.metadataMap = {};
}

TensorFlowModel.prototype.openBuffer = function(buffer, identifier) { 
    try {
        this.model = tensorflow.SavedModel.decode(buffer);
        this.activeGraph = (this.model.metaGraphs.length > 0) ? this.model.metaGraphs[0] : null;
    }
    catch (err) {
        return err;
    }
    return null;
};

TensorFlowModel.prototype.getGraphMetadata = function(graph) {
    var graphMetadata = this.metadataMap[graph];
    if (!graphMetadata) {
        graphMetadata = new TensorFlowGraphMetadata(graph.metaInfoDef);
    }
    return graphMetadata;
};

TensorFlowModel.prototype.formatModelSummary = function() {
    var summary = { properties: [], graphs: [] };

    var format = 'TensorFlow Saved Model' + (this.model.savedModelSchemaVersion ? (' v' + this.model.savedModelSchemaVersion.toString()) : '');
    summary.properties.push({ name: 'Format', value: format });

    for (var i = 0; i < this.model.metaGraphs.length; i++) {
        var graph = this.model.metaGraphs[i];
        summary.graphs.push({
            name: graph.anyInfo ? graph.anyInfo.toString() : ('(' + i.toString() + ')')
        });
    }
    // model.metaGraphs[x].metaInfoDef.tensorflowGitVersion/tensorflowVersion

    // TODO
    return summary;
};

TensorFlowModel.prototype.getActiveGraph = function() {
    return this.activeGraph;
};

TensorFlowModel.prototype.updateActiveGraph = function(name) {
    for (var i = 0; i < this.model.metaGraphs.length; i++) {
        var graph = this.model.metaGraphs[i];
        var graphName = graph.anyInfo ? graph.anyInfo.toString() : ('(' + i.toString() + ')');
        if (name == graphName) {
            this.activeGraph = graph;
            return;            
        }
    }
};

TensorFlowModel.prototype.getGraphs = function() {
    return this.model.metaGraphs;    
};

TensorFlowModel.prototype.getGraphInputs = function(graph) {
    return [];
};

TensorFlowModel.prototype.getGraphOutputs = function(graph) {
    return [];
};

TensorFlowModel.prototype.getGraphInitializers = function(graph) {
    var results = [];
    graph.graphDef.node.forEach(function (node) {
        if (node.op == 'Const') {
            results.push({
                'id': node.name,
                'name': node.name,
                'type': ''
            });
        }
    });
    return results;
};

TensorFlowModel.prototype.getNodes = function(graph) {
   // graph.graphDef.node.forEach(function (node) {
   //     console.log(node.name + ' [' + (!node.input ? "" : node.input.map(s => s).join(',')) + ']');
   // });
    var result = [];
    graph.graphDef.node.forEach(function (node) {
        if (node.op != 'Const') {
            result.push(node);
        }
    });
    return result;
};

TensorFlowModel.prototype.getNodeOperator = function(node) {
    return node.op;
};

TensorFlowModel.prototype.getNodeOperatorDocumentation = function(graph, node) {
    var graphMetadata = this.getGraphMetadata(graph);
    if (graphMetadata) {
        return graphMetadata.getOperatorDocumentation(node.op);       
    }
    return null;
};

TensorFlowModel.prototype.getNodeInputs = function(graph, node) {
    var graphMetadata = this.getGraphMetadata(graph);
    var self = this;
    var result = [];
    if (node.input) {
        node.input.forEach(function(input, index) {
            if (input.startsWith('^')) {
                input = input.substring(1);
            }
            result.push({
                'id': input, 
                'name': graphMetadata ? graphMetadata.getInputName(node.op, index) : ('(' + index.toString() + ')'),
                'type': ''
            });
        });
    }
    return result;
};

TensorFlowModel.prototype.getNodeOutputs = function(graph, node) {
    var graphMetadata = this.getGraphMetadata(graph);
    var index = 0;
    return [ { 
        'id': node.name,
        'name': graphMetadata ? graphMetadata.getOutputName(node.op, index) : ('(' + index.toString() + ')'),
        'type': ''
    } ];
};

TensorFlowModel.prototype.formatNodeProperties = function(node) {
    var result = null;
    if (node.name) {
        result = [];
        if (node.name) {
            result.push({ 'name': 'name', 'value': node.name, 'value_short': function() { return node.name; } });
        }
    }
    return result;
};

TensorFlowModel.prototype.formatNodeAttributes = function(node) {
    var self = this;
    var result = [];
    if (node.attr) {
        Object.keys(node.attr).forEach(function (name) {
            if (name != '_output_shapes') {
                var value = node.attr[name];
                result.push({ 
                    'name': name,
                    'type': '',
                    'value': function() { return '...'; }, 
                    'value_short': function() { return '...'; }
                });    
            }
        });
    }
    return result;
};

TensorFlowModel.prototype.formatTensorType = function(tensor) {
    var result = "?";
    return result;
};

TensorFlowModel.prototype.formatTensor = function(tensor) {
    var result = {};
    result.name = tensor.name;
    result.type = this.formatTensorType(tensor);
    result.value = function() {
        return '?';
    };
    return result;
};

function TensorFlowGraphMetadata(metaInfoDef) {
    var self = this;
    self.schemaMap = {};
    metaInfoDef.strippedOpList.op.forEach(function (opDef) {
        var schema = { inputs: [], outputs: [], attributes: [] };
        opDef.inputArg.forEach(function (inputArg) {
            schema.inputs.push({ name: inputArg.name, typeStr: inputArg.typeAttr });
        });
        opDef.outputArg.forEach(function (outputArg) {
            schema.outputs.push({ name: outputArg.name, typeStr: outputArg.typeAttr });
        });
        opDef.attr.forEach(function (attr) {
            schema.attributes.push({ name: attr.name, type: attr.type });
        });
        self.schemaMap[opDef.name] = schema;
    });
}

TensorFlowGraphMetadata.prototype.getInputName = function(operator, index) {
    var schema = this.schemaMap[operator];
    if (schema) {
        var inputs = schema.inputs;
        if (inputs && index < inputs.length) {
            var input = inputs[index];
            if (input) {
                var name = input.name;
                if (name) {
                    return name;
                }
            }
        }
    }
    return '(' + index.toString() + ')';
};

TensorFlowGraphMetadata.prototype.getOutputName = function(operator, index) {
    var schema = this.schemaMap[operator];
    if (schema) {
        var outputs = schema.outputs;
        if (outputs && index < outputs.length) {
            var output = outputs[index];
            if (output) {
                var name = output.name;
                if (name) {
                    return name;
                }
            }
        }
    }
    return '(' + index.toString() + ')';
};

TensorFlowGraphMetadata.prototype.getOperatorDocumentation = function(operator) {
    var schema = this.schemaMap[operator];
    if (schema) {
        var template = Handlebars.compile(operatorTemplate, 'utf-8');
        return template(schema);
    }
    return null;
};