/*jshint esversion: 6 */

// Experimental

var tensorflow = protobuf.roots.tf.tensorflow;

class TensorFlowModel {

    constructor(hostService) {
        this.metadataMap = {};
    }

    openBuffer(buffer, identifier) { 
        try {
            this.model = tensorflow.SavedModel.decode(buffer);
            this.activeGraph = (this.model.metaGraphs.length > 0) ? this.model.metaGraphs[0] : null;
        }
        catch (err) {
            return err;
        }
        return null;
    }

    getGraphMetadata(graph) {
        var graphMetadata = this.metadataMap[graph];
        if (!graphMetadata) {
            graphMetadata = new TensorFlowGraphMetadata(graph.metaInfoDef);
        }
        return graphMetadata;
    }

    formatModelSummary() {
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
    }

    getActiveGraph() {
        return this.activeGraph;
    }

    updateActiveGraph(name) {
        for (var i = 0; i < this.model.metaGraphs.length; i++) {
            var graph = this.model.metaGraphs[i];
            var graphName = graph.anyInfo ? graph.anyInfo.toString() : ('(' + i.toString() + ')');
            if (name == graphName) {
                this.activeGraph = graph;
                return;            
            }
        }
    }

    getGraphs() {
        return this.model.metaGraphs;    
    }

    getGraphInputs(graph) {
        return [];
    }

    getGraphOutputs(graph) {
        return [];
    }

    getGraphInitializers(graph) {
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
    }

    getNodes(graph) {
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
    }

    getNodeOperator(node) {
        return node.op;
    }

    getNodeOperatorDocumentation(graph, node) {
        var graphMetadata = this.getGraphMetadata(graph);
        if (graphMetadata) {
            return graphMetadata.getOperatorDocumentation(node.op);       
        }
        return null;
    }

    getNodeInputs(graph, node) {
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
    }

    getNodeOutputs(graph, node) {
        var graphMetadata = this.getGraphMetadata(graph);
        var index = 0;
        return [ { 
            'id': node.name,
            'name': graphMetadata ? graphMetadata.getOutputName(node.op, index) : ('(' + index.toString() + ')'),
            'type': ''
        } ];
    }

    formatNodeProperties(node) {
        var result = null;
        if (node.name) {
            result = [];
            if (node.name) {
                result.push({ 'name': 'name', 'value': node.name, 'value_short': function() { return node.name; } });
            }
        }
        return result;
    }

    formatNodeAttributes(node) {
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
    }

    formatTensorType(tensor) {
        var result = "?";
        return result;
    }

    formatTensor(tensor) {
        var result = {};
        result.name = tensor.name;
        result.type = this.formatTensorType(tensor);
        result.value = () => {
            return '?';
        };
        return result;
    }
}

class TensorFlowGraphMetadata {

    constructor(metaInfoDef) {
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

    getInputName(operator, index) {
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
    }

    getOutputName(operator, index) {
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
    }

    getOperatorDocumentation(operator) {
        var schema = this.schemaMap[operator];
        if (schema) {
            var template = Handlebars.compile(operatorTemplate, 'utf-8');
            return template(schema);
        }
        return null;
    }
}