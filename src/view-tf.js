/*jshint esversion: 6 */

// Experimental

var tensorflow = protobuf.roots.tf.tensorflow;

class TensorFlowModel {

    constructor(hostService) {
    }

    openBuffer(buffer, identifier) { 
        try {
            if (identifier == 'saved_model.pb') {
                this._model = tensorflow.SavedModel.decode(buffer);
                this._graphs = [];
                for (var i = 0; i < this._model.metaGraphs.length; i++) {
                    this._graphs.push(new TensorFlowGraph(this, this._model.metaGraphs[i], i));
                }
                this._format = 'TensorFlow Saved Model';
                if (this._model.savedModelSchemaVersion) {
                    this._format += ' v' + this._model.savedModelSchemaVersion.toString();
                }
            }
            else {
                var graphDef = tensorflow.GraphDef.decode(buffer);
                var metaGraph = new tensorflow.MetaGraphDef();
                metaGraph.graphDef = graphDef;
                metaGraph.anyInfo = identifier;
                this._model = new tensorflow.SavedModel();
                this._model.metaGraphs.push(metaGraph);
                this._graphs = [ new TensorFlowGraph(this._model, metaGraph) ];
                this._format = 'TensorFlow Graph Defintion';
            }

            this._activeGraph = (this._graphs.length > 0) ? this._graphs[0] : null;            
        }
        catch (err) {
            return err;
        }
        return null;
    }

    format() {
        var summary = { properties: [], graphs: [] };
        summary.properties.push({ name: 'Format', value: this._format });

        this.graphs.forEach((graph) => {
            summary.graphs.push({
                name: graph.name,
                version: graph.version,
                tags: graph.tags
            });
            // metaInfoDef.tensorflowGitVersion
            // TODO signature
        });
    
        return summary;
    }

    get graphs() {
        return this._graphs;    
    }

    get activeGraph() {
        return this._activeGraph;
    }

    updateActiveGraph(name) {
        this.graphs.forEach((graph) => {
            if (name == graph.name) {
                this._activeGraph = graph;
                return;
            }            
        });
    }
}

class TensorFlowGraph {

    constructor(model, graph, index) {
        this._model = model;
        this._graph = graph;
        this._name = this._graph.anyInfo ? this._graph.anyInfo.toString() : ('(' + index.toString() + ')');

        this._outputMap = {};
        this._graph.graphDef.node.forEach((node) => {
            
        });
    }

    get model() {
        return this._model;
    }

    get name() {
        return this._name;
    }

    get version() {
        if (this._graph.metaInfoDef && this._graph.metaInfoDef.tensorflowVersion) {
            return 'TensorFlow ' + this._graph.metaInfoDef.tensorflowVersion;
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
        return [];
    }

    get outputs() {
        return [];
    }

    get initializers() {
        var results = [];
        this._graph.graphDef.node.forEach((node) => {
            if (node.op == 'Const') {
                var id = node.name;
                if (id.indexOf(':') == -1) {
                    id += ':0';
                }
                var tensor = null;
                Object.keys(node.attr).forEach((name) => {
                    if (name == 'value') {
                        tensor = node.attr[name];
                        return;
                    }
                });
                results.push(new TensorFlowTensor(tensor, id, node.name));
            }
        });
        return results;
    }

    get nodes() {
        // graph.graphDef.node.forEach(function (node) {
        //     console.log(node.name + ' [' + (!node.input ? "" : node.input.map(s => s).join(',')) + ']');
        // });
        var results = [];
        this._graph.graphDef.node.forEach((node) => {
            if (node.op != 'Const') {
                results.push(new TensorFlowNode(this, node));
            }
        });
        return results;
    }

    get metadata() {
        if (!this._metadata) {
            this._metadata = new TensorFlowGraphMetadata(this._graph.metaInfoDef);
        }
        return this._metadata;
    }
}

class TensorFlowNode {

    constructor(graph, node) {
        this._graph = graph;
        this._node = node;
    }

    get operator() {
        return this._node.op;
    }

    get inputs() {
        var graphMetadata = this._graph.metadata;
        var node = this._node;
        var result = [];
        if (node.input) {
            node.input.forEach(function(input, index) {
                if (input.startsWith('^')) {
                    input = input.substring(1);
                }
                if (input.indexOf(':') == -1) {
                    input += ':0';
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

    get outputs() {
        var graphMetadata = this._graph.metadata;
        var node = this._node;
        var index = 0;
        var id = node.name + ':0';
        return [ { 
            'id': id,
            'name': graphMetadata ? graphMetadata.getOutputName(node.op, index) : ('(' + index.toString() + ')'),
            'type': ''
        } ];
    }

    get properties() {
        var node = this._node;
        var result = null;
        if (node.name) {
            result = [];
            if (node.name) {
                result.push({
                    'name': 'name', 
                    'value': node.name,
                    'value_short': function() { return node.name; } 
                });
            }
        }
        return result;
    }

    get attributes() {
        var node = this._node;
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

    get documentation() {
        var graphMetadata = this._graph.metadata;
        if (graphMetadata) {
            return graphMetadata.getOperatorDocumentation(this.operator);       
        }
        return null;
    }
}

class TensorFlowTensor {

    constructor(tensor, id) {
        this._tensor = tensor;
        this._id = id;
    }

    get id() {
        return this._id;
    }

    get name() {
        return this._tensor.name;
    }

    get type() {
        return TensorFlowTensor.formatTensorType(this._tensor);
    }

    get value() {
        return '?';        
    }

    static formatTensorType(tensor) {
        var result = "?";
        return result;
    }
}

class TensorFlowGraphMetadata {

    constructor(metaInfoDef) {
        var self = this;
        self.schemaMap = {};
        if (metaInfoDef && metaInfoDef.strippedOpList && metaInfoDef.strippedOpList.op) {
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