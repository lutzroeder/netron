            /*jshint esversion: 6 */

var openvino = openvino || {};
openvino.ir = openvino.ir || {};
openvino.dot = openvino.dot || {};

openvino.ModelFactory = class {

    match(context) {
        var extension = context.identifier.split('.').pop().toLowerCase();
        if (extension === 'xml') {
            if (context.text.includes('<net')) {
                return true;
            }
        }
        if (extension === 'dot') {
            if (context.text.includes('layer_')) {
                return true;
            }
        }
        return false;
    }

    open(context, host, callback) {
        host.require('./openvino-parser', (err, openvino_parser) => {
            if (err) {
                callback(err, null);
                return;
            }

            var extension = context.identifier.split('.').pop().toLowerCase();
            var parsed = null;
            var model = null;
            if (extension === 'xml') {
                try {
                    parsed = openvino_parser.IrParser.parse(context.text);
                } catch (error) {
                    callback(new openvino.Error('Failed to read OpenVINO IR file.'), null);
                    return;
                }
                try {
                    model = new openvino.ir.Model(parsed);
                } catch (error) {
                    host.exception(error, false);
                    callback(new openvino.Error(error.message), null);
                    return;
                }
            }

            if (extension === 'dot') {
                try {
                    parsed = openvino_parser.DotParser.parse(context.text);
                } catch (error) {
                    callback(new openvino.Error('Failed to read OpenVINO Dot file.'), null);
                    return;
                }
                try {
                    model = new openvino.dot.Model(parsed);
                } catch (error) {
                    host.exception(error, false);
                    callback(new openvino.Error(error.message), null);
                    return;
                }
            }

            openvino.OperatorMetadata.open(host, (err, metadata) => {
                callback(null, model);
            });
        });
    }
};

openvino.ir.Model = class {

    constructor(netDef, init) {
        var graph = new openvino.ir.Graph(netDef, init);
        this._graphs = [ graph ];
    }

    get format() {
        return 'OpenVINO IR';
    }

    get graphs() {
        return this._graphs;
    }
};


openvino.ir.Graph = class {

    constructor(netDef, init) {
        this._name = netDef.net.name || '';
        this._batch = +netDef.net.batch || '';
        this._version = +netDef.net.version || '';

        this._nodes = [];
        this._operators = {};
        this._inputs = [];
        this._outputs = [];

        netDef.layers.forEach((layer) => {
            const node = new openvino.ir.Node(layer, this._version, netDef.edges, netDef.layers);
            this._operators[node.operator] = this._operators[node.operator] ? this._operators[node.operator] + 1 : 1;
            this._nodes.push(node);
        });
    }

    get name() {
        return this._name;
    }

    get inputs() {
        return this._inputs;
    }

    get outputs() {
        return this._outputs;
    }

    get nodes() {
        return this._nodes;
    }

    get operators() {
        return this._operators;
    }
};

openvino.AbstractNode = class {

    get id() {
        return this._id;
    }

    get name() {
        return this._name;
    }

    get device() {
        return this._device || '';
    }

    get operator() {
        return this._type;
    }

    get category() {
        var schema = openvino.OperatorMetadata.operatorMetadata.getSchema(this._type);
        return (schema && schema.category) ? schema.category : null;
    }

    get documentation() {
        var schema = openvino.OperatorMetadata.operatorMetadata.getSchema(this._type);
        if (schema) {
            schema = JSON.parse(JSON.stringify(schema));
            schema.name = operator;
            if (schema.description) {
                schema.description = marked(schema.description);
            }
            if (schema.attributes) {
                schema.attributes.forEach((attribute) => {
                    if (attribute.description) {
                        attribute.description = marked(attribute.description);
                    }
                });
            }
            if (schema.inputs) {
                schema.inputs.forEach((input) => {
                    if (input.description) {
                        input.description = marked(input.description);
                    }
                });
            }
            if (schema.outputs) {
                schema.outputs.forEach((output) => {
                    if (output.description) {
                        output.description = marked(output.description);
                    }
                });
            }
            if (schema.references) {
                schema.references.forEach((reference) => {
                    if (reference) {
                        reference.description = marked(reference.description);
                    }
                });
            }
            return schema;
        }
        return null;
    }

    get attributes() {
        return this._attributes;
    }

    _concatBinaryAttributes(data) {
        return `offset: ${data.offset}, size: ${data.size}`;
    }

    setInputs(inputs, edges, layers) {
        if (!inputs) {
            this._inputs = [];
            return;
        }

        this._inputs = inputs.map((input) => {
            const candidate_edge = edges.find((edge) => {
                return edge['to-layer'] === this._id && edge['to-port'] === input.id;
            });
            if (!candidate_edge) {
                return;
            }
            const parentID = candidate_edge['from-layer'];
            const parent = layers.find((layer) => layer.id === parentID);
            if (!parent) {
                return;
            }
            return parent.id;
        });
    }

    setOutputs(outputs, edges, layers) {
        if (!outputs) {
            this._outputs = [];
            return;
        }

        this._outputs = outputs.map((output) => {
            const candidate_edge = edges.find((edge) => {
                return edge['from-layer'] === this._id && edge['from-port'] === output.id;
            });
            if (!candidate_edge) {
                return;
            }
            const childID = candidate_edge['to-layer'];
            const child = layers.find((layer) => layer.id === childID);
            if (!child) {
                return;
            }
            return child.id;
        });
    }

    get inputs() {
        const list = this._inputs.concat(this._initializers);
        const inputs = openvino.OperatorMetadata.operatorMetadata.getInputs(this._type, list);
        return inputs.map((input) => {
            return new openvino.Argument(input.name, input.connections.map((connection) => {
                if (connection.id instanceof openvino.Tensor) {
                    return new openvino.Connection('', null, connection.id);
                }
                return new openvino.Connection(connection.id, null, null);
            }));
        });
    }

    get outputs() {
        const outputs = openvino.OperatorMetadata.operatorMetadata.getOutputs(this._type, this._outputs, this._id);
        return outputs.map((output) => {
            return new openvino.Argument(output.name, output.connections.map((connection) => {
                return new openvino.Connection(connection.id, null, null);
            }));
        });
    }
};

openvino.ir.Node = class extends openvino.AbstractNode {

    constructor(layer, version, edges, layers) {
        super();

        this._type = layer.type;
        this._name = layer.name || '';
        this._id = layer.id;

        this._inputs = [];
        this._outputs = [];

        this.setInputs(layer.input, edges, layers);
        if (layer.hasOwnProperty(0)) {
             // meaning it has outputs 
             this.setOutputs(layer[0].output, edges, layers);
        }

        this._initializers = [];
        this._attributes = [];

        this._attributes.push(new openvino.Attribute(this, 'precision', layer.precision));

        if (layer.data) {
            this._attributes = Object.keys(layer.data).map((key) => {
                return new openvino.Attribute(this, key, layer.data[key]);
            });
        }

        if (layer.biases) {
            const value = this._concatBinaryAttributes(layer.biases);
            this._attributes.push(new openvino.Attribute(this, 'biases', value));

            // TODO: complex to extract the size of the bias
            // TODO: compute from the overall size?
            // this._initializers.push(new openvinoTensor({data: [],
            //     shape: [layer[0].output[0].dims[1]],
            //     precision: layer.precision
            // }));
        }

        if (layer.weights) {
            const value = this._concatBinaryAttributes(layer.weights);
            this._attributes.push(new openvino.Attribute(this, 'weights', value));


            // this._initializers.push(new openvinoTensor({data: [],
            //     shape: layer[0].output[0].dims,
            //     precision: layer.precision
            // }));
        }
    }
};

openvino.Argument = class {

    constructor(name, connections) {
        this._name = name;
        this._connections = connections;
    }

    get name() {
        return this._name;
    }

    get visible() {
        return true;
    }

    get connections() {
        return this._connections;
    }
};

openvino.Connection = class {

    constructor(id, type, initializer) {
        this._id = id;
        this._type = type || null;
        this._initializer = initializer || null;
    }

    get id() {
        return this._id;
    }

    get type() {
        if (this._initializer) {
            return this._initializer.type;
        }
        return this._type;
    }

    get initializer() {
        return this._initializer;
    }
};

openvino.Attribute = class {

    constructor(node, name, value) {
        this._node = node;
        this._name = name;
        this._value = value;
    }

    get name() {
        return this._name;
    }

    get value() {
        return this._value;
    }
};

openvino.Tensor = class {

    constructor({data, shape, precision}) {
        this._data = data;
        this._shape = shape;
        const dataType = precision === 'FP32' ? 'float32' : '?';
        this._type = new openvino.TensorType(dataType, this._shape);
    }

    get kind() {
        return 'Blob';
    }

    get type() {
        return this._type;
    }

    get state() {
        return this._context().state;
    }

    get value() {
        var context = this._context();
        if (context.state) {
            return null;
        }
        context.limit = Number.MAX_SAFE_INTEGER;
        return this._decode(context, 0);
    }

    toString() {
        var context = this._context();
        if (context.state) {
            return '';
        }
        context.limit = 10000;
        var value = this._decode(context, 0);
        return JSON.stringify(value, null, 4);
    }

    _context() {
        var context = {};
        context.state = null;
        context.index = 0;
        context.count = 0;
        context.data = this._data;
        if (!this._data) {
            context.state = 'Tensor data is empty.';
        }
        return context;
    }

    _decode(context, dimension) {
        var results = [];
        var size = this._shape[dimension];
        if (dimension == this._shape.length - 1) {
            for (var i = 0; i < size; i++) {
                if (context.count > context.limit) {
                    results.push('...');
                    return results;
                }
                results.push(context.data[context.index]);
                context.index++;
                context.count++;
            }
        }
        else {
            for (var j = 0; j < size; j++) {
                if (context.count > context.limit) {
                    results.push('...');
                    return results;
                }
                results.push(this._decode(context, dimension + 1));
            }
        }
        return results;
    }
};

openvino.TensorType = class {

    constructor(dataType, shape) {
        this._dataType = dataType;
        this._shape = shape;
    }

    get dataType() {
        return this._dataType;
    }

    get shape() {
        return this._shape;
    }

    toString() {
        return this.dataType + (this._shape ? ('[' + this._shape.map((dimension) => dimension.toString()).join(',') + ']') : '');
    }
};

openvino.OperatorMetadata = class {

    static open(host, callback) {
        if (!openvino.OperatorMetadata.operatorMetadata) {
            openvino.OperatorMetadata.operatorMetadata = new openvino.OperatorMetadata();
        }
        callback(null, openvino.OperatorMetadata.operatorMetadata);
    }

    constructor(data) {
        this._map = {};
        if (data) {
            var items = JSON.parse(data);
            if (items) {
                items.forEach((item) => {
                    if (item.name && item.schema) {
                        var name = item.name;
                        var schema = item.schema;
                        this._map[name] = schema;
                    }
                });
            }
        }
    }

    getSchema(operator) {
        return this._map[operator];
    }

    getInputs(type, inputs) {
        var results = [];
        var index = 0;

        inputs.slice(index).forEach((input) => {
            const name = (index === 0) ? 'input' : ('(' + index.toString() + ')');
            results.push({
                name: name,
                connections: [{id: input}]
            });
            index++;
        });

        return results;
    }

    getOutputs(type, outputs, layerName) {
        var results = [];
        var index = 0;

        outputs.slice(index).forEach((output) => {
            const name = (index === 0) ? 'output' : ('(' + index.toString() + ')');
            results.push({
                name: name,
                connections: [{id: layerName}]
            });
            index++;
        });
        return results;
    }
};

openvino.dot.Model = class {

    constructor(netDef, init) {
        var graph = new openvino.dot.Graph(netDef, init);
        this._graphs = [ graph ];
    }

    get format() {
        return 'OpenVINO IR Dot';
    }

    get graphs() {
        return this._graphs;
    }
};

openvino.dot.Graph = class {

    constructor(netDef, init) {
        this._name = netDef.id || '';
        this._version = Boolean(netDef.strict).toString();

        this._nodes = [];
        this._operators = {};
        this._inputs = [];
        this._outputs = [];

        const layers = netDef.children.filter((child) => child.type === "node_stmt");
        const edges = netDef.children.filter((child) => child.type === "edge_stmt");

        layers.forEach((layer) => {
            const node = new openvino.dot.Node(layer, this._version, edges, layers);
            this._operators[node.operator] = this._operators[node.operator] ? this._operators[node.operator] + 1 : 1;
            this._nodes.push(node);
        });

        edges.forEach((edge) => {
            const from = edge.edge_list[0];
            const to = edge.edge_list[1];
            const child = this._nodes.find((node) => node._id === to.id);
            if (child) {
                child.updateInputs(from.id);
            }
            const parent = this._nodes.find((node) => node._id === from.id);
            if (parent) {
                parent.updateOutputs(to.id);
            }
        });
    }

    get name() {
        return this._name;
    }

    get inputs() {
        return this._inputs;
    }

    get outputs() {
        return this._outputs;
    }

    get nodes() {
        return this._nodes;
    }

    get operators() {
        return this._operators;
    }
};

openvino.dot.Node = class extends openvino.AbstractNode {

    constructor(layer, version, edges, layers) {
        super();
        this._inputs = [];
        this._outputs = [];
        this._id = layer.node_id.id;

        this._initializers = [];
        this._attributes = [];

        const ownAttributes = ['name', 'shape', 'style', 'fillcolor', 'type'];

        layer.attr_list.forEach(({name, value}) => {
            name = name.toLowerCase().replace(/\s/g, '_');
            if (ownAttributes.includes(name)) {
                this[`_${name}`] = value;
            }

            this._attributes.push(new openvino.Attribute(this, name, value));
        });

        if (!this._type){
            this._type = 'data';
        }
    }

    updateInputs(id) {
        this._inputs.push(id);
    }

    updateOutputs(id) {
        this._outputs.push(id);
    }
};

openvino.Error = class extends Error {

    constructor(message) {
        super(message);
        this.name = 'Error loading OpenVINO model.';
    }
};

if (typeof module !== 'undefined' && typeof module.exports === 'object') {
    module.exports.ModelFactory = openvino.ModelFactory;
}
