/*jshint esversion: 6 */

var openvinoDot = openvinoDot || {};

openvinoDot.ModelFactory = class {
    match(context) {
        return context.identifier.endsWith('.dot');
    }

    open(context, host, callback) {
        host.require('./openvino-dot-proto', (err) => {
            if (err) {
                callback(err, null);
                return;
            }

            try {
                var xml_content = new TextDecoder("utf-8").decode(context.buffer);
            } catch (error) {
                callback(new openvinoDot.Error('File format is not OpenVINO IR Dot compliant.'), null);
                return;
            }

            try {
                var parsed_xml = OpenVINODotParser.parse(xml_content);
            } catch (error) {
                callback(new openvinoDot.Error('Unable to parse OpenVINO IR Dot file.'), null);
                return;
            }

            try {
                var model = new openvinoDot.Model(parsed_xml);
            } catch (error) {
                host.exception(error, false);
                callback(new openvinoDot.Error(error.message), null);
                return;
            }

            openvinoDot.OperatorMetadata.open(host, (err, metadata) => {
                callback(null, model);
            });
        });
    }
}

openvinoDot.Model = class {
    constructor(netDef, init) {
        var graph = new openvinoDot.Graph(netDef, init);
        this._graphs = [graph];
    }

    get format() {
        return 'OpenVINO IR Dot';
    }

    get graphs() {
        return this._graphs;
    }
}

openvinoDot.Error = class extends Error {
    constructor(message) {
        super(message);
        this.name = 'Error loading OpenVINO IR Dot model.';
    }
}

openvinoDot.Graph = class {
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
            const node = new openvinoDot.Node(layer, this._version, edges, layers);
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
}

openvinoDot.Node = class {
    constructor(layer, version, edges, layers) {
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

            this._attributes.push(new openvinoDot.Attribute(this, name, value));
        });
    }

    get name() {
        return this._name;
    }

    get operator() {
        return this._type;
    }

    get category() {
        return openvinoDot.OperatorMetadata.operatorMetadata.getOperatorCategory(this._type);
    }

    get documentation() {
        return openvinoDot.OperatorMetadata.operatorMetadata.getOperatorDocumentation(this._type);
    }

    updateInputs(id) {
        this._inputs.push(id);
    }

    updateOutputs(id) {
        this._outputs.push(id);
    }

    get inputs() {
        const list = this._inputs.concat(this._initializers);
        const inputs = openvinoDot.OperatorMetadata.operatorMetadata.getInputs(this._type, list);
        return inputs.map((input) => {
            return new openvinoDot.Argument(input.name, input.connections.map((connection) => {
                if (connection.id instanceof openvinoDot.Tensor) {
                    return new openvinoDot.Connection('', null, connection.id);
                }
                return new openvinoDot.Connection(connection.id, null, null);
            }));
        });
    }

    get outputs() {
        const outputs = openvinoDot.OperatorMetadata.operatorMetadata.getOutputs(this._type, this._outputs, this._id);
        return outputs.map((output) => {
            return new openvinoDot.Argument(output.name, output.connections.map((connection) => {
                return new openvinoDot.Connection(connection.id, null, null);
            }));
        });
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
            return parent.name;
        })
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
            return child.name;
        })
    }
}

openvinoDot.Argument = class {
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
}

openvinoDot.Connection = class {
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
}

openvinoDot.Attribute = class {
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

    get visible() {
        const meta = openvinoDot.OperatorMetadata.operatorMetadata;
        return meta.getAttributeVisible(this._node.operator, this._name, this._value);
    }
}

openvinoDot.Tensor = class {
    constructor({data, shape, precision}) {
        this._data = data;
        this._shape = shape;
        const dataType = precision === 'FP32' ? 'float32' : '?';
        this._type = new openvinoDot.TensorType(dataType, this._shape);
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
}

openvinoDot.TensorType = class {
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
}

openvinoDot.OperatorMetadata = class {
    static open(host, callback) {
        if (!openvinoDot.OperatorMetadata.operatorMetadata) {
            openvinoDot.OperatorMetadata.operatorMetadata = new openvinoDot.OperatorMetadata();
        }
        callback(null, openvinoDot.OperatorMetadata.operatorMetadata);
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

    getOperatorCategory(operator) {
        var schema = this._map[operator];
        if (schema && schema.category) {
            return schema.category;
        }
        return null;
    }

    getOperatorDocumentation(operator) {
        var schema = this._map[operator];
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
        return '';
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

    getAttributeVisible(operator, name, value) {
        var schema = this._map[operator];
        if (schema && schema.attributes && schema.attributes.length > 0) {
            if (!schema.attributesMap) {
                schema.attributesMap = {};
                schema.attributes.forEach((attribute) => {
                    schema.attributesMap[attribute.name] = attribute;
                });
            }
            var attribute = schema.attributesMap[name];
            if (attribute) {
                if (attribute.hasOwnProperty('visible')) {
                    return attribute.visible;
                }
                if (attribute.hasOwnProperty('default')) {
                    return value != attribute.default.toString();
                }
            }
        }
        return true;
    }
}

if (typeof module !== 'undefined' && typeof module.exports === 'object') {
    module.exports.ModelFactory = openvinoDot.ModelFactory;
}
