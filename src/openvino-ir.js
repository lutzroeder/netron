/*jshint esversion: 6 */

var openvinoIR = openvinoIR || {};

openvinoIR.ModelFactory = class {
    match(context) {
        return context.identifier.endsWith('.xml');
    }

    open(context, host, callback) {
        host.require('./openvino-ir-proto', (err) => {
            if (err) {
                callback(err, null);
                return;
            }

            try {
                var xml_content = new TextDecoder("utf-8").decode(context.buffer);
            } catch (error) {
                callback(new openvinoIR.Error('File format is not OpenVINO IR compliant.'), null);
                return;
            }

            try {
                var parsed_xml = OpenVINOIRParser.parse(xml_content);
            } catch (error) {
                callback(new openvinoIR.Error('Unable to parse OpenVINO IR file.'), null);
                return;
            }

            try {
                var model = new openvinoIR.Model(parsed_xml);
            } catch (error) {
                host.exception(error, false);
                callback(new openvinoIR.Error(error.message), null);
                return;
            }

            openvinoIR.OperatorMetadata.open(host, (err, metadata) => {
                callback(null, model);
            });
        });
    }
}

openvinoIR.Model = class {
    constructor(netDef, init) {
        var graph = new openvinoIR.Graph(netDef, init);
        this._graphs = [graph];
    }

    get format() {
        return 'OpenVINO IR';
    }

    get graphs() {
        return this._graphs;
    }
}

openvinoIR.Error = class extends Error {
    constructor(message) {
        super(message);
        this.name = 'Error loading OpenVINO IR model.';
    }
}

openvinoIR.Graph = class {
    constructor(netDef, init) {
        this._name = netDef.net.name || '';
        this._batch = +netDef.net.batch || '';
        this._version = +netDef.net.version || '';

        this._nodes = [];
        this._operators = {};
        this._inputs = [];
        this._outputs = [];

        netDef.layers.forEach((layer) => {
            const node = new openvinoIR.Node(layer, this._version, netDef.edges, netDef.layers);
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
}

openvinoIR.Node = class {
    constructor(layer, version, edges, layers) {
        switch (version) {
            case 2:
            default:
                this._type = layer.type;
                this._name = layer.name || '';
                this._id = layer.id;
                break;
        }

        this._inputs = [];
        this._outputs = [];

        this.setInputs(layer.input, edges, layers);
        this.setOutputs(layer[0].output, edges, layers);

        this._initializers = [];
        this._attributes = [];

        this._attributes.push(new openvinoIR.Attribute(this, 'precision', layer.precision));

        if (layer.data) {
            this._attributes = Object.keys(layer.data).map((key) => {
                return new openvinoIR.Attribute(this, key, layer.data[key]);
            });
        }

        if (layer.biases) {
            const value = this._concatBinaryAttributes(layer.biases);
            this._attributes.push(new openvinoIR.Attribute(this, 'biases', value));

            // TODO: complex to extract the size of the bias
            // TODO: compute from the overall size?
            // this._initializers.push(new OpenVINOIRTensor({data: [],
            //     shape: [layer[0].output[0].dims[1]],
            //     precision: layer.precision
            // }));
        }

        if (layer.weights) {
            const value = this._concatBinaryAttributes(layer.weights);
            this._attributes.push(new openvinoIR.Attribute(this, 'weights', value));


            // this._initializers.push(new OpenVINOIRTensor({data: [],
            //     shape: layer[0].output[0].dims,
            //     precision: layer.precision
            // }));
        }

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
        return openvinoIR.OperatorMetadata.operatorMetadata.getOperatorCategory(this._type);
    }

    get documentation() {
        return openvinoIR.OperatorMetadata.operatorMetadata.getOperatorDocumentation(this._type);
    }

    get inputs() {
        const list = this._inputs.concat(this._initializers);
        const inputs = openvinoIR.OperatorMetadata.operatorMetadata.getInputs(this._type, list);
        return inputs.map((input) => {
            return new openvinoIR.Argument(input.name, input.connections.map((connection) => {
                if (connection.id instanceof openvinoIR.Tensor) {
                    return new openvinoIR.Connection('', null, connection.id);
                }
                return new openvinoIR.Connection(connection.id, null, null);
            }));
        });
    }

    get outputs() {
        const outputs = openvinoIR.OperatorMetadata.operatorMetadata.getOutputs(this._type, this._outputs, this._name);
        return outputs.map((output) => {
            return new openvinoIR.Argument(output.name, output.connections.map((connection) => {
                return new openvinoIR.Connection(connection.id, null, null);
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
        if (!inputs){
            this._inputs = [];
            return;
        }

        this._inputs = inputs.map((input) => {
            const candidate_edge = edges.find((edge) => {
                return edge['to-layer'] === this._id && edge['to-port'] === input.id;
            });
            if (!candidate_edge){
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
        if (!outputs){
            this._outputs = [];
            return;
        }

        this._outputs = outputs.map((output) => {
            const candidate_edge = edges.find((edge) => {
                return edge['from-layer'] === this._id && edge['from-port'] === output.id;
            });
            if (!candidate_edge){
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

openvinoIR.Argument = class {
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

openvinoIR.Connection = class {
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

openvinoIR.Attribute = class {
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
        const meta = openvinoIR.OperatorMetadata.operatorMetadata;
        return meta.getAttributeVisible(this._node.operator, this._name, this._value);
    }
}

openvinoIR.Tensor = class {
    constructor({data, shape, precision}) {
        this._data = data;
        this._shape = shape;
        const dataType = precision === 'FP32' ? 'float32' : '?';
        this._type = new openvinoIR.TensorType(dataType, this._shape);
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

openvinoIR.TensorType = class {
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

openvinoIR.OperatorMetadata = class {
    static open(host, callback) {
        if (!openvinoIR.OperatorMetadata.operatorMetadata) {
            openvinoIR.OperatorMetadata.operatorMetadata = new openvinoIR.OperatorMetadata();
        }
        callback(null, openvinoIR.OperatorMetadata.operatorMetadata);
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
    module.exports.ModelFactory = openvinoIR.ModelFactory;
}
