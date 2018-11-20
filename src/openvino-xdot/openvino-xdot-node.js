var openvinoXdot = openvinoXdot || {};

openvinoXdot.Node = class {
    constructor(layer, version, edges, layers) {
        this._inputs = [];
        this._outputs = [];
        this._id = layer.node_id.id;

        this._initializers = [];
        this._attributes = [];

        const ownAttributes = ['name', 'shape', 'style', 'fillcolor', 'type'];

        _.each(layer.attr_list, ({name, value}) => {
            name = name.toLowerCase().replace(/\s/g, '_');
            if (_.includes(ownAttributes, name)) {
                this[`_${name}`] = value;
            }

            this._attributes.push(new openvinoXdot.Attribute(this, name, value));
        });
    }

    get name() {
        return this._name;
    }

    get operator() {
        return this._type;
    }

    get category() {
        return openvinoXdot.OperatorMetadata.operatorMetadata.getOperatorCategory(this._type);
    }

    get documentation() {
        return openvinoXdot.OperatorMetadata.operatorMetadata.getOperatorDocumentation(this._type);
    }

    updateInputs(id) {
        this._inputs.push(id);
    }

    updateOutputs(id) {
        this._outputs.push(id);
    }

    get inputs() {
        const list = this._inputs.concat(this._initializers);
        const inputs = openvinoXdot.OperatorMetadata.operatorMetadata.getInputs(this._type, list);
        return inputs.map((input) => {
            return new openvinoXdot.Argument(input.name, input.connections.map((connection) => {
                if (connection.id instanceof CaffeTensor) {
                    return new openvinoXdot.Connection('', null, connection.id);
                }
                return new openvinoXdot.Connection(connection.id, null, null);
            }));
        });
    }

    get outputs() {
        const outputs = openvinoXdot.OperatorMetadata.operatorMetadata.getOutputs(this._type, this._outputs, this._name);
        return outputs.map((output) => {
            return new openvinoXdot.Argument(output.name, output.connections.map((connection) => {
                return new openvinoXdot.Connection(connection.id, null, null);
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

        this._inputs = _.map(inputs, (input) => {
            const candidate_edge = _.find(edges, (edge) => {
                return edge['to-layer'] === this._id && edge['to-port'] === input.id;
            });
            if (!candidate_edge) {
                return;
            }
            const parentID = candidate_edge['from-layer'];
            const parent = _.find(layers, (layer) => layer.id === parentID);
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

        this._outputs = _.map(outputs, (output) => {
            const candidate_edge = _.find(edges, (edge) => {
                return edge['from-layer'] === this._id && edge['from-port'] === output.id;
            });
            if (!candidate_edge) {
                return;
            }
            const childID = candidate_edge['to-layer'];
            const child = _.find(layers, (layer) => layer.id === childID);
            if (!child) {
                return;
            }
            return child.name;
        })
    }
}

openvinoXdot.Argument = class {
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

openvinoXdot.Connection = class {
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

openvinoXdot.Attribute = class {
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
        const meta = openvinoXdot.OperatorMetadata.operatorMetadata;
        return meta.getAttributeVisible(this._node.operator, this._name, this._value);
    }
}

openvinoXdot.Tensor = class {
    constructor({data, shape, precision}) {
        this._data = data;
        this._shape = shape;
        const dataType = precision === 'FP32' ? 'float32' : '?';
        this._type = new openvinoXdot.TensorType(dataType, this._shape);
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

openvinoXdot.TensorType = class {
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

if (typeof module !== 'undefined' && typeof module.exports === 'object') {
    module.exports.Node = openvinoXdot.Node;
}
