class OpenVINOIRNode {
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

        this._attributes.push(new OpenVINOIRAttribute(this, 'precision', layer.precision));

        if (layer.data) {
            this._attributes = _.map(layer.data, (value, key) => {
                return new OpenVINOIRAttribute(this, key, value);
            });
        }

        if (layer.biases) {
            const value = this._concatBinaryAttributes(layer.biases);
            this._attributes.push(new OpenVINOIRAttribute(this, 'biases', value));

            // TODO: complex to extract the size of the bias
            // TODO: compute from the overall size?
            // this._initializers.push(new OpenVINOIRTensor({data: [],
            //     shape: [layer[0].output[0].dims[1]],
            //     precision: layer.precision
            // }));
        }

        if (layer.weights) {
            const value = this._concatBinaryAttributes(layer.weights);
            this._attributes.push(new OpenVINOIRAttribute(this, 'weights', value));


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
        return OpenVINOIROperatorMetadata.operatorMetadata.getOperatorCategory(this._type);
    }

    get documentation() {
        return OpenVINOIROperatorMetadata.operatorMetadata.getOperatorDocumentation(this._type);
    }

    get inputs() {
        const list = this._inputs.concat(this._initializers);
        const inputs = OpenVINOIROperatorMetadata.operatorMetadata.getInputs(this._type, list);
        return inputs.map((input) => {
            return new OpenVINOIRArgument(input.name, input.connections.map((connection) => {
                if (connection.id instanceof CaffeTensor) {
                    return new OpenVINOIRConnection('', null, connection.id);
                }
                return new OpenVINOIRConnection(connection.id, null, null);
            }));
        });
    }

    get outputs() {
        const outputs = OpenVINOIROperatorMetadata.operatorMetadata.getOutputs(this._type, this._outputs, this._name);
        return outputs.map((output) => {
            return new OpenVINOIRArgument(output.name, output.connections.map((connection) => {
                return new OpenVINOIRConnection(connection.id, null, null);
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

        this._inputs = _.map(inputs, (input) => {
            const candidate_edge = _.find(edges, (edge) => {
                return edge['to-layer'] === this._id && edge['to-port'] === input.id;
            });
            if (!candidate_edge){
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
        if (!outputs){
            this._outputs = [];
            return;
        }

        this._outputs = _.map(outputs, (output) => {
            const candidate_edge = _.find(edges, (edge) => {
                return edge['from-layer'] === this._id && edge['from-port'] === output.id;
            });
            if (!candidate_edge){
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

class OpenVINOIRArgument {
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

class OpenVINOIRConnection {
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

class OpenVINOIRAttribute {
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
        const meta = OpenVINOIROperatorMetadata.operatorMetadata;
        return meta.getAttributeVisible(this._node.operator, this._name, this._value);
    }
}

class OpenVINOIRTensor {
    constructor({data, shape, precision}) {
        this._data = data;
        this._shape = shape;
        const dataType = precision === 'FP32' ? 'float32' : '?';
        this._type = new OpenVINOIRTensorType(dataType, this._shape);
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

class OpenVINOIRTensorType {
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
