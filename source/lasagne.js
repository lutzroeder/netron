
// Experimental

var lasagne = lasagne || {};

lasagne.ModelFactory = class {

    match(context) {
        const obj = context.open('pkl');
        if (obj && obj.__class__ && obj.__class__.__module__ === 'nolearn.lasagne.base' && obj.__class__.__name__ == 'NeuralNet') {
            return obj;
        }
        return null;
    }

    async open(context, target) {
        const metadata = await context.metadata('lasagne-metadata.json');
        return new lasagne.Model(metadata, target);
    }
};

lasagne.Model = class {

    constructor(metadata, model) {
        this._graphs = [ new lasagne.Graph(metadata, model) ];
    }

    get format() {
        return 'Lasagne';
    }

    get graphs() {
        return this._graphs;
    }
};

lasagne.Graph = class {

    constructor(metadata, model) {
        this._nodes = [];
        this._inputs = [];
        this._outputs = [];

        const args = new Map();
        const arg = (name, type, initializer) => {
            if (!args.has(name)) {
                args.set(name, new lasagne.Value(name, type));
            }
            const value = args.get(name);
            if (!value.type && type) {
                value.type = type;
            }
            if (!value.initializer && initializer) {
                value.initializer = initializer;
            }
            return value;
        };

        for (const pair of model.layers) {
            const name = pair[0];
            const layer = model.layers_[name];
            if (layer && layer.__class__ && layer.__class__.__module__ === 'lasagne.layers.input' && layer.__class__.__name__ === 'InputLayer') {
                const type = new lasagne.TensorType(layer.input_var.type.dtype, new lasagne.TensorShape(layer.shape));
                this._inputs.push(new lasagne.Argument(layer.name, [ arg(layer.name, type) ]));
                continue;
            }
            this._nodes.push(new lasagne.Node(metadata, layer, arg));
        }

        if (model._output_layer) {
            const output_layer = model._output_layer;
            this._outputs.push(new lasagne.Argument(output_layer.name, [ arg(output_layer.name) ]));
        }
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
};

lasagne.Argument = class {

    constructor(name, value) {
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

lasagne.Value = class {

    constructor(name, type, initializer) {
        if (typeof name !== 'string') {
            throw new lasagne.Error("Invalid value identifier '" + JSON.stringify(name) + "'.");
        }
        this._name= name;
        this._type = type || null;
        this._initializer = initializer || null;
    }

    get name() {
        return this._name;
    }

    get type() {
        if (this._initializer) {
            return this._initializer.type;
        }
        return this._type;
    }

    set type(value) {
        this._type = value;
    }

    get initializer() {
        return this._initializer;
    }

    set initializer(value) {
        this._initializer = value;
    }
};

lasagne.Node = class {

    constructor(metadata, layer, arg) {
        this._name = layer.name || '';
        const type = layer.__class__ ? layer.__class__.__module__ + '.' + layer.__class__.__name__ : '';
        this._type = metadata.type(type) || { name: type };
        this._inputs = [];
        this._outputs = [];
        this._attributes = [];

        const params = new Map();
        for (const key of Object.keys(layer)) {
            if (key === 'name' || key === 'params' || key === 'input_layer' || key === 'input_shape') {
                continue;
            }
            const value = layer[key];
            if (value && value.__class__ && value.__class__.__module__ === 'theano.tensor.sharedvar' && value.__class__.__name__ === 'TensorSharedVariable') {
                params.set(value.name, key);
                continue;
            }
            this._attributes.push(new lasagne.Attribute(null, key, value));
        }

        if (layer.input_layer && layer.input_layer.name) {
            const input_layer = layer.input_layer;
            const type = layer.input_shape ? new lasagne.TensorType('?', new lasagne.TensorShape(layer.input_shape)) : undefined;
            this._inputs.push(new lasagne.Argument('input', [ arg(input_layer.name, type) ]));
        }

        if (layer.params) {
            for (const pair of layer.params) {
                const param = pair[0];
                const param_key = params.get(param.name);
                if (param_key) {
                    const initializer = new lasagne.Tensor(param.container.storage[0]);
                    this._inputs.push(new lasagne.Argument(param_key, [ arg(param.name, null, initializer) ]));
                }
            }
        }

        this._outputs.push(new lasagne.Argument('output', [ arg(this.name) ]));
    }

    get type() {
        return this._type;
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

    get attributes() {
        return this._attributes;
    }
};

lasagne.Attribute = class {

    constructor(metadata, name, value) {
        this._name = name;
        this._value = value;
        if (value && value.__class__) {
            this._type = value.__class__.__module__ + '.' + value.__class__.__name__;
        }
    }

    get name() {
        return this._name;
    }

    get value() {
        return this._value;
    }

    get type() {
        return this._type;
    }
};

lasagne.TensorType = class {

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
        return this._dataType + this._shape.toString();
    }
};

lasagne.TensorShape = class {

    constructor(dimensions) {
        this._dimensions = dimensions;
    }

    get dimensions() {
        return this._dimensions;
    }

    toString() {
        if (this._dimensions && this._dimensions.length > 0) {
            return '[' + this._dimensions.map((dimension) => dimension ? dimension.toString() : '?').join(',') + ']';
        }
        return '';
    }
};

lasagne.Tensor = class {

    constructor(storage) {
        this._type = new lasagne.TensorType(storage.dtype.__name__, new lasagne.TensorShape(storage.shape));
    }

    get type() {
        return this._type;
    }
};

lasagne.Error = class extends Error {

    constructor(message) {
        super(message);
        this.name = 'Lasagne Error';
    }
};

if (typeof module !== 'undefined' && typeof module.exports === 'object') {
    module.exports.ModelFactory = lasagne.ModelFactory;
}