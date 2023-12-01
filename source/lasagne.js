
// Experimental

var lasagne = {};

lasagne.ModelFactory = class {

    match(context) {
        const obj = context.peek('pkl');
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
        this.format = 'Lasagne';
        this.graphs = [ new lasagne.Graph(metadata, model) ];
    }
};

lasagne.Graph = class {

    constructor(metadata, model) {
        this.nodes = [];
        this.inputs = [];
        this.outputs = [];
        const values = new Map();
        values.map = (name, type, initializer) => {
            if (!values.has(name)) {
                values.set(name, new lasagne.Value(name, type));
            }
            const value = values.get(name);
            if (!value.type && type) {
                value.type = type;
            }
            if (!value.initializer && initializer) {
                value.initializer = initializer;
            }
            return value;
        };
        for (const [name] of model.layers) {
            const layer = model.layers_[name];
            if (layer && layer.__class__ && layer.__class__.__module__ === 'lasagne.layers.input' && layer.__class__.__name__ === 'InputLayer') {
                const type = new lasagne.TensorType(layer.input_var.type.dtype, new lasagne.TensorShape(layer.shape));
                const argument = new lasagne.Argument(layer.name, [ values.map(layer.name, type) ]);
                this.inputs.push(argument);
                continue;
            }
            this.nodes.push(new lasagne.Node(metadata, layer, values));
        }
        if (model._output_layer) {
            const output_layer = model._output_layer;
            this.outputs.push(new lasagne.Argument(output_layer.name, [ values.map(output_layer.name) ]));
        }
    }
};

lasagne.Argument = class {

    constructor(name, value) {
        this.name = name;
        this.value = value;
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

    constructor(metadata, layer, values) {
        this.name = layer.name || '';
        const type = layer.__class__ ? layer.__class__.__module__ + '.' + layer.__class__.__name__ : '';
        this.type = metadata.type(type) || { name: type };
        this.inputs = [];
        this.outputs = [];
        this.attributes = [];
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
            this.attributes.push(new lasagne.Attribute(null, key, value));
        }
        if (layer.input_layer && layer.input_layer.name) {
            const input_layer = layer.input_layer;
            const type = layer.input_shape ? new lasagne.TensorType('?', new lasagne.TensorShape(layer.input_shape)) : undefined;
            const argument = new lasagne.Argument('input', [ values.map(input_layer.name, type) ]);
            this.inputs.push(argument);
        }
        if (layer.params) {
            for (const [param] of layer.params) {
                const param_key = params.get(param.name);
                if (param_key) {
                    const initializer = new lasagne.Tensor(param.container.storage[0]);
                    const argument = new lasagne.Argument(param_key, [ values.map(param.name, null, initializer) ]);
                    this.inputs.push(argument);
                }
            }
        }
        this.outputs.push(new lasagne.Argument('output', [ values.map(this.name) ]));
    }
};

lasagne.Attribute = class {

    constructor(metadata, name, value) {
        this.name = name;
        this.value = value;
        if (value && value.__class__) {
            this.type = value.__class__.__module__ + '.' + value.__class__.__name__;
        }
    }
};

lasagne.TensorType = class {

    constructor(dataType, shape) {
        this.dataType = dataType;
        this.shape = shape;
    }

    toString() {
        return this.dataType + this.shape.toString();
    }
};

lasagne.TensorShape = class {

    constructor(dimensions) {
        this.dimensions = dimensions;
    }

    toString() {
        if (this.dimensions && this.dimensions.length > 0) {
            return '[' + this.dimensions.map((dimension) => dimension ? dimension.toString() : '?').join(',') + ']';
        }
        return '';
    }
};

lasagne.Tensor = class {

    constructor(storage) {
        this.type = new lasagne.TensorType(storage.dtype.__name__, new lasagne.TensorShape(storage.shape));
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