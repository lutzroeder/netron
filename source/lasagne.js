
// Experimental

const lasagne = {};

lasagne.ModelFactory = class {

    match(context) {
        const obj = context.peek('pkl');
        if (obj && obj.__class__ && obj.__class__.__module__ === 'nolearn.lasagne.base' && obj.__class__.__name__ === 'NeuralNet') {
            context.type = 'lasagne';
            context.target = obj;
        }
    }

    async open(context) {
        const metadata = await context.metadata('lasagne-metadata.json');
        return new lasagne.Model(metadata, context.target);
    }
};

lasagne.Model = class {

    constructor(metadata, model) {
        this.format = 'Lasagne';
        this.graphs = [new lasagne.Graph(metadata, model)];
    }
};

lasagne.Graph = class {

    constructor(metadata, model) {
        this.nodes = [];
        this.inputs = [];
        this.outputs = [];
        const values = new Map();
        values.map = (name, type, tensor) => {
            if (!values.has(name)) {
                values.set(name, new lasagne.Value(name, type, tensor));
            } else if (tensor) {
                throw new lasagne.Error(`Duplicate value '${name}'.`);
            } else if (type && !type.equals(values.get(name).type)) {
                throw new lasagne.Error(`Duplicate value '${name}'.`);
            }
            return values.get(name);
        };
        for (const [name] of model.layers) {
            const layer = model.layers_[name];
            if (layer.input_layer && layer.input_layer.name) {
                const input_layer = layer.input_layer;
                const dataType = input_layer.input_var ? input_layer.input_var.type.dtype : '?';
                const shape = layer.input_shape ? new lasagne.TensorShape(layer.input_shape) : null;
                const type = shape ? new lasagne.TensorType(dataType, shape) : null;
                values.map(input_layer.name, type);
            }
        }
        for (const [name] of model.layers) {
            const layer = model.layers_[name];
            if (layer && layer.__class__ && layer.__class__.__module__ === 'lasagne.layers.input' && layer.__class__.__name__ === 'InputLayer') {
                const shape = new lasagne.TensorShape(layer.shape);
                const type = new lasagne.TensorType(layer.input_var.type.dtype, shape);
                const argument = new lasagne.Argument(layer.name, [values.map(layer.name, type)]);
                this.inputs.push(argument);
                continue;
            }
            this.nodes.push(new lasagne.Node(metadata, layer, values));
        }
        if (model._output_layer) {
            const output_layer = model._output_layer;
            this.outputs.push(new lasagne.Argument(output_layer.name, [values.map(output_layer.name)]));
        }
    }
};

lasagne.Argument = class {

    constructor(name, value, type) {
        this.name = name;
        this.value = value;
        this.type = type || null;
    }
};

lasagne.Value = class {

    constructor(name, type, initializer) {
        if (typeof name !== 'string') {
            throw new lasagne.Error(`Invalid value identifier '${JSON.stringify(name)}'.`);
        }
        this.name = name;
        this.type = !type && initializer ? initializer.type : type;
        this.initializer = initializer;
    }
};

lasagne.Node = class {

    constructor(metadata, layer, values) {
        this.name = layer.name || '';
        const type = layer.__class__ ? `${layer.__class__.__module__}.${layer.__class__.__name__}` : '';
        this.type = metadata.type(type) || { name: type };
        this.inputs = [];
        this.outputs = [];
        this.attributes = [];
        const params = new Map();
        for (const [key, value] of Object.entries(layer)) {
            if (key === 'name' || key === 'params' || key === 'input_layer' || key === 'input_shape') {
                continue;
            }
            if (value && value.__class__ && value.__class__.__module__ === 'theano.tensor.sharedvar' && value.__class__.__name__ === 'TensorSharedVariable') {
                params.set(value.name, key);
                continue;
            }
            const type = value && value.__class__ ? `${value.__class__.__module__}.${value.__class__.__name__}` : null;
            const attribute = new lasagne.Argument(key, value, type);
            this.attributes.push(attribute);
        }
        if (layer.input_layer && layer.input_layer.name) {
            const value = values.map(layer.input_layer.name);
            const argument = new lasagne.Argument('input', [value]);
            this.inputs.push(argument);
        }
        if (layer.params) {
            for (const [param] of layer.params) {
                const param_key = params.get(param.name);
                if (param_key) {
                    const initializer = new lasagne.Tensor(param.container.storage[0]);
                    const argument = new lasagne.Argument(param_key, [values.map(param.name, null, initializer)]);
                    this.inputs.push(argument);
                }
            }
        }
        this.outputs.push(new lasagne.Argument('output', [values.map(this.name)]));
    }
};

lasagne.TensorType = class {

    constructor(dataType, shape) {
        this.dataType = dataType;
        this.shape = shape;
    }

    equals(obj) {
        return obj && this.dataType === obj.dataType && this.shape && this.shape.equals(obj.shape);
    }

    toString() {
        return this.dataType + this.shape.toString();
    }
};

lasagne.TensorShape = class {

    constructor(dimensions) {
        this.dimensions = dimensions;
    }

    equals(obj) {
        return obj && Array.isArray(obj.dimensions) && Array.isArray(this.dimensions) &&
            this.dimensions.length === obj.dimensions.length &&
            obj.dimensions.every((value, index) => this.dimensions[index] === value);
    }

    toString() {
        if (this.dimensions && this.dimensions.length > 0) {
            return `[${this.dimensions.map((dimension) => dimension ? dimension.toString() : '?').join(',')}]`;
        }
        return '';
    }
};

lasagne.Tensor = class {

    constructor(storage) {
        this.type = new lasagne.TensorType(storage.dtype.__name__, new lasagne.TensorShape(storage.shape));
        this.values = storage.data;
    }
};

lasagne.Error = class extends Error {

    constructor(message) {
        super(message);
        this.name = 'Lasagne Error';
    }
};

export const ModelFactory = lasagne.ModelFactory;
