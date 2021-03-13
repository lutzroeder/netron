/* jshint esversion: 6 */

// Experimental

var lasagne = lasagne || {};

lasagne.ModelFactory = class {

    match(context) {
        const obj = context.open('pkl');
        if (obj && obj.__class__ && obj.__class__.__module__ === 'nolearn.lasagne.base' && obj.__class__.__name__ == 'NeuralNet') {
            return true;
        }
        return false;
    }

    open(context) {
        return lasagne.Metadata.open(context).then((metadata) => {
            const obj = context.open('pkl');
            return new lasagne.Model(metadata, obj);
        });
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
                args.set(name, new lasagne.Argument(name, type));
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
                this._inputs.push(new lasagne.Parameter(layer.name, [ arg(layer.name, type) ]));
                continue;
            }
            this._nodes.push(new lasagne.Node(metadata, layer, arg));
        }

        if (model._output_layer) {
            const output_layer = model._output_layer;
            this._outputs.push(new lasagne.Parameter(output_layer.name, [ arg(output_layer.name) ]));
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

lasagne.Parameter = class {

    constructor(name, args) {
        this._name = name;
        this._arguments = args;
    }

    get name() {
        return this._name;
    }

    get arguments() {
        return this._arguments;
    }

    get visible() {
        return true;
    }
};

lasagne.Argument = class {

    constructor(name, type, initializer) {
        if (typeof name !== 'string') {
            throw new lasagne.Error("Invalid argument identifier '" + JSON.stringify(name) + "'.");
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
        this._type = layer.__class__ ? layer.__class__.__module__ + '.' + layer.__class__.__name__ : '';
        this._metadata = metadata.type(this._type);
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
            this._inputs.push(new lasagne.Parameter('input', [ arg(input_layer.name, type) ]));
        }

        if (layer.params) {
            for (const pair of layer.params) {
                const param = pair[0];
                const param_key = params.get(param.name);
                if (param_key) {
                    const initializer = new lasagne.Tensor(param.container.storage[0]);
                    this._inputs.push(new lasagne.Parameter(param_key, [ arg(param.name, null, initializer) ]));
                }
            }
        }

        this._outputs.push(new lasagne.Parameter('output', [ arg(this.name) ]));
    }

    get type() {
        return this._type;
    }

    get name() {
        return this._name;
    }

    get metadata() {
        return this._metadata;
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
        if (value && value.__class_) {
            this._type = value.__class_.__module__ + '.' + value.__class_.__name__;
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

lasagne.Metadata = class {

    static open(context) {
        if (lasagne.Metadata._metadata) {
            return Promise.resolve(lasagne.Metadata._metadata);
        }
        return context.request('lasagne-metadata.json', 'utf-8', null).then((data) => {
            lasagne.Metadata._metadata = new lasagne.Metadata(data);
            return lasagne.Metadata._metadata;
        }).catch(() => {
            lasagne.Metadata._metadata = new lasagne.Metadata(null);
            return lasagne.Metadata._metadata;
        });
    }

    constructor(data) {
        this._map = new Map();
        if (data) {
            const metadata = JSON.parse(data);
            this._map = new Map(metadata.map((item) => [ item.name, item ]));
        }
    }

    type(name) {
        return this._map.get(name);
    }

    attribute(type, name) {
        const schema = this.type(type);
        if (schema) {
            let attributeMap = schema.attributeMap;
            if (!attributeMap) {
                attributeMap = {};
                if (schema.attributes) {
                    for (const attribute of schema.attributes) {
                        attributeMap[attribute.name] = attribute;
                    }
                }
                schema.attributeMap = attributeMap;
            }
            const attributeSchema = attributeMap[name];
            if (attributeSchema) {
                return attributeSchema;
            }
        }
        return null;
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
        this._type = new lasagne.TensorType(storage.dtype.name, new lasagne.TensorShape(storage.shape));
    }

    get type() {
        return this._type;
    }

    get state() {
        return 'Not implemented.';
    }

    toString() {
        return '';
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