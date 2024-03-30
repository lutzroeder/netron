
const hickle = {};

hickle.ModelFactory = class {

    match(context) {
        const group = context.peek('hdf5');
        if (group && group.attributes && group.attributes.get('CLASS') === 'hickle') {
            context.type = 'hickle';
            context.target = group;
        }
    }

    async open(context) {
        return new hickle.Model(context.target);
    }
};

hickle.Model = class {

    constructor(group) {
        this.format = 'Hickle Weights';
        this.graphs = [new hickle.Graph(group)];
    }
};

hickle.Graph = class {

    constructor(group) {
        this.inputs = [];
        this.outputs = [];
        const deserialize = (group) => {
            if (group && group.attributes.has('type')) {
                const type = group.attributes.get('type');
                if (Array.isArray(type) && type.length && typeof type[0] === 'string') {
                    switch (type[0]) {
                        case 'hickle':
                        case 'dict_item': {
                            if (group.groups.size === 1) {
                                return deserialize(group.groups.values().next().value);
                            }
                            throw new hickle.Error(`Invalid Hickle type value '${type[0]}'.`);
                        }
                        case 'dict': {
                            const dict = new Map();
                            for (const [name, obj] of group.groups) {
                                const value = deserialize(obj);
                                dict.set(name, value);
                            }
                            return dict;
                        }
                        case 'ndarray': {
                            return group.value;
                        }
                        default: {
                            throw new hickle.Error(`Unsupported Hickle type '${type[0]}'`);
                        }
                    }
                }
                throw new hickle.Error(`Unsupported Hickle type '${JSON.stringify(type)}'`);
            }
            throw new hickle.Error('Unsupported Hickle group.');
        };
        const obj = deserialize(group);
        const layers = new Map();
        if (obj && obj instanceof Map && Array.from(obj.values()).every((value) => value.type && value.shape)) {
            for (const [key, value] of obj) {
                const tensor = new hickle.Tensor(key, value.shape, value.type, value.littleEndian, value.type === 'string' ? value.value : value.data);
                const bits = key.split('.');
                const parameter = bits.pop();
                const layer = bits.join('.');
                if (!layers.has(layer)) {
                    layers.set(layer, []);
                }
                layers.get(layer).push({ name: parameter, value: tensor });
            }
        }
        this.nodes = Array.from(layers).map(([name, value]) => new hickle.Node(name, value));
    }
};

hickle.Argument = class {

    constructor(name, value) {
        this.name = name;
        this.value = value;
    }
};

hickle.Value = class {

    constructor(name, type, initializer) {
        if (typeof name !== 'string') {
            throw new hickle.Error(`Invalid value identifier '${JSON.stringify(name)}'.`);
        }
        this.name = name;
        this.type = !type && initializer ? initializer.type : type;
        this.initializer = initializer || null;
    }
};

hickle.Node = class {

    constructor(name, parameters) {
        this.type = { name: 'Weights' };
        this.name = name;
        this.inputs = parameters.map((parameter) => {
            return new hickle.Argument(parameter.name, [
                new hickle.Value(parameter.value.name, null, parameter.value)
            ]);
        });
        this.outputs = [];
        this.attributes = [];
    }
};

hickle.Tensor = class {

    constructor(name, shape, type, littleEndian, data) {
        this.name = name;
        this.type = new hickle.TensorType(type, new hickle.TensorShape(shape));
        this.encoding = littleEndian ? '<' : '>';
        this._data = data;
    }

    get values() {
        if (Array.isArray(this._data) || this._data === null) {
            return null;
        }
        if (this._data instanceof Uint8Array) {
            return this._data;
        }
        return this._data.peek();
    }
};

hickle.TensorType = class {

    constructor(dataType, shape) {
        this.dataType = dataType;
        this.shape = shape;
    }

    toString() {
        return this.dataType + this.shape.toString();
    }
};

hickle.TensorShape = class {

    constructor(dimensions) {
        this.dimensions = dimensions;
    }

    toString() {
        return this.dimensions ? (`[${this.dimensions.map((dimension) => dimension.toString()).join(',')}]`) : '';
    }
};

hickle.Error = class extends Error {

    constructor(message) {
        super(message);
        this.name = 'Error loading Hickle model.';
    }
};

export const ModelFactory = hickle.ModelFactory;
