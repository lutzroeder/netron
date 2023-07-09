
var hickle = hickle || {};

hickle.ModelFactory = class {

    match(context) {
        const group = context.open('hdf5');
        if (group && group.attributes.get('CLASS') === 'hickle') {
            return group;
        }
        return null;
    }

    async open(context, target) {
        return new hickle.Model(target);
    }
};

hickle.Model = class {

    constructor(group) {
        this._graphs = [ new hickle.Graph(group) ];
    }

    get format() {
        return 'Hickle Weights';
    }

    get graphs() {
        return this._graphs;
    }
};

hickle.Graph = class {

    constructor(group) {
        this._inputs = [];
        this._outputs = [];
        const deserialize = (group) => {
            if (group && group.attributes.has('type')) {
                const type = group.attributes.get('type');
                if (Array.isArray(type) && type.length && typeof type[0] === 'string') {
                    switch (type[0]) {
                        case 'hickle':
                        case 'dict_item': {
                            if (group.groups.size == 1) {
                                return deserialize(group.groups.values().next().value);
                            }
                            throw new hickle.Error("Invalid Hickle type value '" + type[0] + "'.");
                        }
                        case 'dict': {
                            const dict = new Map();
                            for (const entry of group.groups) {
                                const name = entry[0];
                                const value = deserialize(entry[1]);
                                dict.set(name, value);
                            }
                            return dict;
                        }
                        case 'ndarray': {
                            return group.value;
                        }
                        default: {
                            throw new hickle.Error("Unsupported Hickle type '" + type[0] + "'");
                        }
                    }
                }
                throw new hickle.Error("Unsupported Hickle type '" + JSON.stringify(type) + "'");
            }
            throw new hickle.Error('Unsupported Hickle group.');
        };
        const obj = deserialize(group);
        const layers = new Map();
        if (obj && obj instanceof Map && Array.from(obj.values()).every((value) => value.type && value.shape)) {
            for (const entry of obj) {
                const key = entry[0];
                const value = entry[1];
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
        this._nodes = Array.from(layers).map((entry) => new hickle.Node(entry[0], entry[1]));
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

hickle.Argument = class {

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

hickle.Value = class {

    constructor(name, type, initializer) {
        if (typeof name !== 'string') {
            throw new hickle.Error("Invalid value identifier '" + JSON.stringify(name) + "'.");
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

    get initializer() {
        return this._initializer;
    }
};

hickle.Node = class {

    constructor(name, parameters) {
        this._type = { name: 'Weights' };
        this._name = name;
        this._inputs = parameters.map((parameter) => {
            return new hickle.Argument(parameter.name, [
                new hickle.Value(parameter.value.name, null, parameter.value)
            ]);
        });
        this._outputs = [];
        this._attributes = [];
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

hickle.Tensor = class {

    constructor(name, shape, type, littleEndian, data) {
        this._name = name;
        this._type = new hickle.TensorType(type, new hickle.TensorShape(shape));
        this._littleEndian = littleEndian;
        this._data = data;
    }

    get name() {
        return this._name;
    }

    get type() {
        return this._type;
    }

    get layout() {
        return this._littleEndian ? '<' : '>';
    }

    get quantization() {
        if (this._quantization && (this._quantization.scale !== 0 || this._quantization.min !== 0)) {
            const scale = this._quantization.scale || 0;
            const min = this._quantization.min || 0;
            return scale.toString() + ' * ' + (min == 0 ? 'q' : ('(q - ' + min.toString() + ')'));
        }
        return null;
    }

    get values() {
        if (Array.isArray(this._data) || this._data === null) {
            return null;
        }
        return this._data instanceof Uint8Array ? this._data : this._data.peek();
    }
};

hickle.TensorType = class {

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

hickle.TensorShape = class {

    constructor(dimensions) {
        this._dimensions = dimensions;
    }

    get dimensions() {
        return this._dimensions;
    }

    toString() {
        return this._dimensions ? ('[' + this._dimensions.map((dimension) => dimension.toString()).join(',') + ']') : '';
    }
};

hickle.Error = class extends Error {

    constructor(message) {
        super(message);
        this.name = 'Error loading Hickle model.';
    }
};

if (typeof module !== 'undefined' && typeof module.exports === 'object') {
    module.exports.ModelFactory = hickle.ModelFactory;
}