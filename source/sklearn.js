
// Experimental

var sklearn = sklearn || {};

sklearn.ModelFactory = class {

    match(context) {
        const obj = context.open('pkl');
        const validate = (obj, name) => {
            if (obj && obj.__class__ && obj.__class__.__module__ && obj.__class__.__name__) {
                const key = obj.__class__.__module__ + '.' + obj.__class__.__name__;
                return key.startsWith(name);
            }
            return false;
        };
        const formats = [
            { name: 'sklearn.', format: 'sklearn' },
            { name: 'xgboost.sklearn.', format: 'sklearn' },
            { name: 'lightgbm.sklearn.', format: 'sklearn' },
            { name: 'scipy.', format: 'scipy' },
            { name: 'hmmlearn.', format: 'hmmlearn' }
        ];
        for (const format of formats) {
            if (validate(obj, format.name)) {
                return format.format;
            }
            if (Array.isArray(obj) && obj.length > 0 && obj.every((item) => validate(item, format.name))) {
                return format.format + '.list';
            }
            if (Object(obj) === obj) {
                const entries = Object.entries(obj);
                if (entries.length > 0 && entries.every((entry) => validate(entry[1], format.name))) {
                    return format.format + '.map';
                }
            }
        }
        return null;
    }

    async open(context, target) {
        const metadata = await context.metadata('sklearn-metadata.json');
        const obj = context.open('pkl');
        return new sklearn.Model(metadata, target, obj);
    }
};

sklearn.Model = class {

    constructor(metadata, target, obj) {
        const formats = new Map([ [ 'sklearn', 'scikit-learn' ], [ 'scipy', 'SciPy' ], [ 'hmmlearn', 'hmmlearn' ] ]);
        this._format = formats.get(target.split('.').shift());
        this._graphs = [];
        const version = [];
        switch (target) {
            case 'sklearn':
            case 'scipy':
            case 'hmmlearn': {
                if (obj._sklearn_version) {
                    version.push(' v' + obj._sklearn_version.toString());
                }
                this._graphs.push(new sklearn.Graph(metadata, '', obj));
                break;
            }
            case 'sklearn.list':
            case 'scipy.list': {
                const list = obj;
                for (let i = 0; i < list.length; i++) {
                    const obj = list[i];
                    this._graphs.push(new sklearn.Graph(metadata, i.toString(), obj));
                    if (obj._sklearn_version) {
                        version.push(' v' + obj._sklearn_version.toString());
                    }
                }
                break;
            }
            case 'sklearn.map':
            case 'scipy.map': {
                for (const entry of Object.entries(obj)) {
                    const obj = entry[1];
                    this._graphs.push(new sklearn.Graph(metadata, entry[0], obj));
                    if (obj._sklearn_version) {
                        version.push(' v' + obj._sklearn_version.toString());
                    }
                }
                break;
            }
            default: {
                throw new sklearn.Error("Unsupported scikit-learn format '" + target + "'.");
            }
        }
        if (version.length > 0 && version.every((value) => value === version[0])) {
            this._format += version[0];
        }
    }

    get format() {
        return this._format;
    }

    get graphs() {
        return this._graphs;
    }
};

sklearn.Graph = class {

    constructor(metadata, name, obj) {
        this._name = name || '';
        this._metadata = metadata;
        this._nodes = [];
        this._groups = false;
        const values = new Map();
        const value = (name) => {
            if (!values.has(name)) {
                values.set(name, new sklearn.Value(name, null, null));
            }
            return values.get(name);
        };
        const concat = (parent, name) => {
            return (parent === '' ?  name : `${parent}/${name}`);
        };
        const process = (group, name, obj, inputs) => {
            const type = obj.__class__.__module__ + '.' + obj.__class__.__name__;
            switch (type) {
                case 'sklearn.pipeline.Pipeline': {
                    this._groups = true;
                    name = name || 'pipeline';
                    const childGroup = concat(group, name);
                    for (const step of obj.steps) {
                        inputs = process(childGroup, step[0], step[1], inputs);
                    }
                    return inputs;
                }
                case 'sklearn.pipeline.FeatureUnion': {
                    this._groups = true;
                    const outputs = [];
                    name = name || 'union';
                    const output = concat(group, name);
                    const subgroup = concat(group, name);
                    this._nodes.push(new sklearn.Node(this._metadata, subgroup, output, obj, inputs, [ output ], value));
                    for (const transformer of obj.transformer_list) {
                        outputs.push(...process(subgroup, transformer[0], transformer[1], [ output ]));
                    }
                    return outputs;
                }
                case 'sklearn.compose._column_transformer.ColumnTransformer': {
                    this._groups = true;
                    name = name || 'transformer';
                    const output = concat(group, name);
                    const subgroup = concat(group, name);
                    const outputs = [];
                    this._nodes.push(new sklearn.Node(this._metadata, subgroup, output, obj, inputs, [ output ], value));
                    for (const transformer of obj.transformers) {
                        if (transformer[1] !== 'passthrough') {
                            outputs.push(...process(subgroup, transformer[0], transformer[1], [ output ]));
                        }
                    }
                    return outputs;
                }
                default: {
                    const output = concat(group, name);
                    this._nodes.push(new sklearn.Node(this._metadata, group, output, obj, inputs, output === '' ? [] : [ output ], value));
                    return [ output ];
                }
            }
        };
        process('', '', obj, ['data']);
    }

    get name() {
        return this._name;
    }

    get groups() {
        return this._groups;
    }

    get inputs() {
        return [];
    }

    get outputs() {
        return [];
    }

    get nodes() {
        return this._nodes;
    }
};

sklearn.Argument = class {

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

sklearn.Value = class {

    constructor(name, type, initializer) {
        if (typeof name !== 'string') {
            throw new sklearn.Error("Invalid value identifier '" + JSON.stringify(name) + "'.");
        }
        this._name = name;
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

sklearn.Node = class {

    constructor(metadata, group, name, obj, inputs, outputs, value) {
        this._group = group || '';
        this._name = name || '';
        const type = obj.__class__ ? obj.__class__.__module__ + '.' + obj.__class__.__name__ : 'Object';
        this._type = metadata.type(type) || { name: type };
        this._inputs = inputs.map((input) => new sklearn.Argument(input, [ value(input) ]));
        this._outputs = outputs.map((output) => new sklearn.Argument(output, [ value(output) ]));
        this._attributes = [];

        for (const entry of Object.entries(obj)) {
            const name = entry[0];
            const value = entry[1];
            if (value && sklearn.Utility.isTensor(value)) {
                const argument = new sklearn.Argument(name, [ new sklearn.Value('', null, new sklearn.Tensor(value)) ]);
                this._inputs.push(argument);
            } else if (Array.isArray(value) && value.every((obj) => sklearn.Utility.isTensor(obj))) {
                const argument = new sklearn.Argument(name, value.map((obj) => new sklearn.Value('', null, new sklearn.Tensor(obj))));
                this._inputs.push(argument);
            } else if (!name.startsWith('_')) {
                const attribute = new sklearn.Attribute(metadata.attribute(type, name), name, value);
                this._attributes.push(attribute);
            }
        }
    }

    get type() {
        return this._type; // .split('.').pop();
    }

    get name() {
        return this._name;
    }

    get group() {
        return this._group ? this._group : null;
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

sklearn.Attribute = class {

    constructor(metadata, name, value) {
        this._name = name;
        this._value = value;
        if (metadata) {
            if (metadata.optional && this._value == null) {
                this._visible = false;
            } else if (metadata.visible === false) {
                this._visible = false;
            } else if (metadata.default !== undefined) {
                if (Array.isArray(value)) {
                    if (Array.isArray(metadata.default)) {
                        this._visible = value.length !== metadata.default || !this.value.every((item, index) => item == metadata.default[index]);
                    } else {
                        this._visible = !this.value.every((item) => item == metadata.default);
                    }
                } else {
                    this._visible = this.value !== metadata.default;
                }
            }
        }
        if (value) {
            if (Array.isArray(value) && value.length > 0 && value.every((obj) => obj.__class__ && obj.__class__.__module__ === value[0].__class__.__module__ && obj.__class__.__name__ === value[0].__class__.__name__)) {
                this._type = value[0].__class__.__module__ + '.' + value[0].__class__.__name__ + '[]';
            } else if (value.__class__) {
                this._type = value.__class__.__module__ + '.' + value.__class__.__name__;
            }
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

    get visible() {
        return this._visible == false ? false : true;
    }
};

sklearn.Tensor = class {

    constructor(array) {
        if (!sklearn.Utility.isTensor(array)) {
            const type = array.__class__.__module__ + '.' + array.__class__.__name__;
            throw new sklearn.Error("Unsupported tensor type '" + type + "'.");
        }
        this._type = new sklearn.TensorType(array.dtype.__name__, new sklearn.TensorShape(array.shape));
        this._byteorder = array.dtype.byteorder;
        this._data = this._type.dataType == 'string' || this._type.dataType == 'object' ? array.tolist() : array.tobytes();
    }

    get type() {
        return this._type;
    }

    get category() {
        return 'NumPy Array';
    }

    get layout() {
        return this._type.dataType == 'string' || this._type.dataType == 'object' ? '|' : this._byteorder;
    }

    get values() {
        return this._data;
    }
};

sklearn.TensorType = class {

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
        return this.dataType + this._shape.toString();
    }
};

sklearn.TensorShape = class {

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

sklearn.Utility = class {

    static isTensor(obj) {
        return obj && obj.__class__ && obj.__class__.__module__ === 'numpy' && obj.__class__.__name__ === 'ndarray';
    }

    static findWeights(obj) {
        const keys = [ '', 'blobs' ];
        for (const key of keys) {
            const dict = key === '' ? obj : obj[key];
            if (dict) {
                const weights = new Map();
                if (dict instanceof Map) {
                    for (const pair of dict) {
                        if (!sklearn.Utility.isTensor(pair[1])) {
                            return null;
                        }
                        weights.set(pair[0], pair[1]);
                    }
                    return weights;
                } else if (!Array.isArray(dict)) {
                    for (const key in dict) {
                        const value = dict[key];
                        if (key != 'weight_order' && key != 'lr') {
                            if (!key || !sklearn.Utility.isTensor(value)) {
                                return null;
                            }
                            weights.set(key, value);
                        }
                    }
                    return weights;
                }
            }
        }
        for (const key of keys) {
            const list = key === '' ? obj : obj[key];
            if (list && Array.isArray(list)) {
                const weights = new Map();
                for (let i = 0; i < list.length; i++) {
                    const value = list[i];
                    if (!sklearn.Utility.isTensor(value, 'numpy.ndarray')) {
                        return null;
                    }
                    weights.set(i.toString(), value);
                }
                return weights;
            }
        }
        return null;
    }
};

sklearn.Error = class extends Error {

    constructor(message) {
        super(message);
        this.name = 'Error loading scikit-learn model.';
    }
};

if (typeof module !== 'undefined' && typeof module.exports === 'object') {
    module.exports.ModelFactory = sklearn.ModelFactory;
}