
// Experimental

var sklearn = {};

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
        const formats = new Map([
            [ 'sklearn', 'scikit-learn' ],
            [ 'scipy', 'SciPy' ],
            [ 'hmmlearn', 'hmmlearn' ]
        ]);
        this.format = formats.get(target.split('.').shift());
        this.graphs = [];
        const version = [];
        switch (target) {
            case 'sklearn':
            case 'scipy':
            case 'hmmlearn': {
                if (obj._sklearn_version) {
                    version.push(' v' + obj._sklearn_version.toString());
                }
                this.graphs.push(new sklearn.Graph(metadata, '', obj));
                break;
            }
            case 'sklearn.list':
            case 'scipy.list': {
                const list = obj;
                for (let i = 0; i < list.length; i++) {
                    const obj = list[i];
                    this.graphs.push(new sklearn.Graph(metadata, i.toString(), obj));
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
                    this.graphs.push(new sklearn.Graph(metadata, entry[0], obj));
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
            this.format += version[0];
        }
    }
};

sklearn.Graph = class {

    constructor(metadata, name, obj) {
        this.name = name || '';
        this.nodes = [];
        this.inputs = [];
        this.outputs = [];
        this.groups = false;
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
                    this.groups = true;
                    name = name || 'pipeline';
                    const childGroup = concat(group, name);
                    for (const step of obj.steps) {
                        inputs = process(childGroup, step[0], step[1], inputs);
                    }
                    return inputs;
                }
                case 'sklearn.pipeline.FeatureUnion': {
                    this.groups = true;
                    const outputs = [];
                    name = name || 'union';
                    const output = concat(group, name);
                    const subgroup = concat(group, name);
                    this.nodes.push(new sklearn.Node(metadata, subgroup, output, obj, inputs, [ output ], value));
                    for (const transformer of obj.transformer_list) {
                        outputs.push(...process(subgroup, transformer[0], transformer[1], [ output ]));
                    }
                    return outputs;
                }
                case 'sklearn.compose._column_transformer.ColumnTransformer': {
                    this.groups = true;
                    name = name || 'transformer';
                    const output = concat(group, name);
                    const subgroup = concat(group, name);
                    const outputs = [];
                    this.nodes.push(new sklearn.Node(metadata, subgroup, output, obj, inputs, [ output ], value));
                    for (const transformer of obj.transformers) {
                        if (transformer[1] !== 'passthrough') {
                            outputs.push(...process(subgroup, transformer[0], transformer[1], [ output ]));
                        }
                    }
                    return outputs;
                }
                default: {
                    const output = concat(group, name);
                    this.nodes.push(new sklearn.Node(metadata, group, output, obj, inputs, output === '' ? [] : [ output ], value));
                    return [ output ];
                }
            }
        };
        process('', '', obj, ['data']);
    }
};

sklearn.Argument = class {

    constructor(name, value) {
        this.name = name;
        this.value = value;
    }
};

sklearn.Value = class {

    constructor(name, type, initializer) {
        if (typeof name !== 'string') {
            throw new sklearn.Error("Invalid value identifier '" + JSON.stringify(name) + "'.");
        }
        this.name = name;
        this._type = type || null;
        this.initializer = initializer || null;
    }

    get type() {
        if (this.initializer) {
            return this.initializer.type;
        }
        return this._type;
    }
};

sklearn.Node = class {

    constructor(metadata, group, name, obj, inputs, outputs, value) {
        this.group = group || null;
        this.name = name || '';
        const type = obj.__class__ ? obj.__class__.__module__ + '.' + obj.__class__.__name__ : 'Object';
        this.type = metadata.type(type) || { name: type };
        this.inputs = inputs.map((input) => new sklearn.Argument(input, [ value(input) ]));
        this.outputs = outputs.map((output) => new sklearn.Argument(output, [ value(output) ]));
        this.attributes = [];
        const isArray = (obj) => {
            return obj && obj.__class__ && obj.__class__.__module__ === 'numpy' && obj.__class__.__name__ === 'ndarray';
        };
        for (const entry of Object.entries(obj)) {
            const name = entry[0];
            const value = entry[1];
            if (value && isArray(value)) {
                const argument = new sklearn.Argument(name, [ new sklearn.Value('', null, new sklearn.Tensor(value)) ]);
                this.inputs.push(argument);
            } else if (Array.isArray(value) && value.length > 0 && value.every((obj) => isArray(obj))) {
                const argument = new sklearn.Argument(name, value.map((obj) => new sklearn.Value('', null, new sklearn.Tensor(obj))));
                this.inputs.push(argument);
            } else if (!name.startsWith('_')) {
                const attribute = new sklearn.Attribute(metadata.attribute(type, name), name, value);
                this.attributes.push(attribute);
            }
        }
    }
};

sklearn.Attribute = class {

    constructor(metadata, name, value) {
        this.name = name;
        this.value = value;
        if (metadata) {
            if (metadata.optional && this.value == null) {
                this.visible = false;
            } else if (metadata.visible === false) {
                this.visible = false;
            } else if (metadata.default !== undefined) {
                if (Array.isArray(value)) {
                    if (Array.isArray(metadata.default)) {
                        this.visible = value.length !== metadata.default || !this.value.every((item, index) => item == metadata.default[index]);
                    } else {
                        this.visible = !this.value.every((item) => item == metadata.default);
                    }
                } else {
                    this.visible = this.value !== metadata.default;
                }
            }
        }
        if (value) {
            if (Array.isArray(value) && value.length > 0 && value.every((obj) => obj.__class__ && obj.__class__.__module__ === value[0].__class__.__module__ && obj.__class__.__name__ === value[0].__class__.__name__)) {
                this.type = value[0].__class__.__module__ + '.' + value[0].__class__.__name__ + '[]';
            } else if (value.__class__) {
                this.type = value.__class__.__module__ + '.' + value.__class__.__name__;
            }
        }
    }
};

sklearn.Tensor = class {

    constructor(array) {
        this.type = new sklearn.TensorType(array.dtype.__name__, new sklearn.TensorShape(array.shape));
        this.encoding = this.type.dataType == 'string' || this.type.dataType == 'object' ? '|' : array.dtype.byteorder;
        this.values = this.type.dataType == 'string' || this.type.dataType == 'object' ? array.tolist() : array.tobytes();
    }
};

sklearn.TensorType = class {

    constructor(dataType, shape) {
        this.dataType = dataType;
        this.shape = shape;
    }

    toString() {
        return this.dataType + this.shape.toString();
    }
};

sklearn.TensorShape = class {

    constructor(dimensions) {
        this.dimensions = dimensions;
    }

    toString() {
        return this.dimensions ? ('[' + this.dimensions.map((dimension) => dimension.toString()).join(',') + ']') : '';
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