
// Experimental

var pickle = {};

pickle.ModelFactory = class {

    match(context) {
        const stream = context.stream;
        const signature = [ 0x80, undefined, 0x8a, 0x0a, 0x6c, 0xfc, 0x9c, 0x46, 0xf9, 0x20, 0x6a, 0xa8, 0x50, 0x19 ];
        if (stream && signature.length <= stream.length && stream.peek(signature.length).every((value, index) => signature[index] === undefined || signature[index] === value)) {
            // Reject PyTorch models with .pkl file extension.
            return null;
        }
        const obj = context.open('pkl');
        if (obj !== undefined) {
            const name = obj && obj.__class__ && obj.__class__.__module__ && obj.__class__.__name__ ? obj.__class__.__module__ + '.' + obj.__class__.__name__ : '';
            if (!name.startsWith('__torch__.')) {
                return obj;
            }
        }
        return null;
    }

    async open(context, target) {
        let format = 'Pickle';
        const obj = target;
        if (obj === null || obj === undefined) {
            context.exception(new pickle.Error("Unsupported Pickle null object in '" + context.identifier + "'."));
        } else if (Array.isArray(obj)) {
            if (obj.length > 0 && obj[0] && obj.every((item) => item && item.__class__ && obj[0].__class__ && item.__class__.__module__ === obj[0].__class__.__module__ && item.__class__.__name__ === obj[0].__class__.__name__)) {
                const type = obj[0].__class__.__module__ + "." + obj[0].__class__.__name__;
                context.exception(new pickle.Error("Unsupported Pickle '" + type + "' array object in '" + context.identifier + "'."));
            } else if (obj.length > 0) {
                context.exception(new pickle.Error("Unsupported Pickle array object in '" + context.identifier + "'."));
            }
        } else if (obj && obj.__class__) {
            const formats = new Map([
                [ 'cuml.ensemble.randomforestclassifier.RandomForestClassifier', 'cuML' ]
            ]);
            const type = obj.__class__.__module__ + "." + obj.__class__.__name__;
            if (formats.has(type)) {
                format = formats.get(type);
            } else {
                context.exception(new pickle.Error("Unsupported Pickle type '" + type +  "'."));
            }
        } else {
            context.exception(new pickle.Error('Unsupported Pickle object.'));
        }
        return new pickle.Model(obj, format);
    }
};

pickle.Model = class {

    constructor(value, format) {
        this.format = format;
        this.graphs = [ new pickle.Graph(value) ];
    }
};

pickle.Graph = class {

    constructor(obj) {
        this.inputs = [];
        this.outputs = [];
        this.nodes = [];
        if (Array.isArray(obj) && (obj.every((item) => item.__class__) || (obj.every((item) => Array.isArray(item))))) {
            for (const item of obj) {
                this.nodes.push(new pickle.Node(item));
            }
        } else if (obj && obj instanceof Map && !Array.from(obj.values()).some((value) => typeof value === 'string' || typeof value === 'number')) {
            for (const entry of obj) {
                this.nodes.push(new pickle.Node(entry[1], entry[0]));
            }
        } else if (obj && obj.__class__) {
            this.nodes.push(new pickle.Node(obj));
        } else if (obj && Object(obj) === obj) {
            this.nodes.push(new pickle.Node(obj));
        }
    }
};

pickle.Node = class {

    constructor(obj, name) {
        this.name = name || '';
        this.inputs = [];
        this.outputs = [];
        this.attributes = [];
        const isArray = (obj) => {
            return obj && obj.__class__ && obj.__class__.__module__ === 'numpy' && obj.__class__.__name__ === 'ndarray';
        };
        if (Array.isArray(obj)) {
            this.type = { name: 'List' };
            const attribute = new pickle.Attribute('value', obj);
            this.attributes.push(attribute);
        } else {
            const type = obj.__class__ ? obj.__class__.__module__ + '.' + obj.__class__.__name__ : 'Object';
            this.type = { name: type };
            const entries = obj instanceof Map ? Array.from(obj.entries()) : Object.entries(obj);
            for (const entry of entries) {
                const name = entry[0];
                const value = entry[1];
                if (value && isArray(value)) {
                    const tensor = new pickle.Tensor(value);
                    const attribute = new pickle.Attribute(name, 'tensor', tensor);
                    this.attributes.push(attribute);
                } else if (Array.isArray(value) && value.length > 0 && value.every((obj) => isArray(obj))) {
                    const values = value.map((value) => new pickle.Tensor(value));
                    const attribute = new pickle.Attribute(name, 'tensor[]', values);
                    this.attributes.push(attribute);
                } else {
                    const attribute = new pickle.Attribute(name, null, value);
                    this.attributes.push(attribute);
                }

            }
        }
    }
};

pickle.Attribute = class {

    constructor(name, type, value) {
        this.name = name;
        this.type = type;
        this.value = value;
        if (!type && value && value.__class__) {
            this.type = value.__class__.__module__ + '.' + value.__class__.__name__;
        }
    }
};

pickle.Tensor = class {

    constructor(array) {
        this.type = new pickle.TensorType(array.dtype.__name__, new pickle.TensorShape(array.shape));
        this.layout = this.type.dataType == 'string' || this.type.dataType == 'object' ? '|' : array.dtype.byteorder;
        this.values = this.type.dataType == 'string' || this.type.dataType == 'object' ? array.tolist() : array.tobytes();
    }
};

pickle.TensorType = class {

    constructor(dataType, shape) {
        this.dataType = dataType;
        this.shape = shape;
    }

    toString() {
        return this.dataType + this.shape.toString();
    }
};

pickle.TensorShape = class {

    constructor(dimensions) {
        this.dimensions = dimensions;
    }

    toString() {
        return this.dimensions ? ('[' + this.dimensions.map((dimension) => dimension.toString()).join(',') + ']') : '';
    }
};

pickle.Error = class extends Error {

    constructor(message) {
        super(message);
        this.name = 'Error loading Pickle model.';
    }
};

if (typeof module !== 'undefined' && typeof module.exports === 'object') {
    module.exports.ModelFactory = pickle.ModelFactory;
}