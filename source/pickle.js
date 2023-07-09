
// Experimental

var pickle = pickle || {};

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
        this._format = format;
        this._graphs = [ new pickle.Graph(value) ];
    }

    get format() {
        return this._format;
    }

    get graphs() {
        return this._graphs;
    }
};

pickle.Graph = class {

    constructor(obj) {
        this._inputs = [];
        this._outputs = [];
        this._nodes = [];

        if (Array.isArray(obj) && (obj.every((item) => item.__class__) || (obj.every((item) => Array.isArray(item))))) {
            for (const item of obj) {
                this._nodes.push(new pickle.Node(item));
            }
        } else if (obj && obj instanceof Map && !Array.from(obj.values()).some((value) => typeof value === 'string' || typeof value === 'number')) {
            for (const entry of obj) {
                this._nodes.push(new pickle.Node(entry[1], entry[0]));
            }
        } else if (obj && obj.__class__) {
            this._nodes.push(new pickle.Node(obj));
        } else if (obj && Object(obj) === obj) {
            this._nodes.push(new pickle.Node(obj));
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

pickle.Node = class {

    constructor(obj, name) {
        this._name = name || '';
        this._inputs = [];
        this._outputs = [];
        this._attributes = [];
        if (Array.isArray(obj)) {
            this._type = { name: 'List' };
            this._attributes.push(new pickle.Attribute('value', obj));
        } else {
            const type = obj.__class__ ? obj.__class__.__module__ + '.' + obj.__class__.__name__ : 'Object';
            this._type = { name: type };
            const entries = obj instanceof Map ? Array.from(obj.entries()) : Object.entries(obj);
            for (const entry of entries) {
                const name = entry[0];
                const value = entry[1];
                this._attributes.push(new pickle.Attribute(name, value));
            }
        }
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

pickle.Attribute = class {

    constructor(name, value) {
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

pickle.Error = class extends Error {

    constructor(message) {
        super(message);
        this.name = 'Error loading Pickle model.';
    }
};

if (typeof module !== 'undefined' && typeof module.exports === 'object') {
    module.exports.ModelFactory = pickle.ModelFactory;
}