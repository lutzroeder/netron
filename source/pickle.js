/* jshint esversion: 6 */

// Experimental

var pickle = pickle || {};
var python = python || require('./python');
var zip = zip || require('./zip');

pickle.ModelFactory = class {

    match(context) {
        const stream = context.stream;
        const signature = [ 0x80, undefined, 0x8a, 0x0a, 0x6c, 0xfc, 0x9c, 0x46, 0xf9, 0x20, 0x6a, 0xa8, 0x50, 0x19 ];
        if (signature.length <= stream.length && stream.peek(signature.length).every((value, index) => signature[index] === undefined || signature[index] === value)) {
            // Reject PyTorch models with .pkl file extension.
            return false;
        }
        const obj = context.open('pkl');
        if (obj !== undefined) {
            return true;
        }
        return false;
    }

    open(context) {
        return new Promise((resolve) => {
            let format = 'Pickle';
            const obj = context.open('pkl');
            if (obj === null || obj === undefined) {
                context.exception(new pickle.Error('Unknown Pickle null object.'));
            }
            else if (Array.isArray(obj)) {
                context.exception(new pickle.Error('Unknown Pickle array object.'));
            }
            else if (obj && obj.__class__) {
                const formats = new Map([
                    [ 'cuml.ensemble.randomforestclassifier.RandomForestClassifier', 'cuML' ]
                ]);
                const type = obj.__class__.__module__ + "." + obj.__class__.__name__;
                if (formats.has(type)) {
                    format = formats.get(type);
                }
                else {
                    context.exception(new pickle.Error("Unknown Pickle type '" + type + "'."));
                }
            }
            else {
                context.exception(new pickle.Error('Unknown Pickle object.'));
            }
            resolve(new pickle.Model(obj, format));
        });
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

        if (Array.isArray(obj) && obj.every((item) => item.__class__)) {
            for (const item of obj) {
                this._nodes.push(new pickle.Node(item));
            }
        }
        else if (obj && obj.__class__) {
            this._nodes.push(new pickle.Node(obj));
        }
        else if (obj && Object(obj) === obj) {
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

    constructor(obj) {
        this._type = obj.__class__ ? obj.__class__.__module__ + '.' + obj.__class__.__name__ : 'Object';
        this._inputs = [];
        this._outputs = [];
        this._attributes = [];
        for (const key of Object.keys(obj)) {
            const value = obj[key];
            this._attributes.push(new pickle.Attribute(key, value));
        }
    }

    get type() {
        return this._type;
    }

    get name() {
        return '';
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