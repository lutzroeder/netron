
var lightgbm = {};
var python = require('./python');

lightgbm.ModelFactory = class {

    match(context) {
        const stream = context.stream;
        const signature = [ 0x74, 0x72, 0x65, 0x65, 0x0A ];
        if (stream && stream.length >= signature.length && stream.peek(signature.length).every((value, index) => value === signature[index])) {
            return 'lightgbm.text';
        }
        const obj = context.open('pkl');
        if (obj && obj.__class__ && obj.__class__.__module__ && obj.__class__.__module__.startsWith('lightgbm.')) {
            return 'lightgbm.pickle';
        }
        return null;
    }

    async open(context, target) {
        let obj;
        let format;
        switch (target) {
            case 'lightgbm.pickle': {
                obj = context.open('pkl');
                format = 'LightGBM Pickle';
                break;
            }
            case 'lightgbm.text': {
                const stream = context.stream;
                const buffer = stream.peek();
                const decoder = new TextDecoder('utf-8');
                const model_str = decoder.decode(buffer);
                const execution = new python.Execution();
                obj = execution.invoke('lightgbm.basic.Booster', []);
                obj.LoadModelFromString(model_str);
                format = 'LightGBM';
                break;
            }
            default: {
                throw new lightgbm.Error("Unsupported LightGBM format '" + target + "'.");
            }
        }
        return new lightgbm.Model(obj, format);
    }
};

lightgbm.Model = class {

    constructor(obj, format) {
        this._format = format + (obj && obj.version ? ' ' + obj.version : '');
        this._graphs = [ new lightgbm.Graph(obj) ];
    }

    get format() {
        return this._format;
    }

    get graphs() {
        return this._graphs;
    }
};

lightgbm.Graph = class {

    constructor(model) {
        this._inputs = [];
        this._outputs = [];
        this._nodes = [];

        const args = [];
        const feature_names = model.feature_names || [];
        for (let i = 0; i < feature_names.length; i++) {
            const name = feature_names[i];
            const info = model.feature_infos && i < model.feature_infos.length ? model.feature_infos[i] : null;
            const value = new lightgbm.Value(name, info);
            args.push(value);
            if (feature_names.length < 1000) {
                this._inputs.push(new lightgbm.Argument(name, [ value ]));
            }
        }
        this._nodes.push(new lightgbm.Node(model, args));
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

lightgbm.Argument = class {

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

lightgbm.Value = class {

    constructor(name, quantization) {
        if (typeof name !== 'string') {
            throw new lightgbm.Error("Invalid value identifier '" + JSON.stringify(name) + "'.");
        }
        this._name = name;
        this._quantization = quantization;
    }

    get name() {
        return this._name;
    }

    get type() {
        return null;
    }

    get quantization() {
        return this._quantization;
    }

    get initializer() {
        return null;
    }
};

lightgbm.Node = class {

    constructor(model, args) {
        const type = model.__class__.__module__ + '.' + model.__class__.__name__;
        this._type = { name: type };
        this._inputs = [];
        this._outputs = [];
        this._attributes = [];
        this._inputs.push(new lightgbm.Argument('features', args));
        for (const entry of Object.entries(model)) {
            const key = entry[0];
            const value = entry[1];
            if (value === undefined) {
                continue;
            }
            switch (key) {
                case 'tree':
                case 'version':
                case 'feature_names':
                case 'feature_infos':
                    break;
                default:
                    this._attributes.push(new lightgbm.Attribute(key, value));
            }
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

lightgbm.Attribute = class {

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

lightgbm.Error = class extends Error {

    constructor(message) {
        super(message);
        this.name = 'Error loading LightGBM model.';
    }
};

if (typeof module !== 'undefined' && typeof module.exports === 'object') {
    module.exports.ModelFactory = lightgbm.ModelFactory;
}
