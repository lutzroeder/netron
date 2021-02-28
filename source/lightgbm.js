/* jshint esversion: 6 */

var lightgbm = lightgbm || {};
var base = base || require('./base');

lightgbm.ModelFactory = class {

    match(context) {
        try {
            const stream = context.stream;
            const reader = base.TextReader.create(stream.peek(), 65536);
            const line = reader.read();
            if (line === 'tree') {
                return true;
            }
        }
        catch (err) {
            // continue regardless of error
        }
        const obj = context.open('pkl');
        if (obj && obj.__class__ && obj.__class__.__module__ && obj.__class__.__module__.startsWith('lightgbm.')) {
            return true;
        }
        return false;
    }

    open(context) {
        return new Promise((resolve, reject) => {
            try {
                let model;
                let format;
                const obj = context.open('pkl');
                if (obj) {
                    format = 'LightGBM Pickle';
                    model = obj;
                    if (model && model.handle && typeof model.handle === 'string') {
                        const reader = base.TextReader.create(model.handle);
                        model = new lightgbm.basic.Booster(reader);
                    }
                }
                else {
                    format = 'LightGBM';
                    const reader = base.TextReader.create(context.stream.peek());
                    model = new lightgbm.basic.Booster(reader);
                }
                resolve(new lightgbm.Model(model, format));
            }
            catch (err) {
                reject(err);
            }
        });
    }
};

lightgbm.Model = class {

    constructor(model, format) {
        this._format = format + (model.meta && model.meta.version ? ' ' + model.meta.version : '');
        this._graphs = [ new lightgbm.Graph(model) ];
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
        if (model.meta && model.meta.feature_names) {
            const feature_names = model.meta.feature_names.split(' ').map((item) => item.trim());
            for (const feature_name of feature_names) {
                const arg = new lightgbm.Argument(feature_name);
                args.push(arg);
                this._inputs.push(new lightgbm.Parameter(feature_name, [ arg ]));
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

lightgbm.Parameter = class {

    constructor(name, args) {
        this._name = name;
        this._arguments = args;
    }

    get name() {
        return this._name;
    }

    get visible() {
        return true;
    }

    get arguments() {
        return this._arguments;
    }
};

lightgbm.Argument = class {

    constructor(name) {
        if (typeof name !== 'string') {
            throw new lightgbm.Error("Invalid argument identifier '" + JSON.stringify(name) + "'.");
        }
        this._name = name;
    }

    get name() {
        return this._name;
    }

    get type() {
        return null;
    }

    get initializer() {
        return null;
    }
};

lightgbm.Node = class {

    constructor(model, args) {
        this._type = model.__class__.__module__ + '.' + model.__class__.__name__;
        this._inputs = [];
        this._outputs = [];
        this._attributes = [];

        this._inputs.push(new lightgbm.Parameter('features', args));

        for (const key of Object.keys(model.params)) {
            this._attributes.push(new lightgbm.Attribute(key, model.params[key]));
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

lightgbm.basic = {};

lightgbm.basic.Booster = class {

    constructor(reader) {

        this.__class__ = {
            __module__: 'lightgbm.basic',
            __name__: 'Booster'
        };

        this.params = {};
        this.feature_importances = {};
        this.meta = {};
        this.trees = [];

        // GBDT::LoadModelFromString() in https://github.com/microsoft/LightGBM/blob/master/src/boosting/gbdt_model_text.cpp
        const signature = reader.read();
        if (!signature || signature.trim() !== 'tree') {
            throw new lightgbm.Error("Invalid signature '" + signature.trim() + "'.");
        }
        let state = '';
        let tree = null;
        // let lineNumber = 0;
        for (;;) {
            // lineNumber++;
            const text = reader.read();
            if (text === undefined) {
                break;
            }
            const line = text.trim();
            if (line.length === 0) {
                continue;
            }
            if (line.startsWith('Tree=')) {
                state = 'tree';
                tree = { index: parseInt(line.split('=').pop(), 10) };
                this.trees.push(tree);
                continue;
            }
            else if (line === 'parameters:') {
                state = 'param';
                continue;
            }
            else if (line === 'feature_importances:' || line === 'feature importances:') {
                state = 'feature_importances';
                continue;
            }
            else if (line === 'end of trees' || line === 'end of parameters') {
                state = '';
                continue;
            }
            else if (line.startsWith('pandas_categorical:')) {
                state = 'pandas_categorical';
                continue;
            }
            switch (state) {
                case '': {
                    const param = line.split('=');
                    if (param.length !== 2) {
                        throw new lightgbm.Error("Invalid property '" + line + "'.");
                    }
                    const name = param[0].trim();
                    const value = param[1].trim();
                    this.meta[name] = value;
                    break;
                }
                case 'param': {
                    if (!line.startsWith('[') || !line.endsWith(']')) {
                        throw new lightgbm.Error("Invalid parameter '" + line + "'.");
                    }
                    const param = line.substring(1, line.length - 2).split(':');
                    if (param.length !== 2) {
                        throw new lightgbm.Error("Invalid param '" + line + "'.");
                    }
                    const name = param[0].trim();
                    const value = param[1].trim();
                    this.params[name] = value;
                    break;
                }
                case 'tree': {
                    const param = line.split('=');
                    if (param.length !== 2) {
                        throw new lightgbm.Error("Invalid property '" + line + "'.");
                    }
                    const name = param[0].trim();
                    const value = param[1].trim();
                    tree[name] = value;
                    break;
                }
                case 'feature_importances': {
                    const param = line.split('=');
                    if (param.length !== 2) {
                        throw new lightgbm.Error("Invalid feature importance '" + line + "'.");
                    }
                    const name = param[0].trim();
                    const value = param[1].trim();
                    this.feature_importances[name] = value;
                    break;
                }
                case 'pandas_categorical': {
                    break;
                }
            }
        }
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
