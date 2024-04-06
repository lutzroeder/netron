
import * as python from './python.js';

const lightgbm = {};

lightgbm.ModelFactory = class {

    match(context) {
        const stream = context.stream;
        const signature = [0x74, 0x72, 0x65, 0x65, 0x0A];
        if (stream && stream.length >= signature.length && stream.peek(signature.length).every((value, index) => value === signature[index])) {
            context.type = 'lightgbm.text';
        } else {
            const obj = context.peek('pkl');
            if (obj && obj.__class__ && obj.__class__.__module__ && obj.__class__.__module__.startsWith('lightgbm.')) {
                context.type = 'lightgbm.pickle';
                context.target = obj;
            }
        }
    }

    async open(context) {
        switch (context.type) {
            case 'lightgbm.pickle': {
                const obj = context.target;
                return new lightgbm.Model(obj, 'LightGBM Pickle');
            }
            case 'lightgbm.text': {
                const stream = context.stream;
                const buffer = stream.peek();
                const decoder = new TextDecoder('utf-8');
                const model_str = decoder.decode(buffer);
                const execution = new python.Execution();
                const obj = execution.invoke('lightgbm.basic.Booster', []);
                obj.LoadModelFromString(model_str);
                return new lightgbm.Model(obj, 'LightGBM');
            }
            default: {
                throw new lightgbm.Error(`Unsupported LightGBM format '${context.type}'.`);
            }
        }
    }
};

lightgbm.Model = class {

    constructor(obj, format) {
        this.format = format + (obj && obj.version ? ` ${obj.version}` : '');
        this.graphs = [new lightgbm.Graph(obj)];
    }
};

lightgbm.Graph = class {

    constructor(model) {
        this.inputs = [];
        this.outputs = [];
        this.nodes = [];
        const values = [];
        const feature_names = model.feature_names || [];
        for (let i = 0; i < feature_names.length; i++) {
            const name = feature_names[i];
            // const info = model.feature_infos && i < model.feature_infos.length ? model.feature_infos[i] : null;
            const value = new lightgbm.Value(name);
            values.push(value);
            if (feature_names.length < 1000) {
                const argument = new lightgbm.Argument(name, [value]);
                this.inputs.push(argument);
            }
        }
        const node = new lightgbm.Node(model, values);
        this.nodes.push(node);
    }
};

lightgbm.Argument = class {

    constructor(name, value) {
        this.name = name;
        this.value = value;
    }
};

lightgbm.Value = class {

    constructor(name) {
        if (typeof name !== 'string') {
            throw new lightgbm.Error(`Invalid value identifier '${JSON.stringify(name)}'.`);
        }
        this.name = name;
    }
};

lightgbm.Node = class {

    constructor(obj, values, stack) {
        const type = obj && obj.__class__ ? `${obj.__class__.__module__}.${obj.__class__.__name__}` : 'builtins.object';
        this.name = '';
        this.type = { name: type };
        this.inputs = [];
        this.outputs = [];
        this.attributes = [];
        if (values) {
            const argument = new lightgbm.Argument('features', values);
            this.inputs.push(argument);
        }
        const isObject = (obj) => {
            if (obj && typeof obj === 'object') {
                const proto = Object.getPrototypeOf(obj);
                return proto === Object.prototype || proto === null;
            }
            return false;
        };
        stack = stack || new Set();
        const entries = Object.entries(obj).filter(([key, value]) => value !== undefined && key !== 'feature_names' && key !== 'feature_infos');
        for (const [key, value] of entries) {
            if (Array.isArray(value) && value.every((obj) => isObject(obj))) {
                const values = value.filter((obj) => !stack.has(obj));
                const nodes = values.map((obj) => {
                    stack.add(obj);
                    const node = new lightgbm.Node(obj, null, stack);
                    stack.delete(obj);
                    return node;
                });
                const attribute = new lightgbm.Attribute('object[]', key, nodes);
                this.attributes.push(attribute);
                continue;
            } else if (isObject(value) && !stack.has(value)) {
                stack.add(obj);
                const node = new lightgbm.Node(obj, null, stack);
                stack.delete(obj);
                const attribute = new lightgbm.Attribute('object', key, node);
                this.attributes.push(attribute);
            } else {
                const attribute = new lightgbm.Attribute(null, key, value);
                this.attributes.push(attribute);
            }
        }
    }
};

lightgbm.Attribute = class {

    constructor(type, name, value) {
        this.type = type;
        this.name = name;
        this.value = value;
    }
};

lightgbm.Error = class extends Error {

    constructor(message) {
        super(message);
        this.name = 'Error loading LightGBM model.';
    }
};

export const ModelFactory = lightgbm.ModelFactory;

