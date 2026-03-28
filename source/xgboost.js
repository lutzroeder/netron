
// Experimental

import * as python from './python.js';

const xgboost = {};

xgboost.ModelFactory = class {

    async match(context) {
        const obj = await context.peek('json');
        if (obj && obj.learner && obj.version && Object.keys(obj).length < 256) {
            return context.set('xgboost.json', obj);
        }
        const stream = context.stream;
        if (stream && stream.length > 4) {
            const buffer = stream.peek(4);
            if (buffer[0] === 0x7B && buffer[1] === 0x4C && buffer[2] === 0x00 && buffer[3] === 0x00) {
                return context.set('xgboost.ubj', stream);
            }
            const signature = String.fromCharCode.apply(null, buffer);
            if (signature.startsWith('binf')) {
                return context.set('xgboost.binf', stream);
            }
            if (signature.startsWith('bs64')) {
                return context.set('xgboost.bs64', stream);
            }
            const reader = await context.read('text', 0x100);
            const line = reader.read('\n');
            if (line !== undefined && line.trim() === 'booster[0]:') {
                return context.set('xgboost.text', stream);
            }
        }
        return null;
    }

    async open(context) {
        if (context.type === 'xgboost.json') {
            const execution = new python.Execution();
            const obj = execution.invoke('xgboost.core.Booster', []);
            obj.load_model(context.value);
            const version = obj.version ? obj.version.join('.') : '';
            return new xgboost.Model(obj, `XGBoost JSON${version ? ` v${version}` : ''}`);
        }
        if (context.type === 'xgboost.ubj') {
            const execution = new python.Execution();
            const obj = execution.invoke('xgboost.core.Booster', []);
            const buffer = context.value.read();
            obj.load_model(buffer);
            const version = obj.version ? obj.version.join('.') : '';
            return new xgboost.Model(obj, `XGBoost UBJSON${version ? ` v${version}` : ''}`);
        }
        if (context.type === 'xgboost.text') {
            throw new xgboost.Error('File contains unsupported XGBoost text data.');
        }
        throw new xgboost.Error('File contains unsupported XGBoost data.');
    }
};

xgboost.Model = class {

    constructor(obj, format) {
        this.format = format;
        this.modules = [new xgboost.Graph(obj)];
    }
};

xgboost.Graph = class {

    constructor(obj) {
        this.inputs = [];
        this.outputs = [];
        this.nodes = [];
        const values = [];
        const feature_names = obj.feature_names || [];
        for (let i = 0; i < feature_names.length; i++) {
            const name = feature_names[i];
            const value = new xgboost.Value(name);
            values.push(value);
            if (feature_names.length < 1000) {
                this.inputs.push(new xgboost.Argument(name, [value]));
            }
        }
        const node = new xgboost.Node(obj, values);
        this.nodes.push(node);
    }
};

xgboost.Argument = class {

    constructor(name, value, type = null) {
        this.name = name;
        this.value = value;
        this.type = type;
    }
};

xgboost.Value = class {

    constructor(name) {
        this.name = name;
    }
};

xgboost.Node = class {

    constructor(obj, values, stack) {
        const type = obj && obj.__class__ ? `${obj.__class__.__module__}.${obj.__class__.__name__}` : 'builtins.object';
        this.name = '';
        this.type = { name: type };
        this.inputs = [];
        this.outputs = [];
        this.attributes = [];
        if (values) {
            this.inputs.push(new xgboost.Argument('features', values));
        }
        const isObject = (obj) => {
            if (obj && typeof obj === 'object') {
                const proto = Object.getPrototypeOf(obj);
                return proto === Object.prototype || proto === null;
            }
            return false;
        };
        stack = stack || new Set();
        const entries = Object.entries(obj).filter(([key, value]) => value !== undefined && key !== 'feature_names' && key !== 'feature_types');
        for (const [key, value] of entries) {
            if (Array.isArray(value) && value.every((item) => isObject(item))) {
                const items = value.filter((item) => !stack.has(item));
                const nodes = items.map((item) => {
                    stack.add(item);
                    const node = new xgboost.Node(item, null, stack);
                    stack.delete(item);
                    return node;
                });
                this.attributes.push(new xgboost.Argument(key, nodes, 'object[]'));
            } else if (isObject(value) && !stack.has(value)) {
                stack.add(value);
                const node = new xgboost.Node(value, null, stack);
                stack.delete(value);
                this.attributes.push(new xgboost.Argument(key, node, 'object'));
            } else {
                this.attributes.push(new xgboost.Argument(key, value));
            }
        }
    }
};

xgboost.Error = class extends Error {

    constructor(message) {
        super(message);
        this.name = 'Error loading XGBoost model.';
    }
};

export const ModelFactory = xgboost.ModelFactory;
