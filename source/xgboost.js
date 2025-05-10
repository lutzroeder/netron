
// Experimental

import * as python from './python.js';

const xgboost = {};

xgboost.ModelFactory = class {

    async match(context) {
        const obj = await context.peek('json');
        if (obj && obj.learner && obj.version) {
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
            const model = execution.invoke('xgboost.core.Booster', []);
            model.load_model(context.value);
            throw new xgboost.Error('File contains unsupported XGBoost JSON data.');
        }
        if (context.type === 'xgboost.text') {
            throw new xgboost.Error('File contains unsupported XGBoost text data.');
        }
        throw new xgboost.Error('File contains unsupported XGBoost data.');
    }
};

xgboost.Error = class extends Error {

    constructor(message) {
        super(message);
        this.name = 'Error loading XGBoost model.';
    }
};

export const ModelFactory = xgboost.ModelFactory;
