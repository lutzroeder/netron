
import * as python from './python.js';

const catboost = {};

catboost.ModelFactory = class {

    async match(context) {
        const stream = context.stream;
        if (stream && stream.length > 4) {
            const buffer = stream.peek(4);
            const signature = Array.from(buffer).map((c) => String.fromCharCode(c)).join('');
            if (signature === 'CBM1') {
                context.type = 'catboost';
            }
        }
    }

    async open(context) {
        const stream = context.stream;
        const execution = new python.Execution();
        const model = execution.invoke('catboost.CatBoostClassifier', []);
        model.load_model(stream);
    }
};

catboost.Error = class extends Error {

    constructor(message) {
        super(message);
        this.name = 'Error loading CatBoost model.';
    }
};

export const ModelFactory = catboost.ModelFactory;

