
// Experimental

const qnn = {};

qnn.ModelFactory = class {

    match(context) {
        const obj = context.peek('json');
        if (obj && obj['model.cpp']) {
            context.type = 'qnn';
            context.target = obj;
        }
    }

    async open(/* context */) {
        throw new qnn.Error("File contains QNN data.");
    }
};

qnn.Error = class extends Error {

    constructor(message) {
        super(message);
        this.name = 'Error loading QNN model.';
    }
};

export const ModelFactory = qnn.ModelFactory;
