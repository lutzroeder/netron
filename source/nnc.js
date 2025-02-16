
const nnc = {};

nnc.ModelFactory = class {

    async match(context) {
        return context.set('nnc');
    }

    async open(/* context */) {
        throw new nnc.Error('File contains undocumented NNC data.');
    }
};

nnc.Error = class extends Error {

    constructor(message) {
        super(message);
        this.name = 'Error loading NNC model.';
    }
};

export const ModelFactory = nnc.ModelFactory;
