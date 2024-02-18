
const imgdnn = {};

imgdnn.ModelFactory = class {

    match(context) {
        const stream = context.stream;
        const signature = [0x49, 0x4d, 0x47, 0x44, 0x4e, 0x4e]; // IMGDNN
        if (stream && stream.length >= signature.length && stream.peek(6).every((value, index) => value === signature[index])) {
            return 'imgdnn';
        }
        return null;
    }

    open(/* context */) {
        throw new imgdnn.Error('Invalid file content. File contains undocumented IMGDNN data.');
    }
};

imgdnn.Model = class {

    constructor(metadata, model) {
        this._format = 'IMGDNN';
        this._graphs = [new imgdnn.Graph(metadata, model)];
    }

    get format() {
        return this._format;
    }

    get graphs() {
        return this._graphs;
    }
};

imgdnn.Graph = class {

    constructor(/* metadata, model */) {
        this._inputs = [];
        this._outputs = [];
        this._nodes = [];
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

imgdnn.Error = class extends Error {

    constructor(message) {
        super(message);
        this.name = 'Error loading IMGDNN model.';
    }
};

export const ModelFactory = imgdnn.ModelFactory;

