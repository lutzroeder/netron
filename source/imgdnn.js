
const imgdnn = {};

imgdnn.ModelFactory = class {

    async match(context) {
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
        this.format = 'IMGDNN';
        this.graphs = [new imgdnn.Graph(metadata, model)];
    }
};

imgdnn.Graph = class {

    constructor(/* metadata, model */) {
        this.inputs = [];
        this.outputs = [];
        this.nodes = [];
    }
};

imgdnn.Error = class extends Error {

    constructor(message) {
        super(message);
        this.name = 'Error loading IMGDNN model.';
    }
};

export const ModelFactory = imgdnn.ModelFactory;

