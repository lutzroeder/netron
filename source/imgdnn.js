
var imgdnn = imgdnn || {};

imgdnn.ModelFactory = class {

    match(context) {
        const stream = context.stream;
        const signature = [ 0x49, 0x4d, 0x47, 0x44, 0x4e, 0x4e ]; // IMGDNN
        if (stream && stream.length >= signature.length && stream.peek(6).every((value, index) => value === signature[index])) {
            return 'imgdnn';
        }
        return null;
    }

    open(context) {
        return imgdnn.Metadata.open(context).then((/* metadata */) => {
            // const stream = context.stream;
            // const buffer = stream.peek();
            throw new imgdnn.Error('Invalid file content. File contains undocumented IMGDNN data.');
            // return new imgdnn.Model(metadata, model);
        });
    }
};

imgdnn.Model = class {

    constructor(metadata, model) {
        this._format = 'IMGDNN';
        this._graphs = [ new imgdnn.Graph(metadata, model) ];
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

imgdnn.Metadata = class {

    static open(/* context */) {
        imgdnn.Metadata._metadata = imgdnn.Metadata._metadata || new imgdnn.Metadata(null);
        return Promise.resolve(imgdnn.Metadata._metadata);
    }

    constructor(data) {
        this._map = new Map();
        if (data) {
            const metadata = JSON.parse(data);
            this._map = new Map(metadata.map((item) => [ item.name, item ]));
        }
    }

    type(name) {
        return this._map.get(name);
    }
};

imgdnn.Error = class extends Error {

    constructor(message) {
        super(message);
        this.name = 'Error loading IMGDNN model.';
    }
};

if (typeof module !== 'undefined' && typeof module.exports === 'object') {
    module.exports.ModelFactory = imgdnn.ModelFactory;
}
