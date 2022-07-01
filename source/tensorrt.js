
var tensorrt = tensorrt || {};
var base = base || require('./base');

tensorrt.ModelFactory = class {

    match(context) {
        const stream = context.stream;
        const engine = tensorrt.Engine.open(stream);
        if (engine) {
            return engine;
        }
        /*
        const extension = context.identifier.split('.').pop().toLowerCase();
        if (extension === 'plan') {
            return tensorrt.Plan.open(stream);
        }
        */
        return undefined;
    }

    open(context, match) {
        return Promise.resolve().then(() => new tensorrt.Model(null, match));
    }
};

tensorrt.Model = class {

    constructor(metadata, model) {
        this._format = model.format;
        this._graphs = [ new tensorrt.Graph(metadata, model) ];
    }

    get format() {
        return this._format;
    }

    get graphs() {
        return this._graphs;
    }
};

tensorrt.Graph = class {

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

// TODO

tensorrt.Engine = class {

    static open(stream) {
        const signature = [ 0x70, 0x74, 0x72, 0x74 ]; // ptrt
        if (stream.length >= 24 && stream.peek(4).every((value, index) => value === signature[index])) {
            return new tensorrt.Engine(stream);
        }
        return null;
    }

    constructor(stream) {
        this._stream = stream;
    }

    get format() {
        this._read();
        return 'TensorRT Engine';
    }

    _read() {
        if (this._stream) {
            const buffer = this._stream.peek(24);
            const reader = new base.BinaryReader(buffer);
            reader.skip(4); // signature
            const version = reader.uint32();
            reader.uint32();
            if (version <= 0x2B) {
                reader.uint32();
            }
            /* const size = */ reader.uint64();
            if (version > 0x2B) {
                reader.uint32();
            }
            throw new tensorrt.Error('Invalid file content. File contains undocumented TensorRT engine data.');
        }
    }
};

/*
tensorrt.Plan = class {

    static open(stream) {
        return new tensorrt.Plan(stream);
    }

    constructor(stream) {
        this._stream = stream;
    }

    get format() {
        this._read();
        return 'TensorRT Plan';
    }

    _read() {
        throw new tensorrt.Error('Invalid file content. File contains undocumented TensorRT plan data.');
    }
};
*/

tensorrt.Error = class extends Error {

    constructor(message) {
        super(message);
        this.name = 'Error loading TensorRT model.';
        this.stack = undefined;
    }
};

if (typeof module !== 'undefined' && typeof module.exports === 'object') {
    module.exports.ModelFactory = tensorrt.ModelFactory;
}
