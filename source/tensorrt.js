
var tensorrt = tensorrt || {};

tensorrt.ModelFactory = class {

    match(context) {
        const stream = context.stream;
        const engine = tensorrt.Engine.open(stream);
        if (engine) {
            return engine;
        }
        const extension = context.identifier.split('.').pop().toLowerCase();
        if (extension === 'plan') {
            return tensorrt.Plan.open(stream);
        }
        return undefined;
    }

    open(context, match) {
        return tensorrt.Metadata.open(context).then((metadata) => {
            return new tensorrt.Model(metadata, match);
        });
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
        if (stream.length >= 4 && stream.peek(4).every((value, index) => value === signature[index])) {
            return new tensorrt.Engine(stream);
        }
        return null;
    }

    constructor(stream) {
        this._stream = stream;
    }

    get format() {
        this._read();
        return 'Tensor RT Engine';
    }

    _read() {
        throw new tensorrt.Error('Invalid file content. File contains undocumented TensorRT engine data.');
    }
};

tensorrt.Plan = class {

    static open(stream) {
        return new tensorrt.Plan(stream);
    }

    constructor(stream) {
        this._stream = stream;
    }

    get format() {
        this._read();
        return 'Tensor RT Plan';
    }

    _read() {
        throw new tensorrt.Error('Invalid file content. File contains undocumented TensorRT plan data.');
    }
};


tensorrt.Metadata = class {

    static open(context) {
        if (tensorrt.Metadata._metadata) {
            return Promise.resolve(tensorrt.Metadata._metadata);
        }
        return context.request('tensorrt-metadata.json', 'utf-8', null).then((data) => {
            tensorrt.Metadata._metadata = new tensorrt.Metadata(data);
            return tensorrt.Metadata._metadata;
        }).catch(() => {
            tensorrt.Metadata._metadata = new tensorrt.Metadata(null);
            return tensorrt.Metadata._metadata;
        });
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
