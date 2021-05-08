/* jshint esversion: 6 */

var tensorrt = tensorrt || {};

tensorrt.ModelFactory = class {

    match(context) {
        const stream = context.stream;
        const signature = [ 0x70, 0x74, 0x72, 0x74 ]; // ptrt
        if (stream.length >= 4 && stream.peek(4).every((value, index) => value === signature[index])) {
            return true;
        }
        return false;
    }

    open(context) {
        return tensorrt.Metadata.open(context).then((metadata) => {
            const stream = context.stream;
            const buffer = stream.peek();
            const model = new tensorrt.Container(buffer);
            return new tensorrt.Model(metadata, model);
        });
    }
};

tensorrt.Model = class {

    constructor(metadata, model) {
        this._format = 'TensorRT';
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

tensorrt.Container = class {

    constructor(/* buffer */) {
        throw new tensorrt.Error('Invalid file content. File contains undocumented TensorRT engine data.');
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
    }
};

if (typeof module !== 'undefined' && typeof module.exports === 'object') {
    module.exports.ModelFactory = tensorrt.ModelFactory;
}
