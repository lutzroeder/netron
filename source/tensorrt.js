
var tensorrt = tensorrt || {};
var base = base || require('./base');

tensorrt.ModelFactory = class {

    match(context) {
        const stream = context.stream;
        return tensorrt.Engine.open(stream) || tensorrt.Container.open(stream);
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

tensorrt.Engine = class {

    static open(stream) {
        const signature = [ 0x70, 0x74, 0x72, 0x74 ]; // ptrt
        if (stream && stream.length >= 24 && stream.peek(4).every((value, index) => value === signature[index])) {
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
            const content = Array.from(buffer).map((c) => (c < 16 ? '0' : '') + c.toString(16)).join('');
            throw new tensorrt.Error("Invalid file content. File contains undocumented TensorRT engine data (" + content.substring(8) + ").");
        }
    }
};

tensorrt.Container = class {

    static open(stream) {
        const buffer = stream.peek(Math.min(512, stream.length));
        if (buffer.length > 12 && buffer[6] === 0x00 && buffer[7] === 0x00) {
            const reader = new base.BinaryReader(buffer);
            const length = reader.uint64();
            if (length === stream.length) {
                let position = reader.position + reader.uint32();
                if (position < reader.length) {
                    reader.seek(position);
                    const offset = reader.uint32();
                    position = reader.position - offset - 4;
                    if (position > 0 && position < reader.length) {
                        reader.seek(position);
                        const length = reader.uint16();
                        if (offset === length) {
                            return new tensorrt.Container(stream);
                        }
                    }
                }
            }
        }
        return null;
    }

    constructor(stream) {
        this._stream = stream;
    }

    get format() {
        this._read();
        return 'TensorRT FlatBuffers';
    }

    _read() {
        const buffer = this._stream.peek(Math.min(24, this._stream.length));
        const content = Array.from(buffer).map((c) => (c < 16 ? '0' : '') + c.toString(16)).join('');
        throw new tensorrt.Error('Invalid file content. File contains undocumented TensorRT data (' + content.substring(16) + ').');
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
