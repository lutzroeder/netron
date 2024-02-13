
import * as base from './base.js';

const tensorrt = {};

tensorrt.ModelFactory = class {

    match(context) {
        const engine = tensorrt.Engine.open(context);
        if (engine) {
            context.target = engine;
            context.type = 'tensorrt.engine';
            return;
        }
        const container = tensorrt.Container.open(context);
        if (container) {
            context.target = container;
            context.type = 'tensorrt.container';
            return;
        }
    }

    async open(context) {
        const target = context.target;
        target.read();
        return new tensorrt.Model(null, target);
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

    static open(context) {
        const stream = context.stream;
        if (stream && stream.length >= 24) {
            const buffer = stream.peek(Math.min(stream.length, 1024));
            const reader = new base.BinaryReader(buffer);
            const match = (reader) => {
                const buffer = reader.peek(4);
                const signature = String.fromCharCode.apply(null, buffer);
                return signature === 'ptrt' || signature === 'ftrt';
            };
            if (match(reader)) {
                return new tensorrt.Engine(context, 0);
            }
            const size = reader.uint32();
            if (size < 1000 && size < (reader.length - 4)) {
                reader.skip(size);
                const position = reader.position;
                if (match(reader)) {
                    return new tensorrt.Engine(context, position);
                }
            }
        }
        return null;
    }

    constructor(context, position) {
        this.context = context;
        this.position = position;
        this.format = 'TensorRT Engine';
    }

    read() {
        const stream = this.context.stream;
        stream.skip(this.position);
        const buffer = stream.peek(24);
        const reader = this.context.reader;
        delete this.context;
        delete this.position;
        reader.skip(4);
        const version = reader.uint32();
        reader.uint32();
        // let size = 0;
        switch (version) {
            case 0x0000:
            case 0x002B: {
                reader.uint32();
                /* size = */ reader.uint64();
                break;
            }
            case 0x0057:
            case 0x0059:
            case 0x0060:
            case 0x0061: {
                /* size = */ reader.uint64();
                reader.uint32();
                break;
            }
            default: {
                const content = Array.from(buffer).map((c) => (c < 16 ? '0' : '') + c.toString(16)).join('');
                throw new tensorrt.Error(`Unsupported TensorRT engine signature (${content.substring(8)}).`);
            }
        }
        // const content = Array.from(buffer).map((c) => (c < 16 ? '0' : '') + c.toString(16)).join('');
        // buffer = this._stream.read(24 + size);
        // reader = new tensorrt.BinaryReader(buffer);
        throw new tensorrt.Error('Invalid file content. File contains undocumented TensorRT engine data.');
    }
};

tensorrt.Container = class {

    static open(context) {
        const stream = context.stream;
        if (stream) {
            const buffer = stream.peek(Math.min(512, stream.length));
            if (buffer.length > 12 && buffer[6] === 0x00 && buffer[7] === 0x00) {
                const reader = new base.BinaryReader(buffer);
                const length = Number(reader.uint64());
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
        }
        return null;
    }

    constructor(stream) {
        this.stream = stream;
        this.format = 'TensorRT FlatBuffers';
    }

    read() {
        delete this.stream;
        // const buffer = this._stream.peek(Math.min(24, this._stream.length));
        // const content = Array.from(buffer).map((c) => (c < 16 ? '0' : '') + c.toString(16)).join('');
        throw new tensorrt.Error('Invalid file content. File contains undocumented TensorRT data.');
    }
};

tensorrt.BinaryReader = class extends base.BinaryReader {

    string() {
        const length = Number(this.uint64());
        const position = this._position;
        this.skip(length);
        const data = this._buffer.subarray(position, this._position);
        this._decoder = this._decoder || new TextDecoder('utf-8');
        return this._decoder.decode(data);
    }
};

tensorrt.Error = class extends Error {

    constructor(message) {
        super(message);
        this.name = 'Error loading TensorRT model.';
    }
};

export const ModelFactory = tensorrt.ModelFactory;
