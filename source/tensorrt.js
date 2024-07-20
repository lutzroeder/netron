
import * as base from './base.js';

const tensorrt = {};

tensorrt.ModelFactory = class {

    match(context) {
        const entries = [
            tensorrt.Engine,
            tensorrt.Container
        ];
        for (const entry of entries) {
            const target = entry.open(context);
            if (target) {
                context.type = target.type;
                context.target = target;
                break;
            }
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
        this.format = model.format;
        this.graphs = [new tensorrt.Graph(metadata, model)];
    }
};

tensorrt.Graph = class {

    constructor(/* metadata, model */) {
        this.inputs = [];
        this.outputs = [];
        this.nodes = [];
    }
};

tensorrt.Engine = class {

    static open(context) {
        const stream = context.stream;
        if (stream && stream.length >= 24) {
            let offset = 0;
            let buffer = stream.peek(Math.min(stream.length, 24));
            if (buffer[3] === 0x00 && buffer[4] === 0x7b) {
                const reader = base.BinaryReader.open(buffer);
                offset = reader.uint32() + 4;
                if ((offset + 4) < stream.length) {
                    const position = stream.position;
                    stream.seek(offset);
                    buffer = stream.peek(4);
                    stream.seek(position);
                }
            }
            const signature = String.fromCharCode.apply(null, buffer.slice(0, 4));
            if (signature === 'ptrt' || signature === 'ftrt') {
                return new tensorrt.Engine(context, offset);
            }
        }
        return null;
    }

    constructor(context, position) {
        this.type = 'tensorrt.engine';
        this.format = 'TensorRT Engine';
        this.context = context;
        this.position = position;
    }

    read() {
        const reader = this.context.read('binary');
        reader.skip(this.position);
        const buffer = reader.peek(24);
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
        // buffer = this.stream.read(24 + size);
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
                const reader = base.BinaryReader.open(buffer);
                const length = reader.uint64().toNumber();
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
        this.type = 'tensorrt.container';
        this.format = 'TensorRT FlatBuffers';
        this.stream = stream;
    }

    read() {
        delete this.stream;
        // const buffer = this.stream.peek(Math.min(24, this.stream.length));
        // const content = Array.from(buffer).map((c) => (c < 16 ? '0' : '') + c.toString(16)).join('');
        throw new tensorrt.Error('Invalid file content. File contains undocumented TensorRT data.');
    }
};

tensorrt.BinaryReader = class {

    constructor(reader) {
        this._reader = reader;
    }

    get position() {
        return this._reader.position;
    }

    uint64() {
        return this._reader.uint64();
    }

    string() {
        const length = this.uint64().toNumber();
        const position = this.position;
        this.skip(length);
        const data = this._buffer.subarray(position, this.position);
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
