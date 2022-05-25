
var dlc = dlc || {};
var text = text || require('./text');

dlc.ModelFactory = class {

    match(context) {
        return dlc.Container.open(context);
    }

    open(context, match) {
        return context.require('./dlc-schema').then(() => {
            dlc.schema = flatbuffers.get('dlc').dlc;
            const container = match;
            return new dlc.Model(container);
        });
    }
};

dlc.Model = class {

    constructor(container) {
        if (container.metadata.size > 0) {
            const converter = container.metadata.get('converter-command');
            if (converter) {
                const source = converter.split(' ').shift().trim();
                if (source.length > 0) {
                    this._source = source;
                    const version = container.metadata.get('converter-version');
                    if (version) {
                        this._source = this._source + ' v' + version;
                    }
                }
            }
        }
        container.model;
        container.params;
        this._graphs = [];
    }

    get format() {
        return 'DLC';
    }

    get source() {
        return this._source;
    }

    get graphs() {
        return this._graphs;
    }
};

dlc.Container = class {

    static open(context) {
        const entries = context.entries('zip');
        if (entries.size > 0) {
            const model = entries.get('model');
            const params = entries.get('model.params');
            if (model || params) {
                return new dlc.Container(model, params, entries.get('dlc.metadata'));
            }
        }
        const stream = context.stream;
        switch (dlc.Container._idenfitier(stream)) {
            case 'NETD':
                return new dlc.Container(stream, null, null);
            case 'NETP':
                return new dlc.Container(null, stream, null);
            default:
                break;
        }
        return null;
    }

    constructor(model, params, metadata) {
        this._model = model || null;
        this._params = params || null;
        this._metadata = metadata || new Uint8Array(0);
    }

    get model() {
        if (this._model && this._model.peek) {
            const stream = this._model;
            const reader = this._open(stream, 'NETD');
            stream.seek(0);
            this._model = dlc.schema.NetDefinition.decode(reader, reader.root);
            throw new dlc.Error("File contains undocumented '" + reader.identifier + "' data.");
        }
        return this._model;
    }

    get params() {
        if (this._params && this._params.peek) {
            const stream = this._params;
            const reader = this._open(stream, 'NETP');
            stream.seek(0);
            this._params = dlc.schema.NetParameter.decode(reader, reader.root);
            throw new dlc.Error("File contains undocumented '" + reader.identifier + "' data.");
        }
        return this._params;
    }

    get metadata() {
        if (this._metadata && this._metadata.peek) {
            const reader = text.Reader.open(this._metadata);
            const metadata = new Map();
            for (;;) {
                const line = reader.read();
                if (line === undefined) {
                    break;
                }
                const index = line.indexOf('=');
                if (index === -1) {
                    break;
                }
                const key = line.substring(0, index);
                const value = line.substring(index + 1);
                metadata.set(key, value);
            }
            this._metadata = metadata;
        }
        return this._metadata;
    }

    _open(stream, identifier) {
        const signature = [ 0xD5, 0x0A, 0x03, 0x00, 0x00, 0x00, 0x00, 0x00 ];
        if (signature.length <= stream.length && stream.peek(signature.length).every((value, index) => value === signature[index])) {
            stream.read(8);
        }
        const buffer = stream.read();
        const reader = flatbuffers.BinaryReader.open(buffer);
        if (identifier != reader.identifier) {
            throw new dlc.Error("File contains undocumented '" + reader.identifier + "' data.");
        }
        return reader;
    }

    static _idenfitier(stream) {
        const signature = [ 0xD5, 0x0A, 0x03, 0x00, 0x00, 0x00, 0x00, 0x00 ];
        if (stream.length > 16 && stream.peek(signature.length).every((value, index) => value === signature[index])) {
            const buffer = stream.peek(16).slice(8, 16);
            const reader = flatbuffers.BinaryReader.open(buffer);
            return reader.identifier;
        }
        else if (stream.length > 8) {
            const buffer = stream.peek(8);
            const reader = flatbuffers.BinaryReader.open(buffer);
            return reader.identifier;
        }
        return '';
    }
};

dlc.Error = class extends Error {

    constructor(message) {
        super(message);
        this.name = 'Error loading DLC model.';
        this.stack = undefined;
    }
};

if (typeof module !== 'undefined' && typeof module.exports === 'object') {
    module.exports.ModelFactory = dlc.ModelFactory;
}
