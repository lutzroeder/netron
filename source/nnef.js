
import * as text from './text.js';

const nnef = {};

nnef.ModelFactory = class {

    match(context) {
        const identifier = context.identifier;
        const extension = identifier.split('.').pop().toLowerCase();
        if (extension === 'nnef') {
            const stream = context.stream;
            if (nnef.TextReader.open(stream)) {
                context.type = 'nnef.graph';
            }
        } else if (extension === 'dat') {
            const stream = context.stream;
            if (stream && stream.length > 2) {
                const buffer = stream.peek(2);
                if (buffer[0] === 0x4E && buffer[1] === 0xEF) {
                    context.type = 'nnef.dat';
                }
            }
        }
    }

    filter(context, type) {
        return context.type !== 'nnef.graph' || type !== 'nnef.dat';
    }

    async open(context) {
        switch (context.type) {
            case 'nnef.graph': {
                const stream = context.stream;
                const reader = nnef.TextReader.open(stream);
                throw new nnef.Error(`NNEF v${reader.version} support not implemented.`);
            }
            case 'nnef.dat': {
                throw new nnef.Error('NNEF dat format support not implemented.');
            }
            default: {
                throw new nnef.Error(`Unsupported NNEF format '${context.type}'.`);
            }
        }
    }
};

nnef.TextReader = class {

    static open(stream) {
        const reader = text.Reader.open(stream);
        for (let i = 0; i < 32; i++) {
            const line = reader.read();
            const match = /version\s*(\d+\.\d+);/.exec(line);
            if (match) {
                return new nnef.TextReader(stream, match[1]);
            }
            if (line === undefined) {
                break;
            }
        }
        return null;
    }

    constructor(stream, version) {
        this._stream = stream;
        this._version = version;
    }

    get version() {
        return this._version;
    }
};

nnef.Error = class extends Error {

    constructor(message) {
        super(message);
        this.name = 'Error loading NNEF model.';
    }
};

export const ModelFactory = nnef.ModelFactory;
