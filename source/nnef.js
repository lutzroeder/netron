
const nnef = {};

nnef.ModelFactory = class {

    match(context) {
        const identifier = context.identifier;
        const extension = identifier.split('.').pop().toLowerCase();
        switch (extension) {
            case 'nnef': {
                const reader = nnef.TextReader.open(context);
                if (reader) {
                    context.type = 'nnef.graph';
                    context.target = reader;
                }
                break;
            }
            case 'dat': {
                const stream = context.stream;
                if (stream && stream.length > 2) {
                    const buffer = stream.peek(2);
                    if (buffer[0] === 0x4E && buffer[1] === 0xEF) {
                        context.type = 'nnef.dat';
                        context.target = stream;
                    }
                }
                break;
            }
            default:
                break;
        }
    }

    filter(context, type) {
        return context.type !== 'nnef.graph' || type !== 'nnef.dat';
    }

    async open(context) {
        switch (context.type) {
            case 'nnef.graph': {
                const reader = context.target;
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

    static open(context) {
        const reader = context.read('text', 65536);
        for (let i = 0; i < 32; i++) {
            const line = reader.read('\n');
            const match = /version\s*(\d+\.\d+);/.exec(line);
            if (match) {
                return new nnef.TextReader(context, match[1]);
            }
            if (line === undefined) {
                break;
            }
        }
        return null;
    }

    constructor(context, version) {
        this.context = context;
        this.version = version;
    }
};

nnef.Error = class extends Error {

    constructor(message) {
        super(message);
        this.name = 'Error loading NNEF model.';
    }
};

export const ModelFactory = nnef.ModelFactory;
