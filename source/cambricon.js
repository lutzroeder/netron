
const cambricon = {};

cambricon.ModelFactory = class {

    match(context) {
        const stream = context.stream;
        if (stream) {
            const buffer = stream.peek(Math.min(20, stream.length));
            const text = Array.from(buffer).map((c) => String.fromCharCode(c)).join('');
            if (text.startsWith('\x7fMEF') || text.startsWith('cambricon_offline')) {
                context.type = 'cambricon';
            }
        }
    }

    async open(/* context */) {
        throw new cambricon.Error("File contains undocumented Cambricon data.");
    }
};

cambricon.Error = class extends Error {

    constructor(message) {
        super(message);
        this.name = 'Error loading Cambricon model.';
    }
};

export const ModelFactory = cambricon.ModelFactory;

