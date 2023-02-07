
var mlir = {};
var text = require('./text');

mlir.ModelFactory = class {

    match(context) {
        const stream = context.stream;
        if (stream) {
            const reader = text.Reader.open(stream, 2048);
            for (;;) {
                const line = reader.read();
                if (line === undefined) {
                    break;
                }
                if (line.indexOf('module ') !== -1) {
                    return 'mlir';
                }
            }
        }
        return null;
    }

    open(context) {
        const stream = context.stream;
        const decoder = text.Decoder.open(stream);
        new mlir.Parser(decoder);
        throw new mlir.Error('MLIR support is not implemented.');
    }
};

mlir.Model = class {

    constructor() {
        this._format = 'MLIR';
        this._graphs = [];
    }

    get format() {
        return this._format;
    }

    get graphs() {
        return this._graphs;
    }
};

mlir.Tokenizer = class {

    constructor(decoder) {
        this._decoder = decoder;
    }
};

mlir.Parser = class {

    constructor(decoder) {
        this._tokenizer = new mlir.Tokenizer(decoder);
    }
};

mlir.Error = class extends Error {

    constructor(message) {
        super(message);
        this.name = 'Error loading MLIR model.';
    }
};

if (typeof module !== 'undefined' && typeof module.exports === 'object') {
    module.exports.ModelFactory = mlir.ModelFactory;
}
