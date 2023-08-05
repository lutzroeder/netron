
var sentencepiece = {};
var protobuf = require('./protobuf');

sentencepiece.ModelFactory = class {

    match(context) {
        const tags = context.tags('pb');
        if ((tags.size >= 3 && tags.size <= 5 &&
            tags.get(1) === 2 && tags.get(2) === 2 & tags.get(3) === 2) &&
            Array.from(tags).every((entry) => entry[0] <= 5 && entry[1] === 2)) {
            const model = context.tags('pb+');
            if (model &&
                model['1'] && model['1']['1'] === 2 && model['1']['2'] === 5 && model['1']['3'] === 0 &&
                model['2'] && model['2']['1'] === 2 && model['2']['2'] === 2 && model['2']['3'] === 0 &&
                model['2']['4'] === 0 && model['2']['10'] === 5 && model['2']['16'] === 0 &&
                model['2']['40'] === 0 && model['2']['41'] === 0 && model['2']['42'] === 0 && model['2']['43'] === 0) {
                return 'sentencepiece';
            }
        }
        return undefined;
    }

    async open(context) {
        await context.require('./sentencepiece-proto');
        let model = null;
        try {
            sentencepiece.proto = protobuf.get('sentencepiece').sentencepiece;
            const stream = context.stream;
            const reader = protobuf.BinaryReader.open(stream);
            model = sentencepiece.proto.ModelProto.decode(reader);
        } catch (error) {
            const message = error && error.message ? error.message : error.toString();
            throw new sentencepiece.Error('File format is not sentencepiece.ModelProto (' + message.replace(/\.$/, '') + ').');
        }
        return new sentencepiece.Model(model);
    }
};

sentencepiece.Model = class {

    constructor() {
        this.format = 'SentencePiece';
        this.graphs = [];
        throw new sentencepiece.Error("Invalid file content. File contains sentencepiece.ModelProto data.");
    }
};

sentencepiece.Error = class extends Error {

    constructor(message) {
        super(message);
        this.name = 'Error loading SentencePiece model.';
    }
};

if (typeof module !== 'undefined' && typeof module.exports === 'object') {
    module.exports.ModelFactory = sentencepiece.ModelFactory;
}
