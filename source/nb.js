/* jshint esversion: 6 */

// Experimental

var nb = nb || {};
var flatbuffers = flatbuffers || require('./flatbuffers');

nb.ModelFactory = class {

    match(context) {
        if (context.stream) {
            switch (context.identifier.toLowerCase()) {
                case '__model__.nb':
                    return 'paddlelite.model.flatbuffers';
                case 'param.nb':
                    return 'paddlelite.data.flatbuffers';
            }
        }
        return '';
    }

    open(/* context, match */) {
        throw new nb.Error('Invalid file content. File contains paddle.lite.fbs.proto data.');
        /*
        return context.require('./nb-schema').then(() => {
            nb.schema = flatbuffers.get('nb').paddle.lite.fbs.proto;
            switch (match) {
                case 'paddlelite.model.flatbuffers': {
                    const stream = context.stream;
                    const reader = flatbuffers.BinaryReader.open(stream);
                    const model = nb.schema.ProgramDesc.create(reader);
                    break;
                }
            }
            throw new nb.Error('Invalid file content. File contains paddle.lite.fbs.proto data.');
        });
        */
    }
};

nb.Error = class extends Error {

    constructor(message, context) {
        super(message);
        this.name = 'Error loading Paddle Lite model.';
        this.context = context === false ? false : true;
    }
};

if (typeof module !== 'undefined' && typeof module.exports === 'object') {
    module.exports.ModelFactory = nb.ModelFactory;
}