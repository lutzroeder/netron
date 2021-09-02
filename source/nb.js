/* jshint esversion: 6 */

// Experimental

var nb = nb || {};
var flatbuffers = flatbuffers || require('./flatbuffers');

nb.ModelFactory = class {

    match(context) {
        if (context.stream) {
            switch (context.identifier.toLowerCase()) {
                case '__model__.nb':
                    return 'paddlelite.model.naivebuffer';
                case 'param.nb':
                    return 'paddlelite.data.naivebuffer';
            }
        }
        return '';
    }

    open(/* context, match */) {
        throw new nb.Error('Invalid file content. File contains Paddle Lite naive buffer data.');
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