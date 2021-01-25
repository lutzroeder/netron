/* jshint esversion: 6 */

// Experimental

var pickle = pickle || {};
var python = python || require('./python');
var zip = zip || require('./zip');

pickle.ModelFactory = class {

    match(context) {
        const stream = context.stream;
        const signature = [ 0x80, undefined, 0x8a, 0x0a, 0x6c, 0xfc, 0x9c, 0x46, 0xf9, 0x20, 0x6a, 0xa8, 0x50, 0x19 ];
        if (signature.length <= stream.length && stream.peek(signature.length).every((value, index) => signature[index] === undefined || signature[index] === value)) {
            // Reject PyTorch models with .pkl file extension.
            return false;
        }
        const tags = context.tags('pkl');
        if (tags.size === 1 || tags.keys().next().value === '') {
            return true;
        }
        return false;
    }

    open(/* context */) {
        return new Promise((resolve) => {
            resolve(new pickle.Model());
        });
    }
};

pickle.Model = class {

    constructor() {
        this._graphs = [];
    }

    get format() {
        return 'Pickle';
    }

    get graphs() {
        return this._graphs;
    }
};


pickle.Error = class extends Error {

    constructor(message) {
        super(message);
        this.name = 'Error loading Pickle model.';
    }
};

if (typeof module !== 'undefined' && typeof module.exports === 'object') {
    module.exports.ModelFactory = pickle.ModelFactory;
}