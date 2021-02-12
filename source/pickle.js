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
        if (tags.size === 1) {
            return true;
        }
        return false;
    }

    open(context) {
        return new Promise((resolve) => {
            const value = context.tags('pkl').values().next().value;
            if (value === null || value === undefined) {
                context.exception(new pickle.Error('Unknown Pickle null object.'));
            }
            else if (Array.isArray(value)) {
                context.exception(new pickle.Error('Unknown Pickle array object.'));
            }
            else if (value && value.__module__ && value.__name__) {
                context.exception(new pickle.Error("Unknown Pickle type '" + value.__module__ + "." + value.__name__ + "'."));
            }
            else {
                context.exception(new pickle.Error('Unknown Pickle object.'));
            }
            resolve(new pickle.Model(value));
        });
    }
};

pickle.Model = class {

    constructor(value) {
        this._graphs = [ new pickle.Graph(value) ];
    }

    get format() {
        return 'Pickle';
    }

    get graphs() {
        return this._graphs;
    }
};

pickle.Graph = class {

    constructor(/* value */) {
        this._inputs = [];
        this._outputs = [];
        this._nodes = [ new pickle.Node() ];
    }

    get inputs() {
        return this._inputs;
    }

    get outputs() {
        return this._outputs;
    }

    get nodes() {
        return this._nodes;
    }
};

pickle.Node = class {

    constructor(/* value */) {
        this._inputs = [];
        this._outputs = [];
        this._attributes = [];
    }

    get type() {
        return '?';
    }

    get name() {
        return '';
    }

    get inputs() {
        return this._inputs;
    }

    get outputs() {
        return this._outputs;
    }

    get attributes() {
        return this._attributes;
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