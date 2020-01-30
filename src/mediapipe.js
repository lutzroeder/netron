/* jshint esversion: 6 */
/* eslint "indent": [ "error", 4, { "SwitchCase": 1 } ] */

var mediapipe = mediapipe || {};
var prototxt = prototxt || require('protobufjs/ext/prototxt');

mediapipe.ModelFactory = class {

    match(context) {
        const identifier = context.identifier;
        const extension = identifier.split('.').pop().toLowerCase();
        if (extension === 'pbtxt') {
            const tags = context.tags('pbtxt');
            if (tags.has('node') && (tags.has('input_side_packet') || tags.has('input_stream') || tags.has('output_stream'))) {
                return true;
            }
        }
        return false;
    }

    open(context /*, host */) {
        const reader = prototxt.TextReader.create(context.text);
        const root = new mediapipe.Node(reader);
        return new mediapipe.Model(root);
    }
};

mediapipe.Model = class {

    constructor(root) {
        this._graphs = [ new mediapipe.Graph(root) ];
    }

    get format() {
        return 'MediaPipe';
    }
}

mediapipe.Graph = class {

    constructor(/* root */) {
        this._inputs = [];
        this._ouputs = [];
        this._nodes = [];
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
}



mediapipe.Node = class {

    constructor(/* reader */) {
        /*
        reader.start();
        while (!reader.end()) {
            // debugger;
            var tag = reader.tag();
        }
        */
    }
}

mediapipe.Error = class extends Error {

    constructor(message) {
        super(message);
        this.name = 'Error loading MediaPipe model.';
    }
};

if (typeof module !== 'undefined' && typeof module.exports === 'object') {
    module.exports.ModelFactory = mediapipe.ModelFactory;
}