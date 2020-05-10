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
            const text = context.text;
            if (tags.has('node') && (text.indexOf('input_stream:') !== -1 || text.indexOf('input_side_packet:') !== -1 || text.indexOf('output_stream:') !== -1)) {
                return true;
            }
        }
        return false;
    }

    open(context, host) {
        const identifier = context.identifier;
        try {
            const reader = prototxt.TextReader.create(context.text);
            const root = new mediapipe.Object(reader);
            return Promise.resolve(new mediapipe.Model(root));
        }
        catch (error) {
            host.exception(error, false);
            const message = error && error.message ? error.message : error.toString();
            return Promise.reject(new mediapipe.Error(message.replace(/\.$/, '') + " in '" + identifier + "'."));
        }
    }
};

mediapipe.Model = class {

    constructor(root) {
        this._graphs = [ new mediapipe.Graph(root) ];
    }

    get format() {
        return 'MediaPipe';
    }

    get graphs() {
        return this._graphs;
    }
};

mediapipe.Graph = class {

    constructor(root) {
        this._inputs = [];
        this._outputs = [];
        this._nodes = [];

        if (root) {
            if (root.input_stream) {
                const inputs = Array.isArray(root.input_stream) ? root.input_stream : [ root.input_stream ];
                for (const input of inputs) {
                    let parts = input.split(':');
                    const type = (parts.length > 1) ? parts.shift() : '';
                    const name = parts.shift();
                    this._inputs.push(new mediapipe.Parameter(name, [
                        new mediapipe.Argument(name, type, null)
                    ]));
                }
            }
            if (root.output_stream) {
                const outputs = Array.isArray(root.output_stream) ? root.output_stream : [ root.output_stream ];
                for (const output of outputs) {
                    let parts = output.split(':');
                    const type = (parts.length > 1) ? parts.shift() : '';
                    const name = parts.shift();
                    this._outputs.push(new mediapipe.Parameter(name, [
                        new mediapipe.Argument(name, type, null)
                    ]));
                }
            }
            if (root.input_side_packet) {
                const inputs = Array.isArray(root.input_side_packet) ? root.input_side_packet : [ root.input_side_packet ];
                for (const input of inputs) {
                    let parts = input.split(':');
                    const type = (parts.length > 1) ? parts.shift() : '';
                    const name = parts.shift();
                    this._inputs.push(new mediapipe.Parameter(name, [
                        new mediapipe.Argument(name, type, null)
                    ]));
                }
            }
            if (root.output_side_packet) {
                const outputs = Array.isArray(root.output_side_packet) ? root.output_side_packet : [ root.output_side_packet ];
                for (const output of outputs) {
                    let parts = output.split(':');
                    const type = (parts.length > 1) ? parts.shift() : '';
                    const name = parts.shift();
                    this._outputs.push(new mediapipe.Parameter(output, [
                        new mediapipe.Argument(name, type, null)
                    ]));
                }
            }
            if (root.node) {
                const nodes = Array.isArray(root.node) ? root.node : [ root.node ];
                for (const node of nodes) {
                    this._nodes.push(new mediapipe.Node(node));
                }
            }
        }
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

mediapipe.Node = class {

    constructor(node) {
        this._type = node.calculator || '?';
        this._type = this._type.replace(/Calculator$/, '');
        this._inputs = [];
        this._outputs = [];
        this._attributes = [];

        if (node.input_stream) {
            let args = [];
            const inputs = Array.isArray(node.input_stream) ? node.input_stream : [ node.input_stream ];
            for (const input of inputs) {
                let parts = input.split(':');
                const type = (parts.length > 1) ? parts.shift() : '';
                const name = parts.shift();
                args.push(new mediapipe.Argument(name, type, null));
            }
            this._inputs.push(new mediapipe.Parameter('input_stream', args));
        }
        if (node.output_stream) {
            let args = [];
            const outputs = Array.isArray(node.output_stream) ? node.output_stream : [ node.output_stream ];
            for (const output of outputs) {
                let parts = output.split(':');
                const type = (parts.length > 1) ? parts.shift() : '';
                const name = parts.shift();
                args.push(new mediapipe.Argument(name, type, null));
            }
            this._outputs.push(new mediapipe.Parameter('output_stream', args));
        }
        if (node.input_side_packet) {
            let args = [];
            const inputs = Array.isArray(node.input_side_packet) ? node.input_side_packet : [ node.input_side_packet ];
            for (const input of inputs) {
                let parts = input.split(':');
                const type = (parts.length > 1) ? parts.shift() : '';
                const name = parts.shift();
                args.push(new mediapipe.Argument(name, type, null));
            }
            this._inputs.push(new mediapipe.Parameter('input_side_packet', args));
        }
        if (node.output_side_packet) {
            let args = [];
            const outputs = Array.isArray(node.output_side_packet) ? node.output_side_packet : [ node.output_side_packet ];
            for (const output of outputs) {
                let parts = output.split(':');
                const type = (parts.length > 1) ? parts.shift() : '';
                const name = parts.shift();
                args.push(new mediapipe.Argument(name, type, null));
            }
            this._outputs.push(new mediapipe.Parameter('output_side_packet', args));
        }
        let options = node.options || node.node_options || null;
        if (options) {
            for (const key of Object.keys(options)) {
                if (key === '__type__') {
                    continue;
                }
                const value = options[key];
                this._attributes.push(new mediapipe.Attribute(key, value));
            }
        }
    }

    get name() {
        return '';
    }

    get operator() {
        return this._type;
    }

    get metadata() {
        return null;
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

mediapipe.Attribute = class {

    constructor(name, value) {
        this._name = name;
        this._value = value;
    }

    get name() {
        return this._name;
    }

    get value() {
        return this._value;
    }

    get visible() {
        return true;
    }
};

mediapipe.Parameter = class {

    constructor(name, args) {
        this._name = name;
        this._arguments = args;
    }

    get name() {
        return this._name;
    }

    get visible() {
        return true;
    }

    get arguments() {
        return this._arguments;
    }
};

mediapipe.Argument = class {

    constructor(name, type, initializer) {
        if (typeof name !== 'string') {
            throw new mediapipe.Error("Invalid argument identifier '" + JSON.stringify(name) + "'.");
        }
        this._name = name;
        this._type = type || null;
        this._initializer = initializer || null;
    }

    get name() {
        return this._name;
    }

    get type() {
        if (this._type) {
            return this._type;
        }
        if (this._initializer) {
            return this._initializer.type;
        }
        return null;
    }

    get initializer() {
        return this._initializer;
    }
};


mediapipe.Object = class {

    constructor(reader) {
        reader.start();

        let close = false;
        const type = reader.peek();
        if (type.startsWith('[') && type.endsWith(']')) {
            this.__type__ = reader.read().substring(0, type.length - 1);
            reader.match(':');
            reader.start();
            close = true;
        }
        let arrayTags = new Set();
        while (!reader.end()) {
            var tag = reader.tag();
            var next = reader.peek();
            var obj = null;
            if (next === '{') {
                obj = new mediapipe.Object(reader);
            }
            else if (next.startsWith('"') && next.endsWith('"')) {
                obj = reader.read().substring(1, next.length - 1);
            }
            else if (next === 'true' || next === 'false') {
                obj = reader.read();
            }
            else if (reader.first()) {
                obj = [];
                while (!reader.last()) {
                    const data = reader.read();
                    if (!isNaN(data)) {
                        obj.push(parseFloat(data));
                    }
                }
            }
            else if (!isNaN(next)) {
                obj = parseFloat(reader.read());
            }
            else {
                obj = reader.read();
            }
            if (this[tag] && (!Array.isArray(this[tag]) || arrayTags.has(tag))) {
                this[tag] = [ this[tag] ];
                arrayTags.delete(tag);
            }
            if (this[tag]) {
                this[tag].push(obj);
            }
            else {
                if (Array.isArray(obj)) {
                    arrayTags.add(tag);
                }
                this[tag] = obj;
            }
        }
        if (close) {
            reader.expect('}');
        }
    }
};

mediapipe.Error = class extends Error {

    constructor(message) {
        super(message);
        this.name = 'Error loading MediaPipe model.';
    }
};

if (typeof module !== 'undefined' && typeof module.exports === 'object') {
    module.exports.ModelFactory = mediapipe.ModelFactory;
}