/* jshint esversion: 6 */

var mediapipe = mediapipe || {};
var protobuf = protobuf || require('./protobuf');

mediapipe.ModelFactory = class {

    match(context) {
        const tags = context.tags('pbtxt');
        if (tags.has('node') && ['input_stream', 'output_stream', 'input_side_packet', 'output_side_packet'].some((key) => tags.has(key) || tags.has('node.' + key))) {
            return true;
        }
        return false;
    }

    open(context, host) {
        return Promise.resolve().then(() => {
            let root;
            try {
                const reader = protobuf.TextReader.create(context.buffer);
                root = new mediapipe.Object(reader);
            }
            catch (error) {
                const message = error && error.message ? error.message : error.toString();
                throw new mediapipe.Error('File text format is not mediapipe.CalculatorGraphConfig (' + message.replace(/\.$/, '') + ').');
            }
            return new mediapipe.Model(root);
        });
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
                    const parts = input.split(':');
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
                    const parts = output.split(':');
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
                    const parts = input.split(':');
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
                    const parts = output.split(':');
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
            const args = [];
            const inputs = Array.isArray(node.input_stream) ? node.input_stream : [ node.input_stream ];
            for (const input of inputs) {
                const parts = input.split(':');
                const type = (parts.length > 1) ? parts.shift() : '';
                const name = parts.shift();
                args.push(new mediapipe.Argument(name, type, null));
            }
            this._inputs.push(new mediapipe.Parameter('input_stream', args));
        }
        if (node.output_stream) {
            const args = [];
            const outputs = Array.isArray(node.output_stream) ? node.output_stream : [ node.output_stream ];
            for (const output of outputs) {
                const parts = output.split(':');
                const type = (parts.length > 1) ? parts.shift() : '';
                const name = parts.shift();
                args.push(new mediapipe.Argument(name, type, null));
            }
            this._outputs.push(new mediapipe.Parameter('output_stream', args));
        }
        if (node.input_side_packet) {
            const args = [];
            const inputs = Array.isArray(node.input_side_packet) ? node.input_side_packet : [ node.input_side_packet ];
            for (const input of inputs) {
                const parts = input.split(':');
                const type = (parts.length > 1) ? parts.shift() : '';
                const name = parts.shift();
                args.push(new mediapipe.Argument(name, type, null));
            }
            this._inputs.push(new mediapipe.Parameter('input_side_packet', args));
        }
        if (node.output_side_packet) {
            const args = [];
            const outputs = Array.isArray(node.output_side_packet) ? node.output_side_packet : [ node.output_side_packet ];
            for (const output of outputs) {
                const parts = output.split(':');
                const type = (parts.length > 1) ? parts.shift() : '';
                const name = parts.shift();
                args.push(new mediapipe.Argument(name, type, null));
            }
            this._outputs.push(new mediapipe.Parameter('output_side_packet', args));
        }
        const options = node.options || node.node_options || null;
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

    get type() {
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
        const type = reader.token();
        if (type.startsWith('[') && type.endsWith(']')) {
            this.__type__ = type.substring(1, type.length - 1);
            reader.next();
            reader.match(':');
            reader.start();
            close = true;
        }
        const arrayTags = new Set();
        while (!reader.end()) {
            const tag = reader.tag();
            const next = reader.token();
            let obj = null;
            if (next === '{') {
                obj = new mediapipe.Object(reader);
            }
            else if (next.startsWith('"') && next.endsWith('"')) {
                obj = next.substring(1, next.length - 1);
                reader.next();
            }
            else if (next === 'true' || next === 'false') {
                obj = next;
                reader.next();
            }
            else if (reader.first()) {
                obj = [];
                while (!reader.last()) {
                    const data = reader.token();
                    reader.next();
                    if (!isNaN(data)) {
                        obj.push(parseFloat(data));
                    }
                }
            }
            else if (!isNaN(next)) {
                obj = parseFloat(next);
                reader.next();
            }
            else {
                obj = next;
                reader.next();
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