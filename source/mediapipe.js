
var mediapipe = {};
var protobuf = require('./protobuf');

mediapipe.ModelFactory = class {

    match(context) {
        const tags = context.tags('pbtxt');
        if (tags.has('node') && ['input_stream', 'output_stream', 'input_side_packet', 'output_side_packet'].some((key) => tags.has(key) || tags.has('node.' + key))) {
            return 'mediapipe.pbtxt';
        }
        return null;
    }

    open(context) {
        return Promise.resolve().then(() => {
        // return context.require('./mediapipe-proto').then(() => {
            mediapipe.proto = protobuf.get('mediapipe');
            let config = null;
            try {
                const stream = context.stream;
                const reader = protobuf.TextReader.open(stream);
                // const config = mediapipe.proto.mediapipe.CalculatorGraphConfig.decodeText(reader);
                config = new mediapipe.Object(reader);
            }
            catch (error) {
                const message = error && error.message ? error.message : error.toString();
                throw new mediapipe.Error('File text format is not mediapipe.CalculatorGraphConfig (' + message.replace(/\.$/, '') + ').');
            }
            return new mediapipe.Model(config);
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
        const type = node.calculator || '?';
        this._type = { name: type.replace(/Calculator$/, '') };
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
        const options = new Map();
        if (node.options) {
            for (const key of Object.keys(node.options)) {
                options.set(key, node.options[key]);
            }
        }
        const node_options = node.node_options ? Array.isArray(node.node_options) ? node.node_options : [ node.node_options ] : [];
        if (mediapipe.proto.google && node_options.every((options) => options instanceof mediapipe.proto.google.protobuf.Any)) {
            for (const entry of node_options) {
                const value = new RegExp(/^\{(.*)\}\s*$/, 's').exec(entry.value);
                const buffer = new TextEncoder('utf-8').encode(value[1]);
                const reader = protobuf.TextReader.open(buffer);
                if (entry.type_url.startsWith('type.googleapis.com/mediapipe.')) {
                    const type = entry.type_url.split('.').pop();
                    if (mediapipe.proto && mediapipe.proto.mediapipe && mediapipe.proto.mediapipe[type]) {
                        const message = mediapipe.proto.mediapipe[type].decodeText(reader);
                        for (const key of Object.keys(message)) {
                            options.set(key, message[key]);
                        }
                        continue;
                    }
                }
                const message = new mediapipe.Object(reader);
                for (const key of Object.keys(message)) {
                    options.set(key, message[key]);
                }
            }
        }
        else {
            for (const entry of node_options) {
                for (const key of Object.keys(entry)) {
                    if (key !== '__type__') {
                        options.set(key, entry[key]);
                    }
                }
            }
        }
        for (const pair of options) {
            this._attributes.push(new mediapipe.Attribute(pair[0], pair[1]));
        }
    }

    get name() {
        return '';
    }

    get type() {
        return this._type;
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

    constructor(reader, block) {
        if (!block) {
            reader.start();
        }
        const type = reader.token();
        if (type.startsWith('[') && type.endsWith(']')) {
            this.__type__ = type.substring(1, type.length - 1);
            reader.next();
            reader.match(':');
            reader.start();
        }
        const arrayTags = new Set();
        while (!reader.end()) {
            const tag = reader.tag();
            const next = reader.token();
            let obj = null;
            if (next === '{') {
                reader.start();
                obj = new mediapipe.Object(reader, true);
                if (obj.__type__) {
                    while (!reader.end()) {
                        if (!Array.isArray(obj)) {
                            obj = [ obj ];
                        }
                        const token = reader.token();
                        if (token.startsWith('[') && token.endsWith(']')) {
                            obj.push(new mediapipe.Object(reader, true));
                            continue;
                        }
                        break;
                    }
                }
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
            reader.match(',');
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