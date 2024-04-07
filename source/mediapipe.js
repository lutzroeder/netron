
import * as protobuf from './protobuf.js';

const mediapipe = {};

mediapipe.ModelFactory = class {

    match(context) {
        const tags = context.tags('pbtxt');
        if (tags.has('node') && ['input_stream', 'output_stream', 'input_side_packet', 'output_side_packet'].some((key) => tags.has(key) || tags.has(`node.${key}`))) {
            context.type = 'mediapipe.pbtxt';
        }
    }

    async open(context) {
        // mediapipe.proto = await context.require('./mediapipe-proto');
        mediapipe.proto = {};
        let config = null;
        try {
            const reader = context.read('protobuf.text');
            // const config = mediapipe.proto.mediapipe.CalculatorGraphConfig.decodeText(reader);
            config = new mediapipe.Object(reader);
        } catch (error) {
            const message = error && error.message ? error.message : error.toString();
            throw new mediapipe.Error(`File text format is not mediapipe.CalculatorGraphConfig (${message.replace(/\.$/, '')}).`);
        }
        return new mediapipe.Model(config);
    }
};

mediapipe.Model = class {

    constructor(config) {
        this.format = 'MediaPipe';
        this.graphs = [new mediapipe.Graph(config)];
    }
};

mediapipe.Graph = class {

    constructor(config) {
        config = config || {};
        this.inputs = [];
        this.outputs = [];
        this.nodes = [];
        const types = new Map();
        const type = (list) => {
            if (!Array.isArray(list)) {
                list = list ? [list] : [];
            }
            return list.map((item) => {
                const parts = item.split(':');
                const name = parts.pop();
                const type = parts.join(':');
                if (!types.has(name)) {
                    const value = new Set();
                    if (type) {
                        value.add(type);
                    }
                    types.set(name, value);
                } else if (type && !types.get(name).has(type)) {
                    types.get(name).add(type);
                }
                return name;
            });
        };
        config.input_stream = type(config.input_stream);
        config.output_stream = type(config.output_stream);
        config.input_side_packet = type(config.input_side_packet);
        config.output_side_packet = type(config.output_side_packet);
        if (!Array.isArray(config.node)) {
            config.node = config.node ? [config.node] : [];
        }
        for (const node of config.node) {
            node.input_stream = type(node.input_stream);
            node.output_stream = type(node.output_stream);
            node.input_side_packet = type(node.input_side_packet);
            node.output_side_packet = type(node.output_side_packet);
        }
        const values = new Map();
        for (const [name, value] of types) {
            const type = Array.from(value).join(',');
            values.set(name, new mediapipe.Value(name, type || null));
        }
        const value = (name) => {
            return values.get(name);
        };
        for (const name of config.input_stream) {
            const argument = new mediapipe.Argument(name, [value(name)]);
            this.inputs.push(argument);
        }
        for (const name of config.output_stream) {
            const argument = new mediapipe.Argument(name, [value(name)]);
            this.outputs.push(argument);
        }
        for (const name of config.input_side_packet) {
            const argument = new mediapipe.Argument(name, [value(name, type)]);
            this.inputs.push(argument);
        }
        for (const output of config.output_side_packet) {
            const parts = output.split(':');
            const type = (parts.length > 1) ? parts.shift() : '';
            const name = parts.shift();
            const argument = new mediapipe.Argument(name, [value(name, type)]);
            this.outputs.push(argument);
        }
        for (const node of config.node) {
            this.nodes.push(new mediapipe.Node(node, value));
        }
    }
};

mediapipe.Node = class {

    constructor(node, value) {
        const type = node.calculator || '?';
        this.name = '';
        this.type = { name: type.replace(/Calculator$/, '') };
        this.inputs = [];
        this.outputs = [];
        this.attributes = [];
        if (node.input_stream) {
            const values = node.input_stream.map((name) => value(name));
            const argument = new mediapipe.Argument('input_stream', values);
            this.inputs.push(argument);
        }
        if (node.output_stream) {
            const values = node.output_stream.map((name) => value(name));
            this.outputs.push(new mediapipe.Argument('output_stream', values));
        }
        if (node.input_side_packet) {
            const values = node.input_side_packet.map((name) => value(name));
            this.inputs.push(new mediapipe.Argument('output_stream', values));
        }
        if (node.output_side_packet) {
            const values = node.output_side_packet.map((name) => value(name));
            this.outputs.push(new mediapipe.Argument('output_side_packet', values));
        }
        const options = new Map();
        if (node.options) {
            for (const key of Object.keys(node.options)) {
                options.set(key, node.options[key]);
            }
        }
        let node_options = node.node_options;
        if (!Array.isArray(node_options)) {
            node_options = node_options ? [node_options] : [];
        }
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
                for (const [name, value] of Object.entries(message)) {
                    options.set(name, value);
                }
            }
        } else {
            for (const option of node_options) {
                for (const [name, value] of Object.entries(option)) {
                    if (name !== '__type__') {
                        options.set(name, value);
                    }
                }
            }
        }
        for (const [name, value] of options) {
            const attribute = new mediapipe.Argument(name, value);
            this.attributes.push(attribute);
        }
    }
};

mediapipe.Argument = class {

    constructor(name, value) {
        this.name = name;
        this.value = value;
    }
};

mediapipe.Value = class {

    constructor(name, type) {
        if (typeof name !== 'string') {
            throw new mediapipe.Error(`Invalid value identifier '${JSON.stringify(name)}'.`);
        }
        this.name = name;
        this.type = type || null;
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
                            obj = [obj];
                        }
                        const token = reader.token();
                        if (token.startsWith('[') && token.endsWith(']')) {
                            obj.push(new mediapipe.Object(reader, true));
                            continue;
                        }
                        break;
                    }
                }
            } else if (next.startsWith('"') && next.endsWith('"')) {
                obj = next.substring(1, next.length - 1);
                reader.next();
            } else if (next === 'true' || next === 'false') {
                obj = next;
                reader.next();
            } else if (reader.first()) {
                obj = [];
                while (!reader.last()) {
                    const data = reader.token();
                    reader.next();
                    if (!isNaN(data)) {
                        obj.push(parseFloat(data));
                    }
                }
            } else if (isNaN(next)) {
                obj = next;
                reader.next();
            } else {
                obj = parseFloat(next);
                reader.next();
            }
            if (this[tag] && (!Array.isArray(this[tag]) || arrayTags.has(tag))) {
                this[tag] = [this[tag]];
                arrayTags.delete(tag);
            }
            if (this[tag]) {
                this[tag].push(obj);
            } else {
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

export const ModelFactory = mediapipe.ModelFactory;
