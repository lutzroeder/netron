/* jshint esversion: 6 */

var rknn = rknn || {};
var json = json || require('./json');

rknn.ModelFactory = class {

    match(context) {
        const buffer = context.buffer;
        if (buffer && buffer.length > 4 && [ 0x52, 0x4B, 0x4E, 0x4E ].every((value, index) => buffer[index] === value)) {
            return true;
        }
        return false;
    }

    open(context, host) {
        return Promise.resolve().then(() => {
            const container = new rknn.Container(context.buffer);
            return new rknn.Model(container.configuration);
        });
    }
};

rknn.Model = class {

    constructor(configuration) {
        this._version = configuration.version;
        this._producer = configuration.ori_network_platform || configuration.network_platform || '';
        this._runtime = configuration.target_platform ? configuration.target_platform.join(',') : '';
        this._graphs = [ new rknn.Graph(configuration) ];
    }

    get format() {
        return 'RKNN v' + this._version;
    }

    get producer() {
        return this._producer;
    }

    get runtime() {
        return this._runtime;
    }

    get graphs() {
        return this._graphs;
    }
};

rknn.Graph = class {

    constructor(configuration) {
        this._name = configuration.name || '';
        this._inputs = [];
        this._outputs = [];
        this._nodes = [];

        for (const node of configuration.nodes) {
            this._nodes.push(new rknn.Node(node));
        }
    }

    get name() {
        return this._name;
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

rknn.Node = class {

    constructor(node) {
        this._name = node.name || '';
        this._type = node.op;
        this._inputs = [];
        this._outputs = [];
        this._attributes = [];

        if (node.nn) {
            const nn = node.nn;
            for (const key of Object.keys(nn)) {
                const params = nn[key];
                for (const name of Object.keys(params)) {
                    const value = params[name];
                    this._attributes.push(new rknn.Attribute(name, value));
                }
            }
        }
    }

    get name() {
        return this._name;
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

rknn.Attribute = class {

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
};

rknn.Container = class {

    constructor(buffer) {
        const view = new DataView(buffer.buffer, buffer.byteOffset, buffer.byteLength);
        const signature = view.getUint64(0, true);
        if (signature.low !== 0x4e4e4b52 || signature.high !== 0) {
            throw new rknn.Error('Invalid RKNN signature.');
        }
        this._version = view.getUint64(8, true).toNumber();
        const blocks = [];
        let position = 16;
        while (position < buffer.length) {
            const size = view.getUint64(position, true).toNumber();
            position += 8;
            blocks.push(buffer.subarray(position, position + size));
            position += size;
        }
        const reader = json.TextReader.create(blocks[1]);
        this._configuration = reader.read();
    }

    get version() {
        return this._version;
    }

    get configuration() {
        return this._configuration;
    }
};

rknn.Error = class extends Error {

    constructor(message) {
        super(message);
        this.name = 'Error loading RKNN model.';
    }
};

if (typeof module !== 'undefined' && typeof module.exports === 'object') {
    module.exports.ModelFactory = rknn.ModelFactory;
}
