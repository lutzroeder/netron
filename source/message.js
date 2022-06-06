
// Experimental

var message = message || {};

message.ModelFactory = class {

    match(context) {
        const obj = context.open('json');
        if (obj && obj.signature && obj.signature.startsWith('netron:')) {
            return obj.signature;
        }
        return '';
    }

    open(context) {
        return Promise.resolve().then(() => {
            const obj = context.open('json');
            return new message.Model(obj);
        });
    }
};

message.Model = class {

    constructor(data) {
        this._format = data.format || '';
        this._producer = data.producer || '';
        this._version = data.version || '';
        this._description = data.description || '';
        this._metadata = (data.metadata || []).map((entry) => { return { name: entry.name, value: entry.value }; });
        this._graphs = (data.graphs || []).map((graph) => new message.Graph(graph));
    }

    get format() {
        return this._format;
    }

    get producer() {
        return this._producer;
    }

    get version() {
        return this._version;
    }

    get description() {
        return this._description;
    }

    get metadata() {
        return this._metadata;
    }

    get graphs() {
        return this._graphs;
    }
};

message.Graph = class {

    constructor(data) {
        this._nodes = (data.nodes || []).map((node) => new message.Node(node));
        this._inputs = (data.inputs || []).map((input) => new message.Parameter(input));
        this._outputs = (data.outputs || []).map((output) => new message.Parameter(output));
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

message.Parameter = class {

    constructor(data) {
        this._name = data.name;
        this._arguments = (data.arguments || []).map((argument) => new message.Argument(argument));
    }

    get name() {
        return this._name;
    }

    get arguments() {
        return this._arguments;
    }

    get visible() {
        return true;
    }
};

message.Argument = class {

    constructor(data) {
        this._name= data.name || '';
        this._type = data.type ? new message.TensorType(data.type) : null;
        this._initializer = data.initializer ? new message.Tensor(data.initializer) : null;
    }

    get name() {
        return this._name;
    }

    get type() {
        if (this._initializer) {
            return this._initializer.type;
        }
        return this._type;
    }

    set type(value) {
        this._type = value;
    }

    get initializer() {
        return this._initializer;
    }
};

message.Node = class {

    constructor(data) {
        this._type = { name: data.type.name, category: data.type.category };
        this._name = data.name;
        this._inputs = (data.inputs || []).map((input) => new message.Parameter(input));
        this._outputs = (data.outputs || []).map((output) => new message.Parameter(output));
        this._attributes = (data.attributes || []).map((attribute) => new message.Attribute(attribute));
    }

    get type() {
        return this._type;
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

    get attributes() {
        return this._attributes;
    }
};

message.Attribute = class {

    constructor(attribute) {
        this._name = attribute.name;
        this._value = attribute.value;
    }

    get name() {
        return this._name;
    }

    get value() {
        return this._value;
    }

    get type() {
        return this._type;
    }
};

message.Error = class extends Error {
    constructor(message) {
        super(message);
        this.name = 'Message Error';
    }
};

if (typeof module !== 'undefined' && typeof module.exports === 'object') {
    module.exports.ModelFactory = message.ModelFactory;
}