
// Experimental

const message = {};

message.ModelFactory = class {

    match(context) {
        const stream = context.stream;
        if (stream) {
            const buffer = stream.peek(Math.min(64, stream.length));
            const content = String.fromCharCode.apply(null, buffer);
            const match = content.match(/^{\s*"signature":\s*"(.*)"\s*,\s*/);
            if (match && match[1].startsWith('netron:')) {
                const obj = context.peek('json');
                if (obj && obj.signature && obj.signature.startsWith('netron:')) {
                    context.type = 'message';
                    context.target = obj;
                }
            }
        }
        return null;
    }

    async open(context) {
        return new message.Model(context.target);
    }
};

message.Model = class {

    constructor(data) {
        this._format = data.format || '';
        this._producer = data.producer || '';
        this._version = data.version || '';
        this._description = data.description || '';
        this._metadata = (data.metadata || []).map((entry) => {
            return { name: entry.name, value: entry.value };
        });
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
        this._inputs = [];
        this._outputs = [];
        this._nodes = [];
        const args = data.arguments ? data.arguments.map((argument) => new message.Value(argument)) : [];
        for (const parameter of data.inputs || []) {
            parameter.arguments = parameter.arguments.map((index) => args[index]).filter((argument) => !argument.initializer);
            if (parameter.arguments.filter((argument) => !argument.initializer).length > 0) {
                this._inputs.push(new message.Argument(parameter));
            }
        }
        for (const parameter of data.outputs || []) {
            parameter.arguments = parameter.arguments.map((index) => args[index]);
            if (parameter.arguments.filter((argument) => !argument.initializer).length > 0) {
                this._outputs.push(new message.Argument(parameter));
            }
        }
        for (const node of data.nodes || []) {
            for (const parameter of node.inputs || []) {
                parameter.arguments = parameter.arguments.map((index) => args[index]);
            }
            for (const parameter of node.outputs || []) {
                parameter.arguments = parameter.arguments.map((index) => args[index]);
            }
            this._nodes.push(new message.Node(node));
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

message.Argument = class {

    constructor(data) {
        this._name = data.name || '';
        this._value = (data.arguments || []);
    }

    get name() {
        return this._name;
    }

    get value() {
        return this._value;
    }
};

message.Value = class {

    constructor(data) {
        this._name = data.name || '';
        this._type = data.type ? new message.TensorType(data.type) : null;
        this._initializer = data.initializer ? new message.Tensor(data.initializer) : null;
    }

    get name() {
        return this._name;
    }

    get type() {
        if (this._initializer && this._initializer.type) {
            return this._initializer.type;
        }
        return this._type;
    }

    get initializer() {
        return this._initializer;
    }
};

message.Node = class {

    constructor(data) {
        this._type = { name: data.type.name, category: data.type.category };
        this._name = data.name;
        this._inputs = (data.inputs || []).map((input) => new message.Argument(input));
        this._outputs = (data.outputs || []).map((output) => new message.Argument(output));
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

    constructor(data) {
        this._type = data.type || '';
        this._name = data.name;
        this._value = data.value;
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

message.TensorType = class {

    constructor(data) {
        this._dataType = data.dataType;
        this._shape = new message.TensorShape(data.shape);
    }

    get dataType() {
        return this._dataType;
    }

    get shape() {
        return this._shape;
    }

    toString() {
        return this._dataType + this._shape.toString();
    }
};

message.TensorShape = class {

    constructor(data) {
        this._dimensions = data.dimensions;
    }

    get dimensions() {
        return this._dimensions;
    }

    toString() {
        return `[${this._dimensions}]`;
    }
};

message.Tensor = class {
};

message.Error = class extends Error {
    constructor(message) {
        super(message);
        this.name = 'Message Error';
    }
};

export const ModelFactory = message.ModelFactory;
