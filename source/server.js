
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
        this.format = data.format || '';
        this.producer = data.producer || '';
        this.version = data.version || '';
        this.description = data.description || '';
        this.metadata = (data.metadata || []).map((entry) => {
            return { name: entry.name, value: entry.value };
        });
        this.graphs = (data.graphs || []).map((graph) => new message.Graph(graph));
    }
};

message.Graph = class {

    constructor(data) {
        this.inputs = [];
        this.outputs = [];
        this.nodes = [];
        const values = data.values ? data.values.map((value) => new message.Value(value)) : [];
        for (const argument of data.inputs || []) {
            argument.value = argument.value.map((index) => values[index]).filter((argument) => !argument.initializer);
            if (argument.value.filter((argument) => !argument.initializer).length > 0) {
                this.inputs.push(new message.Argument(argument));
            }
        }
        for (const argument of data.outputs || []) {
            argument.value = argument.value.map((index) => values[index]);
            if (argument.value.filter((argument) => !argument.initializer).length > 0) {
                this.outputs.push(new message.Argument(argument));
            }
        }
        for (const node of data.nodes || []) {
            for (const argument of node.inputs || []) {
                argument.value = argument.value.map((index) => values[index]);
            }
            for (const argument of node.outputs || []) {
                argument.value = argument.value.map((index) => values[index]);
            }
            this.nodes.push(new message.Node(node));
        }
    }
};

message.Argument = class {

    constructor(data) {
        this.name = data.name || '';
        this.value = data.value || [];
        this.type = data.type || '';
    }
};

message.Value = class {

    constructor(data) {
        this.name = data.name || '';
        this.initializer = data.initializer ? new message.Tensor(data.initializer) : null;
        if (this.initializer && this.initializer.type) {
            this.type = this.initializer.type;
        } else {
            this.type = data.type ? new message.TensorType(data.type) : null;
        }
    }
};

message.Node = class {

    constructor(data) {
        this.type = { name: data.type.name, category: data.type.category };
        this.name = data.name;
        this.inputs = (data.inputs || []).map((input) => new message.Argument(input));
        this.outputs = (data.outputs || []).map((output) => new message.Argument(output));
        this.attributes = (data.attributes || []).map((attribute) => new message.Argument(attribute));
    }
};

message.TensorType = class {

    constructor(data) {
        this.dataType = data.dataType;
        this.shape = new message.TensorShape(data.shape);
    }

    toString() {
        return this.dataType + this.shape.toString();
    }
};

message.TensorShape = class {

    constructor(data) {
        this.dimensions = data.dimensions;
    }

    toString() {
        return `[${this.dimensions}]`;
    }
};

message.Tensor = class {

    constructor(data) {
        this.type = new message.TensorType(data.type);
    }
};

message.Error = class extends Error {
    constructor(message) {
        super(message);
        this.name = 'Message Error';
    }
};

export const ModelFactory = message.ModelFactory;
