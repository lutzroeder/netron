
// Experimental

const message = {};

message.ModelFactory = class {

    async match(context) {
        const stream = context.stream;
        if (stream) {
            const buffer = stream.peek(Math.min(64, stream.length));
            const content = String.fromCharCode.apply(null, buffer);
            const match = content.match(/^{\s*"signature":\s*"(.*)"\s*,\s*/);
            if (match && match[1].startsWith('netron:')) {
                const obj = await context.peek('json');
                if (obj && obj.signature && obj.signature.startsWith('netron:')) {
                    return context.set('message', obj);
                }
            }
        }
        return null;
    }

    async open(context) {
        return new message.Model(context.value);
    }
};

message.Model = class {

    constructor(data) {
        this.format = data.format || '';
        this.format = this.format.replace(/\s+(\d+\.\d+)$/, ' v$1'); // Format v2.0
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
        const values = new Map();
        values.map = (index) => {
            if (!values.has(index)) {
                values.set(index, new message.Value({ name: index.toString() }));
            }
            return values.get(index);
        };
        if (Array.isArray(data.values)) {
            for (let i = 0; i < data.values.length; i++) {
                values.set(i, new message.Value(data.values[i]));
            }
        }
        if (Array.isArray(data.arguments)) {
            for (let i = 0; i < data.arguments.length; i++) {
                values.set(data.arguments[i].name, new message.Value(data.arguments[i]));
            }
        }
        for (const argument of data.inputs || []) {
            argument.value = argument.value.map((index) => values.map(index)).filter((argument) => !argument.initializer);
            if (argument.value.filter((argument) => !argument.initializer).length > 0) {
                this.inputs.push(new message.Argument(argument));
            }
        }
        for (const argument of data.outputs || []) {
            argument.value = argument.value.map((index) => values.map(index));
            if (argument.value.filter((argument) => !argument.initializer).length > 0) {
                this.outputs.push(new message.Argument(argument));
            }
        }
        for (const node of data.nodes || []) {
            for (const argument of node.inputs || []) {
                if (!argument.value && argument.arguments) {
                    argument.value = argument.arguments;
                    delete argument.arguments;
                }
                argument.value = argument.value.map((index) => values.map(index));
            }
            for (const argument of node.outputs || []) {
                if (!argument.value && argument.arguments) {
                    argument.value = argument.arguments;
                    delete argument.arguments;
                }
                argument.value = argument.value.map((index) => values.map(index));
            }
            this.nodes.push(new message.Node(node));
        }
    }
};

message.Argument = class {

    constructor(data) {
        this.name = data.name || '';
        this.value = data.value || [];
        this.type = data.type || null;
    }
};

message.Value = class {

    constructor(data) {
        this.name = data.name ? data.name.toString() : '';
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
        this.name = data.name || '';
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
