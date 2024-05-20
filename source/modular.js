
const modular = {};

modular.ModelFactory = class {

    match(context) {
        const obj = context.peek('json');
        if (obj && obj.signature === "netron:modular") {
            context.type = 'modular';
            context.target = obj;
        }
    }

    async open(context) {
        return new modular.Model(context.target);
    }
};

modular.Model = class {

    constructor(obj) {
        this.format = 'Modular';
        this.graphs = obj.graphs.map((graph) => new modular.Graph(graph));
    }
};

modular.Graph = class {

    constructor(graph) {
        this.nodes = Array.from(graph.nodes.map((node) => new modular.Node(node)));
        this.inputs = [];
        this.outputs = [];
    }
};

modular.Argument = class {

    constructor(name, value) {
        this.name = name;
        this.value = value;
    }
};

modular.Value = class {

    constructor(name, value) {
        if (typeof name !== 'string') {
            throw new modular.Error(`Invalid value identifier '${JSON.stringify(name)}'.`);
        }
        this.name = name;
        this.value = value;
    }
};

modular.Node = class {

    constructor(node) {
        this.name = '';
        this.type = { name: node.type.name };
        if (node.type.category === 'List') {
            this.type.category = 'Data';
        } else if (node.type.category === 'ControlFlow') {
            this.type.category = 'Control';
        } else {
            this.type.category = node.type.category;
        }
        this.attributes = Array.isArray(node.attributes) ? Array.from(node.attributes).map((attribute) => new modular.Argument(attribute.name, attribute.value)) : [];
        this.inputs = Array.from(node.inputs.map((input) => new modular.Argument(input.name, Array.from(input.arguments.map((value) => new modular.Value(value.toString(), input.name))))));
        this.outputs = Array.from(node.outputs.map((output) => new modular.Argument(output.name, Array.from(output.arguments.map((value) => new modular.Value(value.toString(), output.name))))));
    }
};

modular.Error = class extends Error {

    constructor(message) {
        super(message);
        this.name = 'Error loading Modular model.';
    }
};

export const ModelFactory = modular.ModelFactory;
