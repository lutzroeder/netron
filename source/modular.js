
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
        this._graphs = obj.graphs.map((graph) => new modular.Graph(graph));
    }

    get format() {
        return 'Modular';
    }

    get graphs() {
        return this._graphs;
    }
};

modular.Graph = class {

    constructor(graph) {
        this._nodes = Array.from(graph.nodes.map((node) => new modular.Node(node)));
        this._inputs = [];
        this._outputs = [];
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

modular.Argument = class {

    constructor(name, value) {
        this._name = name;
        this._value = Array.from(value.map((value) => new modular.Value(value.toString(), name)));
    }

    get name() {
        return this._name;
    }

    get value() {
        return this._value;
    }
};

modular.Value = class {

    constructor(name, value) {
        if (typeof name !== 'string') {
            throw new modular.Error(`Invalid value identifier '${JSON.stringify(name)}'.`);
        }
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

modular.Node = class {

    constructor(node) {
        this._name = node.type.name;
        if (node.type.category === 'List') {
            this._category = 'Data';
        } else if (node.type.category === 'ControlFlow') {
            this._category = 'Control';
        } else {
            this._category = node.type.category;
        }
        this._type = { name: this._name, category: this._category };
        this._attributes = node.attributes ?
            Array.from(node.attributes).map((attribute) => new modular.Attribute(attribute.name, attribute.value)) :
            [];
        this._inputs = Array.from(node.inputs.map((input) => new modular.Argument(input.name, input.arguments)));
        this._outputs = Array.from(node.outputs.map((output) => new modular.Argument(output.name, output.arguments)));
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

modular.Attribute = class {

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

modular.Error = class extends Error {

    constructor(message) {
        super(message);
        this.name = 'Error loading Modular model.';
    }
};

export const ModelFactory = modular.ModelFactory;
