var openvinoDot = openvinoDot || {};

if (window.require) {
    openvinoDot.Node = openvinoDot.Node || require('./openvino-dot-node').Node;
}

openvinoDot.Graph = class {
    constructor(netDef, init) {
        this._name = netDef.id || '';
        this._version = Boolean(netDef.strict).toString();

        this._nodes = [];
        this._operators = {};
        this._inputs = [];
        this._outputs = [];

        const layers = netDef.children.filter((child) => child.type === "node_stmt");
        const edges = netDef.children.filter((child) => child.type === "edge_stmt");

        layers.forEach((layer) => {
            const node = new openvinoDot.Node(layer, this._version, edges, layers);
            this._operators[node.operator] = this._operators[node.operator] ? this._operators[node.operator] + 1 : 1;
            this._nodes.push(node);
        });

        edges.forEach((edge) => {
            const from = edge.edge_list[0];
            const to = edge.edge_list[1];
            const child = this._nodes.find((node) => node._id === to.id);
            if (child) {
                child.updateInputs(from.id);
            }
            const parent = this._nodes.find((node) => node._id === from.id);
            if (parent) {
                parent.updateOutputs(to.id);
            }
        });
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

    get operators() {
        return this._operators;
    }
}

if (typeof module !== 'undefined' && typeof module.exports === 'object') {
    module.exports.Graph = openvinoDot.Graph;
}
