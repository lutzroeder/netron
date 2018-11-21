var openvinoIR = openvinoIR || {};

if (window.require) {
    openvinoIR.Node = openvinoIR.Node || require('./openvino-ir-node').Node;
}

openvinoIR.Graph = class {
    constructor(netDef, init) {
        this._name = netDef.net.name || '';
        this._batch = +netDef.net.batch || '';
        this._version = +netDef.net.version || '';

        this._nodes = [];
        this._operators = {};
        this._inputs = [];
        this._outputs = [];

        netDef.layers.forEach((layer) => {
            const node = new openvinoIR.Node(layer, this._version, netDef.edges, netDef.layers);
            this._operators[node.operator] = this._operators[node.operator] ? this._operators[node.operator] + 1 : 1;
            this._nodes.push(node);
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
    module.exports.Graph = openvinoIR.Graph;
}
