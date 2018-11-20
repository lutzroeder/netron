class OpenVINOIRGraph {
    constructor(netDef, init) {
        this._name = netDef.net.name || '';
        this._batch = +netDef.net.batch || '';
        this._version = +netDef.net.version || '';

        this._nodes = [];
        this._operators = {};
        this._inputs = [];
        this._outputs = [];

        _.each(netDef.layers, (layer) => {
            const node = new OpenVINOIRNode(layer, this._version, netDef.edges, netDef.layers);
            this._operators[node.operator] = _.get(this._operators, node.operator, 0) + 1;
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
