/* jshint esversion: 6 */
/* eslint "indent": [ "error", 4, { "SwitchCase": 1 } ] */

var openvino = openvino || {};
var marked = marked || require('marked');

openvino.ModelFactory = class {

    match(context) {
        const extension = context.identifier.split('.').pop().toLowerCase();
        if (extension === 'xml') {
            if (context.text.includes('<net')) {
                return true;
            }
        }
        return false;
    }

    open(context, host) {
        return openvino.Metadata.open(host).then((metadata) => {
            const identifier = context.identifier;
            try {
                var errors = false;
                const parser = new DOMParser({ errorHandler: () => { errors = true; } });
                const xml = parser.parseFromString(context.text, 'text/xml');
                if (errors || xml.documentElement == null || xml.getElementsByTagName('parsererror').length > 0) {
                    throw new openvino.Error("File format is not OpenVINO.");
                }
                const net = xml.documentElement;
                if (!net || net.nodeName != 'net' ||
                    openvino.Node.children(net, 'layers').length != 1 ||
                    openvino.Node.children(net, 'edges').length != 1) {
                    throw new openvino.Error("File format is not OpenVINO IR.");
                }
                return new openvino.Model(metadata, net);
            } catch (error) {
                host.exception(error, false);
                var message = error && error.message ? error.message : error.toString();
                message = message.endsWith('.') ? message.substring(0, message.length - 1) : message;
                throw new openvino.Error(message + " in '" + identifier + "'.");
            }
        });
    }
};

openvino.Model = class {

    constructor(metadata, net) {
        var graph = new openvino.Graph(metadata, net);
        this._graphs = [ graph ];
    }

    get format() {
        return 'OpenVINO IR';
    }

    get graphs() {
        return this._graphs;
    }

};

openvino.Graph = class {

    constructor(metadata, net) {
        this._metadata = metadata;
        this._name = net.getAttribute('name') || '';
        this._batch = net.getAttribute('batch') || '';
        this._version = net.getAttribute('version') || '';

        this._nodes = [];
        this._operators = {};
        this._inputs = [];
        this._outputs = [];

        this._arguments = {};

        var layersElement = openvino.Node.children(net, 'layers')[0];
        var edgesElement = openvino.Node.children(net, 'edges')[0];

        var layers = openvino.Node.children(layersElement, 'layer');
        var edges = openvino.Node.children(edgesElement, 'edge');

        var edgeMap = this._collectEdges(edges);

        for (var layer of layers) {
            var operator = layer.getAttribute('type');
            switch (operator) {
                case 'Input':
                    var args = [];
                    var precision = layer.getAttribute('precision');
                    var name = layer.getAttribute('name') || '';
                    var id = layer.getAttribute('id');
                    for (var outputElement of openvino.Node.children(layer, 'output')) {
                        for (var portElement of openvino.Node.children(outputElement, 'port')) {
                            args.push(this._argument(id, precision, portElement, null));
                        }
                    }
                    this._inputs.push(new openvino.Parameter(name, args));
                    break;
                default:
                    this._nodes.push(new openvino.Node(this, this._metadata, layer, edgeMap));
                    break;
            }
        }

        this._replaceTensorIteratorWithSubgraph(layers, edges);
        delete this._arguments;

        // Validation
        // all graph elements are split between inputs and nodes
        // by definition IR is a graph can have inputs of two types: "Input" and "Const"
        // "Input" layers are already moved to inputs when we parse a graph
        // if there are any layers that do not have input arguments and they are no Const ones
        // this means that this graph was not properly processed by the graph building logic
        const allNodesOutputs = this._nodes.reduce((acc, node) => {
            const nodesRes = this._collectConnectionsIds(node._outputs);
            acc = acc.concat(nodesRes);
            return acc;
        }, []);
        const allInputsOutputs = this._collectConnectionsIds(this._inputs);
        const outputSet = new Set([...allNodesOutputs, ...allInputsOutputs]);
        const nodesWithNonExistentInputs = this._nodes.reduce((acc, node) => {
            const nodesInputs = this._collectConnectionsIds(node._inputs);
            if (nodesInputs.filter((value) => !outputSet.has(value)).length > 0) {
                acc.push(node);
            }
            return acc;
        }, []);
        if (nodesWithNonExistentInputs.length !== 0){
            throw new openvino.Error('Graph contains more than one connected component.');
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

    _argument(layer, precision, port, map) {
        var id = layer + ':' + port.getAttribute('id');
        if (map) {
            id = map[id];
        }
        var argument = this._arguments[id];
        if (!argument) {
            var dimensions = [];
            for (var dimElement of Array.prototype.slice.call(port.getElementsByTagName('dim'))) {
                dimensions.push(parseInt(dimElement.textContent.trim()));
            }
            var shape = (dimensions.length == 0) ? null : new openvino.TensorShape(dimensions);
            argument = new openvino.Argument(id, new openvino.TensorType(precision, shape), null);
        }
        return argument;
    }

    _replaceTensorIteratorWithSubgraph(layers, edges) {
        const tiNodes = layers.filter((node) => node.getAttribute('type') === 'TensorIterator');

        for (var singleTensorIteratorNode of tiNodes) {
            const singleTensorIteratorNodeId = singleTensorIteratorNode.getAttribute("id");
            const body = openvino.Node.children(singleTensorIteratorNode, 'body')[0];
            const layersContainer = openvino.Node.children(body, 'layers')[0];
            const edgesContainer = openvino.Node.children(body, 'edges')[0];
            const iteratorLayers = openvino.Node.children(layersContainer, 'layer');
            const iteratorEdges = openvino.Node.children(edgesContainer, 'edge');
            const iteratorEdgeMap = this._collectEdges(iteratorEdges);
            const iteratorBackEdgesContainer = openvino.Node.children(singleTensorIteratorNode, 'back_edges')[0];
            const iteratorBackEdges = openvino.Node.children(iteratorBackEdgesContainer, 'edge')
            const iteratorBackEdgesMap = this._collectEdges(iteratorBackEdges);
            const iteratorAllEdges = Object.assign({}, iteratorEdgeMap, iteratorBackEdgesMap);

            const mappingForNestedIR = this._parseMappingBlock(singleTensorIteratorNode);

            for (var nestedLayer of iteratorLayers) {
                const nestedNode = new openvino.Node(this, this._metadata, nestedLayer, iteratorAllEdges);
                nestedNode._id = `${singleTensorIteratorNodeId}_${nestedLayer.getAttribute('id')}`;
                for (var input of nestedNode._inputs) {
                    for (var input_argument of input.arguments) {
                        // we had a argument with id: 0:1  - meaning from layer "0" and its port "1"
                        // now as we rename all internal nodes to have an id of the TI included
                        // e.g. internal layer with id "0" and TI with id "14" results in internal layer to get id "14_0"
                        if (!input_argument._id){
                            return;
                        }
                        input_argument._id = `${singleTensorIteratorNodeId}_${input_argument._id}`;
                    }
                }

                for (var output of nestedNode._outputs) {
                    for (var output_argument of output.arguments) {
                        // we had a argument with id: 1:1  - meaning from me with id "1" and my port "1"
                        // now as we rename all internal nodes to have an id of the TI included
                        // e.g. my layer with id "1" and TI with id "14" results in internal layer to get id "14_1"
                        if (!output_argument._id){
                            return;
                        }
                        output_argument._id = `${singleTensorIteratorNodeId}_${output_argument._id}`;
                    }
                }
                
                this._nodes.push(nestedNode);
            }

            // We know for sure that edges that appeared in the nested IR are not
            // aware of the external context
            for (var nestedInput of mappingForNestedIR) {
                const nestedNode = this._nodes.find((n) => n._id === `${singleTensorIteratorNodeId}_${nestedInput.internal_layer_id}`);
                const candidate_edges = edges.filter((edge) => {
                    return edge.getAttribute('to-layer') === singleTensorIteratorNodeId && edge.getAttribute('to-port') === nestedInput.external_port_id;
                });
                var candidate_edge;
                if (!candidate_edges.length){
                    return;
                }
                for (candidate_edge of candidate_edges) {
                    const parentLayerID = candidate_edge.getAttribute('from-layer');
                    const parentPortID = candidate_edge.getAttribute('from-port');
                    if (!nestedNode._inputs){
                        throw new openvino.Error(`Tensor Iterator node with name ${nestedNode._name} does not have inputs.`);
                    }
                    const newId = `${parentLayerID}:${parentPortID}`;
                    const inputWithoutId = nestedNode._inputs.find((input) => {
                        return Boolean(input._arguments.find((argument) => !argument._id));
                    });
                    if (inputWithoutId) {
                        const argumentWithoutId = inputWithoutId._arguments.find((argument) => !argument._id);
                        if (argumentWithoutId){
                            argumentWithoutId._id = newId;
                        } 
                    } else {
                        // TODO: no tensor information in the new argument - passed as null for now
                        nestedNode._inputs.push(new openvino.Parameter((nestedNode._inputs.length+1).toString(), [
                            new openvino.Argument(newId, null, null)
                        ]));
                    }
                }
            }

            for (var nestedOutput of mappingForNestedIR.output) {
                const nestedNode = this._nodes.find((n) => n._id === `${singleTensorIteratorNodeId}_${nestedOutput.internal_layer_id}`);
                const candidate_edges = edges.filter((edge) => {
                    return edge.getAttribute('from-layer') === singleTensorIteratorNodeId && edge.getAttribute('from-port') === nestedOutput.external_port_id;
                });
                if (candidate_edges.length === 0){
                    return;
                }
                for (candidate_edge of candidate_edges) {
                    const childLayerID = candidate_edge.getAttribute('to-layer');
                    const child = this._nodes.find((layer) => layer._id === childLayerID);
                    if (!child._inputs || (child._inputs && child._inputs.length === 0)){
                        return;
                    }
                    for (var child_input of child._inputs) {
                        for (var argument of child_input._arguments) {
                            if (!argument._id || (argument._id && argument._id.split(':')[0] !== singleTensorIteratorNodeId)) {
                                return;
                            }
                            const myPort = nestedNode._outputs[0]._arguments[0]._id.split(':')[1];
                            argument._id = `${nestedNode.id}:${myPort}`;
                        }
                    }
                }
            }
            this._nodes = this._nodes.filter((node) => node._type !== 'TensorIterator');
        }
    }

    _collectEdges(edges){
        let edgeMap = {};
        for (var edge of edges) {
            var fromLayer = edge.getAttribute('from-layer');
            var fromPort = edge.getAttribute('from-port');
            var toLayer = edge.getAttribute('to-layer');
            var toPort = edge.getAttribute('to-port');
            edgeMap[toLayer + ':' + toPort] = fromLayer + ':' + fromPort;
        }
        return edgeMap;
    }

    _collectPortsInformation(ports){
        return ports.reduce((acc, port) => {
            acc.push({
                axis: port.getAttribute("axis"),
                external_port_id: port.getAttribute("external_port_id"),
                internal_layer_id: port.getAttribute("internal_layer_id"),
                internal_port_id: port.getAttribute("internal_port_id")
            });
            return acc;
        }, []);
    }

    _parseMappingBlock(singleTensorIteratorNode) {
        const portMap = openvino.Node.children(singleTensorIteratorNode, 'port_map')[0];
        const inputs = openvino.Node.children(portMap, 'input');
        const outputs = openvino.Node.children(portMap, 'output');
        return {
            input: this._collectPortsInformation(inputs),
            output: this._collectPortsInformation(outputs)
        };
    }

    _collectConnectionsIds(where) {
        return where.reduce((accOutput, output) => {
            const res = output._arguments.reduce((accConn, argument) => {
                accConn.push(argument._id);
                return accConn;
            }, []);
            accOutput = accOutput.concat(res);
            return accOutput;
        }, []);
    }
};

openvino.Node = class {

    constructor(graph, metadata, layer, edgeMap) {
        this._metadata = metadata;
        this._type = layer.getAttribute('type');
        this._name = layer.getAttribute('name') || '';
        this._id = layer.getAttribute('id');
        this._inputs = [];
        this._outputs = [];
        this._initializers = [];
        this._attributes = [];

        var precision = layer.getAttribute('precision');

        var inputIndex = 0;
        const input = openvino.Node.children(layer, 'input')[0];
        if (input) {
            for (var port of openvino.Node.children(input, 'port')) {
                var inputName = (inputIndex == 0) ? 'input' : inputIndex.toString(); 
                this._inputs.push(new openvino.Parameter(inputName, [
                    graph._argument(this._id, precision, port, edgeMap)
                ]));
                inputIndex++;
            }
        }

        var outputIndex = 0;
        const output = openvino.Node.children(layer, 'output')[0];
        if (output) {
            for (var portElement of openvino.Node.children(output, 'port')) {
                var outputName = (outputIndex == 0) ? 'output' : outputIndex.toString(); 
                this._outputs.push(new openvino.Parameter(outputName, [
                    graph._argument(this._id, precision, portElement, null)
                ]));
                outputIndex++;
            }
        }

        const data = openvino.Node.children(layer, 'data')[0];
        if (data && data.attributes) {
            for (var attribute of Array.from(data.attributes)) {
                this._attributes.push(new openvino.Attribute(metadata, this, attribute.name, attribute.value));
            }
        }

        const blobs = openvino.Node.children(layer, 'blobs')[0];
        if (blobs){
            for (var blob of Array.from(blobs.childNodes).filter((node) => node.nodeName != '#text')) {
                if (blob.getAttribute && typeof blob.getAttribute === 'function') {
                    var name = blob.nodeName;
                    var offset = parseInt(blob.getAttribute('offset'));
                    var size = parseInt(blob.getAttribute('size'));
                    this._initializers.push(new openvino.Parameter(name, [
                        new openvino.Argument('', null, new openvino.Tensor(precision, null, offset, size))
                    ]));
                }
            }
        }
    }

    get id() {
        return this._id;
    }

    get name() {
        return this._name;
    }

    get device() {
        return this._device || '';
    }

    get operator() {
        return this._type;
    }

    get category() {
        var schema = this._metadata.getSchema(this._type);
        return (schema && schema.category) ? schema.category : '';
    }

    get documentation() {
        var schema = this._metadata.getSchema(this._type);
        if (schema) {
            schema = JSON.parse(JSON.stringify(schema));
            schema.name = this._type;
            if (schema.description) {
                schema.description = marked(schema.description);
            }
            if (schema.attributes) {
                for (var attribute of schema.attributes) {
                    if (attribute.description) {
                        attribute.description = marked(attribute.description);
                    }
                }
            }
            if (schema.inputs) {
                for (var input of schema.inputs) {
                    if (input.description) {
                        input.description = marked(input.description);
                    }
                }
            }
            if (schema.outputs) {
                for (var output of schema.outputs) {
                    if (output.description) {
                        output.description = marked(output.description);
                    }
                }
            }
            if (schema.references) {
                for (var reference of schema.references) {
                    if (reference) {
                        reference.description = marked(reference.description);
                    }
                }
            }
            return schema;
        }
        return '';
    }

    get attributes() {
        return this._attributes;
    }

    get inputs() {
        return this._inputs.concat(this._initializers);
    }

    get outputs() {
        return this._outputs;
    }

    static children(element, name) {
        var children = [];
        var child = element.firstChild;
        while (child != null) {
            if (child.nodeType == 1 && child.nodeName == name) {
                children.push(child);
            }
            child = child.nextSibling;
        }
        return children;
    }
};

openvino.Parameter = class {

    constructor(name, args) {
        this._name = name;
        this._arguments = args;
    }

    get name() {
        return this._name;
    }

    get visible() {
        return true;
    }

    get arguments() {
        return this._arguments;
    }
};

openvino.Argument = class {

    constructor(id, type, initializer) {
        this._id = id;
        this._type = type || null;
        this._initializer = initializer || null;
    }

    get id() {
        return this._id;
    }

    get type() {
        if (this._initializer) {
            return this._initializer.type;
        }
        return this._type;
    }

    get initializer() {
        return this._initializer;
    }
};

openvino.Attribute = class {

    constructor(metadata, node, name, value) {
        this._node = node;
        this._name = name;
        this._value = value;

        var schema = metadata.getAttributeSchema(node.operator, name);
        if (schema) {
            if (Object.prototype.hasOwnProperty.call(schema, 'type')) {
                switch (schema.type) {
                    case 'boolean':
                        switch (value) {
                            case '1':
                            case 'true':
                                this._value = true;
                                break;
                            case '0':
                            case 'false':
                                this._value = false;
                                break;
                        }
                        break;
                    case 'int32':
                        var intValue = Number.parseInt(this._value, 10);
                        this._value = Number.isNaN(this._value - intValue) ? value : intValue;
                        break;
                    case 'float32':
                    case 'float64':
                        var floatValue = Number.parseFloat(this._value);
                        this._value = Number.isNaN(this._value - floatValue) ? value : floatValue;
                        break;
                    case 'int32[]':
                        if (this._value.length > 2) {
                            var ints = [];
                            this._value.split(',').map((item) => {
                                item = item.trim();
                                var intValue = Number.parseInt(item, 10);
                                if (Number.isNaN(item - intValue)) {
                                    ints = null;
                                }
                                else if (ints != null) {
                                    ints.push(intValue);
                                }
                            });
                            if (ints != null) {
                                this._value = ints;
                            }
                        }
                        break;
                    case 'float32[]':
                        if (this._value.length > 2) {
                            var floats = [];
                            this._value.split(',').map((item) => {
                                item = item.trim();
                                var floatValue = Number.parseFloat(item);
                                if (Number.isNaN(item - floatValue)) {
                                    floats = null;
                                }
                                else if (floats != null) {
                                    floats.push(floatValue);
                                }
                            });
                            if (floats != null) {
                                this._value = floats;
                            }
                        }
                        break;
                }
            }
            if (Object.prototype.hasOwnProperty.call(schema, 'visible') && schema.visible == false) {
                this._visible = false;
            }
            else if (Object.prototype.hasOwnProperty.call(schema, 'default')) {
                var defaultValue = schema.default;
                if (this._value == defaultValue) {
                    this._visible = false;
                }
                else if (Array.isArray(this._value) && Array.isArray(defaultValue)) {
                    defaultValue = defaultValue.slice(0, defaultValue.length);
                    if (defaultValue.length > 1 && defaultValue[defaultValue.length - 1] == null) {
                        defaultValue.pop();
                        while (defaultValue.length < this._value.length) {
                            defaultValue.push(defaultValue[defaultValue.length - 1]); 
                        }
                    }
                    if (this._value.every((item, index) => { return item == defaultValue[index]; })) {
                        this._visible = false;
                    }
                }
            }
        }
    }

    get name() {
        return this._name;
    }

    get value() {
        return this._value;
    }

    get visible() {
        return this._visible == false ? false : true;
    }
};

openvino.Tensor = class {

    constructor(precision, shape, offset, size) {
        this._data = null;
        this._reference = '{ offset: ' + offset.toString() + ', size: ' + size.toString() + ' }';
        this._shape = shape;
        this._type = new openvino.TensorType(precision, this._shape);
    }

    get type() {
        return this._type;
    }

    get kind() {
        return 'Blob';
    }

    get reference() {
        return this._reference;
    }

    get state() {
        return 'Tensor data is empty.';
    }

    get value() {
        return null;
    }

    toString() {
        return '';
    }
};

openvino.TensorType = class {

    constructor(precision, shape) {
        switch (precision) {
            case 'FP32':
                this._dataType = 'float32';
                break;
            default:
                this._dataType = precision;
                break;
        }
        this._shape = shape;
    }

    get dataType() {
        return this._dataType;
    }

    get shape() {
        return this._shape;
    }

    toString() {
        if (this._shape == null) {
            return this.dataType + '[?]';
        }
        return this.dataType + this._shape.toString();
    }
};

openvino.TensorShape = class {

    constructor(dimensions) {
        this._dimensions = dimensions;
    }

    get dimensions() {
        return this._dimensions;
    }

    toString() {
        if (!this._dimensions || this._dimensions.length == 0) {
            return '';
        }
        return '[' + this._dimensions.join(',') + ']';
    }
};

openvino.Metadata = class {

    static open(host) {
        if (openvino.Metadata._metadata) {
            return Promise.resolve(openvino.Metadata._metadata);
        }
        return host.request(null, 'openvino-metadata.json', 'utf-8').then((data) => {
            openvino.Metadata._metadata = new openvino.Metadata(data);
            return openvino.Metadata._metadata;
        }).catch(() => {
            openvino.Metadata._metadata = new openvino.Metadata(null);
            return openvino.Metadata._metadata;
        });
    }

    constructor(data) {
        this._map = {};
        this._attributeCache = {};
        if (data) {
            var items = JSON.parse(data);
            if (items) {
                for (var item of items) {
                    if (item.name && item.schema) {
                        var name = item.name;
                        var schema = item.schema;
                        this._map[name] = schema;
                    }
                }
            }
        }
    }

    getSchema(operator) {
        return this._map[operator];
    }

    getAttributeSchema(operator, name) {
        var map = this._attributeCache[operator];
        if (!map) {
            map = {};
            var schema = this.getSchema(operator);
            if (schema && schema.attributes && schema.attributes.length > 0) {
                for (var attribute of schema.attributes) {
                    map[attribute.name] = attribute;
                }
            }
            this._attributeCache[operator] = map;
        }
        return map[name] || null;
    }
};

openvino.Error = class extends Error {

    constructor(message) {
        super(message);
        this.name = 'Error loading OpenVINO model.';
    }
};

if (typeof module !== 'undefined' && typeof module.exports === 'object') {
    module.exports.ModelFactory = openvino.ModelFactory;
}
