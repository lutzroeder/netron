/* jshint esversion: 6 */
/* eslint "indent": [ "error", 4, { "SwitchCase": 1 } ] */

var openvino = openvino || {};
var base = base || require('./base');
var long = long || { Long: require('long') };
var marked = marked || require('marked');

openvino.ModelFactory = class {

    match(context) {
        const extension = context.identifier.split('.').pop().toLowerCase();
        if (extension === 'xml') {
            if (context.text.includes('<net')) {
                return true;
            }
        }
        // if (extension === 'bin') {
        // }
        return false;
    }

    open(context, host) {
        const identifier = context.identifier;
        const extension = identifier.split('.').pop().toLowerCase();
        switch (extension) {
            case 'xml':
                return context.request(identifier.substring(0, identifier.length - 4) + '.bin', null).then((bin) => {
                    return this._openModel(identifier, host, context.text, bin);
                }).catch(() => {
                    return this._openModel(identifier, host, context.text, null);
                });
            case 'bin':
                return context.request(identifier.substring(0, identifier.length - 4) + '.xml', 'utf-8').then((xml) => {
                    return this._openModel(identifier, host, xml, context.buffer);
                });
        }
    }

    _openModel(identifier, host, xml, bin) {
        return openvino.Metadata.open(host).then((metadata) => {
            try {
                let errors = false;
                const parser = new DOMParser({ errorHandler: () => { errors = true; } });
                const xmlDoc = parser.parseFromString(xml, 'text/xml');
                if (errors || xmlDoc.documentElement == null || xmlDoc.getElementsByTagName('parsererror').length > 0) {
                    throw new openvino.Error("File format is not OpenVINO.");
                }
                const net = xmlDoc.documentElement;
                if (!net || net.nodeName != 'net' ||
                    openvino.Node.children(net, 'layers').length != 1 ||
                    openvino.Node.children(net, 'edges').length != 1) {
                    throw new openvino.Error("File format is not OpenVINO IR.");
                }
                return new openvino.Model(metadata, net, bin);
            }
            catch (error) {
                host.exception(error, false);
                let message = error && error.message ? error.message : error.toString();
                message = message.endsWith('.') ? message.substring(0, message.length - 1) : message;
                throw new openvino.Error(message + " in '" + identifier + "'.");
            }
        });
    }
};

openvino.Model = class {

    constructor(metadata, net, bin) {
        let graph = new openvino.Graph(metadata, net, bin);
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

    constructor(metadata, net, bin) {
        this._name = net.getAttribute('name') || '';
        this._batch = net.getAttribute('batch') || '';
        this._version = net.getAttribute('version') || '';
        this._nodes = [];
        this._operators = {};
        this._inputs = [];
        this._outputs = [];
        this._arguments = {};

        const layersElement = openvino.Node.children(net, 'layers')[0];
        const edgesElement = openvino.Node.children(net, 'edges')[0];

        const layers = openvino.Node.children(layersElement, 'layer');
        const edges = openvino.Node.children(edgesElement, 'edge');

        const edgeMap = this._collectEdges(edges);

        for (let layer of layers) {
            const operator = layer.getAttribute('type');
            switch (operator) {
                case 'Input': {
                    let args = [];
                    const name = layer.getAttribute('name') || '';
                    // precision is a part of OpenVINO IR layers of IR v6 and earlier
                    // in IR v7 and newer the port is no longer an attribute of the layer but of each output port
                    const precision = layer.getAttribute('precision');
                    const id = layer.getAttribute('id');
                    for (let outputElement of openvino.Node.children(layer, 'output')) {
                        for (let portElement of openvino.Node.children(outputElement, 'port')) {
                            const portPrecision = portElement.getAttribute('precision') || precision;
                            args.push(this._argument(id, portPrecision, portElement, null));
                        }
                    }
                    // IR input is not just a placeholder, it is conceptually the legitimate layer
                    // in order not to break compatibility with the overall approach
                    // with openvino.Parameter for inputs and openvino.Node for outputs
                    // input openvino.Node would be stored as an optional attribute of openvino.Parameter
                    const inputNode = new openvino.Node(this, metadata, bin, layer, edgeMap);
                    const inputParameter = new openvino.Parameter(name, args);
                    inputParameter._realNode = inputNode;
                    this._inputs.push(inputParameter);
                    break;
                }
                default:
                    this._nodes.push(new openvino.Node(this, metadata, bin, layer, edgeMap));
                    break;
            }
        }

        this._replaceTensorIteratorWithSubgraph(metadata, bin, layers, edges, edgeMap);
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
            const layerNames = nodesWithNonExistentInputs.map((n) => n.name).join(',');
            const message = `Graph seems to contain ${nodesWithNonExistentInputs.length} connected components. Not connected layers: ${layerNames}`;
            throw new openvino.Error(message);
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
        let id = layer + ':' + port.getAttribute('id');
        if (map) {
            id = map[id];
        }
        let argument = this._arguments[id];
        if (!argument) {
            let dimensions = [];
            for (let dimElement of Array.prototype.slice.call(port.getElementsByTagName('dim'))) {
                dimensions.push(parseInt(dimElement.textContent.trim()));
            }
            const shape = (dimensions.length == 0) ? null : new openvino.TensorShape(dimensions);
            argument = new openvino.Argument(id, new openvino.TensorType(precision, shape), null);
        }
        return argument;
    }

    _replaceTensorIteratorWithSubgraph(metadata, bin, layers, edges, edgeMap) {
        const tiNodes = layers.filter((node) => node.getAttribute('type') === 'TensorIterator');
        for (let singleTensorIteratorNode of tiNodes) {
            const singleTensorIteratorNodeId = singleTensorIteratorNode.getAttribute("id");
            const tiNode = this._nodes.find((n) => n._id === `${singleTensorIteratorNodeId}`);
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
            for (let nestedLayer of iteratorLayers) {
                let nestedNode = new openvino.Node(this, metadata, bin, nestedLayer, iteratorAllEdges);
                nestedNode._id = `${singleTensorIteratorNodeId}_${nestedLayer.getAttribute('id')}`;
                for (let input of nestedNode._inputs) {
                    for (let input_argument of input.arguments) {
                        // we had a argument with id: 0:1  - meaning from layer "0" and its port "1"
                        // now as we rename all internal nodes to have an id of the TI included
                        // e.g. internal layer with id "0" and TI with id "14" results in internal layer to get id "14_0"
                        if (!input_argument._id){
                            continue;
                        }
                        input_argument._id = `${singleTensorIteratorNodeId}_${input_argument._id}`;
                    }
                }

                for (let output of nestedNode._outputs) {
                    for (let output_argument of output.arguments) {
                        // we had a argument with id: 1:1  - meaning from me with id "1" and my port "1"
                        // now as we rename all internal nodes to have an id of the TI included
                        // e.g. my layer with id "1" and TI with id "14" results in internal layer to get id "14_1"
                        if (!output_argument._id){
                            continue;
                        }
                        output_argument._id = `${singleTensorIteratorNodeId}_${output_argument._id}`;
                    }
                }
                
                this._nodes.push(nestedNode);
            }

            // We know for sure that edges that appeared in the nested IR are not
            // aware of the external context
            for (let nestedInput of mappingForNestedIR.input) {
                const nestedNode = this._nodes.find((n) => n._id === `${singleTensorIteratorNodeId}_${nestedInput.internal_layer_id}`);
                const candidate_edges = edges.filter((edge) => {
                    return edge.getAttribute('to-layer') === singleTensorIteratorNodeId && edge.getAttribute('to-port') === nestedInput.external_port_id;
                });
                if (!candidate_edges.length){
                    continue;
                }
                for (let candidate_edge of candidate_edges) {
                    const parentLayerID = candidate_edge.getAttribute('from-layer');
                    const parentPortID = candidate_edge.getAttribute('from-port');
                    
                    const parentNode = this._nodes.find((n) => n._id === `${parentLayerID}`);
                    if (!parentNode) {
                        // its parent is a TensorIterator that was removed on the previous cycle
                        // information is still present in the inputs of the current TensorIterator node
                        const potentialParentInput = tiNode._inputs.find((tiInput) => tiInput._name === 'input');
                        if (!potentialParentInput) {
                            return;
                        }
                        const inputWithoutId = nestedNode._inputs.find((input) => {
                            return Boolean(input._arguments.find((argument) => !argument._id));
                        });
                        if (inputWithoutId) {
                            const argumentWithoutId = inputWithoutId._arguments.find((argument) => !argument._id);
                            if (argumentWithoutId){
                                argumentWithoutId._id = potentialParentInput.arguments[0].id;
                            } 
                        }
                    } 
                    else {
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
                        }
                        else {
                            // TODO: no tensor information in the new argument - passed as null for now
                            const inputNode = new openvino.Node(this, metadata, bin, singleTensorIteratorNode, edgeMap);
                            const inputParameter = new openvino.Parameter((nestedNode._inputs.length+1).toString(), [
                                new openvino.Argument(newId, null, null)
                            ]);
                            inputParameter._realNode = inputNode;
                            nestedNode._inputs.push(inputParameter);
                        }
                    }
                }
            }

            for (let nestedOutput of mappingForNestedIR.output) {
                const nestedNode = this._nodes.find((n) => n._id === `${singleTensorIteratorNodeId}_${nestedOutput.internal_layer_id}`);
                const candidate_edges = edges.filter((edge) => {
                    return edge.getAttribute('from-layer') === singleTensorIteratorNodeId && edge.getAttribute('from-port') === nestedOutput.external_port_id;
                });
                if (candidate_edges.length === 0){
                    continue;
                }
                for (let candidate_edge of candidate_edges) {
                    const childLayerID = candidate_edge.getAttribute('to-layer');
                    const child = this._nodes.find((layer) => layer._id === childLayerID);
                    if (!child._inputs || (child._inputs && child._inputs.length === 0)){
                        continue;
                    }
                    if (nestedNode._outputs && nestedNode._outputs[0]) {
                        for (let child_input of child._inputs) {
                            for (let argument of child_input._arguments) {
                                if (!argument._id || (argument._id && argument._id.split(':')[0] !== singleTensorIteratorNodeId)) {
                                    continue;
                                }
                                const myPort = nestedNode._outputs[0]._arguments[0]._id.split(':')[1];
                                argument._id = `${nestedNode.id}:${myPort}`;
                            }
                        }
                    }
                }
            }

            this._nodes = this._nodes.filter((node) => node.id !== singleTensorIteratorNode.id);
        }
    }

    _collectEdges(edges){
        let edgeMap = {};
        for (let edge of edges) {
            const fromLayer = edge.getAttribute('from-layer');
            const fromPort = edge.getAttribute('from-port');
            const toLayer = edge.getAttribute('to-layer');
            const toPort = edge.getAttribute('to-port');
            edgeMap[toLayer + ':' + toPort] = fromLayer + ':' + fromPort;
        }
        return edgeMap;
    }

    _collectPortsInformation(ports) {
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

    constructor(graph, metadata, bin, layer, edgeMap) {
        this._metadata = metadata;
        this._type = layer.getAttribute('type');
        this._name = layer.getAttribute('name') || '';
        this._id = layer.getAttribute('id');
        this._inputs = [];
        this._outputs = [];
        this._initializers = [];
        this._attributes = [];
        const precision = layer.getAttribute('precision');
        let inputIndex = 0;
        const input = openvino.Node.children(layer, 'input')[0];
        if (input) {
            for (let port of openvino.Node.children(input, 'port')) {
                let inputName = (inputIndex == 0) ? 'input' : inputIndex.toString(); 
                this._inputs.push(new openvino.Parameter(inputName, [
                    graph._argument(this._id, precision, port, edgeMap)
                ]));
                inputIndex++;
            }
        }
        let outputIndex = 0;
        const output = openvino.Node.children(layer, 'output')[0];
        if (output) {
            for (let portElement of openvino.Node.children(output, 'port')) {
                let outputName = (outputIndex == 0) ? 'output' : outputIndex.toString();
                const portPrecision = portElement.getAttribute('precision') || precision;
                this._outputs.push(new openvino.Parameter(outputName, [
                    graph._argument(this._id, portPrecision, portElement, null)
                ]));
                outputIndex++;
            }
        }
        let attributes = {};
        const data = openvino.Node.children(layer, 'data')[0];
        if (data && data.attributes) {
            for (let attribute of Array.from(data.attributes)) {
                attributes[attribute.name] = attribute.value;
                this._attributes.push(new openvino.Attribute(metadata, this, attribute.name, attribute.value));
            }
        }
        const blobs = openvino.Node.children(layer, 'blobs')[0];
        if (blobs){
            for (let blob of Array.from(blobs.childNodes).filter((node) => node.nodeName != '#text')) {
                if (blob.getAttribute && typeof blob.getAttribute === 'function') {
                    const name = blob.nodeName;
                    let data = null;
                    let shape = null;
                    const blobPrecision = blob.getAttribute('precision') || precision;
                    if (bin) {
                        const offset = parseInt(blob.getAttribute('offset'));
                        const size = parseInt(blob.getAttribute('size'));
                        if ((offset + size) <= bin.length) {
                            data = bin.slice(offset, offset + size);
                        }
                        const precisionMap = {
                            'FP16': 2, 'FP32': 4,
                            'I8': 1, 'I16': 2, 'I32': 4, 'I64': 8,
                            'U8': 1, 'U16': 2, 'U32': 4, 'U64': 8
                        };
                        if (precisionMap[blobPrecision]) {
                            let itemSize = precisionMap[blobPrecision];
                            switch (this._type) {
                                case 'FullyConnected': {
                                    switch (name) {
                                        case 'weights': {
                                            const outSize = parseInt(attributes['out-size'], 10);
                                            shape = [ size / (outSize * itemSize), outSize ];
                                            break;
                                        }
                                        case 'biases': {
                                            shape = [ parseInt(attributes['out-size'], 10) ];
                                            break;
                                        }
                                    }
                                    break;
                                }
                                case 'Convolution': {
                                    switch (name) {
                                        case 'biases': {
                                            shape = [ size / itemSize ];
                                            break;
                                        }
                                    }
                                    break;
                                }
                            }
                        }
                    }
                    if (shape) {
                        shape = new openvino.TensorShape(shape);
                    }
                    this._initializers.push(new openvino.Parameter(name, [
                        new openvino.Argument('', null, new openvino.Tensor(blobPrecision, shape, data))
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
        const schema = this._metadata.getSchema(this._type);
        return (schema && schema.category) ? schema.category : '';
    }

    get documentation() {
        let schema = this._metadata.getSchema(this._type);
        if (schema) {
            schema = JSON.parse(JSON.stringify(schema));
            schema.name = this._type;
            if (schema.description) {
                schema.description = marked(schema.description);
            }
            if (schema.attributes) {
                for (let attribute of schema.attributes) {
                    if (attribute.description) {
                        attribute.description = marked(attribute.description);
                    }
                }
            }
            if (schema.inputs) {
                for (let input of schema.inputs) {
                    if (input.description) {
                        input.description = marked(input.description);
                    }
                }
            }
            if (schema.outputs) {
                for (let output of schema.outputs) {
                    if (output.description) {
                        output.description = marked(output.description);
                    }
                }
            }
            if (schema.references) {
                for (let reference of schema.references) {
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
        let children = [];
        let child = element.firstChild;
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
        this._realNode = null;
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
        const schema = metadata.getAttributeSchema(node.operator, name);
        if (schema) {
            if (Object.prototype.hasOwnProperty.call(schema, 'type')) {
                this._type = schema.type;
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
                    case 'int32': {
                        const intValue = Number.parseInt(this._value, 10);
                        this._value = Number.isNaN(this._value - intValue) ? value : intValue;
                        break;
                    }
                    case 'float32':
                    case 'float64': {
                        const floatValue = Number.parseFloat(this._value);
                        this._value = Number.isNaN(this._value - floatValue) ? value : floatValue;
                        break;
                    }
                    case 'int32[]':
                        if (this._value.length > 2) {
                            let ints = [];
                            this._value.split(',').map((item) => {
                                item = item.trim();
                                const intValue = Number.parseInt(item, 10);
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
                            let floats = [];
                            this._value.split(',').map((item) => {
                                item = item.trim();
                                let floatValue = Number.parseFloat(item);
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
                let defaultValue = schema.default;
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

    get type() {
        return this._type;
    }

    get visible() {
        return this._visible == false ? false : true;
    }
};

openvino.Tensor = class {

    constructor(precision, shape, data) {
        this._data = data;
        this._type = new openvino.TensorType(precision, shape);
    }

    get type() {
        return this._type;
    }

    get kind() {
        return 'Blob';
    }

    get state() {
        return this._context().state;
    }

    get value() {
        let context = this._context();
        if (context.state) {
            return null;
        }
        context.limit = Number.MAX_SAFE_INTEGER;
        return this._decode(context, 0);
    }

    toString() {
        let context = this._context();
        if (context.state) {
            return '';
        }
        context.limit = 10000;
        let value = this._decode(context, 0);
        return JSON.stringify(value, null, 4);
    }

    _context() {
        let context = {};
        context.state = null;

        if (!this._data) {
            context.state = 'Tensor data is empty.';
            return context;
        }

        if (!this._type.shape) {
            context.state = 'Tensor shape is not defined.';
            return context;
        }

        context.index = 0;
        context.count = 0;
        context.data = new DataView(this._data.buffer, this._data.byteOffset, this._data.byteLength);
        context.dataType = this._type.dataType;
        context.shape = this._type.shape.dimensions;

        return context;
    }

    _decode(context, dimension) {
        let shape = context.shape;
        if (context.shape.length == 0) {
            shape = [ 1 ];
        }
        let results = [];
        let size = shape[dimension];
        if (dimension == shape.length - 1) {
            for (let i = 0; i < size; i++) {
                if (context.count > context.limit) {
                    results.push('...');
                    return results;
                }
                switch (this._type.dataType) {
                    case 'float32':
                        results.push(context.data.getFloat32(context.index, true));
                        context.index += 4;
                        context.count++;
                        break;
                    case 'float16':
                        results.push(context.data.getFloat16(context.index, true));
                        context.index += 2;
                        context.count++;
                        break;
                    case 'int8':
                        results.push(context.data.getInt8(context.index));
                        context.index += 1;
                        context.count++;
                        break;
                    case 'int16':
                        results.push(context.data.getInt16(context.index, true));
                        context.index += 2;
                        context.count++;
                        break;
                    case 'int32':
                        results.push(context.data.getInt32(context.index, true));
                        context.index += 4;
                        context.count++;
                        break;
                    case 'int64':
                        results.push(new long.Long(context.data.getUint32(context.index, true), context.data.getUint32(context.index + 4, true), false));
                        context.index += 8;
                        context.count++;
                        break;
                    case 'uint8':
                        results.push(context.data.getUint8(context.index));
                        context.index += 1;
                        context.count++;
                        break;
                    case 'uint16':
                        results.push(context.data.getUint16(context.index, true));
                        context.index += 2;
                        context.count++;
                        break;
                    case 'uint32':
                        results.push(context.data.getUint32(context.index, true));
                        context.index += 4;
                        context.count++;
                        break;
                    case 'uint64':
                        results.push(new long.Long(context.data.getUint32(context.index, true), context.data.getUint32(context.index + 4, true), true));
                        context.index += 8;
                        context.count++;
                        break;
                }
            }
        }
        else {
            for (let j = 0; j < size; j++) {
                if (context.count > context.limit) {
                    results.push('...');
                    return results;
                }
                results.push(this._decode(context, dimension + 1));
            }
        }
        if (context.shape.length == 0) {
            return results[0];
        }
        return results;
    }
};

openvino.TensorType = class {

    constructor(precision, shape) {
        switch (precision) {
            case 'FP16': this._dataType = 'float16'; break;
            case 'FP32': this._dataType = 'float32'; break;
            case 'I8':   this._dataType = 'int8'; break;
            case 'I16':  this._dataType = 'int16'; break;
            case 'I32':  this._dataType = 'int32'; break;
            case 'I64':  this._dataType = 'int64'; break;
            case 'U8':   this._dataType = 'uint8'; break;
            case 'U16':  this._dataType = 'uint16'; break;
            case 'U32':  this._dataType = 'uint32'; break;
            case 'U64':  this._dataType = 'uint64'; break;
            case 'BOOL': this._dataType = 'boolean'; break;
            case null:  this._dataType = '?'; break;
            case '':  this._dataType = '?'; break;
            default: throw new openvino.Error("Unknown precision '" + precision + "'.");
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
        this._map = new Map();
        this._attributeMap = new Map();
        if (data) {
            let items = JSON.parse(data);
            if (items) {
                for (let item of items) {
                    if (item && item.name && item.schema) {
                        if (this._map.has(item.name)) {
                            throw new openvino.Error("Duplicate metadata key '" + item.name + "'.");
                        }
                        this._map.set(item.name, item.schema);
                    }
                }
            }
        }
    }

    getSchema(operator) {
        return this._map.get(operator) || null;
    }

    getAttributeSchema(operator, name) {
        const key = operator + ':' + name;
        if (!this._attributeMap.has(key)) {
            this._attributeMap.set(key, null);
            const schema = this.getSchema(operator);
            if (schema && schema.attributes) {
                for (let attribute of schema.attributes) {
                    this._attributeMap.set(operator + ':' + attribute.name, attribute);
                }
            }
        }
        return this._attributeMap.get(key);
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
