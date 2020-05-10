/* jshint esversion: 6 */
/* eslint "indent": [ "error", 4, { "SwitchCase": 1 } ] */

var openvino = openvino || {};
var base = base || require('./base');
var long = long || { Long: require('long') };

openvino.ModelFactory = class {

    match(context) {
        const identifier = context.identifier;
        const extension = identifier.split('.').pop().toLowerCase();
        if (extension === 'xml') {
            if (context.text.includes('<net')) {
                return true;
            }
        }
        if (extension === 'bin') {
            switch (identifier) {
                case 'natives_blob.bin':
                case 'snapshot_blob.bin':
                case 'v8_context_snapshot.bin':
                    return false;
            }
            const buffer = context.buffer;
            const signature = [ 0x21, 0xA8, 0xEF, 0xBE, 0xAD, 0xDE ];
            if (buffer && buffer.length > 6 && signature.every((v, i) => v == buffer[i])) {
                return false;
            }
            if (buffer.length > 4) {
                const signature = buffer[0] | buffer[1] << 8 | buffer[2] << 16 | buffer [3] << 24;
                if (signature === 0x00000000 || signature === 0x00000001 ||
                    signature === 0x01306B47 || signature === 0x000D4B38 || signature === 0x0002C056) {
                    return false;
                }
            }
            return true;
        }
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
                }).catch((error) => {
                    host.exception(error, false);
                    const message = error && error.message ? error.message : error.toString();
                    throw new openvino.Error(message.replace(/\.$/, '') + " in '" + identifier + "'.");
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
                if (!xmlDoc.documentElement || xmlDoc.documentElement.nodeName != 'net') {
                    throw new openvino.Error("File format is not OpenVINO IR.");
                }
                const net = openvino.XmlReader.read(xmlDoc.documentElement);
                const model = new openvino.Model(metadata, net, bin);
                if (net.disconnectedLayers) {
                    host.exception(new openvino.Error("Graph contains not connected layers " + JSON.stringify(net.disconnectedLayers) + " in '" + identifier + "'."));
                }
                return model;
            }
            catch (error) {
                host.exception(error, false);
                const message = error && error.message ? error.message : error.toString();
                throw new openvino.Error(message.replace(/\.$/, '') + " in '" + identifier + "'.");
            }
        });
    }
};

openvino.Model = class {

    constructor(metadata, net, bin) {
        this._name = net.name || '';
        this._graphs = [ new openvino.Graph(metadata, net, bin) ];
    }

    get name() {
        return this._name;
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
        this._name = net.name || '';
        this._nodes = [];
        this._inputs = [];
        this._outputs = [];
        this._arguments = {};

        for (const layer of this._const(net.layers, net.edges)) {
            const inputs = layer.inputs.map((input) => this._argument(layer.id, layer.precision, input, net.edges));
            const outputs = layer.outputs.map((output) => this._argument(layer.id, output.precision || layer.precision, output, null));
            switch (layer.type) {
                case 'Input': {
                    const name = layer.name || '';
                    // precision is a part of OpenVINO IR layers of IR v6 and earlier
                    // in IR v7 and newer the port is no longer an attribute of the layer but of each output port
                    // IR input is not just a placeholder, it is conceptually the legitimate layer
                    // in order not to break compatibility with the overall approach
                    // with openvino.Parameter for inputs and openvino.Node for outputs
                    // input openvino.Node would be stored as an optional attribute of openvino.Parameter
                    this._inputs.push(new openvino.Parameter(name, outputs));
                    break;
                }
                default: {
                    this._nodes.push(new openvino.Node(this, metadata, bin, layer, inputs, outputs));
                    break;
                }
            }
        }

        this._replaceTensorIteratorWithSubgraph(metadata, bin, net.layers, net.edges);
        delete this._arguments;

        // Validation
        // all graph elements are split between inputs and nodes
        // by definition IR is a graph can have inputs of two types: "Input" and "Const"
        // "Input" layers are already moved to inputs when we parse a graph
        // if there are any layers that do not have input arguments and they are no Const ones
        // this means that this graph was not properly processed by the graph building logic
        const outputSet = new Set();
        for (const node of this._nodes) {
            for (const output of node.outputs) {
                for (const argument of output.arguments) {
                    outputSet.add(argument.name);
                }
            }
        }
        for (const input of this.inputs) {
            for (const argument of input.arguments) {
                outputSet.add(argument.name);
            }
        }
        const nodesWithNonExistentInputs = new Set();
        for (const node of this._nodes) {
            for (const input of node.inputs) {
                for (const argument of input.arguments) {
                    if (!argument.initializer && !outputSet.has(argument.name)) {
                        nodesWithNonExistentInputs.add(node);
                    }
                }
            }
        }
        if (nodesWithNonExistentInputs.size !== 0){
            net.disconnectedLayers = Array.from(nodesWithNonExistentInputs).map((node) => node.name);
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
        let id = layer + ':' + port.id;
        if (map) {
            id = map[id];
        }
        let argument = this._arguments[id];
        if (!argument) {
            const shape = port.dims.length == 0 ? null : new openvino.TensorShape(port.dims);
            argument = new openvino.Argument(id, new openvino.TensorType(precision, shape), null);
        }
        return argument;
    }

    _replaceTensorIteratorWithSubgraph(metadata, bin, layers, edges) {
        const tensorIteratorLayers = layers.filter((node) => node.type === 'TensorIterator');
        for (const tensorIteratorLayer of tensorIteratorLayers) {
            const singleTensorIteratorNodeId = tensorIteratorLayer.id;
            const tiNode = this._nodes.find((n) => n._id === singleTensorIteratorNodeId);
            const iteratorLayers = tensorIteratorLayer.body.layers;
            const iteratorEdgeMap = tensorIteratorLayer.body.edges;
            const iteratorBackEdgesMap = tensorIteratorLayer.back_edges;
            const iteratorAllEdges = Object.assign({}, iteratorEdgeMap, iteratorBackEdgesMap);
            const mappingForNestedIR = tensorIteratorLayer.port_map;
            for (const nestedLayer of this._const(iteratorLayers, iteratorAllEdges, iteratorBackEdgesMap)) {
                const inputs = nestedLayer.inputs.map((input) => this._argument(nestedLayer.id, nestedLayer.precision, input, iteratorAllEdges));
                const outputs = nestedLayer.outputs.map((output) => this._argument(nestedLayer.id, nestedLayer.precision || output.precision, output, null));
                const nestedNode = new openvino.Node(this, metadata, bin, nestedLayer, inputs, outputs);
                nestedNode._id = singleTensorIteratorNodeId + '_' + nestedLayer.id;
                for (const input of nestedNode._inputs) {
                    for (const input_argument of input.arguments) {
                        // we had a argument with id: 0:1  - meaning from layer "0" and its port "1"
                        // now as we rename all internal nodes to have an id of the TI included
                        // e.g. internal layer with id "0" and TI with id "14" results in internal layer to get id "14_0"
                        if (input_argument.name){
                            input_argument._name = singleTensorIteratorNodeId + '_' + input_argument.name;
                        }
                    }
                }

                for (const output of nestedNode._outputs) {
                    for (const output_argument of output.arguments) {
                        // we had a argument with id: 1:1  - meaning from me with id "1" and my port "1"
                        // now as we rename all internal nodes to have an id of the TI included
                        // e.g. my layer with id "1" and TI with id "14" results in internal layer to get id "14_1"
                        if (output_argument.name){
                            output_argument._name = singleTensorIteratorNodeId + '_' + output_argument.name;
                        }
                    }
                }

                this._nodes.push(nestedNode);
            }

            // We know for sure that edges that appeared in the nested IR are not aware of the external context
            for (const nestedInput of mappingForNestedIR.input) {
                const nestedNode = this._nodes.find((n) => n._id === singleTensorIteratorNodeId + '_' + nestedInput.internal_layer_id);

                const candidate_edge = edges[singleTensorIteratorNodeId + ':' + nestedInput.external_port_id];
                if (candidate_edge) {
                    const parts = candidate_edge.split(':');
                    const parentLayerID = parts[0];
                    const parentPortID = parts[1];
                    const parentNode = this._nodes.find((n) => n._id === parentLayerID);
                    if (!parentNode) {
                        // its parent is a TensorIterator that was removed on the previous cycle
                        // information is still present in the inputs of the current TensorIterator node
                        const potentialParentInput = tiNode._inputs.find((tiInput) => tiInput._name === 'input');
                        if (!potentialParentInput) {
                            return;
                        }
                        const inputWithoutId = nestedNode._inputs.find((input) => {
                            return Boolean(input.arguments.find((argument) => !argument.name));
                        });
                        if (inputWithoutId) {
                            const argumentWithoutId = inputWithoutId.arguments.find((argument) => !argument.name);
                            if (argumentWithoutId){
                                argumentWithoutId._name = potentialParentInput.arguments[0].name;
                            }
                        }
                    }
                    else {
                        if (!nestedNode._inputs){
                            throw new openvino.Error("Tensor Iterator node with name '" + nestedNode._id + "' does not have inputs.");
                        }

                        const newId = parentLayerID + ':' + parentPortID;
                        const inputWithoutId = nestedNode._inputs.find((input) => {
                            return Boolean(input.arguments.find((argument) => !argument.name));
                        });
                        if (inputWithoutId) {
                            const argumentWithoutId = inputWithoutId._arguments.find((argument) => !argument._name);
                            if (argumentWithoutId){
                                argumentWithoutId._name = newId;
                            }
                        }
                        else {
                            // TODO: no tensor information in the new argument - passed as null for now
                            nestedNode._inputs.push(new openvino.Parameter((nestedNode._inputs.length + 1).toString(), [
                                new openvino.Argument(newId, null, null)
                            ]));
                        }
                    }
                }
            }

            for (const nestedOutput of mappingForNestedIR.output) {
                const nestedNode = this._nodes.find((n) => n._id === `${singleTensorIteratorNodeId}_${nestedOutput.internal_layer_id}`);
                const toEdge = singleTensorIteratorNodeId + ':' + nestedOutput.external_port_id;
                const candidate_edges = Object.keys(edges).filter((key) => edges[key] === toEdge);
                for (const candidate_edge of candidate_edges) {
                    const childLayerID = candidate_edge.split(':')[0];
                    const child = this._nodes.find((layer) => layer._id === childLayerID);
                    if (!child._inputs || (child._inputs && child._inputs.length === 0)){
                        continue;
                    }
                    if (nestedNode._outputs && nestedNode._outputs[0]) {
                        for (const child_input of child._inputs) {
                            for (const argument of child_input._arguments) {
                                if (!argument.name || (argument.name && argument.name.split(':')[0] !== singleTensorIteratorNodeId)) {
                                    continue;
                                }
                                const myPort = nestedNode.outputs[0].arguments[0].name.split(':')[1];
                                argument._name = nestedNode.id + ':' + myPort;
                            }
                        }
                    }
                }
            }

            this._nodes = this._nodes.filter((node) => node.id !== tensorIteratorLayer.id);
        }
    }

    _const(layers, edges, back_edges) {
        const results = [];
        back_edges = back_edges || {};
        layers = layers.slice();
        for (const layer of layers) {
            if (layer.type === 'Const' && layer.inputs.length === 0 && layer.outputs.length === 1 &&
                layer.blobs.length === 0 && layer.data && layer.data.length > 3) {
                const data = {};
                for (const attribute of layer.data) {
                    data[attribute.name] = attribute.value;
                }
                if (data['element_type'] && data['offset'] && data['size']) {
                    const element_type = data['element_type'];
                    let precision = null;
                    switch (element_type) {
                        case 'f16': precision = 'FP16'; break;
                        case 'f32': precision = 'FP32'; break;
                        default: precision = element_type.toUpperCase();
                    }
                    const shape = data['shape'] ? data['shape'].split(',').map((dim) => parseInt(dim.trim(), 10)) : null;
                    layer.data = [];
                    layer.blobs.push({ name: 'custom', precision: precision, offset: parseInt(data['offset'], 10), size: parseInt(data['size'], 10), shape: shape });
                }
            }
            if (layer.type === 'Const' && layer.blobs.length === 1 && !layer.blobs[0].shape &&
                layer.inputs.length === 0 && layer.outputs.length === 1 && layer.outputs[0].dims) {
                layer.blobs[0].shape = layer.outputs[0].dims;
            }
        }

        const constMap = new Map();
        for (const layer of layers) {
            if (layer.type === 'Const' && layer.inputs.length === 0 && layer.outputs.length === 1) {
                const from = layer.id + ':' + layer.outputs[0].id;
                constMap.set(from, { layer: layer, counter: 0 });
            }
        }
        for (const to of Object.keys(edges)) {
            const from = edges[to];
            if (constMap.has(from)) {
                constMap.get(from).counter++;
            }
        }
        if (back_edges) {
            for (const to of Object.keys(back_edges)) {
                const from = back_edges[to];
                if (constMap.has(from)) {
                    constMap.get(from).counter++;
                }
            }
        }
        for (const pair of constMap) {
            if (pair[1].counter !== 1) {
                constMap.delete(pair[0]);
            }
        }
        for (const layer of layers) {
            if (layer.blobs.length === 0) {
                for (let i = layer.inputs.length - 1; i > 0; i--) {
                    const input = layer.inputs[i];
                    const to = layer.id + ':' + input.id;
                    const from = edges[to] || back_edges[to];
                    if (!constMap.has(from)) {
                        break;
                    }
                    const constLayer = constMap.get(from).layer;
                    const blob = constLayer.blobs[0];
                    blob.id = constLayer.name || constLayer.id;
                    blob.kind = 'Const';
                    layer.blobs.push(blob);
                    layer.inputs.splice(i, 1);
                    constMap.get(from).layer = null;
                    constMap.get(from).delete = true;
                }
            }
        }

        while (layers.length > 0) {
            const layer = layers.shift();
            if (layer.type === 'Const' && layer.inputs.length === 0 && layer.outputs.length === 1) {
                const from = layer.id + ':' + layer.outputs[0].id;
                if (constMap.has(from) && constMap.get(from).delete) {
                    continue;
                }
            }
            results.push(layer);
        }

        return results;
    }
};

openvino.Node = class {

    constructor(graph, metadata, bin, layer, inputs, outputs) {
        this._metadata = metadata;
        this._type = layer.type;
        this._name = layer.name || '';
        this._id = layer.id;
        this._inputs = [];
        this._outputs = [];
        this._initializers = [];
        this._attributes = [];
        const precision = layer.precision;
        let inputIndex = 0;
        for (const input of inputs) {
            const inputName = (inputIndex == 0) ? 'input' : inputIndex.toString();
            this._inputs.push(new openvino.Parameter(inputName, [ input ]));
            inputIndex++;
        }
        let outputIndex = 0;
        for (const output of outputs) {
            const outputName = (outputIndex == 0) ? 'output' : outputIndex.toString();
            this._outputs.push(new openvino.Parameter(outputName, [ output ]));
            outputIndex++;
        }
        const attributes = {};
        for (const attribute of layer.data) {
            attributes[attribute.name] = attribute.value;
            const attributeSchema = metadata.attribute(this.operator, attribute.name);
            this._attributes.push(new openvino.Attribute(attributeSchema, attribute.name, attribute.value));
        }
        for (const blob of layer.blobs) {
            const name = blob.name;
            const offset = blob.offset;
            const size = blob.size;
            const data = (bin && (offset + size) <= bin.length) ? bin.slice(offset, offset + size) : null;
            let dimensions = blob.shape || null;
            const kind = blob.kind || 'Blob';
            const id = blob.id || '';
            const dataType = blob.precision || precision;
            const precisionMap = {
                'FP16': 2, 'FP32': 4,
                'I8': 1, 'I16': 2, 'I32': 4, 'I64': 8,
                'U8': 1, 'U16': 2, 'U32': 4, 'U64': 8
            };
            const itemSize = precisionMap[dataType];
            if (itemSize) {
                switch (this._type + ':' + name) {
                    case 'FullyConnected:weights': {
                        const outSize = parseInt(attributes['out-size'], 10);
                        dimensions = [ size / (outSize * itemSize), outSize ];
                        break;
                    }
                    case 'FullyConnected:biases': {
                        dimensions = [ parseInt(attributes['out-size'], 10) ];
                        break;
                    }
                    case 'Convolution:weights':
                    case 'Deconvolution:weights': {
                        const c = this.inputs[0].arguments[0].type.shape.dimensions[1];
                        const group = parseInt(attributes['group'] || '1', 10);
                        const kernel = attributes['kernel-x'] && attributes['kernel-y'] ?
                            [ parseInt(attributes['kernel-x'], 10), parseInt(attributes['kernel-y'], 10) ] :
                            attributes['kernel'].split(',').map((v) => parseInt(v.trim(), 10));
                        const n = parseInt(attributes['output'], 10);
                        dimensions = [ Math.floor(c / group), n ].concat(kernel);
                        break;
                    }
                    case 'ScaleShift:weights':
                    case 'ScaleShift:biases':
                    case 'Convolution:biases':
                    case 'Normalize:weights':
                    case 'PReLU:weights': {
                        dimensions = [ Math.floor(size / itemSize) ];
                        break;
                    }
                    case 'Const:custom': {
                        if (this._outputs.length > 0 &&
                            this._outputs[0].arguments.length > 0 &&
                            this._outputs[0].arguments[0].type &&
                            this._outputs[0].arguments[0].type.shape &&
                            this._outputs[0].arguments[0].type.shape.dimensions) {
                            dimensions = this._outputs[0].arguments[0].type.shape.dimensions;
                        }
                        break;
                    }
                }
            }
            const shape = dimensions ? new openvino.TensorShape(dimensions) : null;
            this._initializers.push(new openvino.Parameter(name, [
                new openvino.Argument(id, null, new openvino.Tensor(dataType, shape, data, kind))
            ]));
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

    get metadata() {
        return this._metadata.type(this._type);
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

    constructor(name, type, initializer) {
        // if (typeof name !== 'string') {
        //     throw new openvino.Error("Invalid argument identifier '" + JSON.stringify(name) + "'.");
        // }
        this._name = name;
        this._type = type || null;
        this._initializer = initializer || null;
    }

    get name() {
        return this._name;
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

    constructor(schema, name, value) {
        this._name = name;
        this._value = value;
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
                                const floatValue = Number.parseFloat(item);
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

    constructor(precision, shape, data, kind) {
        this._data = data;
        this._type = new openvino.TensorType(precision, shape);
        this._kind = kind;
    }

    get kind() {
        return this._kind;
    }

    get type() {
        return this._type;
    }

    get state() {
        return this._context().state;
    }

    get value() {
        const context = this._context();
        if (context.state) {
            return null;
        }
        context.limit = Number.MAX_SAFE_INTEGER;
        return this._decode(context, 0);
    }

    toString() {
        const context = this._context();
        if (context.state) {
            return '';
        }
        context.limit = 10000;
        const value = this._decode(context, 0);
        return openvino.Tensor._stringify(value, '', '    ');
    }

    _context() {
        const context = {};
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
        const shape = context.shape.length == 0 ? [ 1 ] : context.shape;
        const results = [];
        const size = shape[dimension];
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

    static _stringify(value, indentation, indent) {
        if (Array.isArray(value)) {
            const result = [];
            result.push(indentation + '[');
            const items = value.map((item) => openvino.Tensor._stringify(item, indentation + indent, indent));
            if (items.length > 0) {
                result.push(items.join(',\n'));
            }
            result.push(indentation + ']');
            return result.join('\n');
        }
        if (typeof value == 'string') {
            return indentation + value;
        }
        if (value == Infinity) {
            return indentation + 'Infinity';
        }
        if (value == -Infinity) {
            return indentation + '-Infinity';
        }
        if (isNaN(value)) {
            return indentation + 'NaN';
        }
        return indentation + value.toString();
    }
};

openvino.TensorType = class {

    constructor(precision, shape) {
        precision = precision ? precision.toLowerCase() : precision;
        switch (precision) {
            case 'f16':  this._dataType = 'float16'; break;
            case 'fp16': this._dataType = 'float16'; break;
            case 'f32':  this._dataType = 'float32'; break;
            case 'fp32': this._dataType = 'float32'; break;
            case 'i8':   this._dataType = 'int8'; break;
            case 'i16':  this._dataType = 'int16'; break;
            case 'i32':  this._dataType = 'int32'; break;
            case 'i64':  this._dataType = 'int64'; break;
            case 'u1':   this._dataType = 'boolean'; break;
            case 'u8':   this._dataType = 'uint8'; break;
            case 'u16':  this._dataType = 'uint16'; break;
            case 'u32':  this._dataType = 'uint32'; break;
            case 'u64':  this._dataType = 'uint64'; break;
            case 'bool': this._dataType = 'boolean'; break;
            case '':     this._dataType = '?'; break;
            case null:   this._dataType = '?'; break;
            default: throw new openvino.Error("Unknown precision '" + JSON.stringify(precision) + "'.");
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
            const items = JSON.parse(data);
            if (items) {
                for (const item of items) {
                    if (item && item.name && item.schema) {
                        if (this._map.has(item.name)) {
                            throw new openvino.Error("Duplicate metadata key '" + item.name + "'.");
                        }
                        item.schema.name = item.name;
                        this._map.set(item.name, item.schema);
                    }
                }
            }
        }
    }

    type(operator) {
        return this._map.get(operator) || null;
    }

    attribute(operator, name) {
        const key = operator + ':' + name;
        if (!this._attributeMap.has(key)) {
            this._attributeMap.set(key, null);
            const schema = this.type(operator);
            if (schema && schema.attributes) {
                for (const attribute of schema.attributes) {
                    this._attributeMap.set(operator + ':' + attribute.name, attribute);
                }
            }
        }
        return this._attributeMap.get(key);
    }
};

openvino.XmlReader = class {

    static read(element) {
        const children = (parent, name) => {
            const children = [];
            let child = parent.firstChild;
            while (child != null) {
                if (child.nodeType == 1 && child.nodeName == name) {
                    children.push(child);
                }
                child = child.nextSibling;
            }
            return children;
        };
        const child = (parent, name) => {
            const elements = children(parent, name);
            if (elements.length > 1) {
                throw new openvino.Error("Element '" + parent.nodeName + "' has multiple '" + name + "' elements.");
            }
            return elements.length > 0 ? elements[0] : null;
        };
        const ports = (parent, name) => {
            const elements = child(parent, name);
            if (elements) {
                return children(elements, 'port').map((element) => {
                    return {
                        id: element.getAttribute('id'),
                        precision: element.getAttribute('precision'),
                        dims: Array.prototype.slice.call(element.getElementsByTagName('dim')).map((dim) => parseInt(dim.textContent.trim(), 10))
                    };
                });
            }
            return [];
        };
        const layers = (parent) => {
            const elements = child(parent, 'layers');
            if (elements) {
                return children(elements, 'layer').map((element) => {
                    const data = child(element, 'data');
                    const blobs = child(element, 'blobs');
                    const layer = {
                        id: element.getAttribute('id'),
                        name: element.getAttribute('name'),
                        type: element.getAttribute('type'),
                        precision: element.getAttribute('precision'),
                        data: !data ? [] : Array.from(data.attributes).map((attribute) => {
                            return { name: attribute.name, value: attribute.value};
                        }),
                        blobs: !blobs ? [] : Array.from(blobs.childNodes).filter((node) => node.nodeType === 1).map((blob) => {
                            return {
                                name: blob.nodeName,
                                precision: blob.getAttribute('precision'),
                                offset: parseInt(blob.getAttribute('offset'), 10),
                                size: parseInt(blob.getAttribute('size'), 10)
                            };
                        }),
                        inputs: ports(element, 'input'),
                        outputs: ports(element, 'output'),
                    };
                    if (layer.type === 'TensorIterator') {
                        layer.back_edges = edges(element, 'back_edges');
                        const body = child(element, 'body');
                        if (body) {
                            layer.body = {
                                layers: layers(body),
                                edges: edges(body)
                            };
                        }
                        const port_map = child(element, 'port_map');
                        if (port_map) {
                            layer.port_map = { input: [], output: [] };
                            for (const port of Array.from(port_map.childNodes).filter((element) => element.nodeType === 1)) {
                                const item = {
                                    axis: port.getAttribute("axis"),
                                    external_port_id: port.getAttribute("external_port_id"),
                                    internal_layer_id: port.getAttribute("internal_layer_id"),
                                    internal_port_id: port.getAttribute("internal_port_id")
                                };
                                switch (port.nodeName) {
                                    case 'input': layer.port_map.input.push(item); break;
                                    case 'output': layer.port_map.output.push(item); break;
                                }
                            }
                        }
                    }
                    return layer;
                });
            }
            return [];
        };
        const edges = (parent, name) => {
            const map = {};
            const elements = child(parent, name || 'edges');
            if (elements) {
                for (const element of children(elements, 'edge')) {
                    const fromLayer = element.getAttribute('from-layer');
                    const fromPort = element.getAttribute('from-port');
                    const toLayer = element.getAttribute('to-layer');
                    const toPort = element.getAttribute('to-port');
                    map[toLayer + ':' + toPort] = fromLayer + ':' + fromPort;
                }
            }
            return map;
        };
        return {
            name: element.getAttribute('name'),
            batch: element.getAttribute('batch'),
            version: element.getAttribute('version'),
            layers: layers(element),
            edges: edges(element)
        };
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
