
var openvino = {};
var xml = require('./xml');

openvino.ModelFactory = class {

    match(context) {
        const identifier = context.identifier;
        const extension = identifier.split('.').pop().toLowerCase();
        if (extension === 'bin') {
            if (identifier === 'natives_blob.bin' || identifier === 'snapshot_blob.bin' || identifier === 'v8_context_snapshot.bin') {
                return undefined;
            }
            const stream = context.stream;
            const signature = [ 0x21, 0xA8, 0xEF, 0xBE, 0xAD, 0xDE ];
            if (signature.length <= stream.length && stream.peek(signature.length).every((value, index) => value === signature[index])) {
                return undefined;
            }
            if (stream.length > 4) {
                const buffer = stream.peek(Math.min(256, stream.length));
                const signature = (buffer[0] | buffer[1] << 8 | buffer[2] << 16 | buffer [3] << 24) >>> 0;
                if (signature === 0x00000000 || signature === 0x00000001 ||
                    signature === 0x01306B47 || signature === 0x000D4B38 || signature === 0x0002C056) {
                    return undefined;
                }
                for (let i = 0; i < buffer.length - 4; i++) {
                    const signature = (buffer[i] | buffer[i + 1] << 8 | buffer[i + 2] << 16 | buffer [i + 3] << 24) >>> 0;
                    if (signature === 0xdeadbeef) {
                        return undefined;
                    }
                }
            }
            return 'openvino.bin';
        }
        const tags = context.tags('xml');
        if (tags.has('net')) {
            return 'openvino.xml';
        }
        return undefined;
    }

    async open(context, target) {
        const open = async (stream, bin) => {
            const metadata = await context.metadata('openvino-metadata.json');
            let document = null;
            try {
                const reader = xml.TextReader.open(stream);
                document = reader.read();
            } catch (error) {
                const message = error && error.message ? error.message : error.toString();
                throw new openvino.Error('File format is not OpenVINO XML (' + message.replace(/\.$/, '') + ').');
            }
            if (!document.documentElement || document.documentElement.localName != 'net') {
                throw new openvino.Error('File format is not OpenVINO IR.');
            }
            const net = openvino.XmlReader.read(document.documentElement);
            return new openvino.Model(metadata, net, bin);
        };
        const identifier = context.identifier;
        switch (target) {
            case 'openvino.xml':
                try {
                    const stream = await context.request(identifier.substring(0, identifier.length - 4) + '.bin', null);
                    const buffer = stream.read();
                    return open(context.stream, buffer);
                } catch (error) {
                    return open(context.stream, null);
                }
            case 'openvino.bin': {
                const stream = await context.request(identifier.substring(0, identifier.length - 4) + '.xml', null);
                return open(stream, context.stream.peek());
            }
            default:
                throw new openvino.Error("Unsupported OpenVINO format '" + target + "'.");
        }
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
        const values = new Map();
        const value = (layer, precision, port, map) => {
            const id = layer + ':' + port.id;
            const name = map ? (map[id] || '') : id;
            if (name === '') {
                throw new openvino.Error('Empty value name.');
            }
            const shape = port.dims.length == 0 ? null : new openvino.TensorShape(port.dims);
            if (!precision && values.has(name)) {
                const value = values.get(name);
                if (value.type && value.type.shape && value.type.shape.equals(shape)) {
                    return value;
                }
            }
            const type = new openvino.TensorType(precision, shape);
            if (!values.has(name)) {
                values.set(name, new openvino.Value(name, type, null));
            } else if (type && !type.equals(values.get(name).type)) {
                return new openvino.Value(name, type, null);
                // TODO throw new openvino.Error("Duplicate value '" + name + "'.");
            }
            return values.get(name);
        };
        const nodes = new Map();
        const replaceTensorIteratorWithSubgraph = (metadata, bin, layers, edges) => {
            const tensorIteratorLayers = layers.filter((node) => node.type === 'TensorIterator');
            for (const tensorIteratorLayer of tensorIteratorLayers) {
                const id = tensorIteratorLayer.id;
                const tiNode = nodes.get(id);
                const iteratorLayers = tensorIteratorLayer.body.layers;
                const iteratorEdgeMap = tensorIteratorLayer.body.edges;
                const iteratorBackEdgesMap = tensorIteratorLayer.back_edges;
                const iteratorAllEdges = Object.assign({}, iteratorEdgeMap, iteratorBackEdgesMap);
                const iteratorAllEdgesMap = {};
                for (const entry of Object.entries(Object.assign({}, iteratorEdgeMap, iteratorBackEdgesMap))) {
                    iteratorAllEdgesMap[id + '_' + entry[0]] = id + '_' + entry[1];
                }
                const mappingForNestedIR = tensorIteratorLayer.port_map;
                const layers = constant(iteratorLayers, iteratorAllEdges, iteratorBackEdgesMap);
                for (const layer of layers) {
                    for (const output of layer.outputs) {
                        value(id + '_' + layer.id, layer.precision || output.precision, output, null);
                    }
                }
                const nestedLayers = new Map(layers.map((layer) => [ id + '_' + layer.id, layer ]));
                const newValues = [];
                for (const nestedInput of mappingForNestedIR.input) {
                    const key = id + '_' + nestedInput.internal_layer_id;
                    const nestedLayer = nestedLayers.get(key);
                    const candidate_edge = edges[id + ':' + nestedInput.external_port_id];
                    if (candidate_edge) {
                        const parts = candidate_edge.split(':');
                        const parentLayerId = parts[0];
                        const parentPortId = parts[1];
                        if (!nodes.has(parentLayerId) && !nestedLayers.has(parentLayerId)) {
                            // its parent is a TensorIterator that was removed on the previous cycle
                            // information is still present in the inputs of the current TensorIterator node
                            const potentialParentInput = tiNode.inputs.find((tiInput) => tiInput.name === 'input');
                            if (!potentialParentInput) {
                                continue;
                            }
                            for (const input of nestedLayer.inputs) {
                                const input_id = id + '_' + nestedLayer.id + ':' + input.id;
                                if (!iteratorAllEdgesMap[input_id]) {
                                    iteratorAllEdgesMap[input_id] = potentialParentInput.value[0].name;
                                    break;
                                }
                            }
                        } else {
                            if (!nestedLayer.inputs) {
                                throw new openvino.Error("Tensor iterator '" + key + "' does not have inputs.");
                            }
                            const newId = parentLayerId + ':' + parentPortId;
                            let newValue = true;
                            for (const input of nestedLayer.inputs) {
                                const input_id = id + '_' + nestedLayer.id + ':' + input.id;
                                if (!iteratorAllEdgesMap[input_id]) {
                                    iteratorAllEdgesMap[input_id] = newId;
                                    newValue = false;
                                    break;
                                }
                            }
                            if (newValue) {
                                // TODO: no tensor information in the new argument - passed as null for now
                                newValues.push({ layer: key, id: newId });
                            }
                        }
                    }
                }
                for (const layer of layers) {
                    const inputs = layer.inputs.map((input) => {
                        return value(id + '_' + layer.id, layer.precision, input, iteratorAllEdgesMap);
                    });
                    const outputs = layer.outputs.map((output) => {
                        return value(id + '_' + layer.id, layer.precision || output.precision, output, null);
                    });
                    const key = id + '_' + layer.id;
                    const node = new openvino.Node(metadata, bin, layer, inputs, outputs);
                    nodes.set(key, node);
                }
                for (const newValue of newValues) {
                    const nestedLayer = nodes.get(newValue.layer);
                    if (!values.has(newValue.id)) {
                        values.set(newValue.id, new openvino.Argument(newValue.id, null, null));
                    }
                    const value = values.get(newValue.id);
                    const argument = new openvino.Argument((nestedLayer.inputs.length + 1).toString(), [ value ]);
                    nestedLayer.inputs.push(argument);
                }
                for (const nestedOutput of mappingForNestedIR.output) {
                    const key = id + '_' + nestedOutput.internal_layer_id;
                    const nestedNode = nodes.get(key);
                    const toEdge = id + ':' + nestedOutput.external_port_id;
                    const candidate_edges = Object.keys(edges).filter((key) => edges[key] === toEdge);
                    for (const candidate_edge of candidate_edges) {
                        const childLayerID = candidate_edge.split(':')[0];
                        const child = nodes.get(childLayerID);
                        if (!child.inputs || (child.inputs && child.inputs.length === 0)) {
                            continue;
                        }
                        for (const child_input of child.inputs) {
                            for (const value of child_input.value) {
                                if (!value.name || (value.name && value.name.split(':')[0] !== id)) {
                                    continue;
                                }
                                if (nestedNode.outputs && nestedNode.outputs.length === 0) {
                                    // it turns out that IRs of version 10 with TensorIterators (TI) can omit
                                    // output section in the output layers of TI body. It seems to happen only
                                    // for cases when TI has a single output. Therefore, there is a workaround that
                                    // creates fake output section for the TI output layer in order to connect it
                                    // with a layer from the main IR.
                                    const myPort = 0;
                                    const newId = key + ':' + myPort;
                                    nestedNode.outputs.push(new openvino.Argument('output', [
                                        new openvino.Value(newId, null, null)
                                    ]));
                                }
                                const myPort = nestedNode.outputs[0].value[0].name.split(':')[1];
                                value._name = key + ':' + myPort;
                            }
                        }
                    }
                }
                nodes.delete(id);
            }
        };
        const constant = (layers, edges, back_edges, omitConstLayers) => {
            back_edges = back_edges || {};
            for (const layer of layers) {
                if (layer.type === 'Const' && layer.inputs.length === 0 && layer.outputs.length === 1 && layer.blobs.length === 0 && layer.data && layer.data.size > 3) {
                    const element_type = layer.data.get('element_type');
                    const offset = layer.data.get('offset');
                    const size = layer.data.get('size');
                    if (element_type && offset !== null && size !== null) {
                        let precision = null;
                        switch (element_type) {
                            case 'f16': precision = 'FP16'; break;
                            case 'f32': precision = 'FP32'; break;
                            case 'f64': precision = 'FP64'; break;
                            default: precision = element_type.toUpperCase();
                        }
                        const shape = layer.data.get('shape');
                        const dims = shape ? shape.split(',').map((dim) => parseInt(dim.trim(), 10)) : null;
                        layer.data.clear();
                        layer.blobs.push({ name: 'custom', precision: precision, offset: parseInt(offset, 10), size: parseInt(size, 10), shape: dims });
                    }
                }
                if (layer.type === 'Const' && layer.blobs.length === 1 && !layer.blobs[0].shape &&
                    layer.inputs.length === 0 && layer.outputs.length === 1 && layer.outputs[0].dims) {
                    layer.blobs[0].shape = layer.outputs[0].dims;
                }
            }
            const constants = new Map();
            for (const layer of layers) {
                if (layer.type === 'Const' && layer.inputs.length === 0 && layer.outputs.length === 1) {
                    const from = layer.id + ':' + layer.outputs[0].id;
                    constants.set(from, { layer: layer, counter: 0 });
                }
            }
            for (const entry of Object.entries(edges)) {
                const from = entry[1];
                if (constants.has(from)) {
                    constants.get(from).counter++;
                }
            }
            if (back_edges) {
                for (const to of Object.keys(back_edges)) {
                    const from = back_edges[to];
                    if (constants.has(from)) {
                        constants.get(from).counter++;
                    }
                }
            }
            for (const entry of constants) {
                if (entry[1].counter !== 1) {
                    constants.delete(entry[0]);
                }
            }
            for (const layer of layers) {
                if (layer.blobs.length === 0) {
                    for (let i = layer.inputs.length - 1; i > 0; i--) {
                        const input = layer.inputs[i];
                        const to = layer.id + ':' + input.id;
                        const from = edges[to] || back_edges[to];
                        if (!constants.has(from)) {
                            break;
                        }
                        const constLayer = constants.get(from).layer;
                        if (constLayer && Array.isArray(constLayer.blobs)) {
                            const blob = constLayer.blobs[0];
                            if (blob) {
                                blob.id = constLayer.name || constLayer.id;
                                blob.kind = 'Const';
                                layer.blobs.push(blob);
                                layer.inputs.splice(i, 1);
                                constants.get(from).layer = null;
                                constants.get(from).delete = true;
                            }
                        }
                    }
                }
            }
            if (omitConstLayers) {
                for (const layer of layers) {
                    if (layer.blobs.length === 0) {
                        for (let i = layer.inputs.length - 1; i > 0; i--) {
                            const input = layer.inputs[i];
                            const to = layer.id + ':' + input.id;
                            const from = edges[to] || back_edges[to];
                            if (!constants.has(from)) {
                                break;
                            }
                            const constLayer = constants.get(from).layer;
                            const blob = constLayer.blobs[0];
                            if (blob) {
                                blob.id = constLayer.name || constLayer.id;
                                blob.kind = 'Const';
                                layer.blobs.push(blob);
                                layer.inputs.splice(i, 1);
                                constants.get(from).layer = null;
                                constants.get(from).delete = true;
                            }
                        }
                    }
                }
            }
            return layers.filter((layer) => {
                if (layer.type === 'Const' && layer.inputs.length === 0 && layer.outputs.length === 1) {
                    const from = layer.id + ':' + layer.outputs[0].id;
                    if (constants.has(from) && constants.get(from).delete) {
                        return false;
                    }
                }
                return true;
            });
        };
        const layers = new Map(net.layers.map((entry) => [ entry.id, entry ]));
        const layer_list = constant(net.layers, net.edges);
        for (const layer of net.layers) {
            for (const output of layer.outputs) {
                const precision = output && output.precision ? output.precision : layer && layer.precision ? layer.precision : null;
                value(layer.id, precision, output, null);
            }
        }
        for (const layer of layer_list) {
            const inputs = layer.inputs.map((input) => {
                const to = layer.id + ':' + input.id;
                if (net.edges[to]) {
                    const output = net.edges[to] ? net.edges[to].split(':') : [];
                    const outputLayerId = output[0];
                    const outputId = output[1];
                    const outputLayer = layers.get(outputLayerId);
                    if (outputLayer && outputId) {
                        const output = outputLayer.outputs.find((output) => output.id === outputId);
                        if (input && output) {
                            input.precision = output.precision;
                        }
                    }
                }
                return value(layer.id, input.precision || layer.precision, input, net.edges);
            });
            const outputs = layer.outputs.map((output) => {
                const precision = output && output.precision ? output.precision : layer && layer.precision ? layer.precision : null;
                return value(layer.id, precision, output, null);
            });
            switch (layer.type) {
                case 'Input': {
                    const name = layer.name || '';
                    // precision is a part of OpenVINO IR layers of IR v6 and earlier
                    // in IR v7 and newer the port is no longer an attribute of the layer but of each output port
                    // IR input is not just a placeholder, it is conceptually the legitimate layer
                    // in order not to break compatibility with the overall approach
                    // with openvino.Parameter for inputs and openvino.Node for outputs
                    // input openvino.Node would be stored as an optional attribute of openvino.Parameter
                    this._inputs.push(new openvino.Argument(name, outputs));
                    break;
                }
                case 'Parameter': {
                    const name = layer.name || '';
                    this._inputs.push(new openvino.Argument(name, outputs));
                    break;
                }
                default: {
                    const node = new openvino.Node(metadata, bin, layer, inputs, outputs);
                    nodes.set(layer.id, node);
                    break;
                }
            }
        }
        replaceTensorIteratorWithSubgraph(metadata, bin, net.layers, net.edges);
        // Validation
        // all graph elements are split between inputs and nodes
        // by definition IR is a graph can have inputs of two types: "Input" and "Const"
        // "Input" layers are already moved to inputs when we parse a graph
        // if there are any layers that do not have input value and they are no Const ones
        // this means that this graph was not properly processed by the graph building logic
        const outputs = new Set();
        for (const node of nodes.values()) {
            for (const output of node.outputs) {
                for (const value of output.value) {
                    outputs.add(value.name);
                }
            }
        }
        for (const input of this.inputs) {
            for (const value of input.value) {
                outputs.add(value.name);
            }
        }
        const nodesWithNonExistentInputs = new Set();
        for (const node of nodes.values()) {
            for (const input of node.inputs) {
                for (const value of input.value) {
                    if (!value.initializer && !outputs.has(value.name)) {
                        nodesWithNonExistentInputs.add(node);
                    }
                }
            }
        }
        if (nodesWithNonExistentInputs.size !== 0) {
            net.disconnectedLayers = Array.from(nodesWithNonExistentInputs).map((node) => node.name);
        }
        this._nodes = Array.from(nodes.values());
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
};

openvino.Node = class {

    constructor(metadata, bin, layer, inputs, outputs) {
        this._name = layer.name || '';
        this._inputs = [];
        this._outputs = [];
        this._attributes = [];
        const type = layer.type;
        this._type = metadata.type(type) || { name: type };
        const precision = layer.precision;
        for (let i = 0; i < inputs.length;) {
            const input = this._type && this._type.inputs && i < this._type.inputs.length ? this._type.inputs[i] : inputs.length === 1 ? { name: 'input' } : { name: i.toString() };
            const count = input.list ? inputs.length - i : 1;
            const list = inputs.slice(i, i + count);
            this._inputs.push(new openvino.Argument(input.name, list));
            i += count;
        }
        for (let i = 0; i < outputs.length;) {
            const output = this._type && this._type.outputs && i < this._type.outputs.length ? this._type.outputs[i] : outputs.length === 1 ? { name: 'output' } : { name: i.toString() };
            const count = output.list ? outputs.length - i : 1;
            const list = outputs.slice(i, i + count);
            this._outputs.push(new openvino.Argument(output.name, list));
            i += count;
        }
        for (const entry of layer.data) {
            const attribute = new openvino.Attribute(metadata.attribute(type, entry[0]), entry[0], entry[1]);
            this._attributes.push(attribute);
        }
        for (const blob of layer.blobs) {
            const name = blob.name;
            const offset = blob.offset;
            const size = blob.size;
            let data = (bin && (offset + size) <= bin.length) ? bin.slice(offset, offset + size) : null;
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
            const weight = (data, name, dimensions) => {
                const shape = dimensions ? new openvino.TensorShape(dimensions) : null;
                const value = new openvino.Value(id, null, new openvino.Tensor(dataType, shape, data, kind));
                this._inputs.push(new openvino.Argument(name, [ value ]));
                const size = dimensions.reduce((a, b) => a * b, 1) * itemSize;
                if (data && data.length !== size) {
                    return data.slice(size, data.length);
                }
                return null;
            };
            if (itemSize) {
                switch (type + ':' + name) {
                    case 'FullyConnected:weights': {
                        const outSize = parseInt(layer.data.get('out-size'), 10);
                        dimensions = [ size / (outSize * itemSize), outSize ];
                        break;
                    }
                    case 'FullyConnected:biases': {
                        dimensions = [ parseInt(layer.data.get('out-size'), 10) ];
                        break;
                    }
                    case 'Convolution:weights':
                    case 'Deconvolution:weights': {
                        const c = this.inputs[0].value[0].type.shape.dimensions[1];
                        const group = parseInt(layer.data.get('group') || '1', 10);
                        const kernel = layer.data.has('kernel-x') && layer.data.has('kernel-y') ?
                            [ parseInt(layer.data.get('kernel-x'), 10), parseInt(layer.data.get('kernel-y'), 10) ] :
                            layer.data.get('kernel').split(',').map((v) => parseInt(v.trim(), 10));
                        const n = parseInt(layer.data.get('output'), 10);
                        dimensions = [ Math.floor(c / group), n ].concat(kernel);
                        break;
                    }
                    case 'LSTMCell:weights': {
                        const input_size = inputs[0].type.shape.dimensions[1];
                        const hidden_size = parseInt(layer.data.get('hidden_size'), 10);
                        data = weight(data, 'W', [ 4 * hidden_size, input_size ]);
                        data = weight(data, 'R', [ 4 * hidden_size, hidden_size ]);
                        break;
                    }
                    case 'LSTMCell:biases': {
                        const hidden_size = parseInt(layer.data.get('hidden_size'), 10);
                        data = weight(data, 'B', [ 4 * hidden_size ]);
                        break;
                    }
                    case 'GRUCell:weights': {
                        const input_size = inputs[0].type.shape.dimensions[1];
                        const hidden_size = parseInt(layer.data.get('hidden_size'), 10);
                        data = weight(data, 'W', [ 3 * hidden_size, input_size ]);
                        data = weight(data, 'R', [ 3 * hidden_size, hidden_size ]);
                        break;
                    }
                    case 'GRUCell:biases': {
                        const linear_before_reset = parseInt(layer.data.get('linear_before_reset'), 10);
                        const hidden_size = parseInt(layer.data.get('hidden_size'), 10);
                        dimensions = linear_before_reset ? [ 4 * hidden_size ] : [ 3 * hidden_size ];
                        data = weight(data, 'B', dimensions);
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
                            this._outputs[0].value.length > 0 &&
                            this._outputs[0].value[0].type &&
                            this._outputs[0].value[0].type.shape &&
                            this._outputs[0].value[0].type.shape.dimensions) {
                            dimensions = this._outputs[0].value[0].type.shape.dimensions;
                        }
                        break;
                    }
                    default:
                        break;
                }
            }
            if (data) {
                weight(data, name, dimensions);
            }
        }
    }

    get name() {
        return this._name;
    }

    get device() {
        return this._device || '';
    }

    get type() {
        return this._type;
    }

    get attributes() {
        return this._attributes;
    }

    get inputs() {
        return this._inputs;
    }

    get outputs() {
        return this._outputs;
    }
};

openvino.Argument = class {

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

openvino.Value = class {

    constructor(name, type, initializer) {
        if (typeof name !== 'string') {
            throw new openvino.Error("Invalid value identifier '" + JSON.stringify(name) + "'.");
        }
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
                    case '':
                    case 'string':
                        break;
                    case 'boolean':
                        switch (value) {
                            case '1':
                            case 'true':
                            case 'True':
                                this._value = true;
                                break;
                            case '0':
                            case 'false':
                            case 'False':
                                this._value = false;
                                break;
                            default:
                                throw new openvino.Error("Unsupported attribute boolean value '" + value + "'.");
                        }
                        break;
                    case 'int32':
                    case 'int64': {
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
                            for (const entry of this._value.split(',')) {
                                const item = entry.trim();
                                const intValue = Number.parseInt(item, 10);
                                if (Number.isNaN(item - intValue)) {
                                    ints = null;
                                } else if (ints != null) {
                                    ints.push(intValue);
                                }
                            }
                            if (ints != null) {
                                this._value = ints;
                            }
                        }
                        break;
                    case 'float32[]':
                        if (this._value.length > 2) {
                            let floats = [];
                            for (const entry of this._value.split(',')) {
                                const item = entry.trim();
                                const floatValue = Number.parseFloat(item);
                                if (Number.isNaN(item - floatValue)) {
                                    floats = null;
                                } else if (floats != null) {
                                    floats.push(floatValue);
                                }
                            }
                            if (floats != null) {
                                this._value = floats;
                            }
                        }
                        break;
                    default:
                        throw new openvino.Error("Unsupported attribute type '" + schema.type + "'.");
                }
            }
            if (schema && schema.visible == false) {
                this._visible = false;
            } else if (Object.prototype.hasOwnProperty.call(schema, 'default')) {
                let defaultValue = schema.default;
                if (this._value == defaultValue) {
                    this._visible = false;
                } else if (Array.isArray(this._value) && Array.isArray(defaultValue)) {
                    defaultValue = defaultValue.slice(0, defaultValue.length);
                    if (defaultValue.length > 1 && defaultValue[defaultValue.length - 1] == null) {
                        defaultValue.pop();
                        while (defaultValue.length < this._value.length) {
                            defaultValue.push(defaultValue[defaultValue.length - 1]);
                        }
                    }
                    if (this._value.every((item, index) => item == defaultValue[index])) {
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

    constructor(precision, shape, data, category) {
        this._type = new openvino.TensorType(precision, shape);
        this._data = data;
        this._category = category;
    }

    get category() {
        return this._category;
    }

    get type() {
        return this._type;
    }

    get values() {
        return this._data;
    }
};

openvino.TensorType = class {

    constructor(precision, shape) {
        precision = precision ? precision.toLowerCase() : precision;
        switch (precision) {
            case 'f16':     this._dataType = 'float16'; break;
            case 'f32':     this._dataType = 'float32'; break;
            case 'f64':     this._dataType = 'float64'; break;
            case 'fp16':    this._dataType = 'float16'; break;
            case 'fp32':    this._dataType = 'float32'; break;
            case 'fp64':    this._dataType = 'float64'; break;
            case 'bf16':    this._dataType = 'bfloat16'; break;
            case 'i4':      this._dataType = 'int4'; break;
            case 'i8':      this._dataType = 'int8'; break;
            case 'i16':     this._dataType = 'int16'; break;
            case 'i32':     this._dataType = 'int32'; break;
            case 'i64':     this._dataType = 'int64'; break;
            case 'u1':      this._dataType = 'boolean'; break;
            case 'u4':      this._dataType = 'uint4'; break;
            case 'u8':      this._dataType = 'uint8'; break;
            case 'u16':     this._dataType = 'uint16'; break;
            case 'u32':     this._dataType = 'uint32'; break;
            case 'u64':     this._dataType = 'uint64'; break;
            case 'bool':    this._dataType = 'boolean'; break;
            case 'boolean': this._dataType = 'boolean'; break;
            case 'bin':     this._dataType = 'bit'; break;
            case '':        this._dataType = '?'; break;
            case null:      this._dataType = '?'; break;
            default:        throw new openvino.Error("Unsupported precision '" + JSON.stringify(precision) + "'.");
        }
        this._shape = shape;
    }

    get dataType() {
        return this._dataType;
    }

    get shape() {
        return this._shape;
    }

    equals(obj) {
        return obj && this._dataType === obj.dataType && this._shape && this._shape.equals(obj.shape);
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

    equals(obj) {
        return obj && Array.isArray(obj.dimensions) &&
            Array.isArray(this._dimensions) && this._dimensions.length === obj.dimensions.length
            && obj.dimensions.every((value, index) => this._dimensions[index] === value);
    }

    toString() {
        if (!this._dimensions || this._dimensions.length == 0) {
            return '';
        }
        return '[' + this._dimensions.join(',') + ']';
    }
};

openvino.XmlReader = class {

    static read(element) {
        const children = (parent, name) => {
            const children = [];
            let child = parent.firstChild;
            while (child != null) {
                if (child.nodeType == 1 && child.prefix === null && child.localName == name) {
                    children.push(child);
                }
                child = child.nextSibling;
            }
            return children;
        };
        const child = (parent, name) => {
            const elements = children(parent, name);
            if (elements.length > 1) {
                throw new openvino.Error("Element '" + parent.localName + "' has multiple '" + name + "' elements.");
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
                        dims: element.getElementsByTagName('dim').map((dim) => parseInt(dim.textContent.trim(), 10))
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
                        data: !data ? new Map() : new Map(Array.from(data.attributes).map((attribute) => [ attribute.localName, attribute.value ])),
                        blobs: !blobs ? [] : Array.from(blobs.childNodes).filter((node) => node.nodeType === 1).map((blob) => {
                            return {
                                name: blob.localName,
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
                                switch (port.localName) {
                                    case 'input': layer.port_map.input.push(item); break;
                                    case 'output': layer.port_map.output.push(item); break;
                                    default: throw new openvino.Error("Unsupported port local name '" + port.localName + "'.");
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
