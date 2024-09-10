
const openvino = {};

openvino.ModelFactory = class {

    match(context) {
        const identifier = context.identifier;
        const extension = identifier.split('.').pop().toLowerCase();
        if (/^.*\.ncnn\.bin$/.test(identifier) ||
            /^.*\.pnnx\.bin$/.test(identifier) ||
            /^.*pytorch_model.*\.bin$/.test(identifier) ||
            /^.*group.+-shard.+of.+\.bin$/.test(identifier) ||
            /^.*param\.bin$/.test(identifier)) {
            return;
        }
        if (extension === 'bin') {
            const stream = context.stream;
            const length = stream.length;
            const signature = [0x21, 0xA8, 0xEF, 0xBE, 0xAD, 0xDE];
            if (signature.length <= length && stream.peek(signature.length).every((value, index) => value === signature[index])) {
                return;
            }
            if (length >= 4) {
                const buffer = stream.peek(Math.min(0x20000, length));
                const signature = (buffer[0] | buffer[1] << 8 | buffer[2] << 16 | buffer [3] << 24) >>> 0;
                for (let i = 0; i < buffer.length - 4; i++) {
                    const signature = (buffer[i] | buffer[i + 1] << 8 | buffer[i + 2] << 16 | buffer [i + 3] << 24) >>> 0;
                    if (signature === 0xdeadbeef || // Core ML
                        signature === 0x01306b47 || signature === 0x000d4b38 || signature === 0x0002c056) { // ncnn
                        return;
                    }
                }
                if (identifier.endsWith('.bin') || identifier.endsWith('.serialized')) {
                    const stream = context.stream;
                    const signatures = [
                        [0x00, 0x00, 0x00, 0x02, 0x00, 0x00, 0x00, 0x02, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00],
                        [0x00, 0x00, 0x00, 0x02, 0x00, 0x00, 0x00, 0x03, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x01]
                    ];
                    if (stream.length >= 16 && signatures.some((signature) => stream.peek(signature.length).every((value, index) => value === signature[index]))) {
                        return;
                    }
                }
                const identifiers = new Set([
                    'config.bin', 'model.bin', '__model__.bin', 'weights.bin',
                    'programs.bin', 'best.bin', 'ncnn.bin',
                    'stories15M.bin','stories42M.bin','stories110M.bin','stories260K.bin'
                ]);
                if (!identifiers.has(identifier)) {
                    const size = Math.min(length, 1024) & 0xFFFC;
                    const buffer = stream.peek(size).slice(0);
                    if (signature === 0x00000001) {
                        return;
                    }
                    const floats = Array.from(new Float32Array(buffer.buffer, buffer.byteOffset, buffer.length >> 2));
                    if (floats.every((x) => x === 0)) {
                        context.type = 'openvino.bin';
                        return;
                    }
                    if (floats[0] !== 0 && floats.every((x) => !Number.isNaN(x) && Number.isFinite(x))) {
                        if (floats.every((x) => x > -20.0 && x < 20.0 && (x >= 0 || x < -0.0000001) && (x <= 0 || x > 0.0000001))) {
                            context.type = 'openvino.bin';
                            return;
                        }
                        if (floats.every((x) => x > -100.0 && x < 100.0 && (x * 10) % 1 === 0)) {
                            context.type = 'openvino.bin';
                            return;
                        }
                    }
                    const ints = Array.from(new Int32Array(buffer.buffer, buffer.byteOffset, buffer.length >> 2));
                    if (ints.length > 32 && ints.slice(0, 32).every((x) => x === 0 || x === 1 || x === 2 || x === 0x7fffffff)) {
                        context.type = 'openvino.bin';
                        return;
                    }
                    if (identifier.indexOf('openvino') !== -1) {
                        context.type = 'openvino.bin';
                        return;
                    }
                }
                return;
            }
        }
        const tags = context.tags('xml');
        if (tags.has('net')) {
            context.type = 'openvino.xml';
        }
    }

    filter(context, type) {
        return context.type !== 'openvino.xml' || type !== 'openvino.bin';
    }

    async open(context) {
        const identifier = context.identifier;
        const base = identifier.substring(0, identifier.length - 4);
        let bin = null;
        switch (context.type) {
            case 'openvino.xml': {
                try {
                    const file = `${base}.bin`;
                    const content = await context.fetch(file);
                    bin = content.stream.peek();
                } catch {
                    // continue regardless of error
                }
                break;
            }
            case 'openvino.bin': {
                try {
                    const file = `${base}.xml`;
                    bin = context.stream.peek();
                    context = await context.fetch(file, null);
                } catch (error) {
                    const message = error && error.message ? error.message : error.toString();
                    throw new openvino.Error(`OpenVINO model definition required (${message.replace(/\.$/, '')}).`);
                }
                break;
            }
            default: {
                throw new openvino.Error(`Unsupported OpenVINO format '${context.type}'.`);
            }
        }
        let document = null;
        try {
            document = context.read('xml');
        } catch (error) {
            const message = error && error.message ? error.message : error.toString();
            throw new openvino.Error(`File format is not OpenVINO XML (${message.replace(/\.$/, '')}).`);
        }
        if (!document.documentElement || document.documentElement.localName !== 'net') {
            throw new openvino.Error('File format is not OpenVINO IR.');
        }
        const element = document.documentElement;
        const metadata = await context.metadata('openvino-metadata.json');
        const object = (element) => {
            const obj = {};
            for (const attribute of element.attributes) {
                obj[attribute.localName] = attribute.value;
            }
            return obj;
        };
        const child = (parent, name) => {
            const elements = parent.getElementsByTagName(name);
            if (elements.length > 1) {
                throw new openvino.Error(`Element '${parent.localName}' has multiple '${name}' elements.`);
            }
            return elements.length > 0 ? elements[0] : null;
        };
        const children = (parent, name, element) => {
            const list = child(parent, name);
            return list ? list.getElementsByTagName(element) : [];
        };
        const edges = (parent, name) => {
            const map = {};
            for (const element of children(parent, name || 'edges', 'edge')) {
                const fromLayer = element.getAttribute('from-layer');
                const fromPort = element.getAttribute('from-port');
                const toLayer = element.getAttribute('to-layer');
                const toPort = element.getAttribute('to-port');
                map[`${toLayer}:${toPort}`] = `${fromLayer}:${fromPort}`;
            }
            return map;
        };
        const layers = (parent) => {
            const ports = (parent, name) => {
                return children(parent, name, 'port').map((element) => {
                    const port = object(element);
                    port.dims = element.getElementsByTagName('dim').map((dim) => parseInt(dim.textContent.trim(), 10));
                    return port;
                });
            };
            return children(parent, 'layers', 'layer').map((element) => {
                const layer = object(element);
                layer.input = ports(element, 'input');
                layer.output = ports(element, 'output');
                const data = child(element, 'data');
                const blobs = child(element, 'blobs');
                layer.data = data ? object(data) : {};
                layer.blobs = blobs ? blobs.getElementsByTagName('*').map((blob) => {
                    const obj = object(blob);
                    obj.name = blob.localName;
                    obj.offset = parseInt(obj.offset, 10);
                    obj.size = parseInt(obj.size, 10);
                    return obj;
                }) : [];
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
                        for (const port of port_map.getElementsByTagName('*')) {
                            const item = object(port);
                            switch (port.localName) {
                                case 'input': layer.port_map.input.push(item); break;
                                case 'output': layer.port_map.output.push(item); break;
                                default: throw new openvino.Error(`Unsupported port local name '${port.localName}'.`);
                            }
                        }
                    }
                }
                return layer;
            });
        };
        const net = object(element);
        net.body =  {
            layers: layers(element),
            edges: edges(element)
        };
        return new openvino.Model(metadata, net, bin);
    }
};

openvino.Model = class {

    constructor(metadata, net, bin) {
        this.name = net.name || '';
        this.graphs = [new openvino.Graph(metadata, net, bin)];
        this.format = 'OpenVINO IR';
    }
};

openvino.Graph = class {

    constructor(metadata, net, bin) {
        this.name = net.name || '';
        this.nodes = [];
        this.inputs = [];
        this.outputs = [];
        const tensors = new Map();
        const values = new Map();
        values.map = (layer, precision, port, map) => {
            const id = `${layer}:${port.id}`;
            const name = map && map[id] ? map[id] : id;
            if (name === '') {
                throw new openvino.Error('Empty value name.');
            }
            const shape = port.dims.length === 0 ? null : new openvino.TensorShape(port.dims);
            if (!precision && values.has(name)) {
                const value = values.get(name);
                if (value.type && value.type.shape && value.type.shape.equals(shape)) {
                    return value;
                }
            }
            const type = new openvino.TensorType(precision, shape);
            let tensor = null;
            if (tensors.has(id)) {
                const blob = tensors.get(id);
                const offset = blob.offset;
                const size = blob.size;
                const shape = new openvino.TensorShape(blob.shape);
                const type = new openvino.TensorType(blob.precision || precision, shape);
                const data = (bin && (offset + size) <= bin.length) ? bin.slice(offset, offset + size) : null;
                tensor = new openvino.Tensor(type, data, 'Const');
            }
            if (!values.has(name)) {
                values.set(name, new openvino.Value(name, type, tensor));
            } else if (type && !type.equals(values.get(name).type)) {
                throw new openvino.Error(`Duplicate value '${name}'.`);
            }
            return values.get(name);
        };
        const nodes = new Map();
        const constant = (layers, edges, back_edges) => {
            back_edges = back_edges || {};
            for (const layer of layers) {
                if (layer.type === 'Const' &&
                    layer.input.length === 0 && layer.output.length === 1 && layer.blobs.length === 0 &&
                    layer.data && layer.data.element_type !== undefined && layer.data.offset !== undefined && layer.data.size !== undefined) {
                    let precision = null;
                    switch (layer.data.element_type) {
                        case 'f16': precision = 'FP16'; break;
                        case 'f32': precision = 'FP32'; break;
                        case 'f64': precision = 'FP64'; break;
                        default: precision = layer.data.element_type.toUpperCase();
                    }
                    const shape = layer.data.shape;
                    layer.blobs.push({
                        name: 'value',
                        precision,
                        offset: parseInt(layer.data.offset, 10),
                        size: parseInt(layer.data.size, 10),
                        shape: shape ? shape.split(',').map((dim) => parseInt(dim.trim(), 10)) : null
                    });
                    layer.data = {};
                }
                if (layer.type === 'Const' && layer.blobs.length === 1 && !layer.blobs[0].shape &&
                    layer.input.length === 0 && layer.output.length === 1 && layer.output[0].dims) {
                    layer.blobs[0].shape = layer.output[0].dims;
                }
            }
            const constants = new Map();
            for (const layer of layers) {
                if (layer.type === 'Const' && layer.input.length === 0 && layer.output.length === 1) {
                    const from = `${layer.id}:${layer.output[0].id}`;
                    constants.set(from, { layer, counter: 0 });
                }
            }
            for (const from of Object.values(edges)) {
                if (constants.has(from)) {
                    constants.get(from).counter++;
                }
            }
            if (back_edges) {
                for (const from of Object.values(back_edges)) {
                    if (constants.has(from)) {
                        constants.get(from).counter++;
                    }
                }
            }
            for (const [name, value] of constants) {
                if (value.counter !== 1) {
                    constants.delete(name);
                }
            }
            for (const layer of layers) {
                if (layer.blobs.length === 0) {
                    for (let i = layer.input.length - 1; i >= 0; i--) {
                        const input = layer.input[i];
                        const to = `${layer.id}:${input.id}`;
                        const from = edges[to] || back_edges[to];
                        if (!constants.has(from)) {
                            break;
                        }
                        const constLayer = constants.get(from).layer;
                        if (constLayer && Array.isArray(constLayer.blobs) && constLayer.blobs.length > 0) {
                            const [blob] = constLayer.blobs;
                            if (blob) {
                                blob.id = constLayer.name || constLayer.id;
                                layer.input[i].blob = blob;
                                constants.get(from).layer = null;
                                constants.get(from).delete = true;
                            }
                        }
                    }
                }
            }
            return layers.filter((layer) => {
                if (layer.type === 'Const' && layer.input.length === 0 && layer.output.length === 1) {
                    const from = `${layer.id}:${layer.output[0].id}`;
                    if (constants.has(from) && constants.get(from).delete) {
                        return false;
                    }
                }
                return true;
            });
        };
        const body = net.body;
        const body_layers = body && Array.isArray(body.layers) ? body.layers : [];
        const body_edges = body && body.edges ? body.edges : {};
        const layers = new Map(body_layers.map((entry) => [entry.id, entry]));
        const ports = new Map();
        if (Array.isArray(net.input)) {
            for (const input of net.input) {
                const value = values.map('', input.precision, input);
                const argument = new openvino.Argument(input.id, [value]);
                this.inputs.push(argument);
                ports.set(input.id, value);
            }
        }
        if (Array.isArray(net.output)) {
            for (const output of net.output) {
                const value = values.map('', output.precision, output);
                const argument = new openvino.Argument(output.id, [value]);
                this.outputs.push(argument);
                ports.set(output.id, value);
            }
        }
        for (const layer of body_layers) {
            for (const output of layer.output) {
                if (!output.precision) {
                    output.precision = layer.precision;
                }
            }
        }
        if (net.port_map) {
            for (const input of net.port_map.input) {
                const external_port = net.input.find((v) => v.id === input.external_port_id);
                const layer = layers.get(input.internal_layer_id);
                if (input.internal_port_id === undefined) {
                    input.internal_port_id = '';
                    layer.input.push({
                        id: input.internal_port_id,
                        precision: layer.data.element_type,
                        dims: layer.data.shape.split(',')
                    });
                }
                const internal_port = layer.input.find((v) => v.id === input.internal_port_id);
                internal_port.precision = external_port.precision;
            }
            for (const output of net.port_map.output) {
                const external_port = net.output.find((v) => v.id === output.external_port_id);
                const layer = layers.get(output.internal_layer_id);
                if (output.internal_port_id === undefined) {
                    output.internal_port_id = '';
                    layer.output.push({
                        id: output.internal_port_id,
                        precision: external_port.precision,
                        dims: external_port.dims
                    });
                }
            }
        }
        const layer_list = constant(body_layers, body_edges);
        for (const layer of layer_list) {
            for (const input of layer.input) {
                if (input.blob) {
                    tensors.set(`${layer.id}:${input.id}`, input.blob);
                }
            }
        }
        for (const layer of layer_list) {
            for (const output of layer.output) {
                values.map(layer.id, output.precision, output, null);
            }
        }
        for (const layer of layer_list) {
            const inputs = layer.input.map((input) => {
                const to = `${layer.id}:${input.id}`;
                if (body.edges[to]) {
                    const output = body.edges[to] ? body.edges[to].split(':') : [];
                    const [outputLayerId, outputId] = output;
                    const outputLayer = layers.get(outputLayerId);
                    if (outputLayer && outputId) {
                        const output = outputLayer.output.find((output) => output.id === outputId);
                        if (input && output) {
                            input.precision = output.precision;
                        }
                    }
                }
                return values.map(layer.id, input.precision || layer.precision, input, body.edges);
            });
            const outputs = layer.output.map((output) => {
                let precision = null;
                if (output && output.precision) {
                    precision = output.precision;
                } else if (layer && layer.precision) {
                    precision = layer.precision;
                }
                return values.map(layer.id, precision, output, null);
            });
            const subgraph = Array.isArray(net.input) || Array.isArray(net.output);
            if (!subgraph && (layer.type === 'Input' || layer.type === 'Parameter')) {
                const name = layer.name || '';
                // precision is a part of OpenVINO IR layers of IR v6 and earlier
                // in IR v7 and newer the port is no longer an attribute of the layer but of each output port
                // IR input is not just a placeholder, it is conceptually the legitimate layer
                // in order not to break compatibility with the overall approach
                // with openvino.Argument for inputs and openvino.Node for outputs
                // input openvino.Node would be stored as an optional attribute of openvino.Parameter
                this.inputs.push(new openvino.Argument(name, outputs));
            } else {
                const node = new openvino.Node(metadata, layer, inputs, outputs, bin);
                nodes.set(layer.id, node);
            }
        }
        this.nodes = Array.from(nodes.values());
        if (net.port_map) {
            const createMapLayer = (obj) => {
                const data = {};
                for (const [name, value] of Object.entries(obj)) {
                    if (name !== 'external_port_id' && name !== 'internal_layer_id' && name !== 'internal_port_id') {
                        data[name] = value;
                    }
                }
                return { type: '-', data };
            };
            for (const input of net.port_map.input) {
                const internal_port = layers.get(input.internal_layer_id).input.find((v) => v.id === input.internal_port_id);
                const inputs = [ports.get(input.external_port_id)];
                const outputs = [values.map(input.internal_layer_id, internal_port.precision, internal_port)];
                const layer = createMapLayer(input);
                this.nodes.push(new openvino.Node(metadata, layer, inputs, outputs));
            }
            for (const output of net.port_map.output) {
                const internal_port = layers.get(output.internal_layer_id).output.find((v) => v.id === output.internal_port_id);
                const inputs = [values.map(output.internal_layer_id, internal_port.precision, internal_port)];
                const outputs = [ports.get(output.external_port_id)];
                const layer = createMapLayer(output);
                this.nodes.push(new openvino.Node(metadata, layer, inputs, outputs));
            }
        }
    }
};

openvino.Node = class {

    constructor(metadata, layer, inputs, outputs, bin) {
        this.name = layer.name || '';
        this.inputs = [];
        this.outputs = [];
        this.attributes = [];
        const type = layer.type;
        this.type = metadata.type(type) || { name: type };
        for (let i = 0; i < inputs.length;) {
            let input = null;
            if (this.type && Array.isArray(this.type.inputs) && i < this.type.inputs.length) {
                input = this.type.inputs[i];
            } else if (inputs.length === 1) {
                input = { name: 'input' };
            } else {
                input = { name: i.toString() };
            }
            const count = input.type === 'Tensor[]' ? inputs.length - i : 1;
            const values = inputs.slice(i, i + count);
            const argument = new openvino.Argument(input.name, values);
            this.inputs.push(argument);
            i += count;
        }
        for (let i = 0; i < outputs.length;) {
            let output = null;
            if (this.type && Array.isArray(this.type.outputs) && i < this.type.outputs.length) {
                output = this.type.outputs[i];
            } else if (outputs.length === 1) {
                output = { name: 'output' };
            } else {
                output = { name: i.toString() };
            }
            const count = output.type === 'Tensor[]' ? outputs.length - i : 1;
            const values = outputs.slice(i, i + count);
            const argument = new openvino.Argument(output.name, values);
            this.outputs.push(argument);
            i += count;
        }
        const op = type;
        for (const [name, obj] of Object.entries(layer.data)) {
            const schema = metadata.attribute(op, name);
            let value = obj;
            let type = null;
            let visible = true;
            if (schema && schema.type !== undefined) {
                type = schema.type;
                switch (schema.type) {
                    case '':
                    case 'graph':
                    case 'string':
                        break;
                    case 'boolean':
                        if (obj === '1' || obj === 'true' || obj === 'True') {
                            value = true;
                        } else if (obj === '0' || obj === 'false' || obj === 'False') {
                            value = false;
                        } else {
                            throw new openvino.Error(`Unsupported attribute boolean value '${obj}'.`);
                        }
                        break;
                    case 'int32':
                    case 'int64': {
                        const intValue = Number.parseInt(obj, 10);
                        value = Number.isNaN(obj - intValue) ? obj : intValue;
                        break;
                    }
                    case 'float32':
                    case 'float64': {
                        const floatValue = Number.parseFloat(obj);
                        value = Number.isNaN(obj - floatValue) ? obj : floatValue;
                        break;
                    }
                    case 'int32[]':
                        if (obj.length > 2) {
                            let ints = [];
                            for (const entry of obj.split(',')) {
                                const item = entry.trim();
                                const intValue = Number.parseInt(item, 10);
                                if (Number.isNaN(item - intValue)) {
                                    ints = null;
                                } else if (ints !== null) {
                                    ints.push(intValue);
                                }
                            }
                            if (ints !== null) {
                                value = ints;
                            }
                        }
                        break;
                    case 'float32[]':
                        if (obj.length > 2) {
                            let floats = [];
                            for (const entry of obj.split(',')) {
                                const item = entry.trim();
                                const floatValue = Number.parseFloat(item);
                                if (Number.isNaN(item - floatValue)) {
                                    floats = null;
                                } else if (floats !== null) {
                                    floats.push(floatValue);
                                }
                            }
                            if (floats !== null) {
                                value = floats;
                            }
                        }
                        break;
                    default:
                        throw new openvino.Error(`Unsupported attribute type '${schema.type}'.`);
                }
            }
            if (schema && schema.visible === false) {
                visible = false;
            } else if (schema && schema.default !== undefined) {
                let defaultValue = schema.default;
                if (value === defaultValue) {
                    visible = false;
                } else if (Array.isArray(value) && Array.isArray(defaultValue)) {
                    defaultValue = defaultValue.slice(0, defaultValue.length);
                    if (defaultValue.length > 1 && defaultValue[defaultValue.length - 1] === null) {
                        defaultValue.pop();
                        while (defaultValue.length < value.length) {
                            defaultValue.push(defaultValue[defaultValue.length - 1]);
                        }
                    }
                    if (value.every((item, index) => item === defaultValue[index])) {
                        visible = false;
                    }
                }
            }
            const attribute = new openvino.Argument(name, value, type, visible);
            this.attributes.push(attribute);
        }
        if (layer.type === 'TensorIterator') {
            const graph = new openvino.Graph(metadata, layer, null);
            const attribute = new openvino.Argument('body', graph, 'graph');
            this.attributes.push(attribute);
        }
        for (const blob of layer.blobs || []) {
            const name = blob.name;
            const offset = blob.offset;
            let data = (bin && (offset + blob.size) <= bin.length) ? bin.slice(offset, offset + blob.size) : null;
            let dimensions = blob.shape || null;
            const category = blob.kind || 'Blob';
            const id = blob.id || '';
            const precision = blob.precision || layer.precision;
            let itemSize = -1;
            switch (precision) {
                case 'BOOL': case 'BOOLEAN':            itemSize = 1; break;
                case 'I1':   case 'U1':                 itemSize = 0.125; break;
                case 'I4':   case 'U4':                 itemSize = 0.5; break;
                case 'I8':   case 'U8':  case 'F8E4M3': itemSize = 1; break;
                case 'I16':  case 'U16': case 'FP16':   itemSize = 2; break;
                case 'I32':  case 'U32': case 'FP32':   itemSize = 4; break;
                case 'I64':  case 'U64': case 'FP64':   itemSize = 8; break;
                default: throw new openvino.Error(`Unsupported data type size '${precision}'.`);
            }
            const weight = (name, precision, dimensions, data) => {
                const shape = dimensions ? new openvino.TensorShape(dimensions) : null;
                const type = new openvino.TensorType(precision, shape);
                const tensor = new openvino.Tensor(type, data, category);
                const value = new openvino.Value(id, null, tensor);
                this.inputs.push(new openvino.Argument(name, [value]));
                const size = Math.ceil(dimensions.reduce((a, b) => a * b, 1) * itemSize);
                if (data && data.length !== size) {
                    return data.slice(size, data.length);
                }
                return null;
            };
            if (itemSize !== -1) {
                switch (`${type}:${name}`) {
                    case 'FullyConnected:weights': {
                        const outSize = parseInt(layer.data['out-size'], 10);
                        dimensions = [layer.input[0].dims[1], outSize];
                        break;
                    }
                    case 'FullyConnected:biases': {
                        dimensions = [parseInt(layer.data['out-size'], 10)];
                        break;
                    }
                    case 'Convolution:weights':
                    case 'Deconvolution:weights': {
                        /* eslint-disable prefer-destructuring */
                        const c = this.inputs[0].value[0].type.shape.dimensions[1];
                        /* eslint-enable prefer-destructuring */
                        const group = parseInt(layer.data.group || '1', 10);
                        const kernel = layer.data['kernel-x'] !== undefined && layer.data['kernel-y'] !== undefined ?
                            [parseInt(layer.data['kernel-x'], 10), parseInt(layer.data['kernel-y'], 10)] :
                            layer.data.kernel.split(',').map((v) => parseInt(v.trim(), 10));
                        const n = parseInt(layer.data.output, 10);
                        dimensions = [Math.floor(c / group), n].concat(kernel);
                        break;
                    }
                    case 'LSTMCell:weights': {
                        /* eslint-disable prefer-destructuring */
                        const input_size = inputs[0].type.shape.dimensions[1];
                        /* eslint-enable prefer-destructuring */
                        const hidden_size = parseInt(layer.data.hidden_size, 10);
                        data = weight('W', precision, [4 * hidden_size, input_size], data);
                        data = weight('R', precision, [4 * hidden_size, hidden_size], data);
                        break;
                    }
                    case 'LSTMCell:biases': {
                        const hidden_size = parseInt(layer.data.hidden_size, 10);
                        data = weight('B', precision, [4 * hidden_size], data);
                        break;
                    }
                    case 'GRUCell:weights': {
                        /* eslint-disable prefer-destructuring */
                        const input_size = inputs[0].type.shape.dimensions[1];
                        /* eslint-enable prefer-destructuring */
                        const hidden_size = parseInt(layer.data.hidden_size, 10);
                        data = weight('W', precision, [3 * hidden_size, input_size], data);
                        data = weight('R', precision, [3 * hidden_size, hidden_size], data);
                        break;
                    }
                    case 'GRUCell:biases': {
                        const linear_before_reset = parseInt(layer.data.linear_before_reset, 10);
                        const hidden_size = parseInt(layer.data.hidden_size, 10);
                        dimensions = linear_before_reset ? [4 * hidden_size] : [3 * hidden_size];
                        data = weight('B', precision, dimensions, data);
                        break;
                    }
                    case 'Convolution:biases': {
                        dimensions = [parseInt(layer.data.output, 10)];
                        break;
                    }
                    case 'ScaleShift:weights':
                    case 'ScaleShift:biases':
                    case 'Normalize:weights': {
                        dimensions = [layer.input[0].dims[1]];
                        break;
                    }
                    case 'PReLU:weights': {
                        dimensions = layer.data.channel_shared === '1' ? [1] : [layer.input[0].dims[1]];
                        break;
                    }
                    case 'Const:custom': {
                        if (this.outputs.length > 0 &&
                            this.outputs[0].value.length > 0 &&
                            this.outputs[0].value[0].type &&
                            this.outputs[0].value[0].type.shape &&
                            this.outputs[0].value[0].type.shape.dimensions) {
                            dimensions = this.outputs[0].value[0].type.shape.dimensions;
                        }
                        break;
                    }
                    default: {
                        break;
                    }
                }
            }
            if (data) {
                weight(name, precision, dimensions, data);
            }
        }
    }
};

openvino.Argument = class {

    constructor(name, value, type, visible) {
        this.name = name;
        this.value = value;
        this.type = type || null;
        this.visible = visible !== false;
    }
};

openvino.Value = class {

    constructor(name, type, initializer) {
        if (typeof name !== 'string') {
            throw new openvino.Error(`Invalid value identifier '${JSON.stringify(name)}'.`);
        }
        this.name = name;
        this.type = initializer ? initializer.type : type;
        this.initializer = initializer || null;
    }
};

openvino.Tensor = class {

    constructor(type, data, category) {
        this.type = type;
        this.values = data;
        this.category = category;
    }
};

openvino.TensorType = class {

    constructor(precision, shape) {
        precision = precision ? precision.toLowerCase() : precision;
        switch (precision) {
            case 'f4e2m1':  this.dataType = 'float4e2m1'; break;
            case 'f8e4m3':  this.dataType = 'float8e4m3'; break;
            case 'f8e5m2':  this.dataType = 'float8e5m2'; break;
            case 'f8e8m0':  this.dataType = 'float8e8m0'; break;
            case 'f16':     this.dataType = 'float16'; break;
            case 'f32':     this.dataType = 'float32'; break;
            case 'f64':     this.dataType = 'float64'; break;
            case 'fp16':    this.dataType = 'float16'; break;
            case 'fp32':    this.dataType = 'float32'; break;
            case 'fp64':    this.dataType = 'float64'; break;
            case 'bf16':    this.dataType = 'bfloat16'; break;
            case 'nf4':     this.dataType = 'nfloat4'; break;
            case 'i4':      this.dataType = 'int4'; break;
            case 'i8':      this.dataType = 'int8'; break;
            case 'i16':     this.dataType = 'int16'; break;
            case 'i32':     this.dataType = 'int32'; break;
            case 'i64':     this.dataType = 'int64'; break;
            case 'u1':      this.dataType = 'boolean'; break;
            case 'u4':      this.dataType = 'uint4'; break;
            case 'u8':      this.dataType = 'uint8'; break;
            case 'u16':     this.dataType = 'uint16'; break;
            case 'u32':     this.dataType = 'uint32'; break;
            case 'u64':     this.dataType = 'uint64'; break;
            case 'bool':    this.dataType = 'boolean'; break;
            case 'boolean': this.dataType = 'boolean'; break;
            case 'bin':     this.dataType = 'bit'; break;
            case 'string':  this.dataType = 'string'; break;
            case '':        this.dataType = '?'; break;
            case null:      this.dataType = '?'; break;
            default:        throw new openvino.Error(`Unsupported precision '${JSON.stringify(precision)}'.`);
        }
        this.shape = shape;
    }

    equals(obj) {
        return obj && this.dataType === obj.dataType &&
            ((this.shape === null && obj.shape === null) || this.shape && this.shape.equals(obj.shape));
    }

    toString() {
        if (this.shape === null) {
            return `${this.dataType}[?]`;
        }
        return this.dataType + this.shape.toString();
    }
};

openvino.TensorShape = class {

    constructor(dimensions) {
        this.dimensions = dimensions;
    }

    equals(obj) {
        return obj && Array.isArray(obj.dimensions) &&
            Array.isArray(this.dimensions) && this.dimensions.length === obj.dimensions.length
            && obj.dimensions.every((value, index) => this.dimensions[index] === value);
    }

    toString() {
        if (!this.dimensions || this.dimensions.length === 0) {
            return '';
        }
        return `[${this.dimensions.join(',')}]`;
    }
};

openvino.Error = class extends Error {

    constructor(message) {
        super(message);
        this.name = 'Error loading OpenVINO model.';
    }
};

export const ModelFactory = openvino.ModelFactory;
