
import * as text from './text.js';
import * as flatbuffers from './flatbuffers.js';

const dlc = {};

dlc.ModelFactory = class {

    match(context) {
        const container = dlc.Container.open(context);
        if (container) {
            context.type = 'dlc';
            context.target = container;
        }
    }

    async open(context) {
        dlc.schema = await context.require('./dlc-schema');
        dlc.schema = dlc.schema.dlc;
        await context.target.read();
        const metadata = await context.metadata('dlc-metadata.json');
        return new dlc.Model(metadata, context.target);
    }
};

dlc.Model = class {

    constructor(metadata, target) {
        this.format = target.format;
        this.metadata = [];
        if (target.metadata.size > 0) {
            const version = target.metadata.get('model-version');
            if (version) {
                this.version = version;
            }
            const converter = target.metadata.get('converter-command');
            if (converter) {
                const source = converter.split(' ').shift().trim();
                if (source.length > 0) {
                    const version = target.metadata.get('converter-version');
                    this.source = version ? `${source} v${version}` : source;
                }
            }
        }
        for (const graph of target.graphs) {
            this.graphs = [new dlc.Graph(metadata, target.version, graph)];
        }
    }
};

dlc.Graph = class {

    constructor(metadata, version, graph) {
        this.name = graph.name;
        this.inputs = [];
        this.outputs = [];
        const values = new Map();
        switch (version) {
            case 3: {
                for (const node of graph.nodes) {
                    for (const name of node.inputs) {
                        if (!values.has(name)) {
                            values.set(name, {});
                        }
                    }
                    for (const name of node.outputs) {
                        if (!values.has(name)) {
                            values.set(name, {});
                        }
                    }
                    let shapes = new Array(node.outputs.length);
                    for (const attribute of node.attributes) {
                        if (attribute.name === 'OutputDims' &&
                            Array.isArray(attribute.attributes) && attribute.attributes.length > 0) {
                            shapes = attribute.data;
                            break;
                        }
                    }
                    for (let i = 0; i < node.outputs.length; i++) {
                        const name = node.outputs[i];
                        const value = values.get(name);
                        if (!value.shape && i < shapes.length) {
                            value.shape = shapes[i];
                        }
                    }
                }
                break;
            }
            case 4: {
                for (const tensor of graph.tensors) {
                    values.set(tensor.name, tensor);
                }
                break;
            }
            default: {
                break;
            }
        }
        for (const [name, tensor] of values) {
            const type = tensor.shape ? new dlc.TensorType(tensor.dtype, tensor.shape) : null;
            const initializer = tensor.data && tensor.data ? new dlc.Tensor(type, tensor.data) : null;
            const value = new dlc.Value(name, type, initializer);
            values.set(name, value);
        }
        const value = (name) => {
            if (!values.has(name)) {
                values.set(name, new dlc.Value(name));
            }
            return values.get(name);
        };
        this.nodes = [];
        for (const node of graph.nodes) {
            if (node.type === 'Input') {
                this.inputs.push(new dlc.Argument(node.name, node.inputs.map((input) => value(input))));
                continue;
            }
            this.nodes.push(new dlc.Node(metadata, version, node, value));
        }
    }
};

dlc.Argument = class {

    constructor(name, value) {
        this.name = name;
        this.value = value;
    }
};

dlc.Value = class {

    constructor(name, type, initializer) {
        if (typeof name !== 'string') {
            throw new dlc.Error(`Invalid value identifier '${JSON.stringify(name)}'.`);
        }
        this.name = name;
        this.type = type;
        this.initializer = initializer;
    }
};

dlc.Node = class {

    constructor(metadata, version, node, value) {
        const type = `${node.type}:v${version}`;
        this.type = { ...metadata.type(type) };
        this.type.name = node.type;
        this.name = node.name;
        this.inputs = [];
        this.outputs = [];
        this.attributes = [];
        const inputs = Array.isArray(node.inputs) ? Array.from(node.inputs).map((input) => value(input)) : [];
        if (Array.isArray(this.type.inputs) && inputs.length === this.type.inputs.length) {
            for (let i = 0; i < inputs.length; i++) {
                const argument = new dlc.Argument(this.type.inputs[i].name, [inputs[i]]);
                this.inputs.push(argument);
            }
        } else if (inputs.length > 0) {
            const argument = new dlc.Argument(inputs.length === 1 ? 'input' : 'inputs', inputs);
            this.inputs.push(argument);
        }
        const outputs = Array.isArray(node.outputs) ? Array.from(node.outputs).map((output) => value(output)) : [];
        if (Array.isArray(this.type.outputs) && outputs.length === this.type.outputs.length) {
            for (let i = 0; i < outputs.length; i++) {
                const argument = new dlc.Argument(this.type.outputs[i].name, [outputs[i]]);
                this.outputs.push(argument);
            }
        } else if (outputs.length > 0) {
            const argument = new dlc.Argument(outputs.length === 1 ? 'output' : 'outputs', outputs);
            this.outputs.push(argument);
        }
        if (node.attributes) {
            for (const attr of node.attributes) {
                if (attr.name === 'OutputDims') {
                    continue;
                }
                const attribute = new dlc.Attribute(metadata.attribute(type, attr.name), version, attr);
                this.attributes.push(attribute);
            }
        }
        if (node.weights) {
            for (const tensor of node.weights) {
                const type = new dlc.TensorType(tensor.data.dtype, tensor.shape);
                const value = new dlc.Value('', type, new dlc.Tensor(type, tensor.data));
                this.inputs.push(new dlc.Argument(tensor.name, [value]));
            }
        }
    }
};

dlc.Attribute = class {

    constructor(metadata, version, attribute) {
        this.name = attribute.name;
        this.type = attribute.type;
        switch (this.type) {
            case 'tensor': {
                const tensor = attribute.data;
                const type = new dlc.TensorType(tensor.dtype, tensor.shape);
                const data = tensor.data;
                this.value = new dlc.Tensor(type, data);
                break;
            }
            default: {
                this.value = attribute.data;
            }
        }
        if (metadata && metadata.type) {
            this.type = metadata.type;
            this.value = dlc.Utility.enum(version, this.type, this.value);
        }
    }
};

dlc.TensorType = class {

    constructor(dataType, shape) {
        this.dataType = dataType || '?';
        this.shape = new dlc.TensorShape(shape);
    }

    toString() {
        return this.dataType + this.shape.toString();
    }
};

dlc.TensorShape = class {

    constructor(dimensions) {
        this.dimensions = Array.from(dimensions);
    }

    toString() {
        if (Array.isArray(this.dimensions) && this.dimensions.length > 0) {
            return `[${this.dimensions.map((dimension) => dimension.toString()).join(',')}]`;
        }
        return '';
    }
};

dlc.Tensor = class {

    constructor(type, data) {
        this.type = type;
        if (data instanceof Uint8Array) {
            this.encoding = '<';
            this.values = data;
        } else {
            this.encoding = '|';
            switch (type.dataType) {
                case 'uint8': this.values = data.bytes; break;
                case 'float32': this.values = data.floats; break;
                default: throw new dlc.Error(`Unsupported tensor data type '${type.dataType}'.`);
            }
        }
    }
};

dlc.Container = class {

    static open(context) {
        const entries = context.peek('zip');
        if (entries instanceof Map && (entries.has('model') || entries.has('model.params'))) {
            return new dlc.Container(context, entries.get('model'), entries.get('model.params'), entries.get('dlc.metadata'));
        }
        const stream = context.stream;
        switch (dlc.Container._signature(stream).split('.').pop()) {
            case 'NETD':
                return new dlc.Container(context, stream, undefined, undefined);
            case 'NETP':
                return new dlc.Container(context, undefined, stream, undefined);
            case 'NR64':
                return new dlc.Container(context, undefined, stream, undefined);
            default:
                return null;
        }
    }

    constructor(context, model, params, metadata) {
        this.context = context;
        this._model = model;
        this._params = params;
        this._metadata = metadata;
    }

    async read() {
        if (this._model === undefined) {
            this._model = await this._fetch('model');
        }
        if (this._params === undefined) {
            this._params = await this._fetch('model.params');
        }
        if (this._metadata === undefined) {
            this._metadata = await this._fetch('dlc.metadata');
        }
        delete this.context;
        this.graphs = [];
        this.metadata = new Map();
        if (this._model) {
            this.format = 'DLC';
            const stream = this._model;
            delete this._model;
            const signature = dlc.Container._signature(stream);
            switch (signature) {
                case '2': {
                    throw new dlc.Error("File contains undocumented DLC v2 data.");
                }
                case '3.NETD':
                case 'NETD': {
                    this.version = 3;
                    this.graph = dlc.Container._model3(stream, signature);
                    this.graphs = [this.graph];
                    break;
                }
                case '4.NETD': {
                    this.version = 4;
                    this.graphs = dlc.Container._model4(stream);
                    break;
                }
                default: {
                    const buffer = stream.peek(Math.min(stream.length, 16));
                    const content = Array.from(buffer).map((c) => (c < 16 ? '0' : '') + c.toString(16)).join('');
                    throw new dlc.Error(`File contains undocumented '${content}' data.`);
                }
            }
        }
        if (this._params) {
            this.format = this.format || 'DLC Weights';
            const stream = this._params;
            delete this._params;
            const signature = dlc.Container._signature(stream);
            switch (signature) {
                case '2': {
                    throw new dlc.Error("File contains undocumented DLC v2 data.");
                }
                case '3.NETP':
                case 'NETP': {
                    this.version = this.graphs.length > 0 ? this.version : 3;
                    this.graph = dlc.Container._params3(stream, signature, this.graph);
                    this.graphs = [this.graph];
                    break;
                }
                case '4.NETP':
                case '4.NR64': {
                    dlc.Container._params4(stream, this.graphs, signature);
                    break;
                }
                default: {
                    const buffer = stream.peek(Math.min(stream.length, 16));
                    const content = Array.from(buffer).map((c) => (c < 16 ? '0' : '') + c.toString(16)).join('');
                    throw new dlc.Error(`File contains undocumented '${content}' data.`);
                }
            }
        }
        if (this._metadata) {
            const stream = this._metadata;
            delete this._metadata;
            const reader = text.Reader.open(stream);
            for (;;) {
                const line = reader.read();
                if (line === undefined) {
                    break;
                }
                const index = line.indexOf('=');
                if (index === -1) {
                    break;
                }
                const key = line.substring(0, index);
                const value = line.substring(index + 1);
                this.metadata.set(key, value);
            }
        }
    }

    static _model3(stream, signature) {
        let model = null;
        try {
            const buffer = new Uint8Array(signature === 'NETD' ? stream.peek() : stream.peek().subarray(8));
            const reader = flatbuffers.BinaryReader.open(buffer);
            model = dlc.schema.v3.Model.decode(reader, reader.root);
        } catch (error) {
            const message = error && error.message ? error.message : error.toString();
            throw new dlc.Error(`File format is not dlc.v3.NETD (${message.replace(/\.$/, '')}).`);
        }
        model.tensors = [];
        const updateAttribute = (attr) => {
            switch (attr.type) {
                case 1: return ['boolean',   attr.bool_value];
                case 2: return ['int32',     attr.int32_value];
                case 3: return ['uint32',    attr.uint32_value];
                case 4: return ['float32',   attr.float32_value];
                case 5: return ['string',    attr.string_value];
                case 7: return ['byte[]',    Array.from(attr.byte_list)];
                case 8: return ['int32[]',   Array.from(attr.int32_list)];
                case 9: return ['float32[]', Array.from(attr.float32_list)];
                case 11: {
                    const obj = {};
                    let index = 0;
                    let list = true;
                    for (const attribute of attr.attributes) {
                        const name = attribute.name;
                        const [, data] = updateAttribute(attribute);
                        obj[name] = data;
                        list = list && index.toString() === attribute.name;
                        index++;
                    }
                    return list ? ['', Object.values(obj)] : ['', obj];
                }
                default:
                    throw new dlc.Error(`Unsupported attribute type '${attr.type}'.`);
            }
        };
        for (const node of model.nodes) {
            for (const attribute of node.attributes) {
                const [type, data] = updateAttribute(attribute);
                attribute.type = type;
                attribute.data = data;
            }
        }
        return model;
    }

    static _model4(stream) {
        let model = null;
        try {
            const buffer = new Uint8Array(stream.peek().subarray(8));
            const reader = flatbuffers.BinaryReader.open(buffer);
            model = dlc.schema.v4.Model.decode(reader, reader.root);
        } catch (error) {
            const message = error && error.message ? error.message : error.toString();
            throw new dlc.Error(`File format is not dlc.v4.NETD (${message.replace(/\.$/, '')}).`);
        }
        const dataType = (value) => {
            switch (value) {
                case 0x0008: return 'int8';
                case 0x0016: return 'int16';
                case 0x0032: return 'int32';
                case 0x0108: return 'int8';
                case 0x0132: return 'int32';
                case 0x0216: return 'float16';
                case 0x0232: return 'float32';
                case 0x0308: return 'qint8';
                case 0x0316: return 'qint16';
                case 0x0332: return 'qint32';
                case 0x0408: return 'uint8';
                case 0x0416: return 'uint16';
                case 0x0432: return 'uint32';
                case 0x0508: return 'boolean';
                default: throw new dlc.Error(`Unsupported data type '${JSON.stringify(value)}'.`);
            }
        };
        const updateTensor = (tensor) => {
            tensor.dtype = dataType(tensor.dtype);
            tensor.output_dtype = dataType(tensor.output_dtype);
        };
        for (const graph of model.graphs) {
            for (const node of graph.nodes) {
                for (const attribute of node.attributes) {
                    switch (attribute.kind) {
                        case 0: {
                            const value = attribute.value;
                            switch (value.kind) {
                                case 0x7fffffff:
                                    attribute.data = value.string_value;
                                    attribute.type = 'string';
                                    break;
                                case 0x0032:
                                    attribute.data = value.int32_value;
                                    break;
                                case 0x0108:
                                    attribute.data = value.int32_value;
                                    attribute.type = 'int8';
                                    break;
                                case 0x0132:
                                    attribute.data = value.int32_value;
                                    attribute.type = 'int32';
                                    break;
                                case 0x0232:
                                    attribute.data = value.float32_value;
                                    attribute.type = 'float32';
                                    break;
                                case 0x0508:
                                    attribute.data = value.int32_value !== 0;
                                    attribute.type = 'boolean';
                                    break;
                                default:
                                    throw new dlc.Error(`Unknown attribute value kind '${value.kind}'.`);
                            }
                            break;
                        }
                        case 1: {
                            const tensor = attribute.tensor;
                            updateTensor(tensor);
                            attribute.type = 'tensor';
                            attribute.data = tensor;
                            break;
                        }
                        default: {
                            throw new dlc.Error(`Unknown attribute kind '${attribute.kind}'.`);
                        }
                    }
                }
            }
            for (const tensor of graph.tensors) {
                updateTensor(tensor);
            }
        }
        return model.graphs;
    }

    static _params3(stream, signature, graph) {
        let params = null;
        try {
            const buffer = new Uint8Array(signature === 'NETP' ? stream.peek() : stream.peek().subarray(8));
            const reader = flatbuffers.BinaryReader.open(buffer);
            params = dlc.schema.v3.ModelParameters.decode(reader, reader.root);
        } catch (error) {
            const message = error && error.message ? error.message : error.toString();
            throw new dlc.Error(`File format is not dlc.v3.NETP (${message.replace(/\.$/, '')}).`);
        }
        if (!graph) {
            graph = new dlc.schema.v3.ModelParameters();
            graph.nodes = new Array(params.nodes.length);
            graph.tensors = [];
            for (let i = 0; i < graph.nodes.length; i++) {
                const node = new dlc.schema.v3.Node();
                node.type = 'Weights';
                node.name = params.nodes[i].name;
                node.inputs = [];
                node.outputs = [];
                node.attributes = [];
                graph.nodes[i] = node;
            }
        }
        const dataType = (value) => {
            switch (value) {
                case null: return '?';
                case 6: return 'uint8';
                case 9: return 'float32';
                default:
                    throw new dlc.Error(`Unsupported data type '${JSON.stringify(value)}'.`);
            }
        };
        const weights = new Map(params.nodes.map((node) => [node.name, node.weights]));
        for (const node of graph.nodes) {
            if (weights.has(node.name)) {
                const tensors = weights.get(node.name);
                for (const tensor of tensors) {
                    tensor.data.dtype = dataType(tensor.data.dtype);
                }
                node.weights = tensors;
            }
        }
        return graph;
    }

    static _params4(stream, graphs, signature) {
        let buffer = stream.peek().subarray(8);
        let buffers = null;
        if (signature === '4.NR64') {
            try {
                const reader = flatbuffers.BinaryReader.open(buffer);
                const nr64 = dlc.schema.v4.ModelParameters64.decode(reader, reader.root);
                buffers = nr64.buffers;
                buffer = nr64.params;
            } catch (error) {
                const message = error && error.message ? error.message : error.toString();
                throw new dlc.Error(`File format is not dlc.v4.NR64 (${message.replace(/\.$/, '')}).`);
            }
        }
        let params = null;
        try {
            const reader = flatbuffers.BinaryReader.open(buffer);
            params = dlc.schema.v4.ModelParameters.decode(reader, reader.root);
        } catch (error) {
            const message = error && error.message ? error.message : error.toString();
            throw new dlc.Error(`File format is not dlc.v4.NETP (${message.replace(/\.$/, '')}).`);
        }
        if (graphs.length === 0) {
            throw new dlc.Error('Model definition not available.');
        }
        const weights = new Map(params.graphs.map((graph) => [graph.name, graph]));
        for (const graph of graphs) {
            const params = weights.get(graph.name);
            const tensors = new Map(params.tensors.map((tensor) => [tensor.name, tensor]));
            let index = 0;
            graph.tensors.sort((a, b) => a.name.localeCompare(b.name));
            for (const tensor of graph.tensors) {
                if (tensor.location === 4) {
                    tensor.data = buffers ? buffers[index++].bytes : tensors.get(tensor.name).bytes;
                }
            }
            for (let i = 0; i < graph.nodes.length; i++) {
                const node = graph.nodes[i];
                const tensors = new Map(params.nodes[i].tensors.map((tensor) => [tensor.name, tensor]));
                for (const attribute of node.attributes) {
                    const tensor = attribute.tensor;
                    if (tensor) {
                        tensor.data = buffers ? buffers[index++].bytes : tensors.get(tensor.name).bytes;
                    }
                }
            }
        }
    }

    async _fetch(name) {
        try {
            const context = await this.context.fetch(name);
            return context.stream;
        } catch {
            return null;
        }
    }

    static _signature(stream) {
        if (stream) {
            const buffer = stream.peek(Math.min(stream.length, 16));
            const match = (signature) => buffer.length >= signature.length && signature.every((value, index) => value === buffer[index]);
            if (match([0xD5, 0x0A, 0x04, 0x00, 0x00, 0x00, 0x00, 0x00]) && buffer.length >= 16) {
                const reader = flatbuffers.BinaryReader.open(stream, 8);
                if (reader) {
                    return `4.${reader.identifier}`;
                }
            }
            if (match([0xD5, 0x0A, 0x03, 0x00, 0x00, 0x00, 0x00, 0x00]) && buffer.length >= 16) {
                const reader = flatbuffers.BinaryReader.open(stream, 8);
                if (reader) {
                    return `3.${reader.identifier}`;
                }
            }
            if (match([0xD5, 0x0A, 0x02, 0x00])) {
                return '2';
            }
            const reader = flatbuffers.BinaryReader.open(stream);
            if (reader) {
                return reader.identifier;
            }
        }
        return '';
    }
};

dlc.Utility = class {

    static enum(version, name, value) {
        switch (version) {
            case 3: version = 'v3'; break;
            case 4: version = 'v4'; break;
            default: version = '';
        }
        const schema = dlc.schema[version];
        if (schema && name) {
            const type = schema[name];
            if (type) {
                dlc.Utility[version] = dlc.Utility[version] || new Map();
                const enums = dlc.Utility[version];
                if (!enums.has(name)) {
                    const entries = new Map(Object.entries(type).map(([key, value]) => [value, key]));
                    enums.set(name, entries);
                }
                const values = enums.get(name);
                if (values.has(value)) {
                    return values.get(value);
                }
            }
        }
        return value;
    }
};

dlc.Error = class extends Error {

    constructor(message) {
        super(message);
        this.name = 'Error loading DLC model.';
    }
};

export const ModelFactory = dlc.ModelFactory;

