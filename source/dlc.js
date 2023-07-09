
var dlc = {};
var text = require('./text');

dlc.ModelFactory = class {

    match(context) {
        return dlc.Container.open(context);
    }

    async open(context, target) {
        await context.require('./dlc-schema');
        dlc.schema = flatbuffers.get('dlc').dlc;
        const container = target;
        let model = null;
        let params = null;
        const metadata_props = container.metadata;
        container.validate();
        try {
            model = container.model;
        } catch (error) {
            const message = error && error.message ? error.message : error.toString();
            throw new dlc.Error('File format is not dlc.NetDef (' + message.replace(/\.$/, '') + ').');
        }
        try {
            params = container.params;
        } catch (error) {
            const message = error && error.message ? error.message : error.toString();
            throw new dlc.Error('File format is not dlc.NetParam (' + message.replace(/\.$/, '') + ').');
        }
        const metadata = await context.metadata('dlc-metadata.json');
        return new dlc.Model(metadata, model, params, metadata_props);
    }
};

dlc.Model = class {

    constructor(metadata, model, params, metadata_props) {
        this._format = model ? 'DLC' : 'DLC Weights';
        this._metadata = [];
        if (metadata_props.size > 0) {
            const version = metadata_props.get('model-version');
            if (version) {
                this._version = version;
            }
            const converter = metadata_props.get('converter-command');
            if (converter) {
                const source = converter.split(' ').shift().trim();
                if (source.length > 0) {
                    const version = metadata_props.get('converter-version');
                    this._metadata.push({ name: 'source', value: version ? source + ' v' + version : source });
                }
            }
        }
        this._graphs = [ new dlc.Graph(metadata, model, params) ];
    }

    get format() {
        return this._format;
    }

    get version() {
        return this._version;
    }

    get metadata() {
        return this._metadata;
    }

    get graphs() {
        return this._graphs;
    }
};

dlc.Graph = class {

    constructor(metadata, model, params) {
        this._inputs = [];
        this._outputs = [];
        const values = new Map();
        const value = (name) => {
            if (!values.has(name)) {
                values.set(name, new dlc.Value(name));
            }
            return values.get(name);
        };
        if (model) {
            for (const node of model.nodes) {
                for (const input of node.inputs) {
                    if (!values.has(input)) {
                        values.set(input, {});
                    }
                }
                const shapes = new Array(node.outputs.length);
                for (const attr of node.attributes) {
                    if (attr.name === 'OutputDims') {
                        for (const entry of Object.entries(attr.attributes)) {
                            const index = parseInt(entry[0], 10);
                            shapes[index] = Array.from(entry[1].int32_list);
                        }
                        break;
                    }
                }
                for (let i = 0; i < node.outputs.length; i++) {
                    const output = node.outputs[i];
                    if (!values.has(output)) {
                        values.set(output, {});
                    }
                    const value = values.get(output);
                    if (i < shapes.length) {
                        value.shape = shapes[i];
                    }
                }
            }
            for (const entry of values) {
                const type = entry[1].shape ? new dlc.TensorType(null, entry[1].shape) : null;
                const value = new dlc.Value(entry[0], type);
                values.set(entry[0], value);
            }
            this._nodes = [];
            const weights = new Map(params ? params.weights.map((weights) => [ weights.name, weights ]) : []);
            for (const node of model.nodes) {
                if (node.type === 'Input') {
                    this._inputs.push(new dlc.Argument(node.name, node.inputs.map((input) => value(input))));
                    continue;
                }
                this._nodes.push(new dlc.Node(metadata, node, weights.get(node.name), value));
            }
        } else {
            this._nodes = params.weights.map((weights) => new dlc.Node(metadata, null, weights, value));
        }
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

dlc.Argument = class {

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

dlc.Value = class {

    constructor(name, type, initializer) {
        if (typeof name !== 'string') {
            throw new dlc.Error("Invalid value identifier '" + JSON.stringify(name) + "'.");
        }
        this._name = name;
        this._type = type;
        this._initializer = initializer;
    }

    get name() {
        return this._name;
    }

    get type() {
        return this._type;
    }

    get initializer() {
        return this._initializer;
    }
};

dlc.Node = class {

    constructor(metadata, node, weights, value) {
        if (node) {
            this._type = metadata.type(node.type);
            this._name = node.name;
            const inputs = Array.from(node.inputs).map((input) => value(input));
            this._inputs = inputs.length === 0 ? [] : [ new dlc.Argument(inputs.length === 1 ? 'input' : 'inputs', inputs) ];
            const outputs = Array.from(node.outputs).map((output) => value(output));
            this._outputs = outputs.length === 0 ? [] : [ new dlc.Argument(outputs.length === 1 ? 'output' : 'outputs', outputs) ];
            this._attributes = [];
            for (const attr of node.attributes) {
                if (attr.name === 'OutputDims') {
                    continue;
                }
                const attribute = new dlc.Attribute(metadata.attribute(node.type, attr.name), attr);
                this._attributes.push(attribute);
            }
            if (weights) {
                for (const tensor of weights.tensors) {
                    const type = new dlc.TensorType(tensor.data.data_type, tensor.shape);
                    const value = new dlc.Value('', type, new dlc.Tensor(type, tensor.data));
                    this._inputs.push(new dlc.Argument(tensor.name, [ value ]));
                }
            }
        } else {
            this._type = { name: 'Weights' };
            this._name = weights.name;
            this._inputs = weights.tensors.map((tensor) => {
                const type = new dlc.TensorType(tensor.data.data_type, tensor.shape);
                const value = new dlc.Value('', type, new dlc.Tensor(type, tensor.data));
                return new dlc.Argument(tensor.name, [ value ]);
            });
            this._outputs = [];
            this._attributes = [];
        }
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

dlc.Attribute = class {

    constructor(metadata, attr) {
        this._name = attr.name;
        const read = (attr) => {
            switch (attr.type) {
                case 1: return [ 'boolean',   attr.bool_value               ];
                case 2: return [ 'int32',     attr.int32_value              ];
                case 3: return [ 'uint32',    attr.uint32_value             ];
                case 4: return [ 'float32',   attr.float32_value            ];
                case 5: return [ 'string',    attr.string_value             ];
                case 7: return [ 'byte[]',    Array.from(attr.byte_list)    ];
                case 8: return [ 'int32[]',   Array.from(attr.int32_list)   ];
                case 9: return [ 'float32[]', Array.from(attr.float32_list) ];
                case 11: {
                    const obj = {};
                    for (const attribute of attr.attributes) {
                        const entry = read(attribute);
                        obj[attribute.name] = entry[1];
                    }
                    return [ '', obj ];
                }
                default:
                    throw new dlc.Error("Unsupported attribute type '" + attr.type + "'.");
            }
        };
        const entry = read(attr);
        if (entry) {
            this._type = entry[0];
            this._value = entry[1];
        }
        if (metadata && metadata.type) {
            this._type = metadata.type;
            this._value = dlc.Utility.enum(this._type, this._value);
        }
    }

    get name() {
        return this._name;
    }

    get type() {
        return this._type;
    }

    get value() {
        return this._value;
    }
};

dlc.TensorType = class {

    constructor(dataType, shape) {
        switch (dataType) {
            case null: this._dataType = '?'; break;
            case 6: this._dataType = 'uint8'; break;
            case 9: this._dataType = 'float32'; break;
            default:
                throw new dlc.Error("Unsupported data type '" + JSON.stringify(dataType) + "'.");
        }
        this._shape = new dlc.TensorShape(shape);
    }

    get dataType() {
        return this._dataType;
    }

    get shape() {
        return this._shape;
    }

    toString() {
        return this.dataType + this._shape.toString();
    }
};

dlc.TensorShape = class {

    constructor(dimensions) {
        this._dimensions = Array.from(dimensions);
    }

    get dimensions() {
        return this._dimensions;
    }

    toString() {
        if (!this._dimensions || this._dimensions.length == 0) {
            return '';
        }
        return '[' + this._dimensions.map((dimension) => dimension.toString()).join(',') + ']';
    }
};

dlc.Tensor = class {

    constructor(type, data) {
        this._type = type;
        switch (type.dataType) {
            case 'uint8': this._values = data.bytes; break;
            case 'float32': this._values = data.floats; break;
            default: throw new dlc.Error("Unsupported tensor data type '" + type.dataType + "'.");
        }
    }

    get type() {
        return this._type;
    }

    get layout() {
        return '|';
    }

    get values() {
        return this._values;
    }
};

dlc.Container = class {

    static open(context) {
        const entries = context.entries('zip');
        if (entries.size > 0) {
            const model = entries.get('model');
            const params = entries.get('model.params');
            const metadata = entries.get('dlc.metadata');
            if (model || params) {
                return new dlc.Container(model, params, metadata);
            }
        }
        const stream = context.stream;
        switch (dlc.Container._signature(stream).split('.').pop()) {
            case 'NETD':
                return new dlc.Container(stream, null, null);
            case 'NETP':
                return new dlc.Container(null, stream, null);
            default:
                return null;
        }
    }

    constructor(model, params, metadata) {
        this._model = { stream: model || null };
        this._params = { stream: params || null };
        this._metadata = { stream: metadata || null, value: new Map() };
    }

    validate() {
        this._model.signature = dlc.Container._signature(this._model.stream);
        this._params.signature = dlc.Container._signature(this._params.stream);
        if (this._model.signature == '2' ||this._params.signature == '2') {
            throw new dlc.Error("File contains undocumented DLC v2 data.");
        }
        if (this._model.signature.startsWith('4.') || this._params.signature.startsWith('4.')) {
            throw new dlc.Error("File contains undocumented DLC v4 data.");
        }
    }

    get model() {
        if (this._model && this._model.stream) {
            const stream = this._model.stream;
            delete this._model.stream;
            switch (dlc.Container._signature(stream)) {
                case '4.NETD': {
                    throw new dlc.Error("File contains undocumented DLC v4 data.");
                }
                case '3.NETD': {
                    const buffer = new Uint8Array(stream.peek().subarray(8));
                    const reader = flatbuffers.BinaryReader.open(buffer);
                    this._model.value = dlc.schema.NetDef.decode(reader, reader.root);
                    break;
                }
                case 'NETD': {
                    const buffer = stream.peek();
                    const reader = flatbuffers.BinaryReader.open(buffer);
                    this._model.value = dlc.schema.NetDef.decode(reader, reader.root);
                    break;
                }
                default: {
                    const buffer = stream.peek(Math.min(stream.length, 16));
                    const content = Array.from(buffer).map((c) => (c < 16 ? '0' : '') + c.toString(16)).join('');
                    throw new dlc.Error("File contains undocumented '" + content + "' data.");
                }
            }
        }
        return this._model.value;
    }

    get params() {
        if (this._params && this._params.stream) {
            const stream = this._params.stream;
            delete this._params.stream;
            switch (dlc.Container._signature(stream)) {
                case '4.NETP': {
                    throw new dlc.Error("File contains undocumented DLC v4 data.");
                }
                case '3.NETP': {
                    const buffer = new Uint8Array(stream.peek().subarray(8));
                    const reader = flatbuffers.BinaryReader.open(buffer);
                    this._params.value = dlc.schema.NetParam.decode(reader, reader.root);
                    break;
                }
                case '2': {
                    throw new dlc.Error("File contains undocumented DLC v2 data.");
                }
                case 'NETP': {
                    const buffer = stream.peek();
                    const reader = flatbuffers.BinaryReader.open(buffer);
                    this._params.value = dlc.schema.NetParam.decode(reader, reader.root);
                    break;
                }
                default: {
                    const buffer = stream.peek(Math.min(stream.length, 16));
                    const content = Array.from(buffer).map((c) => (c < 16 ? '0' : '') + c.toString(16)).join('');
                    throw new dlc.Error("File contains undocumented '" + content + "' data.");
                }
            }
        }
        return this._params.value;
    }

    get metadata() {
        if (this._metadata.stream) {
            const stream = this._metadata.stream;
            delete this._metadata.stream;
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
                this._metadata.value.set(key, value);
            }
        }
        return this._metadata.value;
    }

    static _signature(stream) {
        if (stream) {
            const buffer = stream.peek(Math.min(stream.length, 16));
            const match = (signature) => buffer.length >= signature.length && signature.every((value, index) => value === buffer[index]);
            if (match([ 0xD5, 0x0A, 0x04, 0x00, 0x00, 0x00, 0x00, 0x00 ]) && buffer.length >= 16) {
                const reader = flatbuffers.BinaryReader.open(buffer.slice(8));
                return '4.' + reader.identifier;
            }
            if (match([ 0xD5, 0x0A, 0x03, 0x00, 0x00, 0x00, 0x00, 0x00 ]) && buffer.length >= 16) {
                const reader = flatbuffers.BinaryReader.open(buffer.slice(8));
                return '3.' + reader.identifier;
            }
            if (match([ 0xD5, 0x0A, 0x02, 0x00 ])) {
                return '2';
            }
            if (buffer.length >= 8) {
                const reader = flatbuffers.BinaryReader.open(buffer);
                return reader.identifier;
            }
        }
        return '';
    }
};

dlc.Utility = class {

    static enum(name, value) {
        const type = name && dlc.schema ? dlc.schema[name] : undefined;
        if (type) {
            dlc.Utility._enums = dlc.Utility._enums || new Map();
            if (!dlc.Utility._enums.has(name)) {
                const map = new Map(Object.keys(type).map((key) => [ type[key], key ]));
                dlc.Utility._enums.set(name, map);
            }
            const map = dlc.Utility._enums.get(name);
            if (map.has(value)) {
                return map.get(value);
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

if (typeof module !== 'undefined' && typeof module.exports === 'object') {
    module.exports.ModelFactory = dlc.ModelFactory;
}
