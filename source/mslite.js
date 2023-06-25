
var mslite = {};
var flatbuffers = require('./flatbuffers');

mslite.ModelFactory = class {

    match(context) {
        const stream = context.stream;
        if (stream && stream.length >= 8) {
            const buffer = stream.peek(8);
            const reader = flatbuffers.BinaryReader.open(buffer);
            if (reader.identifier === '' || reader.identifier === 'MSL1' || reader.identifier === 'MSL2') {
                return 'mslite';
            }
        }
        return '';
    }

    async open(context) {
        await context.require('./mslite-schema');
        const stream = context.stream;
        const reader = flatbuffers.BinaryReader.open(stream);
        switch (reader.identifier) {
            case '': {
                throw new mslite.Error('MSL0 format is deprecated.');
            }
            case 'MSL1': {
                throw new mslite.Error('MSL1 format is deprecated.');
            }
            case 'MSL2':
                break;
            default:
                throw new mslite.Error("Unsupported file identifier '" + reader.identifier + "'.");
        }
        let model = null;
        try {
            mslite.schema = flatbuffers.get('mslite').mindspore.schema;
            model = mslite.schema.MetaGraph.create(reader);
        } catch (error) {
            const message = error && error.message ? error.message : error.toString();
            throw new mslite.Error('File format is not mslite.MetaGraph (' + message.replace(/\.$/, '') + ').');
        }
        const metadata = await context.metadata('mslite-metadata.json');
        return new mslite.Model(metadata, model);
    }
};

mslite.Model = class {

    constructor(metadata, model) {
        this._name = model.name || '';
        this._graphs = [];
        const version = model.version ? model.version.match(/^.*(\d\.\d\.\d)$/) : null;
        this._format = 'MindSpore Lite' + (version ? ' v' + version[1] : '');
        const subgraphs = model.subGraph;
        if (Array.isArray(subgraphs)) {
            for (const subgraph of subgraphs) {
                this._graphs.push(new mslite.Graph(metadata, subgraph, model));
            }
        } else {
            this._graphs.push(new mslite.Graph(metadata, model, model));
        }
    }

    get name() {
        return this._name;
    }

    get format() {
        return this._format;
    }

    get graphs() {
        return this._graphs;
    }
};

mslite.Graph = class {

    constructor(metadata, subgraph, model) {
        this._name = subgraph.name || '';
        this._inputs = [];
        this._outputs = [];
        this._nodes = [];
        const args = model.allTensors.map((tensor, index) => {
            const name = tensor.name || index.toString();
            const data = tensor.data;
            const type = new mslite.TensorType(tensor.dataType, tensor.dims);
            const initializer = (data && data.length > 0) ? new mslite.Tensor(type, tensor.data) : null;
            return new mslite.Value(name, tensor, initializer);
        });
        if (subgraph === model) {
            for (let i = 0; i < subgraph.inputIndex.length; i++) {
                const index = subgraph.inputIndex[i];
                this._inputs.push(new mslite.Argument(i.toString(), [ args[index] ]));
            }
            for (let i = 0; i < subgraph.outputIndex.length; i++) {
                const index = subgraph.outputIndex[i];
                this._outputs.push(new mslite.Argument(i.toString(), [ args[index] ]));
            }
            for (let i = 0; i < subgraph.nodes.length; i++) {
                this._nodes.push(new mslite.Node(metadata, subgraph.nodes[i], args));
            }
        } else {
            for (let i = 0; i < subgraph.inputIndices.length; i++) {
                const index = subgraph.inputIndices[i];
                this._inputs.push(new mslite.Argument(i.toString(), [args[index]]));
            }
            for (let i = 0; i < subgraph.outputIndices.length; i++) {
                const index = subgraph.outputIndices[i];
                this._outputs.push(new mslite.Argument(i.toString(), [args[index]]));
            }
            for (let i = 0; i < subgraph.nodeIndices.length; i++) {
                const nodeId = subgraph.nodeIndices[i];
                this._nodes.push(new mslite.Node(metadata, model.nodes[nodeId], args));
            }
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
};

mslite.Node = class {

    constructor(metadata, op, args) {
        this._name = op.name || '';
        this._type = { name: '?' };
        this._attributes = [];
        this._inputs = [];
        this._outputs = [];

        const data = op.primitive.value;
        if (data && data.constructor) {
            const type = data.constructor.name;
            this._type = metadata.type(type);
            this._attributes = Object.keys(data).map((key) => new mslite.Attribute(metadata.attribute(type, key), key.toString(), data[key]));
        }

        const input_num = op.inputIndex.length;
        let i = 0;
        if (this._type && this._type.inputs) {
            for (const input of this._type.inputs) {
                if (i >= input_num) {
                    break;
                }
                const index = op.inputIndex[i];
                this._inputs.push(new mslite.Argument(input.name, [ args[index] ]));
                i += 1;
            }
        }
        for (let j = i; j < input_num; j++) {
            const index = op.inputIndex[j];
            this._inputs.push(new mslite.Argument(j.toString(), [ args[index] ]));
        }

        const output_num = op.outputIndex.length;
        i = 0;
        if (this._type && this._type.outputs) {
            for (const output of this._type.outputs) {
                if (i >= output_num) {
                    break;
                }
                const index = op.outputIndex[i];
                this._outputs.push(new mslite.Argument(output.name, [ args[index] ]));
                i += 1;
            }
        }
        for (let j = i; j < output_num; j++) {
            const index = op.outputIndex[j];
            this._outputs.push(new mslite.Argument(j.toString(), [ args[index] ]));
        }
    }

    get name() {
        return this._name;
    }

    get type() {
        return this._type;
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

mslite.Attribute = class {

    constructor(schema, attrName, value) {
        this._type = null;
        this._name = attrName;
        this._visible = false;
        this._value = ArrayBuffer.isView(value) ? Array.from(value) : value;
        if (schema) {
            if (schema.type) {
                this._type = schema.type;
                if (this._type) {
                    this._value = mslite.Utility.enum(this._type, this._value);
                }
            }
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

    get visible() {
        return this._visible !== false;
    }
};

mslite.Argument = class {

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

mslite.Value = class {

    constructor(name, tensor, initializer) {
        this._name = name;
        this._type = initializer ? null : new mslite.TensorType(tensor.dataType, tensor.dims);
        this._initializer = initializer || null;

        if (tensor.quantParams) {
            const list = [];
            for (let i = 0; i < tensor.quantParams.length; i++) {
                const param = tensor.quantParams[i];
                if (param.scale !== 0 || param.zeroPoint !== 0) {
                    const scale = param.scale;
                    const zeroPoint = param.zeroPoint;
                    let quantization = '';
                    if (scale !== 1) {
                        quantization += scale.toString() + ' * ';
                    }
                    if (zeroPoint === 0) {
                        quantization += 'q';
                    } else if (zeroPoint < 0) {
                        quantization += '(q + ' + -zeroPoint + ')';
                    } else if (zeroPoint > 0) {
                        quantization += '(q - ' + zeroPoint + ')';
                    }
                    list.push(quantization);
                }
            }
            if (list.length > 0 && !list.every((value) => value === 'q')) {
                this._quantization = list.length === 1 ? list[0] : list;
            }
        }
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

    get quantization() {
        return this._quantization;
    }
};

mslite.Tensor = class {

    constructor(type, data) {
        this._type = type;
        this._data = data || null;
    }

    get type() {
        return this._type;
    }

    get layout() {
        switch (this._type.dataType) {
            case 'string': return '|';
            default: return '<';
        }
    }

    get values() {
        switch (this._type.dataType) {
            case 'string': {
                let offset = 0;
                const data = new DataView(this._data.buffer, this._data.byteOffset, this._data.byteLength);
                const count = data.getInt32(0, true);
                offset += 4;
                const offsetTable = [];
                for (let j = 0; j < count; j++) {
                    offsetTable.push(data.getInt32(offset, true));
                    offset += 4;
                }
                offsetTable.push(this._data.length);
                const stringTable = [];
                const utf8Decoder = new TextDecoder('utf-8');
                for (let k = 0; k < count; k++) {
                    const textArray = this._data.subarray(offsetTable[k], offsetTable[k + 1]);
                    stringTable.push(utf8Decoder.decode(textArray));
                }
                return stringTable;
            }
            default: return this._data;
        }
    }
};

mslite.TensorType = class {

    constructor(dataType, dimensions) {
        switch (dataType) {
            case 0:  this._dataType = "?"; break;
            case 1:  this._dataType = "type"; break;
            case 2:  this._dataType = "any"; break;
            case 3:  this._dataType = "object"; break;
            case 4:  this._dataType = "typetype"; break;
            case 5:  this._dataType = "problem"; break;
            case 6:  this._dataType = "external"; break;
            case 7:  this._dataType = "none"; break;
            case 8:  this._dataType = "null"; break;
            case 9:  this._dataType = "ellipsis"; break;
            case 11: this._dataType = "number"; break;
            case 12: this._dataType = "string"; break;
            case 13: this._dataType = "list"; break;
            case 14: this._dataType = "tuple"; break;
            case 15: this._dataType = "slice"; break;
            case 16: this._dataType = "keyword"; break;
            case 17: this._dataType = "tensortype"; break;
            case 18: this._dataType = "rowtensortype"; break;
            case 19: this._dataType = "sparsetensortype"; break;
            case 20: this._dataType = "undeterminedtype"; break;
            case 21: this._dataType = "class"; break;
            case 22: this._dataType = "dictionary"; break;
            case 23: this._dataType = "function"; break;
            case 24: this._dataType = "jtagged"; break;
            case 25: this._dataType = "symbolickeytype"; break;
            case 26: this._dataType = "envtype"; break;
            case 27: this._dataType = "refkey"; break;
            case 28: this._dataType = "ref"; break;
            case 30: this._dataType = "boolean"; break;
            case 31: this._dataType = "int"; break;
            case 32: this._dataType = "int8"; break;
            case 33: this._dataType = "int16"; break;
            case 34: this._dataType = "int32"; break;
            case 35: this._dataType = "int64"; break;
            case 36: this._dataType = "uint"; break;
            case 37: this._dataType = "uint8"; break;
            case 38: this._dataType = "uint16"; break;
            case 39: this._dataType = "uint32"; break;
            case 40: this._dataType = "uint64"; break;
            case 41: this._dataType = "float"; break;
            case 42: this._dataType = "float16"; break;
            case 43: this._dataType = "float32"; break;
            case 44: this._dataType = "float64"; break;
            case 45: this._dataType = "complex64"; break;
            default: throw new mslite.Error("Unsupported data type '" + dataType.toString() + "'.");
        }
        this._shape = new mslite.TensorShape(Array.from(dimensions));
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

mslite.TensorShape = class {

    constructor(dimensions) {
        this._dimensions = dimensions;
    }

    get dimensions() {
        return this._dimensions;
    }

    toString() {
        if (this._dimensions && this._dimensions.length > 0) {
            return '[' + this._dimensions.map((dimension) => dimension ? dimension.toString() : '?').join(',') + ']';
        }
        return '';
    }
};

mslite.Utility = class {

    static enum(name, value) {
        const type = name && mslite.schema ? mslite.schema[name] : undefined;
        if (type) {
            mslite.Utility._enumKeyMap = mslite.Utility._enumKeyMap || new Map();
            if (!mslite.Utility._enumKeyMap.has(name)) {
                const map = new Map();
                for (const key of Object.keys(type)) {
                    map.set(type[key], key);
                }
                mslite.Utility._enumKeyMap.set(name, map);
            }
            const map = mslite.Utility._enumKeyMap.get(name);
            if (map.has(value)) {
                return map.get(value);
            }
        }
        return value;
    }
};

mslite.Error = class extends Error {

    constructor(message) {
        super(message);
        this.name = 'Error loading MindSpore Lite model.';
    }
};

if (typeof module !== 'undefined' && typeof module.exports === 'object') {
    module.exports.ModelFactory = mslite.ModelFactory;
}