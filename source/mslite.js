
const mslite = {};

mslite.ModelFactory = class {

    match(context) {
        const extension = context.identifier.split('.').pop().toLowerCase();
        const reader = context.peek('flatbuffers.binary');
        if (reader) {
            const identifier = reader.identifier;
            if (identifier === 'MSL1' || identifier === 'MSL2' || (identifier === '' && extension === 'ms')) {
                context.type = 'mslite';
                context.target = reader;
            }
        }
    }

    async open(context) {
        const reader = context.target;
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
                throw new mslite.Error(`Unsupported file identifier '${reader.identifier}'.`);
        }
        mslite.schema = await context.require('./mslite-schema');
        mslite.schema = mslite.schema.mindspore.schema;
        let model = null;
        try {
            model = mslite.schema.MetaGraph.create(reader);
        } catch (error) {
            const message = error && error.message ? error.message : error.toString();
            throw new mslite.Error(`File format is not mslite.MetaGraph (${message.replace(/\.$/, '')}).`);
        }
        const metadata = await context.metadata('mslite-metadata.json');
        return new mslite.Model(metadata, model);
    }
};

mslite.Model = class {

    constructor(metadata, model) {
        this.name = model.name || '';
        this.graphs = [];
        const version = model.version ? model.version.match(/^.*(\d\.\d\.\d)$/) : null;
        this.format = `MindSpore Lite${version ? ` v${version[1]}` : ''}`;
        const subgraphs = model.subGraph;
        if (Array.isArray(subgraphs)) {
            for (const subgraph of subgraphs) {
                this.graphs.push(new mslite.Graph(metadata, subgraph, model));
            }
        } else {
            const graph = new mslite.Graph(metadata, model, model);
            this.graphs.push(graph);
        }
    }
};

mslite.Graph = class {

    constructor(metadata, subgraph, model) {
        this.name = subgraph.name || '';
        this.inputs = [];
        this.outputs = [];
        this.nodes = [];
        const values = model.allTensors.map((tensor, index) => {
            const name = tensor.name || index.toString();
            const data = tensor.data;
            const type = new mslite.TensorType(tensor.dataType, tensor.dims);
            const initializer = (data && data.length > 0) ? new mslite.Tensor(type, tensor.data) : null;
            return new mslite.Value(name, tensor, initializer);
        });
        if (subgraph === model) {
            for (let i = 0; i < subgraph.inputIndex.length; i++) {
                const index = subgraph.inputIndex[i];
                this.inputs.push(new mslite.Argument(i.toString(), [values[index]]));
            }
            for (let i = 0; i < subgraph.outputIndex.length; i++) {
                const index = subgraph.outputIndex[i];
                this.outputs.push(new mslite.Argument(i.toString(), [values[index]]));
            }
            for (let i = 0; i < subgraph.nodes.length; i++) {
                this.nodes.push(new mslite.Node(metadata, subgraph.nodes[i], values));
            }
        } else {
            for (let i = 0; i < subgraph.inputIndices.length; i++) {
                const index = subgraph.inputIndices[i];
                this.inputs.push(new mslite.Argument(i.toString(), [values[index]]));
            }
            for (let i = 0; i < subgraph.outputIndices.length; i++) {
                const index = subgraph.outputIndices[i];
                this.outputs.push(new mslite.Argument(i.toString(), [values[index]]));
            }
            for (const name of subgraph.nodeIndices) {
                const node = new mslite.Node(metadata, model.nodes[name], values);
                this.nodes.push(node);
            }
        }
    }
};

mslite.Node = class {

    constructor(metadata, op, values) {
        this.name = op.name || '';
        this.type = { name: '?' };
        this.attributes = [];
        this.inputs = [];
        this.outputs = [];
        const data = op.primitive.value;
        if (data && data.constructor) {
            const type = data.constructor.name;
            this.type = metadata.type(type);
            this.attributes = Object.entries(data).map(([key, obj]) => {
                let value = ArrayBuffer.isView(obj) ? Array.from(obj) : obj;
                let type = null;
                const schema = metadata.attribute(this.type.name, key);
                if (schema && schema.type) {
                    type = schema.type;
                    value = type ? mslite.Utility.enum(type, value) : value;
                }
                return new mslite.Argument(key.toString(), value, type);
            });
        }
        const input_num = op.inputIndex.length;
        let i = 0;
        if (this.type && this.type.inputs) {
            for (const input of this.type.inputs) {
                if (i >= input_num) {
                    break;
                }
                const index = op.inputIndex[i];
                const argument = new mslite.Argument(input.name, [values[index]]);
                this.inputs.push(argument);
                i += 1;
            }
        }
        for (let j = i; j < input_num; j++) {
            const index = op.inputIndex[j];
            const argument = new mslite.Argument(j.toString(), [values[index]]);
            this.inputs.push(argument);
        }
        const output_num = op.outputIndex.length;
        i = 0;
        if (this.type && this.type.outputs) {
            for (const output of this.type.outputs) {
                if (i >= output_num) {
                    break;
                }
                const index = op.outputIndex[i];
                const argument = new mslite.Argument(output.name, [values[index]]);
                this.outputs.push(argument);
                i += 1;
            }
        }
        for (let j = i; j < output_num; j++) {
            const index = op.outputIndex[j];
            const argument = new mslite.Argument(j.toString(), [values[index]]);
            this.outputs.push(argument);
        }
    }
};

mslite.Argument = class {

    constructor(name, value, type) {
        this.name = name;
        this.value = value;
        this.type = type || null;
    }
};

mslite.Value = class {

    constructor(name, tensor, initializer) {
        this.name = name;
        this.type = initializer ? initializer.type : new mslite.TensorType(tensor.dataType, tensor.dims);
        this.initializer = initializer || null;
        if (Array.isArray(tensor.quantParams) && tensor.quantParams.length > 0) {
            this.quantization = {
                type: 'linear',
                scale: [],
                offset: []
            };
            for (let i = 0; i < tensor.quantParams.length; i++) {
                const param = tensor.quantParams[i];
                this.quantization.scale.push(param.scale);
                this.quantization.offset.push(param.zeroPoint);
            }
        }
    }
};

mslite.Tensor = class {

    constructor(type, data) {
        this.type = type;
        this.encoding = type.dataType === 'string' ? '|' : '<';
        this._data = data || null;
    }

    get values() {
        switch (this.type.dataType) {
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
            case 0:  this.dataType = "?"; break;
            case 1:  this.dataType = "type"; break;
            case 2:  this.dataType = "any"; break;
            case 3:  this.dataType = "object"; break;
            case 4:  this.dataType = "typetype"; break;
            case 5:  this.dataType = "problem"; break;
            case 6:  this.dataType = "external"; break;
            case 7:  this.dataType = "none"; break;
            case 8:  this.dataType = "null"; break;
            case 9:  this.dataType = "ellipsis"; break;
            case 11: this.dataType = "number"; break;
            case 12: this.dataType = "string"; break;
            case 13: this.dataType = "list"; break;
            case 14: this.dataType = "tuple"; break;
            case 15: this.dataType = "slice"; break;
            case 16: this.dataType = "keyword"; break;
            case 17: this.dataType = "tensortype"; break;
            case 18: this.dataType = "rowtensortype"; break;
            case 19: this.dataType = "sparsetensortype"; break;
            case 20: this.dataType = "undeterminedtype"; break;
            case 21: this.dataType = "class"; break;
            case 22: this.dataType = "dictionary"; break;
            case 23: this.dataType = "function"; break;
            case 24: this.dataType = "jtagged"; break;
            case 25: this.dataType = "symbolickeytype"; break;
            case 26: this.dataType = "envtype"; break;
            case 27: this.dataType = "refkey"; break;
            case 28: this.dataType = "ref"; break;
            case 30: this.dataType = "boolean"; break;
            // case 31: this.dataType = "int"; break;
            case 32: this.dataType = "int8"; break;
            case 33: this.dataType = "int16"; break;
            case 34: this.dataType = "int32"; break;
            case 35: this.dataType = "int64"; break;
            // case 36: this.dataType = "uint"; break;
            case 37: this.dataType = "uint8"; break;
            case 38: this.dataType = "uint16"; break;
            case 39: this.dataType = "uint32"; break;
            case 40: this.dataType = "uint64"; break;
            // case 41: this.dataType = "float"; break;
            case 42: this.dataType = "float16"; break;
            case 43: this.dataType = "float32"; break;
            case 44: this.dataType = "float64"; break;
            case 45: this.dataType = "bfloat16"; break;
            // case 46: this.dataType = "double"; break;
            // case 47: this.dataType = "complex"; break;
            case 48: this.dataType = "complex64"; break;
            case 49: this.dataType = "complex128"; break;
            case 50: this.dataType = "int4"; break;
            default: throw new mslite.Error(`Unsupported data type '${dataType}'.`);
        }
        this.shape = new mslite.TensorShape(Array.from(dimensions));
    }

    toString() {
        return this.dataType + this.shape.toString();
    }
};

mslite.TensorShape = class {

    constructor(dimensions) {
        this.dimensions = dimensions;
    }

    toString() {
        if (this.dimensions && this.dimensions.length > 0) {
            return `[${this.dimensions.map((dimension) => dimension ? dimension.toString() : '?').join(',')}]`;
        }
        return '';
    }
};

mslite.Utility = class {

    static enum(name, value) {
        mslite.Utility._enumKeyMap = mslite.Utility._enumKeyMap || new Map();
        if (!mslite.Utility._enumKeyMap.has(name)) {
            const type = name && mslite.schema ? mslite.schema[name] : undefined;
            if (type) {
                if (!mslite.Utility._enumKeyMap.has(name)) {
                    const entries = new Map(Object.entries(type).map(([key, value]) => [value, key]));
                    mslite.Utility._enumKeyMap.set(name, entries);
                }
            }
        }
        const map = mslite.Utility._enumKeyMap.get(name);
        if (map && map.has(value)) {
            return map.get(value);
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

export const ModelFactory = mslite.ModelFactory;
