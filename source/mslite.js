/* jshint esversion: 6 */

var mslite = mslite || {};
var flatbuffers = flatbuffers || require('./flatbuffers');

mslite.ModelFactory = class {

    match(context) {
        const stream = context.stream;
        if (stream.length >= 8) {
            const buffer = stream.peek(8);
            const reader = new flatbuffers.Reader(buffer);
            if (reader.identifier === '' || reader.identifier === 'MSL1' || reader.identifier === 'MSL2') {
                return true;
            }
        }
        return false;
    }

    open(context) {
        return context.require('./mslite-schema').then(() => {
            const buffer = context.stream.peek();
            const reader = new flatbuffers.Reader(buffer);
            switch (reader.identifier) {
                case '':
                    throw new mslite.Error('MSL0 format is deprecated.', false);
                case 'MSL1':
                    throw new mslite.Error('MSL1 format is deprecated.', false);
                case 'MSL2':
                    break;
            }
            let model = null;
            try {
                mslite.schema = flatbuffers.get('mslite').mindspore.schema;
                model = mslite.schema.MetaGraph.create(reader);
            }
            catch (error) {
                const message = error && error.message ? error.message : error.toString();
                throw new mslite.Error('File format is not mslite.MetaGraph (' + message.replace(/\.$/, '') + ').');
            }
            return mslite.Metadata.open(context).then((metadata) => {
                return new mslite.Model(metadata, model);
            });
        });
    }
};

mslite.Model = class {

    constructor(metadata, model) {
        this._name = model.name || '';
        this._format = model.version || '';
        this._graphs = [];
        const format = 'MindSpore Lite ';
        if (this._format.startsWith(format)) {
            const version = this._format.substring(format.length).replace(/^v/, '');
            this._format = format + 'v' + version;
        }
        const subgraphs = model.subGraph;
        if (Array.isArray(subgraphs)) {
            this._graphs.push(new mslite.Graph(metadata, model, model));
        }
        else {
            for (const subgraph of subgraphs) {
                this._graphs.push(new mslite.Graph(metadata, subgraph, model));
            }
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
            return new mslite.Argument(name, tensor, initializer);
        });
        if (subgraph === model) {
            for (let i = 0; i < subgraph.inputIndex.length; i++) {
                const index = subgraph.inputIndex[i];
                this._inputs.push(new mslite.Parameter(i.toString(), true, [ args[index] ]));
            }
            for (let i = 0; i < subgraph.outputIndex.length; i++) {
                const index = subgraph.outputIndex[i];
                this._outputs.push(new mslite.Parameter(i.toString(), true, [ args[index] ]));
            }
            for (let i = 0; i < subgraph.nodes.length; i++) {
                this._nodes.push(new mslite.Node(metadata, subgraph.nodes[i], args));
            }
        }
        else {
            for (let i = 0; i < subgraph.inputIndices.length; i++) {
                const index = subgraph.inputIndices[i];
                this._inputs.push(new mslite.Parameter(i.toString(), true, [args[index]]));
            }
            for (let i = 0; i < subgraph.outputIndices.length; i++) {
                const index = subgraph.outputIndices[i];
                this._outputs.push(new mslite.Parameter(i.toString(), true, [args[index]]));
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

    get groups() {
        return false;
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
        this._metadata = metadata;
        this._name = op.name || '';
        this._type = '?';
        this._attributes = [];
        this._inputs = [];
        this._outputs = [];

        let schema = null;
        const data = op.primitive.value;
        if (data && data.constructor) {
            this._type = data.constructor.name;
            schema = metadata.type(this._type);
            this._attributes = Object.keys(data).map((key) => new mslite.Attribute(metadata.attribute(this.type, key), key.toString(), data[key]));
        }

        const input_num = op.inputIndex.length;
        let i = 0;
        if (schema && schema.inputs){
            for (const input of schema.inputs) {
                if (i >= input_num) {
                    break;
                }
                const index = op.inputIndex[i];
                this._inputs.push(new mslite.Parameter(input.name, true, [ args[index] ]));
                i += 1;
            }
        }
        for (let j = i; j < input_num; j++) {
            const index = op.inputIndex[j];
            this._inputs.push(new mslite.Parameter(j.toString(), true, [ args[index] ]));
        }

        const output_num = op.outputIndex.length;
        i = 0;
        if (schema && schema.outputs){
            for (const output of schema.outputs) {
                if (i >= output_num) {
                    break;
                }
                const index = op.outputIndex[i];
                this._outputs.push(new mslite.Parameter(output.name, true, [ args[index] ]));
                i += 1;
            }
        }
        for (let j = i; j < output_num; j++) {
            const index = op.outputIndex[j];
            this._outputs.push(new mslite.Parameter(j.toString(), true, [ args[index] ]));
        }
    }

    get name() {
        return this._name;
    }

    get type() {
        return this._type;
    }

    get metadata() {
        return this._metadata.type(this.type);
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
        this._visible = true;
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

mslite.Parameter = class {

    constructor(name, visible, args) {
        this._name = name;
        this._visible = visible;
        this._arguments = args;
    }

    get name() {
        return this._name;
    }

    get visible() {
        return this._visible;
    }

    get arguments() {
        return this._arguments;
    }
};

mslite.Argument = class {

    constructor(name, tensor, initializer) {
        this._name = name;
        this._type = initializer ? null : new mslite.TensorType(tensor.dataType, tensor.dims);
        this._initializer = initializer || null;

        if (tensor.quantParams) {
            const params = [];
            for (let i = 0; i < tensor.quantParams.length; i++) {
                const param = tensor.quantParams[i];
                if (param.scale !== 0 || param.zeroPoint !== 0) {
                    params.push(param.scale.toString() + ' * x + ' + param.zeroPoint.toString());
                }
            }
            this._quantization = params.join(' -> ');
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
        return JSON.stringify(value, null, 4);
    }

    _context() {
        const context = {};
        context.state = null;
        context.index = 0;
        context.count = 0;

        if (this._data == null || this._data.length === 0) {
            context.state = 'Tensor data is empty.';
            return context;
        }

        context.dataType = this._type.dataType;
        context.shape = this._type.shape.dimensions;
        context.data = new DataView(this._data.buffer, this._data.byteOffset, this._data.byteLength);

        if (this._type.dataType === 'string') {
            let offset = 0;
            const count = context.data.getInt32(0, true);
            offset += 4;
            const offsetTable = [];
            for (let j = 0; j < count; j++) {
                offsetTable.push(context.data.getInt32(offset, true));
                offset += 4;
            }
            offsetTable.push(this._data.length);
            const stringTable = [];
            const utf8Decoder = new TextDecoder('utf-8');
            for (let k = 0; k < count; k++) {
                const textArray = this._data.subarray(offsetTable[k], offsetTable[k + 1]);
                stringTable.push(utf8Decoder.decode(textArray));
            }
            context.data = stringTable;
        }
        return context;
    }

    _decode(context, dimension) {
        const shape = (context.shape.length === 0) ? [ 1 ] : context.shape;
        const size = shape[dimension];
        const results = [];
        if (dimension === shape.length - 1) {
            for (let i = 0; i < size; i++) {
                if (context.count > context.limit) {
                    results.push('...');
                    return results;
                }
                switch (context.dataType) {
                    case 'uint8':
                        results.push(context.data.getUint8(context.index));
                        context.index += 1;
                        context.count++;
                        break;
                    case 'int8':
                        results.push(context.data.getInt8(context.index));
                        context.index += 1;
                        context.count++;
                        break;
                    case 'int16':
                        results.push(context.data.getInt16(context.index));
                        context.index += 2;
                        context.count++;
                        break;
                    case 'int32':
                        results.push(context.data.getInt32(context.index, true));
                        context.index += 4;
                        context.count++;
                        break;
                    case 'int64':
                        results.push(context.data.getInt64(context.index, true));
                        context.index += 8;
                        context.count++;
                        break;
                    case 'float16':
                        results.push(context.data.getFloat16(context.index, true));
                        context.index += 2;
                        context.count++;
                        break;
                    case 'float32':
                        results.push(context.data.getFloat32(context.index, true));
                        context.index += 4;
                        context.count++;
                        break;
                    case 'float64':
                        results.push(context.data.getFloat64(context.index, true));
                        context.index += 8;
                        context.count++;
                        break;
                    case 'string':
                        results.push(context.data[context.index++]);
                        context.count++;
                        break;
                    default:
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
        if (context.shape.length === 0) {
            return results[0];
        }
        return results;
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
            default:
                throw new mslite.Error("Unknown data type '" + dataType.toString() + "'.");
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

mslite.Metadata = class {

    static open(context) {
        if (mslite.Metadata._metadata) {
            return Promise.resolve(mslite.Metadata._metadata);
        }
        return context.request('mslite-metadata.json', 'utf-8', null).then((data) => {
            mslite.Metadata._metadata = new mslite.Metadata(data);
            return mslite.Metadata._metadata;
        }).catch(() => {
            mslite.Metadata._metadata = new mslite.Metadata(null);
            return mslite.Metadata._metadata;
        });
    }

    constructor(data) {
        this._map = new Map();
        if (data) {
            const metadata = JSON.parse(data);
            this._map = new Map(metadata.map((item) => [ item.name, item ]));
        }
    }

    type(name) {
        return this._map.has(name) ? this._map.get(name) : null;
    }

    attribute(type, name) {
        const schema = this.type(type);
        if (schema) {
            let attributeMap = schema.attributeMap;
            if (!attributeMap) {
                attributeMap = {};
                if (schema.attributes) {
                    for (const attribute of schema.attributes) {
                        attributeMap[attribute.name] = attribute;
                    }
                }
                schema.attributeMap = attributeMap;
            }
            const attributeSchema = attributeMap[name];
            if (attributeSchema) {
                return attributeSchema;
            }
        }
        return null;
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

    constructor(message, context) {
        super(message);
        this.name = 'Error loading MindSpore Lite model.';
        this.context = context === false ? false : true;
    }
};

if (typeof module !== 'undefined' && typeof module.exports === 'object') {
    module.exports.ModelFactory = mslite.ModelFactory;
}