/* jshint esversion: 6 */

var mslite = mslite || {};
var flatbuffers = flatbuffers || require('./flatbuffers');

mslite.ModelFactory = class {

    match(context) {
        const signature = 'MSL1';
        const extension = 'ms';
        const buffer = context.buffer;
        const ifSignatureMatch = buffer && buffer.length > 8 && buffer.subarray(4, 8).every((x, i) => x === signature.charCodeAt(i));
        const ifExtensionMatch = context.identifier.split('.').pop().toLowerCase() === extension;
        return ifSignatureMatch && ifExtensionMatch;
    }

    open(context, host) {
        return host.require('./mslite-schema').then(() => {
            const identifier = context.identifier;
            let model = null;
            try {
                mslite.schema = flatbuffers.get('mslite').mindspore.schema;
                const reader = new flatbuffers.Reader(context.buffer);
                if (!mslite.schema.MetaGraph.identifier(reader)) {
                    throw new mslite.Error('Invalid identifier.');
                }
                model = mslite.schema.MetaGraph.create(reader);
            }
            catch (error) {
                host.exception(error, false);
                const message = error && error.message ? error.message : error.toString();
                throw new mslite.Error(message.replace(/\.$/, '') + " in '" + identifier + "'.");
            }
            return mslite.Metadata.open(host).then((metadata) => {
                return new mslite.Model(metadata, model);
            });
        });
    }
};

mslite.Model = class {

    constructor(metadata, model) {
        this._name = model.name || '';
        this._format = model.version || '';

        const tensor_name_dict = this._createTensorMap(model);

        const enum_list = this._enum_list();

        this._graphs = [];
        this._graphs.push(new mslite.Graph(metadata, model, '', model, tensor_name_dict, enum_list));
    }

    _createTensorMap(model) {
        const tensor_name_dict = {};
        for (let i = 0; i < model.inputIndex.length; i++) {
            const id = model.inputIndex[i];
            if (tensor_name_dict && id in tensor_name_dict) {
                continue;
            }
            tensor_name_dict[id] = "graph_input-" + i.toString();
        }

        const special_arg_list = ["Conv2D", "DepthwiseConv2D", "DeConv2D", "DeDepthwiseConv2D", "FullConnection"];

        for (let i = 0; i < model.nodes.length; i++) {
            const node = model.nodes[i];
            for (let j = 0; j < node.outputIndex.length; j++) {
                const id = node.outputIndex[j];
                if (tensor_name_dict && id in tensor_name_dict) {
                    continue;
                }
                tensor_name_dict[id] = node.name + "/output-" + j.toString();
            }

            const nodeType = node.primitive.value.constructor.name;
            if (node.inputIndex.length > 1 && special_arg_list.indexOf(nodeType) !== -1) {
                tensor_name_dict[node.inputIndex[1]] = node.name + "/weight";
                if (node.inputIndex.length > 2) {
                    tensor_name_dict[node.inputIndex[2]] = node.name + "/bias";
                }
            }
            else {
                for (let j = 0; j < node.inputIndex.length; j++) {
                    const id = node.inputIndex[j];
                    if (tensor_name_dict && id in tensor_name_dict) {
                        continue;
                    }
                    tensor_name_dict[id] = node.name + "/input-" + j.toString();
                }
            }
        }
        return tensor_name_dict;
    }

    _enum_list() {
        const enum_list = [];
        for (const obj_name in mslite.schema) {
            if (mslite.schema[obj_name].constructor.name === 'Object') {
                enum_list.push(obj_name);
            }
        }
        return enum_list;
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

    constructor(metadata, subgraph, subgraph_name, model, tensor_name_dict, enum_list) {
        this._name = subgraph.name || subgraph_name;

        this._nodes = [];
        for (let i = 0; i < subgraph.nodes.length; i++) {
            this._nodes.push(new mslite.Node(metadata, subgraph.nodes[i], model, tensor_name_dict, enum_list));
        }

        this._inputs = [];
        for (let i = 0; i < subgraph.inputIndex.length; i++) {
            const id = subgraph.inputIndex[i];
            const name = this._name + "input-" + i.toString();
            this._inputs.push(this._createParameter(subgraph.allTensors[id], i.toString(), tensor_name_dict[id], name));
        }

        this._outputs = [];
        for (let i = 0; i < subgraph.outputIndex.length; i++) {
            const id = subgraph.outputIndex[i];
            const name = this._name + "output-" + i.toString();
            this._outputs.push(this._createParameter(subgraph.allTensors[id], i.toString(), tensor_name_dict[id], name));
        }
    }

    _createParameter(tensor, tensor_idx, arg_name, param_name) {
        const data = tensor.data;
        const initializer = (data && data.length > 0) ?
            new mslite.Tensor(tensor_idx, tensor.data, tensor.dataType, tensor.dims) : null;
        const args = new mslite.Argument(arg_name, tensor, initializer);
        return new mslite.Parameter(param_name, true, [args]);
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

    constructor(metadata, op, model, tensor_name_dict, enum_list) {
        this._name = op.name || '';

        const prim = op.primitive.value;
        let schema = '';
        if (prim.constructor) {
            this._type = prim.constructor.name;
            schema = metadata.type(this._type);
        }
        else {
            this._type = "Unsupported op: " + this._name;
        }

        this._attributes = [];
        for (const key of Object.keys(prim)) {
            const meta = metadata.attribute(this.type, key);
            this._attributes.push(new mslite.Attribute(meta, key.toString(), prim[key], enum_list));
        }

        this._inputs = [];
        const input_num = op.inputIndex.length;
        let i = 0;
        if (schema && schema.inputs){
            for (const input of schema.inputs) {
                if (i >= input_num) {
                    break;
                }
                const id = op.inputIndex[i];
                this._inputs.push(this._createParameter(model.allTensors[id], i.toString(), tensor_name_dict[id], input.name));
                i += 1;
            }
        }
        for (let j = i; j < input_num; j++) {
            const id = op.inputIndex[j];
            this._inputs.push(this._createParameter(model.allTensors[id], j.toString(), tensor_name_dict[id], j.toString()));
        }

        this._outputs = [];
        const output_num = op.outputIndex.length;
        i = 0;
        if (schema && schema.outputs){
            for (const output of schema.outputs) {
                if (i >= output_num) {
                    break;
                }
                const id = op.outputIndex[i];
                this._outputs.push(this._createParameter(model.allTensors[id], i.toString(), tensor_name_dict[id], output.name));
                i += 1;
            }
        }
        for (let j = i; j < output_num; j++) {
            const id = op.outputIndex[j];
            this._outputs.push(this._createParameter(model.allTensors[id], j.toString(), tensor_name_dict[id], j.toString()));
        }
    }

    _createParameter(tensor, tensor_idx, arg_name, param_name) {
        const data = tensor.data;
        const initializer = (data && data.length > 0) ?
            new mslite.Tensor(tensor_idx, tensor.data, tensor.dataType, tensor.dims) : null;
        const args = new mslite.Argument(arg_name, tensor, initializer);
        return new mslite.Parameter(param_name, true, [args]);
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

    constructor(schema, attrName, value, enum_list) {
        this._type = null;
        this._name = attrName;
        this._visible = true;
        this._value = ArrayBuffer.isView(value) ? Array.from(value) : value;

        if (schema) {
            if (schema.type) {
                this._type = schema.type;
                for (let i = 0; i < enum_list.length; i++) {
                    if (this._type === enum_list[i]) {
                        this._value = mslite.Utility.getEnumValue(this._type, this._value);
                        return;
                    }
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
        this._type = new mslite.TensorType(tensor.dataType, tensor.dims, name);
        this._initializer = initializer || null;

        if (tensor.quantParams) {
            this._quantParams = [];
            for (let i = 0; i < tensor.quantParams.length; i++) {
                const params = tensor.quantParams[i];
                const ten = [i.toString() + " : " + "scale=" + params.scale.toString() + ', zeroPoint=' + params.zeroPoint.toString()];
                this._quantParams.push(ten);
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
        return this._quantParams;
    }
};

mslite.Tensor = class {

    constructor(name, data, dataType, dims) {
        this._name = name;
        this._type = new mslite.TensorType(dataType || 0, dims || [], name);
        this._data = data || null;
    }

    get name() {
        return this._name;
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

    constructor(dataType, dims, info) {
        this._dataType = mslite.Utility.convertDataType(dataType, info);
        this._shape = new mslite.TensorShape(Array.from(dims));
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

    static open(host) {
        if (mslite.Metadata._metadata) {
            return Promise.resolve(mslite.Metadata._metadata);
        }
        return host.request(null, 'mslite-metadata.json', 'utf-8').then((data) => {
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
            const items = JSON.parse(data);
            if (items) {
                for (const item of items) {
                    if (item.name && item.schema) {
                        item.schema.name = item.name;
                        this._map.set(item.name, item.schema);
                    }
                }
            }
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

    static getEnumValue(name, value) {
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

    static convertDataType(dataType, info) {
        let type = '';
        switch (dataType) {
            case 0: type = "Unknown dataType"; break;
            case 30: type = "boolean"; break;
            case 31: type = "int"; break;
            case 32: type = "int8"; break;
            case 33: type = "int16"; break;
            case 34: type = "int32"; break;
            case 35: type = "int64"; break;
            case 36: type = "uint"; break;
            case 37: type = "uint8"; break;
            case 38: type = "uint16"; break;
            case 39: type = "uint32"; break;
            case 40: type = "uint64"; break;
            case 41: type = "float"; break;
            case 42: type = "float16"; break;
            case 43: type = "float32"; break;
            case 44: type = "float64"; break;
            case 45: type = "complex64"; break;
            default:
                throw new mslite.Error(" Wrong dataType = " + dataType + ' of '+ info);
        }
        return type;
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
