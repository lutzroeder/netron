/**
 * @fileoverview
 * This file contains the implementation of the mind_ir namespace, including the ModelFactory, Model, and Graph classes.
 * The ModelFactory class is responsible for matching and opening mind_ir models.
 * The Model class represents a mind_ir model and contains information about the model's format, name, and graphs.
 * The Graph class represents a graph within a mind_ir model and contains information about the graph's inputs, outputs, and nodes.
 */
const mind_ir = {
    ModelFactory: class {},
    Model: class {},
    Graph: class {},
    Parameter: class {},
    Argument: class {},
    Node: class {},
    Utility: class {},
    AttributeType: {},
    Attribute: class {},
    Tensor: class {},
    TensorType: class {},
    TensorShape: class {},
    Error: class extends Error {
        constructor(message) {
            super(message);
            this.name = 'SnapML Model Load Error.';
        }
    }
};
//const mind_ir = mind_ir || {};

mind_ir.ModelFactory = class {

    match(context) {
        const identifier = context.identifier;
        const extension = identifier.split('.').pop().toLowerCase();
        if (extension=="mindir") {
            return 'mind_ir';
        }
        return undefined;

    }


    open(context) {
        return context.require('./mind_ir-proto').then(() => {
            let model = null;
            try {
                mind_ir.proto = protobuf.get('mind_ir').mind_ir;
                const stream = context.stream;
                const reader = protobuf.BinaryReader.open(stream);
                model = mind_ir.proto.ModelProto.decode(reader);

            } catch (error) {
                const message = error && error.message ? error.message : error.toString();
                throw new mind_ir.Error(`File format is not mind_ir.Graph (${message.replace(/\.$/, '')}).`);
            }
            return context.metadata('mind_ir-metadata.json').then((metadata) => {
                return new mind_ir.Model(metadata, model);
            });
        });
    }
};





mind_ir.Model = class {


    constructor(metadata, model) {

        this._format = model.producer_name || ` ${model.model_version}` || `, ${model.ir_version}`;

        for (let i = 0; i < model.graph.node.length; i++) {
            const node = model.graph.node[i];
            node.tensors = node.attribute[0].tensors;
            if (node.attribute[0].tensors.dims) {
                node.dims=node.attribute[0].tensors.dims;
                const int32Array = new Int32Array(node.dims.length);


                node.dims.forEach((int64, index) => {
                    int32Array[index] = int64.low;
                });
                node.dims = int32Array;
                const type = new mind_ir.TensorType(node.attribute[0].tensors.data_type, node.dims);
                node.data_type = type;
            }

            if (node.op_type.startsWith('REF::')) {

                const primitiveNode = model.primitives.find((p) => p.name === node.op_type.split('::')[1]);

                if (primitiveNode) {

                    node.attribute = primitiveNode.attribute;
                    node.op_type = primitiveNode.op_type;
                    node.primitive = primitiveNode;
                }
            }
        }


        this._graphs = [new mind_ir.Graph(metadata, model.graph)];
    }

    get format() {
        return this._format;
    }

    get name() {
        return this._name;
    }

    get graphs() {
        return this._graphs;
    }
};


mind_ir.Graph = class {


    constructor(metadata, model) {



        this._inputs = [];
        this._outputs = [];
        this._nodes = [];


        //const scope = {};
        //const index = 0;




        const args_map = new Map();

        for (const parameter of model.input) {
            args_map.set(parameter.name, new mind_ir.Argument("input", parameter.tensor[0], null, null, true));
        }

        for (const parameter of model.output) {
            args_map.set(parameter.name, new mind_ir.Argument("output", parameter.tensor[0], null, null, true));
        }



        for (const para of model.parameter) {
            const data = para.raw_data;
            const int32Array = new Int32Array(para.dims.length);
            para.dims.forEach((int64, index) => {
                int32Array[index] = int64.low;
            });
            para.dims=int32Array;
            const type = new mind_ir.TensorType(para.data_type, para.dims);
            para.data_type = type;
            const initializer = (data && data.length > 0) ? new mind_ir.Tensor(type, data) : null;
            if (para.ref_key) {
                //debugger;
                args_map.set(para.name, new mind_ir.Argument(para.ref_key, para, initializer, null, true));
            } else {
                args_map.set(para.name, new mind_ir.Argument(para.name, para, initializer, null, true));
            }
        }

        for (const node of model.node) {
            if (node.domain) {
                if (node.domain.includes("Load")) {
                    args_map.set(node.name, args_map.get(node.input[0]));
                } else {
                    args_map.set(node.name, new mind_ir.Argument(node.domain, node.tensors, null, null, true));
                }

            } else {
                args_map.set(node.name, new mind_ir.Argument(node.name, node.tensors, null, null, true));
            }
        }


        for (const input of model.input) {
            const argument = args_map.get(input.name);
            this._inputs.push(new mind_ir.Parameter("input", [argument]));

        }


        for (const output of model.output) {
            this._outputs.push(new mind_ir.Parameter("output", [args_map.get(output.name)]));
        }

        for (const node of model.node) {
            if (node.op_type !== "MakeTuple" && node.op_type !== "UpdateState" && node.op_type !== "Load" && node.op_type !== "Constant") {
                this._nodes.push(new mind_ir.Node(metadata, node, args_map));
            }
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





mind_ir.Parameter = class {


    constructor(name, args) {
        this._name = name;
        this._arguments = args;
    }


    get name() {
        return this._name;
    }


    get arguments() {
        return this._arguments;
    }



    get visible() {

        if (this._name.includes("Load") || this._name.includes("MakeTuple") || this._name.includes("Constant") || this._name.includes("UpdateState")) {
            return false;
        }
        return true;
    }
};


mind_ir.Argument = class {


    constructor(name, tensor, initializer, quantization, visible) {

        if (typeof name !== 'string') {
            throw new mind_ir.Error(`无效的参数标识符 '${JSON.stringify(name)}'。`);
        }

        this._name = name;
        this._initializer = initializer || null;
        //let tensor_=null;
        if (Array.isArray(tensor)) {
            [tensor] = tensor;
        }
        if (!(tensor && tensor.dims)) {
            this._type = null;
        } else {
            this._type = initializer ? initializer._type : new mind_ir.TensorType(tensor.data_type, tensor.dims);
        }

        this._quantization = quantization || null;
        this._visible = visible || false;
    }


    get name() {
        return this._name;
    }


    get type() {
        return this._type;
    }


    get quantization() {
        return null;
    }


    get initializer() {
        return this._initializer;
    }



    get visible() {

        if (this._name.includes("Load") || this._name.includes("MakeTuple") || this._name.includes("Constant") || this._name.includes("UpdateState")) {
            return false;
        }
        return true;
    }
};



mind_ir.Node = class {


    constructor(metadata, node, args_map) {


        this._name = node.domain;
        this._type = { name: node.op_type };
        this._type = metadata.type(this._type.name);
        this._inputs = [];
        this._outputs = [];
        this._attributes = [];


        const input_num = node.input.length;
        let i = 0;
        if (this._type && this._type.inputs) {
            for (const input of this._type.inputs) {
                if (i >= input_num) {
                    break;
                }
                const arg = args_map.get(node.input[i]);
                //debugger;
                if (!(arg.name.includes("MakeTuple") || arg.name.includes("UpdateState") || arg.name.includes("Load") || arg.name.includes("Constant"))) {
                    this._inputs.push(new mind_ir.Parameter(input.name, arg ? [arg] : []));
                }
                i += 1;
            }
        }
        for (let j = i; j < input_num; j++) {
            const arg = args_map.get(node.input[j]);
            if (!(arg.name.includes("MakeTuple") || arg.name.includes("UpdateState") || arg.name.includes("Load") || arg.name.includes("Constant"))) {
                this._inputs.push(new mind_ir.Parameter(j.toString(), arg ? [arg] : []));
            }
        }

        const output_num = node.output.length;
        i = 0;
        if (this._type && this._type.outputs) {
            for (const output of this._type.outputs) {
                if (i >= output_num) {
                    break;
                }
                const arg = args_map.get(node.output[i]);
                if (!(arg.name.includes("MakeTuple") || arg.name.includes("UpdateState") || arg.name.includes("Load") || arg.name.includes("Constant"))) {
                    this._outputs.push(new mind_ir.Parameter(output.name, arg ? [arg] : []));
                }
                i += 1;
            }
        }
        for (let j = i; j < output_num; j++) {
            const arg = args_map.get(node.output[j]);
            if (!(arg.name.includes("MakeTuple") || arg.name.includes("UpdateState") || arg.name.includes("Load") || arg.name.includes("Constant"))) {
                this._outputs.push(new mind_ir.Parameter(j.toString(), arg ? [arg] : []));
            }
        }

        for (const attr of node.attribute) {
            //debugger;
            if (!attr.name.includes("input_names") && !attr.name.includes("output_names")) {
                this._attributes.push(new mind_ir.Attribute(this, attr));
            }
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

mind_ir.Utility = class {

    static convertToArray(values) {
        return values.map((value) => {
            for (const array of Object.values(value)) {
                if (Array.isArray(array) && array.length > 0) {
                    return array[0];
                }
            }
            return null;
        });
    }

    static convertList(lst) {
        const int32Array = new Int32Array(lst.length);

        lst.forEach((int64, index) => {
            int32Array[index] = int64.low;
        });
        return int32Array;
    }

    static enum(name, value) {
        const type = name && mind_ir.schema ? mind_ir.schema[name] : undefined;
        if (type) {
            mind_ir.Utility._enumKeyMap = mind_ir.Utility._enumKeyMap || new Map();
            if (!mind_ir.Utility._enumKeyMap.has(name)) {
                const map = new Map();
                for (const key of Object.keys(type)) {
                    map.set(type[key], key);
                }
                mind_ir.Utility._enumKeyMap.set(name, map);
            }
            const map = mind_ir.Utility._enumKeyMap.get(name);
            if (map.has(value)) {
                return map.get(value);
            }
        }
        return value;
    }
};


mind_ir.AttributeType = {
    UNDEFINED: 0,
    FLOAT: 1,
    UINT8: 2,
    INT8: 3,
    UINT16: 4,
    INT16: 5,
    INT32: 6,
    INT64: 7,
    STRING: 8,
    BOOL: 9,
    FLOAT16: 10,
    DOUBLE: 11,
    UINT32: 12,
    UINT64: 13,
    COMPLEX64: 14,
    COMPLEX128: 15,
    BFLOAT16: 16,
    TENSOR: 17,
    GRAPH: 18,
    TENSORS: 19,
    TUPLE: 20,
    LIST: 21,
    DICT: 22,
    UMONAD: 23,
    IOMONAD: 24,
    NONE: 25,
    PRIMITIVECLOSURE: 26,
    FUNCGRAPHCLOSURE: 27,
    PARTIALCLOSURE: 28,
    UNIONFUNCCLOSURE: 29,
    CSR_TENSOR: 30,
    COO_TENSOR: 31,
    ROW_TENSOR: 32,
    CLASS_TYPE: 33,
    NAME_SPACE: 34,
    SYMBOL: 35,
    TYPE_NULL: 36,
    MAP_TENSOR: 37
};



mind_ir.Attribute = class {

    constructor(context, attribute) {

        this._name = attribute.name;
        this._description = attribute.doc_string || '';
        this._type = null;
        this._value = null;

        switch (attribute.type) {
            case mind_ir.AttributeType.UNDEFINED:
                break;
            case mind_ir.AttributeType.FLOAT:
                this._value = attribute.f;
                this._type = 'float32';
                break;
            case mind_ir.AttributeType.UINT8:
                this._value = attribute.i;
                this._type = 'uint8';
                break;
            case mind_ir.AttributeType.INT8:
                this._value = attribute.i;
                this._type = 'int8';
                break;
            case mind_ir.AttributeType.UINT16:
                this._value = attribute.i;
                this._type = 'uint16';
                break;
            case mind_ir.AttributeType.INT16:
                this._value = attribute.i;
                this._type = 'int16';
                break;
            case mind_ir.AttributeType.INT32:
                this._value = attribute.i;
                this._type = 'int32';
                break;
            case mind_ir.AttributeType.INT64:
                if (attribute.ints!=0) {
                    this._value = attribute.ints;
                } else {
                    this._value = attribute.i;
                }
                this._type = 'int64';
                break;
            case mind_ir.AttributeType.STRING:
                if (attribute.s==0) {
                    this._value = attribute.strings;
                } else {
                    this._value = attribute.s;
                }

                this._type = 'string';
                break;
            case mind_ir.AttributeType.BOOL:
                this._value = attribute.i !== 0;
                this._type = 'bool';
                break;
            case mind_ir.AttributeType.FLOAT16:
                this._value = attribute.f;
                this._type = 'float16';
                break;
            case mind_ir.AttributeType.DOUBLE:
                this._value = attribute.f;
                this._type = 'double';
                break;
            case mind_ir.AttributeType.UINT32:
                this._value = attribute.i;
                this._type = 'uint32';
                break;
            case mind_ir.AttributeType.UINT64:
                this._value = attribute.i;
                this._type = 'uint64';
                break;
            case mind_ir.AttributeType.COMPLEX64:
                this._value = attribute.f;
                this._type = 'complex64';
                break;
            case mind_ir.AttributeType.COMPLEX128:
                this._value = attribute.f;
                this._type = 'complex128';
                break;
            case mind_ir.AttributeType.BFLOAT16:
                this._value = attribute.f;
                this._type = 'bfloat16';
                break;
            case mind_ir.AttributeType.TENSOR:
                this._value = new mind_ir.Tensor(context, attribute.t);
                this._type = 'tensor';
                break;
            case mind_ir.AttributeType.GRAPH:
                this._value = context.graph(attribute.g);
                this._type = 'graph';
                break;
            case mind_ir.AttributeType.TENSORS:
                this._value = attribute.tensors.map((tensor) => new mind_ir.Tensor(context, tensor));
                this._type = 'tensor[]';
                break;
            case mind_ir.AttributeType.TUPLE: {
                const processedAttributes__ = [];
                for (let i = 0; i < attribute.values.length; i++) {
                    const item = attribute.values[i];
                    const innerAttribute = new mind_ir.Attribute(this, item);
                    processedAttributes__.push(innerAttribute._value);
                }
                this._value = processedAttributes__;
                this._type = 'tuple';
                break;
            }
            case mind_ir.AttributeType.LIST: {
                const processedAttributes_ = [];
                for (let i = 0; i < attribute.values.length; i++) {
                    const item = attribute.values[i];
                    const innerAttribute = new mind_ir.Attribute(this, item);
                    //debugger;
                    processedAttributes_.push(innerAttribute._value);
                }
                this._value = processedAttributes_;
                this._type = 'list';
                break;
            }
            case mind_ir.AttributeType.DICT:
                this._value = attribute.dict;
                this._type = 'dict';
                break;
            case mind_ir.AttributeType.UMONAD:

                break;
            case mind_ir.AttributeType.IOMONAD:

                break;
            case mind_ir.AttributeType.NONE:
                this._value = null;
                this._type = 'none';
                break;
            case mind_ir.AttributeType.PRIMITIVECLOSURE:

                break;
            case mind_ir.AttributeType.FUNCGRAPHCLOSURE:

                break;
            case mind_ir.AttributeType.PARTIALCLOSURE:

                break;
            case mind_ir.AttributeType.UNIONFUNCCLOSURE:

                break;
            case mind_ir.AttributeType.CSR_TENSOR:
                this._value = new mind_ir.Tensor(context, attribute.csr_tensor);
                this._type = 'csr_tensor';
                break;
            case mind_ir.AttributeType.COO_TENSOR:
                this._value = new mind_ir.Tensor(context, attribute.coo_tensor);
                this._type = 'coo_tensor';
                break;
            case mind_ir.AttributeType.ROW_TENSOR:
                this._value = new mind_ir.Tensor(context, attribute.row_tensor);
                this._type = 'row_tensor';
                break;
            case mind_ir.AttributeType.CLASS_TYPE:

                break;
            case mind_ir.AttributeType.NAME_SPACE:

                break;
            case mind_ir.AttributeType.SYMBOL:

                break;
            case mind_ir.AttributeType.TYPE_NULL:
                this._value = null;
                this._type = 'type_null';
                break;
            case mind_ir.AttributeType.MAP_TENSOR:

                break;
            default:
                throw new Error(`Unsupported attribute type '${attribute.type}'.`);

        }
    }




    get name() {
        return this._name;
    }

    get type() {
        return this._type;
    }

    get value() {
        if (this._type == 'string') {

            const decoder = new TextDecoder('utf-8');


            const decodedString = decoder.decode(this._value);
            return decodedString;
        }
        return this._value;
    }

    get visible() {

        if (this._name.includes("output_names")||this._name.includes("output_names")) {
            return false;
        }
        return this._visible !== false;
    }
};


mind_ir.Tensor = class {

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



mind_ir.TensorType = class {
    constructor(dataType, dimensions) {
        switch (dataType+1) {
            case 1: this._dataType = "undefined"; break;
            case 2: this._dataType = "float32"; break;
            case 3: this._dataType = "uint8"; break;
            case 4: this._dataType = "uint16"; break;
            case 5: this._dataType = "int16"; break;
            case 6: this._dataType = "int32"; break;
            case 7: this._dataType = "int64"; break;
            case 8: this._dataType = "string"; break;
            case 9: this._dataType = "bool"; break;
            case 10: this._dataType = "float16"; break;
            case 11: this._dataType = "double"; break;
            case 12: this._dataType = "uint32"; break;
            case 13: this._dataType = "uint64"; break;
            case 14: this._dataType = "complex64"; break;
            case 15: this._dataType = "complex128"; break;
            case 16: this._dataType = "bfloat16"; break;
            case 17: this._dataType = "float64"; break;
            case 19: this._dataType = "string"; break;
            case 20: this._dataType = "complex64"; break;
            case 21: this._dataType = "complex128"; break;
            default: throw new Error(`Unsupported data type '${dataType.toString()}'.`);
        }

        this._shape = new mind_ir.TensorShape(dimensions);
    }

    get dataType() {
        return this._dataType;
    }

    get shape() {
        return this._shape;
    }

    toString() {
        return this._dataType + this._shape.toString();
    }
};



mind_ir.TensorShape = class {

    constructor(dimensions) {
        // if (dimensions instanceof Int32Array) {
        //     dimensions = dimensions;
        // }
        if (typeof(dimensions)=='object') {
            dimensions=mind_ir.Utility.convertList(dimensions);
        }
        //debugger;
        this._dimensions = Array.from(dimensions);
    }

    get dimensions() {
        return this._dimensions;
    }

    toString() {
        if (this._dimensions && this._dimensions.length > 0) {

            return `[${this._dimensions.map((dimension) => dimension ? dimension.toString() : '?').join(',')}]`;
        }
        return '';
    }
};


mind_ir.Error = class extends Error {


    constructor(message) {
        super(message);
        this.name = 'SnapML Model Load Error.';
    }
};


if (typeof module !== 'undefined' && typeof module.exports === 'object') {
    module.exports.ModelFactory = mind_ir.ModelFactory;
}
