
export const mind_ir = {};

mind_ir.Version = {
    "IR_VERSION_START": 0,
    "IR_VERSION": 1,
    "IR_VERSION_WITH_PRIM_FUNCTION": 2
};

mind_ir.AttributeProto = class AttributeProto {

    constructor() {
        this.floats = [];
        this.doubles = [];
        this.ints = [];
        this.strings = [];
        this.tensors = [];
        this.graphs = [];
        this.values = [];
    }

    static decode(reader, length) {
        const message = new mind_ir.AttributeProto();
        const end = length === undefined ? reader.length : reader.position + length;
        while (reader.position < end) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.name = reader.string();
                    break;
                case 2:
                    message.f = reader.float();
                    break;
                case 3:
                    message.i = reader.int64();
                    break;
                case 4:
                    message.d = reader.double();
                    break;
                case 5:
                    message.s = reader.bytes();
                    break;
                case 6:
                    message.t = mind_ir.TensorProto.decode(reader, reader.uint32());
                    break;
                case 7:
                    message.g = mind_ir.GraphProto.decode(reader, reader.uint32());
                    break;
                case 8:
                    message.floats = reader.floats(message.floats, tag);
                    break;
                case 9:
                    message.doubles = reader.doubles(message.doubles, tag);
                    break;
                case 10:
                    message.ints = reader.array(message.ints, () => reader.int64(), tag);
                    break;
                case 11:
                    message.strings.push(reader.bytes());
                    break;
                case 12:
                    message.tensors.push(mind_ir.TensorProto.decode(reader, reader.uint32()));
                    break;
                case 13:
                    message.graphs.push(mind_ir.GraphProto.decode(reader, reader.uint32()));
                    break;
                case 14:
                    message.doc_string = reader.string();
                    break;
                case 15:
                    message.ref_attr_name = reader.string();
                    break;
                case 16:
                    message.type = reader.int32();
                    break;
                case 17:
                    message.values.push(mind_ir.AttributeProto.decode(reader, reader.uint32()));
                    break;
                case 18:
                    message.seq_info = mind_ir.AttributeProto.SeqInfoProto.decode(reader, reader.uint32());
                    break;
                case 19:
                    message.functor = mind_ir.FunctorProto.decode(reader, reader.uint32());
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }

    static decodeText(reader) {
        const message = new mind_ir.AttributeProto();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "name":
                    message.name = reader.string();
                    break;
                case "f":
                    message.f = reader.float();
                    break;
                case "i":
                    message.i = reader.int64();
                    break;
                case "d":
                    message.d = reader.double();
                    break;
                case "s":
                    message.s = reader.bytes();
                    break;
                case "t":
                    message.t = mind_ir.TensorProto.decodeText(reader);
                    break;
                case "g":
                    message.g = mind_ir.GraphProto.decodeText(reader);
                    break;
                case "floats":
                    reader.array(message.floats, () => reader.float());
                    break;
                case "doubles":
                    reader.array(message.doubles, () => reader.double());
                    break;
                case "ints":
                    reader.array(message.ints, () => reader.int64());
                    break;
                case "strings":
                    reader.array(message.strings, () => reader.bytes());
                    break;
                case "tensors":
                    message.tensors.push(mind_ir.TensorProto.decodeText(reader));
                    break;
                case "graphs":
                    message.graphs.push(mind_ir.GraphProto.decodeText(reader));
                    break;
                case "doc_string":
                    message.doc_string = reader.string();
                    break;
                case "ref_attr_name":
                    message.ref_attr_name = reader.string();
                    break;
                case "type":
                    message.type = reader.enum(mind_ir.AttributeProto.AttributeType);
                    break;
                case "values":
                    message.values.push(mind_ir.AttributeProto.decodeText(reader));
                    break;
                case "seq_info":
                    message.seq_info = mind_ir.AttributeProto.SeqInfoProto.decodeText(reader);
                    break;
                case "functor":
                    message.functor = mind_ir.FunctorProto.decodeText(reader);
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

mind_ir.AttributeProto.prototype.name = "";
mind_ir.AttributeProto.prototype.f = 0;
mind_ir.AttributeProto.prototype.i = 0n;
mind_ir.AttributeProto.prototype.d = 0;
mind_ir.AttributeProto.prototype.s = new Uint8Array([]);
mind_ir.AttributeProto.prototype.t = null;
mind_ir.AttributeProto.prototype.g = null;
mind_ir.AttributeProto.prototype.doc_string = "";
mind_ir.AttributeProto.prototype.ref_attr_name = "";
mind_ir.AttributeProto.prototype.type = 0;
mind_ir.AttributeProto.prototype.seq_info = null;
mind_ir.AttributeProto.prototype.functor = null;

mind_ir.AttributeProto.AttributeType = {
    "UNDEFINED": 0,
    "FLOAT": 1,
    "UINT8": 2,
    "INT8": 3,
    "UINT16": 4,
    "INT16": 5,
    "INT32": 6,
    "INT64": 7,
    "STRING": 8,
    "BOOL": 9,
    "FLOAT16": 10,
    "DOUBLE": 11,
    "UINT32": 12,
    "UINT64": 13,
    "COMPLEX64": 14,
    "COMPLEX128": 15,
    "BFLOAT16": 16,
    "TENSOR": 17,
    "GRAPH": 18,
    "TENSORS": 19,
    "TUPLE": 20,
    "LIST": 21,
    "DICT": 22,
    "UMONAD": 23,
    "IOMONAD": 24,
    "NONE": 25,
    "PRIMITIVECLOSURE": 26,
    "FUNCGRAPHCLOSURE": 27,
    "PARTIALCLOSURE": 28,
    "UNIONFUNCCLOSURE": 29,
    "CSR_TENSOR": 30,
    "COO_TENSOR": 31,
    "ROW_TENSOR": 32,
    "CLASS_TYPE": 33,
    "NAME_SPACE": 34,
    "SYMBOL": 35,
    "TYPE_NULL": 36,
    "MAP_TENSOR": 37,
    "FUNCTOR": 38,
    "SCALAR": 39
};

mind_ir.AttributeProto.SeqInfoProto = class SeqInfoProto {

    static decode(reader, length) {
        const message = new mind_ir.AttributeProto.SeqInfoProto();
        const end = length === undefined ? reader.length : reader.position + length;
        while (reader.position < end) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.is_dyn_len = reader.bool();
                    break;
                case 2:
                    message.tuple_elem_item = mind_ir.AttributeProto.decode(reader, reader.uint32());
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }

    static decodeText(reader) {
        const message = new mind_ir.AttributeProto.SeqInfoProto();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "is_dyn_len":
                    message.is_dyn_len = reader.bool();
                    break;
                case "tuple_elem_item":
                    message.tuple_elem_item = mind_ir.AttributeProto.decodeText(reader);
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

mind_ir.AttributeProto.SeqInfoProto.prototype.is_dyn_len = false;
mind_ir.AttributeProto.SeqInfoProto.prototype.tuple_elem_item = null;

mind_ir.FunctorProto = class FunctorProto {

    constructor() {
        this.values = [];
    }

    static decode(reader, length) {
        const message = new mind_ir.FunctorProto();
        const end = length === undefined ? reader.length : reader.position + length;
        while (reader.position < end) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.type = reader.int32();
                    break;
                case 2:
                    message.name = reader.string();
                    break;
                case 3:
                    message.values.push(mind_ir.AttributeProto.decode(reader, reader.uint32()));
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }

    static decodeText(reader) {
        const message = new mind_ir.FunctorProto();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "type":
                    message.type = reader.enum(mind_ir.FunctorProto.FunctorType);
                    break;
                case "name":
                    message.name = reader.string();
                    break;
                case "values":
                    message.values.push(mind_ir.AttributeProto.decodeText(reader));
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

mind_ir.FunctorProto.prototype.type = 1;
mind_ir.FunctorProto.prototype.name = "";

mind_ir.FunctorProto.FunctorType = {
    "SHAPE_CALC_FUNCTOR": 1
};

mind_ir.ValueInfoProto = class ValueInfoProto {

    constructor() {
        this.tensor = [];
    }

    static decode(reader, length) {
        const message = new mind_ir.ValueInfoProto();
        const end = length === undefined ? reader.length : reader.position + length;
        while (reader.position < end) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.name = reader.string();
                    break;
                case 2:
                    message.tensor.push(mind_ir.TensorProto.decode(reader, reader.uint32()));
                    break;
                case 3:
                    message.doc_string = reader.string();
                    break;
                case 4:
                    message.denotation = reader.string();
                    break;
                case 5:
                    message.attr_info = mind_ir.AttributeProto.decode(reader, reader.uint32());
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }

    static decodeText(reader) {
        const message = new mind_ir.ValueInfoProto();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "name":
                    message.name = reader.string();
                    break;
                case "tensor":
                    message.tensor.push(mind_ir.TensorProto.decodeText(reader));
                    break;
                case "doc_string":
                    message.doc_string = reader.string();
                    break;
                case "denotation":
                    message.denotation = reader.string();
                    break;
                case "attr_info":
                    message.attr_info = mind_ir.AttributeProto.decodeText(reader);
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

mind_ir.ValueInfoProto.prototype.name = "";
mind_ir.ValueInfoProto.prototype.doc_string = "";
mind_ir.ValueInfoProto.prototype.denotation = "";
mind_ir.ValueInfoProto.prototype.attr_info = null;

mind_ir.NodeProto = class NodeProto {

    constructor() {
        this.input = [];
        this.output = [];
        this.attribute = [];
        this.node_attr = [];
        this.primal_attr = [];
    }

    static decode(reader, length) {
        const message = new mind_ir.NodeProto();
        const end = length === undefined ? reader.length : reader.position + length;
        while (reader.position < end) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.input.push(reader.string());
                    break;
                case 2:
                    message.output.push(reader.string());
                    break;
                case 3:
                    message.name = reader.string();
                    break;
                case 4:
                    message.op_type = reader.string();
                    break;
                case 5:
                    message.attribute.push(mind_ir.AttributeProto.decode(reader, reader.uint32()));
                    break;
                case 6:
                    message.doc_string = reader.string();
                    break;
                case 7:
                    message.domain = reader.string();
                    break;
                case 8:
                    message.node_attr.push(mind_ir.AttributeProto.decode(reader, reader.uint32()));
                    break;
                case 9:
                    message.primal_attr.push(mind_ir.AttributeProto.decode(reader, reader.uint32()));
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }

    static decodeText(reader) {
        const message = new mind_ir.NodeProto();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "input":
                    reader.array(message.input, () => reader.string());
                    break;
                case "output":
                    reader.array(message.output, () => reader.string());
                    break;
                case "name":
                    message.name = reader.string();
                    break;
                case "op_type":
                    message.op_type = reader.string();
                    break;
                case "attribute":
                    message.attribute.push(mind_ir.AttributeProto.decodeText(reader));
                    break;
                case "doc_string":
                    message.doc_string = reader.string();
                    break;
                case "domain":
                    message.domain = reader.string();
                    break;
                case "node_attr":
                    message.node_attr.push(mind_ir.AttributeProto.decodeText(reader));
                    break;
                case "primal_attr":
                    message.primal_attr.push(mind_ir.AttributeProto.decodeText(reader));
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

mind_ir.NodeProto.prototype.name = "";
mind_ir.NodeProto.prototype.op_type = "";
mind_ir.NodeProto.prototype.doc_string = "";
mind_ir.NodeProto.prototype.domain = "";

mind_ir.ModelProto = class ModelProto {

    constructor() {
        this.functions = [];
        this.primitives = [];
        this.user_info = {};
    }

    static decode(reader, length) {
        const message = new mind_ir.ModelProto();
        const end = length === undefined ? reader.length : reader.position + length;
        while (reader.position < end) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.ir_version = reader.string();
                    break;
                case 2:
                    message.producer_name = reader.string();
                    break;
                case 3:
                    message.producer_version = reader.string();
                    break;
                case 4:
                    message.domain = reader.string();
                    break;
                case 5:
                    message.model_version = reader.string();
                    break;
                case 6:
                    message.doc_string = reader.string();
                    break;
                case 7:
                    message.graph = mind_ir.GraphProto.decode(reader, reader.uint32());
                    break;
                case 8:
                    message.functions.push(mind_ir.GraphProto.decode(reader, reader.uint32()));
                    break;
                case 9:
                    message.preprocessor = mind_ir.PreprocessorProto.decode(reader, reader.uint32());
                    break;
                case 10:
                    message.little_endian = reader.bool();
                    break;
                case 11:
                    message.parallel = mind_ir.ParallelProto.decode(reader, reader.uint32());
                    break;
                case 12:
                    message.primitives.push(mind_ir.PrimitiveProto.decode(reader, reader.uint32()));
                    break;
                case 13:
                    message.mind_ir_version = reader.int64();
                    break;
                case 14:
                    reader.entry(message.user_info, () => reader.string(), () => reader.string());
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }

    static decodeText(reader) {
        const message = new mind_ir.ModelProto();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "ir_version":
                    message.ir_version = reader.string();
                    break;
                case "producer_name":
                    message.producer_name = reader.string();
                    break;
                case "producer_version":
                    message.producer_version = reader.string();
                    break;
                case "domain":
                    message.domain = reader.string();
                    break;
                case "model_version":
                    message.model_version = reader.string();
                    break;
                case "doc_string":
                    message.doc_string = reader.string();
                    break;
                case "graph":
                    message.graph = mind_ir.GraphProto.decodeText(reader);
                    break;
                case "functions":
                    message.functions.push(mind_ir.GraphProto.decodeText(reader));
                    break;
                case "preprocessor":
                    message.preprocessor = mind_ir.PreprocessorProto.decodeText(reader);
                    break;
                case "little_endian":
                    message.little_endian = reader.bool();
                    break;
                case "parallel":
                    message.parallel = mind_ir.ParallelProto.decodeText(reader);
                    break;
                case "primitives":
                    message.primitives.push(mind_ir.PrimitiveProto.decodeText(reader));
                    break;
                case "mind_ir_version":
                    message.mind_ir_version = reader.int64();
                    break;
                case "user_info":
                    reader.entry(message.user_info, () => reader.string(), () => reader.string());
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

mind_ir.ModelProto.prototype.ir_version = "";
mind_ir.ModelProto.prototype.producer_name = "";
mind_ir.ModelProto.prototype.producer_version = "";
mind_ir.ModelProto.prototype.domain = "";
mind_ir.ModelProto.prototype.model_version = "";
mind_ir.ModelProto.prototype.doc_string = "";
mind_ir.ModelProto.prototype.graph = null;
mind_ir.ModelProto.prototype.preprocessor = null;
mind_ir.ModelProto.prototype.little_endian = false;
mind_ir.ModelProto.prototype.parallel = null;
mind_ir.ModelProto.prototype.mind_ir_version = 0n;

mind_ir.PreprocessorProto = class PreprocessorProto {

    constructor() {
        this.op = [];
    }

    static decode(reader, length) {
        const message = new mind_ir.PreprocessorProto();
        const end = length === undefined ? reader.length : reader.position + length;
        while (reader.position < end) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.op.push(mind_ir.PreprocessOpProto.decode(reader, reader.uint32()));
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }

    static decodeText(reader) {
        const message = new mind_ir.PreprocessorProto();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "op":
                    message.op.push(mind_ir.PreprocessOpProto.decodeText(reader));
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

mind_ir.PreprocessOpProto = class PreprocessOpProto {

    static decode(reader, length) {
        const message = new mind_ir.PreprocessOpProto();
        const end = length === undefined ? reader.length : reader.position + length;
        while (reader.position < end) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.input_columns = reader.string();
                    break;
                case 2:
                    message.output_columns = reader.string();
                    break;
                case 3:
                    message.project_columns = reader.string();
                    break;
                case 4:
                    message.op_type = reader.string();
                    break;
                case 5:
                    message.operations = reader.string();
                    break;
                case 6:
                    message.offload = reader.bool();
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }

    static decodeText(reader) {
        const message = new mind_ir.PreprocessOpProto();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "input_columns":
                    message.input_columns = reader.string();
                    break;
                case "output_columns":
                    message.output_columns = reader.string();
                    break;
                case "project_columns":
                    message.project_columns = reader.string();
                    break;
                case "op_type":
                    message.op_type = reader.string();
                    break;
                case "operations":
                    message.operations = reader.string();
                    break;
                case "offload":
                    message.offload = reader.bool();
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

mind_ir.PreprocessOpProto.prototype.input_columns = "";
mind_ir.PreprocessOpProto.prototype.output_columns = "";
mind_ir.PreprocessOpProto.prototype.project_columns = "";
mind_ir.PreprocessOpProto.prototype.op_type = "";
mind_ir.PreprocessOpProto.prototype.operations = "";
mind_ir.PreprocessOpProto.prototype.offload = false;

mind_ir.GraphProto = class GraphProto {

    constructor() {
        this.node = [];
        this.parameter = [];
        this.input = [];
        this.output = [];
        this.attribute = [];
        this.map_parameter = [];
    }

    static decode(reader, length) {
        const message = new mind_ir.GraphProto();
        const end = length === undefined ? reader.length : reader.position + length;
        while (reader.position < end) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.node.push(mind_ir.NodeProto.decode(reader, reader.uint32()));
                    break;
                case 2:
                    message.name = reader.string();
                    break;
                case 3:
                    message.parameter.push(mind_ir.TensorProto.decode(reader, reader.uint32()));
                    break;
                case 4:
                    message.doc_string = reader.string();
                    break;
                case 5:
                    message.input.push(mind_ir.ValueInfoProto.decode(reader, reader.uint32()));
                    break;
                case 6:
                    message.output.push(mind_ir.ValueInfoProto.decode(reader, reader.uint32()));
                    break;
                case 7:
                    message.bprop_hash = reader.string();
                    break;
                case 8:
                    message.attribute.push(mind_ir.AttributeProto.decode(reader, reader.uint32()));
                    break;
                case 9:
                    message.bprop_filepath = reader.string();
                    break;
                case 10:
                    message.map_parameter.push(mind_ir.MapTensorProto.decode(reader, reader.uint32()));
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }

    static decodeText(reader) {
        const message = new mind_ir.GraphProto();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "node":
                    message.node.push(mind_ir.NodeProto.decodeText(reader));
                    break;
                case "name":
                    message.name = reader.string();
                    break;
                case "parameter":
                    message.parameter.push(mind_ir.TensorProto.decodeText(reader));
                    break;
                case "doc_string":
                    message.doc_string = reader.string();
                    break;
                case "input":
                    message.input.push(mind_ir.ValueInfoProto.decodeText(reader));
                    break;
                case "output":
                    message.output.push(mind_ir.ValueInfoProto.decodeText(reader));
                    break;
                case "bprop_hash":
                    message.bprop_hash = reader.string();
                    break;
                case "attribute":
                    message.attribute.push(mind_ir.AttributeProto.decodeText(reader));
                    break;
                case "bprop_filepath":
                    message.bprop_filepath = reader.string();
                    break;
                case "map_parameter":
                    message.map_parameter.push(mind_ir.MapTensorProto.decodeText(reader));
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

mind_ir.GraphProto.prototype.name = "";
mind_ir.GraphProto.prototype.doc_string = "";
mind_ir.GraphProto.prototype.bprop_hash = "";
mind_ir.GraphProto.prototype.bprop_filepath = "";

mind_ir.TensorProto = class TensorProto {

    constructor() {
        this.dims = [];
        this.float_data = [];
        this.int32_data = [];
        this.string_data = [];
        this.int64_data = [];
        this.double_data = [];
        this.uint64_data = [];
        this.min_dims = [];
        this.max_dims = [];
        this.quant_params = [];
    }

    static decode(reader, length) {
        const message = new mind_ir.TensorProto();
        const end = length === undefined ? reader.length : reader.position + length;
        while (reader.position < end) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.dims = reader.array(message.dims, () => reader.int64(), tag);
                    break;
                case 2:
                    message.data_type = reader.int32();
                    break;
                case 3:
                    message.float_data = reader.floats(message.float_data, tag);
                    break;
                case 4:
                    message.int32_data = reader.array(message.int32_data, () => reader.int32(), tag);
                    break;
                case 5:
                    message.string_data.push(reader.bytes());
                    break;
                case 6:
                    message.int64_data = reader.array(message.int64_data, () => reader.int64(), tag);
                    break;
                case 7:
                    message.name = reader.string();
                    break;
                case 8:
                    message.doc_string = reader.string();
                    break;
                case 9:
                    message.raw_data = reader.bytes();
                    break;
                case 10:
                    message.double_data = reader.doubles(message.double_data, tag);
                    break;
                case 11:
                    message.uint64_data = reader.array(message.uint64_data, () => reader.uint64(), tag);
                    break;
                case 12:
                    message.external_data = mind_ir.TensorProto.ExternalDataProto.decode(reader, reader.uint32());
                    break;
                case 13:
                    message.ref_key = reader.string();
                    break;
                case 14:
                    message.min_dims = reader.array(message.min_dims, () => reader.int64(), tag);
                    break;
                case 15:
                    message.max_dims = reader.array(message.max_dims, () => reader.int64(), tag);
                    break;
                case 16:
                    message.compression_type = reader.int32();
                    break;
                case 17:
                    message.quant_params.push(mind_ir.TensorProto.QuantParamProto.decode(reader, reader.uint32()));
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }

    static decodeText(reader) {
        const message = new mind_ir.TensorProto();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "dims":
                    reader.array(message.dims, () => reader.int64());
                    break;
                case "data_type":
                    message.data_type = reader.int32();
                    break;
                case "float_data":
                    reader.array(message.float_data, () => reader.float());
                    break;
                case "int32_data":
                    reader.array(message.int32_data, () => reader.int32());
                    break;
                case "string_data":
                    reader.array(message.string_data, () => reader.bytes());
                    break;
                case "int64_data":
                    reader.array(message.int64_data, () => reader.int64());
                    break;
                case "name":
                    message.name = reader.string();
                    break;
                case "doc_string":
                    message.doc_string = reader.string();
                    break;
                case "raw_data":
                    message.raw_data = reader.bytes();
                    break;
                case "double_data":
                    reader.array(message.double_data, () => reader.double());
                    break;
                case "uint64_data":
                    reader.array(message.uint64_data, () => reader.uint64());
                    break;
                case "external_data":
                    message.external_data = mind_ir.TensorProto.ExternalDataProto.decodeText(reader);
                    break;
                case "ref_key":
                    message.ref_key = reader.string();
                    break;
                case "min_dims":
                    reader.array(message.min_dims, () => reader.int64());
                    break;
                case "max_dims":
                    reader.array(message.max_dims, () => reader.int64());
                    break;
                case "compression_type":
                    message.compression_type = reader.enum(mind_ir.TensorProto.CompressionType);
                    break;
                case "quant_params":
                    message.quant_params.push(mind_ir.TensorProto.QuantParamProto.decodeText(reader));
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

mind_ir.TensorProto.prototype.data_type = 0;
mind_ir.TensorProto.prototype.name = "";
mind_ir.TensorProto.prototype.doc_string = "";
mind_ir.TensorProto.prototype.raw_data = new Uint8Array([]);
mind_ir.TensorProto.prototype.external_data = null;
mind_ir.TensorProto.prototype.ref_key = "";
mind_ir.TensorProto.prototype.compression_type = 0;

mind_ir.TensorProto.DataType = {
    "UNDEFINED": 0,
    "FLOAT": 1,
    "UINT8": 2,
    "INT8": 3,
    "UINT16": 4,
    "INT16": 5,
    "INT32": 6,
    "INT64": 7,
    "STRING": 8,
    "BOOL": 9,
    "FLOAT16": 10,
    "DOUBLE": 11,
    "UINT32": 12,
    "UINT64": 13,
    "COMPLEX64": 14,
    "COMPLEX128": 15,
    "BFLOAT16": 16,
    "FLOAT64": 17,
    "QINT4X2": 18
};

mind_ir.TensorProto.CompressionType = {
    "NO_COMPRESSION": 0,
    "INDEXING": 1,
    "SPARSE": 2,
    "FSE": 3,
    "BIT_PACKING": 4,
    "FSE_INT": 5,
    "FSE_INFER": 6
};

mind_ir.TensorProto.ExternalDataProto = class ExternalDataProto {

    static decode(reader, length) {
        const message = new mind_ir.TensorProto.ExternalDataProto();
        const end = length === undefined ? reader.length : reader.position + length;
        while (reader.position < end) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.location = reader.string();
                    break;
                case 2:
                    message.offset = reader.int64();
                    break;
                case 3:
                    message.length = reader.int64();
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }

    static decodeText(reader) {
        const message = new mind_ir.TensorProto.ExternalDataProto();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "location":
                    message.location = reader.string();
                    break;
                case "offset":
                    message.offset = reader.int64();
                    break;
                case "length":
                    message.length = reader.int64();
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

mind_ir.TensorProto.ExternalDataProto.prototype.location = "";
mind_ir.TensorProto.ExternalDataProto.prototype.offset = 0n;
mind_ir.TensorProto.ExternalDataProto.prototype.length = 0n;

mind_ir.TensorProto.QuantParamProto = class QuantParamProto {

    constructor() {
        this.attribute = [];
    }

    static decode(reader, length) {
        const message = new mind_ir.TensorProto.QuantParamProto();
        const end = length === undefined ? reader.length : reader.position + length;
        while (reader.position < end) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.quant_algo_name = reader.string();
                    break;
                case 2:
                    message.attribute.push(mind_ir.AttributeProto.decode(reader, reader.uint32()));
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        if (!Object.prototype.hasOwnProperty.call(message, 'quant_algo_name')) {
            throw new Error("Expected 'quant_algo_name'.");
        }
        return message;
    }

    static decodeText(reader) {
        const message = new mind_ir.TensorProto.QuantParamProto();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "quant_algo_name":
                    message.quant_algo_name = reader.string();
                    break;
                case "attribute":
                    message.attribute.push(mind_ir.AttributeProto.decodeText(reader));
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        if (!Object.prototype.hasOwnProperty.call(message, "quant_algo_name")) {
            throw new Error("Expected 'quant_algo_name'.");
        }
        return message;
    }
};

mind_ir.TensorProto.QuantParamProto.prototype.quant_algo_name = "";

mind_ir.MapTensorProto = class MapTensorProto {

    static decode(reader, length) {
        const message = new mind_ir.MapTensorProto();
        const end = length === undefined ? reader.length : reader.position + length;
        while (reader.position < end) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.name = reader.string();
                    break;
                case 2:
                    message.default_value = mind_ir.AttributeProto.decode(reader, reader.uint32());
                    break;
                case 3:
                    message.key_tensor = mind_ir.TensorProto.decode(reader, reader.uint32());
                    break;
                case 4:
                    message.value_tensor = mind_ir.TensorProto.decode(reader, reader.uint32());
                    break;
                case 5:
                    message.status_tensor = mind_ir.TensorProto.decode(reader, reader.uint32());
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        if (!Object.prototype.hasOwnProperty.call(message, 'name')) {
            throw new Error("Expected 'name'.");
        }
        if (!Object.prototype.hasOwnProperty.call(message, 'default_value')) {
            throw new Error("Expected 'default_value'.");
        }
        if (!Object.prototype.hasOwnProperty.call(message, 'key_tensor')) {
            throw new Error("Expected 'key_tensor'.");
        }
        if (!Object.prototype.hasOwnProperty.call(message, 'value_tensor')) {
            throw new Error("Expected 'value_tensor'.");
        }
        if (!Object.prototype.hasOwnProperty.call(message, 'status_tensor')) {
            throw new Error("Expected 'status_tensor'.");
        }
        return message;
    }

    static decodeText(reader) {
        const message = new mind_ir.MapTensorProto();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "name":
                    message.name = reader.string();
                    break;
                case "default_value":
                    message.default_value = mind_ir.AttributeProto.decodeText(reader);
                    break;
                case "key_tensor":
                    message.key_tensor = mind_ir.TensorProto.decodeText(reader);
                    break;
                case "value_tensor":
                    message.value_tensor = mind_ir.TensorProto.decodeText(reader);
                    break;
                case "status_tensor":
                    message.status_tensor = mind_ir.TensorProto.decodeText(reader);
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        if (!Object.prototype.hasOwnProperty.call(message, "name")) {
            throw new Error("Expected 'name'.");
        }
        if (!Object.prototype.hasOwnProperty.call(message, "default_value")) {
            throw new Error("Expected 'default_value'.");
        }
        if (!Object.prototype.hasOwnProperty.call(message, "key_tensor")) {
            throw new Error("Expected 'key_tensor'.");
        }
        if (!Object.prototype.hasOwnProperty.call(message, "value_tensor")) {
            throw new Error("Expected 'value_tensor'.");
        }
        if (!Object.prototype.hasOwnProperty.call(message, "status_tensor")) {
            throw new Error("Expected 'status_tensor'.");
        }
        return message;
    }
};

mind_ir.MapTensorProto.prototype.name = "";
mind_ir.MapTensorProto.prototype.default_value = null;
mind_ir.MapTensorProto.prototype.key_tensor = null;
mind_ir.MapTensorProto.prototype.value_tensor = null;
mind_ir.MapTensorProto.prototype.status_tensor = null;

mind_ir.ParallelProto = class ParallelProto {

    constructor() {
        this.layout = [];
    }

    static decode(reader, length) {
        const message = new mind_ir.ParallelProto();
        const end = length === undefined ? reader.length : reader.position + length;
        while (reader.position < end) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.layout.push(mind_ir.LayoutProto.decode(reader, reader.uint32()));
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }

    static decodeText(reader) {
        const message = new mind_ir.ParallelProto();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "layout":
                    message.layout.push(mind_ir.LayoutProto.decodeText(reader));
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

mind_ir.LayoutProto = class LayoutProto {

    constructor() {
        this.device_arrangement_int = [];
        this.tensor_map_int = [];
        this.slice_shape_int = [];
    }

    static decode(reader, length) {
        const message = new mind_ir.LayoutProto();
        const end = length === undefined ? reader.length : reader.position + length;
        while (reader.position < end) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.name = reader.string();
                    break;
                case 2:
                    message.device_arrangement_int = reader.array(message.device_arrangement_int, () => reader.int64(), tag);
                    break;
                case 3:
                    message.tensor_map_int = reader.array(message.tensor_map_int, () => reader.int64(), tag);
                    break;
                case 4:
                    message.slice_shape_int = reader.array(message.slice_shape_int, () => reader.int64(), tag);
                    break;
                case 5:
                    message.field_size = reader.int64();
                    break;
                case 6:
                    message.uniform_split = reader.bool();
                    break;
                case 7:
                    message.opt_shard_group = reader.string();
                    break;
                case 8:
                    message.pipeline_shared = reader.bool();
                    break;
                case 9:
                    message.is_send = reader.bool();
                    break;
                case 10:
                    message.peer_rank = reader.int64();
                    break;
                case 11:
                    message.sr_tag = reader.int64();
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }

    static decodeText(reader) {
        const message = new mind_ir.LayoutProto();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "name":
                    message.name = reader.string();
                    break;
                case "device_arrangement_int":
                    reader.array(message.device_arrangement_int, () => reader.int64());
                    break;
                case "tensor_map_int":
                    reader.array(message.tensor_map_int, () => reader.int64());
                    break;
                case "slice_shape_int":
                    reader.array(message.slice_shape_int, () => reader.int64());
                    break;
                case "field_size":
                    message.field_size = reader.int64();
                    break;
                case "uniform_split":
                    message.uniform_split = reader.bool();
                    break;
                case "opt_shard_group":
                    message.opt_shard_group = reader.string();
                    break;
                case "pipeline_shared":
                    message.pipeline_shared = reader.bool();
                    break;
                case "is_send":
                    message.is_send = reader.bool();
                    break;
                case "peer_rank":
                    message.peer_rank = reader.int64();
                    break;
                case "sr_tag":
                    message.sr_tag = reader.int64();
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

mind_ir.LayoutProto.prototype.name = "";
mind_ir.LayoutProto.prototype.field_size = 0n;
mind_ir.LayoutProto.prototype.uniform_split = false;
mind_ir.LayoutProto.prototype.opt_shard_group = "";
mind_ir.LayoutProto.prototype.pipeline_shared = false;
mind_ir.LayoutProto.prototype.is_send = false;
mind_ir.LayoutProto.prototype.peer_rank = 0n;
mind_ir.LayoutProto.prototype.sr_tag = 0n;

mind_ir.PrimitiveProto = class PrimitiveProto {

    constructor() {
        this.attribute = [];
    }

    static decode(reader, length) {
        const message = new mind_ir.PrimitiveProto();
        const end = length === undefined ? reader.length : reader.position + length;
        while (reader.position < end) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.name = reader.string();
                    break;
                case 2:
                    message.op_type = reader.string();
                    break;
                case 3:
                    message.attribute.push(mind_ir.AttributeProto.decode(reader, reader.uint32()));
                    break;
                case 4:
                    message.instance_name = reader.string();
                    break;
                case 5:
                    message.prim_type = reader.int32();
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }

    static decodeText(reader) {
        const message = new mind_ir.PrimitiveProto();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "name":
                    message.name = reader.string();
                    break;
                case "op_type":
                    message.op_type = reader.string();
                    break;
                case "attribute":
                    message.attribute.push(mind_ir.AttributeProto.decodeText(reader));
                    break;
                case "instance_name":
                    message.instance_name = reader.string();
                    break;
                case "prim_type":
                    message.prim_type = reader.enum(mind_ir.PrimitiveProto.PrimType);
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

mind_ir.PrimitiveProto.prototype.name = "";
mind_ir.PrimitiveProto.prototype.op_type = "";
mind_ir.PrimitiveProto.prototype.instance_name = "";
mind_ir.PrimitiveProto.prototype.prim_type = 1;

mind_ir.PrimitiveProto.PrimType = {
    "PRIMITIVE": 1,
    "PRIMITIVE_FUNCTION": 2
};
