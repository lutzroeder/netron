export const protobuf = {};
const $root = protobuf.get('mind_ir');

$root.mind_ir = {};

$root.mind_ir.Version = {
    "IR_VERSION_START": 0,
    "IR_VERSION": 1
};

$root.mind_ir.AttributeProto = class AttributeProto {

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
        const message = new $root.mind_ir.AttributeProto();
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
                    message.t = $root.mind_ir.TensorProto.decode(reader, reader.uint32());
                    break;
                case 7:
                    message.g = $root.mind_ir.GraphProto.decode(reader, reader.uint32());
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
                    message.tensors.push($root.mind_ir.TensorProto.decode(reader, reader.uint32()));
                    break;
                case 13:
                    message.graphs.push($root.mind_ir.GraphProto.decode(reader, reader.uint32()));
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
                    message.values.push($root.mind_ir.AttributeProto.decode(reader, reader.uint32()));
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }
};

$root.mind_ir.AttributeProto.prototype.name = "";
$root.mind_ir.AttributeProto.prototype.f = 0;
$root.mind_ir.AttributeProto.prototype.i = protobuf.Int64.create(0);
$root.mind_ir.AttributeProto.prototype.d = 0;
$root.mind_ir.AttributeProto.prototype.s = new Uint8Array([]);
$root.mind_ir.AttributeProto.prototype.t = null;
$root.mind_ir.AttributeProto.prototype.g = null;
$root.mind_ir.AttributeProto.prototype.doc_string = "";
$root.mind_ir.AttributeProto.prototype.ref_attr_name = "";
$root.mind_ir.AttributeProto.prototype.type = 0;

$root.mind_ir.AttributeProto.AttributeType = {
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
    "CLASS_TYPE": 33  //数据类型是否齐全？
};

$root.mind_ir.ValueInfoProto = class ValueInfoProto {

    constructor() {
        this.tensor = [];
    }

    static decode(reader, length) {
        const message = new $root.mind_ir.ValueInfoProto();
        const end = length === undefined ? reader.length : reader.position + length;
        while (reader.position < end) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.name = reader.string();
                    break;
                case 2:
                    message.tensor.push($root.mind_ir.TensorProto.decode(reader, reader.uint32()));
                    break;
                case 3:
                    message.doc_string = reader.string();
                    break;
                case 4:
                    message.denotation = reader.string();
                    break;
                case 5:
                    message.attr_info = $root.mind_ir.AttributeProto.decode(reader, reader.uint32());
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }
};

$root.mind_ir.ValueInfoProto.prototype.name = "";
$root.mind_ir.ValueInfoProto.prototype.doc_string = "";
$root.mind_ir.ValueInfoProto.prototype.denotation = "";
$root.mind_ir.ValueInfoProto.prototype.attr_info = null;

$root.mind_ir.NodeProto = class NodeProto {

    constructor() {
        this.input = [];
        this.output = [];
        this.attribute = [];
    }

    static decode(reader, length) {
        const message = new $root.mind_ir.NodeProto();
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
                    message.attribute.push($root.mind_ir.AttributeProto.decode(reader, reader.uint32()));
                    break;
                case 6:
                    message.doc_string = reader.string();
                    break;
                case 7:
                    message.domain = reader.string();
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }
};

$root.mind_ir.NodeProto.prototype.name = "";
$root.mind_ir.NodeProto.prototype.op_type = "";
$root.mind_ir.NodeProto.prototype.doc_string = "";
$root.mind_ir.NodeProto.prototype.domain = "";

$root.mind_ir.ModelProto = class ModelProto {

    constructor() {
        this.functions = [];
        this.primitives = [];
    }

    static decode(reader, length) {
        const message = new $root.mind_ir.ModelProto();
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
                    message.graph = $root.mind_ir.GraphProto.decode(reader, reader.uint32());
                    break;
                case 8:
                    message.functions.push($root.mind_ir.GraphProto.decode(reader, reader.uint32()));
                    break;
                case 9:
                    message.preprocessor = $root.mind_ir.PreprocessorProto.decode(reader, reader.uint32());
                    break;
                case 10:
                    message.little_endian = reader.bool();
                    break;
                case 11:
                    message.parallel = $root.mind_ir.ParallelProto.decode(reader, reader.uint32());
                    break;
                case 12:
                    message.primitives.push($root.mind_ir.PrimitiveProto.decode(reader, reader.uint32()));
                    break;
                case 13:
                    message.mind_ir_version = reader.int64();
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }
};

$root.mind_ir.ModelProto.prototype.ir_version = "";
$root.mind_ir.ModelProto.prototype.producer_name = "";
$root.mind_ir.ModelProto.prototype.producer_version = "";
$root.mind_ir.ModelProto.prototype.domain = "";
$root.mind_ir.ModelProto.prototype.model_version = "";
$root.mind_ir.ModelProto.prototype.doc_string = "";
$root.mind_ir.ModelProto.prototype.graph = null;
$root.mind_ir.ModelProto.prototype.preprocessor = null;
$root.mind_ir.ModelProto.prototype.little_endian = false;
$root.mind_ir.ModelProto.prototype.parallel = null;
$root.mind_ir.ModelProto.prototype.mind_ir_version = protobuf.Int64.create(0);

$root.mind_ir.PreprocessorProto = class PreprocessorProto {

    constructor() {
        this.op = [];
    }

    static decode(reader, length) {
        const message = new $root.mind_ir.PreprocessorProto();
        const end = length === undefined ? reader.length : reader.position + length;
        while (reader.position < end) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.op.push($root.mind_ir.PreprocessOpProto.decode(reader, reader.uint32()));
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }
};

$root.mind_ir.PreprocessOpProto = class PreprocessOpProto {

    static decode(reader, length) {
        const message = new $root.mind_ir.PreprocessOpProto();
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
};

$root.mind_ir.PreprocessOpProto.prototype.input_columns = "";
$root.mind_ir.PreprocessOpProto.prototype.output_columns = "";
$root.mind_ir.PreprocessOpProto.prototype.project_columns = "";
$root.mind_ir.PreprocessOpProto.prototype.op_type = "";
$root.mind_ir.PreprocessOpProto.prototype.operations = "";
$root.mind_ir.PreprocessOpProto.prototype.offload = false;

$root.mind_ir.GraphProto = class GraphProto {

    constructor() {
        this.node = [];
        this.parameter = [];
        this.input = [];
        this.output = [];
        this.attribute = [];
    }

    static decode(reader, length) {
        const message = new $root.mind_ir.GraphProto();
        const end = length === undefined ? reader.length : reader.position + length;
        while (reader.position < end) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.node.push($root.mind_ir.NodeProto.decode(reader, reader.uint32()));
                    break;
                case 2:
                    message.name = reader.string();
                    break;
                case 3:
                    message.parameter.push($root.mind_ir.TensorProto.decode(reader, reader.uint32()));
                    break;
                case 4:
                    message.doc_string = reader.string();
                    break;
                case 5:
                    message.input.push($root.mind_ir.ValueInfoProto.decode(reader, reader.uint32()));
                    break;
                case 6:
                    message.output.push($root.mind_ir.ValueInfoProto.decode(reader, reader.uint32()));
                    break;
                case 7:
                    message.bprop_hash = reader.string();
                    break;
                case 8:
                    message.attribute.push($root.mind_ir.AttributeProto.decode(reader, reader.uint32()));
                    break;
                case 9:
                    message.bprop_filepath = reader.string();
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }
};

$root.mind_ir.GraphProto.prototype.name = "";
$root.mind_ir.GraphProto.prototype.doc_string = "";
$root.mind_ir.GraphProto.prototype.bprop_hash = "";
$root.mind_ir.GraphProto.prototype.bprop_filepath = "";

$root.mind_ir.TensorProto = class TensorProto {

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
        const message = new $root.mind_ir.TensorProto();
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
                    message.external_data = $root.mind_ir.TensorProto.ExternalDataProto.decode(reader, reader.uint32());
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
                    message.quant_params.push($root.mind_ir.TensorProto.QuantParamProto.decode(reader, reader.uint32()));
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }
};

$root.mind_ir.TensorProto.prototype.data_type = 0;
$root.mind_ir.TensorProto.prototype.name = "";
$root.mind_ir.TensorProto.prototype.doc_string = "";
$root.mind_ir.TensorProto.prototype.raw_data = new Uint8Array([]);
$root.mind_ir.TensorProto.prototype.external_data = null;
$root.mind_ir.TensorProto.prototype.ref_key = "";
$root.mind_ir.TensorProto.prototype.compression_type = 0;

$root.mind_ir.TensorProto.DataType = {
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
    "FLOAT64": 17
};

$root.mind_ir.TensorProto.CompressionType = {
    "NO_COMPRESSION": 0,
    "INDEXING": 1,
    "SPARSE": 2,
    "FSE": 3,
    "BIT_PACKING": 4,
    "FSE_INT": 5,
    "FSE_INFER": 6
};

$root.mind_ir.TensorProto.ExternalDataProto = class ExternalDataProto {

    static decode(reader, length) {
        const message = new $root.mind_ir.TensorProto.ExternalDataProto();
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
                case 4:
                    message.checksum = reader.string();
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }
};

$root.mind_ir.TensorProto.ExternalDataProto.prototype.location = "";
$root.mind_ir.TensorProto.ExternalDataProto.prototype.offset = protobuf.Int64.create(0);
$root.mind_ir.TensorProto.ExternalDataProto.prototype.length = protobuf.Int64.create(0);
$root.mind_ir.TensorProto.ExternalDataProto.prototype.checksum = "";

$root.mind_ir.TensorProto.QuantParamProto = class QuantParamProto {

    constructor() {
        this.attribute = [];
    }

    static decode(reader, length) {
        const message = new $root.mind_ir.TensorProto.QuantParamProto();
        const end = length === undefined ? reader.length : reader.position + length;
        while (reader.position < end) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.quant_algo_name = reader.string();
                    break;
                case 2:
                    message.attribute.push($root.mind_ir.AttributeProto.decode(reader, reader.uint32()));
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        if (!Object.prototype.hasOwnProperty.call(message, 'quant_algo_name')) {
            throw new protobuf.Error("Excepted 'quant_algo_name'.");
        }
        return message;
    }
};

$root.mind_ir.TensorProto.QuantParamProto.prototype.quant_algo_name = "";

$root.mind_ir.ParallelProto = class ParallelProto {

    constructor() {
        this.layout = [];
    }

    static decode(reader, length) {
        const message = new $root.mind_ir.ParallelProto();
        const end = length === undefined ? reader.length : reader.position + length;
        while (reader.position < end) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.layout.push($root.mind_ir.LayoutProto.decode(reader, reader.uint32()));
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }
};

$root.mind_ir.LayoutProto = class LayoutProto {

    constructor() {
        this.device_arrangement_int = [];
        this.tensor_map_int = [];
        this.slice_shape_int = [];
    }

    static decode(reader, length) {
        const message = new $root.mind_ir.LayoutProto();
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
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }
};

$root.mind_ir.LayoutProto.prototype.name = "";
$root.mind_ir.LayoutProto.prototype.field_size = protobuf.Int64.create(0);
$root.mind_ir.LayoutProto.prototype.uniform_split = false;
$root.mind_ir.LayoutProto.prototype.opt_shard_group = "";

$root.mind_ir.PrimitiveProto = class PrimitiveProto {

    constructor() {
        this.attribute = [];
    }

    static decode(reader, length) {
        const message = new $root.mind_ir.PrimitiveProto();
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
                    message.attribute.push($root.mind_ir.AttributeProto.decode(reader, reader.uint32()));
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }
};

$root.mind_ir.PrimitiveProto.prototype.name = "";
$root.mind_ir.PrimitiveProto.prototype.op_type = "";
