var $root = protobuf.get('onnx');

$root.onnx = {};

$root.onnx.Version = {
    "_START_VERSION": 0,
    "IR_VERSION_2017_10_10": 1,
    "IR_VERSION_2017_10_30": 2,
    "IR_VERSION_2017_11_3": 3,
    "IR_VERSION_2019_1_22": 4,
    "IR_VERSION_2019_3_18": 5,
    "IR_VERSION_2019_9_19": 6,
    "IR_VERSION_2020_5_8": 7,
    "IR_VERSION_2021_7_30": 8,
    "IR_VERSION": 9
};

$root.onnx.AttributeProto = class AttributeProto {

    constructor() {
        this.floats = [];
        this.ints = [];
        this.strings = [];
        this.tensors = [];
        this.graphs = [];
        this.sparse_tensors = [];
        this.type_protos = [];
    }

    static decode(reader, length) {
        const message = new $root.onnx.AttributeProto();
        const end = length !== undefined ? reader.position + length : reader.length;
        while (reader.position < end) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.name = reader.string();
                    break;
                case 21:
                    message.ref_attr_name = reader.string();
                    break;
                case 13:
                    message.doc_string = reader.string();
                    break;
                case 20:
                    message.type = reader.int32();
                    break;
                case 2:
                    message.f = reader.float();
                    break;
                case 3:
                    message.i = reader.int64();
                    break;
                case 4:
                    message.s = reader.bytes();
                    break;
                case 5:
                    message.t = $root.onnx.TensorProto.decode(reader, reader.uint32());
                    break;
                case 6:
                    message.g = $root.onnx.GraphProto.decode(reader, reader.uint32());
                    break;
                case 22:
                    message.sparse_tensor = $root.onnx.SparseTensorProto.decode(reader, reader.uint32());
                    break;
                case 14:
                    message.tp = $root.onnx.TypeProto.decode(reader, reader.uint32());
                    break;
                case 7:
                    message.floats = reader.floats(message.floats, tag);
                    break;
                case 8:
                    message.ints = reader.array(message.ints, () => reader.int64(), tag);
                    break;
                case 9:
                    message.strings.push(reader.bytes());
                    break;
                case 10:
                    message.tensors.push($root.onnx.TensorProto.decode(reader, reader.uint32()));
                    break;
                case 11:
                    message.graphs.push($root.onnx.GraphProto.decode(reader, reader.uint32()));
                    break;
                case 23:
                    message.sparse_tensors.push($root.onnx.SparseTensorProto.decode(reader, reader.uint32()));
                    break;
                case 15:
                    message.type_protos.push($root.onnx.TypeProto.decode(reader, reader.uint32()));
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }

    static decodeText(reader) {
        const message = new $root.onnx.AttributeProto();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "name":
                    message.name = reader.string();
                    break;
                case "ref_attr_name":
                    message.ref_attr_name = reader.string();
                    break;
                case "doc_string":
                    message.doc_string = reader.string();
                    break;
                case "type":
                    message.type = reader.enum($root.onnx.AttributeProto.AttributeType);
                    break;
                case "f":
                    message.f = reader.float();
                    break;
                case "i":
                    message.i = reader.int64();
                    break;
                case "s":
                    message.s = reader.bytes();
                    break;
                case "t":
                    message.t = $root.onnx.TensorProto.decodeText(reader);
                    break;
                case "g":
                    message.g = $root.onnx.GraphProto.decodeText(reader);
                    break;
                case "sparse_tensor":
                    message.sparse_tensor = $root.onnx.SparseTensorProto.decodeText(reader);
                    break;
                case "tp":
                    message.tp = $root.onnx.TypeProto.decodeText(reader);
                    break;
                case "floats":
                    reader.array(message.floats, () => reader.float());
                    break;
                case "ints":
                    reader.array(message.ints, () => reader.int64());
                    break;
                case "strings":
                    reader.array(message.strings, () => reader.bytes());
                    break;
                case "tensors":
                    message.tensors.push($root.onnx.TensorProto.decodeText(reader));
                    break;
                case "graphs":
                    message.graphs.push($root.onnx.GraphProto.decodeText(reader));
                    break;
                case "sparse_tensors":
                    message.sparse_tensors.push($root.onnx.SparseTensorProto.decodeText(reader));
                    break;
                case "type_protos":
                    message.type_protos.push($root.onnx.TypeProto.decodeText(reader));
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

$root.onnx.AttributeProto.prototype.name = "";
$root.onnx.AttributeProto.prototype.ref_attr_name = "";
$root.onnx.AttributeProto.prototype.doc_string = "";
$root.onnx.AttributeProto.prototype.type = 0;
$root.onnx.AttributeProto.prototype.f = 0;
$root.onnx.AttributeProto.prototype.i = protobuf.Int64.create(0);
$root.onnx.AttributeProto.prototype.s = new Uint8Array([]);
$root.onnx.AttributeProto.prototype.t = null;
$root.onnx.AttributeProto.prototype.g = null;
$root.onnx.AttributeProto.prototype.sparse_tensor = null;
$root.onnx.AttributeProto.prototype.tp = null;

$root.onnx.AttributeProto.AttributeType = {
    "UNDEFINED": 0,
    "FLOAT": 1,
    "INT": 2,
    "STRING": 3,
    "TENSOR": 4,
    "GRAPH": 5,
    "SPARSE_TENSOR": 11,
    "TYPE_PROTO": 13,
    "FLOATS": 6,
    "INTS": 7,
    "STRINGS": 8,
    "TENSORS": 9,
    "GRAPHS": 10,
    "SPARSE_TENSORS": 12,
    "TYPE_PROTOS": 14
};

$root.onnx.ValueInfoProto = class ValueInfoProto {

    constructor() {
    }

    static decode(reader, length) {
        const message = new $root.onnx.ValueInfoProto();
        const end = length !== undefined ? reader.position + length : reader.length;
        while (reader.position < end) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.name = reader.string();
                    break;
                case 2:
                    message.type = $root.onnx.TypeProto.decode(reader, reader.uint32());
                    break;
                case 3:
                    message.doc_string = reader.string();
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }

    static decodeText(reader) {
        const message = new $root.onnx.ValueInfoProto();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "name":
                    message.name = reader.string();
                    break;
                case "type":
                    message.type = $root.onnx.TypeProto.decodeText(reader);
                    break;
                case "doc_string":
                    message.doc_string = reader.string();
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

$root.onnx.ValueInfoProto.prototype.name = "";
$root.onnx.ValueInfoProto.prototype.type = null;
$root.onnx.ValueInfoProto.prototype.doc_string = "";

$root.onnx.NodeProto = class NodeProto {

    constructor() {
        this.input = [];
        this.output = [];
        this.attribute = [];
    }

    static decode(reader, length) {
        const message = new $root.onnx.NodeProto();
        const end = length !== undefined ? reader.position + length : reader.length;
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
                case 7:
                    message.domain = reader.string();
                    break;
                case 5:
                    message.attribute.push($root.onnx.AttributeProto.decode(reader, reader.uint32()));
                    break;
                case 6:
                    message.doc_string = reader.string();
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }

    static decodeText(reader) {
        const message = new $root.onnx.NodeProto();
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
                case "domain":
                    message.domain = reader.string();
                    break;
                case "attribute":
                    message.attribute.push($root.onnx.AttributeProto.decodeText(reader));
                    break;
                case "doc_string":
                    message.doc_string = reader.string();
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

$root.onnx.NodeProto.prototype.name = "";
$root.onnx.NodeProto.prototype.op_type = "";
$root.onnx.NodeProto.prototype.domain = "";
$root.onnx.NodeProto.prototype.doc_string = "";

$root.onnx.TrainingInfoProto = class TrainingInfoProto {

    constructor() {
        this.initialization_binding = [];
        this.update_binding = [];
    }

    static decode(reader, length) {
        const message = new $root.onnx.TrainingInfoProto();
        const end = length !== undefined ? reader.position + length : reader.length;
        while (reader.position < end) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.initialization = $root.onnx.GraphProto.decode(reader, reader.uint32());
                    break;
                case 2:
                    message.algorithm = $root.onnx.GraphProto.decode(reader, reader.uint32());
                    break;
                case 3:
                    message.initialization_binding.push($root.onnx.StringStringEntryProto.decode(reader, reader.uint32()));
                    break;
                case 4:
                    message.update_binding.push($root.onnx.StringStringEntryProto.decode(reader, reader.uint32()));
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }

    static decodeText(reader) {
        const message = new $root.onnx.TrainingInfoProto();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "initialization":
                    message.initialization = $root.onnx.GraphProto.decodeText(reader);
                    break;
                case "algorithm":
                    message.algorithm = $root.onnx.GraphProto.decodeText(reader);
                    break;
                case "initialization_binding":
                    message.initialization_binding.push($root.onnx.StringStringEntryProto.decodeText(reader));
                    break;
                case "update_binding":
                    message.update_binding.push($root.onnx.StringStringEntryProto.decodeText(reader));
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

$root.onnx.TrainingInfoProto.prototype.initialization = null;
$root.onnx.TrainingInfoProto.prototype.algorithm = null;

$root.onnx.ModelProto = class ModelProto {

    constructor() {
        this.opset_import = [];
        this.metadata_props = [];
        this.training_info = [];
        this.functions = [];
    }

    static decode(reader, length) {
        const message = new $root.onnx.ModelProto();
        const end = length !== undefined ? reader.position + length : reader.length;
        while (reader.position < end) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.ir_version = reader.int64();
                    break;
                case 8:
                    message.opset_import.push($root.onnx.OperatorSetIdProto.decode(reader, reader.uint32()));
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
                    message.model_version = reader.int64();
                    break;
                case 6:
                    message.doc_string = reader.string();
                    break;
                case 7:
                    message.graph = $root.onnx.GraphProto.decode(reader, reader.uint32());
                    break;
                case 14:
                    message.metadata_props.push($root.onnx.StringStringEntryProto.decode(reader, reader.uint32()));
                    break;
                case 20:
                    message.training_info.push($root.onnx.TrainingInfoProto.decode(reader, reader.uint32()));
                    break;
                case 25:
                    message.functions.push($root.onnx.FunctionProto.decode(reader, reader.uint32()));
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }

    static decodeText(reader) {
        const message = new $root.onnx.ModelProto();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "ir_version":
                    message.ir_version = reader.int64();
                    break;
                case "opset_import":
                    message.opset_import.push($root.onnx.OperatorSetIdProto.decodeText(reader));
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
                    message.model_version = reader.int64();
                    break;
                case "doc_string":
                    message.doc_string = reader.string();
                    break;
                case "graph":
                    message.graph = $root.onnx.GraphProto.decodeText(reader);
                    break;
                case "metadata_props":
                    message.metadata_props.push($root.onnx.StringStringEntryProto.decodeText(reader));
                    break;
                case "training_info":
                    message.training_info.push($root.onnx.TrainingInfoProto.decodeText(reader));
                    break;
                case "functions":
                    message.functions.push($root.onnx.FunctionProto.decodeText(reader));
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

$root.onnx.ModelProto.prototype.ir_version = protobuf.Int64.create(0);
$root.onnx.ModelProto.prototype.producer_name = "";
$root.onnx.ModelProto.prototype.producer_version = "";
$root.onnx.ModelProto.prototype.domain = "";
$root.onnx.ModelProto.prototype.model_version = protobuf.Int64.create(0);
$root.onnx.ModelProto.prototype.doc_string = "";
$root.onnx.ModelProto.prototype.graph = null;

$root.onnx.StringStringEntryProto = class StringStringEntryProto {

    constructor() {
    }

    static decode(reader, length) {
        const message = new $root.onnx.StringStringEntryProto();
        const end = length !== undefined ? reader.position + length : reader.length;
        while (reader.position < end) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.key = reader.string();
                    break;
                case 2:
                    message.value = reader.string();
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }

    static decodeText(reader) {
        const message = new $root.onnx.StringStringEntryProto();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "key":
                    message.key = reader.string();
                    break;
                case "value":
                    message.value = reader.string();
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

$root.onnx.StringStringEntryProto.prototype.key = "";
$root.onnx.StringStringEntryProto.prototype.value = "";

$root.onnx.TensorAnnotation = class TensorAnnotation {

    constructor() {
        this.quant_parameter_tensor_names = [];
    }

    static decode(reader, length) {
        const message = new $root.onnx.TensorAnnotation();
        const end = length !== undefined ? reader.position + length : reader.length;
        while (reader.position < end) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.tensor_name = reader.string();
                    break;
                case 2:
                    message.quant_parameter_tensor_names.push($root.onnx.StringStringEntryProto.decode(reader, reader.uint32()));
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }

    static decodeText(reader) {
        const message = new $root.onnx.TensorAnnotation();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "tensor_name":
                    message.tensor_name = reader.string();
                    break;
                case "quant_parameter_tensor_names":
                    message.quant_parameter_tensor_names.push($root.onnx.StringStringEntryProto.decodeText(reader));
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

$root.onnx.TensorAnnotation.prototype.tensor_name = "";

$root.onnx.GraphProto = class GraphProto {

    constructor() {
        this.node = [];
        this.initializer = [];
        this.sparse_initializer = [];
        this.input = [];
        this.output = [];
        this.value_info = [];
        this.quantization_annotation = [];
    }

    static decode(reader, length) {
        const message = new $root.onnx.GraphProto();
        const end = length !== undefined ? reader.position + length : reader.length;
        while (reader.position < end) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.node.push($root.onnx.NodeProto.decode(reader, reader.uint32()));
                    break;
                case 2:
                    message.name = reader.string();
                    break;
                case 5:
                    message.initializer.push($root.onnx.TensorProto.decode(reader, reader.uint32()));
                    break;
                case 15:
                    message.sparse_initializer.push($root.onnx.SparseTensorProto.decode(reader, reader.uint32()));
                    break;
                case 10:
                    message.doc_string = reader.string();
                    break;
                case 11:
                    message.input.push($root.onnx.ValueInfoProto.decode(reader, reader.uint32()));
                    break;
                case 12:
                    message.output.push($root.onnx.ValueInfoProto.decode(reader, reader.uint32()));
                    break;
                case 13:
                    message.value_info.push($root.onnx.ValueInfoProto.decode(reader, reader.uint32()));
                    break;
                case 14:
                    message.quantization_annotation.push($root.onnx.TensorAnnotation.decode(reader, reader.uint32()));
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }

    static decodeText(reader) {
        const message = new $root.onnx.GraphProto();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "node":
                    message.node.push($root.onnx.NodeProto.decodeText(reader));
                    break;
                case "name":
                    message.name = reader.string();
                    break;
                case "initializer":
                    message.initializer.push($root.onnx.TensorProto.decodeText(reader));
                    break;
                case "sparse_initializer":
                    message.sparse_initializer.push($root.onnx.SparseTensorProto.decodeText(reader));
                    break;
                case "doc_string":
                    message.doc_string = reader.string();
                    break;
                case "input":
                    message.input.push($root.onnx.ValueInfoProto.decodeText(reader));
                    break;
                case "output":
                    message.output.push($root.onnx.ValueInfoProto.decodeText(reader));
                    break;
                case "value_info":
                    message.value_info.push($root.onnx.ValueInfoProto.decodeText(reader));
                    break;
                case "quantization_annotation":
                    message.quantization_annotation.push($root.onnx.TensorAnnotation.decodeText(reader));
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

$root.onnx.GraphProto.prototype.name = "";
$root.onnx.GraphProto.prototype.doc_string = "";

$root.onnx.TensorProto = class TensorProto {

    constructor() {
        this.dims = [];
        this.float_data = [];
        this.int32_data = [];
        this.string_data = [];
        this.int64_data = [];
        this.external_data = [];
        this.double_data = [];
        this.uint64_data = [];
    }

    static decode(reader, length) {
        const message = new $root.onnx.TensorProto();
        const end = length !== undefined ? reader.position + length : reader.length;
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
                    message.segment = $root.onnx.TensorProto.Segment.decode(reader, reader.uint32());
                    break;
                case 4:
                    message.float_data = reader.floats(message.float_data, tag);
                    break;
                case 5:
                    message.int32_data = reader.array(message.int32_data, () => reader.int32(), tag);
                    break;
                case 6:
                    message.string_data.push(reader.bytes());
                    break;
                case 7:
                    message.int64_data = reader.array(message.int64_data, () => reader.int64(), tag);
                    break;
                case 8:
                    message.name = reader.string();
                    break;
                case 12:
                    message.doc_string = reader.string();
                    break;
                case 9:
                    message.raw_data = reader.bytes();
                    break;
                case 13:
                    message.external_data.push($root.onnx.StringStringEntryProto.decode(reader, reader.uint32()));
                    break;
                case 14:
                    message.data_location = reader.int32();
                    break;
                case 10:
                    message.double_data = reader.doubles(message.double_data, tag);
                    break;
                case 11:
                    message.uint64_data = reader.array(message.uint64_data, () => reader.uint64(), tag);
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }

    static decodeText(reader) {
        const message = new $root.onnx.TensorProto();
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
                case "segment":
                    message.segment = $root.onnx.TensorProto.Segment.decodeText(reader);
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
                case "external_data":
                    message.external_data.push($root.onnx.StringStringEntryProto.decodeText(reader));
                    break;
                case "data_location":
                    message.data_location = reader.enum($root.onnx.TensorProto.DataLocation);
                    break;
                case "double_data":
                    reader.array(message.double_data, () => reader.double());
                    break;
                case "uint64_data":
                    reader.array(message.uint64_data, () => reader.uint64());
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

$root.onnx.TensorProto.prototype.data_type = 0;
$root.onnx.TensorProto.prototype.segment = null;
$root.onnx.TensorProto.prototype.name = "";
$root.onnx.TensorProto.prototype.doc_string = "";
$root.onnx.TensorProto.prototype.raw_data = new Uint8Array([]);
$root.onnx.TensorProto.prototype.data_location = 0;

$root.onnx.TensorProto.DataType = {
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
    "FLOAT8E4M3FN": 17,
    "FLOAT8E4M3FNUZ": 18,
    "FLOAT8E5M2": 19,
    "FLOAT8E5M2FNUZ": 20
};

$root.onnx.TensorProto.Segment = class Segment {

    constructor() {
    }

    static decode(reader, length) {
        const message = new $root.onnx.TensorProto.Segment();
        const end = length !== undefined ? reader.position + length : reader.length;
        while (reader.position < end) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.begin = reader.int64();
                    break;
                case 2:
                    message.end = reader.int64();
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }

    static decodeText(reader) {
        const message = new $root.onnx.TensorProto.Segment();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "begin":
                    message.begin = reader.int64();
                    break;
                case "end":
                    message.end = reader.int64();
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

$root.onnx.TensorProto.Segment.prototype.begin = protobuf.Int64.create(0);
$root.onnx.TensorProto.Segment.prototype.end = protobuf.Int64.create(0);

$root.onnx.TensorProto.DataLocation = {
    "DEFAULT": 0,
    "EXTERNAL": 1
};

$root.onnx.SparseTensorProto = class SparseTensorProto {

    constructor() {
        this.dims = [];
    }

    static decode(reader, length) {
        const message = new $root.onnx.SparseTensorProto();
        const end = length !== undefined ? reader.position + length : reader.length;
        while (reader.position < end) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.values = $root.onnx.TensorProto.decode(reader, reader.uint32());
                    break;
                case 2:
                    message.indices = $root.onnx.TensorProto.decode(reader, reader.uint32());
                    break;
                case 3:
                    message.dims = reader.array(message.dims, () => reader.int64(), tag);
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }

    static decodeText(reader) {
        const message = new $root.onnx.SparseTensorProto();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "values":
                    message.values = $root.onnx.TensorProto.decodeText(reader);
                    break;
                case "indices":
                    message.indices = $root.onnx.TensorProto.decodeText(reader);
                    break;
                case "dims":
                    reader.array(message.dims, () => reader.int64());
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

$root.onnx.SparseTensorProto.prototype.values = null;
$root.onnx.SparseTensorProto.prototype.indices = null;

$root.onnx.TensorShapeProto = class TensorShapeProto {

    constructor() {
        this.dim = [];
    }

    static decode(reader, length) {
        const message = new $root.onnx.TensorShapeProto();
        const end = length !== undefined ? reader.position + length : reader.length;
        while (reader.position < end) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.dim.push($root.onnx.TensorShapeProto.Dimension.decode(reader, reader.uint32()));
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }

    static decodeText(reader) {
        const message = new $root.onnx.TensorShapeProto();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "dim":
                    message.dim.push($root.onnx.TensorShapeProto.Dimension.decodeText(reader));
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

$root.onnx.TensorShapeProto.Dimension = class Dimension {

    constructor() {
    }

    get value() {
        $root.onnx.TensorShapeProto.Dimension.valueSet = $root.onnx.TensorShapeProto.Dimension.valueSet || new Set([ "dim_value", "dim_param"]);
        return Object.keys(this).find((key) => $root.onnx.TensorShapeProto.Dimension.valueSet.has(key) && this[key] != null);
    }

    static decode(reader, length) {
        const message = new $root.onnx.TensorShapeProto.Dimension();
        const end = length !== undefined ? reader.position + length : reader.length;
        while (reader.position < end) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.dim_value = reader.int64();
                    break;
                case 2:
                    message.dim_param = reader.string();
                    break;
                case 3:
                    message.denotation = reader.string();
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }

    static decodeText(reader) {
        const message = new $root.onnx.TensorShapeProto.Dimension();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "dim_value":
                    message.dim_value = reader.int64();
                    break;
                case "dim_param":
                    message.dim_param = reader.string();
                    break;
                case "denotation":
                    message.denotation = reader.string();
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

$root.onnx.TensorShapeProto.Dimension.prototype.denotation = "";

$root.onnx.TypeProto = class TypeProto {

    constructor() {
    }

    get value() {
        $root.onnx.TypeProto.valueSet = $root.onnx.TypeProto.valueSet || new Set([ "tensor_type", "sequence_type", "map_type", "optional_type", "sparse_tensor_type", "opaque_type"]);
        return Object.keys(this).find((key) => $root.onnx.TypeProto.valueSet.has(key) && this[key] != null);
    }

    static decode(reader, length) {
        const message = new $root.onnx.TypeProto();
        const end = length !== undefined ? reader.position + length : reader.length;
        while (reader.position < end) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.tensor_type = $root.onnx.TypeProto.Tensor.decode(reader, reader.uint32());
                    break;
                case 4:
                    message.sequence_type = $root.onnx.TypeProto.Sequence.decode(reader, reader.uint32());
                    break;
                case 5:
                    message.map_type = $root.onnx.TypeProto.Map.decode(reader, reader.uint32());
                    break;
                case 9:
                    message.optional_type = $root.onnx.TypeProto.Optional.decode(reader, reader.uint32());
                    break;
                case 8:
                    message.sparse_tensor_type = $root.onnx.TypeProto.SparseTensor.decode(reader, reader.uint32());
                    break;
                case 7:
                    message.opaque_type = $root.onnx.TypeProto.Opaque.decode(reader, reader.uint32());
                    break;
                case 6:
                    message.denotation = reader.string();
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }

    static decodeText(reader) {
        const message = new $root.onnx.TypeProto();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "tensor_type":
                    message.tensor_type = $root.onnx.TypeProto.Tensor.decodeText(reader);
                    break;
                case "sequence_type":
                    message.sequence_type = $root.onnx.TypeProto.Sequence.decodeText(reader);
                    break;
                case "map_type":
                    message.map_type = $root.onnx.TypeProto.Map.decodeText(reader);
                    break;
                case "optional_type":
                    message.optional_type = $root.onnx.TypeProto.Optional.decodeText(reader);
                    break;
                case "sparse_tensor_type":
                    message.sparse_tensor_type = $root.onnx.TypeProto.SparseTensor.decodeText(reader);
                    break;
                case "opaque_type":
                    message.opaque_type = $root.onnx.TypeProto.Opaque.decodeText(reader);
                    break;
                case "denotation":
                    message.denotation = reader.string();
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

$root.onnx.TypeProto.prototype.denotation = "";

$root.onnx.TypeProto.Tensor = class Tensor {

    constructor() {
    }

    static decode(reader, length) {
        const message = new $root.onnx.TypeProto.Tensor();
        const end = length !== undefined ? reader.position + length : reader.length;
        while (reader.position < end) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.elem_type = reader.int32();
                    break;
                case 2:
                    message.shape = $root.onnx.TensorShapeProto.decode(reader, reader.uint32());
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }

    static decodeText(reader) {
        const message = new $root.onnx.TypeProto.Tensor();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "elem_type":
                    message.elem_type = reader.int32();
                    break;
                case "shape":
                    message.shape = $root.onnx.TensorShapeProto.decodeText(reader);
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

$root.onnx.TypeProto.Tensor.prototype.elem_type = 0;
$root.onnx.TypeProto.Tensor.prototype.shape = null;

$root.onnx.TypeProto.Sequence = class Sequence {

    constructor() {
    }

    static decode(reader, length) {
        const message = new $root.onnx.TypeProto.Sequence();
        const end = length !== undefined ? reader.position + length : reader.length;
        while (reader.position < end) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.elem_type = $root.onnx.TypeProto.decode(reader, reader.uint32());
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }

    static decodeText(reader) {
        const message = new $root.onnx.TypeProto.Sequence();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "elem_type":
                    message.elem_type = $root.onnx.TypeProto.decodeText(reader);
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

$root.onnx.TypeProto.Sequence.prototype.elem_type = null;

$root.onnx.TypeProto.Map = class Map {

    constructor() {
    }

    static decode(reader, length) {
        const message = new $root.onnx.TypeProto.Map();
        const end = length !== undefined ? reader.position + length : reader.length;
        while (reader.position < end) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.key_type = reader.int32();
                    break;
                case 2:
                    message.value_type = $root.onnx.TypeProto.decode(reader, reader.uint32());
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }

    static decodeText(reader) {
        const message = new $root.onnx.TypeProto.Map();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "key_type":
                    message.key_type = reader.int32();
                    break;
                case "value_type":
                    message.value_type = $root.onnx.TypeProto.decodeText(reader);
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

$root.onnx.TypeProto.Map.prototype.key_type = 0;
$root.onnx.TypeProto.Map.prototype.value_type = null;

$root.onnx.TypeProto.Optional = class Optional {

    constructor() {
    }

    static decode(reader, length) {
        const message = new $root.onnx.TypeProto.Optional();
        const end = length !== undefined ? reader.position + length : reader.length;
        while (reader.position < end) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.elem_type = $root.onnx.TypeProto.decode(reader, reader.uint32());
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }

    static decodeText(reader) {
        const message = new $root.onnx.TypeProto.Optional();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "elem_type":
                    message.elem_type = $root.onnx.TypeProto.decodeText(reader);
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

$root.onnx.TypeProto.Optional.prototype.elem_type = null;

$root.onnx.TypeProto.SparseTensor = class SparseTensor {

    constructor() {
    }

    static decode(reader, length) {
        const message = new $root.onnx.TypeProto.SparseTensor();
        const end = length !== undefined ? reader.position + length : reader.length;
        while (reader.position < end) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.elem_type = reader.int32();
                    break;
                case 2:
                    message.shape = $root.onnx.TensorShapeProto.decode(reader, reader.uint32());
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }

    static decodeText(reader) {
        const message = new $root.onnx.TypeProto.SparseTensor();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "elem_type":
                    message.elem_type = reader.int32();
                    break;
                case "shape":
                    message.shape = $root.onnx.TensorShapeProto.decodeText(reader);
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

$root.onnx.TypeProto.SparseTensor.prototype.elem_type = 0;
$root.onnx.TypeProto.SparseTensor.prototype.shape = null;

$root.onnx.TypeProto.Opaque = class Opaque {

    constructor() {
    }

    static decode(reader, length) {
        const message = new $root.onnx.TypeProto.Opaque();
        const end = length !== undefined ? reader.position + length : reader.length;
        while (reader.position < end) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.domain = reader.string();
                    break;
                case 2:
                    message.name = reader.string();
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }

    static decodeText(reader) {
        const message = new $root.onnx.TypeProto.Opaque();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "domain":
                    message.domain = reader.string();
                    break;
                case "name":
                    message.name = reader.string();
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

$root.onnx.TypeProto.Opaque.prototype.domain = "";
$root.onnx.TypeProto.Opaque.prototype.name = "";

$root.onnx.OperatorSetIdProto = class OperatorSetIdProto {

    constructor() {
    }

    static decode(reader, length) {
        const message = new $root.onnx.OperatorSetIdProto();
        const end = length !== undefined ? reader.position + length : reader.length;
        while (reader.position < end) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.domain = reader.string();
                    break;
                case 2:
                    message.version = reader.int64();
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }

    static decodeText(reader) {
        const message = new $root.onnx.OperatorSetIdProto();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "domain":
                    message.domain = reader.string();
                    break;
                case "version":
                    message.version = reader.int64();
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

$root.onnx.OperatorSetIdProto.prototype.domain = "";
$root.onnx.OperatorSetIdProto.prototype.version = protobuf.Int64.create(0);

$root.onnx.OperatorStatus = {
    "EXPERIMENTAL": 0,
    "STABLE": 1
};

$root.onnx.FunctionProto = class FunctionProto {

    constructor() {
        this.input = [];
        this.output = [];
        this.attribute = [];
        this.attribute_proto = [];
        this.node = [];
        this.opset_import = [];
    }

    static decode(reader, length) {
        const message = new $root.onnx.FunctionProto();
        const end = length !== undefined ? reader.position + length : reader.length;
        while (reader.position < end) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.name = reader.string();
                    break;
                case 4:
                    message.input.push(reader.string());
                    break;
                case 5:
                    message.output.push(reader.string());
                    break;
                case 6:
                    message.attribute.push(reader.string());
                    break;
                case 11:
                    message.attribute_proto.push($root.onnx.AttributeProto.decode(reader, reader.uint32()));
                    break;
                case 7:
                    message.node.push($root.onnx.NodeProto.decode(reader, reader.uint32()));
                    break;
                case 8:
                    message.doc_string = reader.string();
                    break;
                case 9:
                    message.opset_import.push($root.onnx.OperatorSetIdProto.decode(reader, reader.uint32()));
                    break;
                case 10:
                    message.domain = reader.string();
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }

    static decodeText(reader) {
        const message = new $root.onnx.FunctionProto();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "name":
                    message.name = reader.string();
                    break;
                case "input":
                    reader.array(message.input, () => reader.string());
                    break;
                case "output":
                    reader.array(message.output, () => reader.string());
                    break;
                case "attribute":
                    reader.array(message.attribute, () => reader.string());
                    break;
                case "attribute_proto":
                    message.attribute_proto.push($root.onnx.AttributeProto.decodeText(reader));
                    break;
                case "node":
                    message.node.push($root.onnx.NodeProto.decodeText(reader));
                    break;
                case "doc_string":
                    message.doc_string = reader.string();
                    break;
                case "opset_import":
                    message.opset_import.push($root.onnx.OperatorSetIdProto.decodeText(reader));
                    break;
                case "domain":
                    message.domain = reader.string();
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

$root.onnx.FunctionProto.prototype.name = "";
$root.onnx.FunctionProto.prototype.doc_string = "";
$root.onnx.FunctionProto.prototype.domain = "";

$root.onnx.OperatorProto = class OperatorProto {

    constructor() {
    }

    static decode(reader, length) {
        const message = new $root.onnx.OperatorProto();
        const end = length !== undefined ? reader.position + length : reader.length;
        while (reader.position < end) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.op_type = reader.string();
                    break;
                case 2:
                    message.since_version = reader.int64();
                    break;
                case 3:
                    message.status = reader.int32();
                    break;
                case 10:
                    message.doc_string = reader.string();
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }

    static decodeText(reader) {
        const message = new $root.onnx.OperatorProto();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "op_type":
                    message.op_type = reader.string();
                    break;
                case "since_version":
                    message.since_version = reader.int64();
                    break;
                case "status":
                    message.status = reader.enum($root.onnx.OperatorStatus);
                    break;
                case "doc_string":
                    message.doc_string = reader.string();
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

$root.onnx.OperatorProto.prototype.op_type = "";
$root.onnx.OperatorProto.prototype.since_version = protobuf.Int64.create(0);
$root.onnx.OperatorProto.prototype.status = 0;
$root.onnx.OperatorProto.prototype.doc_string = "";

$root.onnx.OperatorSetProto = class OperatorSetProto {

    constructor() {
        this.operator = [];
        this.functions = [];
    }

    static decode(reader, length) {
        const message = new $root.onnx.OperatorSetProto();
        const end = length !== undefined ? reader.position + length : reader.length;
        while (reader.position < end) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.magic = reader.string();
                    break;
                case 2:
                    message.ir_version = reader.int64();
                    break;
                case 3:
                    message.ir_version_prerelease = reader.string();
                    break;
                case 7:
                    message.ir_build_metadata = reader.string();
                    break;
                case 4:
                    message.domain = reader.string();
                    break;
                case 5:
                    message.opset_version = reader.int64();
                    break;
                case 6:
                    message.doc_string = reader.string();
                    break;
                case 8:
                    message.operator.push($root.onnx.OperatorProto.decode(reader, reader.uint32()));
                    break;
                case 9:
                    message.functions.push($root.onnx.FunctionProto.decode(reader, reader.uint32()));
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }

    static decodeText(reader) {
        const message = new $root.onnx.OperatorSetProto();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "magic":
                    message.magic = reader.string();
                    break;
                case "ir_version":
                    message.ir_version = reader.int64();
                    break;
                case "ir_version_prerelease":
                    message.ir_version_prerelease = reader.string();
                    break;
                case "ir_build_metadata":
                    message.ir_build_metadata = reader.string();
                    break;
                case "domain":
                    message.domain = reader.string();
                    break;
                case "opset_version":
                    message.opset_version = reader.int64();
                    break;
                case "doc_string":
                    message.doc_string = reader.string();
                    break;
                case "operator":
                    message.operator.push($root.onnx.OperatorProto.decodeText(reader));
                    break;
                case "functions":
                    message.functions.push($root.onnx.FunctionProto.decodeText(reader));
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

$root.onnx.OperatorSetProto.prototype.magic = "";
$root.onnx.OperatorSetProto.prototype.ir_version = protobuf.Int64.create(0);
$root.onnx.OperatorSetProto.prototype.ir_version_prerelease = "";
$root.onnx.OperatorSetProto.prototype.ir_build_metadata = "";
$root.onnx.OperatorSetProto.prototype.domain = "";
$root.onnx.OperatorSetProto.prototype.opset_version = protobuf.Int64.create(0);
$root.onnx.OperatorSetProto.prototype.doc_string = "";
