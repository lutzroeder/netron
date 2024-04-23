
export const caffe2 = {};

caffe2.TensorProto = class TensorProto {

    constructor() {
        this.dims = [];
        this.float_data = [];
        this.int32_data = [];
        this.string_data = [];
        this.double_data = [];
        this.int64_data = [];
    }

    static decode(reader, length) {
        const message = new caffe2.TensorProto();
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
                case 15:
                    message.data_format = reader.uint32();
                    break;
                case 3:
                    message.float_data = reader.floats(message.float_data, tag);
                    break;
                case 4:
                    message.int32_data = reader.array(message.int32_data, () => reader.int32(), tag);
                    break;
                case 5:
                    message.byte_data = reader.bytes();
                    break;
                case 6:
                    message.string_data.push(reader.bytes());
                    break;
                case 9:
                    message.double_data = reader.doubles(message.double_data, tag);
                    break;
                case 10:
                    message.int64_data = reader.array(message.int64_data, () => reader.int64(), tag);
                    break;
                case 13:
                    message.raw_data = reader.bytes();
                    break;
                case 7:
                    message.name = reader.string();
                    break;
                case 8:
                    message.device_detail = caffe2.DeviceOption.decode(reader, reader.uint32());
                    break;
                case 11:
                    message.segment = caffe2.TensorProto.Segment.decode(reader, reader.uint32());
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }

    static decodeText(reader) {
        const message = new caffe2.TensorProto();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "dims":
                    reader.array(message.dims, () => reader.int64());
                    break;
                case "data_type":
                    message.data_type = reader.enum(caffe2.TensorProto.DataType);
                    break;
                case "data_format":
                    message.data_format = reader.uint32();
                    break;
                case "float_data":
                    reader.array(message.float_data, () => reader.float());
                    break;
                case "int32_data":
                    reader.array(message.int32_data, () => reader.int32());
                    break;
                case "byte_data":
                    message.byte_data = reader.bytes();
                    break;
                case "string_data":
                    reader.array(message.string_data, () => reader.bytes());
                    break;
                case "double_data":
                    reader.array(message.double_data, () => reader.double());
                    break;
                case "int64_data":
                    reader.array(message.int64_data, () => reader.int64());
                    break;
                case "raw_data":
                    message.raw_data = reader.bytes();
                    break;
                case "name":
                    message.name = reader.string();
                    break;
                case "device_detail":
                    message.device_detail = caffe2.DeviceOption.decodeText(reader);
                    break;
                case "segment":
                    message.segment = caffe2.TensorProto.Segment.decodeText(reader);
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

caffe2.TensorProto.prototype.data_type = 1;
caffe2.TensorProto.prototype.data_format = 0;
caffe2.TensorProto.prototype.byte_data = new Uint8Array([]);
caffe2.TensorProto.prototype.raw_data = new Uint8Array([]);
caffe2.TensorProto.prototype.name = "";
caffe2.TensorProto.prototype.device_detail = null;
caffe2.TensorProto.prototype.segment = null;

caffe2.TensorProto.DataType = {
    "UNDEFINED": 0,
    "FLOAT": 1,
    "INT32": 2,
    "BYTE": 3,
    "STRING": 4,
    "BOOL": 5,
    "UINT8": 6,
    "INT8": 7,
    "UINT16": 8,
    "INT16": 9,
    "INT64": 10,
    "FLOAT16": 12,
    "DOUBLE": 13,
    "ZERO_COLLISION_HASH": 14,
    "REBATCHING_BUFFER": 15
};

caffe2.TensorProto.SerializationFormat = {
    "FMT_PROTOBUF": 0,
    "FMT_BFLOAT16": 1
};

caffe2.TensorProto.Segment = class Segment {

    static decode(reader, length) {
        const message = new caffe2.TensorProto.Segment();
        const end = length === undefined ? reader.length : reader.position + length;
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
        if (!Object.prototype.hasOwnProperty.call(message, 'begin')) {
            throw new Error("Excepted 'begin'.");
        }
        if (!Object.prototype.hasOwnProperty.call(message, 'end')) {
            throw new Error("Excepted 'end'.");
        }
        return message;
    }

    static decodeText(reader) {
        const message = new caffe2.TensorProto.Segment();
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
        if (!Object.prototype.hasOwnProperty.call(message, "begin")) {
            throw new Error("Excepted 'begin'.");
        }
        if (!Object.prototype.hasOwnProperty.call(message, "end")) {
            throw new Error("Excepted 'end'.");
        }
        return message;
    }
};

caffe2.TensorProto.Segment.prototype.begin = 0n;
caffe2.TensorProto.Segment.prototype.end = 0n;

caffe2.QTensorProto = class QTensorProto {

    constructor() {
        this.dims = [];
        this.data = [];
        this.scales = [];
        this.biases = [];
    }

    static decode(reader, length) {
        const message = new caffe2.QTensorProto();
        const end = length === undefined ? reader.length : reader.position + length;
        while (reader.position < end) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.dims = reader.array(message.dims, () => reader.int64(), tag);
                    break;
                case 2:
                    message.precision = reader.int32();
                    break;
                case 3:
                    message.scale = reader.double();
                    break;
                case 4:
                    message.bias = reader.double();
                    break;
                case 5:
                    message.is_signed = reader.bool();
                    break;
                case 6:
                    message.data = reader.array(message.data, () => reader.int32(), tag);
                    break;
                case 7:
                    message.name = reader.string();
                    break;
                case 8:
                    message.data_type = reader.int32();
                    break;
                case 9:
                    message.scales = reader.doubles(message.scales, tag);
                    break;
                case 10:
                    message.biases = reader.doubles(message.biases, tag);
                    break;
                case 11:
                    message.axis = reader.int32();
                    break;
                case 12:
                    message.is_multiparam = reader.bool();
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        if (!Object.prototype.hasOwnProperty.call(message, 'precision')) {
            throw new Error("Excepted 'precision'.");
        }
        if (!Object.prototype.hasOwnProperty.call(message, 'scale')) {
            throw new Error("Excepted 'scale'.");
        }
        if (!Object.prototype.hasOwnProperty.call(message, 'bias')) {
            throw new Error("Excepted 'bias'.");
        }
        if (!Object.prototype.hasOwnProperty.call(message, 'is_signed')) {
            throw new Error("Excepted 'is_signed'.");
        }
        return message;
    }

    static decodeText(reader) {
        const message = new caffe2.QTensorProto();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "dims":
                    reader.array(message.dims, () => reader.int64());
                    break;
                case "precision":
                    message.precision = reader.int32();
                    break;
                case "scale":
                    message.scale = reader.double();
                    break;
                case "bias":
                    message.bias = reader.double();
                    break;
                case "is_signed":
                    message.is_signed = reader.bool();
                    break;
                case "data":
                    reader.array(message.data, () => reader.int32());
                    break;
                case "name":
                    message.name = reader.string();
                    break;
                case "data_type":
                    message.data_type = reader.enum(caffe2.TensorProto.DataType);
                    break;
                case "scales":
                    reader.array(message.scales, () => reader.double());
                    break;
                case "biases":
                    reader.array(message.biases, () => reader.double());
                    break;
                case "axis":
                    message.axis = reader.int32();
                    break;
                case "is_multiparam":
                    message.is_multiparam = reader.bool();
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        if (!Object.prototype.hasOwnProperty.call(message, "precision")) {
            throw new Error("Excepted 'precision'.");
        }
        if (!Object.prototype.hasOwnProperty.call(message, "scale")) {
            throw new Error("Excepted 'scale'.");
        }
        if (!Object.prototype.hasOwnProperty.call(message, "bias")) {
            throw new Error("Excepted 'bias'.");
        }
        if (!Object.prototype.hasOwnProperty.call(message, "is_signed")) {
            throw new Error("Excepted 'is_signed'.");
        }
        return message;
    }
};

caffe2.QTensorProto.prototype.precision = 0;
caffe2.QTensorProto.prototype.scale = 0;
caffe2.QTensorProto.prototype.bias = 0;
caffe2.QTensorProto.prototype.is_signed = false;
caffe2.QTensorProto.prototype.name = "";
caffe2.QTensorProto.prototype.data_type = 2;
caffe2.QTensorProto.prototype.axis = 0;
caffe2.QTensorProto.prototype.is_multiparam = false;

caffe2.TensorProtos = class TensorProtos {

    constructor() {
        this.protos = [];
    }

    static decode(reader, length) {
        const message = new caffe2.TensorProtos();
        const end = length === undefined ? reader.length : reader.position + length;
        while (reader.position < end) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.protos.push(caffe2.TensorProto.decode(reader, reader.uint32()));
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }

    static decodeText(reader) {
        const message = new caffe2.TensorProtos();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "protos":
                    message.protos.push(caffe2.TensorProto.decodeText(reader));
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

caffe2.TensorShape = class TensorShape {

    constructor() {
        this.dims = [];
        this.unknown_dims = [];
    }

    static decode(reader, length) {
        const message = new caffe2.TensorShape();
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
                    message.unknown_dims = reader.array(message.unknown_dims, () => reader.int32(), tag);
                    break;
                case 4:
                    message.unknown_shape = reader.bool();
                    break;
                case 5:
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
        const message = new caffe2.TensorShape();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "dims":
                    reader.array(message.dims, () => reader.int64());
                    break;
                case "data_type":
                    message.data_type = reader.enum(caffe2.TensorProto.DataType);
                    break;
                case "unknown_dims":
                    reader.array(message.unknown_dims, () => reader.int32());
                    break;
                case "unknown_shape":
                    message.unknown_shape = reader.bool();
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

caffe2.TensorShape.prototype.data_type = 1;
caffe2.TensorShape.prototype.unknown_shape = false;
caffe2.TensorShape.prototype.name = "";

caffe2.TensorShapes = class TensorShapes {

    constructor() {
        this.shapes = [];
    }

    static decode(reader, length) {
        const message = new caffe2.TensorShapes();
        const end = length === undefined ? reader.length : reader.position + length;
        while (reader.position < end) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.shapes.push(caffe2.TensorShape.decode(reader, reader.uint32()));
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }

    static decodeText(reader) {
        const message = new caffe2.TensorShapes();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "shapes":
                    message.shapes.push(caffe2.TensorShape.decodeText(reader));
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

caffe2.TensorBoundShape = class TensorBoundShape {

    constructor() {
        this.dim_type = [];
    }

    static decode(reader, length) {
        const message = new caffe2.TensorBoundShape();
        const end = length === undefined ? reader.length : reader.position + length;
        while (reader.position < end) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.shape = caffe2.TensorShape.decode(reader, reader.uint32());
                    break;
                case 2:
                    message.dim_type = reader.array(message.dim_type, () => reader.int32(), tag);
                    break;
                case 3:
                    message.name = reader.string();
                    break;
                case 4:
                    message.shape_is_final = reader.bool();
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }

    static decodeText(reader) {
        const message = new caffe2.TensorBoundShape();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "shape":
                    message.shape = caffe2.TensorShape.decodeText(reader);
                    break;
                case "dim_type":
                    reader.array(message.dim_type, () => reader.enum(caffe2.TensorBoundShape.DimType));
                    break;
                case "name":
                    message.name = reader.string();
                    break;
                case "shape_is_final":
                    message.shape_is_final = reader.bool();
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

caffe2.TensorBoundShape.prototype.shape = null;
caffe2.TensorBoundShape.prototype.name = "";
caffe2.TensorBoundShape.prototype.shape_is_final = false;

caffe2.TensorBoundShape.DimType = {
    "UNKNOWN": 0,
    "CONSTANT": 1,
    "BATCH": 2,
    "BATCH_OF_FEATURE_MAX": 3,
    "BATCH_OF_FEATURE_MAX_DEFAULT": 4,
    "FEATURE_MAX": 5,
    "FEATURE_MAX_DEFAULT": 6
};

caffe2.TensorBoundShapes = class TensorBoundShapes {

    constructor() {
        this.shapes = [];
    }

    static decode(reader, length) {
        const message = new caffe2.TensorBoundShapes();
        const end = length === undefined ? reader.length : reader.position + length;
        while (reader.position < end) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.shapes.push(caffe2.TensorBoundShape.decode(reader, reader.uint32()));
                    break;
                case 2:
                    message.max_batch_size = reader.int64();
                    break;
                case 3:
                    message.max_feature_len = reader.int64();
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }

    static decodeText(reader) {
        const message = new caffe2.TensorBoundShapes();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "shapes":
                    message.shapes.push(caffe2.TensorBoundShape.decodeText(reader));
                    break;
                case "max_batch_size":
                    message.max_batch_size = reader.int64();
                    break;
                case "max_feature_len":
                    message.max_feature_len = reader.int64();
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

caffe2.TensorBoundShapes.prototype.max_batch_size = 0n;
caffe2.TensorBoundShapes.prototype.max_feature_len = 0n;

caffe2.AOTConfig = class AOTConfig {

    static decode(reader, length) {
        const message = new caffe2.AOTConfig();
        const end = length === undefined ? reader.length : reader.position + length;
        while (reader.position < end) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.max_batch_size = reader.int64();
                    break;
                case 2:
                    message.max_seq_size = reader.int64();
                    break;
                case 3:
                    message.in_batch_broadcast = reader.bool();
                    break;
                case 4:
                    message.onnxifi_blacklist_ops = reader.string();
                    break;
                case 5:
                    message.onnxifi_min_ops = reader.int32();
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        if (!Object.prototype.hasOwnProperty.call(message, 'max_batch_size')) {
            throw new Error("Excepted 'max_batch_size'.");
        }
        if (!Object.prototype.hasOwnProperty.call(message, 'max_seq_size')) {
            throw new Error("Excepted 'max_seq_size'.");
        }
        if (!Object.prototype.hasOwnProperty.call(message, 'in_batch_broadcast')) {
            throw new Error("Excepted 'in_batch_broadcast'.");
        }
        return message;
    }

    static decodeText(reader) {
        const message = new caffe2.AOTConfig();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "max_batch_size":
                    message.max_batch_size = reader.int64();
                    break;
                case "max_seq_size":
                    message.max_seq_size = reader.int64();
                    break;
                case "in_batch_broadcast":
                    message.in_batch_broadcast = reader.bool();
                    break;
                case "onnxifi_blacklist_ops":
                    message.onnxifi_blacklist_ops = reader.string();
                    break;
                case "onnxifi_min_ops":
                    message.onnxifi_min_ops = reader.int32();
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        if (!Object.prototype.hasOwnProperty.call(message, "max_batch_size")) {
            throw new Error("Excepted 'max_batch_size'.");
        }
        if (!Object.prototype.hasOwnProperty.call(message, "max_seq_size")) {
            throw new Error("Excepted 'max_seq_size'.");
        }
        if (!Object.prototype.hasOwnProperty.call(message, "in_batch_broadcast")) {
            throw new Error("Excepted 'in_batch_broadcast'.");
        }
        return message;
    }
};

caffe2.AOTConfig.prototype.max_batch_size = 0n;
caffe2.AOTConfig.prototype.max_seq_size = 0n;
caffe2.AOTConfig.prototype.in_batch_broadcast = false;
caffe2.AOTConfig.prototype.onnxifi_blacklist_ops = "";
caffe2.AOTConfig.prototype.onnxifi_min_ops = 0;

caffe2.Argument = class Argument {

    constructor() {
        this.floats = [];
        this.ints = [];
        this.strings = [];
        this.tensors = [];
        this.nets = [];
        this.qtensors = [];
    }

    static decode(reader, length) {
        const message = new caffe2.Argument();
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
                    message.s = reader.bytes();
                    break;
                case 10:
                    message.t = caffe2.TensorProto.decode(reader, reader.uint32());
                    break;
                case 8:
                    message.n = caffe2.NetDef.decode(reader, reader.uint32());
                    break;
                case 5:
                    message.floats = reader.floats(message.floats, tag);
                    break;
                case 6:
                    message.ints = reader.array(message.ints, () => reader.int64(), tag);
                    break;
                case 7:
                    message.strings.push(reader.bytes());
                    break;
                case 11:
                    message.tensors.push(caffe2.TensorProto.decode(reader, reader.uint32()));
                    break;
                case 9:
                    message.nets.push(caffe2.NetDef.decode(reader, reader.uint32()));
                    break;
                case 12:
                    message.qtensors.push(caffe2.QTensorProto.decode(reader, reader.uint32()));
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }

    static decodeText(reader) {
        const message = new caffe2.Argument();
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
                case "s":
                    message.s = reader.bytes();
                    break;
                case "t":
                    message.t = caffe2.TensorProto.decodeText(reader);
                    break;
                case "n":
                    message.n = caffe2.NetDef.decodeText(reader);
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
                    message.tensors.push(caffe2.TensorProto.decodeText(reader));
                    break;
                case "nets":
                    message.nets.push(caffe2.NetDef.decodeText(reader));
                    break;
                case "qtensors":
                    message.qtensors.push(caffe2.QTensorProto.decodeText(reader));
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

caffe2.Argument.prototype.name = "";
caffe2.Argument.prototype.f = 0;
caffe2.Argument.prototype.i = 0n;
caffe2.Argument.prototype.s = new Uint8Array([]);
caffe2.Argument.prototype.t = null;
caffe2.Argument.prototype.n = null;

caffe2.DeviceTypeProto = {
    "PROTO_CPU": 0,
    "PROTO_CUDA": 1,
    "PROTO_MKLDNN": 2,
    "PROTO_OPENGL": 3,
    "PROTO_OPENCL": 4,
    "PROTO_IDEEP": 5,
    "PROTO_HIP": 6,
    "PROTO_FPGA": 7,
    "PROTO_MAIA": 8,
    "PROTO_XLA": 9,
    "PROTO_MPS": 10,
    "PROTO_COMPILE_TIME_MAX_DEVICE_TYPES": 11
};

caffe2.DeviceOption = class DeviceOption {

    constructor() {
        this.extra_info = [];
    }

    static decode(reader, length) {
        const message = new caffe2.DeviceOption();
        const end = length === undefined ? reader.length : reader.position + length;
        while (reader.position < end) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.device_type = reader.int32();
                    break;
                case 2:
                    message.device_id = reader.int32();
                    break;
                case 3:
                    message.random_seed = reader.uint32();
                    break;
                case 4:
                    message.node_name = reader.string();
                    break;
                case 5:
                    message.numa_node_id = reader.int32();
                    break;
                case 6:
                    message.extra_info.push(reader.string());
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }

    static decodeText(reader) {
        const message = new caffe2.DeviceOption();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "device_type":
                    message.device_type = reader.int32();
                    break;
                case "device_id":
                    message.device_id = reader.int32();
                    break;
                case "random_seed":
                    message.random_seed = reader.uint32();
                    break;
                case "node_name":
                    message.node_name = reader.string();
                    break;
                case "numa_node_id":
                    message.numa_node_id = reader.int32();
                    break;
                case "extra_info":
                    reader.array(message.extra_info, () => reader.string());
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

caffe2.DeviceOption.prototype.device_type = 0;
caffe2.DeviceOption.prototype.device_id = 0;
caffe2.DeviceOption.prototype.random_seed = 0;
caffe2.DeviceOption.prototype.node_name = "";
caffe2.DeviceOption.prototype.numa_node_id = 0;

caffe2.OperatorDef = class OperatorDef {

    constructor() {
        this.input = [];
        this.output = [];
        this.arg = [];
        this.control_input = [];
    }

    static decode(reader, length) {
        const message = new caffe2.OperatorDef();
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
                    message.type = reader.string();
                    break;
                case 5:
                    message.arg.push(caffe2.Argument.decode(reader, reader.uint32()));
                    break;
                case 6:
                    message.device_option = caffe2.DeviceOption.decode(reader, reader.uint32());
                    break;
                case 7:
                    message.engine = reader.string();
                    break;
                case 8:
                    message.control_input.push(reader.string());
                    break;
                case 9:
                    message.is_gradient_op = reader.bool();
                    break;
                case 10:
                    message.debug_info = reader.string();
                    break;
                case 11:
                    message.domain = reader.string();
                    break;
                case 12:
                    message.op_version = reader.int64();
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }

    static decodeText(reader) {
        const message = new caffe2.OperatorDef();
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
                case "type":
                    message.type = reader.string();
                    break;
                case "arg":
                    message.arg.push(caffe2.Argument.decodeText(reader));
                    break;
                case "device_option":
                    message.device_option = caffe2.DeviceOption.decodeText(reader);
                    break;
                case "engine":
                    message.engine = reader.string();
                    break;
                case "control_input":
                    reader.array(message.control_input, () => reader.string());
                    break;
                case "is_gradient_op":
                    message.is_gradient_op = reader.bool();
                    break;
                case "debug_info":
                    message.debug_info = reader.string();
                    break;
                case "domain":
                    message.domain = reader.string();
                    break;
                case "op_version":
                    message.op_version = reader.int64();
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

caffe2.OperatorDef.prototype.name = "";
caffe2.OperatorDef.prototype.type = "";
caffe2.OperatorDef.prototype.device_option = null;
caffe2.OperatorDef.prototype.engine = "";
caffe2.OperatorDef.prototype.is_gradient_op = false;
caffe2.OperatorDef.prototype.debug_info = "";
caffe2.OperatorDef.prototype.domain = "";
caffe2.OperatorDef.prototype.op_version = 0n;

caffe2.MapFieldEntry = class MapFieldEntry {

    static decode(reader, length) {
        const message = new caffe2.MapFieldEntry();
        const end = length === undefined ? reader.length : reader.position + length;
        while (reader.position < end) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.key = reader.string();
                    break;
                case 2:
                    message.val = reader.string();
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        if (!Object.prototype.hasOwnProperty.call(message, 'key')) {
            throw new Error("Excepted 'key'.");
        }
        if (!Object.prototype.hasOwnProperty.call(message, 'val')) {
            throw new Error("Excepted 'val'.");
        }
        return message;
    }

    static decodeText(reader) {
        const message = new caffe2.MapFieldEntry();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "key":
                    message.key = reader.string();
                    break;
                case "val":
                    message.val = reader.string();
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        if (!Object.prototype.hasOwnProperty.call(message, "key")) {
            throw new Error("Excepted 'key'.");
        }
        if (!Object.prototype.hasOwnProperty.call(message, "val")) {
            throw new Error("Excepted 'val'.");
        }
        return message;
    }
};

caffe2.MapFieldEntry.prototype.key = "";
caffe2.MapFieldEntry.prototype.val = "";

caffe2.BackendOptions = class BackendOptions {

    constructor() {
        this.option = [];
    }

    static decode(reader, length) {
        const message = new caffe2.BackendOptions();
        const end = length === undefined ? reader.length : reader.position + length;
        while (reader.position < end) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.backend_name = reader.string();
                    break;
                case 2:
                    message.option.push(caffe2.MapFieldEntry.decode(reader, reader.uint32()));
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        if (!Object.prototype.hasOwnProperty.call(message, 'backend_name')) {
            throw new Error("Excepted 'backend_name'.");
        }
        return message;
    }

    static decodeText(reader) {
        const message = new caffe2.BackendOptions();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "backend_name":
                    message.backend_name = reader.string();
                    break;
                case "option":
                    message.option.push(caffe2.MapFieldEntry.decodeText(reader));
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        if (!Object.prototype.hasOwnProperty.call(message, "backend_name")) {
            throw new Error("Excepted 'backend_name'.");
        }
        return message;
    }
};

caffe2.BackendOptions.prototype.backend_name = "";

caffe2.PartitionInfo = class PartitionInfo {

    constructor() {
        this.device_id = [];
        this.backend_options = [];
    }

    static decode(reader, length) {
        const message = new caffe2.PartitionInfo();
        const end = length === undefined ? reader.length : reader.position + length;
        while (reader.position < end) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.name = reader.string();
                    break;
                case 2:
                    message.device_id = reader.array(message.device_id, () => reader.int32(), tag);
                    break;
                case 3:
                    message.extra_info = reader.string();
                    break;
                case 4:
                    message.backend_options.push(caffe2.BackendOptions.decode(reader, reader.uint32()));
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        if (!Object.prototype.hasOwnProperty.call(message, 'name')) {
            throw new Error("Excepted 'name'.");
        }
        return message;
    }

    static decodeText(reader) {
        const message = new caffe2.PartitionInfo();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "name":
                    message.name = reader.string();
                    break;
                case "device_id":
                    reader.array(message.device_id, () => reader.int32());
                    break;
                case "extra_info":
                    message.extra_info = reader.string();
                    break;
                case "backend_options":
                    message.backend_options.push(caffe2.BackendOptions.decodeText(reader));
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        if (!Object.prototype.hasOwnProperty.call(message, "name")) {
            throw new Error("Excepted 'name'.");
        }
        return message;
    }
};

caffe2.PartitionInfo.prototype.name = "";
caffe2.PartitionInfo.prototype.extra_info = "";

caffe2.NetDef = class NetDef {

    constructor() {
        this.op = [];
        this.arg = [];
        this.external_input = [];
        this.external_output = [];
        this.partition_info = [];
    }

    static decode(reader, length) {
        const message = new caffe2.NetDef();
        const end = length === undefined ? reader.length : reader.position + length;
        while (reader.position < end) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.name = reader.string();
                    break;
                case 2:
                    message.op.push(caffe2.OperatorDef.decode(reader, reader.uint32()));
                    break;
                case 3:
                    message.type = reader.string();
                    break;
                case 4:
                    message.num_workers = reader.int32();
                    break;
                case 5:
                    message.device_option = caffe2.DeviceOption.decode(reader, reader.uint32());
                    break;
                case 6:
                    message.arg.push(caffe2.Argument.decode(reader, reader.uint32()));
                    break;
                case 7:
                    message.external_input.push(reader.string());
                    break;
                case 8:
                    message.external_output.push(reader.string());
                    break;
                case 9:
                    message.partition_info.push(caffe2.PartitionInfo.decode(reader, reader.uint32()));
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }

    static decodeText(reader) {
        const message = new caffe2.NetDef();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "name":
                    message.name = reader.string();
                    break;
                case "op":
                    message.op.push(caffe2.OperatorDef.decodeText(reader));
                    break;
                case "type":
                    message.type = reader.string();
                    break;
                case "num_workers":
                    message.num_workers = reader.int32();
                    break;
                case "device_option":
                    message.device_option = caffe2.DeviceOption.decodeText(reader);
                    break;
                case "arg":
                    message.arg.push(caffe2.Argument.decodeText(reader));
                    break;
                case "external_input":
                    reader.array(message.external_input, () => reader.string());
                    break;
                case "external_output":
                    reader.array(message.external_output, () => reader.string());
                    break;
                case "partition_info":
                    message.partition_info.push(caffe2.PartitionInfo.decodeText(reader));
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

caffe2.NetDef.prototype.name = "";
caffe2.NetDef.prototype.type = "";
caffe2.NetDef.prototype.num_workers = 0;
caffe2.NetDef.prototype.device_option = null;

caffe2.ExecutionStep = class ExecutionStep {

    constructor() {
        this.substep = [];
        this.network = [];
    }

    static decode(reader, length) {
        const message = new caffe2.ExecutionStep();
        const end = length === undefined ? reader.length : reader.position + length;
        while (reader.position < end) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.name = reader.string();
                    break;
                case 2:
                    message.substep.push(caffe2.ExecutionStep.decode(reader, reader.uint32()));
                    break;
                case 3:
                    message.network.push(reader.string());
                    break;
                case 4:
                    message.num_iter = reader.int64();
                    break;
                case 5:
                    message.criteria_network = reader.string();
                    break;
                case 7:
                    message.report_net = reader.string();
                    break;
                case 8:
                    message.report_interval = reader.int32();
                    break;
                case 11:
                    message.run_every_ms = reader.int64();
                    break;
                case 6:
                    message.concurrent_substeps = reader.bool();
                    break;
                case 9:
                    message.should_stop_blob = reader.string();
                    break;
                case 10:
                    message.only_once = reader.bool();
                    break;
                case 12:
                    message.create_workspace = reader.bool();
                    break;
                case 13:
                    message.num_concurrent_instances = reader.int32();
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }

    static decodeText(reader) {
        const message = new caffe2.ExecutionStep();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "name":
                    message.name = reader.string();
                    break;
                case "substep":
                    message.substep.push(caffe2.ExecutionStep.decodeText(reader));
                    break;
                case "network":
                    reader.array(message.network, () => reader.string());
                    break;
                case "num_iter":
                    message.num_iter = reader.int64();
                    break;
                case "criteria_network":
                    message.criteria_network = reader.string();
                    break;
                case "report_net":
                    message.report_net = reader.string();
                    break;
                case "report_interval":
                    message.report_interval = reader.int32();
                    break;
                case "run_every_ms":
                    message.run_every_ms = reader.int64();
                    break;
                case "concurrent_substeps":
                    message.concurrent_substeps = reader.bool();
                    break;
                case "should_stop_blob":
                    message.should_stop_blob = reader.string();
                    break;
                case "only_once":
                    message.only_once = reader.bool();
                    break;
                case "create_workspace":
                    message.create_workspace = reader.bool();
                    break;
                case "num_concurrent_instances":
                    message.num_concurrent_instances = reader.int32();
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

caffe2.ExecutionStep.prototype.name = "";
caffe2.ExecutionStep.prototype.num_iter = 0n;
caffe2.ExecutionStep.prototype.criteria_network = "";
caffe2.ExecutionStep.prototype.report_net = "";
caffe2.ExecutionStep.prototype.report_interval = 0;
caffe2.ExecutionStep.prototype.run_every_ms = 0n;
caffe2.ExecutionStep.prototype.concurrent_substeps = false;
caffe2.ExecutionStep.prototype.should_stop_blob = "";
caffe2.ExecutionStep.prototype.only_once = false;
caffe2.ExecutionStep.prototype.create_workspace = false;
caffe2.ExecutionStep.prototype.num_concurrent_instances = 0;

caffe2.PlanDef = class PlanDef {

    constructor() {
        this.network = [];
        this.execution_step = [];
    }

    static decode(reader, length) {
        const message = new caffe2.PlanDef();
        const end = length === undefined ? reader.length : reader.position + length;
        while (reader.position < end) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.name = reader.string();
                    break;
                case 2:
                    message.network.push(caffe2.NetDef.decode(reader, reader.uint32()));
                    break;
                case 3:
                    message.execution_step.push(caffe2.ExecutionStep.decode(reader, reader.uint32()));
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }

    static decodeText(reader) {
        const message = new caffe2.PlanDef();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "name":
                    message.name = reader.string();
                    break;
                case "network":
                    message.network.push(caffe2.NetDef.decodeText(reader));
                    break;
                case "execution_step":
                    message.execution_step.push(caffe2.ExecutionStep.decodeText(reader));
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

caffe2.PlanDef.prototype.name = "";

caffe2.BlobProto = class BlobProto {

    static decode(reader, length) {
        const message = new caffe2.BlobProto();
        const end = length === undefined ? reader.length : reader.position + length;
        while (reader.position < end) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.name = reader.string();
                    break;
                case 2:
                    message.type = reader.string();
                    break;
                case 3:
                    message.tensor = caffe2.TensorProto.decode(reader, reader.uint32());
                    break;
                case 4:
                    message.content = reader.bytes();
                    break;
                case 5:
                    message.qtensor = caffe2.QTensorProto.decode(reader, reader.uint32());
                    break;
                case 6:
                    message.content_num_chunks = reader.int32();
                    break;
                case 7:
                    message.content_chunk_id = reader.int32();
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }

    static decodeText(reader) {
        const message = new caffe2.BlobProto();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "name":
                    message.name = reader.string();
                    break;
                case "type":
                    message.type = reader.string();
                    break;
                case "tensor":
                    message.tensor = caffe2.TensorProto.decodeText(reader);
                    break;
                case "content":
                    message.content = reader.bytes();
                    break;
                case "qtensor":
                    message.qtensor = caffe2.QTensorProto.decodeText(reader);
                    break;
                case "content_num_chunks":
                    message.content_num_chunks = reader.int32();
                    break;
                case "content_chunk_id":
                    message.content_chunk_id = reader.int32();
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

caffe2.BlobProto.prototype.name = "";
caffe2.BlobProto.prototype.type = "";
caffe2.BlobProto.prototype.tensor = null;
caffe2.BlobProto.prototype.content = new Uint8Array([]);
caffe2.BlobProto.prototype.qtensor = null;
caffe2.BlobProto.prototype.content_num_chunks = 0;
caffe2.BlobProto.prototype.content_chunk_id = 0;

caffe2.DBReaderProto = class DBReaderProto {

    static decode(reader, length) {
        const message = new caffe2.DBReaderProto();
        const end = length === undefined ? reader.length : reader.position + length;
        while (reader.position < end) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.name = reader.string();
                    break;
                case 2:
                    message.source = reader.string();
                    break;
                case 3:
                    message.db_type = reader.string();
                    break;
                case 4:
                    message.key = reader.string();
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }

    static decodeText(reader) {
        const message = new caffe2.DBReaderProto();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "name":
                    message.name = reader.string();
                    break;
                case "source":
                    message.source = reader.string();
                    break;
                case "db_type":
                    message.db_type = reader.string();
                    break;
                case "key":
                    message.key = reader.string();
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

caffe2.DBReaderProto.prototype.name = "";
caffe2.DBReaderProto.prototype.source = "";
caffe2.DBReaderProto.prototype.db_type = "";
caffe2.DBReaderProto.prototype.key = "";

caffe2.BlobSerializationOptions = class BlobSerializationOptions {

    static decode(reader, length) {
        const message = new caffe2.BlobSerializationOptions();
        const end = length === undefined ? reader.length : reader.position + length;
        while (reader.position < end) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.blob_name_regex = reader.string();
                    break;
                case 2:
                    message.chunk_size = reader.int64();
                    break;
                case 3:
                    message.float_format = reader.int32();
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }

    static decodeText(reader) {
        const message = new caffe2.BlobSerializationOptions();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "blob_name_regex":
                    message.blob_name_regex = reader.string();
                    break;
                case "chunk_size":
                    message.chunk_size = reader.int64();
                    break;
                case "float_format":
                    message.float_format = reader.enum(caffe2.BlobSerializationOptions.FloatFormat);
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

caffe2.BlobSerializationOptions.prototype.blob_name_regex = "";
caffe2.BlobSerializationOptions.prototype.chunk_size = 0n;
caffe2.BlobSerializationOptions.prototype.float_format = 0;

caffe2.BlobSerializationOptions.FloatFormat = {
    "FLOAT_DEFAULT": 0,
    "FLOAT_PROTOBUF": 1,
    "FLOAT_BFLOAT16": 2
};

caffe2.SerializationOptions = class SerializationOptions {

    constructor() {
        this.options = [];
    }

    static decode(reader, length) {
        const message = new caffe2.SerializationOptions();
        const end = length === undefined ? reader.length : reader.position + length;
        while (reader.position < end) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.options.push(caffe2.BlobSerializationOptions.decode(reader, reader.uint32()));
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }

    static decodeText(reader) {
        const message = new caffe2.SerializationOptions();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "options":
                    message.options.push(caffe2.BlobSerializationOptions.decodeText(reader));
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};
