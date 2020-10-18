var $root = protobuf.get('paddle');

$root.paddle = {};

$root.paddle.framework = {};

$root.paddle.framework.proto = {};

$root.paddle.framework.proto.Version = class Version {

    constructor() {
    }

    static decode(reader, length) {
        const message = new $root.paddle.framework.proto.Version();
        const end = reader.next(length);
        while (reader.end(end)) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.version = reader.int64();
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }
};

$root.paddle.framework.proto.Version.prototype.version = protobuf.Int64.create(0);

$root.paddle.framework.proto.AttrType = {
    "INT": 0,
    "FLOAT": 1,
    "STRING": 2,
    "INTS": 3,
    "FLOATS": 4,
    "STRINGS": 5,
    "BOOLEAN": 6,
    "BOOLEANS": 7,
    "BLOCK": 8,
    "LONG": 9,
    "BLOCKS": 10,
    "LONGS": 11
};

$root.paddle.framework.proto.OpDesc = class OpDesc {

    constructor() {
        this.inputs = [];
        this.outputs = [];
        this.attrs = [];
    }

    static decode(reader, length) {
        const message = new $root.paddle.framework.proto.OpDesc();
        const end = reader.next(length);
        while (reader.end(end)) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 3:
                    message.type = reader.string();
                    break;
                case 1:
                    message.inputs.push($root.paddle.framework.proto.OpDesc.Var.decode(reader, reader.uint32()));
                    break;
                case 2:
                    message.outputs.push($root.paddle.framework.proto.OpDesc.Var.decode(reader, reader.uint32()));
                    break;
                case 4:
                    message.attrs.push($root.paddle.framework.proto.OpDesc.Attr.decode(reader, reader.uint32()));
                    break;
                case 5:
                    message.is_target = reader.bool();
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        if (!Object.prototype.hasOwnProperty.call(message, 'type')) {
            throw new protobuf.Error("Excepted 'type'.");
        }
        return message;
    }
};

$root.paddle.framework.proto.OpDesc.prototype.type = "";
$root.paddle.framework.proto.OpDesc.prototype.is_target = false;

$root.paddle.framework.proto.OpDesc.Attr = class Attr {

    constructor() {
        this.ints = [];
        this.floats = [];
        this.strings = [];
        this.bools = [];
        this.blocks_idx = [];
        this.longs = [];
    }

    static decode(reader, length) {
        const message = new $root.paddle.framework.proto.OpDesc.Attr();
        const end = reader.next(length);
        while (reader.end(end)) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.name = reader.string();
                    break;
                case 2:
                    message.type = reader.int32();
                    break;
                case 3:
                    message.i = reader.int32();
                    break;
                case 4:
                    message.f = reader.float();
                    break;
                case 5:
                    message.s = reader.string();
                    break;
                case 6:
                    message.ints = reader.array(message.ints, () => reader.int32(), tag);
                    break;
                case 7:
                    message.floats = reader.floats(message.floats, tag);
                    break;
                case 8:
                    message.strings.push(reader.string());
                    break;
                case 10:
                    message.b = reader.bool();
                    break;
                case 11:
                    message.bools = reader.array(message.bools, () => reader.bool(), tag);
                    break;
                case 12:
                    message.block_idx = reader.int32();
                    break;
                case 13:
                    message.l = reader.int64();
                    break;
                case 14:
                    message.blocks_idx = reader.array(message.blocks_idx, () => reader.int32(), tag);
                    break;
                case 15:
                    message.longs = reader.array(message.longs, () => reader.int64(), tag);
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        if (!Object.prototype.hasOwnProperty.call(message, 'name')) {
            throw new protobuf.Error("Excepted 'name'.");
        }
        if (!Object.prototype.hasOwnProperty.call(message, 'type')) {
            throw new protobuf.Error("Excepted 'type'.");
        }
        return message;
    }
};

$root.paddle.framework.proto.OpDesc.Attr.prototype.name = "";
$root.paddle.framework.proto.OpDesc.Attr.prototype.type = 0;
$root.paddle.framework.proto.OpDesc.Attr.prototype.i = 0;
$root.paddle.framework.proto.OpDesc.Attr.prototype.f = 0;
$root.paddle.framework.proto.OpDesc.Attr.prototype.s = "";
$root.paddle.framework.proto.OpDesc.Attr.prototype.b = false;
$root.paddle.framework.proto.OpDesc.Attr.prototype.block_idx = 0;
$root.paddle.framework.proto.OpDesc.Attr.prototype.l = protobuf.Int64.create(0);

$root.paddle.framework.proto.OpDesc.Var = class Var {

    constructor() {
        this["arguments"] = [];
    }

    static decode(reader, length) {
        const message = new $root.paddle.framework.proto.OpDesc.Var();
        const end = reader.next(length);
        while (reader.end(end)) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.parameter = reader.string();
                    break;
                case 2:
                    message["arguments"].push(reader.string());
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        if (!Object.prototype.hasOwnProperty.call(message, 'parameter')) {
            throw new protobuf.Error("Excepted 'parameter'.");
        }
        return message;
    }
};

$root.paddle.framework.proto.OpDesc.Var.prototype.parameter = "";

$root.paddle.framework.proto.OpProto = class OpProto {

    constructor() {
        this.inputs = [];
        this.outputs = [];
        this.attrs = [];
    }

    static decode(reader, length) {
        const message = new $root.paddle.framework.proto.OpProto();
        const end = reader.next(length);
        while (reader.end(end)) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.type = reader.string();
                    break;
                case 2:
                    message.inputs.push($root.paddle.framework.proto.OpProto.Var.decode(reader, reader.uint32()));
                    break;
                case 3:
                    message.outputs.push($root.paddle.framework.proto.OpProto.Var.decode(reader, reader.uint32()));
                    break;
                case 4:
                    message.attrs.push($root.paddle.framework.proto.OpProto.Attr.decode(reader, reader.uint32()));
                    break;
                case 5:
                    message.comment = reader.string();
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        if (!Object.prototype.hasOwnProperty.call(message, 'type')) {
            throw new protobuf.Error("Excepted 'type'.");
        }
        if (!Object.prototype.hasOwnProperty.call(message, 'comment')) {
            throw new protobuf.Error("Excepted 'comment'.");
        }
        return message;
    }
};

$root.paddle.framework.proto.OpProto.prototype.type = "";
$root.paddle.framework.proto.OpProto.prototype.comment = "";

$root.paddle.framework.proto.OpProto.Var = class Var {

    constructor() {
    }

    static decode(reader, length) {
        const message = new $root.paddle.framework.proto.OpProto.Var();
        const end = reader.next(length);
        while (reader.end(end)) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.name = reader.string();
                    break;
                case 2:
                    message.comment = reader.string();
                    break;
                case 3:
                    message.duplicable = reader.bool();
                    break;
                case 4:
                    message.intermediate = reader.bool();
                    break;
                case 5:
                    message.dispensable = reader.bool();
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        if (!Object.prototype.hasOwnProperty.call(message, 'name')) {
            throw new protobuf.Error("Excepted 'name'.");
        }
        if (!Object.prototype.hasOwnProperty.call(message, 'comment')) {
            throw new protobuf.Error("Excepted 'comment'.");
        }
        return message;
    }
};

$root.paddle.framework.proto.OpProto.Var.prototype.name = "";
$root.paddle.framework.proto.OpProto.Var.prototype.comment = "";
$root.paddle.framework.proto.OpProto.Var.prototype.duplicable = false;
$root.paddle.framework.proto.OpProto.Var.prototype.intermediate = false;
$root.paddle.framework.proto.OpProto.Var.prototype.dispensable = false;

$root.paddle.framework.proto.OpProto.Attr = class Attr {

    constructor() {
    }

    static decode(reader, length) {
        const message = new $root.paddle.framework.proto.OpProto.Attr();
        const end = reader.next(length);
        while (reader.end(end)) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.name = reader.string();
                    break;
                case 2:
                    message.type = reader.int32();
                    break;
                case 3:
                    message.comment = reader.string();
                    break;
                case 4:
                    message.generated = reader.bool();
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        if (!Object.prototype.hasOwnProperty.call(message, 'name')) {
            throw new protobuf.Error("Excepted 'name'.");
        }
        if (!Object.prototype.hasOwnProperty.call(message, 'type')) {
            throw new protobuf.Error("Excepted 'type'.");
        }
        if (!Object.prototype.hasOwnProperty.call(message, 'comment')) {
            throw new protobuf.Error("Excepted 'comment'.");
        }
        return message;
    }
};

$root.paddle.framework.proto.OpProto.Attr.prototype.name = "";
$root.paddle.framework.proto.OpProto.Attr.prototype.type = 0;
$root.paddle.framework.proto.OpProto.Attr.prototype.comment = "";
$root.paddle.framework.proto.OpProto.Attr.prototype.generated = false;

$root.paddle.framework.proto.VarType = class VarType {

    constructor() {
    }

    static decode(reader, length) {
        const message = new $root.paddle.framework.proto.VarType();
        const end = reader.next(length);
        while (reader.end(end)) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.type = reader.int32();
                    break;
                case 2:
                    message.selected_rows = $root.paddle.framework.proto.VarType.TensorDesc.decode(reader, reader.uint32());
                    break;
                case 3:
                    message.lod_tensor = $root.paddle.framework.proto.VarType.LoDTensorDesc.decode(reader, reader.uint32());
                    break;
                case 4:
                    message.tensor_array = $root.paddle.framework.proto.VarType.LoDTensorArrayDesc.decode(reader, reader.uint32());
                    break;
                case 5:
                    message.reader = $root.paddle.framework.proto.VarType.ReaderDesc.decode(reader, reader.uint32());
                    break;
                case 7:
                    message.tuple = $root.paddle.framework.proto.VarType.Tuple.decode(reader, reader.uint32());
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        if (!Object.prototype.hasOwnProperty.call(message, 'type')) {
            throw new protobuf.Error("Excepted 'type'.");
        }
        return message;
    }
};

$root.paddle.framework.proto.VarType.prototype.type = 0;
$root.paddle.framework.proto.VarType.prototype.selected_rows = null;
$root.paddle.framework.proto.VarType.prototype.lod_tensor = null;
$root.paddle.framework.proto.VarType.prototype.tensor_array = null;
$root.paddle.framework.proto.VarType.prototype.reader = null;
$root.paddle.framework.proto.VarType.prototype.tuple = null;

$root.paddle.framework.proto.VarType.Type = {
    "BOOL": 0,
    "INT16": 1,
    "INT32": 2,
    "INT64": 3,
    "FP16": 4,
    "FP32": 5,
    "FP64": 6,
    "SIZE_T": 19,
    "UINT8": 20,
    "INT8": 21,
    "BF16": 22,
    "LOD_TENSOR": 7,
    "SELECTED_ROWS": 8,
    "FEED_MINIBATCH": 9,
    "FETCH_LIST": 10,
    "STEP_SCOPES": 11,
    "LOD_RANK_TABLE": 12,
    "LOD_TENSOR_ARRAY": 13,
    "PLACE_LIST": 14,
    "READER": 15,
    "RAW": 17,
    "TUPLE": 18
};

$root.paddle.framework.proto.VarType.TensorDesc = class TensorDesc {

    constructor() {
        this.dims = [];
    }

    static decode(reader, length) {
        const message = new $root.paddle.framework.proto.VarType.TensorDesc();
        const end = reader.next(length);
        while (reader.end(end)) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.data_type = reader.int32();
                    break;
                case 2:
                    message.dims = reader.array(message.dims, () => reader.int64(), tag);
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        if (!Object.prototype.hasOwnProperty.call(message, 'data_type')) {
            throw new protobuf.Error("Excepted 'data_type'.");
        }
        return message;
    }
};

$root.paddle.framework.proto.VarType.TensorDesc.prototype.data_type = 0;

$root.paddle.framework.proto.VarType.LoDTensorDesc = class LoDTensorDesc {

    constructor() {
    }

    static decode(reader, length) {
        const message = new $root.paddle.framework.proto.VarType.LoDTensorDesc();
        const end = reader.next(length);
        while (reader.end(end)) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.tensor = $root.paddle.framework.proto.VarType.TensorDesc.decode(reader, reader.uint32());
                    break;
                case 2:
                    message.lod_level = reader.int32();
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        if (!Object.prototype.hasOwnProperty.call(message, 'tensor')) {
            throw new protobuf.Error("Excepted 'tensor'.");
        }
        return message;
    }
};

$root.paddle.framework.proto.VarType.LoDTensorDesc.prototype.tensor = null;
$root.paddle.framework.proto.VarType.LoDTensorDesc.prototype.lod_level = 0;

$root.paddle.framework.proto.VarType.LoDTensorArrayDesc = class LoDTensorArrayDesc {

    constructor() {
    }

    static decode(reader, length) {
        const message = new $root.paddle.framework.proto.VarType.LoDTensorArrayDesc();
        const end = reader.next(length);
        while (reader.end(end)) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.tensor = $root.paddle.framework.proto.VarType.TensorDesc.decode(reader, reader.uint32());
                    break;
                case 2:
                    message.lod_level = reader.int32();
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        if (!Object.prototype.hasOwnProperty.call(message, 'tensor')) {
            throw new protobuf.Error("Excepted 'tensor'.");
        }
        return message;
    }
};

$root.paddle.framework.proto.VarType.LoDTensorArrayDesc.prototype.tensor = null;
$root.paddle.framework.proto.VarType.LoDTensorArrayDesc.prototype.lod_level = 0;

$root.paddle.framework.proto.VarType.ReaderDesc = class ReaderDesc {

    constructor() {
        this.lod_tensor = [];
    }

    static decode(reader, length) {
        const message = new $root.paddle.framework.proto.VarType.ReaderDesc();
        const end = reader.next(length);
        while (reader.end(end)) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.lod_tensor.push($root.paddle.framework.proto.VarType.LoDTensorDesc.decode(reader, reader.uint32()));
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }
};

$root.paddle.framework.proto.VarType.Tuple = class Tuple {

    constructor() {
        this.element_type = [];
    }

    static decode(reader, length) {
        const message = new $root.paddle.framework.proto.VarType.Tuple();
        const end = reader.next(length);
        while (reader.end(end)) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.element_type = reader.array(message.element_type, () => reader.int32(), tag);
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }
};

$root.paddle.framework.proto.VarDesc = class VarDesc {

    constructor() {
    }

    static decode(reader, length) {
        const message = new $root.paddle.framework.proto.VarDesc();
        const end = reader.next(length);
        while (reader.end(end)) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.name = reader.string();
                    break;
                case 2:
                    message.type = $root.paddle.framework.proto.VarType.decode(reader, reader.uint32());
                    break;
                case 3:
                    message.persistable = reader.bool();
                    break;
                case 4:
                    message.need_check_feed = reader.bool();
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        if (!Object.prototype.hasOwnProperty.call(message, 'name')) {
            throw new protobuf.Error("Excepted 'name'.");
        }
        if (!Object.prototype.hasOwnProperty.call(message, 'type')) {
            throw new protobuf.Error("Excepted 'type'.");
        }
        return message;
    }
};

$root.paddle.framework.proto.VarDesc.prototype.name = "";
$root.paddle.framework.proto.VarDesc.prototype.type = null;
$root.paddle.framework.proto.VarDesc.prototype.persistable = false;
$root.paddle.framework.proto.VarDesc.prototype.need_check_feed = false;

$root.paddle.framework.proto.BlockDesc = class BlockDesc {

    constructor() {
        this.vars = [];
        this.ops = [];
    }

    static decode(reader, length) {
        const message = new $root.paddle.framework.proto.BlockDesc();
        const end = reader.next(length);
        while (reader.end(end)) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.idx = reader.int32();
                    break;
                case 2:
                    message.parent_idx = reader.int32();
                    break;
                case 3:
                    message.vars.push($root.paddle.framework.proto.VarDesc.decode(reader, reader.uint32()));
                    break;
                case 4:
                    message.ops.push($root.paddle.framework.proto.OpDesc.decode(reader, reader.uint32()));
                    break;
                case 5:
                    message.forward_block_idx = reader.int32();
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        if (!Object.prototype.hasOwnProperty.call(message, 'idx')) {
            throw new protobuf.Error("Excepted 'idx'.");
        }
        if (!Object.prototype.hasOwnProperty.call(message, 'parent_idx')) {
            throw new protobuf.Error("Excepted 'parent_idx'.");
        }
        return message;
    }
};

$root.paddle.framework.proto.BlockDesc.prototype.idx = 0;
$root.paddle.framework.proto.BlockDesc.prototype.parent_idx = 0;
$root.paddle.framework.proto.BlockDesc.prototype.forward_block_idx = -1;

$root.paddle.framework.proto.OpVersion = class OpVersion {

    constructor() {
    }

    static decode(reader, length) {
        const message = new $root.paddle.framework.proto.OpVersion();
        const end = reader.next(length);
        while (reader.end(end)) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.version = reader.int32();
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        if (!Object.prototype.hasOwnProperty.call(message, 'version')) {
            throw new protobuf.Error("Excepted 'version'.");
        }
        return message;
    }
};

$root.paddle.framework.proto.OpVersion.prototype.version = 0;

$root.paddle.framework.proto.OpVersionMap = class OpVersionMap {

    constructor() {
        this.pair = [];
    }

    static decode(reader, length) {
        const message = new $root.paddle.framework.proto.OpVersionMap();
        const end = reader.next(length);
        while (reader.end(end)) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.pair.push($root.paddle.framework.proto.OpVersionMap.OpVersionPair.decode(reader, reader.uint32()));
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }
};

$root.paddle.framework.proto.OpVersionMap.OpVersionPair = class OpVersionPair {

    constructor() {
    }

    static decode(reader, length) {
        const message = new $root.paddle.framework.proto.OpVersionMap.OpVersionPair();
        const end = reader.next(length);
        while (reader.end(end)) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.op_name = reader.string();
                    break;
                case 2:
                    message.op_version = $root.paddle.framework.proto.OpVersion.decode(reader, reader.uint32());
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        if (!Object.prototype.hasOwnProperty.call(message, 'op_name')) {
            throw new protobuf.Error("Excepted 'op_name'.");
        }
        if (!Object.prototype.hasOwnProperty.call(message, 'op_version')) {
            throw new protobuf.Error("Excepted 'op_version'.");
        }
        return message;
    }
};

$root.paddle.framework.proto.OpVersionMap.OpVersionPair.prototype.op_name = "";
$root.paddle.framework.proto.OpVersionMap.OpVersionPair.prototype.op_version = null;

$root.paddle.framework.proto.ProgramDesc = class ProgramDesc {

    constructor() {
        this.blocks = [];
    }

    static decode(reader, length) {
        const message = new $root.paddle.framework.proto.ProgramDesc();
        const end = reader.next(length);
        while (reader.end(end)) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.blocks.push($root.paddle.framework.proto.BlockDesc.decode(reader, reader.uint32()));
                    break;
                case 4:
                    message.version = $root.paddle.framework.proto.Version.decode(reader, reader.uint32());
                    break;
                case 5:
                    message.op_version_map = $root.paddle.framework.proto.OpVersionMap.decode(reader, reader.uint32());
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }
};

$root.paddle.framework.proto.ProgramDesc.prototype.version = null;
$root.paddle.framework.proto.ProgramDesc.prototype.op_version_map = null;
