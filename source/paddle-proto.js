var $root = protobuf.get('paddle');

$root.paddle = {};

$root.paddle.framework = {};

$root.paddle.framework.proto = {};

$root.paddle.framework.proto.Version = class Version {

    constructor() {
    }

    static decode(reader, length) {
        const message = new $root.paddle.framework.proto.Version();
        const end = length !== undefined ? reader.position + length : reader.length;
        while (reader.position < end) {
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

    static decodeText(reader) {
        const message = new $root.paddle.framework.proto.Version();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
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
    "LONGS": 11,
    "FLOAT64S": 12,
    "VAR": 13,
    "VARS": 14,
    "FLOAT64": 15,
    "SCALAR": 16,
    "SCALARS": 17
};

$root.paddle.framework.proto.Complex = class Complex {

    constructor() {
    }

    static decode(reader, length) {
        const message = new $root.paddle.framework.proto.Complex();
        const end = length !== undefined ? reader.position + length : reader.length;
        while (reader.position < end) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.r = reader.double();
                    break;
                case 2:
                    message.i = reader.double();
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        if (!Object.prototype.hasOwnProperty.call(message, 'r')) {
            throw new protobuf.Error("Excepted 'r'.");
        }
        if (!Object.prototype.hasOwnProperty.call(message, 'i')) {
            throw new protobuf.Error("Excepted 'i'.");
        }
        return message;
    }

    static decodeText(reader) {
        const message = new $root.paddle.framework.proto.Complex();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "r":
                    message.r = reader.double();
                    break;
                case "i":
                    message.i = reader.double();
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        if (!Object.prototype.hasOwnProperty.call(message, "r")) {
            throw new protobuf.Error("Excepted 'r'.");
        }
        if (!Object.prototype.hasOwnProperty.call(message, "i")) {
            throw new protobuf.Error("Excepted 'i'.");
        }
        return message;
    }
};

$root.paddle.framework.proto.Complex.prototype.r = 0;
$root.paddle.framework.proto.Complex.prototype.i = 0;

$root.paddle.framework.proto.Scalar = class Scalar {

    constructor() {
    }

    static decode(reader, length) {
        const message = new $root.paddle.framework.proto.Scalar();
        const end = length !== undefined ? reader.position + length : reader.length;
        while (reader.position < end) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.type = reader.int32();
                    break;
                case 2:
                    message.b = reader.bool();
                    break;
                case 3:
                    message.i = reader.int64();
                    break;
                case 4:
                    message.r = reader.double();
                    break;
                case 5:
                    message.c = $root.paddle.framework.proto.Complex.decode(reader, reader.uint32());
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

    static decodeText(reader) {
        const message = new $root.paddle.framework.proto.Scalar();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "type":
                    message.type = reader.enum($root.paddle.framework.proto.Scalar.Type);
                    break;
                case "b":
                    message.b = reader.bool();
                    break;
                case "i":
                    message.i = reader.int64();
                    break;
                case "r":
                    message.r = reader.double();
                    break;
                case "c":
                    message.c = $root.paddle.framework.proto.Complex.decodeText(reader);
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        if (!Object.prototype.hasOwnProperty.call(message, "type")) {
            throw new protobuf.Error("Excepted 'type'.");
        }
        return message;
    }
};

$root.paddle.framework.proto.Scalar.prototype.type = 1;
$root.paddle.framework.proto.Scalar.prototype.b = false;
$root.paddle.framework.proto.Scalar.prototype.i = protobuf.Int64.create(0);
$root.paddle.framework.proto.Scalar.prototype.r = 0;
$root.paddle.framework.proto.Scalar.prototype.c = null;

$root.paddle.framework.proto.Scalar.Type = {
    "BOOLEAN": 1,
    "LONG": 2,
    "FLOAT64": 3,
    "COMPLEX128": 4
};

$root.paddle.framework.proto.OpDesc = class OpDesc {

    constructor() {
        this.inputs = [];
        this.outputs = [];
        this.attrs = [];
    }

    static decode(reader, length) {
        const message = new $root.paddle.framework.proto.OpDesc();
        const end = length !== undefined ? reader.position + length : reader.length;
        while (reader.position < end) {
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

    static decodeText(reader) {
        const message = new $root.paddle.framework.proto.OpDesc();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "type":
                    message.type = reader.string();
                    break;
                case "inputs":
                    message.inputs.push($root.paddle.framework.proto.OpDesc.Var.decodeText(reader));
                    break;
                case "outputs":
                    message.outputs.push($root.paddle.framework.proto.OpDesc.Var.decodeText(reader));
                    break;
                case "attrs":
                    message.attrs.push($root.paddle.framework.proto.OpDesc.Attr.decodeText(reader));
                    break;
                case "is_target":
                    message.is_target = reader.bool();
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        if (!Object.prototype.hasOwnProperty.call(message, "type")) {
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
        this.float64s = [];
        this.vars_name = [];
        this.scalars = [];
    }

    static decode(reader, length) {
        const message = new $root.paddle.framework.proto.OpDesc.Attr();
        const end = length !== undefined ? reader.position + length : reader.length;
        while (reader.position < end) {
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
                case 16:
                    message.float64s = reader.doubles(message.float64s, tag);
                    break;
                case 17:
                    message.var_name = reader.string();
                    break;
                case 18:
                    message.vars_name.push(reader.string());
                    break;
                case 19:
                    message.float64 = reader.double();
                    break;
                case 20:
                    message.scalar = $root.paddle.framework.proto.Scalar.decode(reader, reader.uint32());
                    break;
                case 21:
                    message.scalars.push($root.paddle.framework.proto.Scalar.decode(reader, reader.uint32()));
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

    static decodeText(reader) {
        const message = new $root.paddle.framework.proto.OpDesc.Attr();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "name":
                    message.name = reader.string();
                    break;
                case "type":
                    message.type = reader.enum($root.paddle.framework.proto.AttrType);
                    break;
                case "i":
                    message.i = reader.int32();
                    break;
                case "f":
                    message.f = reader.float();
                    break;
                case "s":
                    message.s = reader.string();
                    break;
                case "ints":
                    reader.array(message.ints, () => reader.int32());
                    break;
                case "floats":
                    reader.array(message.floats, () => reader.float());
                    break;
                case "strings":
                    reader.array(message.strings, () => reader.string());
                    break;
                case "b":
                    message.b = reader.bool();
                    break;
                case "bools":
                    reader.array(message.bools, () => reader.bool());
                    break;
                case "block_idx":
                    message.block_idx = reader.int32();
                    break;
                case "l":
                    message.l = reader.int64();
                    break;
                case "blocks_idx":
                    reader.array(message.blocks_idx, () => reader.int32());
                    break;
                case "longs":
                    reader.array(message.longs, () => reader.int64());
                    break;
                case "float64s":
                    reader.array(message.float64s, () => reader.double());
                    break;
                case "var_name":
                    message.var_name = reader.string();
                    break;
                case "vars_name":
                    reader.array(message.vars_name, () => reader.string());
                    break;
                case "float64":
                    message.float64 = reader.double();
                    break;
                case "scalar":
                    message.scalar = $root.paddle.framework.proto.Scalar.decodeText(reader);
                    break;
                case "scalars":
                    message.scalars.push($root.paddle.framework.proto.Scalar.decodeText(reader));
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        if (!Object.prototype.hasOwnProperty.call(message, "name")) {
            throw new protobuf.Error("Excepted 'name'.");
        }
        if (!Object.prototype.hasOwnProperty.call(message, "type")) {
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
$root.paddle.framework.proto.OpDesc.Attr.prototype.var_name = "";
$root.paddle.framework.proto.OpDesc.Attr.prototype.float64 = 0;
$root.paddle.framework.proto.OpDesc.Attr.prototype.scalar = null;

$root.paddle.framework.proto.OpDesc.Var = class Var {

    constructor() {
        this["arguments"] = [];
    }

    static decode(reader, length) {
        const message = new $root.paddle.framework.proto.OpDesc.Var();
        const end = length !== undefined ? reader.position + length : reader.length;
        while (reader.position < end) {
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

    static decodeText(reader) {
        const message = new $root.paddle.framework.proto.OpDesc.Var();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "parameter":
                    message.parameter = reader.string();
                    break;
                case "arguments":
                    reader.array(message["arguments"], () => reader.string());
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        if (!Object.prototype.hasOwnProperty.call(message, "parameter")) {
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
        const end = length !== undefined ? reader.position + length : reader.length;
        while (reader.position < end) {
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

    static decodeText(reader) {
        const message = new $root.paddle.framework.proto.OpProto();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "type":
                    message.type = reader.string();
                    break;
                case "inputs":
                    message.inputs.push($root.paddle.framework.proto.OpProto.Var.decodeText(reader));
                    break;
                case "outputs":
                    message.outputs.push($root.paddle.framework.proto.OpProto.Var.decodeText(reader));
                    break;
                case "attrs":
                    message.attrs.push($root.paddle.framework.proto.OpProto.Attr.decodeText(reader));
                    break;
                case "comment":
                    message.comment = reader.string();
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        if (!Object.prototype.hasOwnProperty.call(message, "type")) {
            throw new protobuf.Error("Excepted 'type'.");
        }
        if (!Object.prototype.hasOwnProperty.call(message, "comment")) {
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
        const end = length !== undefined ? reader.position + length : reader.length;
        while (reader.position < end) {
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
                case 6:
                    message.extra = reader.bool();
                    break;
                case 7:
                    message.quant = reader.bool();
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

    static decodeText(reader) {
        const message = new $root.paddle.framework.proto.OpProto.Var();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "name":
                    message.name = reader.string();
                    break;
                case "comment":
                    message.comment = reader.string();
                    break;
                case "duplicable":
                    message.duplicable = reader.bool();
                    break;
                case "intermediate":
                    message.intermediate = reader.bool();
                    break;
                case "dispensable":
                    message.dispensable = reader.bool();
                    break;
                case "extra":
                    message.extra = reader.bool();
                    break;
                case "quant":
                    message.quant = reader.bool();
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        if (!Object.prototype.hasOwnProperty.call(message, "name")) {
            throw new protobuf.Error("Excepted 'name'.");
        }
        if (!Object.prototype.hasOwnProperty.call(message, "comment")) {
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
$root.paddle.framework.proto.OpProto.Var.prototype.extra = false;
$root.paddle.framework.proto.OpProto.Var.prototype.quant = false;

$root.paddle.framework.proto.OpProto.Attr = class Attr {

    constructor() {
    }

    static decode(reader, length) {
        const message = new $root.paddle.framework.proto.OpProto.Attr();
        const end = length !== undefined ? reader.position + length : reader.length;
        while (reader.position < end) {
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
                case 5:
                    message.extra = reader.bool();
                    break;
                case 6:
                    message.quant = reader.bool();
                    break;
                case 7:
                    message.support_tensor = reader.bool();
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

    static decodeText(reader) {
        const message = new $root.paddle.framework.proto.OpProto.Attr();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "name":
                    message.name = reader.string();
                    break;
                case "type":
                    message.type = reader.enum($root.paddle.framework.proto.AttrType);
                    break;
                case "comment":
                    message.comment = reader.string();
                    break;
                case "generated":
                    message.generated = reader.bool();
                    break;
                case "extra":
                    message.extra = reader.bool();
                    break;
                case "quant":
                    message.quant = reader.bool();
                    break;
                case "support_tensor":
                    message.support_tensor = reader.bool();
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        if (!Object.prototype.hasOwnProperty.call(message, "name")) {
            throw new protobuf.Error("Excepted 'name'.");
        }
        if (!Object.prototype.hasOwnProperty.call(message, "type")) {
            throw new protobuf.Error("Excepted 'type'.");
        }
        if (!Object.prototype.hasOwnProperty.call(message, "comment")) {
            throw new protobuf.Error("Excepted 'comment'.");
        }
        return message;
    }
};

$root.paddle.framework.proto.OpProto.Attr.prototype.name = "";
$root.paddle.framework.proto.OpProto.Attr.prototype.type = 0;
$root.paddle.framework.proto.OpProto.Attr.prototype.comment = "";
$root.paddle.framework.proto.OpProto.Attr.prototype.generated = false;
$root.paddle.framework.proto.OpProto.Attr.prototype.extra = false;
$root.paddle.framework.proto.OpProto.Attr.prototype.quant = false;
$root.paddle.framework.proto.OpProto.Attr.prototype.support_tensor = false;

$root.paddle.framework.proto.VarType = class VarType {

    constructor() {
    }

    static decode(reader, length) {
        const message = new $root.paddle.framework.proto.VarType();
        const end = length !== undefined ? reader.position + length : reader.length;
        while (reader.position < end) {
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
                case 8:
                    message.string = $root.paddle.framework.proto.VarType.TensorDesc.decode(reader, reader.uint32());
                    break;
                case 9:
                    message.strings = $root.paddle.framework.proto.VarType.TensorDesc.decode(reader, reader.uint32());
                    break;
                case 10:
                    message.vocab = $root.paddle.framework.proto.VarType.TensorDesc.decode(reader, reader.uint32());
                    break;
                case 11:
                    message.sparse_coo = $root.paddle.framework.proto.VarType.TensorDesc.decode(reader, reader.uint32());
                    break;
                case 12:
                    message.sparse_csr = $root.paddle.framework.proto.VarType.TensorDesc.decode(reader, reader.uint32());
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

    static decodeText(reader) {
        const message = new $root.paddle.framework.proto.VarType();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "type":
                    message.type = reader.enum($root.paddle.framework.proto.VarType.Type);
                    break;
                case "selected_rows":
                    message.selected_rows = $root.paddle.framework.proto.VarType.TensorDesc.decodeText(reader);
                    break;
                case "lod_tensor":
                    message.lod_tensor = $root.paddle.framework.proto.VarType.LoDTensorDesc.decodeText(reader);
                    break;
                case "tensor_array":
                    message.tensor_array = $root.paddle.framework.proto.VarType.LoDTensorArrayDesc.decodeText(reader);
                    break;
                case "reader":
                    message.reader = $root.paddle.framework.proto.VarType.ReaderDesc.decodeText(reader);
                    break;
                case "tuple":
                    message.tuple = $root.paddle.framework.proto.VarType.Tuple.decodeText(reader);
                    break;
                case "string":
                    message.string = $root.paddle.framework.proto.VarType.TensorDesc.decodeText(reader);
                    break;
                case "strings":
                    message.strings = $root.paddle.framework.proto.VarType.TensorDesc.decodeText(reader);
                    break;
                case "vocab":
                    message.vocab = $root.paddle.framework.proto.VarType.TensorDesc.decodeText(reader);
                    break;
                case "sparse_coo":
                    message.sparse_coo = $root.paddle.framework.proto.VarType.TensorDesc.decodeText(reader);
                    break;
                case "sparse_csr":
                    message.sparse_csr = $root.paddle.framework.proto.VarType.TensorDesc.decodeText(reader);
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        if (!Object.prototype.hasOwnProperty.call(message, "type")) {
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
$root.paddle.framework.proto.VarType.prototype.string = null;
$root.paddle.framework.proto.VarType.prototype.strings = null;
$root.paddle.framework.proto.VarType.prototype.vocab = null;
$root.paddle.framework.proto.VarType.prototype.sparse_coo = null;
$root.paddle.framework.proto.VarType.prototype.sparse_csr = null;

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
    "COMPLEX64": 23,
    "COMPLEX128": 24,
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
    "TUPLE": 18,
    "STRING": 25,
    "STRINGS": 26,
    "VOCAB": 27,
    "FEED_LIST": 28,
    "PSTRING": 29,
    "SPARSE_COO": 30,
    "SPARSE_CSR": 31
};

$root.paddle.framework.proto.VarType.TensorDesc = class TensorDesc {

    constructor() {
        this.dims = [];
    }

    static decode(reader, length) {
        const message = new $root.paddle.framework.proto.VarType.TensorDesc();
        const end = length !== undefined ? reader.position + length : reader.length;
        while (reader.position < end) {
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

    static decodeText(reader) {
        const message = new $root.paddle.framework.proto.VarType.TensorDesc();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "data_type":
                    message.data_type = reader.enum($root.paddle.framework.proto.VarType.Type);
                    break;
                case "dims":
                    reader.array(message.dims, () => reader.int64());
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        if (!Object.prototype.hasOwnProperty.call(message, "data_type")) {
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
        const end = length !== undefined ? reader.position + length : reader.length;
        while (reader.position < end) {
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

    static decodeText(reader) {
        const message = new $root.paddle.framework.proto.VarType.LoDTensorDesc();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "tensor":
                    message.tensor = $root.paddle.framework.proto.VarType.TensorDesc.decodeText(reader);
                    break;
                case "lod_level":
                    message.lod_level = reader.int32();
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        if (!Object.prototype.hasOwnProperty.call(message, "tensor")) {
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
        const end = length !== undefined ? reader.position + length : reader.length;
        while (reader.position < end) {
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

    static decodeText(reader) {
        const message = new $root.paddle.framework.proto.VarType.LoDTensorArrayDesc();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "tensor":
                    message.tensor = $root.paddle.framework.proto.VarType.TensorDesc.decodeText(reader);
                    break;
                case "lod_level":
                    message.lod_level = reader.int32();
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        if (!Object.prototype.hasOwnProperty.call(message, "tensor")) {
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
        const end = length !== undefined ? reader.position + length : reader.length;
        while (reader.position < end) {
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

    static decodeText(reader) {
        const message = new $root.paddle.framework.proto.VarType.ReaderDesc();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "lod_tensor":
                    message.lod_tensor.push($root.paddle.framework.proto.VarType.LoDTensorDesc.decodeText(reader));
                    break;
                default:
                    reader.field(tag, message);
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
        const end = length !== undefined ? reader.position + length : reader.length;
        while (reader.position < end) {
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

    static decodeText(reader) {
        const message = new $root.paddle.framework.proto.VarType.Tuple();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "element_type":
                    reader.array(message.element_type, () => reader.enum($root.paddle.framework.proto.VarType.Type));
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

$root.paddle.framework.proto.VarDesc = class VarDesc {

    constructor() {
        this.attrs = [];
    }

    static decode(reader, length) {
        const message = new $root.paddle.framework.proto.VarDesc();
        const end = length !== undefined ? reader.position + length : reader.length;
        while (reader.position < end) {
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
                case 5:
                    message.is_parameter = reader.bool();
                    break;
                case 6:
                    message.stop_gradient = reader.bool();
                    break;
                case 7:
                    message.attrs.push($root.paddle.framework.proto.VarDesc.Attr.decode(reader, reader.uint32()));
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

    static decodeText(reader) {
        const message = new $root.paddle.framework.proto.VarDesc();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "name":
                    message.name = reader.string();
                    break;
                case "type":
                    message.type = $root.paddle.framework.proto.VarType.decodeText(reader);
                    break;
                case "persistable":
                    message.persistable = reader.bool();
                    break;
                case "need_check_feed":
                    message.need_check_feed = reader.bool();
                    break;
                case "is_parameter":
                    message.is_parameter = reader.bool();
                    break;
                case "stop_gradient":
                    message.stop_gradient = reader.bool();
                    break;
                case "attrs":
                    message.attrs.push($root.paddle.framework.proto.VarDesc.Attr.decodeText(reader));
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        if (!Object.prototype.hasOwnProperty.call(message, "name")) {
            throw new protobuf.Error("Excepted 'name'.");
        }
        if (!Object.prototype.hasOwnProperty.call(message, "type")) {
            throw new protobuf.Error("Excepted 'type'.");
        }
        return message;
    }
};

$root.paddle.framework.proto.VarDesc.prototype.name = "";
$root.paddle.framework.proto.VarDesc.prototype.type = null;
$root.paddle.framework.proto.VarDesc.prototype.persistable = false;
$root.paddle.framework.proto.VarDesc.prototype.need_check_feed = false;
$root.paddle.framework.proto.VarDesc.prototype.is_parameter = false;
$root.paddle.framework.proto.VarDesc.prototype.stop_gradient = false;

$root.paddle.framework.proto.VarDesc.Attr = class Attr {

    constructor() {
        this.ints = [];
    }

    static decode(reader, length) {
        const message = new $root.paddle.framework.proto.VarDesc.Attr();
        const end = length !== undefined ? reader.position + length : reader.length;
        while (reader.position < end) {
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
                    message.s = reader.string();
                    break;
                case 5:
                    message.ints = reader.array(message.ints, () => reader.int32(), tag);
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

    static decodeText(reader) {
        const message = new $root.paddle.framework.proto.VarDesc.Attr();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "name":
                    message.name = reader.string();
                    break;
                case "type":
                    message.type = reader.enum($root.paddle.framework.proto.AttrType);
                    break;
                case "i":
                    message.i = reader.int32();
                    break;
                case "s":
                    message.s = reader.string();
                    break;
                case "ints":
                    reader.array(message.ints, () => reader.int32());
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        if (!Object.prototype.hasOwnProperty.call(message, "name")) {
            throw new protobuf.Error("Excepted 'name'.");
        }
        if (!Object.prototype.hasOwnProperty.call(message, "type")) {
            throw new protobuf.Error("Excepted 'type'.");
        }
        return message;
    }
};

$root.paddle.framework.proto.VarDesc.Attr.prototype.name = "";
$root.paddle.framework.proto.VarDesc.Attr.prototype.type = 0;
$root.paddle.framework.proto.VarDesc.Attr.prototype.i = 0;
$root.paddle.framework.proto.VarDesc.Attr.prototype.s = "";

$root.paddle.framework.proto.BlockDesc = class BlockDesc {

    constructor() {
        this.vars = [];
        this.ops = [];
    }

    static decode(reader, length) {
        const message = new $root.paddle.framework.proto.BlockDesc();
        const end = length !== undefined ? reader.position + length : reader.length;
        while (reader.position < end) {
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

    static decodeText(reader) {
        const message = new $root.paddle.framework.proto.BlockDesc();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "idx":
                    message.idx = reader.int32();
                    break;
                case "parent_idx":
                    message.parent_idx = reader.int32();
                    break;
                case "vars":
                    message.vars.push($root.paddle.framework.proto.VarDesc.decodeText(reader));
                    break;
                case "ops":
                    message.ops.push($root.paddle.framework.proto.OpDesc.decodeText(reader));
                    break;
                case "forward_block_idx":
                    message.forward_block_idx = reader.int32();
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        if (!Object.prototype.hasOwnProperty.call(message, "idx")) {
            throw new protobuf.Error("Excepted 'idx'.");
        }
        if (!Object.prototype.hasOwnProperty.call(message, "parent_idx")) {
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
        const end = length !== undefined ? reader.position + length : reader.length;
        while (reader.position < end) {
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

    static decodeText(reader) {
        const message = new $root.paddle.framework.proto.OpVersion();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "version":
                    message.version = reader.int32();
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        if (!Object.prototype.hasOwnProperty.call(message, "version")) {
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
        const end = length !== undefined ? reader.position + length : reader.length;
        while (reader.position < end) {
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

    static decodeText(reader) {
        const message = new $root.paddle.framework.proto.OpVersionMap();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "pair":
                    message.pair.push($root.paddle.framework.proto.OpVersionMap.OpVersionPair.decodeText(reader));
                    break;
                default:
                    reader.field(tag, message);
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
        const end = length !== undefined ? reader.position + length : reader.length;
        while (reader.position < end) {
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

    static decodeText(reader) {
        const message = new $root.paddle.framework.proto.OpVersionMap.OpVersionPair();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "op_name":
                    message.op_name = reader.string();
                    break;
                case "op_version":
                    message.op_version = $root.paddle.framework.proto.OpVersion.decodeText(reader);
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        if (!Object.prototype.hasOwnProperty.call(message, "op_name")) {
            throw new protobuf.Error("Excepted 'op_name'.");
        }
        if (!Object.prototype.hasOwnProperty.call(message, "op_version")) {
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
        const end = length !== undefined ? reader.position + length : reader.length;
        while (reader.position < end) {
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

    static decodeText(reader) {
        const message = new $root.paddle.framework.proto.ProgramDesc();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "blocks":
                    message.blocks.push($root.paddle.framework.proto.BlockDesc.decodeText(reader));
                    break;
                case "version":
                    message.version = $root.paddle.framework.proto.Version.decodeText(reader);
                    break;
                case "op_version_map":
                    message.op_version_map = $root.paddle.framework.proto.OpVersionMap.decodeText(reader);
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

$root.paddle.framework.proto.ProgramDesc.prototype.version = null;
$root.paddle.framework.proto.ProgramDesc.prototype.op_version_map = null;
