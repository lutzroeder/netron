
export const paddle = {};

paddle.framework = {};

paddle.framework.proto = {};

paddle.framework.proto.Version = class Version {

    static decode(reader, length) {
        const message = new paddle.framework.proto.Version();
        const end = length === undefined ? reader.length : reader.position + length;
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
        const message = new paddle.framework.proto.Version();
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

paddle.framework.proto.Version.prototype.version = 0n;

paddle.framework.proto.AttrType = {
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

paddle.framework.proto.Complex = class Complex {

    static decode(reader, length) {
        const message = new paddle.framework.proto.Complex();
        const end = length === undefined ? reader.length : reader.position + length;
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
            throw new Error("Excepted 'r'.");
        }
        if (!Object.prototype.hasOwnProperty.call(message, 'i')) {
            throw new Error("Excepted 'i'.");
        }
        return message;
    }

    static decodeText(reader) {
        const message = new paddle.framework.proto.Complex();
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
            throw new Error("Excepted 'r'.");
        }
        if (!Object.prototype.hasOwnProperty.call(message, "i")) {
            throw new Error("Excepted 'i'.");
        }
        return message;
    }
};

paddle.framework.proto.Complex.prototype.r = 0;
paddle.framework.proto.Complex.prototype.i = 0;

paddle.framework.proto.Scalar = class Scalar {

    static decode(reader, length) {
        const message = new paddle.framework.proto.Scalar();
        const end = length === undefined ? reader.length : reader.position + length;
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
                    message.c = paddle.framework.proto.Complex.decode(reader, reader.uint32());
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        if (!Object.prototype.hasOwnProperty.call(message, 'type')) {
            throw new Error("Excepted 'type'.");
        }
        return message;
    }

    static decodeText(reader) {
        const message = new paddle.framework.proto.Scalar();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "type":
                    message.type = reader.enum(paddle.framework.proto.Scalar.Type);
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
                    message.c = paddle.framework.proto.Complex.decodeText(reader);
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        if (!Object.prototype.hasOwnProperty.call(message, "type")) {
            throw new Error("Excepted 'type'.");
        }
        return message;
    }
};

paddle.framework.proto.Scalar.prototype.type = 1;
paddle.framework.proto.Scalar.prototype.b = false;
paddle.framework.proto.Scalar.prototype.i = 0n;
paddle.framework.proto.Scalar.prototype.r = 0;
paddle.framework.proto.Scalar.prototype.c = null;

paddle.framework.proto.Scalar.Type = {
    "BOOLEAN": 1,
    "LONG": 2,
    "FLOAT64": 3,
    "COMPLEX128": 4
};

paddle.framework.proto.OpDesc = class OpDesc {

    constructor() {
        this.inputs = [];
        this.outputs = [];
        this.attrs = [];
    }

    static decode(reader, length) {
        const message = new paddle.framework.proto.OpDesc();
        const end = length === undefined ? reader.length : reader.position + length;
        while (reader.position < end) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 3:
                    message.type = reader.string();
                    break;
                case 1:
                    message.inputs.push(paddle.framework.proto.OpDesc.Var.decode(reader, reader.uint32()));
                    break;
                case 2:
                    message.outputs.push(paddle.framework.proto.OpDesc.Var.decode(reader, reader.uint32()));
                    break;
                case 4:
                    message.attrs.push(paddle.framework.proto.OpDesc.Attr.decode(reader, reader.uint32()));
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
            throw new Error("Excepted 'type'.");
        }
        return message;
    }

    static decodeText(reader) {
        const message = new paddle.framework.proto.OpDesc();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "type":
                    message.type = reader.string();
                    break;
                case "inputs":
                    message.inputs.push(paddle.framework.proto.OpDesc.Var.decodeText(reader));
                    break;
                case "outputs":
                    message.outputs.push(paddle.framework.proto.OpDesc.Var.decodeText(reader));
                    break;
                case "attrs":
                    message.attrs.push(paddle.framework.proto.OpDesc.Attr.decodeText(reader));
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
            throw new Error("Excepted 'type'.");
        }
        return message;
    }
};

paddle.framework.proto.OpDesc.prototype.type = "";
paddle.framework.proto.OpDesc.prototype.is_target = false;

paddle.framework.proto.OpDesc.Attr = class Attr {

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
        const message = new paddle.framework.proto.OpDesc.Attr();
        const end = length === undefined ? reader.length : reader.position + length;
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
                    message.scalar = paddle.framework.proto.Scalar.decode(reader, reader.uint32());
                    break;
                case 21:
                    message.scalars.push(paddle.framework.proto.Scalar.decode(reader, reader.uint32()));
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        if (!Object.prototype.hasOwnProperty.call(message, 'name')) {
            throw new Error("Excepted 'name'.");
        }
        if (!Object.prototype.hasOwnProperty.call(message, 'type')) {
            throw new Error("Excepted 'type'.");
        }
        return message;
    }

    static decodeText(reader) {
        const message = new paddle.framework.proto.OpDesc.Attr();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "name":
                    message.name = reader.string();
                    break;
                case "type":
                    message.type = reader.enum(paddle.framework.proto.AttrType);
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
                    message.scalar = paddle.framework.proto.Scalar.decodeText(reader);
                    break;
                case "scalars":
                    message.scalars.push(paddle.framework.proto.Scalar.decodeText(reader));
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        if (!Object.prototype.hasOwnProperty.call(message, "name")) {
            throw new Error("Excepted 'name'.");
        }
        if (!Object.prototype.hasOwnProperty.call(message, "type")) {
            throw new Error("Excepted 'type'.");
        }
        return message;
    }
};

paddle.framework.proto.OpDesc.Attr.prototype.name = "";
paddle.framework.proto.OpDesc.Attr.prototype.type = 0;
paddle.framework.proto.OpDesc.Attr.prototype.i = 0;
paddle.framework.proto.OpDesc.Attr.prototype.f = 0;
paddle.framework.proto.OpDesc.Attr.prototype.s = "";
paddle.framework.proto.OpDesc.Attr.prototype.b = false;
paddle.framework.proto.OpDesc.Attr.prototype.block_idx = 0;
paddle.framework.proto.OpDesc.Attr.prototype.l = 0n;
paddle.framework.proto.OpDesc.Attr.prototype.var_name = "";
paddle.framework.proto.OpDesc.Attr.prototype.float64 = 0;
paddle.framework.proto.OpDesc.Attr.prototype.scalar = null;

paddle.framework.proto.OpDesc.Var = class Var {

    constructor() {
        this.arguments = [];
    }

    static decode(reader, length) {
        const message = new paddle.framework.proto.OpDesc.Var();
        const end = length === undefined ? reader.length : reader.position + length;
        while (reader.position < end) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.parameter = reader.string();
                    break;
                case 2:
                    message.arguments.push(reader.string());
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        if (!Object.prototype.hasOwnProperty.call(message, 'parameter')) {
            throw new Error("Excepted 'parameter'.");
        }
        return message;
    }

    static decodeText(reader) {
        const message = new paddle.framework.proto.OpDesc.Var();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "parameter":
                    message.parameter = reader.string();
                    break;
                case "arguments":
                    reader.array(message.arguments, () => reader.string());
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        if (!Object.prototype.hasOwnProperty.call(message, "parameter")) {
            throw new Error("Excepted 'parameter'.");
        }
        return message;
    }
};

paddle.framework.proto.OpDesc.Var.prototype.parameter = "";

paddle.framework.proto.OpProto = class OpProto {

    constructor() {
        this.inputs = [];
        this.outputs = [];
        this.attrs = [];
    }

    static decode(reader, length) {
        const message = new paddle.framework.proto.OpProto();
        const end = length === undefined ? reader.length : reader.position + length;
        while (reader.position < end) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.type = reader.string();
                    break;
                case 2:
                    message.inputs.push(paddle.framework.proto.OpProto.Var.decode(reader, reader.uint32()));
                    break;
                case 3:
                    message.outputs.push(paddle.framework.proto.OpProto.Var.decode(reader, reader.uint32()));
                    break;
                case 4:
                    message.attrs.push(paddle.framework.proto.OpProto.Attr.decode(reader, reader.uint32()));
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
            throw new Error("Excepted 'type'.");
        }
        if (!Object.prototype.hasOwnProperty.call(message, 'comment')) {
            throw new Error("Excepted 'comment'.");
        }
        return message;
    }

    static decodeText(reader) {
        const message = new paddle.framework.proto.OpProto();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "type":
                    message.type = reader.string();
                    break;
                case "inputs":
                    message.inputs.push(paddle.framework.proto.OpProto.Var.decodeText(reader));
                    break;
                case "outputs":
                    message.outputs.push(paddle.framework.proto.OpProto.Var.decodeText(reader));
                    break;
                case "attrs":
                    message.attrs.push(paddle.framework.proto.OpProto.Attr.decodeText(reader));
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
            throw new Error("Excepted 'type'.");
        }
        if (!Object.prototype.hasOwnProperty.call(message, "comment")) {
            throw new Error("Excepted 'comment'.");
        }
        return message;
    }
};

paddle.framework.proto.OpProto.prototype.type = "";
paddle.framework.proto.OpProto.prototype.comment = "";

paddle.framework.proto.OpProto.Var = class Var {

    static decode(reader, length) {
        const message = new paddle.framework.proto.OpProto.Var();
        const end = length === undefined ? reader.length : reader.position + length;
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
            throw new Error("Excepted 'name'.");
        }
        if (!Object.prototype.hasOwnProperty.call(message, 'comment')) {
            throw new Error("Excepted 'comment'.");
        }
        return message;
    }

    static decodeText(reader) {
        const message = new paddle.framework.proto.OpProto.Var();
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
            throw new Error("Excepted 'name'.");
        }
        if (!Object.prototype.hasOwnProperty.call(message, "comment")) {
            throw new Error("Excepted 'comment'.");
        }
        return message;
    }
};

paddle.framework.proto.OpProto.Var.prototype.name = "";
paddle.framework.proto.OpProto.Var.prototype.comment = "";
paddle.framework.proto.OpProto.Var.prototype.duplicable = false;
paddle.framework.proto.OpProto.Var.prototype.intermediate = false;
paddle.framework.proto.OpProto.Var.prototype.dispensable = false;
paddle.framework.proto.OpProto.Var.prototype.extra = false;
paddle.framework.proto.OpProto.Var.prototype.quant = false;

paddle.framework.proto.OpProto.Attr = class Attr {

    static decode(reader, length) {
        const message = new paddle.framework.proto.OpProto.Attr();
        const end = length === undefined ? reader.length : reader.position + length;
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
            throw new Error("Excepted 'name'.");
        }
        if (!Object.prototype.hasOwnProperty.call(message, 'type')) {
            throw new Error("Excepted 'type'.");
        }
        if (!Object.prototype.hasOwnProperty.call(message, 'comment')) {
            throw new Error("Excepted 'comment'.");
        }
        return message;
    }

    static decodeText(reader) {
        const message = new paddle.framework.proto.OpProto.Attr();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "name":
                    message.name = reader.string();
                    break;
                case "type":
                    message.type = reader.enum(paddle.framework.proto.AttrType);
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
            throw new Error("Excepted 'name'.");
        }
        if (!Object.prototype.hasOwnProperty.call(message, "type")) {
            throw new Error("Excepted 'type'.");
        }
        if (!Object.prototype.hasOwnProperty.call(message, "comment")) {
            throw new Error("Excepted 'comment'.");
        }
        return message;
    }
};

paddle.framework.proto.OpProto.Attr.prototype.name = "";
paddle.framework.proto.OpProto.Attr.prototype.type = 0;
paddle.framework.proto.OpProto.Attr.prototype.comment = "";
paddle.framework.proto.OpProto.Attr.prototype.generated = false;
paddle.framework.proto.OpProto.Attr.prototype.extra = false;
paddle.framework.proto.OpProto.Attr.prototype.quant = false;
paddle.framework.proto.OpProto.Attr.prototype.support_tensor = false;

paddle.framework.proto.VarType = class VarType {

    static decode(reader, length) {
        const message = new paddle.framework.proto.VarType();
        const end = length === undefined ? reader.length : reader.position + length;
        while (reader.position < end) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.type = reader.int32();
                    break;
                case 2:
                    message.selected_rows = paddle.framework.proto.VarType.TensorDesc.decode(reader, reader.uint32());
                    break;
                case 3:
                    message.lod_tensor = paddle.framework.proto.VarType.LoDTensorDesc.decode(reader, reader.uint32());
                    break;
                case 4:
                    message.tensor_array = paddle.framework.proto.VarType.LoDTensorArrayDesc.decode(reader, reader.uint32());
                    break;
                case 5:
                    message.reader = paddle.framework.proto.VarType.ReaderDesc.decode(reader, reader.uint32());
                    break;
                case 7:
                    message.tuple = paddle.framework.proto.VarType.Tuple.decode(reader, reader.uint32());
                    break;
                case 8:
                    message.string = paddle.framework.proto.VarType.TensorDesc.decode(reader, reader.uint32());
                    break;
                case 9:
                    message.strings = paddle.framework.proto.VarType.TensorDesc.decode(reader, reader.uint32());
                    break;
                case 10:
                    message.vocab = paddle.framework.proto.VarType.TensorDesc.decode(reader, reader.uint32());
                    break;
                case 11:
                    message.sparse_coo = paddle.framework.proto.VarType.TensorDesc.decode(reader, reader.uint32());
                    break;
                case 12:
                    message.sparse_csr = paddle.framework.proto.VarType.TensorDesc.decode(reader, reader.uint32());
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        if (!Object.prototype.hasOwnProperty.call(message, 'type')) {
            throw new Error("Excepted 'type'.");
        }
        return message;
    }

    static decodeText(reader) {
        const message = new paddle.framework.proto.VarType();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "type":
                    message.type = reader.enum(paddle.framework.proto.VarType.Type);
                    break;
                case "selected_rows":
                    message.selected_rows = paddle.framework.proto.VarType.TensorDesc.decodeText(reader);
                    break;
                case "lod_tensor":
                    message.lod_tensor = paddle.framework.proto.VarType.LoDTensorDesc.decodeText(reader);
                    break;
                case "tensor_array":
                    message.tensor_array = paddle.framework.proto.VarType.LoDTensorArrayDesc.decodeText(reader);
                    break;
                case "reader":
                    message.reader = paddle.framework.proto.VarType.ReaderDesc.decodeText(reader);
                    break;
                case "tuple":
                    message.tuple = paddle.framework.proto.VarType.Tuple.decodeText(reader);
                    break;
                case "string":
                    message.string = paddle.framework.proto.VarType.TensorDesc.decodeText(reader);
                    break;
                case "strings":
                    message.strings = paddle.framework.proto.VarType.TensorDesc.decodeText(reader);
                    break;
                case "vocab":
                    message.vocab = paddle.framework.proto.VarType.TensorDesc.decodeText(reader);
                    break;
                case "sparse_coo":
                    message.sparse_coo = paddle.framework.proto.VarType.TensorDesc.decodeText(reader);
                    break;
                case "sparse_csr":
                    message.sparse_csr = paddle.framework.proto.VarType.TensorDesc.decodeText(reader);
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        if (!Object.prototype.hasOwnProperty.call(message, "type")) {
            throw new Error("Excepted 'type'.");
        }
        return message;
    }
};

paddle.framework.proto.VarType.prototype.type = 0;
paddle.framework.proto.VarType.prototype.selected_rows = null;
paddle.framework.proto.VarType.prototype.lod_tensor = null;
paddle.framework.proto.VarType.prototype.tensor_array = null;
paddle.framework.proto.VarType.prototype.reader = null;
paddle.framework.proto.VarType.prototype.tuple = null;
paddle.framework.proto.VarType.prototype.string = null;
paddle.framework.proto.VarType.prototype.strings = null;
paddle.framework.proto.VarType.prototype.vocab = null;
paddle.framework.proto.VarType.prototype.sparse_coo = null;
paddle.framework.proto.VarType.prototype.sparse_csr = null;

paddle.framework.proto.VarType.Type = {
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
    "FP8_E4M3FN": 32,
    "FP8_E5M2": 33,
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

paddle.framework.proto.VarType.TensorDesc = class TensorDesc {

    constructor() {
        this.dims = [];
    }

    static decode(reader, length) {
        const message = new paddle.framework.proto.VarType.TensorDesc();
        const end = length === undefined ? reader.length : reader.position + length;
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
            throw new Error("Excepted 'data_type'.");
        }
        return message;
    }

    static decodeText(reader) {
        const message = new paddle.framework.proto.VarType.TensorDesc();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "data_type":
                    message.data_type = reader.enum(paddle.framework.proto.VarType.Type);
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
            throw new Error("Excepted 'data_type'.");
        }
        return message;
    }
};

paddle.framework.proto.VarType.TensorDesc.prototype.data_type = 0;

paddle.framework.proto.VarType.LoDTensorDesc = class LoDTensorDesc {

    static decode(reader, length) {
        const message = new paddle.framework.proto.VarType.LoDTensorDesc();
        const end = length === undefined ? reader.length : reader.position + length;
        while (reader.position < end) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.tensor = paddle.framework.proto.VarType.TensorDesc.decode(reader, reader.uint32());
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
            throw new Error("Excepted 'tensor'.");
        }
        return message;
    }

    static decodeText(reader) {
        const message = new paddle.framework.proto.VarType.LoDTensorDesc();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "tensor":
                    message.tensor = paddle.framework.proto.VarType.TensorDesc.decodeText(reader);
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
            throw new Error("Excepted 'tensor'.");
        }
        return message;
    }
};

paddle.framework.proto.VarType.LoDTensorDesc.prototype.tensor = null;
paddle.framework.proto.VarType.LoDTensorDesc.prototype.lod_level = 0;

paddle.framework.proto.VarType.LoDTensorArrayDesc = class LoDTensorArrayDesc {

    static decode(reader, length) {
        const message = new paddle.framework.proto.VarType.LoDTensorArrayDesc();
        const end = length === undefined ? reader.length : reader.position + length;
        while (reader.position < end) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.tensor = paddle.framework.proto.VarType.TensorDesc.decode(reader, reader.uint32());
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
            throw new Error("Excepted 'tensor'.");
        }
        return message;
    }

    static decodeText(reader) {
        const message = new paddle.framework.proto.VarType.LoDTensorArrayDesc();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "tensor":
                    message.tensor = paddle.framework.proto.VarType.TensorDesc.decodeText(reader);
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
            throw new Error("Excepted 'tensor'.");
        }
        return message;
    }
};

paddle.framework.proto.VarType.LoDTensorArrayDesc.prototype.tensor = null;
paddle.framework.proto.VarType.LoDTensorArrayDesc.prototype.lod_level = 0;

paddle.framework.proto.VarType.ReaderDesc = class ReaderDesc {

    constructor() {
        this.lod_tensor = [];
    }

    static decode(reader, length) {
        const message = new paddle.framework.proto.VarType.ReaderDesc();
        const end = length === undefined ? reader.length : reader.position + length;
        while (reader.position < end) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.lod_tensor.push(paddle.framework.proto.VarType.LoDTensorDesc.decode(reader, reader.uint32()));
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }

    static decodeText(reader) {
        const message = new paddle.framework.proto.VarType.ReaderDesc();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "lod_tensor":
                    message.lod_tensor.push(paddle.framework.proto.VarType.LoDTensorDesc.decodeText(reader));
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

paddle.framework.proto.VarType.Tuple = class Tuple {

    constructor() {
        this.element_type = [];
    }

    static decode(reader, length) {
        const message = new paddle.framework.proto.VarType.Tuple();
        const end = length === undefined ? reader.length : reader.position + length;
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
        const message = new paddle.framework.proto.VarType.Tuple();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "element_type":
                    reader.array(message.element_type, () => reader.enum(paddle.framework.proto.VarType.Type));
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

paddle.framework.proto.VarDesc = class VarDesc {

    constructor() {
        this.attrs = [];
    }

    static decode(reader, length) {
        const message = new paddle.framework.proto.VarDesc();
        const end = length === undefined ? reader.length : reader.position + length;
        while (reader.position < end) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.name = reader.string();
                    break;
                case 2:
                    message.type = paddle.framework.proto.VarType.decode(reader, reader.uint32());
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
                    message.attrs.push(paddle.framework.proto.VarDesc.Attr.decode(reader, reader.uint32()));
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        if (!Object.prototype.hasOwnProperty.call(message, 'name')) {
            throw new Error("Excepted 'name'.");
        }
        if (!Object.prototype.hasOwnProperty.call(message, 'type')) {
            throw new Error("Excepted 'type'.");
        }
        return message;
    }

    static decodeText(reader) {
        const message = new paddle.framework.proto.VarDesc();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "name":
                    message.name = reader.string();
                    break;
                case "type":
                    message.type = paddle.framework.proto.VarType.decodeText(reader);
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
                    message.attrs.push(paddle.framework.proto.VarDesc.Attr.decodeText(reader));
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        if (!Object.prototype.hasOwnProperty.call(message, "name")) {
            throw new Error("Excepted 'name'.");
        }
        if (!Object.prototype.hasOwnProperty.call(message, "type")) {
            throw new Error("Excepted 'type'.");
        }
        return message;
    }
};

paddle.framework.proto.VarDesc.prototype.name = "";
paddle.framework.proto.VarDesc.prototype.type = null;
paddle.framework.proto.VarDesc.prototype.persistable = false;
paddle.framework.proto.VarDesc.prototype.need_check_feed = false;
paddle.framework.proto.VarDesc.prototype.is_parameter = false;
paddle.framework.proto.VarDesc.prototype.stop_gradient = false;

paddle.framework.proto.VarDesc.Attr = class Attr {

    constructor() {
        this.ints = [];
    }

    static decode(reader, length) {
        const message = new paddle.framework.proto.VarDesc.Attr();
        const end = length === undefined ? reader.length : reader.position + length;
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
            throw new Error("Excepted 'name'.");
        }
        if (!Object.prototype.hasOwnProperty.call(message, 'type')) {
            throw new Error("Excepted 'type'.");
        }
        return message;
    }

    static decodeText(reader) {
        const message = new paddle.framework.proto.VarDesc.Attr();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "name":
                    message.name = reader.string();
                    break;
                case "type":
                    message.type = reader.enum(paddle.framework.proto.AttrType);
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
            throw new Error("Excepted 'name'.");
        }
        if (!Object.prototype.hasOwnProperty.call(message, "type")) {
            throw new Error("Excepted 'type'.");
        }
        return message;
    }
};

paddle.framework.proto.VarDesc.Attr.prototype.name = "";
paddle.framework.proto.VarDesc.Attr.prototype.type = 0;
paddle.framework.proto.VarDesc.Attr.prototype.i = 0;
paddle.framework.proto.VarDesc.Attr.prototype.s = "";

paddle.framework.proto.BlockDesc = class BlockDesc {

    constructor() {
        this.vars = [];
        this.ops = [];
    }

    static decode(reader, length) {
        const message = new paddle.framework.proto.BlockDesc();
        const end = length === undefined ? reader.length : reader.position + length;
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
                    message.vars.push(paddle.framework.proto.VarDesc.decode(reader, reader.uint32()));
                    break;
                case 4:
                    message.ops.push(paddle.framework.proto.OpDesc.decode(reader, reader.uint32()));
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
            throw new Error("Excepted 'idx'.");
        }
        if (!Object.prototype.hasOwnProperty.call(message, 'parent_idx')) {
            throw new Error("Excepted 'parent_idx'.");
        }
        return message;
    }

    static decodeText(reader) {
        const message = new paddle.framework.proto.BlockDesc();
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
                    message.vars.push(paddle.framework.proto.VarDesc.decodeText(reader));
                    break;
                case "ops":
                    message.ops.push(paddle.framework.proto.OpDesc.decodeText(reader));
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
            throw new Error("Excepted 'idx'.");
        }
        if (!Object.prototype.hasOwnProperty.call(message, "parent_idx")) {
            throw new Error("Excepted 'parent_idx'.");
        }
        return message;
    }
};

paddle.framework.proto.BlockDesc.prototype.idx = 0;
paddle.framework.proto.BlockDesc.prototype.parent_idx = 0;
paddle.framework.proto.BlockDesc.prototype.forward_block_idx = -1;

paddle.framework.proto.OpVersion = class OpVersion {

    static decode(reader, length) {
        const message = new paddle.framework.proto.OpVersion();
        const end = length === undefined ? reader.length : reader.position + length;
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
            throw new Error("Excepted 'version'.");
        }
        return message;
    }

    static decodeText(reader) {
        const message = new paddle.framework.proto.OpVersion();
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
            throw new Error("Excepted 'version'.");
        }
        return message;
    }
};

paddle.framework.proto.OpVersion.prototype.version = 0;

paddle.framework.proto.OpVersionMap = class OpVersionMap {

    constructor() {
        this.pair = [];
    }

    static decode(reader, length) {
        const message = new paddle.framework.proto.OpVersionMap();
        const end = length === undefined ? reader.length : reader.position + length;
        while (reader.position < end) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.pair.push(paddle.framework.proto.OpVersionMap.OpVersionPair.decode(reader, reader.uint32()));
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }

    static decodeText(reader) {
        const message = new paddle.framework.proto.OpVersionMap();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "pair":
                    message.pair.push(paddle.framework.proto.OpVersionMap.OpVersionPair.decodeText(reader));
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

paddle.framework.proto.OpVersionMap.OpVersionPair = class OpVersionPair {

    static decode(reader, length) {
        const message = new paddle.framework.proto.OpVersionMap.OpVersionPair();
        const end = length === undefined ? reader.length : reader.position + length;
        while (reader.position < end) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.op_name = reader.string();
                    break;
                case 2:
                    message.op_version = paddle.framework.proto.OpVersion.decode(reader, reader.uint32());
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        if (!Object.prototype.hasOwnProperty.call(message, 'op_name')) {
            throw new Error("Excepted 'op_name'.");
        }
        if (!Object.prototype.hasOwnProperty.call(message, 'op_version')) {
            throw new Error("Excepted 'op_version'.");
        }
        return message;
    }

    static decodeText(reader) {
        const message = new paddle.framework.proto.OpVersionMap.OpVersionPair();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "op_name":
                    message.op_name = reader.string();
                    break;
                case "op_version":
                    message.op_version = paddle.framework.proto.OpVersion.decodeText(reader);
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        if (!Object.prototype.hasOwnProperty.call(message, "op_name")) {
            throw new Error("Excepted 'op_name'.");
        }
        if (!Object.prototype.hasOwnProperty.call(message, "op_version")) {
            throw new Error("Excepted 'op_version'.");
        }
        return message;
    }
};

paddle.framework.proto.OpVersionMap.OpVersionPair.prototype.op_name = "";
paddle.framework.proto.OpVersionMap.OpVersionPair.prototype.op_version = null;

paddle.framework.proto.ProgramDesc = class ProgramDesc {

    constructor() {
        this.blocks = [];
    }

    static decode(reader, length) {
        const message = new paddle.framework.proto.ProgramDesc();
        const end = length === undefined ? reader.length : reader.position + length;
        while (reader.position < end) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.blocks.push(paddle.framework.proto.BlockDesc.decode(reader, reader.uint32()));
                    break;
                case 4:
                    message.version = paddle.framework.proto.Version.decode(reader, reader.uint32());
                    break;
                case 5:
                    message.op_version_map = paddle.framework.proto.OpVersionMap.decode(reader, reader.uint32());
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }

    static decodeText(reader) {
        const message = new paddle.framework.proto.ProgramDesc();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "blocks":
                    message.blocks.push(paddle.framework.proto.BlockDesc.decodeText(reader));
                    break;
                case "version":
                    message.version = paddle.framework.proto.Version.decodeText(reader);
                    break;
                case "op_version_map":
                    message.op_version_map = paddle.framework.proto.OpVersionMap.decodeText(reader);
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

paddle.framework.proto.ProgramDesc.prototype.version = null;
paddle.framework.proto.ProgramDesc.prototype.op_version_map = null;
