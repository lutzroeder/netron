(function($protobuf) {
    "use strict";

    const $root = $protobuf.get('paddle');

    $root.paddle = (function() {

        const paddle = {};

        paddle.framework = (function() {

            const framework = {};

            framework.proto = (function() {

                const proto = {};

                proto.Version = (function() {

                    function Version() {
                    }

                    Version.prototype.version = $protobuf.Long ? $protobuf.Long.fromBits(0, 0, false) : 0;

                    Version.decode = function (reader, length) {
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
                    };

                    return Version;
                })();

                proto.AttrType = (function() {
                    const values = {};
                    values["INT"] = 0;
                    values["FLOAT"] = 1;
                    values["STRING"] = 2;
                    values["INTS"] = 3;
                    values["FLOATS"] = 4;
                    values["STRINGS"] = 5;
                    values["BOOLEAN"] = 6;
                    values["BOOLEANS"] = 7;
                    values["BLOCK"] = 8;
                    values["LONG"] = 9;
                    values["BLOCKS"] = 10;
                    values["LONGS"] = 11;
                    return values;
                })();

                proto.OpDesc = (function() {

                    function OpDesc() {
                        this.inputs = [];
                        this.outputs = [];
                        this.attrs = [];
                    }

                    OpDesc.prototype.type = "";
                    OpDesc.prototype.inputs = [];
                    OpDesc.prototype.outputs = [];
                    OpDesc.prototype.attrs = [];
                    OpDesc.prototype.is_target = false;

                    OpDesc.decode = function (reader, length) {
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
                            throw $protobuf.Error("Excepted 'type'.");
                        }
                        return message;
                    };

                    OpDesc.Attr = (function() {

                        function Attr() {
                            this.ints = [];
                            this.floats = [];
                            this.strings = [];
                            this.bools = [];
                            this.blocks_idx = [];
                            this.longs = [];
                        }

                        Attr.prototype.name = "";
                        Attr.prototype.type = 0;
                        Attr.prototype.i = 0;
                        Attr.prototype.f = 0;
                        Attr.prototype.s = "";
                        Attr.prototype.ints = [];
                        Attr.prototype.floats = [];
                        Attr.prototype.strings = [];
                        Attr.prototype.b = false;
                        Attr.prototype.bools = [];
                        Attr.prototype.block_idx = 0;
                        Attr.prototype.l = $protobuf.Long ? $protobuf.Long.fromBits(0, 0, false) : 0;
                        Attr.prototype.blocks_idx = [];
                        Attr.prototype.longs = [];

                        Attr.decode = function (reader, length) {
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
                                throw $protobuf.Error("Excepted 'name'.");
                            }
                            if (!Object.prototype.hasOwnProperty.call(message, 'type')) {
                                throw $protobuf.Error("Excepted 'type'.");
                            }
                            return message;
                        };

                        return Attr;
                    })();

                    OpDesc.Var = (function() {

                        function Var() {
                            this["arguments"] = [];
                        }

                        Var.prototype.parameter = "";
                        Var.prototype["arguments"] = [];

                        Var.decode = function (reader, length) {
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
                                throw $protobuf.Error("Excepted 'parameter'.");
                            }
                            return message;
                        };

                        return Var;
                    })();

                    return OpDesc;
                })();

                proto.OpProto = (function() {

                    function OpProto() {
                        this.inputs = [];
                        this.outputs = [];
                        this.attrs = [];
                    }

                    OpProto.prototype.type = "";
                    OpProto.prototype.inputs = [];
                    OpProto.prototype.outputs = [];
                    OpProto.prototype.attrs = [];
                    OpProto.prototype.comment = "";

                    OpProto.decode = function (reader, length) {
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
                            throw $protobuf.Error("Excepted 'type'.");
                        }
                        if (!Object.prototype.hasOwnProperty.call(message, 'comment')) {
                            throw $protobuf.Error("Excepted 'comment'.");
                        }
                        return message;
                    };

                    OpProto.Var = (function() {

                        function Var() {
                        }

                        Var.prototype.name = "";
                        Var.prototype.comment = "";
                        Var.prototype.duplicable = false;
                        Var.prototype.intermediate = false;
                        Var.prototype.dispensable = false;

                        Var.decode = function (reader, length) {
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
                                throw $protobuf.Error("Excepted 'name'.");
                            }
                            if (!Object.prototype.hasOwnProperty.call(message, 'comment')) {
                                throw $protobuf.Error("Excepted 'comment'.");
                            }
                            return message;
                        };

                        return Var;
                    })();

                    OpProto.Attr = (function() {

                        function Attr() {
                        }

                        Attr.prototype.name = "";
                        Attr.prototype.type = 0;
                        Attr.prototype.comment = "";
                        Attr.prototype.generated = false;

                        Attr.decode = function (reader, length) {
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
                                throw $protobuf.Error("Excepted 'name'.");
                            }
                            if (!Object.prototype.hasOwnProperty.call(message, 'type')) {
                                throw $protobuf.Error("Excepted 'type'.");
                            }
                            if (!Object.prototype.hasOwnProperty.call(message, 'comment')) {
                                throw $protobuf.Error("Excepted 'comment'.");
                            }
                            return message;
                        };

                        return Attr;
                    })();

                    return OpProto;
                })();

                proto.VarType = (function() {

                    function VarType() {
                    }

                    VarType.prototype.type = 0;
                    VarType.prototype.selected_rows = null;
                    VarType.prototype.lod_tensor = null;
                    VarType.prototype.tensor_array = null;
                    VarType.prototype.reader = null;
                    VarType.prototype.tuple = null;

                    VarType.decode = function (reader, length) {
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
                            throw $protobuf.Error("Excepted 'type'.");
                        }
                        return message;
                    };

                    VarType.Type = (function() {
                        const values = {};
                        values["BOOL"] = 0;
                        values["INT16"] = 1;
                        values["INT32"] = 2;
                        values["INT64"] = 3;
                        values["FP16"] = 4;
                        values["FP32"] = 5;
                        values["FP64"] = 6;
                        values["SIZE_T"] = 19;
                        values["UINT8"] = 20;
                        values["INT8"] = 21;
                        values["LOD_TENSOR"] = 7;
                        values["SELECTED_ROWS"] = 8;
                        values["FEED_MINIBATCH"] = 9;
                        values["FETCH_LIST"] = 10;
                        values["STEP_SCOPES"] = 11;
                        values["LOD_RANK_TABLE"] = 12;
                        values["LOD_TENSOR_ARRAY"] = 13;
                        values["PLACE_LIST"] = 14;
                        values["READER"] = 15;
                        values["RAW"] = 17;
                        values["TUPLE"] = 18;
                        return values;
                    })();

                    VarType.TensorDesc = (function() {

                        function TensorDesc() {
                            this.dims = [];
                        }

                        TensorDesc.prototype.data_type = 0;
                        TensorDesc.prototype.dims = [];

                        TensorDesc.decode = function (reader, length) {
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
                                throw $protobuf.Error("Excepted 'data_type'.");
                            }
                            return message;
                        };

                        return TensorDesc;
                    })();

                    VarType.LoDTensorDesc = (function() {

                        function LoDTensorDesc() {
                        }

                        LoDTensorDesc.prototype.tensor = null;
                        LoDTensorDesc.prototype.lod_level = 0;

                        LoDTensorDesc.decode = function (reader, length) {
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
                                throw $protobuf.Error("Excepted 'tensor'.");
                            }
                            return message;
                        };

                        return LoDTensorDesc;
                    })();

                    VarType.LoDTensorArrayDesc = (function() {

                        function LoDTensorArrayDesc() {
                        }

                        LoDTensorArrayDesc.prototype.tensor = null;
                        LoDTensorArrayDesc.prototype.lod_level = 0;

                        LoDTensorArrayDesc.decode = function (reader, length) {
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
                                throw $protobuf.Error("Excepted 'tensor'.");
                            }
                            return message;
                        };

                        return LoDTensorArrayDesc;
                    })();

                    VarType.ReaderDesc = (function() {

                        function ReaderDesc() {
                            this.lod_tensor = [];
                        }

                        ReaderDesc.prototype.lod_tensor = [];

                        ReaderDesc.decode = function (reader, length) {
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
                        };

                        return ReaderDesc;
                    })();

                    VarType.Tuple = (function() {

                        function Tuple() {
                            this.element_type = [];
                        }

                        Tuple.prototype.element_type = [];

                        Tuple.decode = function (reader, length) {
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
                        };

                        return Tuple;
                    })();

                    return VarType;
                })();

                proto.VarDesc = (function() {

                    function VarDesc() {
                    }

                    VarDesc.prototype.name = "";
                    VarDesc.prototype.type = null;
                    VarDesc.prototype.persistable = false;
                    VarDesc.prototype.need_check_feed = false;

                    VarDesc.decode = function (reader, length) {
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
                            throw $protobuf.Error("Excepted 'name'.");
                        }
                        if (!Object.prototype.hasOwnProperty.call(message, 'type')) {
                            throw $protobuf.Error("Excepted 'type'.");
                        }
                        return message;
                    };

                    return VarDesc;
                })();

                proto.BlockDesc = (function() {

                    function BlockDesc() {
                        this.vars = [];
                        this.ops = [];
                    }

                    BlockDesc.prototype.idx = 0;
                    BlockDesc.prototype.parent_idx = 0;
                    BlockDesc.prototype.vars = [];
                    BlockDesc.prototype.ops = [];
                    BlockDesc.prototype.forward_block_idx = -1;

                    BlockDesc.decode = function (reader, length) {
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
                            throw $protobuf.Error("Excepted 'idx'.");
                        }
                        if (!Object.prototype.hasOwnProperty.call(message, 'parent_idx')) {
                            throw $protobuf.Error("Excepted 'parent_idx'.");
                        }
                        return message;
                    };

                    return BlockDesc;
                })();

                proto.CompatibleInfo = (function() {

                    function CompatibleInfo() {
                    }

                    CompatibleInfo.prototype.version = "";
                    CompatibleInfo.prototype.type = 0;

                    CompatibleInfo.decode = function (reader, length) {
                        const message = new $root.paddle.framework.proto.CompatibleInfo();
                        const end = reader.next(length);
                        while (reader.end(end)) {
                            const tag = reader.uint32();
                            switch (tag >>> 3) {
                                case 1:
                                    message.version = reader.string();
                                    break;
                                case 2:
                                    message.type = reader.int32();
                                    break;
                                default:
                                    reader.skipType(tag & 7);
                                    break;
                            }
                        }
                        if (!Object.prototype.hasOwnProperty.call(message, 'version')) {
                            throw $protobuf.Error("Excepted 'version'.");
                        }
                        if (!Object.prototype.hasOwnProperty.call(message, 'type')) {
                            throw $protobuf.Error("Excepted 'type'.");
                        }
                        return message;
                    };

                    CompatibleInfo.Type = (function() {
                        const values = {};
                        values["COMPATIBLE"] = 0;
                        values["DEFINITELY_NOT"] = 1;
                        values["POSSIBLE"] = 2;
                        values["BUG_FIX"] = 3;
                        values["PRECISION_CHANGE"] = 4;
                        return values;
                    })();

                    return CompatibleInfo;
                })();

                proto.OpCompatibleMap = (function() {

                    function OpCompatibleMap() {
                        this.pair = [];
                    }

                    OpCompatibleMap.prototype.pair = [];
                    OpCompatibleMap.prototype.default_required_version = "";

                    OpCompatibleMap.decode = function (reader, length) {
                        const message = new $root.paddle.framework.proto.OpCompatibleMap();
                        const end = reader.next(length);
                        while (reader.end(end)) {
                            const tag = reader.uint32();
                            switch (tag >>> 3) {
                                case 1:
                                    message.pair.push($root.paddle.framework.proto.OpCompatibleMap.OpCompatiblePair.decode(reader, reader.uint32()));
                                    break;
                                case 2:
                                    message.default_required_version = reader.string();
                                    break;
                                default:
                                    reader.skipType(tag & 7);
                                    break;
                            }
                        }
                        return message;
                    };

                    OpCompatibleMap.OpCompatiblePair = (function() {

                        function OpCompatiblePair() {
                        }

                        OpCompatiblePair.prototype.op_name = "";
                        OpCompatiblePair.prototype.compatible_info = null;

                        OpCompatiblePair.decode = function (reader, length) {
                            const message = new $root.paddle.framework.proto.OpCompatibleMap.OpCompatiblePair();
                            const end = reader.next(length);
                            while (reader.end(end)) {
                                const tag = reader.uint32();
                                switch (tag >>> 3) {
                                    case 1:
                                        message.op_name = reader.string();
                                        break;
                                    case 2:
                                        message.compatible_info = $root.paddle.framework.proto.CompatibleInfo.decode(reader, reader.uint32());
                                        break;
                                    default:
                                        reader.skipType(tag & 7);
                                        break;
                                }
                            }
                            if (!Object.prototype.hasOwnProperty.call(message, 'op_name')) {
                                throw $protobuf.Error("Excepted 'op_name'.");
                            }
                            if (!Object.prototype.hasOwnProperty.call(message, 'compatible_info')) {
                                throw $protobuf.Error("Excepted 'compatible_info'.");
                            }
                            return message;
                        };

                        return OpCompatiblePair;
                    })();

                    return OpCompatibleMap;
                })();

                proto.ProgramDesc = (function() {

                    function ProgramDesc() {
                        this.blocks = [];
                    }

                    ProgramDesc.prototype.blocks = [];
                    ProgramDesc.prototype.version = null;
                    ProgramDesc.prototype.op_compatible_map = null;

                    ProgramDesc.decode = function (reader, length) {
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
                                case 3:
                                    message.op_compatible_map = $root.paddle.framework.proto.OpCompatibleMap.decode(reader, reader.uint32());
                                    break;
                                default:
                                    reader.skipType(tag & 7);
                                    break;
                            }
                        }
                        return message;
                    };

                    return ProgramDesc;
                })();

                return proto;
            })();

            return framework;
        })();

        return paddle;
    })();
    return $root;
})(protobuf);
