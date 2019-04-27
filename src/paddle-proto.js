/*eslint-disable block-scoped-var, id-length, no-control-regex, no-magic-numbers, no-prototype-builtins, no-redeclare, no-shadow, no-var, sort-vars*/
(function($protobuf) {
    "use strict";

    var $Reader = $protobuf.Reader, $util = $protobuf.util;
    
    var $root = $protobuf.roots.paddle || ($protobuf.roots.paddle = {});
    
    $root.paddle = (function() {
    
        var paddle = {};
    
        paddle.framework = (function() {
    
            var framework = {};
    
            framework.proto = (function() {
    
                var proto = {};
    
                proto.Version = (function() {
    
                    function Version(properties) {
                        if (properties)
                            for (var keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                                if (properties[keys[i]] != null)
                                    this[keys[i]] = properties[keys[i]];
                    }
    
                    Version.prototype.version = $util.Long ? $util.Long.fromBits(0,0,false) : 0;
    
                    Version.decode = function decode(reader, length) {
                        if (!(reader instanceof $Reader))
                            reader = $Reader.create(reader);
                        var end = length === undefined ? reader.len : reader.pos + length, message = new $root.paddle.framework.proto.Version();
                        while (reader.pos < end) {
                            var tag = reader.uint32();
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
                    var valuesById = {}, values = Object.create(valuesById);
                    values[valuesById[0] = "INT"] = 0;
                    values[valuesById[1] = "FLOAT"] = 1;
                    values[valuesById[2] = "STRING"] = 2;
                    values[valuesById[3] = "INTS"] = 3;
                    values[valuesById[4] = "FLOATS"] = 4;
                    values[valuesById[5] = "STRINGS"] = 5;
                    values[valuesById[6] = "BOOLEAN"] = 6;
                    values[valuesById[7] = "BOOLEANS"] = 7;
                    values[valuesById[8] = "BLOCK"] = 8;
                    values[valuesById[9] = "LONG"] = 9;
                    values[valuesById[10] = "BLOCKS"] = 10;
                    values[valuesById[11] = "LONGS"] = 11;
                    return values;
                })();
    
                proto.OpDesc = (function() {
    
                    function OpDesc(properties) {
                        this.inputs = [];
                        this.outputs = [];
                        this.attrs = [];
                        if (properties)
                            for (var keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                                if (properties[keys[i]] != null)
                                    this[keys[i]] = properties[keys[i]];
                    }
    
                    OpDesc.prototype.type = "";
                    OpDesc.prototype.inputs = $util.emptyArray;
                    OpDesc.prototype.outputs = $util.emptyArray;
                    OpDesc.prototype.attrs = $util.emptyArray;
                    OpDesc.prototype.is_target = false;
    
                    OpDesc.decode = function decode(reader, length) {
                        if (!(reader instanceof $Reader))
                            reader = $Reader.create(reader);
                        var end = length === undefined ? reader.len : reader.pos + length, message = new $root.paddle.framework.proto.OpDesc();
                        while (reader.pos < end) {
                            var tag = reader.uint32();
                            switch (tag >>> 3) {
                            case 3:
                                message.type = reader.string();
                                break;
                            case 1:
                                if (!(message.inputs && message.inputs.length))
                                    message.inputs = [];
                                message.inputs.push($root.paddle.framework.proto.OpDesc.Var.decode(reader, reader.uint32()));
                                break;
                            case 2:
                                if (!(message.outputs && message.outputs.length))
                                    message.outputs = [];
                                message.outputs.push($root.paddle.framework.proto.OpDesc.Var.decode(reader, reader.uint32()));
                                break;
                            case 4:
                                if (!(message.attrs && message.attrs.length))
                                    message.attrs = [];
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
                        if (!message.hasOwnProperty("type"))
                            throw $util.ProtocolError("missing required 'type'", { instance: message });
                        return message;
                    };
    
                    OpDesc.Attr = (function() {
    
                        function Attr(properties) {
                            this.ints = [];
                            this.floats = [];
                            this.strings = [];
                            this.bools = [];
                            this.blocks_idx = [];
                            this.longs = [];
                            if (properties)
                                for (var keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                                    if (properties[keys[i]] != null)
                                        this[keys[i]] = properties[keys[i]];
                        }
    
                        Attr.prototype.name = "";
                        Attr.prototype.type = 0;
                        Attr.prototype.i = 0;
                        Attr.prototype.f = 0;
                        Attr.prototype.s = "";
                        Attr.prototype.ints = $util.emptyArray;
                        Attr.prototype.floats = $util.emptyArray;
                        Attr.prototype.strings = $util.emptyArray;
                        Attr.prototype.b = false;
                        Attr.prototype.bools = $util.emptyArray;
                        Attr.prototype.block_idx = 0;
                        Attr.prototype.l = $util.Long ? $util.Long.fromBits(0,0,false) : 0;
                        Attr.prototype.blocks_idx = $util.emptyArray;
                        Attr.prototype.longs = $util.emptyArray;
    
                        Attr.decode = function decode(reader, length) {
                            if (!(reader instanceof $Reader))
                                reader = $Reader.create(reader);
                            var end = length === undefined ? reader.len : reader.pos + length, message = new $root.paddle.framework.proto.OpDesc.Attr();
                            while (reader.pos < end) {
                                var tag = reader.uint32();
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
                                    if (!(message.ints && message.ints.length))
                                        message.ints = [];
                                    if ((tag & 7) === 2) {
                                        var end2 = reader.uint32() + reader.pos;
                                        while (reader.pos < end2)
                                            message.ints.push(reader.int32());
                                    } else
                                        message.ints.push(reader.int32());
                                    break;
                                case 7:
                                    if (!(message.floats && message.floats.length))
                                        message.floats = [];
                                    if ((tag & 7) === 2) {
                                        var end2 = reader.uint32() + reader.pos;
                                        while (reader.pos < end2)
                                            message.floats.push(reader.float());
                                    } else
                                        message.floats.push(reader.float());
                                    break;
                                case 8:
                                    if (!(message.strings && message.strings.length))
                                        message.strings = [];
                                    message.strings.push(reader.string());
                                    break;
                                case 10:
                                    message.b = reader.bool();
                                    break;
                                case 11:
                                    if (!(message.bools && message.bools.length))
                                        message.bools = [];
                                    if ((tag & 7) === 2) {
                                        var end2 = reader.uint32() + reader.pos;
                                        while (reader.pos < end2)
                                            message.bools.push(reader.bool());
                                    } else
                                        message.bools.push(reader.bool());
                                    break;
                                case 12:
                                    message.block_idx = reader.int32();
                                    break;
                                case 13:
                                    message.l = reader.int64();
                                    break;
                                case 14:
                                    if (!(message.blocks_idx && message.blocks_idx.length))
                                        message.blocks_idx = [];
                                    if ((tag & 7) === 2) {
                                        var end2 = reader.uint32() + reader.pos;
                                        while (reader.pos < end2)
                                            message.blocks_idx.push(reader.int32());
                                    } else
                                        message.blocks_idx.push(reader.int32());
                                    break;
                                case 15:
                                    if (!(message.longs && message.longs.length))
                                        message.longs = [];
                                    if ((tag & 7) === 2) {
                                        var end2 = reader.uint32() + reader.pos;
                                        while (reader.pos < end2)
                                            message.longs.push(reader.int64());
                                    } else
                                        message.longs.push(reader.int64());
                                    break;
                                default:
                                    reader.skipType(tag & 7);
                                    break;
                                }
                            }
                            if (!message.hasOwnProperty("name"))
                                throw $util.ProtocolError("missing required 'name'", { instance: message });
                            if (!message.hasOwnProperty("type"))
                                throw $util.ProtocolError("missing required 'type'", { instance: message });
                            return message;
                        };
    
                        return Attr;
                    })();
    
                    OpDesc.Var = (function() {
    
                        function Var(properties) {
                            this["arguments"] = [];
                            if (properties)
                                for (var keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                                    if (properties[keys[i]] != null)
                                        this[keys[i]] = properties[keys[i]];
                        }
    
                        Var.prototype.parameter = "";
                        Var.prototype["arguments"] = $util.emptyArray;
    
                        Var.decode = function decode(reader, length) {
                            if (!(reader instanceof $Reader))
                                reader = $Reader.create(reader);
                            var end = length === undefined ? reader.len : reader.pos + length, message = new $root.paddle.framework.proto.OpDesc.Var();
                            while (reader.pos < end) {
                                var tag = reader.uint32();
                                switch (tag >>> 3) {
                                case 1:
                                    message.parameter = reader.string();
                                    break;
                                case 2:
                                    if (!(message["arguments"] && message["arguments"].length))
                                        message["arguments"] = [];
                                    message["arguments"].push(reader.string());
                                    break;
                                default:
                                    reader.skipType(tag & 7);
                                    break;
                                }
                            }
                            if (!message.hasOwnProperty("parameter"))
                                throw $util.ProtocolError("missing required 'parameter'", { instance: message });
                            return message;
                        };
    
                        return Var;
                    })();
    
                    return OpDesc;
                })();
    
                proto.OpProto = (function() {
    
                    function OpProto(properties) {
                        this.inputs = [];
                        this.outputs = [];
                        this.attrs = [];
                        if (properties)
                            for (var keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                                if (properties[keys[i]] != null)
                                    this[keys[i]] = properties[keys[i]];
                    }
    
                    OpProto.prototype.type = "";
                    OpProto.prototype.inputs = $util.emptyArray;
                    OpProto.prototype.outputs = $util.emptyArray;
                    OpProto.prototype.attrs = $util.emptyArray;
                    OpProto.prototype.comment = "";
    
                    OpProto.decode = function decode(reader, length) {
                        if (!(reader instanceof $Reader))
                            reader = $Reader.create(reader);
                        var end = length === undefined ? reader.len : reader.pos + length, message = new $root.paddle.framework.proto.OpProto();
                        while (reader.pos < end) {
                            var tag = reader.uint32();
                            switch (tag >>> 3) {
                            case 1:
                                message.type = reader.string();
                                break;
                            case 2:
                                if (!(message.inputs && message.inputs.length))
                                    message.inputs = [];
                                message.inputs.push($root.paddle.framework.proto.OpProto.Var.decode(reader, reader.uint32()));
                                break;
                            case 3:
                                if (!(message.outputs && message.outputs.length))
                                    message.outputs = [];
                                message.outputs.push($root.paddle.framework.proto.OpProto.Var.decode(reader, reader.uint32()));
                                break;
                            case 4:
                                if (!(message.attrs && message.attrs.length))
                                    message.attrs = [];
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
                        if (!message.hasOwnProperty("type"))
                            throw $util.ProtocolError("missing required 'type'", { instance: message });
                        if (!message.hasOwnProperty("comment"))
                            throw $util.ProtocolError("missing required 'comment'", { instance: message });
                        return message;
                    };
    
                    OpProto.Var = (function() {
    
                        function Var(properties) {
                            if (properties)
                                for (var keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                                    if (properties[keys[i]] != null)
                                        this[keys[i]] = properties[keys[i]];
                        }
    
                        Var.prototype.name = "";
                        Var.prototype.comment = "";
                        Var.prototype.duplicable = false;
                        Var.prototype.intermediate = false;
                        Var.prototype.dispensable = false;
    
                        Var.decode = function decode(reader, length) {
                            if (!(reader instanceof $Reader))
                                reader = $Reader.create(reader);
                            var end = length === undefined ? reader.len : reader.pos + length, message = new $root.paddle.framework.proto.OpProto.Var();
                            while (reader.pos < end) {
                                var tag = reader.uint32();
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
                            if (!message.hasOwnProperty("name"))
                                throw $util.ProtocolError("missing required 'name'", { instance: message });
                            if (!message.hasOwnProperty("comment"))
                                throw $util.ProtocolError("missing required 'comment'", { instance: message });
                            return message;
                        };
    
                        return Var;
                    })();
    
                    OpProto.Attr = (function() {
    
                        function Attr(properties) {
                            if (properties)
                                for (var keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                                    if (properties[keys[i]] != null)
                                        this[keys[i]] = properties[keys[i]];
                        }
    
                        Attr.prototype.name = "";
                        Attr.prototype.type = 0;
                        Attr.prototype.comment = "";
                        Attr.prototype.generated = false;
    
                        Attr.decode = function decode(reader, length) {
                            if (!(reader instanceof $Reader))
                                reader = $Reader.create(reader);
                            var end = length === undefined ? reader.len : reader.pos + length, message = new $root.paddle.framework.proto.OpProto.Attr();
                            while (reader.pos < end) {
                                var tag = reader.uint32();
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
                            if (!message.hasOwnProperty("name"))
                                throw $util.ProtocolError("missing required 'name'", { instance: message });
                            if (!message.hasOwnProperty("type"))
                                throw $util.ProtocolError("missing required 'type'", { instance: message });
                            if (!message.hasOwnProperty("comment"))
                                throw $util.ProtocolError("missing required 'comment'", { instance: message });
                            return message;
                        };
    
                        return Attr;
                    })();
    
                    return OpProto;
                })();
    
                proto.VarType = (function() {
    
                    function VarType(properties) {
                        if (properties)
                            for (var keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                                if (properties[keys[i]] != null)
                                    this[keys[i]] = properties[keys[i]];
                    }
    
                    VarType.prototype.type = 0;
                    VarType.prototype.selected_rows = null;
                    VarType.prototype.lod_tensor = null;
                    VarType.prototype.tensor_array = null;
                    VarType.prototype.reader = null;
                    VarType.prototype.tuple = null;
    
                    VarType.decode = function decode(reader, length) {
                        if (!(reader instanceof $Reader))
                            reader = $Reader.create(reader);
                        var end = length === undefined ? reader.len : reader.pos + length, message = new $root.paddle.framework.proto.VarType();
                        while (reader.pos < end) {
                            var tag = reader.uint32();
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
                        if (!message.hasOwnProperty("type"))
                            throw $util.ProtocolError("missing required 'type'", { instance: message });
                        return message;
                    };
    
                    VarType.Type = (function() {
                        var valuesById = {}, values = Object.create(valuesById);
                        values[valuesById[0] = "BOOL"] = 0;
                        values[valuesById[1] = "INT16"] = 1;
                        values[valuesById[2] = "INT32"] = 2;
                        values[valuesById[3] = "INT64"] = 3;
                        values[valuesById[4] = "FP16"] = 4;
                        values[valuesById[5] = "FP32"] = 5;
                        values[valuesById[6] = "FP64"] = 6;
                        values[valuesById[19] = "SIZE_T"] = 19;
                        values[valuesById[20] = "UINT8"] = 20;
                        values[valuesById[21] = "INT8"] = 21;
                        values[valuesById[7] = "LOD_TENSOR"] = 7;
                        values[valuesById[8] = "SELECTED_ROWS"] = 8;
                        values[valuesById[9] = "FEED_MINIBATCH"] = 9;
                        values[valuesById[10] = "FETCH_LIST"] = 10;
                        values[valuesById[11] = "STEP_SCOPES"] = 11;
                        values[valuesById[12] = "LOD_RANK_TABLE"] = 12;
                        values[valuesById[13] = "LOD_TENSOR_ARRAY"] = 13;
                        values[valuesById[14] = "PLACE_LIST"] = 14;
                        values[valuesById[15] = "READER"] = 15;
                        values[valuesById[17] = "RAW"] = 17;
                        values[valuesById[18] = "TUPLE"] = 18;
                        return values;
                    })();
    
                    VarType.TensorDesc = (function() {
    
                        function TensorDesc(properties) {
                            this.dims = [];
                            if (properties)
                                for (var keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                                    if (properties[keys[i]] != null)
                                        this[keys[i]] = properties[keys[i]];
                        }
    
                        TensorDesc.prototype.data_type = 0;
                        TensorDesc.prototype.dims = $util.emptyArray;
    
                        TensorDesc.decode = function decode(reader, length) {
                            if (!(reader instanceof $Reader))
                                reader = $Reader.create(reader);
                            var end = length === undefined ? reader.len : reader.pos + length, message = new $root.paddle.framework.proto.VarType.TensorDesc();
                            while (reader.pos < end) {
                                var tag = reader.uint32();
                                switch (tag >>> 3) {
                                case 1:
                                    message.data_type = reader.int32();
                                    break;
                                case 2:
                                    if (!(message.dims && message.dims.length))
                                        message.dims = [];
                                    if ((tag & 7) === 2) {
                                        var end2 = reader.uint32() + reader.pos;
                                        while (reader.pos < end2)
                                            message.dims.push(reader.int64());
                                    } else
                                        message.dims.push(reader.int64());
                                    break;
                                default:
                                    reader.skipType(tag & 7);
                                    break;
                                }
                            }
                            if (!message.hasOwnProperty("data_type"))
                                throw $util.ProtocolError("missing required 'data_type'", { instance: message });
                            return message;
                        };
    
                        return TensorDesc;
                    })();
    
                    VarType.LoDTensorDesc = (function() {
    
                        function LoDTensorDesc(properties) {
                            if (properties)
                                for (var keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                                    if (properties[keys[i]] != null)
                                        this[keys[i]] = properties[keys[i]];
                        }
    
                        LoDTensorDesc.prototype.tensor = null;
                        LoDTensorDesc.prototype.lod_level = 0;
    
                        LoDTensorDesc.decode = function decode(reader, length) {
                            if (!(reader instanceof $Reader))
                                reader = $Reader.create(reader);
                            var end = length === undefined ? reader.len : reader.pos + length, message = new $root.paddle.framework.proto.VarType.LoDTensorDesc();
                            while (reader.pos < end) {
                                var tag = reader.uint32();
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
                            if (!message.hasOwnProperty("tensor"))
                                throw $util.ProtocolError("missing required 'tensor'", { instance: message });
                            return message;
                        };
    
                        return LoDTensorDesc;
                    })();
    
                    VarType.LoDTensorArrayDesc = (function() {
    
                        function LoDTensorArrayDesc(properties) {
                            if (properties)
                                for (var keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                                    if (properties[keys[i]] != null)
                                        this[keys[i]] = properties[keys[i]];
                        }
    
                        LoDTensorArrayDesc.prototype.tensor = null;
                        LoDTensorArrayDesc.prototype.lod_level = 0;
    
                        LoDTensorArrayDesc.decode = function decode(reader, length) {
                            if (!(reader instanceof $Reader))
                                reader = $Reader.create(reader);
                            var end = length === undefined ? reader.len : reader.pos + length, message = new $root.paddle.framework.proto.VarType.LoDTensorArrayDesc();
                            while (reader.pos < end) {
                                var tag = reader.uint32();
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
                            if (!message.hasOwnProperty("tensor"))
                                throw $util.ProtocolError("missing required 'tensor'", { instance: message });
                            return message;
                        };
    
                        return LoDTensorArrayDesc;
                    })();
    
                    VarType.ReaderDesc = (function() {
    
                        function ReaderDesc(properties) {
                            this.lod_tensor = [];
                            if (properties)
                                for (var keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                                    if (properties[keys[i]] != null)
                                        this[keys[i]] = properties[keys[i]];
                        }
    
                        ReaderDesc.prototype.lod_tensor = $util.emptyArray;
    
                        ReaderDesc.decode = function decode(reader, length) {
                            if (!(reader instanceof $Reader))
                                reader = $Reader.create(reader);
                            var end = length === undefined ? reader.len : reader.pos + length, message = new $root.paddle.framework.proto.VarType.ReaderDesc();
                            while (reader.pos < end) {
                                var tag = reader.uint32();
                                switch (tag >>> 3) {
                                case 1:
                                    if (!(message.lod_tensor && message.lod_tensor.length))
                                        message.lod_tensor = [];
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
    
                        function Tuple(properties) {
                            this.element_type = [];
                            if (properties)
                                for (var keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                                    if (properties[keys[i]] != null)
                                        this[keys[i]] = properties[keys[i]];
                        }
    
                        Tuple.prototype.element_type = $util.emptyArray;
    
                        Tuple.decode = function decode(reader, length) {
                            if (!(reader instanceof $Reader))
                                reader = $Reader.create(reader);
                            var end = length === undefined ? reader.len : reader.pos + length, message = new $root.paddle.framework.proto.VarType.Tuple();
                            while (reader.pos < end) {
                                var tag = reader.uint32();
                                switch (tag >>> 3) {
                                case 1:
                                    if (!(message.element_type && message.element_type.length))
                                        message.element_type = [];
                                    if ((tag & 7) === 2) {
                                        var end2 = reader.uint32() + reader.pos;
                                        while (reader.pos < end2)
                                            message.element_type.push(reader.int32());
                                    } else
                                        message.element_type.push(reader.int32());
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
    
                    function VarDesc(properties) {
                        if (properties)
                            for (var keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                                if (properties[keys[i]] != null)
                                    this[keys[i]] = properties[keys[i]];
                    }
    
                    VarDesc.prototype.name = "";
                    VarDesc.prototype.type = null;
                    VarDesc.prototype.persistable = false;
    
                    VarDesc.decode = function decode(reader, length) {
                        if (!(reader instanceof $Reader))
                            reader = $Reader.create(reader);
                        var end = length === undefined ? reader.len : reader.pos + length, message = new $root.paddle.framework.proto.VarDesc();
                        while (reader.pos < end) {
                            var tag = reader.uint32();
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
                            default:
                                reader.skipType(tag & 7);
                                break;
                            }
                        }
                        if (!message.hasOwnProperty("name"))
                            throw $util.ProtocolError("missing required 'name'", { instance: message });
                        if (!message.hasOwnProperty("type"))
                            throw $util.ProtocolError("missing required 'type'", { instance: message });
                        return message;
                    };
    
                    return VarDesc;
                })();
    
                proto.BlockDesc = (function() {
    
                    function BlockDesc(properties) {
                        this.vars = [];
                        this.ops = [];
                        if (properties)
                            for (var keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                                if (properties[keys[i]] != null)
                                    this[keys[i]] = properties[keys[i]];
                    }
    
                    BlockDesc.prototype.idx = 0;
                    BlockDesc.prototype.parent_idx = 0;
                    BlockDesc.prototype.vars = $util.emptyArray;
                    BlockDesc.prototype.ops = $util.emptyArray;
                    BlockDesc.prototype.forward_block_idx = -1;
    
                    BlockDesc.decode = function decode(reader, length) {
                        if (!(reader instanceof $Reader))
                            reader = $Reader.create(reader);
                        var end = length === undefined ? reader.len : reader.pos + length, message = new $root.paddle.framework.proto.BlockDesc();
                        while (reader.pos < end) {
                            var tag = reader.uint32();
                            switch (tag >>> 3) {
                            case 1:
                                message.idx = reader.int32();
                                break;
                            case 2:
                                message.parent_idx = reader.int32();
                                break;
                            case 3:
                                if (!(message.vars && message.vars.length))
                                    message.vars = [];
                                message.vars.push($root.paddle.framework.proto.VarDesc.decode(reader, reader.uint32()));
                                break;
                            case 4:
                                if (!(message.ops && message.ops.length))
                                    message.ops = [];
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
                        if (!message.hasOwnProperty("idx"))
                            throw $util.ProtocolError("missing required 'idx'", { instance: message });
                        if (!message.hasOwnProperty("parent_idx"))
                            throw $util.ProtocolError("missing required 'parent_idx'", { instance: message });
                        return message;
                    };
    
                    return BlockDesc;
                })();
    
                proto.ProgramDesc = (function() {
    
                    function ProgramDesc(properties) {
                        this.blocks = [];
                        if (properties)
                            for (var keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                                if (properties[keys[i]] != null)
                                    this[keys[i]] = properties[keys[i]];
                    }
    
                    ProgramDesc.prototype.blocks = $util.emptyArray;
                    ProgramDesc.prototype.version = null;
    
                    ProgramDesc.decode = function decode(reader, length) {
                        if (!(reader instanceof $Reader))
                            reader = $Reader.create(reader);
                        var end = length === undefined ? reader.len : reader.pos + length, message = new $root.paddle.framework.proto.ProgramDesc();
                        while (reader.pos < end) {
                            var tag = reader.uint32();
                            switch (tag >>> 3) {
                            case 1:
                                if (!(message.blocks && message.blocks.length))
                                    message.blocks = [];
                                message.blocks.push($root.paddle.framework.proto.BlockDesc.decode(reader, reader.uint32()));
                                break;
                            case 2:
                                message.version = $root.paddle.framework.proto.Version.decode(reader, reader.uint32());
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
