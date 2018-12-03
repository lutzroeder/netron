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
    
                    Version.create = function create(properties) {
                        return new Version(properties);
                    };
    
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
    
                    Version.verify = function verify(message) {
                        if (typeof message !== "object" || message === null)
                            return "object expected";
                        if (message.version != null && message.hasOwnProperty("version"))
                            if (!$util.isInteger(message.version) && !(message.version && $util.isInteger(message.version.low) && $util.isInteger(message.version.high)))
                                return "version: integer|Long expected";
                        return null;
                    };
    
                    Version.fromObject = function fromObject(object) {
                        if (object instanceof $root.paddle.framework.proto.Version)
                            return object;
                        var message = new $root.paddle.framework.proto.Version();
                        if (object.version != null)
                            if ($util.Long)
                                (message.version = $util.Long.fromValue(object.version)).unsigned = false;
                            else if (typeof object.version === "string")
                                message.version = parseInt(object.version, 10);
                            else if (typeof object.version === "number")
                                message.version = object.version;
                            else if (typeof object.version === "object")
                                message.version = new $util.LongBits(object.version.low >>> 0, object.version.high >>> 0).toNumber();
                        return message;
                    };
    
                    Version.toObject = function toObject(message, options) {
                        if (!options)
                            options = {};
                        var object = {};
                        if (options.defaults)
                            if ($util.Long) {
                                var long = new $util.Long(0, 0, false);
                                object.version = options.longs === String ? long.toString() : options.longs === Number ? long.toNumber() : long;
                            } else
                                object.version = options.longs === String ? "0" : 0;
                        if (message.version != null && message.hasOwnProperty("version"))
                            if (typeof message.version === "number")
                                object.version = options.longs === String ? String(message.version) : message.version;
                            else
                                object.version = options.longs === String ? $util.Long.prototype.toString.call(message.version) : options.longs === Number ? new $util.LongBits(message.version.low >>> 0, message.version.high >>> 0).toNumber() : message.version;
                        return object;
                    };
    
                    Version.prototype.toJSON = function toJSON() {
                        return this.constructor.toObject(this, $protobuf.util.toJSONOptions);
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
    
                    OpDesc.create = function create(properties) {
                        return new OpDesc(properties);
                    };
    
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
    
                    OpDesc.verify = function verify(message) {
                        if (typeof message !== "object" || message === null)
                            return "object expected";
                        if (!$util.isString(message.type))
                            return "type: string expected";
                        if (message.inputs != null && message.hasOwnProperty("inputs")) {
                            if (!Array.isArray(message.inputs))
                                return "inputs: array expected";
                            for (var i = 0; i < message.inputs.length; ++i) {
                                var error = $root.paddle.framework.proto.OpDesc.Var.verify(message.inputs[i]);
                                if (error)
                                    return "inputs." + error;
                            }
                        }
                        if (message.outputs != null && message.hasOwnProperty("outputs")) {
                            if (!Array.isArray(message.outputs))
                                return "outputs: array expected";
                            for (var i = 0; i < message.outputs.length; ++i) {
                                var error = $root.paddle.framework.proto.OpDesc.Var.verify(message.outputs[i]);
                                if (error)
                                    return "outputs." + error;
                            }
                        }
                        if (message.attrs != null && message.hasOwnProperty("attrs")) {
                            if (!Array.isArray(message.attrs))
                                return "attrs: array expected";
                            for (var i = 0; i < message.attrs.length; ++i) {
                                var error = $root.paddle.framework.proto.OpDesc.Attr.verify(message.attrs[i]);
                                if (error)
                                    return "attrs." + error;
                            }
                        }
                        if (message.is_target != null && message.hasOwnProperty("is_target"))
                            if (typeof message.is_target !== "boolean")
                                return "is_target: boolean expected";
                        return null;
                    };
    
                    OpDesc.fromObject = function fromObject(object) {
                        if (object instanceof $root.paddle.framework.proto.OpDesc)
                            return object;
                        var message = new $root.paddle.framework.proto.OpDesc();
                        if (object.type != null)
                            message.type = String(object.type);
                        if (object.inputs) {
                            if (!Array.isArray(object.inputs))
                                throw TypeError(".paddle.framework.proto.OpDesc.inputs: array expected");
                            message.inputs = [];
                            for (var i = 0; i < object.inputs.length; ++i) {
                                if (typeof object.inputs[i] !== "object")
                                    throw TypeError(".paddle.framework.proto.OpDesc.inputs: object expected");
                                message.inputs[i] = $root.paddle.framework.proto.OpDesc.Var.fromObject(object.inputs[i]);
                            }
                        }
                        if (object.outputs) {
                            if (!Array.isArray(object.outputs))
                                throw TypeError(".paddle.framework.proto.OpDesc.outputs: array expected");
                            message.outputs = [];
                            for (var i = 0; i < object.outputs.length; ++i) {
                                if (typeof object.outputs[i] !== "object")
                                    throw TypeError(".paddle.framework.proto.OpDesc.outputs: object expected");
                                message.outputs[i] = $root.paddle.framework.proto.OpDesc.Var.fromObject(object.outputs[i]);
                            }
                        }
                        if (object.attrs) {
                            if (!Array.isArray(object.attrs))
                                throw TypeError(".paddle.framework.proto.OpDesc.attrs: array expected");
                            message.attrs = [];
                            for (var i = 0; i < object.attrs.length; ++i) {
                                if (typeof object.attrs[i] !== "object")
                                    throw TypeError(".paddle.framework.proto.OpDesc.attrs: object expected");
                                message.attrs[i] = $root.paddle.framework.proto.OpDesc.Attr.fromObject(object.attrs[i]);
                            }
                        }
                        if (object.is_target != null)
                            message.is_target = Boolean(object.is_target);
                        return message;
                    };
    
                    OpDesc.toObject = function toObject(message, options) {
                        if (!options)
                            options = {};
                        var object = {};
                        if (options.arrays || options.defaults) {
                            object.inputs = [];
                            object.outputs = [];
                            object.attrs = [];
                        }
                        if (options.defaults) {
                            object.type = "";
                            object.is_target = false;
                        }
                        if (message.inputs && message.inputs.length) {
                            object.inputs = [];
                            for (var j = 0; j < message.inputs.length; ++j)
                                object.inputs[j] = $root.paddle.framework.proto.OpDesc.Var.toObject(message.inputs[j], options);
                        }
                        if (message.outputs && message.outputs.length) {
                            object.outputs = [];
                            for (var j = 0; j < message.outputs.length; ++j)
                                object.outputs[j] = $root.paddle.framework.proto.OpDesc.Var.toObject(message.outputs[j], options);
                        }
                        if (message.type != null && message.hasOwnProperty("type"))
                            object.type = message.type;
                        if (message.attrs && message.attrs.length) {
                            object.attrs = [];
                            for (var j = 0; j < message.attrs.length; ++j)
                                object.attrs[j] = $root.paddle.framework.proto.OpDesc.Attr.toObject(message.attrs[j], options);
                        }
                        if (message.is_target != null && message.hasOwnProperty("is_target"))
                            object.is_target = message.is_target;
                        return object;
                    };
    
                    OpDesc.prototype.toJSON = function toJSON() {
                        return this.constructor.toObject(this, $protobuf.util.toJSONOptions);
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
    
                        Attr.create = function create(properties) {
                            return new Attr(properties);
                        };
    
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
    
                        Attr.verify = function verify(message) {
                            if (typeof message !== "object" || message === null)
                                return "object expected";
                            if (!$util.isString(message.name))
                                return "name: string expected";
                            switch (message.type) {
                            default:
                                return "type: enum value expected";
                            case 0:
                            case 1:
                            case 2:
                            case 3:
                            case 4:
                            case 5:
                            case 6:
                            case 7:
                            case 8:
                            case 9:
                            case 10:
                            case 11:
                                break;
                            }
                            if (message.i != null && message.hasOwnProperty("i"))
                                if (!$util.isInteger(message.i))
                                    return "i: integer expected";
                            if (message.f != null && message.hasOwnProperty("f"))
                                if (typeof message.f !== "number")
                                    return "f: number expected";
                            if (message.s != null && message.hasOwnProperty("s"))
                                if (!$util.isString(message.s))
                                    return "s: string expected";
                            if (message.ints != null && message.hasOwnProperty("ints")) {
                                if (!Array.isArray(message.ints))
                                    return "ints: array expected";
                                for (var i = 0; i < message.ints.length; ++i)
                                    if (!$util.isInteger(message.ints[i]))
                                        return "ints: integer[] expected";
                            }
                            if (message.floats != null && message.hasOwnProperty("floats")) {
                                if (!Array.isArray(message.floats))
                                    return "floats: array expected";
                                for (var i = 0; i < message.floats.length; ++i)
                                    if (typeof message.floats[i] !== "number")
                                        return "floats: number[] expected";
                            }
                            if (message.strings != null && message.hasOwnProperty("strings")) {
                                if (!Array.isArray(message.strings))
                                    return "strings: array expected";
                                for (var i = 0; i < message.strings.length; ++i)
                                    if (!$util.isString(message.strings[i]))
                                        return "strings: string[] expected";
                            }
                            if (message.b != null && message.hasOwnProperty("b"))
                                if (typeof message.b !== "boolean")
                                    return "b: boolean expected";
                            if (message.bools != null && message.hasOwnProperty("bools")) {
                                if (!Array.isArray(message.bools))
                                    return "bools: array expected";
                                for (var i = 0; i < message.bools.length; ++i)
                                    if (typeof message.bools[i] !== "boolean")
                                        return "bools: boolean[] expected";
                            }
                            if (message.block_idx != null && message.hasOwnProperty("block_idx"))
                                if (!$util.isInteger(message.block_idx))
                                    return "block_idx: integer expected";
                            if (message.l != null && message.hasOwnProperty("l"))
                                if (!$util.isInteger(message.l) && !(message.l && $util.isInteger(message.l.low) && $util.isInteger(message.l.high)))
                                    return "l: integer|Long expected";
                            if (message.blocks_idx != null && message.hasOwnProperty("blocks_idx")) {
                                if (!Array.isArray(message.blocks_idx))
                                    return "blocks_idx: array expected";
                                for (var i = 0; i < message.blocks_idx.length; ++i)
                                    if (!$util.isInteger(message.blocks_idx[i]))
                                        return "blocks_idx: integer[] expected";
                            }
                            if (message.longs != null && message.hasOwnProperty("longs")) {
                                if (!Array.isArray(message.longs))
                                    return "longs: array expected";
                                for (var i = 0; i < message.longs.length; ++i)
                                    if (!$util.isInteger(message.longs[i]) && !(message.longs[i] && $util.isInteger(message.longs[i].low) && $util.isInteger(message.longs[i].high)))
                                        return "longs: integer|Long[] expected";
                            }
                            return null;
                        };
    
                        Attr.fromObject = function fromObject(object) {
                            if (object instanceof $root.paddle.framework.proto.OpDesc.Attr)
                                return object;
                            var message = new $root.paddle.framework.proto.OpDesc.Attr();
                            if (object.name != null)
                                message.name = String(object.name);
                            switch (object.type) {
                            case "INT":
                            case 0:
                                message.type = 0;
                                break;
                            case "FLOAT":
                            case 1:
                                message.type = 1;
                                break;
                            case "STRING":
                            case 2:
                                message.type = 2;
                                break;
                            case "INTS":
                            case 3:
                                message.type = 3;
                                break;
                            case "FLOATS":
                            case 4:
                                message.type = 4;
                                break;
                            case "STRINGS":
                            case 5:
                                message.type = 5;
                                break;
                            case "BOOLEAN":
                            case 6:
                                message.type = 6;
                                break;
                            case "BOOLEANS":
                            case 7:
                                message.type = 7;
                                break;
                            case "BLOCK":
                            case 8:
                                message.type = 8;
                                break;
                            case "LONG":
                            case 9:
                                message.type = 9;
                                break;
                            case "BLOCKS":
                            case 10:
                                message.type = 10;
                                break;
                            case "LONGS":
                            case 11:
                                message.type = 11;
                                break;
                            }
                            if (object.i != null)
                                message.i = object.i | 0;
                            if (object.f != null)
                                message.f = Number(object.f);
                            if (object.s != null)
                                message.s = String(object.s);
                            if (object.ints) {
                                if (!Array.isArray(object.ints))
                                    throw TypeError(".paddle.framework.proto.OpDesc.Attr.ints: array expected");
                                message.ints = [];
                                for (var i = 0; i < object.ints.length; ++i)
                                    message.ints[i] = object.ints[i] | 0;
                            }
                            if (object.floats) {
                                if (!Array.isArray(object.floats))
                                    throw TypeError(".paddle.framework.proto.OpDesc.Attr.floats: array expected");
                                message.floats = [];
                                for (var i = 0; i < object.floats.length; ++i)
                                    message.floats[i] = Number(object.floats[i]);
                            }
                            if (object.strings) {
                                if (!Array.isArray(object.strings))
                                    throw TypeError(".paddle.framework.proto.OpDesc.Attr.strings: array expected");
                                message.strings = [];
                                for (var i = 0; i < object.strings.length; ++i)
                                    message.strings[i] = String(object.strings[i]);
                            }
                            if (object.b != null)
                                message.b = Boolean(object.b);
                            if (object.bools) {
                                if (!Array.isArray(object.bools))
                                    throw TypeError(".paddle.framework.proto.OpDesc.Attr.bools: array expected");
                                message.bools = [];
                                for (var i = 0; i < object.bools.length; ++i)
                                    message.bools[i] = Boolean(object.bools[i]);
                            }
                            if (object.block_idx != null)
                                message.block_idx = object.block_idx | 0;
                            if (object.l != null)
                                if ($util.Long)
                                    (message.l = $util.Long.fromValue(object.l)).unsigned = false;
                                else if (typeof object.l === "string")
                                    message.l = parseInt(object.l, 10);
                                else if (typeof object.l === "number")
                                    message.l = object.l;
                                else if (typeof object.l === "object")
                                    message.l = new $util.LongBits(object.l.low >>> 0, object.l.high >>> 0).toNumber();
                            if (object.blocks_idx) {
                                if (!Array.isArray(object.blocks_idx))
                                    throw TypeError(".paddle.framework.proto.OpDesc.Attr.blocks_idx: array expected");
                                message.blocks_idx = [];
                                for (var i = 0; i < object.blocks_idx.length; ++i)
                                    message.blocks_idx[i] = object.blocks_idx[i] | 0;
                            }
                            if (object.longs) {
                                if (!Array.isArray(object.longs))
                                    throw TypeError(".paddle.framework.proto.OpDesc.Attr.longs: array expected");
                                message.longs = [];
                                for (var i = 0; i < object.longs.length; ++i)
                                    if ($util.Long)
                                        (message.longs[i] = $util.Long.fromValue(object.longs[i])).unsigned = false;
                                    else if (typeof object.longs[i] === "string")
                                        message.longs[i] = parseInt(object.longs[i], 10);
                                    else if (typeof object.longs[i] === "number")
                                        message.longs[i] = object.longs[i];
                                    else if (typeof object.longs[i] === "object")
                                        message.longs[i] = new $util.LongBits(object.longs[i].low >>> 0, object.longs[i].high >>> 0).toNumber();
                            }
                            return message;
                        };
    
                        Attr.toObject = function toObject(message, options) {
                            if (!options)
                                options = {};
                            var object = {};
                            if (options.arrays || options.defaults) {
                                object.ints = [];
                                object.floats = [];
                                object.strings = [];
                                object.bools = [];
                                object.blocks_idx = [];
                                object.longs = [];
                            }
                            if (options.defaults) {
                                object.name = "";
                                object.type = options.enums === String ? "INT" : 0;
                                object.i = 0;
                                object.f = 0;
                                object.s = "";
                                object.b = false;
                                object.block_idx = 0;
                                if ($util.Long) {
                                    var long = new $util.Long(0, 0, false);
                                    object.l = options.longs === String ? long.toString() : options.longs === Number ? long.toNumber() : long;
                                } else
                                    object.l = options.longs === String ? "0" : 0;
                            }
                            if (message.name != null && message.hasOwnProperty("name"))
                                object.name = message.name;
                            if (message.type != null && message.hasOwnProperty("type"))
                                object.type = options.enums === String ? $root.paddle.framework.proto.AttrType[message.type] : message.type;
                            if (message.i != null && message.hasOwnProperty("i"))
                                object.i = message.i;
                            if (message.f != null && message.hasOwnProperty("f"))
                                object.f = options.json && !isFinite(message.f) ? String(message.f) : message.f;
                            if (message.s != null && message.hasOwnProperty("s"))
                                object.s = message.s;
                            if (message.ints && message.ints.length) {
                                object.ints = [];
                                for (var j = 0; j < message.ints.length; ++j)
                                    object.ints[j] = message.ints[j];
                            }
                            if (message.floats && message.floats.length) {
                                object.floats = [];
                                for (var j = 0; j < message.floats.length; ++j)
                                    object.floats[j] = options.json && !isFinite(message.floats[j]) ? String(message.floats[j]) : message.floats[j];
                            }
                            if (message.strings && message.strings.length) {
                                object.strings = [];
                                for (var j = 0; j < message.strings.length; ++j)
                                    object.strings[j] = message.strings[j];
                            }
                            if (message.b != null && message.hasOwnProperty("b"))
                                object.b = message.b;
                            if (message.bools && message.bools.length) {
                                object.bools = [];
                                for (var j = 0; j < message.bools.length; ++j)
                                    object.bools[j] = message.bools[j];
                            }
                            if (message.block_idx != null && message.hasOwnProperty("block_idx"))
                                object.block_idx = message.block_idx;
                            if (message.l != null && message.hasOwnProperty("l"))
                                if (typeof message.l === "number")
                                    object.l = options.longs === String ? String(message.l) : message.l;
                                else
                                    object.l = options.longs === String ? $util.Long.prototype.toString.call(message.l) : options.longs === Number ? new $util.LongBits(message.l.low >>> 0, message.l.high >>> 0).toNumber() : message.l;
                            if (message.blocks_idx && message.blocks_idx.length) {
                                object.blocks_idx = [];
                                for (var j = 0; j < message.blocks_idx.length; ++j)
                                    object.blocks_idx[j] = message.blocks_idx[j];
                            }
                            if (message.longs && message.longs.length) {
                                object.longs = [];
                                for (var j = 0; j < message.longs.length; ++j)
                                    if (typeof message.longs[j] === "number")
                                        object.longs[j] = options.longs === String ? String(message.longs[j]) : message.longs[j];
                                    else
                                        object.longs[j] = options.longs === String ? $util.Long.prototype.toString.call(message.longs[j]) : options.longs === Number ? new $util.LongBits(message.longs[j].low >>> 0, message.longs[j].high >>> 0).toNumber() : message.longs[j];
                            }
                            return object;
                        };
    
                        Attr.prototype.toJSON = function toJSON() {
                            return this.constructor.toObject(this, $protobuf.util.toJSONOptions);
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
    
                        Var.create = function create(properties) {
                            return new Var(properties);
                        };
    
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
    
                        Var.verify = function verify(message) {
                            if (typeof message !== "object" || message === null)
                                return "object expected";
                            if (!$util.isString(message.parameter))
                                return "parameter: string expected";
                            if (message["arguments"] != null && message.hasOwnProperty("arguments")) {
                                if (!Array.isArray(message["arguments"]))
                                    return "arguments: array expected";
                                for (var i = 0; i < message["arguments"].length; ++i)
                                    if (!$util.isString(message["arguments"][i]))
                                        return "arguments: string[] expected";
                            }
                            return null;
                        };
    
                        Var.fromObject = function fromObject(object) {
                            if (object instanceof $root.paddle.framework.proto.OpDesc.Var)
                                return object;
                            var message = new $root.paddle.framework.proto.OpDesc.Var();
                            if (object.parameter != null)
                                message.parameter = String(object.parameter);
                            if (object["arguments"]) {
                                if (!Array.isArray(object["arguments"]))
                                    throw TypeError(".paddle.framework.proto.OpDesc.Var.arguments: array expected");
                                message["arguments"] = [];
                                for (var i = 0; i < object["arguments"].length; ++i)
                                    message["arguments"][i] = String(object["arguments"][i]);
                            }
                            return message;
                        };
    
                        Var.toObject = function toObject(message, options) {
                            if (!options)
                                options = {};
                            var object = {};
                            if (options.arrays || options.defaults)
                                object["arguments"] = [];
                            if (options.defaults)
                                object.parameter = "";
                            if (message.parameter != null && message.hasOwnProperty("parameter"))
                                object.parameter = message.parameter;
                            if (message["arguments"] && message["arguments"].length) {
                                object["arguments"] = [];
                                for (var j = 0; j < message["arguments"].length; ++j)
                                    object["arguments"][j] = message["arguments"][j];
                            }
                            return object;
                        };
    
                        Var.prototype.toJSON = function toJSON() {
                            return this.constructor.toObject(this, $protobuf.util.toJSONOptions);
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
    
                    OpProto.create = function create(properties) {
                        return new OpProto(properties);
                    };
    
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
    
                    OpProto.verify = function verify(message) {
                        if (typeof message !== "object" || message === null)
                            return "object expected";
                        if (!$util.isString(message.type))
                            return "type: string expected";
                        if (message.inputs != null && message.hasOwnProperty("inputs")) {
                            if (!Array.isArray(message.inputs))
                                return "inputs: array expected";
                            for (var i = 0; i < message.inputs.length; ++i) {
                                var error = $root.paddle.framework.proto.OpProto.Var.verify(message.inputs[i]);
                                if (error)
                                    return "inputs." + error;
                            }
                        }
                        if (message.outputs != null && message.hasOwnProperty("outputs")) {
                            if (!Array.isArray(message.outputs))
                                return "outputs: array expected";
                            for (var i = 0; i < message.outputs.length; ++i) {
                                var error = $root.paddle.framework.proto.OpProto.Var.verify(message.outputs[i]);
                                if (error)
                                    return "outputs." + error;
                            }
                        }
                        if (message.attrs != null && message.hasOwnProperty("attrs")) {
                            if (!Array.isArray(message.attrs))
                                return "attrs: array expected";
                            for (var i = 0; i < message.attrs.length; ++i) {
                                var error = $root.paddle.framework.proto.OpProto.Attr.verify(message.attrs[i]);
                                if (error)
                                    return "attrs." + error;
                            }
                        }
                        if (!$util.isString(message.comment))
                            return "comment: string expected";
                        return null;
                    };
    
                    OpProto.fromObject = function fromObject(object) {
                        if (object instanceof $root.paddle.framework.proto.OpProto)
                            return object;
                        var message = new $root.paddle.framework.proto.OpProto();
                        if (object.type != null)
                            message.type = String(object.type);
                        if (object.inputs) {
                            if (!Array.isArray(object.inputs))
                                throw TypeError(".paddle.framework.proto.OpProto.inputs: array expected");
                            message.inputs = [];
                            for (var i = 0; i < object.inputs.length; ++i) {
                                if (typeof object.inputs[i] !== "object")
                                    throw TypeError(".paddle.framework.proto.OpProto.inputs: object expected");
                                message.inputs[i] = $root.paddle.framework.proto.OpProto.Var.fromObject(object.inputs[i]);
                            }
                        }
                        if (object.outputs) {
                            if (!Array.isArray(object.outputs))
                                throw TypeError(".paddle.framework.proto.OpProto.outputs: array expected");
                            message.outputs = [];
                            for (var i = 0; i < object.outputs.length; ++i) {
                                if (typeof object.outputs[i] !== "object")
                                    throw TypeError(".paddle.framework.proto.OpProto.outputs: object expected");
                                message.outputs[i] = $root.paddle.framework.proto.OpProto.Var.fromObject(object.outputs[i]);
                            }
                        }
                        if (object.attrs) {
                            if (!Array.isArray(object.attrs))
                                throw TypeError(".paddle.framework.proto.OpProto.attrs: array expected");
                            message.attrs = [];
                            for (var i = 0; i < object.attrs.length; ++i) {
                                if (typeof object.attrs[i] !== "object")
                                    throw TypeError(".paddle.framework.proto.OpProto.attrs: object expected");
                                message.attrs[i] = $root.paddle.framework.proto.OpProto.Attr.fromObject(object.attrs[i]);
                            }
                        }
                        if (object.comment != null)
                            message.comment = String(object.comment);
                        return message;
                    };
    
                    OpProto.toObject = function toObject(message, options) {
                        if (!options)
                            options = {};
                        var object = {};
                        if (options.arrays || options.defaults) {
                            object.inputs = [];
                            object.outputs = [];
                            object.attrs = [];
                        }
                        if (options.defaults) {
                            object.type = "";
                            object.comment = "";
                        }
                        if (message.type != null && message.hasOwnProperty("type"))
                            object.type = message.type;
                        if (message.inputs && message.inputs.length) {
                            object.inputs = [];
                            for (var j = 0; j < message.inputs.length; ++j)
                                object.inputs[j] = $root.paddle.framework.proto.OpProto.Var.toObject(message.inputs[j], options);
                        }
                        if (message.outputs && message.outputs.length) {
                            object.outputs = [];
                            for (var j = 0; j < message.outputs.length; ++j)
                                object.outputs[j] = $root.paddle.framework.proto.OpProto.Var.toObject(message.outputs[j], options);
                        }
                        if (message.attrs && message.attrs.length) {
                            object.attrs = [];
                            for (var j = 0; j < message.attrs.length; ++j)
                                object.attrs[j] = $root.paddle.framework.proto.OpProto.Attr.toObject(message.attrs[j], options);
                        }
                        if (message.comment != null && message.hasOwnProperty("comment"))
                            object.comment = message.comment;
                        return object;
                    };
    
                    OpProto.prototype.toJSON = function toJSON() {
                        return this.constructor.toObject(this, $protobuf.util.toJSONOptions);
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
    
                        Var.create = function create(properties) {
                            return new Var(properties);
                        };
    
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
    
                        Var.verify = function verify(message) {
                            if (typeof message !== "object" || message === null)
                                return "object expected";
                            if (!$util.isString(message.name))
                                return "name: string expected";
                            if (!$util.isString(message.comment))
                                return "comment: string expected";
                            if (message.duplicable != null && message.hasOwnProperty("duplicable"))
                                if (typeof message.duplicable !== "boolean")
                                    return "duplicable: boolean expected";
                            if (message.intermediate != null && message.hasOwnProperty("intermediate"))
                                if (typeof message.intermediate !== "boolean")
                                    return "intermediate: boolean expected";
                            if (message.dispensable != null && message.hasOwnProperty("dispensable"))
                                if (typeof message.dispensable !== "boolean")
                                    return "dispensable: boolean expected";
                            return null;
                        };
    
                        Var.fromObject = function fromObject(object) {
                            if (object instanceof $root.paddle.framework.proto.OpProto.Var)
                                return object;
                            var message = new $root.paddle.framework.proto.OpProto.Var();
                            if (object.name != null)
                                message.name = String(object.name);
                            if (object.comment != null)
                                message.comment = String(object.comment);
                            if (object.duplicable != null)
                                message.duplicable = Boolean(object.duplicable);
                            if (object.intermediate != null)
                                message.intermediate = Boolean(object.intermediate);
                            if (object.dispensable != null)
                                message.dispensable = Boolean(object.dispensable);
                            return message;
                        };
    
                        Var.toObject = function toObject(message, options) {
                            if (!options)
                                options = {};
                            var object = {};
                            if (options.defaults) {
                                object.name = "";
                                object.comment = "";
                                object.duplicable = false;
                                object.intermediate = false;
                                object.dispensable = false;
                            }
                            if (message.name != null && message.hasOwnProperty("name"))
                                object.name = message.name;
                            if (message.comment != null && message.hasOwnProperty("comment"))
                                object.comment = message.comment;
                            if (message.duplicable != null && message.hasOwnProperty("duplicable"))
                                object.duplicable = message.duplicable;
                            if (message.intermediate != null && message.hasOwnProperty("intermediate"))
                                object.intermediate = message.intermediate;
                            if (message.dispensable != null && message.hasOwnProperty("dispensable"))
                                object.dispensable = message.dispensable;
                            return object;
                        };
    
                        Var.prototype.toJSON = function toJSON() {
                            return this.constructor.toObject(this, $protobuf.util.toJSONOptions);
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
    
                        Attr.create = function create(properties) {
                            return new Attr(properties);
                        };
    
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
    
                        Attr.verify = function verify(message) {
                            if (typeof message !== "object" || message === null)
                                return "object expected";
                            if (!$util.isString(message.name))
                                return "name: string expected";
                            switch (message.type) {
                            default:
                                return "type: enum value expected";
                            case 0:
                            case 1:
                            case 2:
                            case 3:
                            case 4:
                            case 5:
                            case 6:
                            case 7:
                            case 8:
                            case 9:
                            case 10:
                            case 11:
                                break;
                            }
                            if (!$util.isString(message.comment))
                                return "comment: string expected";
                            if (message.generated != null && message.hasOwnProperty("generated"))
                                if (typeof message.generated !== "boolean")
                                    return "generated: boolean expected";
                            return null;
                        };
    
                        Attr.fromObject = function fromObject(object) {
                            if (object instanceof $root.paddle.framework.proto.OpProto.Attr)
                                return object;
                            var message = new $root.paddle.framework.proto.OpProto.Attr();
                            if (object.name != null)
                                message.name = String(object.name);
                            switch (object.type) {
                            case "INT":
                            case 0:
                                message.type = 0;
                                break;
                            case "FLOAT":
                            case 1:
                                message.type = 1;
                                break;
                            case "STRING":
                            case 2:
                                message.type = 2;
                                break;
                            case "INTS":
                            case 3:
                                message.type = 3;
                                break;
                            case "FLOATS":
                            case 4:
                                message.type = 4;
                                break;
                            case "STRINGS":
                            case 5:
                                message.type = 5;
                                break;
                            case "BOOLEAN":
                            case 6:
                                message.type = 6;
                                break;
                            case "BOOLEANS":
                            case 7:
                                message.type = 7;
                                break;
                            case "BLOCK":
                            case 8:
                                message.type = 8;
                                break;
                            case "LONG":
                            case 9:
                                message.type = 9;
                                break;
                            case "BLOCKS":
                            case 10:
                                message.type = 10;
                                break;
                            case "LONGS":
                            case 11:
                                message.type = 11;
                                break;
                            }
                            if (object.comment != null)
                                message.comment = String(object.comment);
                            if (object.generated != null)
                                message.generated = Boolean(object.generated);
                            return message;
                        };
    
                        Attr.toObject = function toObject(message, options) {
                            if (!options)
                                options = {};
                            var object = {};
                            if (options.defaults) {
                                object.name = "";
                                object.type = options.enums === String ? "INT" : 0;
                                object.comment = "";
                                object.generated = false;
                            }
                            if (message.name != null && message.hasOwnProperty("name"))
                                object.name = message.name;
                            if (message.type != null && message.hasOwnProperty("type"))
                                object.type = options.enums === String ? $root.paddle.framework.proto.AttrType[message.type] : message.type;
                            if (message.comment != null && message.hasOwnProperty("comment"))
                                object.comment = message.comment;
                            if (message.generated != null && message.hasOwnProperty("generated"))
                                object.generated = message.generated;
                            return object;
                        };
    
                        Attr.prototype.toJSON = function toJSON() {
                            return this.constructor.toObject(this, $protobuf.util.toJSONOptions);
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
    
                    VarType.create = function create(properties) {
                        return new VarType(properties);
                    };
    
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
    
                    VarType.verify = function verify(message) {
                        if (typeof message !== "object" || message === null)
                            return "object expected";
                        switch (message.type) {
                        default:
                            return "type: enum value expected";
                        case 0:
                        case 1:
                        case 2:
                        case 3:
                        case 4:
                        case 5:
                        case 6:
                        case 19:
                        case 20:
                        case 21:
                        case 7:
                        case 8:
                        case 9:
                        case 10:
                        case 11:
                        case 12:
                        case 13:
                        case 14:
                        case 15:
                        case 17:
                        case 18:
                            break;
                        }
                        if (message.selected_rows != null && message.hasOwnProperty("selected_rows")) {
                            var error = $root.paddle.framework.proto.VarType.TensorDesc.verify(message.selected_rows);
                            if (error)
                                return "selected_rows." + error;
                        }
                        if (message.lod_tensor != null && message.hasOwnProperty("lod_tensor")) {
                            var error = $root.paddle.framework.proto.VarType.LoDTensorDesc.verify(message.lod_tensor);
                            if (error)
                                return "lod_tensor." + error;
                        }
                        if (message.tensor_array != null && message.hasOwnProperty("tensor_array")) {
                            var error = $root.paddle.framework.proto.VarType.LoDTensorArrayDesc.verify(message.tensor_array);
                            if (error)
                                return "tensor_array." + error;
                        }
                        if (message.reader != null && message.hasOwnProperty("reader")) {
                            var error = $root.paddle.framework.proto.VarType.ReaderDesc.verify(message.reader);
                            if (error)
                                return "reader." + error;
                        }
                        if (message.tuple != null && message.hasOwnProperty("tuple")) {
                            var error = $root.paddle.framework.proto.VarType.Tuple.verify(message.tuple);
                            if (error)
                                return "tuple." + error;
                        }
                        return null;
                    };
    
                    VarType.fromObject = function fromObject(object) {
                        if (object instanceof $root.paddle.framework.proto.VarType)
                            return object;
                        var message = new $root.paddle.framework.proto.VarType();
                        switch (object.type) {
                        case "BOOL":
                        case 0:
                            message.type = 0;
                            break;
                        case "INT16":
                        case 1:
                            message.type = 1;
                            break;
                        case "INT32":
                        case 2:
                            message.type = 2;
                            break;
                        case "INT64":
                        case 3:
                            message.type = 3;
                            break;
                        case "FP16":
                        case 4:
                            message.type = 4;
                            break;
                        case "FP32":
                        case 5:
                            message.type = 5;
                            break;
                        case "FP64":
                        case 6:
                            message.type = 6;
                            break;
                        case "SIZE_T":
                        case 19:
                            message.type = 19;
                            break;
                        case "UINT8":
                        case 20:
                            message.type = 20;
                            break;
                        case "INT8":
                        case 21:
                            message.type = 21;
                            break;
                        case "LOD_TENSOR":
                        case 7:
                            message.type = 7;
                            break;
                        case "SELECTED_ROWS":
                        case 8:
                            message.type = 8;
                            break;
                        case "FEED_MINIBATCH":
                        case 9:
                            message.type = 9;
                            break;
                        case "FETCH_LIST":
                        case 10:
                            message.type = 10;
                            break;
                        case "STEP_SCOPES":
                        case 11:
                            message.type = 11;
                            break;
                        case "LOD_RANK_TABLE":
                        case 12:
                            message.type = 12;
                            break;
                        case "LOD_TENSOR_ARRAY":
                        case 13:
                            message.type = 13;
                            break;
                        case "PLACE_LIST":
                        case 14:
                            message.type = 14;
                            break;
                        case "READER":
                        case 15:
                            message.type = 15;
                            break;
                        case "RAW":
                        case 17:
                            message.type = 17;
                            break;
                        case "TUPLE":
                        case 18:
                            message.type = 18;
                            break;
                        }
                        if (object.selected_rows != null) {
                            if (typeof object.selected_rows !== "object")
                                throw TypeError(".paddle.framework.proto.VarType.selected_rows: object expected");
                            message.selected_rows = $root.paddle.framework.proto.VarType.TensorDesc.fromObject(object.selected_rows);
                        }
                        if (object.lod_tensor != null) {
                            if (typeof object.lod_tensor !== "object")
                                throw TypeError(".paddle.framework.proto.VarType.lod_tensor: object expected");
                            message.lod_tensor = $root.paddle.framework.proto.VarType.LoDTensorDesc.fromObject(object.lod_tensor);
                        }
                        if (object.tensor_array != null) {
                            if (typeof object.tensor_array !== "object")
                                throw TypeError(".paddle.framework.proto.VarType.tensor_array: object expected");
                            message.tensor_array = $root.paddle.framework.proto.VarType.LoDTensorArrayDesc.fromObject(object.tensor_array);
                        }
                        if (object.reader != null) {
                            if (typeof object.reader !== "object")
                                throw TypeError(".paddle.framework.proto.VarType.reader: object expected");
                            message.reader = $root.paddle.framework.proto.VarType.ReaderDesc.fromObject(object.reader);
                        }
                        if (object.tuple != null) {
                            if (typeof object.tuple !== "object")
                                throw TypeError(".paddle.framework.proto.VarType.tuple: object expected");
                            message.tuple = $root.paddle.framework.proto.VarType.Tuple.fromObject(object.tuple);
                        }
                        return message;
                    };
    
                    VarType.toObject = function toObject(message, options) {
                        if (!options)
                            options = {};
                        var object = {};
                        if (options.defaults) {
                            object.type = options.enums === String ? "BOOL" : 0;
                            object.selected_rows = null;
                            object.lod_tensor = null;
                            object.tensor_array = null;
                            object.reader = null;
                            object.tuple = null;
                        }
                        if (message.type != null && message.hasOwnProperty("type"))
                            object.type = options.enums === String ? $root.paddle.framework.proto.VarType.Type[message.type] : message.type;
                        if (message.selected_rows != null && message.hasOwnProperty("selected_rows"))
                            object.selected_rows = $root.paddle.framework.proto.VarType.TensorDesc.toObject(message.selected_rows, options);
                        if (message.lod_tensor != null && message.hasOwnProperty("lod_tensor"))
                            object.lod_tensor = $root.paddle.framework.proto.VarType.LoDTensorDesc.toObject(message.lod_tensor, options);
                        if (message.tensor_array != null && message.hasOwnProperty("tensor_array"))
                            object.tensor_array = $root.paddle.framework.proto.VarType.LoDTensorArrayDesc.toObject(message.tensor_array, options);
                        if (message.reader != null && message.hasOwnProperty("reader"))
                            object.reader = $root.paddle.framework.proto.VarType.ReaderDesc.toObject(message.reader, options);
                        if (message.tuple != null && message.hasOwnProperty("tuple"))
                            object.tuple = $root.paddle.framework.proto.VarType.Tuple.toObject(message.tuple, options);
                        return object;
                    };
    
                    VarType.prototype.toJSON = function toJSON() {
                        return this.constructor.toObject(this, $protobuf.util.toJSONOptions);
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
    
                        TensorDesc.create = function create(properties) {
                            return new TensorDesc(properties);
                        };
    
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
    
                        TensorDesc.verify = function verify(message) {
                            if (typeof message !== "object" || message === null)
                                return "object expected";
                            switch (message.data_type) {
                            default:
                                return "data_type: enum value expected";
                            case 0:
                            case 1:
                            case 2:
                            case 3:
                            case 4:
                            case 5:
                            case 6:
                            case 19:
                            case 20:
                            case 21:
                            case 7:
                            case 8:
                            case 9:
                            case 10:
                            case 11:
                            case 12:
                            case 13:
                            case 14:
                            case 15:
                            case 17:
                            case 18:
                                break;
                            }
                            if (message.dims != null && message.hasOwnProperty("dims")) {
                                if (!Array.isArray(message.dims))
                                    return "dims: array expected";
                                for (var i = 0; i < message.dims.length; ++i)
                                    if (!$util.isInteger(message.dims[i]) && !(message.dims[i] && $util.isInteger(message.dims[i].low) && $util.isInteger(message.dims[i].high)))
                                        return "dims: integer|Long[] expected";
                            }
                            return null;
                        };
    
                        TensorDesc.fromObject = function fromObject(object) {
                            if (object instanceof $root.paddle.framework.proto.VarType.TensorDesc)
                                return object;
                            var message = new $root.paddle.framework.proto.VarType.TensorDesc();
                            switch (object.data_type) {
                            case "BOOL":
                            case 0:
                                message.data_type = 0;
                                break;
                            case "INT16":
                            case 1:
                                message.data_type = 1;
                                break;
                            case "INT32":
                            case 2:
                                message.data_type = 2;
                                break;
                            case "INT64":
                            case 3:
                                message.data_type = 3;
                                break;
                            case "FP16":
                            case 4:
                                message.data_type = 4;
                                break;
                            case "FP32":
                            case 5:
                                message.data_type = 5;
                                break;
                            case "FP64":
                            case 6:
                                message.data_type = 6;
                                break;
                            case "SIZE_T":
                            case 19:
                                message.data_type = 19;
                                break;
                            case "UINT8":
                            case 20:
                                message.data_type = 20;
                                break;
                            case "INT8":
                            case 21:
                                message.data_type = 21;
                                break;
                            case "LOD_TENSOR":
                            case 7:
                                message.data_type = 7;
                                break;
                            case "SELECTED_ROWS":
                            case 8:
                                message.data_type = 8;
                                break;
                            case "FEED_MINIBATCH":
                            case 9:
                                message.data_type = 9;
                                break;
                            case "FETCH_LIST":
                            case 10:
                                message.data_type = 10;
                                break;
                            case "STEP_SCOPES":
                            case 11:
                                message.data_type = 11;
                                break;
                            case "LOD_RANK_TABLE":
                            case 12:
                                message.data_type = 12;
                                break;
                            case "LOD_TENSOR_ARRAY":
                            case 13:
                                message.data_type = 13;
                                break;
                            case "PLACE_LIST":
                            case 14:
                                message.data_type = 14;
                                break;
                            case "READER":
                            case 15:
                                message.data_type = 15;
                                break;
                            case "RAW":
                            case 17:
                                message.data_type = 17;
                                break;
                            case "TUPLE":
                            case 18:
                                message.data_type = 18;
                                break;
                            }
                            if (object.dims) {
                                if (!Array.isArray(object.dims))
                                    throw TypeError(".paddle.framework.proto.VarType.TensorDesc.dims: array expected");
                                message.dims = [];
                                for (var i = 0; i < object.dims.length; ++i)
                                    if ($util.Long)
                                        (message.dims[i] = $util.Long.fromValue(object.dims[i])).unsigned = false;
                                    else if (typeof object.dims[i] === "string")
                                        message.dims[i] = parseInt(object.dims[i], 10);
                                    else if (typeof object.dims[i] === "number")
                                        message.dims[i] = object.dims[i];
                                    else if (typeof object.dims[i] === "object")
                                        message.dims[i] = new $util.LongBits(object.dims[i].low >>> 0, object.dims[i].high >>> 0).toNumber();
                            }
                            return message;
                        };
    
                        TensorDesc.toObject = function toObject(message, options) {
                            if (!options)
                                options = {};
                            var object = {};
                            if (options.arrays || options.defaults)
                                object.dims = [];
                            if (options.defaults)
                                object.data_type = options.enums === String ? "BOOL" : 0;
                            if (message.data_type != null && message.hasOwnProperty("data_type"))
                                object.data_type = options.enums === String ? $root.paddle.framework.proto.VarType.Type[message.data_type] : message.data_type;
                            if (message.dims && message.dims.length) {
                                object.dims = [];
                                for (var j = 0; j < message.dims.length; ++j)
                                    if (typeof message.dims[j] === "number")
                                        object.dims[j] = options.longs === String ? String(message.dims[j]) : message.dims[j];
                                    else
                                        object.dims[j] = options.longs === String ? $util.Long.prototype.toString.call(message.dims[j]) : options.longs === Number ? new $util.LongBits(message.dims[j].low >>> 0, message.dims[j].high >>> 0).toNumber() : message.dims[j];
                            }
                            return object;
                        };
    
                        TensorDesc.prototype.toJSON = function toJSON() {
                            return this.constructor.toObject(this, $protobuf.util.toJSONOptions);
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
    
                        LoDTensorDesc.create = function create(properties) {
                            return new LoDTensorDesc(properties);
                        };
    
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
    
                        LoDTensorDesc.verify = function verify(message) {
                            if (typeof message !== "object" || message === null)
                                return "object expected";
                            {
                                var error = $root.paddle.framework.proto.VarType.TensorDesc.verify(message.tensor);
                                if (error)
                                    return "tensor." + error;
                            }
                            if (message.lod_level != null && message.hasOwnProperty("lod_level"))
                                if (!$util.isInteger(message.lod_level))
                                    return "lod_level: integer expected";
                            return null;
                        };
    
                        LoDTensorDesc.fromObject = function fromObject(object) {
                            if (object instanceof $root.paddle.framework.proto.VarType.LoDTensorDesc)
                                return object;
                            var message = new $root.paddle.framework.proto.VarType.LoDTensorDesc();
                            if (object.tensor != null) {
                                if (typeof object.tensor !== "object")
                                    throw TypeError(".paddle.framework.proto.VarType.LoDTensorDesc.tensor: object expected");
                                message.tensor = $root.paddle.framework.proto.VarType.TensorDesc.fromObject(object.tensor);
                            }
                            if (object.lod_level != null)
                                message.lod_level = object.lod_level | 0;
                            return message;
                        };
    
                        LoDTensorDesc.toObject = function toObject(message, options) {
                            if (!options)
                                options = {};
                            var object = {};
                            if (options.defaults) {
                                object.tensor = null;
                                object.lod_level = 0;
                            }
                            if (message.tensor != null && message.hasOwnProperty("tensor"))
                                object.tensor = $root.paddle.framework.proto.VarType.TensorDesc.toObject(message.tensor, options);
                            if (message.lod_level != null && message.hasOwnProperty("lod_level"))
                                object.lod_level = message.lod_level;
                            return object;
                        };
    
                        LoDTensorDesc.prototype.toJSON = function toJSON() {
                            return this.constructor.toObject(this, $protobuf.util.toJSONOptions);
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
    
                        LoDTensorArrayDesc.create = function create(properties) {
                            return new LoDTensorArrayDesc(properties);
                        };
    
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
    
                        LoDTensorArrayDesc.verify = function verify(message) {
                            if (typeof message !== "object" || message === null)
                                return "object expected";
                            {
                                var error = $root.paddle.framework.proto.VarType.TensorDesc.verify(message.tensor);
                                if (error)
                                    return "tensor." + error;
                            }
                            if (message.lod_level != null && message.hasOwnProperty("lod_level"))
                                if (!$util.isInteger(message.lod_level))
                                    return "lod_level: integer expected";
                            return null;
                        };
    
                        LoDTensorArrayDesc.fromObject = function fromObject(object) {
                            if (object instanceof $root.paddle.framework.proto.VarType.LoDTensorArrayDesc)
                                return object;
                            var message = new $root.paddle.framework.proto.VarType.LoDTensorArrayDesc();
                            if (object.tensor != null) {
                                if (typeof object.tensor !== "object")
                                    throw TypeError(".paddle.framework.proto.VarType.LoDTensorArrayDesc.tensor: object expected");
                                message.tensor = $root.paddle.framework.proto.VarType.TensorDesc.fromObject(object.tensor);
                            }
                            if (object.lod_level != null)
                                message.lod_level = object.lod_level | 0;
                            return message;
                        };
    
                        LoDTensorArrayDesc.toObject = function toObject(message, options) {
                            if (!options)
                                options = {};
                            var object = {};
                            if (options.defaults) {
                                object.tensor = null;
                                object.lod_level = 0;
                            }
                            if (message.tensor != null && message.hasOwnProperty("tensor"))
                                object.tensor = $root.paddle.framework.proto.VarType.TensorDesc.toObject(message.tensor, options);
                            if (message.lod_level != null && message.hasOwnProperty("lod_level"))
                                object.lod_level = message.lod_level;
                            return object;
                        };
    
                        LoDTensorArrayDesc.prototype.toJSON = function toJSON() {
                            return this.constructor.toObject(this, $protobuf.util.toJSONOptions);
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
    
                        ReaderDesc.create = function create(properties) {
                            return new ReaderDesc(properties);
                        };
    
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
    
                        ReaderDesc.verify = function verify(message) {
                            if (typeof message !== "object" || message === null)
                                return "object expected";
                            if (message.lod_tensor != null && message.hasOwnProperty("lod_tensor")) {
                                if (!Array.isArray(message.lod_tensor))
                                    return "lod_tensor: array expected";
                                for (var i = 0; i < message.lod_tensor.length; ++i) {
                                    var error = $root.paddle.framework.proto.VarType.LoDTensorDesc.verify(message.lod_tensor[i]);
                                    if (error)
                                        return "lod_tensor." + error;
                                }
                            }
                            return null;
                        };
    
                        ReaderDesc.fromObject = function fromObject(object) {
                            if (object instanceof $root.paddle.framework.proto.VarType.ReaderDesc)
                                return object;
                            var message = new $root.paddle.framework.proto.VarType.ReaderDesc();
                            if (object.lod_tensor) {
                                if (!Array.isArray(object.lod_tensor))
                                    throw TypeError(".paddle.framework.proto.VarType.ReaderDesc.lod_tensor: array expected");
                                message.lod_tensor = [];
                                for (var i = 0; i < object.lod_tensor.length; ++i) {
                                    if (typeof object.lod_tensor[i] !== "object")
                                        throw TypeError(".paddle.framework.proto.VarType.ReaderDesc.lod_tensor: object expected");
                                    message.lod_tensor[i] = $root.paddle.framework.proto.VarType.LoDTensorDesc.fromObject(object.lod_tensor[i]);
                                }
                            }
                            return message;
                        };
    
                        ReaderDesc.toObject = function toObject(message, options) {
                            if (!options)
                                options = {};
                            var object = {};
                            if (options.arrays || options.defaults)
                                object.lod_tensor = [];
                            if (message.lod_tensor && message.lod_tensor.length) {
                                object.lod_tensor = [];
                                for (var j = 0; j < message.lod_tensor.length; ++j)
                                    object.lod_tensor[j] = $root.paddle.framework.proto.VarType.LoDTensorDesc.toObject(message.lod_tensor[j], options);
                            }
                            return object;
                        };
    
                        ReaderDesc.prototype.toJSON = function toJSON() {
                            return this.constructor.toObject(this, $protobuf.util.toJSONOptions);
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
    
                        Tuple.create = function create(properties) {
                            return new Tuple(properties);
                        };
    
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
    
                        Tuple.verify = function verify(message) {
                            if (typeof message !== "object" || message === null)
                                return "object expected";
                            if (message.element_type != null && message.hasOwnProperty("element_type")) {
                                if (!Array.isArray(message.element_type))
                                    return "element_type: array expected";
                                for (var i = 0; i < message.element_type.length; ++i)
                                    switch (message.element_type[i]) {
                                    default:
                                        return "element_type: enum value[] expected";
                                    case 0:
                                    case 1:
                                    case 2:
                                    case 3:
                                    case 4:
                                    case 5:
                                    case 6:
                                    case 19:
                                    case 20:
                                    case 21:
                                    case 7:
                                    case 8:
                                    case 9:
                                    case 10:
                                    case 11:
                                    case 12:
                                    case 13:
                                    case 14:
                                    case 15:
                                    case 17:
                                    case 18:
                                        break;
                                    }
                            }
                            return null;
                        };
    
                        Tuple.fromObject = function fromObject(object) {
                            if (object instanceof $root.paddle.framework.proto.VarType.Tuple)
                                return object;
                            var message = new $root.paddle.framework.proto.VarType.Tuple();
                            if (object.element_type) {
                                if (!Array.isArray(object.element_type))
                                    throw TypeError(".paddle.framework.proto.VarType.Tuple.element_type: array expected");
                                message.element_type = [];
                                for (var i = 0; i < object.element_type.length; ++i)
                                    switch (object.element_type[i]) {
                                    default:
                                    case "BOOL":
                                    case 0:
                                        message.element_type[i] = 0;
                                        break;
                                    case "INT16":
                                    case 1:
                                        message.element_type[i] = 1;
                                        break;
                                    case "INT32":
                                    case 2:
                                        message.element_type[i] = 2;
                                        break;
                                    case "INT64":
                                    case 3:
                                        message.element_type[i] = 3;
                                        break;
                                    case "FP16":
                                    case 4:
                                        message.element_type[i] = 4;
                                        break;
                                    case "FP32":
                                    case 5:
                                        message.element_type[i] = 5;
                                        break;
                                    case "FP64":
                                    case 6:
                                        message.element_type[i] = 6;
                                        break;
                                    case "SIZE_T":
                                    case 19:
                                        message.element_type[i] = 19;
                                        break;
                                    case "UINT8":
                                    case 20:
                                        message.element_type[i] = 20;
                                        break;
                                    case "INT8":
                                    case 21:
                                        message.element_type[i] = 21;
                                        break;
                                    case "LOD_TENSOR":
                                    case 7:
                                        message.element_type[i] = 7;
                                        break;
                                    case "SELECTED_ROWS":
                                    case 8:
                                        message.element_type[i] = 8;
                                        break;
                                    case "FEED_MINIBATCH":
                                    case 9:
                                        message.element_type[i] = 9;
                                        break;
                                    case "FETCH_LIST":
                                    case 10:
                                        message.element_type[i] = 10;
                                        break;
                                    case "STEP_SCOPES":
                                    case 11:
                                        message.element_type[i] = 11;
                                        break;
                                    case "LOD_RANK_TABLE":
                                    case 12:
                                        message.element_type[i] = 12;
                                        break;
                                    case "LOD_TENSOR_ARRAY":
                                    case 13:
                                        message.element_type[i] = 13;
                                        break;
                                    case "PLACE_LIST":
                                    case 14:
                                        message.element_type[i] = 14;
                                        break;
                                    case "READER":
                                    case 15:
                                        message.element_type[i] = 15;
                                        break;
                                    case "RAW":
                                    case 17:
                                        message.element_type[i] = 17;
                                        break;
                                    case "TUPLE":
                                    case 18:
                                        message.element_type[i] = 18;
                                        break;
                                    }
                            }
                            return message;
                        };
    
                        Tuple.toObject = function toObject(message, options) {
                            if (!options)
                                options = {};
                            var object = {};
                            if (options.arrays || options.defaults)
                                object.element_type = [];
                            if (message.element_type && message.element_type.length) {
                                object.element_type = [];
                                for (var j = 0; j < message.element_type.length; ++j)
                                    object.element_type[j] = options.enums === String ? $root.paddle.framework.proto.VarType.Type[message.element_type[j]] : message.element_type[j];
                            }
                            return object;
                        };
    
                        Tuple.prototype.toJSON = function toJSON() {
                            return this.constructor.toObject(this, $protobuf.util.toJSONOptions);
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
    
                    VarDesc.create = function create(properties) {
                        return new VarDesc(properties);
                    };
    
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
    
                    VarDesc.verify = function verify(message) {
                        if (typeof message !== "object" || message === null)
                            return "object expected";
                        if (!$util.isString(message.name))
                            return "name: string expected";
                        {
                            var error = $root.paddle.framework.proto.VarType.verify(message.type);
                            if (error)
                                return "type." + error;
                        }
                        if (message.persistable != null && message.hasOwnProperty("persistable"))
                            if (typeof message.persistable !== "boolean")
                                return "persistable: boolean expected";
                        return null;
                    };
    
                    VarDesc.fromObject = function fromObject(object) {
                        if (object instanceof $root.paddle.framework.proto.VarDesc)
                            return object;
                        var message = new $root.paddle.framework.proto.VarDesc();
                        if (object.name != null)
                            message.name = String(object.name);
                        if (object.type != null) {
                            if (typeof object.type !== "object")
                                throw TypeError(".paddle.framework.proto.VarDesc.type: object expected");
                            message.type = $root.paddle.framework.proto.VarType.fromObject(object.type);
                        }
                        if (object.persistable != null)
                            message.persistable = Boolean(object.persistable);
                        return message;
                    };
    
                    VarDesc.toObject = function toObject(message, options) {
                        if (!options)
                            options = {};
                        var object = {};
                        if (options.defaults) {
                            object.name = "";
                            object.type = null;
                            object.persistable = false;
                        }
                        if (message.name != null && message.hasOwnProperty("name"))
                            object.name = message.name;
                        if (message.type != null && message.hasOwnProperty("type"))
                            object.type = $root.paddle.framework.proto.VarType.toObject(message.type, options);
                        if (message.persistable != null && message.hasOwnProperty("persistable"))
                            object.persistable = message.persistable;
                        return object;
                    };
    
                    VarDesc.prototype.toJSON = function toJSON() {
                        return this.constructor.toObject(this, $protobuf.util.toJSONOptions);
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
    
                    BlockDesc.create = function create(properties) {
                        return new BlockDesc(properties);
                    };
    
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
    
                    BlockDesc.verify = function verify(message) {
                        if (typeof message !== "object" || message === null)
                            return "object expected";
                        if (!$util.isInteger(message.idx))
                            return "idx: integer expected";
                        if (!$util.isInteger(message.parent_idx))
                            return "parent_idx: integer expected";
                        if (message.vars != null && message.hasOwnProperty("vars")) {
                            if (!Array.isArray(message.vars))
                                return "vars: array expected";
                            for (var i = 0; i < message.vars.length; ++i) {
                                var error = $root.paddle.framework.proto.VarDesc.verify(message.vars[i]);
                                if (error)
                                    return "vars." + error;
                            }
                        }
                        if (message.ops != null && message.hasOwnProperty("ops")) {
                            if (!Array.isArray(message.ops))
                                return "ops: array expected";
                            for (var i = 0; i < message.ops.length; ++i) {
                                var error = $root.paddle.framework.proto.OpDesc.verify(message.ops[i]);
                                if (error)
                                    return "ops." + error;
                            }
                        }
                        if (message.forward_block_idx != null && message.hasOwnProperty("forward_block_idx"))
                            if (!$util.isInteger(message.forward_block_idx))
                                return "forward_block_idx: integer expected";
                        return null;
                    };
    
                    BlockDesc.fromObject = function fromObject(object) {
                        if (object instanceof $root.paddle.framework.proto.BlockDesc)
                            return object;
                        var message = new $root.paddle.framework.proto.BlockDesc();
                        if (object.idx != null)
                            message.idx = object.idx | 0;
                        if (object.parent_idx != null)
                            message.parent_idx = object.parent_idx | 0;
                        if (object.vars) {
                            if (!Array.isArray(object.vars))
                                throw TypeError(".paddle.framework.proto.BlockDesc.vars: array expected");
                            message.vars = [];
                            for (var i = 0; i < object.vars.length; ++i) {
                                if (typeof object.vars[i] !== "object")
                                    throw TypeError(".paddle.framework.proto.BlockDesc.vars: object expected");
                                message.vars[i] = $root.paddle.framework.proto.VarDesc.fromObject(object.vars[i]);
                            }
                        }
                        if (object.ops) {
                            if (!Array.isArray(object.ops))
                                throw TypeError(".paddle.framework.proto.BlockDesc.ops: array expected");
                            message.ops = [];
                            for (var i = 0; i < object.ops.length; ++i) {
                                if (typeof object.ops[i] !== "object")
                                    throw TypeError(".paddle.framework.proto.BlockDesc.ops: object expected");
                                message.ops[i] = $root.paddle.framework.proto.OpDesc.fromObject(object.ops[i]);
                            }
                        }
                        if (object.forward_block_idx != null)
                            message.forward_block_idx = object.forward_block_idx | 0;
                        return message;
                    };
    
                    BlockDesc.toObject = function toObject(message, options) {
                        if (!options)
                            options = {};
                        var object = {};
                        if (options.arrays || options.defaults) {
                            object.vars = [];
                            object.ops = [];
                        }
                        if (options.defaults) {
                            object.idx = 0;
                            object.parent_idx = 0;
                            object.forward_block_idx = -1;
                        }
                        if (message.idx != null && message.hasOwnProperty("idx"))
                            object.idx = message.idx;
                        if (message.parent_idx != null && message.hasOwnProperty("parent_idx"))
                            object.parent_idx = message.parent_idx;
                        if (message.vars && message.vars.length) {
                            object.vars = [];
                            for (var j = 0; j < message.vars.length; ++j)
                                object.vars[j] = $root.paddle.framework.proto.VarDesc.toObject(message.vars[j], options);
                        }
                        if (message.ops && message.ops.length) {
                            object.ops = [];
                            for (var j = 0; j < message.ops.length; ++j)
                                object.ops[j] = $root.paddle.framework.proto.OpDesc.toObject(message.ops[j], options);
                        }
                        if (message.forward_block_idx != null && message.hasOwnProperty("forward_block_idx"))
                            object.forward_block_idx = message.forward_block_idx;
                        return object;
                    };
    
                    BlockDesc.prototype.toJSON = function toJSON() {
                        return this.constructor.toObject(this, $protobuf.util.toJSONOptions);
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
    
                    ProgramDesc.create = function create(properties) {
                        return new ProgramDesc(properties);
                    };
    
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
    
                    ProgramDesc.verify = function verify(message) {
                        if (typeof message !== "object" || message === null)
                            return "object expected";
                        if (message.blocks != null && message.hasOwnProperty("blocks")) {
                            if (!Array.isArray(message.blocks))
                                return "blocks: array expected";
                            for (var i = 0; i < message.blocks.length; ++i) {
                                var error = $root.paddle.framework.proto.BlockDesc.verify(message.blocks[i]);
                                if (error)
                                    return "blocks." + error;
                            }
                        }
                        if (message.version != null && message.hasOwnProperty("version")) {
                            var error = $root.paddle.framework.proto.Version.verify(message.version);
                            if (error)
                                return "version." + error;
                        }
                        return null;
                    };
    
                    ProgramDesc.fromObject = function fromObject(object) {
                        if (object instanceof $root.paddle.framework.proto.ProgramDesc)
                            return object;
                        var message = new $root.paddle.framework.proto.ProgramDesc();
                        if (object.blocks) {
                            if (!Array.isArray(object.blocks))
                                throw TypeError(".paddle.framework.proto.ProgramDesc.blocks: array expected");
                            message.blocks = [];
                            for (var i = 0; i < object.blocks.length; ++i) {
                                if (typeof object.blocks[i] !== "object")
                                    throw TypeError(".paddle.framework.proto.ProgramDesc.blocks: object expected");
                                message.blocks[i] = $root.paddle.framework.proto.BlockDesc.fromObject(object.blocks[i]);
                            }
                        }
                        if (object.version != null) {
                            if (typeof object.version !== "object")
                                throw TypeError(".paddle.framework.proto.ProgramDesc.version: object expected");
                            message.version = $root.paddle.framework.proto.Version.fromObject(object.version);
                        }
                        return message;
                    };
    
                    ProgramDesc.toObject = function toObject(message, options) {
                        if (!options)
                            options = {};
                        var object = {};
                        if (options.arrays || options.defaults)
                            object.blocks = [];
                        if (options.defaults)
                            object.version = null;
                        if (message.blocks && message.blocks.length) {
                            object.blocks = [];
                            for (var j = 0; j < message.blocks.length; ++j)
                                object.blocks[j] = $root.paddle.framework.proto.BlockDesc.toObject(message.blocks[j], options);
                        }
                        if (message.version != null && message.hasOwnProperty("version"))
                            object.version = $root.paddle.framework.proto.Version.toObject(message.version, options);
                        return object;
                    };
    
                    ProgramDesc.prototype.toJSON = function toJSON() {
                        return this.constructor.toObject(this, $protobuf.util.toJSONOptions);
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
