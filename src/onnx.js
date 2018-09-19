/*eslint-disable block-scoped-var, id-length, no-control-regex, no-magic-numbers, no-prototype-builtins, no-redeclare, no-shadow, no-var, sort-vars*/
(function($protobuf) {
    "use strict";

    var $Reader = $protobuf.Reader, $util = $protobuf.util;
    
    var $root = $protobuf.roots.onnx || ($protobuf.roots.onnx = {});
    
    $root.onnx = (function() {
    
        var onnx = {};
    
        onnx.Version = (function() {
            var valuesById = {}, values = Object.create(valuesById);
            values[valuesById[0] = "_START_VERSION"] = 0;
            values[valuesById[1] = "IR_VERSION_2017_10_10"] = 1;
            values[valuesById[2] = "IR_VERSION_2017_10_30"] = 2;
            values[valuesById[3] = "IR_VERSION"] = 3;
            return values;
        })();
    
        onnx.AttributeProto = (function() {
    
            function AttributeProto(properties) {
                this.floats = [];
                this.ints = [];
                this.strings = [];
                this.tensors = [];
                this.graphs = [];
                if (properties)
                    for (var keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                        if (properties[keys[i]] != null)
                            this[keys[i]] = properties[keys[i]];
            }
    
            AttributeProto.prototype.name = "";
            AttributeProto.prototype.refAttrName = "";
            AttributeProto.prototype.docString = "";
            AttributeProto.prototype.type = 0;
            AttributeProto.prototype.f = 0;
            AttributeProto.prototype.i = $util.Long ? $util.Long.fromBits(0,0,false) : 0;
            AttributeProto.prototype.s = $util.newBuffer([]);
            AttributeProto.prototype.t = null;
            AttributeProto.prototype.g = null;
            AttributeProto.prototype.floats = $util.emptyArray;
            AttributeProto.prototype.ints = $util.emptyArray;
            AttributeProto.prototype.strings = $util.emptyArray;
            AttributeProto.prototype.tensors = $util.emptyArray;
            AttributeProto.prototype.graphs = $util.emptyArray;
    
            AttributeProto.create = function create(properties) {
                return new AttributeProto(properties);
            };
    
            AttributeProto.decode = function decode(reader, length) {
                if (!(reader instanceof $Reader))
                    reader = $Reader.create(reader);
                var end = length === undefined ? reader.len : reader.pos + length, message = new $root.onnx.AttributeProto();
                while (reader.pos < end) {
                    var tag = reader.uint32();
                    switch (tag >>> 3) {
                    case 1:
                        message.name = reader.string();
                        break;
                    case 21:
                        message.refAttrName = reader.string();
                        break;
                    case 13:
                        message.docString = reader.string();
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
                        if (!(message.ints && message.ints.length))
                            message.ints = [];
                        if ((tag & 7) === 2) {
                            var end2 = reader.uint32() + reader.pos;
                            while (reader.pos < end2)
                                message.ints.push(reader.int64());
                        } else
                            message.ints.push(reader.int64());
                        break;
                    case 9:
                        if (!(message.strings && message.strings.length))
                            message.strings = [];
                        message.strings.push(reader.bytes());
                        break;
                    case 10:
                        if (!(message.tensors && message.tensors.length))
                            message.tensors = [];
                        message.tensors.push($root.onnx.TensorProto.decode(reader, reader.uint32()));
                        break;
                    case 11:
                        if (!(message.graphs && message.graphs.length))
                            message.graphs = [];
                        message.graphs.push($root.onnx.GraphProto.decode(reader, reader.uint32()));
                        break;
                    default:
                        reader.skipType(tag & 7);
                        break;
                    }
                }
                return message;
            };
    
            AttributeProto.verify = function verify(message) {
                if (typeof message !== "object" || message === null)
                    return "object expected";
                if (message.name != null && message.hasOwnProperty("name"))
                    if (!$util.isString(message.name))
                        return "name: string expected";
                if (message.refAttrName != null && message.hasOwnProperty("refAttrName"))
                    if (!$util.isString(message.refAttrName))
                        return "refAttrName: string expected";
                if (message.docString != null && message.hasOwnProperty("docString"))
                    if (!$util.isString(message.docString))
                        return "docString: string expected";
                if (message.type != null && message.hasOwnProperty("type"))
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
                        break;
                    }
                if (message.f != null && message.hasOwnProperty("f"))
                    if (typeof message.f !== "number")
                        return "f: number expected";
                if (message.i != null && message.hasOwnProperty("i"))
                    if (!$util.isInteger(message.i) && !(message.i && $util.isInteger(message.i.low) && $util.isInteger(message.i.high)))
                        return "i: integer|Long expected";
                if (message.s != null && message.hasOwnProperty("s"))
                    if (!(message.s && typeof message.s.length === "number" || $util.isString(message.s)))
                        return "s: buffer expected";
                if (message.t != null && message.hasOwnProperty("t")) {
                    var error = $root.onnx.TensorProto.verify(message.t);
                    if (error)
                        return "t." + error;
                }
                if (message.g != null && message.hasOwnProperty("g")) {
                    var error = $root.onnx.GraphProto.verify(message.g);
                    if (error)
                        return "g." + error;
                }
                if (message.floats != null && message.hasOwnProperty("floats")) {
                    if (!Array.isArray(message.floats))
                        return "floats: array expected";
                    for (var i = 0; i < message.floats.length; ++i)
                        if (typeof message.floats[i] !== "number")
                            return "floats: number[] expected";
                }
                if (message.ints != null && message.hasOwnProperty("ints")) {
                    if (!Array.isArray(message.ints))
                        return "ints: array expected";
                    for (var i = 0; i < message.ints.length; ++i)
                        if (!$util.isInteger(message.ints[i]) && !(message.ints[i] && $util.isInteger(message.ints[i].low) && $util.isInteger(message.ints[i].high)))
                            return "ints: integer|Long[] expected";
                }
                if (message.strings != null && message.hasOwnProperty("strings")) {
                    if (!Array.isArray(message.strings))
                        return "strings: array expected";
                    for (var i = 0; i < message.strings.length; ++i)
                        if (!(message.strings[i] && typeof message.strings[i].length === "number" || $util.isString(message.strings[i])))
                            return "strings: buffer[] expected";
                }
                if (message.tensors != null && message.hasOwnProperty("tensors")) {
                    if (!Array.isArray(message.tensors))
                        return "tensors: array expected";
                    for (var i = 0; i < message.tensors.length; ++i) {
                        var error = $root.onnx.TensorProto.verify(message.tensors[i]);
                        if (error)
                            return "tensors." + error;
                    }
                }
                if (message.graphs != null && message.hasOwnProperty("graphs")) {
                    if (!Array.isArray(message.graphs))
                        return "graphs: array expected";
                    for (var i = 0; i < message.graphs.length; ++i) {
                        var error = $root.onnx.GraphProto.verify(message.graphs[i]);
                        if (error)
                            return "graphs." + error;
                    }
                }
                return null;
            };
    
            AttributeProto.fromObject = function fromObject(object) {
                if (object instanceof $root.onnx.AttributeProto)
                    return object;
                var message = new $root.onnx.AttributeProto();
                if (object.name != null)
                    message.name = String(object.name);
                if (object.refAttrName != null)
                    message.refAttrName = String(object.refAttrName);
                if (object.docString != null)
                    message.docString = String(object.docString);
                switch (object.type) {
                case "UNDEFINED":
                case 0:
                    message.type = 0;
                    break;
                case "FLOAT":
                case 1:
                    message.type = 1;
                    break;
                case "INT":
                case 2:
                    message.type = 2;
                    break;
                case "STRING":
                case 3:
                    message.type = 3;
                    break;
                case "TENSOR":
                case 4:
                    message.type = 4;
                    break;
                case "GRAPH":
                case 5:
                    message.type = 5;
                    break;
                case "FLOATS":
                case 6:
                    message.type = 6;
                    break;
                case "INTS":
                case 7:
                    message.type = 7;
                    break;
                case "STRINGS":
                case 8:
                    message.type = 8;
                    break;
                case "TENSORS":
                case 9:
                    message.type = 9;
                    break;
                case "GRAPHS":
                case 10:
                    message.type = 10;
                    break;
                }
                if (object.f != null)
                    message.f = Number(object.f);
                if (object.i != null)
                    if ($util.Long)
                        (message.i = $util.Long.fromValue(object.i)).unsigned = false;
                    else if (typeof object.i === "string")
                        message.i = parseInt(object.i, 10);
                    else if (typeof object.i === "number")
                        message.i = object.i;
                    else if (typeof object.i === "object")
                        message.i = new $util.LongBits(object.i.low >>> 0, object.i.high >>> 0).toNumber();
                if (object.s != null)
                    if (typeof object.s === "string")
                        $util.base64.decode(object.s, message.s = $util.newBuffer($util.base64.length(object.s)), 0);
                    else if (object.s.length)
                        message.s = object.s;
                if (object.t != null) {
                    if (typeof object.t !== "object")
                        throw TypeError(".onnx.AttributeProto.t: object expected");
                    message.t = $root.onnx.TensorProto.fromObject(object.t);
                }
                if (object.g != null) {
                    if (typeof object.g !== "object")
                        throw TypeError(".onnx.AttributeProto.g: object expected");
                    message.g = $root.onnx.GraphProto.fromObject(object.g);
                }
                if (object.floats) {
                    if (!Array.isArray(object.floats))
                        throw TypeError(".onnx.AttributeProto.floats: array expected");
                    message.floats = [];
                    for (var i = 0; i < object.floats.length; ++i)
                        message.floats[i] = Number(object.floats[i]);
                }
                if (object.ints) {
                    if (!Array.isArray(object.ints))
                        throw TypeError(".onnx.AttributeProto.ints: array expected");
                    message.ints = [];
                    for (var i = 0; i < object.ints.length; ++i)
                        if ($util.Long)
                            (message.ints[i] = $util.Long.fromValue(object.ints[i])).unsigned = false;
                        else if (typeof object.ints[i] === "string")
                            message.ints[i] = parseInt(object.ints[i], 10);
                        else if (typeof object.ints[i] === "number")
                            message.ints[i] = object.ints[i];
                        else if (typeof object.ints[i] === "object")
                            message.ints[i] = new $util.LongBits(object.ints[i].low >>> 0, object.ints[i].high >>> 0).toNumber();
                }
                if (object.strings) {
                    if (!Array.isArray(object.strings))
                        throw TypeError(".onnx.AttributeProto.strings: array expected");
                    message.strings = [];
                    for (var i = 0; i < object.strings.length; ++i)
                        if (typeof object.strings[i] === "string")
                            $util.base64.decode(object.strings[i], message.strings[i] = $util.newBuffer($util.base64.length(object.strings[i])), 0);
                        else if (object.strings[i].length)
                            message.strings[i] = object.strings[i];
                }
                if (object.tensors) {
                    if (!Array.isArray(object.tensors))
                        throw TypeError(".onnx.AttributeProto.tensors: array expected");
                    message.tensors = [];
                    for (var i = 0; i < object.tensors.length; ++i) {
                        if (typeof object.tensors[i] !== "object")
                            throw TypeError(".onnx.AttributeProto.tensors: object expected");
                        message.tensors[i] = $root.onnx.TensorProto.fromObject(object.tensors[i]);
                    }
                }
                if (object.graphs) {
                    if (!Array.isArray(object.graphs))
                        throw TypeError(".onnx.AttributeProto.graphs: array expected");
                    message.graphs = [];
                    for (var i = 0; i < object.graphs.length; ++i) {
                        if (typeof object.graphs[i] !== "object")
                            throw TypeError(".onnx.AttributeProto.graphs: object expected");
                        message.graphs[i] = $root.onnx.GraphProto.fromObject(object.graphs[i]);
                    }
                }
                return message;
            };
    
            AttributeProto.toObject = function toObject(message, options) {
                if (!options)
                    options = {};
                var object = {};
                if (options.arrays || options.defaults) {
                    object.floats = [];
                    object.ints = [];
                    object.strings = [];
                    object.tensors = [];
                    object.graphs = [];
                }
                if (options.defaults) {
                    object.name = "";
                    object.f = 0;
                    if ($util.Long) {
                        var long = new $util.Long(0, 0, false);
                        object.i = options.longs === String ? long.toString() : options.longs === Number ? long.toNumber() : long;
                    } else
                        object.i = options.longs === String ? "0" : 0;
                    if (options.bytes === String)
                        object.s = "";
                    else {
                        object.s = [];
                        if (options.bytes !== Array)
                            object.s = $util.newBuffer(object.s);
                    }
                    object.t = null;
                    object.g = null;
                    object.docString = "";
                    object.type = options.enums === String ? "UNDEFINED" : 0;
                    object.refAttrName = "";
                }
                if (message.name != null && message.hasOwnProperty("name"))
                    object.name = message.name;
                if (message.f != null && message.hasOwnProperty("f"))
                    object.f = options.json && !isFinite(message.f) ? String(message.f) : message.f;
                if (message.i != null && message.hasOwnProperty("i"))
                    if (typeof message.i === "number")
                        object.i = options.longs === String ? String(message.i) : message.i;
                    else
                        object.i = options.longs === String ? $util.Long.prototype.toString.call(message.i) : options.longs === Number ? new $util.LongBits(message.i.low >>> 0, message.i.high >>> 0).toNumber() : message.i;
                if (message.s != null && message.hasOwnProperty("s"))
                    object.s = options.bytes === String ? $util.base64.encode(message.s, 0, message.s.length) : options.bytes === Array ? Array.prototype.slice.call(message.s) : message.s;
                if (message.t != null && message.hasOwnProperty("t"))
                    object.t = $root.onnx.TensorProto.toObject(message.t, options);
                if (message.g != null && message.hasOwnProperty("g"))
                    object.g = $root.onnx.GraphProto.toObject(message.g, options);
                if (message.floats && message.floats.length) {
                    object.floats = [];
                    for (var j = 0; j < message.floats.length; ++j)
                        object.floats[j] = options.json && !isFinite(message.floats[j]) ? String(message.floats[j]) : message.floats[j];
                }
                if (message.ints && message.ints.length) {
                    object.ints = [];
                    for (var j = 0; j < message.ints.length; ++j)
                        if (typeof message.ints[j] === "number")
                            object.ints[j] = options.longs === String ? String(message.ints[j]) : message.ints[j];
                        else
                            object.ints[j] = options.longs === String ? $util.Long.prototype.toString.call(message.ints[j]) : options.longs === Number ? new $util.LongBits(message.ints[j].low >>> 0, message.ints[j].high >>> 0).toNumber() : message.ints[j];
                }
                if (message.strings && message.strings.length) {
                    object.strings = [];
                    for (var j = 0; j < message.strings.length; ++j)
                        object.strings[j] = options.bytes === String ? $util.base64.encode(message.strings[j], 0, message.strings[j].length) : options.bytes === Array ? Array.prototype.slice.call(message.strings[j]) : message.strings[j];
                }
                if (message.tensors && message.tensors.length) {
                    object.tensors = [];
                    for (var j = 0; j < message.tensors.length; ++j)
                        object.tensors[j] = $root.onnx.TensorProto.toObject(message.tensors[j], options);
                }
                if (message.graphs && message.graphs.length) {
                    object.graphs = [];
                    for (var j = 0; j < message.graphs.length; ++j)
                        object.graphs[j] = $root.onnx.GraphProto.toObject(message.graphs[j], options);
                }
                if (message.docString != null && message.hasOwnProperty("docString"))
                    object.docString = message.docString;
                if (message.type != null && message.hasOwnProperty("type"))
                    object.type = options.enums === String ? $root.onnx.AttributeProto.AttributeType[message.type] : message.type;
                if (message.refAttrName != null && message.hasOwnProperty("refAttrName"))
                    object.refAttrName = message.refAttrName;
                return object;
            };
    
            AttributeProto.prototype.toJSON = function toJSON() {
                return this.constructor.toObject(this, $protobuf.util.toJSONOptions);
            };
    
            AttributeProto.AttributeType = (function() {
                var valuesById = {}, values = Object.create(valuesById);
                values[valuesById[0] = "UNDEFINED"] = 0;
                values[valuesById[1] = "FLOAT"] = 1;
                values[valuesById[2] = "INT"] = 2;
                values[valuesById[3] = "STRING"] = 3;
                values[valuesById[4] = "TENSOR"] = 4;
                values[valuesById[5] = "GRAPH"] = 5;
                values[valuesById[6] = "FLOATS"] = 6;
                values[valuesById[7] = "INTS"] = 7;
                values[valuesById[8] = "STRINGS"] = 8;
                values[valuesById[9] = "TENSORS"] = 9;
                values[valuesById[10] = "GRAPHS"] = 10;
                return values;
            })();
    
            return AttributeProto;
        })();
    
        onnx.ValueInfoProto = (function() {
    
            function ValueInfoProto(properties) {
                if (properties)
                    for (var keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                        if (properties[keys[i]] != null)
                            this[keys[i]] = properties[keys[i]];
            }
    
            ValueInfoProto.prototype.name = "";
            ValueInfoProto.prototype.type = null;
            ValueInfoProto.prototype.docString = "";
    
            ValueInfoProto.create = function create(properties) {
                return new ValueInfoProto(properties);
            };
    
            ValueInfoProto.decode = function decode(reader, length) {
                if (!(reader instanceof $Reader))
                    reader = $Reader.create(reader);
                var end = length === undefined ? reader.len : reader.pos + length, message = new $root.onnx.ValueInfoProto();
                while (reader.pos < end) {
                    var tag = reader.uint32();
                    switch (tag >>> 3) {
                    case 1:
                        message.name = reader.string();
                        break;
                    case 2:
                        message.type = $root.onnx.TypeProto.decode(reader, reader.uint32());
                        break;
                    case 3:
                        message.docString = reader.string();
                        break;
                    default:
                        reader.skipType(tag & 7);
                        break;
                    }
                }
                return message;
            };
    
            ValueInfoProto.verify = function verify(message) {
                if (typeof message !== "object" || message === null)
                    return "object expected";
                if (message.name != null && message.hasOwnProperty("name"))
                    if (!$util.isString(message.name))
                        return "name: string expected";
                if (message.type != null && message.hasOwnProperty("type")) {
                    var error = $root.onnx.TypeProto.verify(message.type);
                    if (error)
                        return "type." + error;
                }
                if (message.docString != null && message.hasOwnProperty("docString"))
                    if (!$util.isString(message.docString))
                        return "docString: string expected";
                return null;
            };
    
            ValueInfoProto.fromObject = function fromObject(object) {
                if (object instanceof $root.onnx.ValueInfoProto)
                    return object;
                var message = new $root.onnx.ValueInfoProto();
                if (object.name != null)
                    message.name = String(object.name);
                if (object.type != null) {
                    if (typeof object.type !== "object")
                        throw TypeError(".onnx.ValueInfoProto.type: object expected");
                    message.type = $root.onnx.TypeProto.fromObject(object.type);
                }
                if (object.docString != null)
                    message.docString = String(object.docString);
                return message;
            };
    
            ValueInfoProto.toObject = function toObject(message, options) {
                if (!options)
                    options = {};
                var object = {};
                if (options.defaults) {
                    object.name = "";
                    object.type = null;
                    object.docString = "";
                }
                if (message.name != null && message.hasOwnProperty("name"))
                    object.name = message.name;
                if (message.type != null && message.hasOwnProperty("type"))
                    object.type = $root.onnx.TypeProto.toObject(message.type, options);
                if (message.docString != null && message.hasOwnProperty("docString"))
                    object.docString = message.docString;
                return object;
            };
    
            ValueInfoProto.prototype.toJSON = function toJSON() {
                return this.constructor.toObject(this, $protobuf.util.toJSONOptions);
            };
    
            return ValueInfoProto;
        })();
    
        onnx.NodeProto = (function() {
    
            function NodeProto(properties) {
                this.input = [];
                this.output = [];
                this.attribute = [];
                if (properties)
                    for (var keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                        if (properties[keys[i]] != null)
                            this[keys[i]] = properties[keys[i]];
            }
    
            NodeProto.prototype.input = $util.emptyArray;
            NodeProto.prototype.output = $util.emptyArray;
            NodeProto.prototype.name = "";
            NodeProto.prototype.opType = "";
            NodeProto.prototype.domain = "";
            NodeProto.prototype.attribute = $util.emptyArray;
            NodeProto.prototype.docString = "";
    
            NodeProto.create = function create(properties) {
                return new NodeProto(properties);
            };
    
            NodeProto.decode = function decode(reader, length) {
                if (!(reader instanceof $Reader))
                    reader = $Reader.create(reader);
                var end = length === undefined ? reader.len : reader.pos + length, message = new $root.onnx.NodeProto();
                while (reader.pos < end) {
                    var tag = reader.uint32();
                    switch (tag >>> 3) {
                    case 1:
                        if (!(message.input && message.input.length))
                            message.input = [];
                        message.input.push(reader.string());
                        break;
                    case 2:
                        if (!(message.output && message.output.length))
                            message.output = [];
                        message.output.push(reader.string());
                        break;
                    case 3:
                        message.name = reader.string();
                        break;
                    case 4:
                        message.opType = reader.string();
                        break;
                    case 7:
                        message.domain = reader.string();
                        break;
                    case 5:
                        if (!(message.attribute && message.attribute.length))
                            message.attribute = [];
                        message.attribute.push($root.onnx.AttributeProto.decode(reader, reader.uint32()));
                        break;
                    case 6:
                        message.docString = reader.string();
                        break;
                    default:
                        reader.skipType(tag & 7);
                        break;
                    }
                }
                return message;
            };
    
            NodeProto.verify = function verify(message) {
                if (typeof message !== "object" || message === null)
                    return "object expected";
                if (message.input != null && message.hasOwnProperty("input")) {
                    if (!Array.isArray(message.input))
                        return "input: array expected";
                    for (var i = 0; i < message.input.length; ++i)
                        if (!$util.isString(message.input[i]))
                            return "input: string[] expected";
                }
                if (message.output != null && message.hasOwnProperty("output")) {
                    if (!Array.isArray(message.output))
                        return "output: array expected";
                    for (var i = 0; i < message.output.length; ++i)
                        if (!$util.isString(message.output[i]))
                            return "output: string[] expected";
                }
                if (message.name != null && message.hasOwnProperty("name"))
                    if (!$util.isString(message.name))
                        return "name: string expected";
                if (message.opType != null && message.hasOwnProperty("opType"))
                    if (!$util.isString(message.opType))
                        return "opType: string expected";
                if (message.domain != null && message.hasOwnProperty("domain"))
                    if (!$util.isString(message.domain))
                        return "domain: string expected";
                if (message.attribute != null && message.hasOwnProperty("attribute")) {
                    if (!Array.isArray(message.attribute))
                        return "attribute: array expected";
                    for (var i = 0; i < message.attribute.length; ++i) {
                        var error = $root.onnx.AttributeProto.verify(message.attribute[i]);
                        if (error)
                            return "attribute." + error;
                    }
                }
                if (message.docString != null && message.hasOwnProperty("docString"))
                    if (!$util.isString(message.docString))
                        return "docString: string expected";
                return null;
            };
    
            NodeProto.fromObject = function fromObject(object) {
                if (object instanceof $root.onnx.NodeProto)
                    return object;
                var message = new $root.onnx.NodeProto();
                if (object.input) {
                    if (!Array.isArray(object.input))
                        throw TypeError(".onnx.NodeProto.input: array expected");
                    message.input = [];
                    for (var i = 0; i < object.input.length; ++i)
                        message.input[i] = String(object.input[i]);
                }
                if (object.output) {
                    if (!Array.isArray(object.output))
                        throw TypeError(".onnx.NodeProto.output: array expected");
                    message.output = [];
                    for (var i = 0; i < object.output.length; ++i)
                        message.output[i] = String(object.output[i]);
                }
                if (object.name != null)
                    message.name = String(object.name);
                if (object.opType != null)
                    message.opType = String(object.opType);
                if (object.domain != null)
                    message.domain = String(object.domain);
                if (object.attribute) {
                    if (!Array.isArray(object.attribute))
                        throw TypeError(".onnx.NodeProto.attribute: array expected");
                    message.attribute = [];
                    for (var i = 0; i < object.attribute.length; ++i) {
                        if (typeof object.attribute[i] !== "object")
                            throw TypeError(".onnx.NodeProto.attribute: object expected");
                        message.attribute[i] = $root.onnx.AttributeProto.fromObject(object.attribute[i]);
                    }
                }
                if (object.docString != null)
                    message.docString = String(object.docString);
                return message;
            };
    
            NodeProto.toObject = function toObject(message, options) {
                if (!options)
                    options = {};
                var object = {};
                if (options.arrays || options.defaults) {
                    object.input = [];
                    object.output = [];
                    object.attribute = [];
                }
                if (options.defaults) {
                    object.name = "";
                    object.opType = "";
                    object.docString = "";
                    object.domain = "";
                }
                if (message.input && message.input.length) {
                    object.input = [];
                    for (var j = 0; j < message.input.length; ++j)
                        object.input[j] = message.input[j];
                }
                if (message.output && message.output.length) {
                    object.output = [];
                    for (var j = 0; j < message.output.length; ++j)
                        object.output[j] = message.output[j];
                }
                if (message.name != null && message.hasOwnProperty("name"))
                    object.name = message.name;
                if (message.opType != null && message.hasOwnProperty("opType"))
                    object.opType = message.opType;
                if (message.attribute && message.attribute.length) {
                    object.attribute = [];
                    for (var j = 0; j < message.attribute.length; ++j)
                        object.attribute[j] = $root.onnx.AttributeProto.toObject(message.attribute[j], options);
                }
                if (message.docString != null && message.hasOwnProperty("docString"))
                    object.docString = message.docString;
                if (message.domain != null && message.hasOwnProperty("domain"))
                    object.domain = message.domain;
                return object;
            };
    
            NodeProto.prototype.toJSON = function toJSON() {
                return this.constructor.toObject(this, $protobuf.util.toJSONOptions);
            };
    
            return NodeProto;
        })();
    
        onnx.ModelProto = (function() {
    
            function ModelProto(properties) {
                this.opsetImport = [];
                this.metadataProps = [];
                if (properties)
                    for (var keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                        if (properties[keys[i]] != null)
                            this[keys[i]] = properties[keys[i]];
            }
    
            ModelProto.prototype.irVersion = $util.Long ? $util.Long.fromBits(0,0,false) : 0;
            ModelProto.prototype.opsetImport = $util.emptyArray;
            ModelProto.prototype.producerName = "";
            ModelProto.prototype.producerVersion = "";
            ModelProto.prototype.domain = "";
            ModelProto.prototype.modelVersion = $util.Long ? $util.Long.fromBits(0,0,false) : 0;
            ModelProto.prototype.docString = "";
            ModelProto.prototype.graph = null;
            ModelProto.prototype.metadataProps = $util.emptyArray;
    
            ModelProto.create = function create(properties) {
                return new ModelProto(properties);
            };
    
            ModelProto.decode = function decode(reader, length) {
                if (!(reader instanceof $Reader))
                    reader = $Reader.create(reader);
                var end = length === undefined ? reader.len : reader.pos + length, message = new $root.onnx.ModelProto();
                while (reader.pos < end) {
                    var tag = reader.uint32();
                    switch (tag >>> 3) {
                    case 1:
                        message.irVersion = reader.int64();
                        break;
                    case 8:
                        if (!(message.opsetImport && message.opsetImport.length))
                            message.opsetImport = [];
                        message.opsetImport.push($root.onnx.OperatorSetIdProto.decode(reader, reader.uint32()));
                        break;
                    case 2:
                        message.producerName = reader.string();
                        break;
                    case 3:
                        message.producerVersion = reader.string();
                        break;
                    case 4:
                        message.domain = reader.string();
                        break;
                    case 5:
                        message.modelVersion = reader.int64();
                        break;
                    case 6:
                        message.docString = reader.string();
                        break;
                    case 7:
                        message.graph = $root.onnx.GraphProto.decode(reader, reader.uint32());
                        break;
                    case 14:
                        if (!(message.metadataProps && message.metadataProps.length))
                            message.metadataProps = [];
                        message.metadataProps.push($root.onnx.StringStringEntryProto.decode(reader, reader.uint32()));
                        break;
                    default:
                        reader.skipType(tag & 7);
                        break;
                    }
                }
                return message;
            };
    
            ModelProto.verify = function verify(message) {
                if (typeof message !== "object" || message === null)
                    return "object expected";
                if (message.irVersion != null && message.hasOwnProperty("irVersion"))
                    if (!$util.isInteger(message.irVersion) && !(message.irVersion && $util.isInteger(message.irVersion.low) && $util.isInteger(message.irVersion.high)))
                        return "irVersion: integer|Long expected";
                if (message.opsetImport != null && message.hasOwnProperty("opsetImport")) {
                    if (!Array.isArray(message.opsetImport))
                        return "opsetImport: array expected";
                    for (var i = 0; i < message.opsetImport.length; ++i) {
                        var error = $root.onnx.OperatorSetIdProto.verify(message.opsetImport[i]);
                        if (error)
                            return "opsetImport." + error;
                    }
                }
                if (message.producerName != null && message.hasOwnProperty("producerName"))
                    if (!$util.isString(message.producerName))
                        return "producerName: string expected";
                if (message.producerVersion != null && message.hasOwnProperty("producerVersion"))
                    if (!$util.isString(message.producerVersion))
                        return "producerVersion: string expected";
                if (message.domain != null && message.hasOwnProperty("domain"))
                    if (!$util.isString(message.domain))
                        return "domain: string expected";
                if (message.modelVersion != null && message.hasOwnProperty("modelVersion"))
                    if (!$util.isInteger(message.modelVersion) && !(message.modelVersion && $util.isInteger(message.modelVersion.low) && $util.isInteger(message.modelVersion.high)))
                        return "modelVersion: integer|Long expected";
                if (message.docString != null && message.hasOwnProperty("docString"))
                    if (!$util.isString(message.docString))
                        return "docString: string expected";
                if (message.graph != null && message.hasOwnProperty("graph")) {
                    var error = $root.onnx.GraphProto.verify(message.graph);
                    if (error)
                        return "graph." + error;
                }
                if (message.metadataProps != null && message.hasOwnProperty("metadataProps")) {
                    if (!Array.isArray(message.metadataProps))
                        return "metadataProps: array expected";
                    for (var i = 0; i < message.metadataProps.length; ++i) {
                        var error = $root.onnx.StringStringEntryProto.verify(message.metadataProps[i]);
                        if (error)
                            return "metadataProps." + error;
                    }
                }
                return null;
            };
    
            ModelProto.fromObject = function fromObject(object) {
                if (object instanceof $root.onnx.ModelProto)
                    return object;
                var message = new $root.onnx.ModelProto();
                if (object.irVersion != null)
                    if ($util.Long)
                        (message.irVersion = $util.Long.fromValue(object.irVersion)).unsigned = false;
                    else if (typeof object.irVersion === "string")
                        message.irVersion = parseInt(object.irVersion, 10);
                    else if (typeof object.irVersion === "number")
                        message.irVersion = object.irVersion;
                    else if (typeof object.irVersion === "object")
                        message.irVersion = new $util.LongBits(object.irVersion.low >>> 0, object.irVersion.high >>> 0).toNumber();
                if (object.opsetImport) {
                    if (!Array.isArray(object.opsetImport))
                        throw TypeError(".onnx.ModelProto.opsetImport: array expected");
                    message.opsetImport = [];
                    for (var i = 0; i < object.opsetImport.length; ++i) {
                        if (typeof object.opsetImport[i] !== "object")
                            throw TypeError(".onnx.ModelProto.opsetImport: object expected");
                        message.opsetImport[i] = $root.onnx.OperatorSetIdProto.fromObject(object.opsetImport[i]);
                    }
                }
                if (object.producerName != null)
                    message.producerName = String(object.producerName);
                if (object.producerVersion != null)
                    message.producerVersion = String(object.producerVersion);
                if (object.domain != null)
                    message.domain = String(object.domain);
                if (object.modelVersion != null)
                    if ($util.Long)
                        (message.modelVersion = $util.Long.fromValue(object.modelVersion)).unsigned = false;
                    else if (typeof object.modelVersion === "string")
                        message.modelVersion = parseInt(object.modelVersion, 10);
                    else if (typeof object.modelVersion === "number")
                        message.modelVersion = object.modelVersion;
                    else if (typeof object.modelVersion === "object")
                        message.modelVersion = new $util.LongBits(object.modelVersion.low >>> 0, object.modelVersion.high >>> 0).toNumber();
                if (object.docString != null)
                    message.docString = String(object.docString);
                if (object.graph != null) {
                    if (typeof object.graph !== "object")
                        throw TypeError(".onnx.ModelProto.graph: object expected");
                    message.graph = $root.onnx.GraphProto.fromObject(object.graph);
                }
                if (object.metadataProps) {
                    if (!Array.isArray(object.metadataProps))
                        throw TypeError(".onnx.ModelProto.metadataProps: array expected");
                    message.metadataProps = [];
                    for (var i = 0; i < object.metadataProps.length; ++i) {
                        if (typeof object.metadataProps[i] !== "object")
                            throw TypeError(".onnx.ModelProto.metadataProps: object expected");
                        message.metadataProps[i] = $root.onnx.StringStringEntryProto.fromObject(object.metadataProps[i]);
                    }
                }
                return message;
            };
    
            ModelProto.toObject = function toObject(message, options) {
                if (!options)
                    options = {};
                var object = {};
                if (options.arrays || options.defaults) {
                    object.opsetImport = [];
                    object.metadataProps = [];
                }
                if (options.defaults) {
                    if ($util.Long) {
                        var long = new $util.Long(0, 0, false);
                        object.irVersion = options.longs === String ? long.toString() : options.longs === Number ? long.toNumber() : long;
                    } else
                        object.irVersion = options.longs === String ? "0" : 0;
                    object.producerName = "";
                    object.producerVersion = "";
                    object.domain = "";
                    if ($util.Long) {
                        var long = new $util.Long(0, 0, false);
                        object.modelVersion = options.longs === String ? long.toString() : options.longs === Number ? long.toNumber() : long;
                    } else
                        object.modelVersion = options.longs === String ? "0" : 0;
                    object.docString = "";
                    object.graph = null;
                }
                if (message.irVersion != null && message.hasOwnProperty("irVersion"))
                    if (typeof message.irVersion === "number")
                        object.irVersion = options.longs === String ? String(message.irVersion) : message.irVersion;
                    else
                        object.irVersion = options.longs === String ? $util.Long.prototype.toString.call(message.irVersion) : options.longs === Number ? new $util.LongBits(message.irVersion.low >>> 0, message.irVersion.high >>> 0).toNumber() : message.irVersion;
                if (message.producerName != null && message.hasOwnProperty("producerName"))
                    object.producerName = message.producerName;
                if (message.producerVersion != null && message.hasOwnProperty("producerVersion"))
                    object.producerVersion = message.producerVersion;
                if (message.domain != null && message.hasOwnProperty("domain"))
                    object.domain = message.domain;
                if (message.modelVersion != null && message.hasOwnProperty("modelVersion"))
                    if (typeof message.modelVersion === "number")
                        object.modelVersion = options.longs === String ? String(message.modelVersion) : message.modelVersion;
                    else
                        object.modelVersion = options.longs === String ? $util.Long.prototype.toString.call(message.modelVersion) : options.longs === Number ? new $util.LongBits(message.modelVersion.low >>> 0, message.modelVersion.high >>> 0).toNumber() : message.modelVersion;
                if (message.docString != null && message.hasOwnProperty("docString"))
                    object.docString = message.docString;
                if (message.graph != null && message.hasOwnProperty("graph"))
                    object.graph = $root.onnx.GraphProto.toObject(message.graph, options);
                if (message.opsetImport && message.opsetImport.length) {
                    object.opsetImport = [];
                    for (var j = 0; j < message.opsetImport.length; ++j)
                        object.opsetImport[j] = $root.onnx.OperatorSetIdProto.toObject(message.opsetImport[j], options);
                }
                if (message.metadataProps && message.metadataProps.length) {
                    object.metadataProps = [];
                    for (var j = 0; j < message.metadataProps.length; ++j)
                        object.metadataProps[j] = $root.onnx.StringStringEntryProto.toObject(message.metadataProps[j], options);
                }
                return object;
            };
    
            ModelProto.prototype.toJSON = function toJSON() {
                return this.constructor.toObject(this, $protobuf.util.toJSONOptions);
            };
    
            return ModelProto;
        })();
    
        onnx.StringStringEntryProto = (function() {
    
            function StringStringEntryProto(properties) {
                if (properties)
                    for (var keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                        if (properties[keys[i]] != null)
                            this[keys[i]] = properties[keys[i]];
            }
    
            StringStringEntryProto.prototype.key = "";
            StringStringEntryProto.prototype.value = "";
    
            StringStringEntryProto.create = function create(properties) {
                return new StringStringEntryProto(properties);
            };
    
            StringStringEntryProto.decode = function decode(reader, length) {
                if (!(reader instanceof $Reader))
                    reader = $Reader.create(reader);
                var end = length === undefined ? reader.len : reader.pos + length, message = new $root.onnx.StringStringEntryProto();
                while (reader.pos < end) {
                    var tag = reader.uint32();
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
            };
    
            StringStringEntryProto.verify = function verify(message) {
                if (typeof message !== "object" || message === null)
                    return "object expected";
                if (message.key != null && message.hasOwnProperty("key"))
                    if (!$util.isString(message.key))
                        return "key: string expected";
                if (message.value != null && message.hasOwnProperty("value"))
                    if (!$util.isString(message.value))
                        return "value: string expected";
                return null;
            };
    
            StringStringEntryProto.fromObject = function fromObject(object) {
                if (object instanceof $root.onnx.StringStringEntryProto)
                    return object;
                var message = new $root.onnx.StringStringEntryProto();
                if (object.key != null)
                    message.key = String(object.key);
                if (object.value != null)
                    message.value = String(object.value);
                return message;
            };
    
            StringStringEntryProto.toObject = function toObject(message, options) {
                if (!options)
                    options = {};
                var object = {};
                if (options.defaults) {
                    object.key = "";
                    object.value = "";
                }
                if (message.key != null && message.hasOwnProperty("key"))
                    object.key = message.key;
                if (message.value != null && message.hasOwnProperty("value"))
                    object.value = message.value;
                return object;
            };
    
            StringStringEntryProto.prototype.toJSON = function toJSON() {
                return this.constructor.toObject(this, $protobuf.util.toJSONOptions);
            };
    
            return StringStringEntryProto;
        })();
    
        onnx.GraphProto = (function() {
    
            function GraphProto(properties) {
                this.node = [];
                this.initializer = [];
                this.input = [];
                this.output = [];
                this.valueInfo = [];
                if (properties)
                    for (var keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                        if (properties[keys[i]] != null)
                            this[keys[i]] = properties[keys[i]];
            }
    
            GraphProto.prototype.node = $util.emptyArray;
            GraphProto.prototype.name = "";
            GraphProto.prototype.initializer = $util.emptyArray;
            GraphProto.prototype.docString = "";
            GraphProto.prototype.input = $util.emptyArray;
            GraphProto.prototype.output = $util.emptyArray;
            GraphProto.prototype.valueInfo = $util.emptyArray;
    
            GraphProto.create = function create(properties) {
                return new GraphProto(properties);
            };
    
            GraphProto.decode = function decode(reader, length) {
                if (!(reader instanceof $Reader))
                    reader = $Reader.create(reader);
                var end = length === undefined ? reader.len : reader.pos + length, message = new $root.onnx.GraphProto();
                while (reader.pos < end) {
                    var tag = reader.uint32();
                    switch (tag >>> 3) {
                    case 1:
                        if (!(message.node && message.node.length))
                            message.node = [];
                        message.node.push($root.onnx.NodeProto.decode(reader, reader.uint32()));
                        break;
                    case 2:
                        message.name = reader.string();
                        break;
                    case 5:
                        if (!(message.initializer && message.initializer.length))
                            message.initializer = [];
                        message.initializer.push($root.onnx.TensorProto.decode(reader, reader.uint32()));
                        break;
                    case 10:
                        message.docString = reader.string();
                        break;
                    case 11:
                        if (!(message.input && message.input.length))
                            message.input = [];
                        message.input.push($root.onnx.ValueInfoProto.decode(reader, reader.uint32()));
                        break;
                    case 12:
                        if (!(message.output && message.output.length))
                            message.output = [];
                        message.output.push($root.onnx.ValueInfoProto.decode(reader, reader.uint32()));
                        break;
                    case 13:
                        if (!(message.valueInfo && message.valueInfo.length))
                            message.valueInfo = [];
                        message.valueInfo.push($root.onnx.ValueInfoProto.decode(reader, reader.uint32()));
                        break;
                    default:
                        reader.skipType(tag & 7);
                        break;
                    }
                }
                return message;
            };
    
            GraphProto.verify = function verify(message) {
                if (typeof message !== "object" || message === null)
                    return "object expected";
                if (message.node != null && message.hasOwnProperty("node")) {
                    if (!Array.isArray(message.node))
                        return "node: array expected";
                    for (var i = 0; i < message.node.length; ++i) {
                        var error = $root.onnx.NodeProto.verify(message.node[i]);
                        if (error)
                            return "node." + error;
                    }
                }
                if (message.name != null && message.hasOwnProperty("name"))
                    if (!$util.isString(message.name))
                        return "name: string expected";
                if (message.initializer != null && message.hasOwnProperty("initializer")) {
                    if (!Array.isArray(message.initializer))
                        return "initializer: array expected";
                    for (var i = 0; i < message.initializer.length; ++i) {
                        var error = $root.onnx.TensorProto.verify(message.initializer[i]);
                        if (error)
                            return "initializer." + error;
                    }
                }
                if (message.docString != null && message.hasOwnProperty("docString"))
                    if (!$util.isString(message.docString))
                        return "docString: string expected";
                if (message.input != null && message.hasOwnProperty("input")) {
                    if (!Array.isArray(message.input))
                        return "input: array expected";
                    for (var i = 0; i < message.input.length; ++i) {
                        var error = $root.onnx.ValueInfoProto.verify(message.input[i]);
                        if (error)
                            return "input." + error;
                    }
                }
                if (message.output != null && message.hasOwnProperty("output")) {
                    if (!Array.isArray(message.output))
                        return "output: array expected";
                    for (var i = 0; i < message.output.length; ++i) {
                        var error = $root.onnx.ValueInfoProto.verify(message.output[i]);
                        if (error)
                            return "output." + error;
                    }
                }
                if (message.valueInfo != null && message.hasOwnProperty("valueInfo")) {
                    if (!Array.isArray(message.valueInfo))
                        return "valueInfo: array expected";
                    for (var i = 0; i < message.valueInfo.length; ++i) {
                        var error = $root.onnx.ValueInfoProto.verify(message.valueInfo[i]);
                        if (error)
                            return "valueInfo." + error;
                    }
                }
                return null;
            };
    
            GraphProto.fromObject = function fromObject(object) {
                if (object instanceof $root.onnx.GraphProto)
                    return object;
                var message = new $root.onnx.GraphProto();
                if (object.node) {
                    if (!Array.isArray(object.node))
                        throw TypeError(".onnx.GraphProto.node: array expected");
                    message.node = [];
                    for (var i = 0; i < object.node.length; ++i) {
                        if (typeof object.node[i] !== "object")
                            throw TypeError(".onnx.GraphProto.node: object expected");
                        message.node[i] = $root.onnx.NodeProto.fromObject(object.node[i]);
                    }
                }
                if (object.name != null)
                    message.name = String(object.name);
                if (object.initializer) {
                    if (!Array.isArray(object.initializer))
                        throw TypeError(".onnx.GraphProto.initializer: array expected");
                    message.initializer = [];
                    for (var i = 0; i < object.initializer.length; ++i) {
                        if (typeof object.initializer[i] !== "object")
                            throw TypeError(".onnx.GraphProto.initializer: object expected");
                        message.initializer[i] = $root.onnx.TensorProto.fromObject(object.initializer[i]);
                    }
                }
                if (object.docString != null)
                    message.docString = String(object.docString);
                if (object.input) {
                    if (!Array.isArray(object.input))
                        throw TypeError(".onnx.GraphProto.input: array expected");
                    message.input = [];
                    for (var i = 0; i < object.input.length; ++i) {
                        if (typeof object.input[i] !== "object")
                            throw TypeError(".onnx.GraphProto.input: object expected");
                        message.input[i] = $root.onnx.ValueInfoProto.fromObject(object.input[i]);
                    }
                }
                if (object.output) {
                    if (!Array.isArray(object.output))
                        throw TypeError(".onnx.GraphProto.output: array expected");
                    message.output = [];
                    for (var i = 0; i < object.output.length; ++i) {
                        if (typeof object.output[i] !== "object")
                            throw TypeError(".onnx.GraphProto.output: object expected");
                        message.output[i] = $root.onnx.ValueInfoProto.fromObject(object.output[i]);
                    }
                }
                if (object.valueInfo) {
                    if (!Array.isArray(object.valueInfo))
                        throw TypeError(".onnx.GraphProto.valueInfo: array expected");
                    message.valueInfo = [];
                    for (var i = 0; i < object.valueInfo.length; ++i) {
                        if (typeof object.valueInfo[i] !== "object")
                            throw TypeError(".onnx.GraphProto.valueInfo: object expected");
                        message.valueInfo[i] = $root.onnx.ValueInfoProto.fromObject(object.valueInfo[i]);
                    }
                }
                return message;
            };
    
            GraphProto.toObject = function toObject(message, options) {
                if (!options)
                    options = {};
                var object = {};
                if (options.arrays || options.defaults) {
                    object.node = [];
                    object.initializer = [];
                    object.input = [];
                    object.output = [];
                    object.valueInfo = [];
                }
                if (options.defaults) {
                    object.name = "";
                    object.docString = "";
                }
                if (message.node && message.node.length) {
                    object.node = [];
                    for (var j = 0; j < message.node.length; ++j)
                        object.node[j] = $root.onnx.NodeProto.toObject(message.node[j], options);
                }
                if (message.name != null && message.hasOwnProperty("name"))
                    object.name = message.name;
                if (message.initializer && message.initializer.length) {
                    object.initializer = [];
                    for (var j = 0; j < message.initializer.length; ++j)
                        object.initializer[j] = $root.onnx.TensorProto.toObject(message.initializer[j], options);
                }
                if (message.docString != null && message.hasOwnProperty("docString"))
                    object.docString = message.docString;
                if (message.input && message.input.length) {
                    object.input = [];
                    for (var j = 0; j < message.input.length; ++j)
                        object.input[j] = $root.onnx.ValueInfoProto.toObject(message.input[j], options);
                }
                if (message.output && message.output.length) {
                    object.output = [];
                    for (var j = 0; j < message.output.length; ++j)
                        object.output[j] = $root.onnx.ValueInfoProto.toObject(message.output[j], options);
                }
                if (message.valueInfo && message.valueInfo.length) {
                    object.valueInfo = [];
                    for (var j = 0; j < message.valueInfo.length; ++j)
                        object.valueInfo[j] = $root.onnx.ValueInfoProto.toObject(message.valueInfo[j], options);
                }
                return object;
            };
    
            GraphProto.prototype.toJSON = function toJSON() {
                return this.constructor.toObject(this, $protobuf.util.toJSONOptions);
            };
    
            return GraphProto;
        })();
    
        onnx.TensorProto = (function() {
    
            function TensorProto(properties) {
                this.dims = [];
                this.floatData = [];
                this.int32Data = [];
                this.stringData = [];
                this.int64Data = [];
                this.doubleData = [];
                this.uint64Data = [];
                if (properties)
                    for (var keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                        if (properties[keys[i]] != null)
                            this[keys[i]] = properties[keys[i]];
            }
    
            TensorProto.prototype.dims = $util.emptyArray;
            TensorProto.prototype.dataType = 0;
            TensorProto.prototype.segment = null;
            TensorProto.prototype.floatData = $util.emptyArray;
            TensorProto.prototype.int32Data = $util.emptyArray;
            TensorProto.prototype.stringData = $util.emptyArray;
            TensorProto.prototype.int64Data = $util.emptyArray;
            TensorProto.prototype.name = "";
            TensorProto.prototype.docString = "";
            TensorProto.prototype.rawData = $util.newBuffer([]);
            TensorProto.prototype.doubleData = $util.emptyArray;
            TensorProto.prototype.uint64Data = $util.emptyArray;
    
            TensorProto.create = function create(properties) {
                return new TensorProto(properties);
            };
    
            TensorProto.decode = function decode(reader, length) {
                if (!(reader instanceof $Reader))
                    reader = $Reader.create(reader);
                var end = length === undefined ? reader.len : reader.pos + length, message = new $root.onnx.TensorProto();
                while (reader.pos < end) {
                    var tag = reader.uint32();
                    switch (tag >>> 3) {
                    case 1:
                        if (!(message.dims && message.dims.length))
                            message.dims = [];
                        if ((tag & 7) === 2) {
                            var end2 = reader.uint32() + reader.pos;
                            while (reader.pos < end2)
                                message.dims.push(reader.int64());
                        } else
                            message.dims.push(reader.int64());
                        break;
                    case 2:
                        message.dataType = reader.int32();
                        break;
                    case 3:
                        message.segment = $root.onnx.TensorProto.Segment.decode(reader, reader.uint32());
                        break;
                    case 4:
                        if (!(message.floatData && message.floatData.length))
                            message.floatData = [];
                        if ((tag & 7) === 2) {
                            var end2 = reader.uint32() + reader.pos;
                            if (message.floatData.length == 0 && (end2 - reader.pos) > 1048576) {
                                var floatDataLength = end2 - reader.pos;
                                var floatDataView = new DataView(reader.buf.buffer, reader.buf.byteOffset + reader.pos, floatDataLength);
                                floatDataLength = floatDataLength >>> 2;
                                var floatData = new Float32Array(floatDataLength);
                                for (var i = 0; i < floatDataLength; i++) {
                                    floatData[i] = floatDataView.getFloat32(i << 2, true);
                                }
                                message.floatData = floatData;
                                reader.pos = end2;
                            }
                            else {
                                while (reader.pos < end2)
                                    message.floatData.push(reader.float());
                            }
                        } else
                            message.floatData.push(reader.float());
                        break;
                    case 5:
                        if (!(message.int32Data && message.int32Data.length))
                            message.int32Data = [];
                        if ((tag & 7) === 2) {
                            var end2 = reader.uint32() + reader.pos;
                            while (reader.pos < end2)
                                message.int32Data.push(reader.int32());
                        } else
                            message.int32Data.push(reader.int32());
                        break;
                    case 6:
                        if (!(message.stringData && message.stringData.length))
                            message.stringData = [];
                        message.stringData.push(reader.bytes());
                        break;
                    case 7:
                        if (!(message.int64Data && message.int64Data.length))
                            message.int64Data = [];
                        if ((tag & 7) === 2) {
                            var end2 = reader.uint32() + reader.pos;
                            while (reader.pos < end2)
                                message.int64Data.push(reader.int64());
                        } else
                            message.int64Data.push(reader.int64());
                        break;
                    case 8:
                        message.name = reader.string();
                        break;
                    case 12:
                        message.docString = reader.string();
                        break;
                    case 9:
                        message.rawData = reader.bytes();
                        break;
                    case 10:
                        if (!(message.doubleData && message.doubleData.length))
                            message.doubleData = [];
                        if ((tag & 7) === 2) {
                            var end2 = reader.uint32() + reader.pos;
                            if (message.doubleData.length == 0 && (end2 - reader.pos) > 1048576) {
                                var doubleDataLength = end2 - reader.pos;
                                var doubleDataView = new DataView(reader.buf.buffer, reader.buf.byteOffset + reader.pos, doubleDataLength);
                                doubleDataLength = doubleDataLength >>> 3;
                                var doubleData = new Float64Array(doubleDataLength);
                                for (var i = 0; i < doubleDataLength; i++) {
                                    doubleData[i] = doubleDataView.getFloat64(i << 3, true);
                                }
                                message.doubleData = doubleData;
                                reader.pos = end2;
                            }
                            else {
                                while (reader.pos < end2)
                                    message.doubleData.push(reader.double());
                            }
                        } else
                            message.doubleData.push(reader.double());
                        break;
                    case 11:
                        if (!(message.uint64Data && message.uint64Data.length))
                            message.uint64Data = [];
                        if ((tag & 7) === 2) {
                            var end2 = reader.uint32() + reader.pos;
                            while (reader.pos < end2)
                                message.uint64Data.push(reader.uint64());
                        } else
                            message.uint64Data.push(reader.uint64());
                        break;
                    default:
                        reader.skipType(tag & 7);
                        break;
                    }
                }
                return message;
            };
    
            TensorProto.verify = function verify(message) {
                if (typeof message !== "object" || message === null)
                    return "object expected";
                if (message.dims != null && message.hasOwnProperty("dims")) {
                    if (!Array.isArray(message.dims))
                        return "dims: array expected";
                    for (var i = 0; i < message.dims.length; ++i)
                        if (!$util.isInteger(message.dims[i]) && !(message.dims[i] && $util.isInteger(message.dims[i].low) && $util.isInteger(message.dims[i].high)))
                            return "dims: integer|Long[] expected";
                }
                if (message.dataType != null && message.hasOwnProperty("dataType"))
                    switch (message.dataType) {
                    default:
                        return "dataType: enum value expected";
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
                    case 12:
                    case 13:
                    case 14:
                    case 15:
                        break;
                    }
                if (message.segment != null && message.hasOwnProperty("segment")) {
                    var error = $root.onnx.TensorProto.Segment.verify(message.segment);
                    if (error)
                        return "segment." + error;
                }
                if (message.floatData != null && message.hasOwnProperty("floatData")) {
                    if (!Array.isArray(message.floatData))
                        return "floatData: array expected";
                    for (var i = 0; i < message.floatData.length; ++i)
                        if (typeof message.floatData[i] !== "number")
                            return "floatData: number[] expected";
                }
                if (message.int32Data != null && message.hasOwnProperty("int32Data")) {
                    if (!Array.isArray(message.int32Data))
                        return "int32Data: array expected";
                    for (var i = 0; i < message.int32Data.length; ++i)
                        if (!$util.isInteger(message.int32Data[i]))
                            return "int32Data: integer[] expected";
                }
                if (message.stringData != null && message.hasOwnProperty("stringData")) {
                    if (!Array.isArray(message.stringData))
                        return "stringData: array expected";
                    for (var i = 0; i < message.stringData.length; ++i)
                        if (!(message.stringData[i] && typeof message.stringData[i].length === "number" || $util.isString(message.stringData[i])))
                            return "stringData: buffer[] expected";
                }
                if (message.int64Data != null && message.hasOwnProperty("int64Data")) {
                    if (!Array.isArray(message.int64Data))
                        return "int64Data: array expected";
                    for (var i = 0; i < message.int64Data.length; ++i)
                        if (!$util.isInteger(message.int64Data[i]) && !(message.int64Data[i] && $util.isInteger(message.int64Data[i].low) && $util.isInteger(message.int64Data[i].high)))
                            return "int64Data: integer|Long[] expected";
                }
                if (message.name != null && message.hasOwnProperty("name"))
                    if (!$util.isString(message.name))
                        return "name: string expected";
                if (message.docString != null && message.hasOwnProperty("docString"))
                    if (!$util.isString(message.docString))
                        return "docString: string expected";
                if (message.rawData != null && message.hasOwnProperty("rawData"))
                    if (!(message.rawData && typeof message.rawData.length === "number" || $util.isString(message.rawData)))
                        return "rawData: buffer expected";
                if (message.doubleData != null && message.hasOwnProperty("doubleData")) {
                    if (!Array.isArray(message.doubleData))
                        return "doubleData: array expected";
                    for (var i = 0; i < message.doubleData.length; ++i)
                        if (typeof message.doubleData[i] !== "number")
                            return "doubleData: number[] expected";
                }
                if (message.uint64Data != null && message.hasOwnProperty("uint64Data")) {
                    if (!Array.isArray(message.uint64Data))
                        return "uint64Data: array expected";
                    for (var i = 0; i < message.uint64Data.length; ++i)
                        if (!$util.isInteger(message.uint64Data[i]) && !(message.uint64Data[i] && $util.isInteger(message.uint64Data[i].low) && $util.isInteger(message.uint64Data[i].high)))
                            return "uint64Data: integer|Long[] expected";
                }
                return null;
            };
    
            TensorProto.fromObject = function fromObject(object) {
                if (object instanceof $root.onnx.TensorProto)
                    return object;
                var message = new $root.onnx.TensorProto();
                if (object.dims) {
                    if (!Array.isArray(object.dims))
                        throw TypeError(".onnx.TensorProto.dims: array expected");
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
                switch (object.dataType) {
                case "UNDEFINED":
                case 0:
                    message.dataType = 0;
                    break;
                case "FLOAT":
                case 1:
                    message.dataType = 1;
                    break;
                case "UINT8":
                case 2:
                    message.dataType = 2;
                    break;
                case "INT8":
                case 3:
                    message.dataType = 3;
                    break;
                case "UINT16":
                case 4:
                    message.dataType = 4;
                    break;
                case "INT16":
                case 5:
                    message.dataType = 5;
                    break;
                case "INT32":
                case 6:
                    message.dataType = 6;
                    break;
                case "INT64":
                case 7:
                    message.dataType = 7;
                    break;
                case "STRING":
                case 8:
                    message.dataType = 8;
                    break;
                case "BOOL":
                case 9:
                    message.dataType = 9;
                    break;
                case "FLOAT16":
                case 10:
                    message.dataType = 10;
                    break;
                case "DOUBLE":
                case 11:
                    message.dataType = 11;
                    break;
                case "UINT32":
                case 12:
                    message.dataType = 12;
                    break;
                case "UINT64":
                case 13:
                    message.dataType = 13;
                    break;
                case "COMPLEX64":
                case 14:
                    message.dataType = 14;
                    break;
                case "COMPLEX128":
                case 15:
                    message.dataType = 15;
                    break;
                }
                if (object.segment != null) {
                    if (typeof object.segment !== "object")
                        throw TypeError(".onnx.TensorProto.segment: object expected");
                    message.segment = $root.onnx.TensorProto.Segment.fromObject(object.segment);
                }
                if (object.floatData) {
                    if (!Array.isArray(object.floatData))
                        throw TypeError(".onnx.TensorProto.floatData: array expected");
                    message.floatData = [];
                    for (var i = 0; i < object.floatData.length; ++i)
                        message.floatData[i] = Number(object.floatData[i]);
                }
                if (object.int32Data) {
                    if (!Array.isArray(object.int32Data))
                        throw TypeError(".onnx.TensorProto.int32Data: array expected");
                    message.int32Data = [];
                    for (var i = 0; i < object.int32Data.length; ++i)
                        message.int32Data[i] = object.int32Data[i] | 0;
                }
                if (object.stringData) {
                    if (!Array.isArray(object.stringData))
                        throw TypeError(".onnx.TensorProto.stringData: array expected");
                    message.stringData = [];
                    for (var i = 0; i < object.stringData.length; ++i)
                        if (typeof object.stringData[i] === "string")
                            $util.base64.decode(object.stringData[i], message.stringData[i] = $util.newBuffer($util.base64.length(object.stringData[i])), 0);
                        else if (object.stringData[i].length)
                            message.stringData[i] = object.stringData[i];
                }
                if (object.int64Data) {
                    if (!Array.isArray(object.int64Data))
                        throw TypeError(".onnx.TensorProto.int64Data: array expected");
                    message.int64Data = [];
                    for (var i = 0; i < object.int64Data.length; ++i)
                        if ($util.Long)
                            (message.int64Data[i] = $util.Long.fromValue(object.int64Data[i])).unsigned = false;
                        else if (typeof object.int64Data[i] === "string")
                            message.int64Data[i] = parseInt(object.int64Data[i], 10);
                        else if (typeof object.int64Data[i] === "number")
                            message.int64Data[i] = object.int64Data[i];
                        else if (typeof object.int64Data[i] === "object")
                            message.int64Data[i] = new $util.LongBits(object.int64Data[i].low >>> 0, object.int64Data[i].high >>> 0).toNumber();
                }
                if (object.name != null)
                    message.name = String(object.name);
                if (object.docString != null)
                    message.docString = String(object.docString);
                if (object.rawData != null)
                    if (typeof object.rawData === "string")
                        $util.base64.decode(object.rawData, message.rawData = $util.newBuffer($util.base64.length(object.rawData)), 0);
                    else if (object.rawData.length)
                        message.rawData = object.rawData;
                if (object.doubleData) {
                    if (!Array.isArray(object.doubleData))
                        throw TypeError(".onnx.TensorProto.doubleData: array expected");
                    message.doubleData = [];
                    for (var i = 0; i < object.doubleData.length; ++i)
                        message.doubleData[i] = Number(object.doubleData[i]);
                }
                if (object.uint64Data) {
                    if (!Array.isArray(object.uint64Data))
                        throw TypeError(".onnx.TensorProto.uint64Data: array expected");
                    message.uint64Data = [];
                    for (var i = 0; i < object.uint64Data.length; ++i)
                        if ($util.Long)
                            (message.uint64Data[i] = $util.Long.fromValue(object.uint64Data[i])).unsigned = true;
                        else if (typeof object.uint64Data[i] === "string")
                            message.uint64Data[i] = parseInt(object.uint64Data[i], 10);
                        else if (typeof object.uint64Data[i] === "number")
                            message.uint64Data[i] = object.uint64Data[i];
                        else if (typeof object.uint64Data[i] === "object")
                            message.uint64Data[i] = new $util.LongBits(object.uint64Data[i].low >>> 0, object.uint64Data[i].high >>> 0).toNumber(true);
                }
                return message;
            };
    
            TensorProto.toObject = function toObject(message, options) {
                if (!options)
                    options = {};
                var object = {};
                if (options.arrays || options.defaults) {
                    object.dims = [];
                    object.floatData = [];
                    object.int32Data = [];
                    object.stringData = [];
                    object.int64Data = [];
                    object.doubleData = [];
                    object.uint64Data = [];
                }
                if (options.defaults) {
                    object.dataType = options.enums === String ? "UNDEFINED" : 0;
                    object.segment = null;
                    object.name = "";
                    if (options.bytes === String)
                        object.rawData = "";
                    else {
                        object.rawData = [];
                        if (options.bytes !== Array)
                            object.rawData = $util.newBuffer(object.rawData);
                    }
                    object.docString = "";
                }
                if (message.dims && message.dims.length) {
                    object.dims = [];
                    for (var j = 0; j < message.dims.length; ++j)
                        if (typeof message.dims[j] === "number")
                            object.dims[j] = options.longs === String ? String(message.dims[j]) : message.dims[j];
                        else
                            object.dims[j] = options.longs === String ? $util.Long.prototype.toString.call(message.dims[j]) : options.longs === Number ? new $util.LongBits(message.dims[j].low >>> 0, message.dims[j].high >>> 0).toNumber() : message.dims[j];
                }
                if (message.dataType != null && message.hasOwnProperty("dataType"))
                    object.dataType = options.enums === String ? $root.onnx.TensorProto.DataType[message.dataType] : message.dataType;
                if (message.segment != null && message.hasOwnProperty("segment"))
                    object.segment = $root.onnx.TensorProto.Segment.toObject(message.segment, options);
                if (message.floatData && message.floatData.length) {
                    object.floatData = [];
                    for (var j = 0; j < message.floatData.length; ++j)
                        object.floatData[j] = options.json && !isFinite(message.floatData[j]) ? String(message.floatData[j]) : message.floatData[j];
                }
                if (message.int32Data && message.int32Data.length) {
                    object.int32Data = [];
                    for (var j = 0; j < message.int32Data.length; ++j)
                        object.int32Data[j] = message.int32Data[j];
                }
                if (message.stringData && message.stringData.length) {
                    object.stringData = [];
                    for (var j = 0; j < message.stringData.length; ++j)
                        object.stringData[j] = options.bytes === String ? $util.base64.encode(message.stringData[j], 0, message.stringData[j].length) : options.bytes === Array ? Array.prototype.slice.call(message.stringData[j]) : message.stringData[j];
                }
                if (message.int64Data && message.int64Data.length) {
                    object.int64Data = [];
                    for (var j = 0; j < message.int64Data.length; ++j)
                        if (typeof message.int64Data[j] === "number")
                            object.int64Data[j] = options.longs === String ? String(message.int64Data[j]) : message.int64Data[j];
                        else
                            object.int64Data[j] = options.longs === String ? $util.Long.prototype.toString.call(message.int64Data[j]) : options.longs === Number ? new $util.LongBits(message.int64Data[j].low >>> 0, message.int64Data[j].high >>> 0).toNumber() : message.int64Data[j];
                }
                if (message.name != null && message.hasOwnProperty("name"))
                    object.name = message.name;
                if (message.rawData != null && message.hasOwnProperty("rawData"))
                    object.rawData = options.bytes === String ? $util.base64.encode(message.rawData, 0, message.rawData.length) : options.bytes === Array ? Array.prototype.slice.call(message.rawData) : message.rawData;
                if (message.doubleData && message.doubleData.length) {
                    object.doubleData = [];
                    for (var j = 0; j < message.doubleData.length; ++j)
                        object.doubleData[j] = options.json && !isFinite(message.doubleData[j]) ? String(message.doubleData[j]) : message.doubleData[j];
                }
                if (message.uint64Data && message.uint64Data.length) {
                    object.uint64Data = [];
                    for (var j = 0; j < message.uint64Data.length; ++j)
                        if (typeof message.uint64Data[j] === "number")
                            object.uint64Data[j] = options.longs === String ? String(message.uint64Data[j]) : message.uint64Data[j];
                        else
                            object.uint64Data[j] = options.longs === String ? $util.Long.prototype.toString.call(message.uint64Data[j]) : options.longs === Number ? new $util.LongBits(message.uint64Data[j].low >>> 0, message.uint64Data[j].high >>> 0).toNumber(true) : message.uint64Data[j];
                }
                if (message.docString != null && message.hasOwnProperty("docString"))
                    object.docString = message.docString;
                return object;
            };
    
            TensorProto.prototype.toJSON = function toJSON() {
                return this.constructor.toObject(this, $protobuf.util.toJSONOptions);
            };
    
            TensorProto.DataType = (function() {
                var valuesById = {}, values = Object.create(valuesById);
                values[valuesById[0] = "UNDEFINED"] = 0;
                values[valuesById[1] = "FLOAT"] = 1;
                values[valuesById[2] = "UINT8"] = 2;
                values[valuesById[3] = "INT8"] = 3;
                values[valuesById[4] = "UINT16"] = 4;
                values[valuesById[5] = "INT16"] = 5;
                values[valuesById[6] = "INT32"] = 6;
                values[valuesById[7] = "INT64"] = 7;
                values[valuesById[8] = "STRING"] = 8;
                values[valuesById[9] = "BOOL"] = 9;
                values[valuesById[10] = "FLOAT16"] = 10;
                values[valuesById[11] = "DOUBLE"] = 11;
                values[valuesById[12] = "UINT32"] = 12;
                values[valuesById[13] = "UINT64"] = 13;
                values[valuesById[14] = "COMPLEX64"] = 14;
                values[valuesById[15] = "COMPLEX128"] = 15;
                return values;
            })();
    
            TensorProto.Segment = (function() {
    
                function Segment(properties) {
                    if (properties)
                        for (var keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                            if (properties[keys[i]] != null)
                                this[keys[i]] = properties[keys[i]];
                }
    
                Segment.prototype.begin = $util.Long ? $util.Long.fromBits(0,0,false) : 0;
                Segment.prototype.end = $util.Long ? $util.Long.fromBits(0,0,false) : 0;
    
                Segment.create = function create(properties) {
                    return new Segment(properties);
                };
    
                Segment.decode = function decode(reader, length) {
                    if (!(reader instanceof $Reader))
                        reader = $Reader.create(reader);
                    var end = length === undefined ? reader.len : reader.pos + length, message = new $root.onnx.TensorProto.Segment();
                    while (reader.pos < end) {
                        var tag = reader.uint32();
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
                };
    
                Segment.verify = function verify(message) {
                    if (typeof message !== "object" || message === null)
                        return "object expected";
                    if (message.begin != null && message.hasOwnProperty("begin"))
                        if (!$util.isInteger(message.begin) && !(message.begin && $util.isInteger(message.begin.low) && $util.isInteger(message.begin.high)))
                            return "begin: integer|Long expected";
                    if (message.end != null && message.hasOwnProperty("end"))
                        if (!$util.isInteger(message.end) && !(message.end && $util.isInteger(message.end.low) && $util.isInteger(message.end.high)))
                            return "end: integer|Long expected";
                    return null;
                };
    
                Segment.fromObject = function fromObject(object) {
                    if (object instanceof $root.onnx.TensorProto.Segment)
                        return object;
                    var message = new $root.onnx.TensorProto.Segment();
                    if (object.begin != null)
                        if ($util.Long)
                            (message.begin = $util.Long.fromValue(object.begin)).unsigned = false;
                        else if (typeof object.begin === "string")
                            message.begin = parseInt(object.begin, 10);
                        else if (typeof object.begin === "number")
                            message.begin = object.begin;
                        else if (typeof object.begin === "object")
                            message.begin = new $util.LongBits(object.begin.low >>> 0, object.begin.high >>> 0).toNumber();
                    if (object.end != null)
                        if ($util.Long)
                            (message.end = $util.Long.fromValue(object.end)).unsigned = false;
                        else if (typeof object.end === "string")
                            message.end = parseInt(object.end, 10);
                        else if (typeof object.end === "number")
                            message.end = object.end;
                        else if (typeof object.end === "object")
                            message.end = new $util.LongBits(object.end.low >>> 0, object.end.high >>> 0).toNumber();
                    return message;
                };
    
                Segment.toObject = function toObject(message, options) {
                    if (!options)
                        options = {};
                    var object = {};
                    if (options.defaults) {
                        if ($util.Long) {
                            var long = new $util.Long(0, 0, false);
                            object.begin = options.longs === String ? long.toString() : options.longs === Number ? long.toNumber() : long;
                        } else
                            object.begin = options.longs === String ? "0" : 0;
                        if ($util.Long) {
                            var long = new $util.Long(0, 0, false);
                            object.end = options.longs === String ? long.toString() : options.longs === Number ? long.toNumber() : long;
                        } else
                            object.end = options.longs === String ? "0" : 0;
                    }
                    if (message.begin != null && message.hasOwnProperty("begin"))
                        if (typeof message.begin === "number")
                            object.begin = options.longs === String ? String(message.begin) : message.begin;
                        else
                            object.begin = options.longs === String ? $util.Long.prototype.toString.call(message.begin) : options.longs === Number ? new $util.LongBits(message.begin.low >>> 0, message.begin.high >>> 0).toNumber() : message.begin;
                    if (message.end != null && message.hasOwnProperty("end"))
                        if (typeof message.end === "number")
                            object.end = options.longs === String ? String(message.end) : message.end;
                        else
                            object.end = options.longs === String ? $util.Long.prototype.toString.call(message.end) : options.longs === Number ? new $util.LongBits(message.end.low >>> 0, message.end.high >>> 0).toNumber() : message.end;
                    return object;
                };
    
                Segment.prototype.toJSON = function toJSON() {
                    return this.constructor.toObject(this, $protobuf.util.toJSONOptions);
                };
    
                return Segment;
            })();
    
            return TensorProto;
        })();
    
        onnx.TensorShapeProto = (function() {
    
            function TensorShapeProto(properties) {
                this.dim = [];
                if (properties)
                    for (var keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                        if (properties[keys[i]] != null)
                            this[keys[i]] = properties[keys[i]];
            }
    
            TensorShapeProto.prototype.dim = $util.emptyArray;
    
            TensorShapeProto.create = function create(properties) {
                return new TensorShapeProto(properties);
            };
    
            TensorShapeProto.decode = function decode(reader, length) {
                if (!(reader instanceof $Reader))
                    reader = $Reader.create(reader);
                var end = length === undefined ? reader.len : reader.pos + length, message = new $root.onnx.TensorShapeProto();
                while (reader.pos < end) {
                    var tag = reader.uint32();
                    switch (tag >>> 3) {
                    case 1:
                        if (!(message.dim && message.dim.length))
                            message.dim = [];
                        message.dim.push($root.onnx.TensorShapeProto.Dimension.decode(reader, reader.uint32()));
                        break;
                    default:
                        reader.skipType(tag & 7);
                        break;
                    }
                }
                return message;
            };
    
            TensorShapeProto.verify = function verify(message) {
                if (typeof message !== "object" || message === null)
                    return "object expected";
                if (message.dim != null && message.hasOwnProperty("dim")) {
                    if (!Array.isArray(message.dim))
                        return "dim: array expected";
                    for (var i = 0; i < message.dim.length; ++i) {
                        var error = $root.onnx.TensorShapeProto.Dimension.verify(message.dim[i]);
                        if (error)
                            return "dim." + error;
                    }
                }
                return null;
            };
    
            TensorShapeProto.fromObject = function fromObject(object) {
                if (object instanceof $root.onnx.TensorShapeProto)
                    return object;
                var message = new $root.onnx.TensorShapeProto();
                if (object.dim) {
                    if (!Array.isArray(object.dim))
                        throw TypeError(".onnx.TensorShapeProto.dim: array expected");
                    message.dim = [];
                    for (var i = 0; i < object.dim.length; ++i) {
                        if (typeof object.dim[i] !== "object")
                            throw TypeError(".onnx.TensorShapeProto.dim: object expected");
                        message.dim[i] = $root.onnx.TensorShapeProto.Dimension.fromObject(object.dim[i]);
                    }
                }
                return message;
            };
    
            TensorShapeProto.toObject = function toObject(message, options) {
                if (!options)
                    options = {};
                var object = {};
                if (options.arrays || options.defaults)
                    object.dim = [];
                if (message.dim && message.dim.length) {
                    object.dim = [];
                    for (var j = 0; j < message.dim.length; ++j)
                        object.dim[j] = $root.onnx.TensorShapeProto.Dimension.toObject(message.dim[j], options);
                }
                return object;
            };
    
            TensorShapeProto.prototype.toJSON = function toJSON() {
                return this.constructor.toObject(this, $protobuf.util.toJSONOptions);
            };
    
            TensorShapeProto.Dimension = (function() {
    
                function Dimension(properties) {
                    if (properties)
                        for (var keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                            if (properties[keys[i]] != null)
                                this[keys[i]] = properties[keys[i]];
                }
    
                Dimension.prototype.dimValue = $util.Long ? $util.Long.fromBits(0,0,false) : 0;
                Dimension.prototype.dimParam = "";
                Dimension.prototype.denotation = "";
    
                var $oneOfFields;
    
                Object.defineProperty(Dimension.prototype, "value", {
                    get: $util.oneOfGetter($oneOfFields = ["dimValue", "dimParam"]),
                    set: $util.oneOfSetter($oneOfFields)
                });
    
                Dimension.create = function create(properties) {
                    return new Dimension(properties);
                };
    
                Dimension.decode = function decode(reader, length) {
                    if (!(reader instanceof $Reader))
                        reader = $Reader.create(reader);
                    var end = length === undefined ? reader.len : reader.pos + length, message = new $root.onnx.TensorShapeProto.Dimension();
                    while (reader.pos < end) {
                        var tag = reader.uint32();
                        switch (tag >>> 3) {
                        case 1:
                            message.dimValue = reader.int64();
                            break;
                        case 2:
                            message.dimParam = reader.string();
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
                };
    
                Dimension.verify = function verify(message) {
                    if (typeof message !== "object" || message === null)
                        return "object expected";
                    var properties = {};
                    if (message.dimValue != null && message.hasOwnProperty("dimValue")) {
                        properties.value = 1;
                        if (!$util.isInteger(message.dimValue) && !(message.dimValue && $util.isInteger(message.dimValue.low) && $util.isInteger(message.dimValue.high)))
                            return "dimValue: integer|Long expected";
                    }
                    if (message.dimParam != null && message.hasOwnProperty("dimParam")) {
                        if (properties.value === 1)
                            return "value: multiple values";
                        properties.value = 1;
                        if (!$util.isString(message.dimParam))
                            return "dimParam: string expected";
                    }
                    if (message.denotation != null && message.hasOwnProperty("denotation"))
                        if (!$util.isString(message.denotation))
                            return "denotation: string expected";
                    return null;
                };
    
                Dimension.fromObject = function fromObject(object) {
                    if (object instanceof $root.onnx.TensorShapeProto.Dimension)
                        return object;
                    var message = new $root.onnx.TensorShapeProto.Dimension();
                    if (object.dimValue != null)
                        if ($util.Long)
                            (message.dimValue = $util.Long.fromValue(object.dimValue)).unsigned = false;
                        else if (typeof object.dimValue === "string")
                            message.dimValue = parseInt(object.dimValue, 10);
                        else if (typeof object.dimValue === "number")
                            message.dimValue = object.dimValue;
                        else if (typeof object.dimValue === "object")
                            message.dimValue = new $util.LongBits(object.dimValue.low >>> 0, object.dimValue.high >>> 0).toNumber();
                    if (object.dimParam != null)
                        message.dimParam = String(object.dimParam);
                    if (object.denotation != null)
                        message.denotation = String(object.denotation);
                    return message;
                };
    
                Dimension.toObject = function toObject(message, options) {
                    if (!options)
                        options = {};
                    var object = {};
                    if (options.defaults)
                        object.denotation = "";
                    if (message.dimValue != null && message.hasOwnProperty("dimValue")) {
                        if (typeof message.dimValue === "number")
                            object.dimValue = options.longs === String ? String(message.dimValue) : message.dimValue;
                        else
                            object.dimValue = options.longs === String ? $util.Long.prototype.toString.call(message.dimValue) : options.longs === Number ? new $util.LongBits(message.dimValue.low >>> 0, message.dimValue.high >>> 0).toNumber() : message.dimValue;
                        if (options.oneofs)
                            object.value = "dimValue";
                    }
                    if (message.dimParam != null && message.hasOwnProperty("dimParam")) {
                        object.dimParam = message.dimParam;
                        if (options.oneofs)
                            object.value = "dimParam";
                    }
                    if (message.denotation != null && message.hasOwnProperty("denotation"))
                        object.denotation = message.denotation;
                    return object;
                };
    
                Dimension.prototype.toJSON = function toJSON() {
                    return this.constructor.toObject(this, $protobuf.util.toJSONOptions);
                };
    
                return Dimension;
            })();
    
            return TensorShapeProto;
        })();
    
        onnx.TypeProto = (function() {
    
            function TypeProto(properties) {
                if (properties)
                    for (var keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                        if (properties[keys[i]] != null)
                            this[keys[i]] = properties[keys[i]];
            }
    
            TypeProto.prototype.tensorType = null;
            TypeProto.prototype.sequenceType = null;
            TypeProto.prototype.mapType = null;
            TypeProto.prototype.denotation = "";
    
            var $oneOfFields;
    
            Object.defineProperty(TypeProto.prototype, "value", {
                get: $util.oneOfGetter($oneOfFields = ["tensorType", "sequenceType", "mapType"]),
                set: $util.oneOfSetter($oneOfFields)
            });
    
            TypeProto.create = function create(properties) {
                return new TypeProto(properties);
            };
    
            TypeProto.decode = function decode(reader, length) {
                if (!(reader instanceof $Reader))
                    reader = $Reader.create(reader);
                var end = length === undefined ? reader.len : reader.pos + length, message = new $root.onnx.TypeProto();
                while (reader.pos < end) {
                    var tag = reader.uint32();
                    switch (tag >>> 3) {
                    case 1:
                        message.tensorType = $root.onnx.TypeProto.Tensor.decode(reader, reader.uint32());
                        break;
                    case 4:
                        message.sequenceType = $root.onnx.TypeProto.Sequence.decode(reader, reader.uint32());
                        break;
                    case 5:
                        message.mapType = $root.onnx.TypeProto.Map.decode(reader, reader.uint32());
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
            };
    
            TypeProto.verify = function verify(message) {
                if (typeof message !== "object" || message === null)
                    return "object expected";
                var properties = {};
                if (message.tensorType != null && message.hasOwnProperty("tensorType")) {
                    properties.value = 1;
                    {
                        var error = $root.onnx.TypeProto.Tensor.verify(message.tensorType);
                        if (error)
                            return "tensorType." + error;
                    }
                }
                if (message.sequenceType != null && message.hasOwnProperty("sequenceType")) {
                    if (properties.value === 1)
                        return "value: multiple values";
                    properties.value = 1;
                    {
                        var error = $root.onnx.TypeProto.Sequence.verify(message.sequenceType);
                        if (error)
                            return "sequenceType." + error;
                    }
                }
                if (message.mapType != null && message.hasOwnProperty("mapType")) {
                    if (properties.value === 1)
                        return "value: multiple values";
                    properties.value = 1;
                    {
                        var error = $root.onnx.TypeProto.Map.verify(message.mapType);
                        if (error)
                            return "mapType." + error;
                    }
                }
                if (message.denotation != null && message.hasOwnProperty("denotation"))
                    if (!$util.isString(message.denotation))
                        return "denotation: string expected";
                return null;
            };
    
            TypeProto.fromObject = function fromObject(object) {
                if (object instanceof $root.onnx.TypeProto)
                    return object;
                var message = new $root.onnx.TypeProto();
                if (object.tensorType != null) {
                    if (typeof object.tensorType !== "object")
                        throw TypeError(".onnx.TypeProto.tensorType: object expected");
                    message.tensorType = $root.onnx.TypeProto.Tensor.fromObject(object.tensorType);
                }
                if (object.sequenceType != null) {
                    if (typeof object.sequenceType !== "object")
                        throw TypeError(".onnx.TypeProto.sequenceType: object expected");
                    message.sequenceType = $root.onnx.TypeProto.Sequence.fromObject(object.sequenceType);
                }
                if (object.mapType != null) {
                    if (typeof object.mapType !== "object")
                        throw TypeError(".onnx.TypeProto.mapType: object expected");
                    message.mapType = $root.onnx.TypeProto.Map.fromObject(object.mapType);
                }
                if (object.denotation != null)
                    message.denotation = String(object.denotation);
                return message;
            };
    
            TypeProto.toObject = function toObject(message, options) {
                if (!options)
                    options = {};
                var object = {};
                if (options.defaults)
                    object.denotation = "";
                if (message.tensorType != null && message.hasOwnProperty("tensorType")) {
                    object.tensorType = $root.onnx.TypeProto.Tensor.toObject(message.tensorType, options);
                    if (options.oneofs)
                        object.value = "tensorType";
                }
                if (message.sequenceType != null && message.hasOwnProperty("sequenceType")) {
                    object.sequenceType = $root.onnx.TypeProto.Sequence.toObject(message.sequenceType, options);
                    if (options.oneofs)
                        object.value = "sequenceType";
                }
                if (message.mapType != null && message.hasOwnProperty("mapType")) {
                    object.mapType = $root.onnx.TypeProto.Map.toObject(message.mapType, options);
                    if (options.oneofs)
                        object.value = "mapType";
                }
                if (message.denotation != null && message.hasOwnProperty("denotation"))
                    object.denotation = message.denotation;
                return object;
            };
    
            TypeProto.prototype.toJSON = function toJSON() {
                return this.constructor.toObject(this, $protobuf.util.toJSONOptions);
            };
    
            TypeProto.Tensor = (function() {
    
                function Tensor(properties) {
                    if (properties)
                        for (var keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                            if (properties[keys[i]] != null)
                                this[keys[i]] = properties[keys[i]];
                }
    
                Tensor.prototype.elemType = 0;
                Tensor.prototype.shape = null;
    
                Tensor.create = function create(properties) {
                    return new Tensor(properties);
                };
    
                Tensor.decode = function decode(reader, length) {
                    if (!(reader instanceof $Reader))
                        reader = $Reader.create(reader);
                    var end = length === undefined ? reader.len : reader.pos + length, message = new $root.onnx.TypeProto.Tensor();
                    while (reader.pos < end) {
                        var tag = reader.uint32();
                        switch (tag >>> 3) {
                        case 1:
                            message.elemType = reader.int32();
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
                };
    
                Tensor.verify = function verify(message) {
                    if (typeof message !== "object" || message === null)
                        return "object expected";
                    if (message.elemType != null && message.hasOwnProperty("elemType"))
                        switch (message.elemType) {
                        default:
                            return "elemType: enum value expected";
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
                        case 12:
                        case 13:
                        case 14:
                        case 15:
                            break;
                        }
                    if (message.shape != null && message.hasOwnProperty("shape")) {
                        var error = $root.onnx.TensorShapeProto.verify(message.shape);
                        if (error)
                            return "shape." + error;
                    }
                    return null;
                };
    
                Tensor.fromObject = function fromObject(object) {
                    if (object instanceof $root.onnx.TypeProto.Tensor)
                        return object;
                    var message = new $root.onnx.TypeProto.Tensor();
                    switch (object.elemType) {
                    case "UNDEFINED":
                    case 0:
                        message.elemType = 0;
                        break;
                    case "FLOAT":
                    case 1:
                        message.elemType = 1;
                        break;
                    case "UINT8":
                    case 2:
                        message.elemType = 2;
                        break;
                    case "INT8":
                    case 3:
                        message.elemType = 3;
                        break;
                    case "UINT16":
                    case 4:
                        message.elemType = 4;
                        break;
                    case "INT16":
                    case 5:
                        message.elemType = 5;
                        break;
                    case "INT32":
                    case 6:
                        message.elemType = 6;
                        break;
                    case "INT64":
                    case 7:
                        message.elemType = 7;
                        break;
                    case "STRING":
                    case 8:
                        message.elemType = 8;
                        break;
                    case "BOOL":
                    case 9:
                        message.elemType = 9;
                        break;
                    case "FLOAT16":
                    case 10:
                        message.elemType = 10;
                        break;
                    case "DOUBLE":
                    case 11:
                        message.elemType = 11;
                        break;
                    case "UINT32":
                    case 12:
                        message.elemType = 12;
                        break;
                    case "UINT64":
                    case 13:
                        message.elemType = 13;
                        break;
                    case "COMPLEX64":
                    case 14:
                        message.elemType = 14;
                        break;
                    case "COMPLEX128":
                    case 15:
                        message.elemType = 15;
                        break;
                    }
                    if (object.shape != null) {
                        if (typeof object.shape !== "object")
                            throw TypeError(".onnx.TypeProto.Tensor.shape: object expected");
                        message.shape = $root.onnx.TensorShapeProto.fromObject(object.shape);
                    }
                    return message;
                };
    
                Tensor.toObject = function toObject(message, options) {
                    if (!options)
                        options = {};
                    var object = {};
                    if (options.defaults) {
                        object.elemType = options.enums === String ? "UNDEFINED" : 0;
                        object.shape = null;
                    }
                    if (message.elemType != null && message.hasOwnProperty("elemType"))
                        object.elemType = options.enums === String ? $root.onnx.TensorProto.DataType[message.elemType] : message.elemType;
                    if (message.shape != null && message.hasOwnProperty("shape"))
                        object.shape = $root.onnx.TensorShapeProto.toObject(message.shape, options);
                    return object;
                };
    
                Tensor.prototype.toJSON = function toJSON() {
                    return this.constructor.toObject(this, $protobuf.util.toJSONOptions);
                };
    
                return Tensor;
            })();
    
            TypeProto.Sequence = (function() {
    
                function Sequence(properties) {
                    if (properties)
                        for (var keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                            if (properties[keys[i]] != null)
                                this[keys[i]] = properties[keys[i]];
                }
    
                Sequence.prototype.elemType = null;
    
                Sequence.create = function create(properties) {
                    return new Sequence(properties);
                };
    
                Sequence.decode = function decode(reader, length) {
                    if (!(reader instanceof $Reader))
                        reader = $Reader.create(reader);
                    var end = length === undefined ? reader.len : reader.pos + length, message = new $root.onnx.TypeProto.Sequence();
                    while (reader.pos < end) {
                        var tag = reader.uint32();
                        switch (tag >>> 3) {
                        case 1:
                            message.elemType = $root.onnx.TypeProto.decode(reader, reader.uint32());
                            break;
                        default:
                            reader.skipType(tag & 7);
                            break;
                        }
                    }
                    return message;
                };
    
                Sequence.verify = function verify(message) {
                    if (typeof message !== "object" || message === null)
                        return "object expected";
                    if (message.elemType != null && message.hasOwnProperty("elemType")) {
                        var error = $root.onnx.TypeProto.verify(message.elemType);
                        if (error)
                            return "elemType." + error;
                    }
                    return null;
                };
    
                Sequence.fromObject = function fromObject(object) {
                    if (object instanceof $root.onnx.TypeProto.Sequence)
                        return object;
                    var message = new $root.onnx.TypeProto.Sequence();
                    if (object.elemType != null) {
                        if (typeof object.elemType !== "object")
                            throw TypeError(".onnx.TypeProto.Sequence.elemType: object expected");
                        message.elemType = $root.onnx.TypeProto.fromObject(object.elemType);
                    }
                    return message;
                };
    
                Sequence.toObject = function toObject(message, options) {
                    if (!options)
                        options = {};
                    var object = {};
                    if (options.defaults)
                        object.elemType = null;
                    if (message.elemType != null && message.hasOwnProperty("elemType"))
                        object.elemType = $root.onnx.TypeProto.toObject(message.elemType, options);
                    return object;
                };
    
                Sequence.prototype.toJSON = function toJSON() {
                    return this.constructor.toObject(this, $protobuf.util.toJSONOptions);
                };
    
                return Sequence;
            })();
    
            TypeProto.Map = (function() {
    
                function Map(properties) {
                    if (properties)
                        for (var keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                            if (properties[keys[i]] != null)
                                this[keys[i]] = properties[keys[i]];
                }
    
                Map.prototype.keyType = 0;
                Map.prototype.valueType = null;
    
                Map.create = function create(properties) {
                    return new Map(properties);
                };
    
                Map.decode = function decode(reader, length) {
                    if (!(reader instanceof $Reader))
                        reader = $Reader.create(reader);
                    var end = length === undefined ? reader.len : reader.pos + length, message = new $root.onnx.TypeProto.Map();
                    while (reader.pos < end) {
                        var tag = reader.uint32();
                        switch (tag >>> 3) {
                        case 1:
                            message.keyType = reader.int32();
                            break;
                        case 2:
                            message.valueType = $root.onnx.TypeProto.decode(reader, reader.uint32());
                            break;
                        default:
                            reader.skipType(tag & 7);
                            break;
                        }
                    }
                    return message;
                };
    
                Map.verify = function verify(message) {
                    if (typeof message !== "object" || message === null)
                        return "object expected";
                    if (message.keyType != null && message.hasOwnProperty("keyType"))
                        switch (message.keyType) {
                        default:
                            return "keyType: enum value expected";
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
                        case 12:
                        case 13:
                        case 14:
                        case 15:
                            break;
                        }
                    if (message.valueType != null && message.hasOwnProperty("valueType")) {
                        var error = $root.onnx.TypeProto.verify(message.valueType);
                        if (error)
                            return "valueType." + error;
                    }
                    return null;
                };
    
                Map.fromObject = function fromObject(object) {
                    if (object instanceof $root.onnx.TypeProto.Map)
                        return object;
                    var message = new $root.onnx.TypeProto.Map();
                    switch (object.keyType) {
                    case "UNDEFINED":
                    case 0:
                        message.keyType = 0;
                        break;
                    case "FLOAT":
                    case 1:
                        message.keyType = 1;
                        break;
                    case "UINT8":
                    case 2:
                        message.keyType = 2;
                        break;
                    case "INT8":
                    case 3:
                        message.keyType = 3;
                        break;
                    case "UINT16":
                    case 4:
                        message.keyType = 4;
                        break;
                    case "INT16":
                    case 5:
                        message.keyType = 5;
                        break;
                    case "INT32":
                    case 6:
                        message.keyType = 6;
                        break;
                    case "INT64":
                    case 7:
                        message.keyType = 7;
                        break;
                    case "STRING":
                    case 8:
                        message.keyType = 8;
                        break;
                    case "BOOL":
                    case 9:
                        message.keyType = 9;
                        break;
                    case "FLOAT16":
                    case 10:
                        message.keyType = 10;
                        break;
                    case "DOUBLE":
                    case 11:
                        message.keyType = 11;
                        break;
                    case "UINT32":
                    case 12:
                        message.keyType = 12;
                        break;
                    case "UINT64":
                    case 13:
                        message.keyType = 13;
                        break;
                    case "COMPLEX64":
                    case 14:
                        message.keyType = 14;
                        break;
                    case "COMPLEX128":
                    case 15:
                        message.keyType = 15;
                        break;
                    }
                    if (object.valueType != null) {
                        if (typeof object.valueType !== "object")
                            throw TypeError(".onnx.TypeProto.Map.valueType: object expected");
                        message.valueType = $root.onnx.TypeProto.fromObject(object.valueType);
                    }
                    return message;
                };
    
                Map.toObject = function toObject(message, options) {
                    if (!options)
                        options = {};
                    var object = {};
                    if (options.defaults) {
                        object.keyType = options.enums === String ? "UNDEFINED" : 0;
                        object.valueType = null;
                    }
                    if (message.keyType != null && message.hasOwnProperty("keyType"))
                        object.keyType = options.enums === String ? $root.onnx.TensorProto.DataType[message.keyType] : message.keyType;
                    if (message.valueType != null && message.hasOwnProperty("valueType"))
                        object.valueType = $root.onnx.TypeProto.toObject(message.valueType, options);
                    return object;
                };
    
                Map.prototype.toJSON = function toJSON() {
                    return this.constructor.toObject(this, $protobuf.util.toJSONOptions);
                };
    
                return Map;
            })();
    
            return TypeProto;
        })();
    
        onnx.OperatorSetIdProto = (function() {
    
            function OperatorSetIdProto(properties) {
                if (properties)
                    for (var keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                        if (properties[keys[i]] != null)
                            this[keys[i]] = properties[keys[i]];
            }
    
            OperatorSetIdProto.prototype.domain = "";
            OperatorSetIdProto.prototype.version = $util.Long ? $util.Long.fromBits(0,0,false) : 0;
    
            OperatorSetIdProto.create = function create(properties) {
                return new OperatorSetIdProto(properties);
            };
    
            OperatorSetIdProto.decode = function decode(reader, length) {
                if (!(reader instanceof $Reader))
                    reader = $Reader.create(reader);
                var end = length === undefined ? reader.len : reader.pos + length, message = new $root.onnx.OperatorSetIdProto();
                while (reader.pos < end) {
                    var tag = reader.uint32();
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
            };
    
            OperatorSetIdProto.verify = function verify(message) {
                if (typeof message !== "object" || message === null)
                    return "object expected";
                if (message.domain != null && message.hasOwnProperty("domain"))
                    if (!$util.isString(message.domain))
                        return "domain: string expected";
                if (message.version != null && message.hasOwnProperty("version"))
                    if (!$util.isInteger(message.version) && !(message.version && $util.isInteger(message.version.low) && $util.isInteger(message.version.high)))
                        return "version: integer|Long expected";
                return null;
            };
    
            OperatorSetIdProto.fromObject = function fromObject(object) {
                if (object instanceof $root.onnx.OperatorSetIdProto)
                    return object;
                var message = new $root.onnx.OperatorSetIdProto();
                if (object.domain != null)
                    message.domain = String(object.domain);
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
    
            OperatorSetIdProto.toObject = function toObject(message, options) {
                if (!options)
                    options = {};
                var object = {};
                if (options.defaults) {
                    object.domain = "";
                    if ($util.Long) {
                        var long = new $util.Long(0, 0, false);
                        object.version = options.longs === String ? long.toString() : options.longs === Number ? long.toNumber() : long;
                    } else
                        object.version = options.longs === String ? "0" : 0;
                }
                if (message.domain != null && message.hasOwnProperty("domain"))
                    object.domain = message.domain;
                if (message.version != null && message.hasOwnProperty("version"))
                    if (typeof message.version === "number")
                        object.version = options.longs === String ? String(message.version) : message.version;
                    else
                        object.version = options.longs === String ? $util.Long.prototype.toString.call(message.version) : options.longs === Number ? new $util.LongBits(message.version.low >>> 0, message.version.high >>> 0).toNumber() : message.version;
                return object;
            };
    
            OperatorSetIdProto.prototype.toJSON = function toJSON() {
                return this.constructor.toObject(this, $protobuf.util.toJSONOptions);
            };
    
            return OperatorSetIdProto;
        })();
    
        onnx.OperatorStatus = (function() {
            var valuesById = {}, values = Object.create(valuesById);
            values[valuesById[0] = "EXPERIMENTAL"] = 0;
            values[valuesById[1] = "STABLE"] = 1;
            return values;
        })();
    
        onnx.FunctionProto = (function() {
    
            function FunctionProto(properties) {
                this.input = [];
                this.output = [];
                this.attribute = [];
                this.node = [];
                if (properties)
                    for (var keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                        if (properties[keys[i]] != null)
                            this[keys[i]] = properties[keys[i]];
            }
    
            FunctionProto.prototype.name = "";
            FunctionProto.prototype.sinceVersion = $util.Long ? $util.Long.fromBits(0,0,false) : 0;
            FunctionProto.prototype.status = 0;
            FunctionProto.prototype.input = $util.emptyArray;
            FunctionProto.prototype.output = $util.emptyArray;
            FunctionProto.prototype.attribute = $util.emptyArray;
            FunctionProto.prototype.node = $util.emptyArray;
            FunctionProto.prototype.docString = "";
    
            FunctionProto.create = function create(properties) {
                return new FunctionProto(properties);
            };
    
            FunctionProto.decode = function decode(reader, length) {
                if (!(reader instanceof $Reader))
                    reader = $Reader.create(reader);
                var end = length === undefined ? reader.len : reader.pos + length, message = new $root.onnx.FunctionProto();
                while (reader.pos < end) {
                    var tag = reader.uint32();
                    switch (tag >>> 3) {
                    case 1:
                        message.name = reader.string();
                        break;
                    case 2:
                        message.sinceVersion = reader.int64();
                        break;
                    case 3:
                        message.status = reader.int32();
                        break;
                    case 4:
                        if (!(message.input && message.input.length))
                            message.input = [];
                        message.input.push(reader.string());
                        break;
                    case 5:
                        if (!(message.output && message.output.length))
                            message.output = [];
                        message.output.push(reader.string());
                        break;
                    case 6:
                        if (!(message.attribute && message.attribute.length))
                            message.attribute = [];
                        message.attribute.push(reader.string());
                        break;
                    case 7:
                        if (!(message.node && message.node.length))
                            message.node = [];
                        message.node.push($root.onnx.NodeProto.decode(reader, reader.uint32()));
                        break;
                    case 8:
                        message.docString = reader.string();
                        break;
                    default:
                        reader.skipType(tag & 7);
                        break;
                    }
                }
                return message;
            };
    
            FunctionProto.verify = function verify(message) {
                if (typeof message !== "object" || message === null)
                    return "object expected";
                if (message.name != null && message.hasOwnProperty("name"))
                    if (!$util.isString(message.name))
                        return "name: string expected";
                if (message.sinceVersion != null && message.hasOwnProperty("sinceVersion"))
                    if (!$util.isInteger(message.sinceVersion) && !(message.sinceVersion && $util.isInteger(message.sinceVersion.low) && $util.isInteger(message.sinceVersion.high)))
                        return "sinceVersion: integer|Long expected";
                if (message.status != null && message.hasOwnProperty("status"))
                    switch (message.status) {
                    default:
                        return "status: enum value expected";
                    case 0:
                    case 1:
                        break;
                    }
                if (message.input != null && message.hasOwnProperty("input")) {
                    if (!Array.isArray(message.input))
                        return "input: array expected";
                    for (var i = 0; i < message.input.length; ++i)
                        if (!$util.isString(message.input[i]))
                            return "input: string[] expected";
                }
                if (message.output != null && message.hasOwnProperty("output")) {
                    if (!Array.isArray(message.output))
                        return "output: array expected";
                    for (var i = 0; i < message.output.length; ++i)
                        if (!$util.isString(message.output[i]))
                            return "output: string[] expected";
                }
                if (message.attribute != null && message.hasOwnProperty("attribute")) {
                    if (!Array.isArray(message.attribute))
                        return "attribute: array expected";
                    for (var i = 0; i < message.attribute.length; ++i)
                        if (!$util.isString(message.attribute[i]))
                            return "attribute: string[] expected";
                }
                if (message.node != null && message.hasOwnProperty("node")) {
                    if (!Array.isArray(message.node))
                        return "node: array expected";
                    for (var i = 0; i < message.node.length; ++i) {
                        var error = $root.onnx.NodeProto.verify(message.node[i]);
                        if (error)
                            return "node." + error;
                    }
                }
                if (message.docString != null && message.hasOwnProperty("docString"))
                    if (!$util.isString(message.docString))
                        return "docString: string expected";
                return null;
            };
    
            FunctionProto.fromObject = function fromObject(object) {
                if (object instanceof $root.onnx.FunctionProto)
                    return object;
                var message = new $root.onnx.FunctionProto();
                if (object.name != null)
                    message.name = String(object.name);
                if (object.sinceVersion != null)
                    if ($util.Long)
                        (message.sinceVersion = $util.Long.fromValue(object.sinceVersion)).unsigned = false;
                    else if (typeof object.sinceVersion === "string")
                        message.sinceVersion = parseInt(object.sinceVersion, 10);
                    else if (typeof object.sinceVersion === "number")
                        message.sinceVersion = object.sinceVersion;
                    else if (typeof object.sinceVersion === "object")
                        message.sinceVersion = new $util.LongBits(object.sinceVersion.low >>> 0, object.sinceVersion.high >>> 0).toNumber();
                switch (object.status) {
                case "EXPERIMENTAL":
                case 0:
                    message.status = 0;
                    break;
                case "STABLE":
                case 1:
                    message.status = 1;
                    break;
                }
                if (object.input) {
                    if (!Array.isArray(object.input))
                        throw TypeError(".onnx.FunctionProto.input: array expected");
                    message.input = [];
                    for (var i = 0; i < object.input.length; ++i)
                        message.input[i] = String(object.input[i]);
                }
                if (object.output) {
                    if (!Array.isArray(object.output))
                        throw TypeError(".onnx.FunctionProto.output: array expected");
                    message.output = [];
                    for (var i = 0; i < object.output.length; ++i)
                        message.output[i] = String(object.output[i]);
                }
                if (object.attribute) {
                    if (!Array.isArray(object.attribute))
                        throw TypeError(".onnx.FunctionProto.attribute: array expected");
                    message.attribute = [];
                    for (var i = 0; i < object.attribute.length; ++i)
                        message.attribute[i] = String(object.attribute[i]);
                }
                if (object.node) {
                    if (!Array.isArray(object.node))
                        throw TypeError(".onnx.FunctionProto.node: array expected");
                    message.node = [];
                    for (var i = 0; i < object.node.length; ++i) {
                        if (typeof object.node[i] !== "object")
                            throw TypeError(".onnx.FunctionProto.node: object expected");
                        message.node[i] = $root.onnx.NodeProto.fromObject(object.node[i]);
                    }
                }
                if (object.docString != null)
                    message.docString = String(object.docString);
                return message;
            };
    
            FunctionProto.toObject = function toObject(message, options) {
                if (!options)
                    options = {};
                var object = {};
                if (options.arrays || options.defaults) {
                    object.input = [];
                    object.output = [];
                    object.attribute = [];
                    object.node = [];
                }
                if (options.defaults) {
                    object.name = "";
                    if ($util.Long) {
                        var long = new $util.Long(0, 0, false);
                        object.sinceVersion = options.longs === String ? long.toString() : options.longs === Number ? long.toNumber() : long;
                    } else
                        object.sinceVersion = options.longs === String ? "0" : 0;
                    object.status = options.enums === String ? "EXPERIMENTAL" : 0;
                    object.docString = "";
                }
                if (message.name != null && message.hasOwnProperty("name"))
                    object.name = message.name;
                if (message.sinceVersion != null && message.hasOwnProperty("sinceVersion"))
                    if (typeof message.sinceVersion === "number")
                        object.sinceVersion = options.longs === String ? String(message.sinceVersion) : message.sinceVersion;
                    else
                        object.sinceVersion = options.longs === String ? $util.Long.prototype.toString.call(message.sinceVersion) : options.longs === Number ? new $util.LongBits(message.sinceVersion.low >>> 0, message.sinceVersion.high >>> 0).toNumber() : message.sinceVersion;
                if (message.status != null && message.hasOwnProperty("status"))
                    object.status = options.enums === String ? $root.onnx.OperatorStatus[message.status] : message.status;
                if (message.input && message.input.length) {
                    object.input = [];
                    for (var j = 0; j < message.input.length; ++j)
                        object.input[j] = message.input[j];
                }
                if (message.output && message.output.length) {
                    object.output = [];
                    for (var j = 0; j < message.output.length; ++j)
                        object.output[j] = message.output[j];
                }
                if (message.attribute && message.attribute.length) {
                    object.attribute = [];
                    for (var j = 0; j < message.attribute.length; ++j)
                        object.attribute[j] = message.attribute[j];
                }
                if (message.node && message.node.length) {
                    object.node = [];
                    for (var j = 0; j < message.node.length; ++j)
                        object.node[j] = $root.onnx.NodeProto.toObject(message.node[j], options);
                }
                if (message.docString != null && message.hasOwnProperty("docString"))
                    object.docString = message.docString;
                return object;
            };
    
            FunctionProto.prototype.toJSON = function toJSON() {
                return this.constructor.toObject(this, $protobuf.util.toJSONOptions);
            };
    
            return FunctionProto;
        })();
    
        onnx.OperatorProto = (function() {
    
            function OperatorProto(properties) {
                if (properties)
                    for (var keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                        if (properties[keys[i]] != null)
                            this[keys[i]] = properties[keys[i]];
            }
    
            OperatorProto.prototype.opType = "";
            OperatorProto.prototype.sinceVersion = $util.Long ? $util.Long.fromBits(0,0,false) : 0;
            OperatorProto.prototype.status = 0;
            OperatorProto.prototype.docString = "";
    
            OperatorProto.create = function create(properties) {
                return new OperatorProto(properties);
            };
    
            OperatorProto.decode = function decode(reader, length) {
                if (!(reader instanceof $Reader))
                    reader = $Reader.create(reader);
                var end = length === undefined ? reader.len : reader.pos + length, message = new $root.onnx.OperatorProto();
                while (reader.pos < end) {
                    var tag = reader.uint32();
                    switch (tag >>> 3) {
                    case 1:
                        message.opType = reader.string();
                        break;
                    case 2:
                        message.sinceVersion = reader.int64();
                        break;
                    case 3:
                        message.status = reader.int32();
                        break;
                    case 10:
                        message.docString = reader.string();
                        break;
                    default:
                        reader.skipType(tag & 7);
                        break;
                    }
                }
                return message;
            };
    
            OperatorProto.verify = function verify(message) {
                if (typeof message !== "object" || message === null)
                    return "object expected";
                if (message.opType != null && message.hasOwnProperty("opType"))
                    if (!$util.isString(message.opType))
                        return "opType: string expected";
                if (message.sinceVersion != null && message.hasOwnProperty("sinceVersion"))
                    if (!$util.isInteger(message.sinceVersion) && !(message.sinceVersion && $util.isInteger(message.sinceVersion.low) && $util.isInteger(message.sinceVersion.high)))
                        return "sinceVersion: integer|Long expected";
                if (message.status != null && message.hasOwnProperty("status"))
                    switch (message.status) {
                    default:
                        return "status: enum value expected";
                    case 0:
                    case 1:
                        break;
                    }
                if (message.docString != null && message.hasOwnProperty("docString"))
                    if (!$util.isString(message.docString))
                        return "docString: string expected";
                return null;
            };
    
            OperatorProto.fromObject = function fromObject(object) {
                if (object instanceof $root.onnx.OperatorProto)
                    return object;
                var message = new $root.onnx.OperatorProto();
                if (object.opType != null)
                    message.opType = String(object.opType);
                if (object.sinceVersion != null)
                    if ($util.Long)
                        (message.sinceVersion = $util.Long.fromValue(object.sinceVersion)).unsigned = false;
                    else if (typeof object.sinceVersion === "string")
                        message.sinceVersion = parseInt(object.sinceVersion, 10);
                    else if (typeof object.sinceVersion === "number")
                        message.sinceVersion = object.sinceVersion;
                    else if (typeof object.sinceVersion === "object")
                        message.sinceVersion = new $util.LongBits(object.sinceVersion.low >>> 0, object.sinceVersion.high >>> 0).toNumber();
                switch (object.status) {
                case "EXPERIMENTAL":
                case 0:
                    message.status = 0;
                    break;
                case "STABLE":
                case 1:
                    message.status = 1;
                    break;
                }
                if (object.docString != null)
                    message.docString = String(object.docString);
                return message;
            };
    
            OperatorProto.toObject = function toObject(message, options) {
                if (!options)
                    options = {};
                var object = {};
                if (options.defaults) {
                    object.opType = "";
                    if ($util.Long) {
                        var long = new $util.Long(0, 0, false);
                        object.sinceVersion = options.longs === String ? long.toString() : options.longs === Number ? long.toNumber() : long;
                    } else
                        object.sinceVersion = options.longs === String ? "0" : 0;
                    object.status = options.enums === String ? "EXPERIMENTAL" : 0;
                    object.docString = "";
                }
                if (message.opType != null && message.hasOwnProperty("opType"))
                    object.opType = message.opType;
                if (message.sinceVersion != null && message.hasOwnProperty("sinceVersion"))
                    if (typeof message.sinceVersion === "number")
                        object.sinceVersion = options.longs === String ? String(message.sinceVersion) : message.sinceVersion;
                    else
                        object.sinceVersion = options.longs === String ? $util.Long.prototype.toString.call(message.sinceVersion) : options.longs === Number ? new $util.LongBits(message.sinceVersion.low >>> 0, message.sinceVersion.high >>> 0).toNumber() : message.sinceVersion;
                if (message.status != null && message.hasOwnProperty("status"))
                    object.status = options.enums === String ? $root.onnx.OperatorStatus[message.status] : message.status;
                if (message.docString != null && message.hasOwnProperty("docString"))
                    object.docString = message.docString;
                return object;
            };
    
            OperatorProto.prototype.toJSON = function toJSON() {
                return this.constructor.toObject(this, $protobuf.util.toJSONOptions);
            };
    
            return OperatorProto;
        })();
    
        onnx.OperatorSetProto = (function() {
    
            function OperatorSetProto(properties) {
                this.operator = [];
                this.functions = [];
                if (properties)
                    for (var keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                        if (properties[keys[i]] != null)
                            this[keys[i]] = properties[keys[i]];
            }
    
            OperatorSetProto.prototype.magic = "";
            OperatorSetProto.prototype.irVersion = 0;
            OperatorSetProto.prototype.irVersionPrerelease = "";
            OperatorSetProto.prototype.irBuildMetadata = "";
            OperatorSetProto.prototype.domain = "";
            OperatorSetProto.prototype.opsetVersion = $util.Long ? $util.Long.fromBits(0,0,false) : 0;
            OperatorSetProto.prototype.docString = "";
            OperatorSetProto.prototype.operator = $util.emptyArray;
            OperatorSetProto.prototype.functions = $util.emptyArray;
    
            OperatorSetProto.create = function create(properties) {
                return new OperatorSetProto(properties);
            };
    
            OperatorSetProto.decode = function decode(reader, length) {
                if (!(reader instanceof $Reader))
                    reader = $Reader.create(reader);
                var end = length === undefined ? reader.len : reader.pos + length, message = new $root.onnx.OperatorSetProto();
                while (reader.pos < end) {
                    var tag = reader.uint32();
                    switch (tag >>> 3) {
                    case 1:
                        message.magic = reader.string();
                        break;
                    case 2:
                        message.irVersion = reader.int32();
                        break;
                    case 3:
                        message.irVersionPrerelease = reader.string();
                        break;
                    case 7:
                        message.irBuildMetadata = reader.string();
                        break;
                    case 4:
                        message.domain = reader.string();
                        break;
                    case 5:
                        message.opsetVersion = reader.int64();
                        break;
                    case 6:
                        message.docString = reader.string();
                        break;
                    case 8:
                        if (!(message.operator && message.operator.length))
                            message.operator = [];
                        message.operator.push($root.onnx.OperatorProto.decode(reader, reader.uint32()));
                        break;
                    case 9:
                        if (!(message.functions && message.functions.length))
                            message.functions = [];
                        message.functions.push($root.onnx.FunctionProto.decode(reader, reader.uint32()));
                        break;
                    default:
                        reader.skipType(tag & 7);
                        break;
                    }
                }
                return message;
            };
    
            OperatorSetProto.verify = function verify(message) {
                if (typeof message !== "object" || message === null)
                    return "object expected";
                if (message.magic != null && message.hasOwnProperty("magic"))
                    if (!$util.isString(message.magic))
                        return "magic: string expected";
                if (message.irVersion != null && message.hasOwnProperty("irVersion"))
                    if (!$util.isInteger(message.irVersion))
                        return "irVersion: integer expected";
                if (message.irVersionPrerelease != null && message.hasOwnProperty("irVersionPrerelease"))
                    if (!$util.isString(message.irVersionPrerelease))
                        return "irVersionPrerelease: string expected";
                if (message.irBuildMetadata != null && message.hasOwnProperty("irBuildMetadata"))
                    if (!$util.isString(message.irBuildMetadata))
                        return "irBuildMetadata: string expected";
                if (message.domain != null && message.hasOwnProperty("domain"))
                    if (!$util.isString(message.domain))
                        return "domain: string expected";
                if (message.opsetVersion != null && message.hasOwnProperty("opsetVersion"))
                    if (!$util.isInteger(message.opsetVersion) && !(message.opsetVersion && $util.isInteger(message.opsetVersion.low) && $util.isInteger(message.opsetVersion.high)))
                        return "opsetVersion: integer|Long expected";
                if (message.docString != null && message.hasOwnProperty("docString"))
                    if (!$util.isString(message.docString))
                        return "docString: string expected";
                if (message.operator != null && message.hasOwnProperty("operator")) {
                    if (!Array.isArray(message.operator))
                        return "operator: array expected";
                    for (var i = 0; i < message.operator.length; ++i) {
                        var error = $root.onnx.OperatorProto.verify(message.operator[i]);
                        if (error)
                            return "operator." + error;
                    }
                }
                if (message.functions != null && message.hasOwnProperty("functions")) {
                    if (!Array.isArray(message.functions))
                        return "functions: array expected";
                    for (var i = 0; i < message.functions.length; ++i) {
                        var error = $root.onnx.FunctionProto.verify(message.functions[i]);
                        if (error)
                            return "functions." + error;
                    }
                }
                return null;
            };
    
            OperatorSetProto.fromObject = function fromObject(object) {
                if (object instanceof $root.onnx.OperatorSetProto)
                    return object;
                var message = new $root.onnx.OperatorSetProto();
                if (object.magic != null)
                    message.magic = String(object.magic);
                if (object.irVersion != null)
                    message.irVersion = object.irVersion | 0;
                if (object.irVersionPrerelease != null)
                    message.irVersionPrerelease = String(object.irVersionPrerelease);
                if (object.irBuildMetadata != null)
                    message.irBuildMetadata = String(object.irBuildMetadata);
                if (object.domain != null)
                    message.domain = String(object.domain);
                if (object.opsetVersion != null)
                    if ($util.Long)
                        (message.opsetVersion = $util.Long.fromValue(object.opsetVersion)).unsigned = false;
                    else if (typeof object.opsetVersion === "string")
                        message.opsetVersion = parseInt(object.opsetVersion, 10);
                    else if (typeof object.opsetVersion === "number")
                        message.opsetVersion = object.opsetVersion;
                    else if (typeof object.opsetVersion === "object")
                        message.opsetVersion = new $util.LongBits(object.opsetVersion.low >>> 0, object.opsetVersion.high >>> 0).toNumber();
                if (object.docString != null)
                    message.docString = String(object.docString);
                if (object.operator) {
                    if (!Array.isArray(object.operator))
                        throw TypeError(".onnx.OperatorSetProto.operator: array expected");
                    message.operator = [];
                    for (var i = 0; i < object.operator.length; ++i) {
                        if (typeof object.operator[i] !== "object")
                            throw TypeError(".onnx.OperatorSetProto.operator: object expected");
                        message.operator[i] = $root.onnx.OperatorProto.fromObject(object.operator[i]);
                    }
                }
                if (object.functions) {
                    if (!Array.isArray(object.functions))
                        throw TypeError(".onnx.OperatorSetProto.functions: array expected");
                    message.functions = [];
                    for (var i = 0; i < object.functions.length; ++i) {
                        if (typeof object.functions[i] !== "object")
                            throw TypeError(".onnx.OperatorSetProto.functions: object expected");
                        message.functions[i] = $root.onnx.FunctionProto.fromObject(object.functions[i]);
                    }
                }
                return message;
            };
    
            OperatorSetProto.toObject = function toObject(message, options) {
                if (!options)
                    options = {};
                var object = {};
                if (options.arrays || options.defaults) {
                    object.operator = [];
                    object.functions = [];
                }
                if (options.defaults) {
                    object.magic = "";
                    object.irVersion = 0;
                    object.irVersionPrerelease = "";
                    object.domain = "";
                    if ($util.Long) {
                        var long = new $util.Long(0, 0, false);
                        object.opsetVersion = options.longs === String ? long.toString() : options.longs === Number ? long.toNumber() : long;
                    } else
                        object.opsetVersion = options.longs === String ? "0" : 0;
                    object.docString = "";
                    object.irBuildMetadata = "";
                }
                if (message.magic != null && message.hasOwnProperty("magic"))
                    object.magic = message.magic;
                if (message.irVersion != null && message.hasOwnProperty("irVersion"))
                    object.irVersion = message.irVersion;
                if (message.irVersionPrerelease != null && message.hasOwnProperty("irVersionPrerelease"))
                    object.irVersionPrerelease = message.irVersionPrerelease;
                if (message.domain != null && message.hasOwnProperty("domain"))
                    object.domain = message.domain;
                if (message.opsetVersion != null && message.hasOwnProperty("opsetVersion"))
                    if (typeof message.opsetVersion === "number")
                        object.opsetVersion = options.longs === String ? String(message.opsetVersion) : message.opsetVersion;
                    else
                        object.opsetVersion = options.longs === String ? $util.Long.prototype.toString.call(message.opsetVersion) : options.longs === Number ? new $util.LongBits(message.opsetVersion.low >>> 0, message.opsetVersion.high >>> 0).toNumber() : message.opsetVersion;
                if (message.docString != null && message.hasOwnProperty("docString"))
                    object.docString = message.docString;
                if (message.irBuildMetadata != null && message.hasOwnProperty("irBuildMetadata"))
                    object.irBuildMetadata = message.irBuildMetadata;
                if (message.operator && message.operator.length) {
                    object.operator = [];
                    for (var j = 0; j < message.operator.length; ++j)
                        object.operator[j] = $root.onnx.OperatorProto.toObject(message.operator[j], options);
                }
                if (message.functions && message.functions.length) {
                    object.functions = [];
                    for (var j = 0; j < message.functions.length; ++j)
                        object.functions[j] = $root.onnx.FunctionProto.toObject(message.functions[j], options);
                }
                return object;
            };
    
            OperatorSetProto.prototype.toJSON = function toJSON() {
                return this.constructor.toObject(this, $protobuf.util.toJSONOptions);
            };
    
            return OperatorSetProto;
        })();
    
        return onnx;
    })();

    return $root;
})(protobuf);
