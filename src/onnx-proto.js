/*eslint-disable block-scoped-var, id-length, no-control-regex, no-magic-numbers, no-prototype-builtins, no-redeclare, no-shadow, no-var, sort-vars*/
(function($protobuf) {
    "use strict";

    var $Reader = $protobuf.Reader, $TextReader = $protobuf.TextReader, $util = $protobuf.util;
    
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
            AttributeProto.prototype.ref_attr_name = "";
            AttributeProto.prototype.doc_string = "";
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
    
            AttributeProto.decodeText = function decodeText(reader, block) {
                if (!(reader instanceof $TextReader))
                    reader = $TextReader.create(reader);
                var message = new $root.onnx.AttributeProto();
                reader.start(block);
                while (!reader.end(block)) {
                    var tag = reader.tag();
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
                        message.t = $root.onnx.TensorProto.decodeText(reader, true);
                        break;
                    case "g":
                        message.g = $root.onnx.GraphProto.decodeText(reader, true);
                        break;
                    case "floats":
                        if (!(message.floats && message.floats.length))
                            message.floats = [];
                        message.floats.push(reader.float());
                        break;
                    case "ints":
                        if (!(message.ints && message.ints.length))
                            message.ints = [];
                        message.ints.push(reader.int64());
                        break;
                    case "strings":
                        if (!(message.strings && message.strings.length))
                            message.strings = [];
                        message.strings.push(reader.bytes());
                        break;
                    case "tensors":
                        if (!(message.tensors && message.tensors.length))
                            message.tensors = [];
                        message.tensors.push($root.onnx.TensorProto.decodeText(reader, true));
                        break;
                    case "graphs":
                        if (!(message.graphs && message.graphs.length))
                            message.graphs = [];
                        message.graphs.push($root.onnx.GraphProto.decodeText(reader, true));
                        break;
                    default:
                        reader.handle(tag, message);
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
                if (message.ref_attr_name != null && message.hasOwnProperty("ref_attr_name"))
                    if (!$util.isString(message.ref_attr_name))
                        return "ref_attr_name: string expected";
                if (message.doc_string != null && message.hasOwnProperty("doc_string"))
                    if (!$util.isString(message.doc_string))
                        return "doc_string: string expected";
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
                if (object.ref_attr_name != null)
                    message.ref_attr_name = String(object.ref_attr_name);
                if (object.doc_string != null)
                    message.doc_string = String(object.doc_string);
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
                    object.doc_string = "";
                    object.type = options.enums === String ? "UNDEFINED" : 0;
                    object.ref_attr_name = "";
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
                if (message.doc_string != null && message.hasOwnProperty("doc_string"))
                    object.doc_string = message.doc_string;
                if (message.type != null && message.hasOwnProperty("type"))
                    object.type = options.enums === String ? $root.onnx.AttributeProto.AttributeType[message.type] : message.type;
                if (message.ref_attr_name != null && message.hasOwnProperty("ref_attr_name"))
                    object.ref_attr_name = message.ref_attr_name;
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
            ValueInfoProto.prototype.doc_string = "";
    
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
                        message.doc_string = reader.string();
                        break;
                    default:
                        reader.skipType(tag & 7);
                        break;
                    }
                }
                return message;
            };
    
            ValueInfoProto.decodeText = function decodeText(reader, block) {
                if (!(reader instanceof $TextReader))
                    reader = $TextReader.create(reader);
                var message = new $root.onnx.ValueInfoProto();
                reader.start(block);
                while (!reader.end(block)) {
                    var tag = reader.tag();
                    switch (tag) {
                    case "name":
                        message.name = reader.string();
                        break;
                    case "type":
                        message.type = $root.onnx.TypeProto.decodeText(reader, true);
                        break;
                    case "doc_string":
                        message.doc_string = reader.string();
                        break;
                    default:
                        reader.handle(tag, message);
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
                if (message.doc_string != null && message.hasOwnProperty("doc_string"))
                    if (!$util.isString(message.doc_string))
                        return "doc_string: string expected";
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
                if (object.doc_string != null)
                    message.doc_string = String(object.doc_string);
                return message;
            };
    
            ValueInfoProto.toObject = function toObject(message, options) {
                if (!options)
                    options = {};
                var object = {};
                if (options.defaults) {
                    object.name = "";
                    object.type = null;
                    object.doc_string = "";
                }
                if (message.name != null && message.hasOwnProperty("name"))
                    object.name = message.name;
                if (message.type != null && message.hasOwnProperty("type"))
                    object.type = $root.onnx.TypeProto.toObject(message.type, options);
                if (message.doc_string != null && message.hasOwnProperty("doc_string"))
                    object.doc_string = message.doc_string;
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
            NodeProto.prototype.op_type = "";
            NodeProto.prototype.domain = "";
            NodeProto.prototype.attribute = $util.emptyArray;
            NodeProto.prototype.doc_string = "";
    
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
                        message.op_type = reader.string();
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
                        message.doc_string = reader.string();
                        break;
                    default:
                        reader.skipType(tag & 7);
                        break;
                    }
                }
                return message;
            };
    
            NodeProto.decodeText = function decodeText(reader, block) {
                if (!(reader instanceof $TextReader))
                    reader = $TextReader.create(reader);
                var message = new $root.onnx.NodeProto();
                reader.start(block);
                while (!reader.end(block)) {
                    var tag = reader.tag();
                    switch (tag) {
                    case "input":
                        if (!(message.input && message.input.length))
                            message.input = [];
                        message.input.push(reader.string());
                        break;
                    case "output":
                        if (!(message.output && message.output.length))
                            message.output = [];
                        message.output.push(reader.string());
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
                        if (!(message.attribute && message.attribute.length))
                            message.attribute = [];
                        message.attribute.push($root.onnx.AttributeProto.decodeText(reader, true));
                        break;
                    case "doc_string":
                        message.doc_string = reader.string();
                        break;
                    default:
                        reader.handle(tag, message);
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
                if (message.op_type != null && message.hasOwnProperty("op_type"))
                    if (!$util.isString(message.op_type))
                        return "op_type: string expected";
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
                if (message.doc_string != null && message.hasOwnProperty("doc_string"))
                    if (!$util.isString(message.doc_string))
                        return "doc_string: string expected";
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
                if (object.op_type != null)
                    message.op_type = String(object.op_type);
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
                if (object.doc_string != null)
                    message.doc_string = String(object.doc_string);
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
                    object.op_type = "";
                    object.doc_string = "";
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
                if (message.op_type != null && message.hasOwnProperty("op_type"))
                    object.op_type = message.op_type;
                if (message.attribute && message.attribute.length) {
                    object.attribute = [];
                    for (var j = 0; j < message.attribute.length; ++j)
                        object.attribute[j] = $root.onnx.AttributeProto.toObject(message.attribute[j], options);
                }
                if (message.doc_string != null && message.hasOwnProperty("doc_string"))
                    object.doc_string = message.doc_string;
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
                this.opset_import = [];
                this.metadata_props = [];
                if (properties)
                    for (var keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                        if (properties[keys[i]] != null)
                            this[keys[i]] = properties[keys[i]];
            }
    
            ModelProto.prototype.ir_version = $util.Long ? $util.Long.fromBits(0,0,false) : 0;
            ModelProto.prototype.opset_import = $util.emptyArray;
            ModelProto.prototype.producer_name = "";
            ModelProto.prototype.producer_version = "";
            ModelProto.prototype.domain = "";
            ModelProto.prototype.model_version = $util.Long ? $util.Long.fromBits(0,0,false) : 0;
            ModelProto.prototype.doc_string = "";
            ModelProto.prototype.graph = null;
            ModelProto.prototype.metadata_props = $util.emptyArray;
    
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
                        message.ir_version = reader.int64();
                        break;
                    case 8:
                        if (!(message.opset_import && message.opset_import.length))
                            message.opset_import = [];
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
                        if (!(message.metadata_props && message.metadata_props.length))
                            message.metadata_props = [];
                        message.metadata_props.push($root.onnx.StringStringEntryProto.decode(reader, reader.uint32()));
                        break;
                    default:
                        reader.skipType(tag & 7);
                        break;
                    }
                }
                return message;
            };
    
            ModelProto.decodeText = function decodeText(reader, block) {
                if (!(reader instanceof $TextReader))
                    reader = $TextReader.create(reader);
                var message = new $root.onnx.ModelProto();
                reader.start(block);
                while (!reader.end(block)) {
                    var tag = reader.tag();
                    switch (tag) {
                    case "ir_version":
                        message.ir_version = reader.int64();
                        break;
                    case "opset_import":
                        if (!(message.opset_import && message.opset_import.length))
                            message.opset_import = [];
                        message.opset_import.push($root.onnx.OperatorSetIdProto.decodeText(reader, true));
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
                        message.graph = $root.onnx.GraphProto.decodeText(reader, true);
                        break;
                    case "metadata_props":
                        if (!(message.metadata_props && message.metadata_props.length))
                            message.metadata_props = [];
                        message.metadata_props.push($root.onnx.StringStringEntryProto.decodeText(reader, true));
                        break;
                    default:
                        reader.handle(tag, message);
                        break;
                    }
                }
                return message;
            };
    
            ModelProto.verify = function verify(message) {
                if (typeof message !== "object" || message === null)
                    return "object expected";
                if (message.ir_version != null && message.hasOwnProperty("ir_version"))
                    if (!$util.isInteger(message.ir_version) && !(message.ir_version && $util.isInteger(message.ir_version.low) && $util.isInteger(message.ir_version.high)))
                        return "ir_version: integer|Long expected";
                if (message.opset_import != null && message.hasOwnProperty("opset_import")) {
                    if (!Array.isArray(message.opset_import))
                        return "opset_import: array expected";
                    for (var i = 0; i < message.opset_import.length; ++i) {
                        var error = $root.onnx.OperatorSetIdProto.verify(message.opset_import[i]);
                        if (error)
                            return "opset_import." + error;
                    }
                }
                if (message.producer_name != null && message.hasOwnProperty("producer_name"))
                    if (!$util.isString(message.producer_name))
                        return "producer_name: string expected";
                if (message.producer_version != null && message.hasOwnProperty("producer_version"))
                    if (!$util.isString(message.producer_version))
                        return "producer_version: string expected";
                if (message.domain != null && message.hasOwnProperty("domain"))
                    if (!$util.isString(message.domain))
                        return "domain: string expected";
                if (message.model_version != null && message.hasOwnProperty("model_version"))
                    if (!$util.isInteger(message.model_version) && !(message.model_version && $util.isInteger(message.model_version.low) && $util.isInteger(message.model_version.high)))
                        return "model_version: integer|Long expected";
                if (message.doc_string != null && message.hasOwnProperty("doc_string"))
                    if (!$util.isString(message.doc_string))
                        return "doc_string: string expected";
                if (message.graph != null && message.hasOwnProperty("graph")) {
                    var error = $root.onnx.GraphProto.verify(message.graph);
                    if (error)
                        return "graph." + error;
                }
                if (message.metadata_props != null && message.hasOwnProperty("metadata_props")) {
                    if (!Array.isArray(message.metadata_props))
                        return "metadata_props: array expected";
                    for (var i = 0; i < message.metadata_props.length; ++i) {
                        var error = $root.onnx.StringStringEntryProto.verify(message.metadata_props[i]);
                        if (error)
                            return "metadata_props." + error;
                    }
                }
                return null;
            };
    
            ModelProto.fromObject = function fromObject(object) {
                if (object instanceof $root.onnx.ModelProto)
                    return object;
                var message = new $root.onnx.ModelProto();
                if (object.ir_version != null)
                    if ($util.Long)
                        (message.ir_version = $util.Long.fromValue(object.ir_version)).unsigned = false;
                    else if (typeof object.ir_version === "string")
                        message.ir_version = parseInt(object.ir_version, 10);
                    else if (typeof object.ir_version === "number")
                        message.ir_version = object.ir_version;
                    else if (typeof object.ir_version === "object")
                        message.ir_version = new $util.LongBits(object.ir_version.low >>> 0, object.ir_version.high >>> 0).toNumber();
                if (object.opset_import) {
                    if (!Array.isArray(object.opset_import))
                        throw TypeError(".onnx.ModelProto.opset_import: array expected");
                    message.opset_import = [];
                    for (var i = 0; i < object.opset_import.length; ++i) {
                        if (typeof object.opset_import[i] !== "object")
                            throw TypeError(".onnx.ModelProto.opset_import: object expected");
                        message.opset_import[i] = $root.onnx.OperatorSetIdProto.fromObject(object.opset_import[i]);
                    }
                }
                if (object.producer_name != null)
                    message.producer_name = String(object.producer_name);
                if (object.producer_version != null)
                    message.producer_version = String(object.producer_version);
                if (object.domain != null)
                    message.domain = String(object.domain);
                if (object.model_version != null)
                    if ($util.Long)
                        (message.model_version = $util.Long.fromValue(object.model_version)).unsigned = false;
                    else if (typeof object.model_version === "string")
                        message.model_version = parseInt(object.model_version, 10);
                    else if (typeof object.model_version === "number")
                        message.model_version = object.model_version;
                    else if (typeof object.model_version === "object")
                        message.model_version = new $util.LongBits(object.model_version.low >>> 0, object.model_version.high >>> 0).toNumber();
                if (object.doc_string != null)
                    message.doc_string = String(object.doc_string);
                if (object.graph != null) {
                    if (typeof object.graph !== "object")
                        throw TypeError(".onnx.ModelProto.graph: object expected");
                    message.graph = $root.onnx.GraphProto.fromObject(object.graph);
                }
                if (object.metadata_props) {
                    if (!Array.isArray(object.metadata_props))
                        throw TypeError(".onnx.ModelProto.metadata_props: array expected");
                    message.metadata_props = [];
                    for (var i = 0; i < object.metadata_props.length; ++i) {
                        if (typeof object.metadata_props[i] !== "object")
                            throw TypeError(".onnx.ModelProto.metadata_props: object expected");
                        message.metadata_props[i] = $root.onnx.StringStringEntryProto.fromObject(object.metadata_props[i]);
                    }
                }
                return message;
            };
    
            ModelProto.toObject = function toObject(message, options) {
                if (!options)
                    options = {};
                var object = {};
                if (options.arrays || options.defaults) {
                    object.opset_import = [];
                    object.metadata_props = [];
                }
                if (options.defaults) {
                    if ($util.Long) {
                        var long = new $util.Long(0, 0, false);
                        object.ir_version = options.longs === String ? long.toString() : options.longs === Number ? long.toNumber() : long;
                    } else
                        object.ir_version = options.longs === String ? "0" : 0;
                    object.producer_name = "";
                    object.producer_version = "";
                    object.domain = "";
                    if ($util.Long) {
                        var long = new $util.Long(0, 0, false);
                        object.model_version = options.longs === String ? long.toString() : options.longs === Number ? long.toNumber() : long;
                    } else
                        object.model_version = options.longs === String ? "0" : 0;
                    object.doc_string = "";
                    object.graph = null;
                }
                if (message.ir_version != null && message.hasOwnProperty("ir_version"))
                    if (typeof message.ir_version === "number")
                        object.ir_version = options.longs === String ? String(message.ir_version) : message.ir_version;
                    else
                        object.ir_version = options.longs === String ? $util.Long.prototype.toString.call(message.ir_version) : options.longs === Number ? new $util.LongBits(message.ir_version.low >>> 0, message.ir_version.high >>> 0).toNumber() : message.ir_version;
                if (message.producer_name != null && message.hasOwnProperty("producer_name"))
                    object.producer_name = message.producer_name;
                if (message.producer_version != null && message.hasOwnProperty("producer_version"))
                    object.producer_version = message.producer_version;
                if (message.domain != null && message.hasOwnProperty("domain"))
                    object.domain = message.domain;
                if (message.model_version != null && message.hasOwnProperty("model_version"))
                    if (typeof message.model_version === "number")
                        object.model_version = options.longs === String ? String(message.model_version) : message.model_version;
                    else
                        object.model_version = options.longs === String ? $util.Long.prototype.toString.call(message.model_version) : options.longs === Number ? new $util.LongBits(message.model_version.low >>> 0, message.model_version.high >>> 0).toNumber() : message.model_version;
                if (message.doc_string != null && message.hasOwnProperty("doc_string"))
                    object.doc_string = message.doc_string;
                if (message.graph != null && message.hasOwnProperty("graph"))
                    object.graph = $root.onnx.GraphProto.toObject(message.graph, options);
                if (message.opset_import && message.opset_import.length) {
                    object.opset_import = [];
                    for (var j = 0; j < message.opset_import.length; ++j)
                        object.opset_import[j] = $root.onnx.OperatorSetIdProto.toObject(message.opset_import[j], options);
                }
                if (message.metadata_props && message.metadata_props.length) {
                    object.metadata_props = [];
                    for (var j = 0; j < message.metadata_props.length; ++j)
                        object.metadata_props[j] = $root.onnx.StringStringEntryProto.toObject(message.metadata_props[j], options);
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
    
            StringStringEntryProto.decodeText = function decodeText(reader, block) {
                if (!(reader instanceof $TextReader))
                    reader = $TextReader.create(reader);
                var message = new $root.onnx.StringStringEntryProto();
                reader.start(block);
                while (!reader.end(block)) {
                    var tag = reader.tag();
                    switch (tag) {
                    case "key":
                        message.key = reader.string();
                        break;
                    case "value":
                        message.value = reader.string();
                        break;
                    default:
                        reader.handle(tag, message);
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
                this.value_info = [];
                if (properties)
                    for (var keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                        if (properties[keys[i]] != null)
                            this[keys[i]] = properties[keys[i]];
            }
    
            GraphProto.prototype.node = $util.emptyArray;
            GraphProto.prototype.name = "";
            GraphProto.prototype.initializer = $util.emptyArray;
            GraphProto.prototype.doc_string = "";
            GraphProto.prototype.input = $util.emptyArray;
            GraphProto.prototype.output = $util.emptyArray;
            GraphProto.prototype.value_info = $util.emptyArray;
    
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
                        message.doc_string = reader.string();
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
                        if (!(message.value_info && message.value_info.length))
                            message.value_info = [];
                        message.value_info.push($root.onnx.ValueInfoProto.decode(reader, reader.uint32()));
                        break;
                    default:
                        reader.skipType(tag & 7);
                        break;
                    }
                }
                return message;
            };
    
            GraphProto.decodeText = function decodeText(reader, block) {
                if (!(reader instanceof $TextReader))
                    reader = $TextReader.create(reader);
                var message = new $root.onnx.GraphProto();
                reader.start(block);
                while (!reader.end(block)) {
                    var tag = reader.tag();
                    switch (tag) {
                    case "node":
                        if (!(message.node && message.node.length))
                            message.node = [];
                        message.node.push($root.onnx.NodeProto.decodeText(reader, true));
                        break;
                    case "name":
                        message.name = reader.string();
                        break;
                    case "initializer":
                        if (!(message.initializer && message.initializer.length))
                            message.initializer = [];
                        message.initializer.push($root.onnx.TensorProto.decodeText(reader, true));
                        break;
                    case "doc_string":
                        message.doc_string = reader.string();
                        break;
                    case "input":
                        if (!(message.input && message.input.length))
                            message.input = [];
                        message.input.push($root.onnx.ValueInfoProto.decodeText(reader, true));
                        break;
                    case "output":
                        if (!(message.output && message.output.length))
                            message.output = [];
                        message.output.push($root.onnx.ValueInfoProto.decodeText(reader, true));
                        break;
                    case "value_info":
                        if (!(message.value_info && message.value_info.length))
                            message.value_info = [];
                        message.value_info.push($root.onnx.ValueInfoProto.decodeText(reader, true));
                        break;
                    default:
                        reader.handle(tag, message);
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
                if (message.doc_string != null && message.hasOwnProperty("doc_string"))
                    if (!$util.isString(message.doc_string))
                        return "doc_string: string expected";
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
                if (message.value_info != null && message.hasOwnProperty("value_info")) {
                    if (!Array.isArray(message.value_info))
                        return "value_info: array expected";
                    for (var i = 0; i < message.value_info.length; ++i) {
                        var error = $root.onnx.ValueInfoProto.verify(message.value_info[i]);
                        if (error)
                            return "value_info." + error;
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
                if (object.doc_string != null)
                    message.doc_string = String(object.doc_string);
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
                if (object.value_info) {
                    if (!Array.isArray(object.value_info))
                        throw TypeError(".onnx.GraphProto.value_info: array expected");
                    message.value_info = [];
                    for (var i = 0; i < object.value_info.length; ++i) {
                        if (typeof object.value_info[i] !== "object")
                            throw TypeError(".onnx.GraphProto.value_info: object expected");
                        message.value_info[i] = $root.onnx.ValueInfoProto.fromObject(object.value_info[i]);
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
                    object.value_info = [];
                }
                if (options.defaults) {
                    object.name = "";
                    object.doc_string = "";
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
                if (message.doc_string != null && message.hasOwnProperty("doc_string"))
                    object.doc_string = message.doc_string;
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
                if (message.value_info && message.value_info.length) {
                    object.value_info = [];
                    for (var j = 0; j < message.value_info.length; ++j)
                        object.value_info[j] = $root.onnx.ValueInfoProto.toObject(message.value_info[j], options);
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
                this.float_data = [];
                this.int32_data = [];
                this.string_data = [];
                this.int64_data = [];
                this.double_data = [];
                this.uint64_data = [];
                if (properties)
                    for (var keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                        if (properties[keys[i]] != null)
                            this[keys[i]] = properties[keys[i]];
            }
    
            TensorProto.prototype.dims = $util.emptyArray;
            TensorProto.prototype.data_type = 0;
            TensorProto.prototype.segment = null;
            TensorProto.prototype.float_data = $util.emptyArray;
            TensorProto.prototype.int32_data = $util.emptyArray;
            TensorProto.prototype.string_data = $util.emptyArray;
            TensorProto.prototype.int64_data = $util.emptyArray;
            TensorProto.prototype.name = "";
            TensorProto.prototype.doc_string = "";
            TensorProto.prototype.raw_data = $util.newBuffer([]);
            TensorProto.prototype.double_data = $util.emptyArray;
            TensorProto.prototype.uint64_data = $util.emptyArray;
    
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
                        message.data_type = reader.int32();
                        break;
                    case 3:
                        message.segment = $root.onnx.TensorProto.Segment.decode(reader, reader.uint32());
                        break;
                    case 4:
                        if (!(message.float_data && message.float_data.length))
                            message.float_data = [];
                        if ((tag & 7) === 2) {
                            var end2 = reader.uint32() + reader.pos;
                            if (message.float_data.length == 0 && (end2 - reader.pos) > 1048576) {
                                var float_dataLength = end2 - reader.pos;
                                var float_dataView = new DataView(reader.buf.buffer, reader.buf.byteOffset + reader.pos, float_dataLength);
                                float_dataLength = float_dataLength >>> 2;
                                var float_data = new Float32Array(float_dataLength);
                                for (var i = 0; i < float_dataLength; i++) {
                                    float_data[i] = float_dataView.getFloat32(i << 2, true);
                                }
                                message.float_data = float_data;
                                reader.pos = end2;
                            }
                            else {
                                while (reader.pos < end2)
                                    message.float_data.push(reader.float());
                            }
                        } else
                            message.float_data.push(reader.float());
                        break;
                    case 5:
                        if (!(message.int32_data && message.int32_data.length))
                            message.int32_data = [];
                        if ((tag & 7) === 2) {
                            var end2 = reader.uint32() + reader.pos;
                            while (reader.pos < end2)
                                message.int32_data.push(reader.int32());
                        } else
                            message.int32_data.push(reader.int32());
                        break;
                    case 6:
                        if (!(message.string_data && message.string_data.length))
                            message.string_data = [];
                        message.string_data.push(reader.bytes());
                        break;
                    case 7:
                        if (!(message.int64_data && message.int64_data.length))
                            message.int64_data = [];
                        if ((tag & 7) === 2) {
                            var end2 = reader.uint32() + reader.pos;
                            while (reader.pos < end2)
                                message.int64_data.push(reader.int64());
                        } else
                            message.int64_data.push(reader.int64());
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
                    case 10:
                        if (!(message.double_data && message.double_data.length))
                            message.double_data = [];
                        if ((tag & 7) === 2) {
                            var end2 = reader.uint32() + reader.pos;
                            if (message.double_data.length == 0 && (end2 - reader.pos) > 1048576) {
                                var double_dataLength = end2 - reader.pos;
                                var double_dataView = new DataView(reader.buf.buffer, reader.buf.byteOffset + reader.pos, double_dataLength);
                                double_dataLength = double_dataLength >>> 3;
                                var double_data = new Float64Array(double_dataLength);
                                for (var i = 0; i < double_dataLength; i++) {
                                    double_data[i] = double_dataView.getFloat64(i << 3, true);
                                }
                                message.double_data = double_data;
                                reader.pos = end2;
                            }
                            else {
                                while (reader.pos < end2)
                                    message.double_data.push(reader.double());
                            }
                        } else
                            message.double_data.push(reader.double());
                        break;
                    case 11:
                        if (!(message.uint64_data && message.uint64_data.length))
                            message.uint64_data = [];
                        if ((tag & 7) === 2) {
                            var end2 = reader.uint32() + reader.pos;
                            while (reader.pos < end2)
                                message.uint64_data.push(reader.uint64());
                        } else
                            message.uint64_data.push(reader.uint64());
                        break;
                    default:
                        reader.skipType(tag & 7);
                        break;
                    }
                }
                return message;
            };
    
            TensorProto.decodeText = function decodeText(reader, block) {
                if (!(reader instanceof $TextReader))
                    reader = $TextReader.create(reader);
                var message = new $root.onnx.TensorProto();
                reader.start(block);
                while (!reader.end(block)) {
                    var tag = reader.tag();
                    switch (tag) {
                    case "dims":
                        if (!(message.dims && message.dims.length))
                            message.dims = [];
                        message.dims.push(reader.int64());
                        break;
                    case "data_type":
                        message.data_type = reader.int32();
                        break;
                    case "segment":
                        message.segment = $root.onnx.TensorProto.Segment.decodeText(reader, true);
                        break;
                    case "float_data":
                        if (!(message.float_data && message.float_data.length))
                            message.float_data = [];
                        message.float_data.push(reader.float());
                        break;
                    case "int32_data":
                        if (!(message.int32_data && message.int32_data.length))
                            message.int32_data = [];
                        message.int32_data.push(reader.int32());
                        break;
                    case "string_data":
                        if (!(message.string_data && message.string_data.length))
                            message.string_data = [];
                        message.string_data.push(reader.bytes());
                        break;
                    case "int64_data":
                        if (!(message.int64_data && message.int64_data.length))
                            message.int64_data = [];
                        message.int64_data.push(reader.int64());
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
                    case "double_data":
                        if (!(message.double_data && message.double_data.length))
                            message.double_data = [];
                        message.double_data.push(reader.double());
                        break;
                    case "uint64_data":
                        if (!(message.uint64_data && message.uint64_data.length))
                            message.uint64_data = [];
                        message.uint64_data.push(reader.uint64());
                        break;
                    default:
                        reader.handle(tag, message);
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
                if (message.data_type != null && message.hasOwnProperty("data_type"))
                    if (!$util.isInteger(message.data_type))
                        return "data_type: integer expected";
                if (message.segment != null && message.hasOwnProperty("segment")) {
                    var error = $root.onnx.TensorProto.Segment.verify(message.segment);
                    if (error)
                        return "segment." + error;
                }
                if (message.float_data != null && message.hasOwnProperty("float_data")) {
                    if (!Array.isArray(message.float_data))
                        return "float_data: array expected";
                    for (var i = 0; i < message.float_data.length; ++i)
                        if (typeof message.float_data[i] !== "number")
                            return "float_data: number[] expected";
                }
                if (message.int32_data != null && message.hasOwnProperty("int32_data")) {
                    if (!Array.isArray(message.int32_data))
                        return "int32_data: array expected";
                    for (var i = 0; i < message.int32_data.length; ++i)
                        if (!$util.isInteger(message.int32_data[i]))
                            return "int32_data: integer[] expected";
                }
                if (message.string_data != null && message.hasOwnProperty("string_data")) {
                    if (!Array.isArray(message.string_data))
                        return "string_data: array expected";
                    for (var i = 0; i < message.string_data.length; ++i)
                        if (!(message.string_data[i] && typeof message.string_data[i].length === "number" || $util.isString(message.string_data[i])))
                            return "string_data: buffer[] expected";
                }
                if (message.int64_data != null && message.hasOwnProperty("int64_data")) {
                    if (!Array.isArray(message.int64_data))
                        return "int64_data: array expected";
                    for (var i = 0; i < message.int64_data.length; ++i)
                        if (!$util.isInteger(message.int64_data[i]) && !(message.int64_data[i] && $util.isInteger(message.int64_data[i].low) && $util.isInteger(message.int64_data[i].high)))
                            return "int64_data: integer|Long[] expected";
                }
                if (message.name != null && message.hasOwnProperty("name"))
                    if (!$util.isString(message.name))
                        return "name: string expected";
                if (message.doc_string != null && message.hasOwnProperty("doc_string"))
                    if (!$util.isString(message.doc_string))
                        return "doc_string: string expected";
                if (message.raw_data != null && message.hasOwnProperty("raw_data"))
                    if (!(message.raw_data && typeof message.raw_data.length === "number" || $util.isString(message.raw_data)))
                        return "raw_data: buffer expected";
                if (message.double_data != null && message.hasOwnProperty("double_data")) {
                    if (!Array.isArray(message.double_data))
                        return "double_data: array expected";
                    for (var i = 0; i < message.double_data.length; ++i)
                        if (typeof message.double_data[i] !== "number")
                            return "double_data: number[] expected";
                }
                if (message.uint64_data != null && message.hasOwnProperty("uint64_data")) {
                    if (!Array.isArray(message.uint64_data))
                        return "uint64_data: array expected";
                    for (var i = 0; i < message.uint64_data.length; ++i)
                        if (!$util.isInteger(message.uint64_data[i]) && !(message.uint64_data[i] && $util.isInteger(message.uint64_data[i].low) && $util.isInteger(message.uint64_data[i].high)))
                            return "uint64_data: integer|Long[] expected";
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
                if (object.data_type != null)
                    message.data_type = object.data_type | 0;
                if (object.segment != null) {
                    if (typeof object.segment !== "object")
                        throw TypeError(".onnx.TensorProto.segment: object expected");
                    message.segment = $root.onnx.TensorProto.Segment.fromObject(object.segment);
                }
                if (object.float_data) {
                    if (!Array.isArray(object.float_data))
                        throw TypeError(".onnx.TensorProto.float_data: array expected");
                    message.float_data = [];
                    for (var i = 0; i < object.float_data.length; ++i)
                        message.float_data[i] = Number(object.float_data[i]);
                }
                if (object.int32_data) {
                    if (!Array.isArray(object.int32_data))
                        throw TypeError(".onnx.TensorProto.int32_data: array expected");
                    message.int32_data = [];
                    for (var i = 0; i < object.int32_data.length; ++i)
                        message.int32_data[i] = object.int32_data[i] | 0;
                }
                if (object.string_data) {
                    if (!Array.isArray(object.string_data))
                        throw TypeError(".onnx.TensorProto.string_data: array expected");
                    message.string_data = [];
                    for (var i = 0; i < object.string_data.length; ++i)
                        if (typeof object.string_data[i] === "string")
                            $util.base64.decode(object.string_data[i], message.string_data[i] = $util.newBuffer($util.base64.length(object.string_data[i])), 0);
                        else if (object.string_data[i].length)
                            message.string_data[i] = object.string_data[i];
                }
                if (object.int64_data) {
                    if (!Array.isArray(object.int64_data))
                        throw TypeError(".onnx.TensorProto.int64_data: array expected");
                    message.int64_data = [];
                    for (var i = 0; i < object.int64_data.length; ++i)
                        if ($util.Long)
                            (message.int64_data[i] = $util.Long.fromValue(object.int64_data[i])).unsigned = false;
                        else if (typeof object.int64_data[i] === "string")
                            message.int64_data[i] = parseInt(object.int64_data[i], 10);
                        else if (typeof object.int64_data[i] === "number")
                            message.int64_data[i] = object.int64_data[i];
                        else if (typeof object.int64_data[i] === "object")
                            message.int64_data[i] = new $util.LongBits(object.int64_data[i].low >>> 0, object.int64_data[i].high >>> 0).toNumber();
                }
                if (object.name != null)
                    message.name = String(object.name);
                if (object.doc_string != null)
                    message.doc_string = String(object.doc_string);
                if (object.raw_data != null)
                    if (typeof object.raw_data === "string")
                        $util.base64.decode(object.raw_data, message.raw_data = $util.newBuffer($util.base64.length(object.raw_data)), 0);
                    else if (object.raw_data.length)
                        message.raw_data = object.raw_data;
                if (object.double_data) {
                    if (!Array.isArray(object.double_data))
                        throw TypeError(".onnx.TensorProto.double_data: array expected");
                    message.double_data = [];
                    for (var i = 0; i < object.double_data.length; ++i)
                        message.double_data[i] = Number(object.double_data[i]);
                }
                if (object.uint64_data) {
                    if (!Array.isArray(object.uint64_data))
                        throw TypeError(".onnx.TensorProto.uint64_data: array expected");
                    message.uint64_data = [];
                    for (var i = 0; i < object.uint64_data.length; ++i)
                        if ($util.Long)
                            (message.uint64_data[i] = $util.Long.fromValue(object.uint64_data[i])).unsigned = true;
                        else if (typeof object.uint64_data[i] === "string")
                            message.uint64_data[i] = parseInt(object.uint64_data[i], 10);
                        else if (typeof object.uint64_data[i] === "number")
                            message.uint64_data[i] = object.uint64_data[i];
                        else if (typeof object.uint64_data[i] === "object")
                            message.uint64_data[i] = new $util.LongBits(object.uint64_data[i].low >>> 0, object.uint64_data[i].high >>> 0).toNumber(true);
                }
                return message;
            };
    
            TensorProto.toObject = function toObject(message, options) {
                if (!options)
                    options = {};
                var object = {};
                if (options.arrays || options.defaults) {
                    object.dims = [];
                    object.float_data = [];
                    object.int32_data = [];
                    object.string_data = [];
                    object.int64_data = [];
                    object.double_data = [];
                    object.uint64_data = [];
                }
                if (options.defaults) {
                    object.data_type = 0;
                    object.segment = null;
                    object.name = "";
                    if (options.bytes === String)
                        object.raw_data = "";
                    else {
                        object.raw_data = [];
                        if (options.bytes !== Array)
                            object.raw_data = $util.newBuffer(object.raw_data);
                    }
                    object.doc_string = "";
                }
                if (message.dims && message.dims.length) {
                    object.dims = [];
                    for (var j = 0; j < message.dims.length; ++j)
                        if (typeof message.dims[j] === "number")
                            object.dims[j] = options.longs === String ? String(message.dims[j]) : message.dims[j];
                        else
                            object.dims[j] = options.longs === String ? $util.Long.prototype.toString.call(message.dims[j]) : options.longs === Number ? new $util.LongBits(message.dims[j].low >>> 0, message.dims[j].high >>> 0).toNumber() : message.dims[j];
                }
                if (message.data_type != null && message.hasOwnProperty("data_type"))
                    object.data_type = message.data_type;
                if (message.segment != null && message.hasOwnProperty("segment"))
                    object.segment = $root.onnx.TensorProto.Segment.toObject(message.segment, options);
                if (message.float_data && message.float_data.length) {
                    object.float_data = [];
                    for (var j = 0; j < message.float_data.length; ++j)
                        object.float_data[j] = options.json && !isFinite(message.float_data[j]) ? String(message.float_data[j]) : message.float_data[j];
                }
                if (message.int32_data && message.int32_data.length) {
                    object.int32_data = [];
                    for (var j = 0; j < message.int32_data.length; ++j)
                        object.int32_data[j] = message.int32_data[j];
                }
                if (message.string_data && message.string_data.length) {
                    object.string_data = [];
                    for (var j = 0; j < message.string_data.length; ++j)
                        object.string_data[j] = options.bytes === String ? $util.base64.encode(message.string_data[j], 0, message.string_data[j].length) : options.bytes === Array ? Array.prototype.slice.call(message.string_data[j]) : message.string_data[j];
                }
                if (message.int64_data && message.int64_data.length) {
                    object.int64_data = [];
                    for (var j = 0; j < message.int64_data.length; ++j)
                        if (typeof message.int64_data[j] === "number")
                            object.int64_data[j] = options.longs === String ? String(message.int64_data[j]) : message.int64_data[j];
                        else
                            object.int64_data[j] = options.longs === String ? $util.Long.prototype.toString.call(message.int64_data[j]) : options.longs === Number ? new $util.LongBits(message.int64_data[j].low >>> 0, message.int64_data[j].high >>> 0).toNumber() : message.int64_data[j];
                }
                if (message.name != null && message.hasOwnProperty("name"))
                    object.name = message.name;
                if (message.raw_data != null && message.hasOwnProperty("raw_data"))
                    object.raw_data = options.bytes === String ? $util.base64.encode(message.raw_data, 0, message.raw_data.length) : options.bytes === Array ? Array.prototype.slice.call(message.raw_data) : message.raw_data;
                if (message.double_data && message.double_data.length) {
                    object.double_data = [];
                    for (var j = 0; j < message.double_data.length; ++j)
                        object.double_data[j] = options.json && !isFinite(message.double_data[j]) ? String(message.double_data[j]) : message.double_data[j];
                }
                if (message.uint64_data && message.uint64_data.length) {
                    object.uint64_data = [];
                    for (var j = 0; j < message.uint64_data.length; ++j)
                        if (typeof message.uint64_data[j] === "number")
                            object.uint64_data[j] = options.longs === String ? String(message.uint64_data[j]) : message.uint64_data[j];
                        else
                            object.uint64_data[j] = options.longs === String ? $util.Long.prototype.toString.call(message.uint64_data[j]) : options.longs === Number ? new $util.LongBits(message.uint64_data[j].low >>> 0, message.uint64_data[j].high >>> 0).toNumber(true) : message.uint64_data[j];
                }
                if (message.doc_string != null && message.hasOwnProperty("doc_string"))
                    object.doc_string = message.doc_string;
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
                values[valuesById[16] = "BFLOAT16"] = 16;
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
    
                Segment.decodeText = function decodeText(reader, block) {
                    if (!(reader instanceof $TextReader))
                        reader = $TextReader.create(reader);
                    var message = new $root.onnx.TensorProto.Segment();
                    reader.start(block);
                    while (!reader.end(block)) {
                        var tag = reader.tag();
                        switch (tag) {
                        case "begin":
                            message.begin = reader.int64();
                            break;
                        case "end":
                            message.end = reader.int64();
                            break;
                        default:
                            reader.handle(tag, message);
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
    
            TensorShapeProto.decodeText = function decodeText(reader, block) {
                if (!(reader instanceof $TextReader))
                    reader = $TextReader.create(reader);
                var message = new $root.onnx.TensorShapeProto();
                reader.start(block);
                while (!reader.end(block)) {
                    var tag = reader.tag();
                    switch (tag) {
                    case "dim":
                        if (!(message.dim && message.dim.length))
                            message.dim = [];
                        message.dim.push($root.onnx.TensorShapeProto.Dimension.decodeText(reader, true));
                        break;
                    default:
                        reader.handle(tag, message);
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
    
                Dimension.prototype.dim_value = $util.Long ? $util.Long.fromBits(0,0,false) : 0;
                Dimension.prototype.dim_param = "";
                Dimension.prototype.denotation = "";
    
                var $oneOfFields;
    
                Object.defineProperty(Dimension.prototype, "value", {
                    get: $util.oneOfGetter($oneOfFields = ["dim_value", "dim_param"]),
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
                };
    
                Dimension.decodeText = function decodeText(reader, block) {
                    if (!(reader instanceof $TextReader))
                        reader = $TextReader.create(reader);
                    var message = new $root.onnx.TensorShapeProto.Dimension();
                    reader.start(block);
                    while (!reader.end(block)) {
                        var tag = reader.tag();
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
                            reader.handle(tag, message);
                            break;
                        }
                    }
                    return message;
                };
    
                Dimension.verify = function verify(message) {
                    if (typeof message !== "object" || message === null)
                        return "object expected";
                    var properties = {};
                    if (message.dim_value != null && message.hasOwnProperty("dim_value")) {
                        properties.value = 1;
                        if (!$util.isInteger(message.dim_value) && !(message.dim_value && $util.isInteger(message.dim_value.low) && $util.isInteger(message.dim_value.high)))
                            return "dim_value: integer|Long expected";
                    }
                    if (message.dim_param != null && message.hasOwnProperty("dim_param")) {
                        if (properties.value === 1)
                            return "value: multiple values";
                        properties.value = 1;
                        if (!$util.isString(message.dim_param))
                            return "dim_param: string expected";
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
                    if (object.dim_value != null)
                        if ($util.Long)
                            (message.dim_value = $util.Long.fromValue(object.dim_value)).unsigned = false;
                        else if (typeof object.dim_value === "string")
                            message.dim_value = parseInt(object.dim_value, 10);
                        else if (typeof object.dim_value === "number")
                            message.dim_value = object.dim_value;
                        else if (typeof object.dim_value === "object")
                            message.dim_value = new $util.LongBits(object.dim_value.low >>> 0, object.dim_value.high >>> 0).toNumber();
                    if (object.dim_param != null)
                        message.dim_param = String(object.dim_param);
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
                    if (message.dim_value != null && message.hasOwnProperty("dim_value")) {
                        if (typeof message.dim_value === "number")
                            object.dim_value = options.longs === String ? String(message.dim_value) : message.dim_value;
                        else
                            object.dim_value = options.longs === String ? $util.Long.prototype.toString.call(message.dim_value) : options.longs === Number ? new $util.LongBits(message.dim_value.low >>> 0, message.dim_value.high >>> 0).toNumber() : message.dim_value;
                        if (options.oneofs)
                            object.value = "dim_value";
                    }
                    if (message.dim_param != null && message.hasOwnProperty("dim_param")) {
                        object.dim_param = message.dim_param;
                        if (options.oneofs)
                            object.value = "dim_param";
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
    
            TypeProto.prototype.tensor_type = null;
            TypeProto.prototype.sequence_type = null;
            TypeProto.prototype.map_type = null;
            TypeProto.prototype.opaque_type = null;
            TypeProto.prototype.sparse_tensor_type = null;
            TypeProto.prototype.denotation = "";
    
            var $oneOfFields;
    
            Object.defineProperty(TypeProto.prototype, "value", {
                get: $util.oneOfGetter($oneOfFields = ["tensor_type", "sequence_type", "map_type", "opaque_type", "sparse_tensor_type"]),
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
                        message.tensor_type = $root.onnx.TypeProto.Tensor.decode(reader, reader.uint32());
                        break;
                    case 4:
                        message.sequence_type = $root.onnx.TypeProto.Sequence.decode(reader, reader.uint32());
                        break;
                    case 5:
                        message.map_type = $root.onnx.TypeProto.Map.decode(reader, reader.uint32());
                        break;
                    case 7:
                        message.opaque_type = $root.onnx.TypeProto.Opaque.decode(reader, reader.uint32());
                        break;
                    case 8:
                        message.sparse_tensor_type = $root.onnx.TypeProto.SparseTensor.decode(reader, reader.uint32());
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
    
            TypeProto.decodeText = function decodeText(reader, block) {
                if (!(reader instanceof $TextReader))
                    reader = $TextReader.create(reader);
                var message = new $root.onnx.TypeProto();
                reader.start(block);
                while (!reader.end(block)) {
                    var tag = reader.tag();
                    switch (tag) {
                    case "tensor_type":
                        message.tensor_type = $root.onnx.TypeProto.Tensor.decodeText(reader, true);
                        break;
                    case "sequence_type":
                        message.sequence_type = $root.onnx.TypeProto.Sequence.decodeText(reader, true);
                        break;
                    case "map_type":
                        message.map_type = $root.onnx.TypeProto.Map.decodeText(reader, true);
                        break;
                    case "opaque_type":
                        message.opaque_type = $root.onnx.TypeProto.Opaque.decodeText(reader, true);
                        break;
                    case "sparse_tensor_type":
                        message.sparse_tensor_type = $root.onnx.TypeProto.SparseTensor.decodeText(reader, true);
                        break;
                    case "denotation":
                        message.denotation = reader.string();
                        break;
                    default:
                        reader.handle(tag, message);
                        break;
                    }
                }
                return message;
            };
    
            TypeProto.verify = function verify(message) {
                if (typeof message !== "object" || message === null)
                    return "object expected";
                var properties = {};
                if (message.tensor_type != null && message.hasOwnProperty("tensor_type")) {
                    properties.value = 1;
                    {
                        var error = $root.onnx.TypeProto.Tensor.verify(message.tensor_type);
                        if (error)
                            return "tensor_type." + error;
                    }
                }
                if (message.sequence_type != null && message.hasOwnProperty("sequence_type")) {
                    if (properties.value === 1)
                        return "value: multiple values";
                    properties.value = 1;
                    {
                        var error = $root.onnx.TypeProto.Sequence.verify(message.sequence_type);
                        if (error)
                            return "sequence_type." + error;
                    }
                }
                if (message.map_type != null && message.hasOwnProperty("map_type")) {
                    if (properties.value === 1)
                        return "value: multiple values";
                    properties.value = 1;
                    {
                        var error = $root.onnx.TypeProto.Map.verify(message.map_type);
                        if (error)
                            return "map_type." + error;
                    }
                }
                if (message.opaque_type != null && message.hasOwnProperty("opaque_type")) {
                    if (properties.value === 1)
                        return "value: multiple values";
                    properties.value = 1;
                    {
                        var error = $root.onnx.TypeProto.Opaque.verify(message.opaque_type);
                        if (error)
                            return "opaque_type." + error;
                    }
                }
                if (message.sparse_tensor_type != null && message.hasOwnProperty("sparse_tensor_type")) {
                    if (properties.value === 1)
                        return "value: multiple values";
                    properties.value = 1;
                    {
                        var error = $root.onnx.TypeProto.SparseTensor.verify(message.sparse_tensor_type);
                        if (error)
                            return "sparse_tensor_type." + error;
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
                if (object.tensor_type != null) {
                    if (typeof object.tensor_type !== "object")
                        throw TypeError(".onnx.TypeProto.tensor_type: object expected");
                    message.tensor_type = $root.onnx.TypeProto.Tensor.fromObject(object.tensor_type);
                }
                if (object.sequence_type != null) {
                    if (typeof object.sequence_type !== "object")
                        throw TypeError(".onnx.TypeProto.sequence_type: object expected");
                    message.sequence_type = $root.onnx.TypeProto.Sequence.fromObject(object.sequence_type);
                }
                if (object.map_type != null) {
                    if (typeof object.map_type !== "object")
                        throw TypeError(".onnx.TypeProto.map_type: object expected");
                    message.map_type = $root.onnx.TypeProto.Map.fromObject(object.map_type);
                }
                if (object.opaque_type != null) {
                    if (typeof object.opaque_type !== "object")
                        throw TypeError(".onnx.TypeProto.opaque_type: object expected");
                    message.opaque_type = $root.onnx.TypeProto.Opaque.fromObject(object.opaque_type);
                }
                if (object.sparse_tensor_type != null) {
                    if (typeof object.sparse_tensor_type !== "object")
                        throw TypeError(".onnx.TypeProto.sparse_tensor_type: object expected");
                    message.sparse_tensor_type = $root.onnx.TypeProto.SparseTensor.fromObject(object.sparse_tensor_type);
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
                if (message.tensor_type != null && message.hasOwnProperty("tensor_type")) {
                    object.tensor_type = $root.onnx.TypeProto.Tensor.toObject(message.tensor_type, options);
                    if (options.oneofs)
                        object.value = "tensor_type";
                }
                if (message.sequence_type != null && message.hasOwnProperty("sequence_type")) {
                    object.sequence_type = $root.onnx.TypeProto.Sequence.toObject(message.sequence_type, options);
                    if (options.oneofs)
                        object.value = "sequence_type";
                }
                if (message.map_type != null && message.hasOwnProperty("map_type")) {
                    object.map_type = $root.onnx.TypeProto.Map.toObject(message.map_type, options);
                    if (options.oneofs)
                        object.value = "map_type";
                }
                if (message.denotation != null && message.hasOwnProperty("denotation"))
                    object.denotation = message.denotation;
                if (message.opaque_type != null && message.hasOwnProperty("opaque_type")) {
                    object.opaque_type = $root.onnx.TypeProto.Opaque.toObject(message.opaque_type, options);
                    if (options.oneofs)
                        object.value = "opaque_type";
                }
                if (message.sparse_tensor_type != null && message.hasOwnProperty("sparse_tensor_type")) {
                    object.sparse_tensor_type = $root.onnx.TypeProto.SparseTensor.toObject(message.sparse_tensor_type, options);
                    if (options.oneofs)
                        object.value = "sparse_tensor_type";
                }
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
    
                Tensor.prototype.elem_type = 0;
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
                };
    
                Tensor.decodeText = function decodeText(reader, block) {
                    if (!(reader instanceof $TextReader))
                        reader = $TextReader.create(reader);
                    var message = new $root.onnx.TypeProto.Tensor();
                    reader.start(block);
                    while (!reader.end(block)) {
                        var tag = reader.tag();
                        switch (tag) {
                        case "elem_type":
                            message.elem_type = reader.int32();
                            break;
                        case "shape":
                            message.shape = $root.onnx.TensorShapeProto.decodeText(reader, true);
                            break;
                        default:
                            reader.handle(tag, message);
                            break;
                        }
                    }
                    return message;
                };
    
                Tensor.verify = function verify(message) {
                    if (typeof message !== "object" || message === null)
                        return "object expected";
                    if (message.elem_type != null && message.hasOwnProperty("elem_type"))
                        if (!$util.isInteger(message.elem_type))
                            return "elem_type: integer expected";
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
                    if (object.elem_type != null)
                        message.elem_type = object.elem_type | 0;
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
                        object.elem_type = 0;
                        object.shape = null;
                    }
                    if (message.elem_type != null && message.hasOwnProperty("elem_type"))
                        object.elem_type = message.elem_type;
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
    
                Sequence.prototype.elem_type = null;
    
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
                            message.elem_type = $root.onnx.TypeProto.decode(reader, reader.uint32());
                            break;
                        default:
                            reader.skipType(tag & 7);
                            break;
                        }
                    }
                    return message;
                };
    
                Sequence.decodeText = function decodeText(reader, block) {
                    if (!(reader instanceof $TextReader))
                        reader = $TextReader.create(reader);
                    var message = new $root.onnx.TypeProto.Sequence();
                    reader.start(block);
                    while (!reader.end(block)) {
                        var tag = reader.tag();
                        switch (tag) {
                        case "elem_type":
                            message.elem_type = $root.onnx.TypeProto.decodeText(reader, true);
                            break;
                        default:
                            reader.handle(tag, message);
                            break;
                        }
                    }
                    return message;
                };
    
                Sequence.verify = function verify(message) {
                    if (typeof message !== "object" || message === null)
                        return "object expected";
                    if (message.elem_type != null && message.hasOwnProperty("elem_type")) {
                        var error = $root.onnx.TypeProto.verify(message.elem_type);
                        if (error)
                            return "elem_type." + error;
                    }
                    return null;
                };
    
                Sequence.fromObject = function fromObject(object) {
                    if (object instanceof $root.onnx.TypeProto.Sequence)
                        return object;
                    var message = new $root.onnx.TypeProto.Sequence();
                    if (object.elem_type != null) {
                        if (typeof object.elem_type !== "object")
                            throw TypeError(".onnx.TypeProto.Sequence.elem_type: object expected");
                        message.elem_type = $root.onnx.TypeProto.fromObject(object.elem_type);
                    }
                    return message;
                };
    
                Sequence.toObject = function toObject(message, options) {
                    if (!options)
                        options = {};
                    var object = {};
                    if (options.defaults)
                        object.elem_type = null;
                    if (message.elem_type != null && message.hasOwnProperty("elem_type"))
                        object.elem_type = $root.onnx.TypeProto.toObject(message.elem_type, options);
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
    
                Map.prototype.key_type = 0;
                Map.prototype.value_type = null;
    
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
                };
    
                Map.decodeText = function decodeText(reader, block) {
                    if (!(reader instanceof $TextReader))
                        reader = $TextReader.create(reader);
                    var message = new $root.onnx.TypeProto.Map();
                    reader.start(block);
                    while (!reader.end(block)) {
                        var tag = reader.tag();
                        switch (tag) {
                        case "key_type":
                            message.key_type = reader.int32();
                            break;
                        case "value_type":
                            message.value_type = $root.onnx.TypeProto.decodeText(reader, true);
                            break;
                        default:
                            reader.handle(tag, message);
                            break;
                        }
                    }
                    return message;
                };
    
                Map.verify = function verify(message) {
                    if (typeof message !== "object" || message === null)
                        return "object expected";
                    if (message.key_type != null && message.hasOwnProperty("key_type"))
                        if (!$util.isInteger(message.key_type))
                            return "key_type: integer expected";
                    if (message.value_type != null && message.hasOwnProperty("value_type")) {
                        var error = $root.onnx.TypeProto.verify(message.value_type);
                        if (error)
                            return "value_type." + error;
                    }
                    return null;
                };
    
                Map.fromObject = function fromObject(object) {
                    if (object instanceof $root.onnx.TypeProto.Map)
                        return object;
                    var message = new $root.onnx.TypeProto.Map();
                    if (object.key_type != null)
                        message.key_type = object.key_type | 0;
                    if (object.value_type != null) {
                        if (typeof object.value_type !== "object")
                            throw TypeError(".onnx.TypeProto.Map.value_type: object expected");
                        message.value_type = $root.onnx.TypeProto.fromObject(object.value_type);
                    }
                    return message;
                };
    
                Map.toObject = function toObject(message, options) {
                    if (!options)
                        options = {};
                    var object = {};
                    if (options.defaults) {
                        object.key_type = 0;
                        object.value_type = null;
                    }
                    if (message.key_type != null && message.hasOwnProperty("key_type"))
                        object.key_type = message.key_type;
                    if (message.value_type != null && message.hasOwnProperty("value_type"))
                        object.value_type = $root.onnx.TypeProto.toObject(message.value_type, options);
                    return object;
                };
    
                Map.prototype.toJSON = function toJSON() {
                    return this.constructor.toObject(this, $protobuf.util.toJSONOptions);
                };
    
                return Map;
            })();
    
            TypeProto.Opaque = (function() {
    
                function Opaque(properties) {
                    if (properties)
                        for (var keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                            if (properties[keys[i]] != null)
                                this[keys[i]] = properties[keys[i]];
                }
    
                Opaque.prototype.domain = "";
                Opaque.prototype.name = "";
    
                Opaque.create = function create(properties) {
                    return new Opaque(properties);
                };
    
                Opaque.decode = function decode(reader, length) {
                    if (!(reader instanceof $Reader))
                        reader = $Reader.create(reader);
                    var end = length === undefined ? reader.len : reader.pos + length, message = new $root.onnx.TypeProto.Opaque();
                    while (reader.pos < end) {
                        var tag = reader.uint32();
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
                };
    
                Opaque.decodeText = function decodeText(reader, block) {
                    if (!(reader instanceof $TextReader))
                        reader = $TextReader.create(reader);
                    var message = new $root.onnx.TypeProto.Opaque();
                    reader.start(block);
                    while (!reader.end(block)) {
                        var tag = reader.tag();
                        switch (tag) {
                        case "domain":
                            message.domain = reader.string();
                            break;
                        case "name":
                            message.name = reader.string();
                            break;
                        default:
                            reader.handle(tag, message);
                            break;
                        }
                    }
                    return message;
                };
    
                Opaque.verify = function verify(message) {
                    if (typeof message !== "object" || message === null)
                        return "object expected";
                    if (message.domain != null && message.hasOwnProperty("domain"))
                        if (!$util.isString(message.domain))
                            return "domain: string expected";
                    if (message.name != null && message.hasOwnProperty("name"))
                        if (!$util.isString(message.name))
                            return "name: string expected";
                    return null;
                };
    
                Opaque.fromObject = function fromObject(object) {
                    if (object instanceof $root.onnx.TypeProto.Opaque)
                        return object;
                    var message = new $root.onnx.TypeProto.Opaque();
                    if (object.domain != null)
                        message.domain = String(object.domain);
                    if (object.name != null)
                        message.name = String(object.name);
                    return message;
                };
    
                Opaque.toObject = function toObject(message, options) {
                    if (!options)
                        options = {};
                    var object = {};
                    if (options.defaults) {
                        object.domain = "";
                        object.name = "";
                    }
                    if (message.domain != null && message.hasOwnProperty("domain"))
                        object.domain = message.domain;
                    if (message.name != null && message.hasOwnProperty("name"))
                        object.name = message.name;
                    return object;
                };
    
                Opaque.prototype.toJSON = function toJSON() {
                    return this.constructor.toObject(this, $protobuf.util.toJSONOptions);
                };
    
                return Opaque;
            })();
    
            TypeProto.SparseTensor = (function() {
    
                function SparseTensor(properties) {
                    if (properties)
                        for (var keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                            if (properties[keys[i]] != null)
                                this[keys[i]] = properties[keys[i]];
                }
    
                SparseTensor.prototype.elem_type = 0;
                SparseTensor.prototype.shape = null;
    
                SparseTensor.create = function create(properties) {
                    return new SparseTensor(properties);
                };
    
                SparseTensor.decode = function decode(reader, length) {
                    if (!(reader instanceof $Reader))
                        reader = $Reader.create(reader);
                    var end = length === undefined ? reader.len : reader.pos + length, message = new $root.onnx.TypeProto.SparseTensor();
                    while (reader.pos < end) {
                        var tag = reader.uint32();
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
                };
    
                SparseTensor.decodeText = function decodeText(reader, block) {
                    if (!(reader instanceof $TextReader))
                        reader = $TextReader.create(reader);
                    var message = new $root.onnx.TypeProto.SparseTensor();
                    reader.start(block);
                    while (!reader.end(block)) {
                        var tag = reader.tag();
                        switch (tag) {
                        case "elem_type":
                            message.elem_type = reader.int32();
                            break;
                        case "shape":
                            message.shape = $root.onnx.TensorShapeProto.decodeText(reader, true);
                            break;
                        default:
                            reader.handle(tag, message);
                            break;
                        }
                    }
                    return message;
                };
    
                SparseTensor.verify = function verify(message) {
                    if (typeof message !== "object" || message === null)
                        return "object expected";
                    if (message.elem_type != null && message.hasOwnProperty("elem_type"))
                        if (!$util.isInteger(message.elem_type))
                            return "elem_type: integer expected";
                    if (message.shape != null && message.hasOwnProperty("shape")) {
                        var error = $root.onnx.TensorShapeProto.verify(message.shape);
                        if (error)
                            return "shape." + error;
                    }
                    return null;
                };
    
                SparseTensor.fromObject = function fromObject(object) {
                    if (object instanceof $root.onnx.TypeProto.SparseTensor)
                        return object;
                    var message = new $root.onnx.TypeProto.SparseTensor();
                    if (object.elem_type != null)
                        message.elem_type = object.elem_type | 0;
                    if (object.shape != null) {
                        if (typeof object.shape !== "object")
                            throw TypeError(".onnx.TypeProto.SparseTensor.shape: object expected");
                        message.shape = $root.onnx.TensorShapeProto.fromObject(object.shape);
                    }
                    return message;
                };
    
                SparseTensor.toObject = function toObject(message, options) {
                    if (!options)
                        options = {};
                    var object = {};
                    if (options.defaults) {
                        object.elem_type = 0;
                        object.shape = null;
                    }
                    if (message.elem_type != null && message.hasOwnProperty("elem_type"))
                        object.elem_type = message.elem_type;
                    if (message.shape != null && message.hasOwnProperty("shape"))
                        object.shape = $root.onnx.TensorShapeProto.toObject(message.shape, options);
                    return object;
                };
    
                SparseTensor.prototype.toJSON = function toJSON() {
                    return this.constructor.toObject(this, $protobuf.util.toJSONOptions);
                };
    
                return SparseTensor;
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
    
            OperatorSetIdProto.decodeText = function decodeText(reader, block) {
                if (!(reader instanceof $TextReader))
                    reader = $TextReader.create(reader);
                var message = new $root.onnx.OperatorSetIdProto();
                reader.start(block);
                while (!reader.end(block)) {
                    var tag = reader.tag();
                    switch (tag) {
                    case "domain":
                        message.domain = reader.string();
                        break;
                    case "version":
                        message.version = reader.int64();
                        break;
                    default:
                        reader.handle(tag, message);
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
            FunctionProto.prototype.since_version = $util.Long ? $util.Long.fromBits(0,0,false) : 0;
            FunctionProto.prototype.status = 0;
            FunctionProto.prototype.input = $util.emptyArray;
            FunctionProto.prototype.output = $util.emptyArray;
            FunctionProto.prototype.attribute = $util.emptyArray;
            FunctionProto.prototype.node = $util.emptyArray;
            FunctionProto.prototype.doc_string = "";
    
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
                        message.since_version = reader.int64();
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
                        message.doc_string = reader.string();
                        break;
                    default:
                        reader.skipType(tag & 7);
                        break;
                    }
                }
                return message;
            };
    
            FunctionProto.decodeText = function decodeText(reader, block) {
                if (!(reader instanceof $TextReader))
                    reader = $TextReader.create(reader);
                var message = new $root.onnx.FunctionProto();
                reader.start(block);
                while (!reader.end(block)) {
                    var tag = reader.tag();
                    switch (tag) {
                    case "name":
                        message.name = reader.string();
                        break;
                    case "since_version":
                        message.since_version = reader.int64();
                        break;
                    case "status":
                        message.status = reader.enum($root.onnx.OperatorStatus);
                        break;
                    case "input":
                        if (!(message.input && message.input.length))
                            message.input = [];
                        message.input.push(reader.string());
                        break;
                    case "output":
                        if (!(message.output && message.output.length))
                            message.output = [];
                        message.output.push(reader.string());
                        break;
                    case "attribute":
                        if (!(message.attribute && message.attribute.length))
                            message.attribute = [];
                        message.attribute.push(reader.string());
                        break;
                    case "node":
                        if (!(message.node && message.node.length))
                            message.node = [];
                        message.node.push($root.onnx.NodeProto.decodeText(reader, true));
                        break;
                    case "doc_string":
                        message.doc_string = reader.string();
                        break;
                    default:
                        reader.handle(tag, message);
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
                if (message.since_version != null && message.hasOwnProperty("since_version"))
                    if (!$util.isInteger(message.since_version) && !(message.since_version && $util.isInteger(message.since_version.low) && $util.isInteger(message.since_version.high)))
                        return "since_version: integer|Long expected";
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
                if (message.doc_string != null && message.hasOwnProperty("doc_string"))
                    if (!$util.isString(message.doc_string))
                        return "doc_string: string expected";
                return null;
            };
    
            FunctionProto.fromObject = function fromObject(object) {
                if (object instanceof $root.onnx.FunctionProto)
                    return object;
                var message = new $root.onnx.FunctionProto();
                if (object.name != null)
                    message.name = String(object.name);
                if (object.since_version != null)
                    if ($util.Long)
                        (message.since_version = $util.Long.fromValue(object.since_version)).unsigned = false;
                    else if (typeof object.since_version === "string")
                        message.since_version = parseInt(object.since_version, 10);
                    else if (typeof object.since_version === "number")
                        message.since_version = object.since_version;
                    else if (typeof object.since_version === "object")
                        message.since_version = new $util.LongBits(object.since_version.low >>> 0, object.since_version.high >>> 0).toNumber();
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
                if (object.doc_string != null)
                    message.doc_string = String(object.doc_string);
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
                        object.since_version = options.longs === String ? long.toString() : options.longs === Number ? long.toNumber() : long;
                    } else
                        object.since_version = options.longs === String ? "0" : 0;
                    object.status = options.enums === String ? "EXPERIMENTAL" : 0;
                    object.doc_string = "";
                }
                if (message.name != null && message.hasOwnProperty("name"))
                    object.name = message.name;
                if (message.since_version != null && message.hasOwnProperty("since_version"))
                    if (typeof message.since_version === "number")
                        object.since_version = options.longs === String ? String(message.since_version) : message.since_version;
                    else
                        object.since_version = options.longs === String ? $util.Long.prototype.toString.call(message.since_version) : options.longs === Number ? new $util.LongBits(message.since_version.low >>> 0, message.since_version.high >>> 0).toNumber() : message.since_version;
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
                if (message.doc_string != null && message.hasOwnProperty("doc_string"))
                    object.doc_string = message.doc_string;
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
    
            OperatorProto.prototype.op_type = "";
            OperatorProto.prototype.since_version = $util.Long ? $util.Long.fromBits(0,0,false) : 0;
            OperatorProto.prototype.status = 0;
            OperatorProto.prototype.doc_string = "";
    
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
            };
    
            OperatorProto.decodeText = function decodeText(reader, block) {
                if (!(reader instanceof $TextReader))
                    reader = $TextReader.create(reader);
                var message = new $root.onnx.OperatorProto();
                reader.start(block);
                while (!reader.end(block)) {
                    var tag = reader.tag();
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
                        reader.handle(tag, message);
                        break;
                    }
                }
                return message;
            };
    
            OperatorProto.verify = function verify(message) {
                if (typeof message !== "object" || message === null)
                    return "object expected";
                if (message.op_type != null && message.hasOwnProperty("op_type"))
                    if (!$util.isString(message.op_type))
                        return "op_type: string expected";
                if (message.since_version != null && message.hasOwnProperty("since_version"))
                    if (!$util.isInteger(message.since_version) && !(message.since_version && $util.isInteger(message.since_version.low) && $util.isInteger(message.since_version.high)))
                        return "since_version: integer|Long expected";
                if (message.status != null && message.hasOwnProperty("status"))
                    switch (message.status) {
                    default:
                        return "status: enum value expected";
                    case 0:
                    case 1:
                        break;
                    }
                if (message.doc_string != null && message.hasOwnProperty("doc_string"))
                    if (!$util.isString(message.doc_string))
                        return "doc_string: string expected";
                return null;
            };
    
            OperatorProto.fromObject = function fromObject(object) {
                if (object instanceof $root.onnx.OperatorProto)
                    return object;
                var message = new $root.onnx.OperatorProto();
                if (object.op_type != null)
                    message.op_type = String(object.op_type);
                if (object.since_version != null)
                    if ($util.Long)
                        (message.since_version = $util.Long.fromValue(object.since_version)).unsigned = false;
                    else if (typeof object.since_version === "string")
                        message.since_version = parseInt(object.since_version, 10);
                    else if (typeof object.since_version === "number")
                        message.since_version = object.since_version;
                    else if (typeof object.since_version === "object")
                        message.since_version = new $util.LongBits(object.since_version.low >>> 0, object.since_version.high >>> 0).toNumber();
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
                if (object.doc_string != null)
                    message.doc_string = String(object.doc_string);
                return message;
            };
    
            OperatorProto.toObject = function toObject(message, options) {
                if (!options)
                    options = {};
                var object = {};
                if (options.defaults) {
                    object.op_type = "";
                    if ($util.Long) {
                        var long = new $util.Long(0, 0, false);
                        object.since_version = options.longs === String ? long.toString() : options.longs === Number ? long.toNumber() : long;
                    } else
                        object.since_version = options.longs === String ? "0" : 0;
                    object.status = options.enums === String ? "EXPERIMENTAL" : 0;
                    object.doc_string = "";
                }
                if (message.op_type != null && message.hasOwnProperty("op_type"))
                    object.op_type = message.op_type;
                if (message.since_version != null && message.hasOwnProperty("since_version"))
                    if (typeof message.since_version === "number")
                        object.since_version = options.longs === String ? String(message.since_version) : message.since_version;
                    else
                        object.since_version = options.longs === String ? $util.Long.prototype.toString.call(message.since_version) : options.longs === Number ? new $util.LongBits(message.since_version.low >>> 0, message.since_version.high >>> 0).toNumber() : message.since_version;
                if (message.status != null && message.hasOwnProperty("status"))
                    object.status = options.enums === String ? $root.onnx.OperatorStatus[message.status] : message.status;
                if (message.doc_string != null && message.hasOwnProperty("doc_string"))
                    object.doc_string = message.doc_string;
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
            OperatorSetProto.prototype.ir_version = 0;
            OperatorSetProto.prototype.ir_version_prerelease = "";
            OperatorSetProto.prototype.ir_build_metadata = "";
            OperatorSetProto.prototype.domain = "";
            OperatorSetProto.prototype.opset_version = $util.Long ? $util.Long.fromBits(0,0,false) : 0;
            OperatorSetProto.prototype.doc_string = "";
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
                        message.ir_version = reader.int32();
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
    
            OperatorSetProto.decodeText = function decodeText(reader, block) {
                if (!(reader instanceof $TextReader))
                    reader = $TextReader.create(reader);
                var message = new $root.onnx.OperatorSetProto();
                reader.start(block);
                while (!reader.end(block)) {
                    var tag = reader.tag();
                    switch (tag) {
                    case "magic":
                        message.magic = reader.string();
                        break;
                    case "ir_version":
                        message.ir_version = reader.int32();
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
                        if (!(message.operator && message.operator.length))
                            message.operator = [];
                        message.operator.push($root.onnx.OperatorProto.decodeText(reader, true));
                        break;
                    case "functions":
                        if (!(message.functions && message.functions.length))
                            message.functions = [];
                        message.functions.push($root.onnx.FunctionProto.decodeText(reader, true));
                        break;
                    default:
                        reader.handle(tag, message);
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
                if (message.ir_version != null && message.hasOwnProperty("ir_version"))
                    if (!$util.isInteger(message.ir_version))
                        return "ir_version: integer expected";
                if (message.ir_version_prerelease != null && message.hasOwnProperty("ir_version_prerelease"))
                    if (!$util.isString(message.ir_version_prerelease))
                        return "ir_version_prerelease: string expected";
                if (message.ir_build_metadata != null && message.hasOwnProperty("ir_build_metadata"))
                    if (!$util.isString(message.ir_build_metadata))
                        return "ir_build_metadata: string expected";
                if (message.domain != null && message.hasOwnProperty("domain"))
                    if (!$util.isString(message.domain))
                        return "domain: string expected";
                if (message.opset_version != null && message.hasOwnProperty("opset_version"))
                    if (!$util.isInteger(message.opset_version) && !(message.opset_version && $util.isInteger(message.opset_version.low) && $util.isInteger(message.opset_version.high)))
                        return "opset_version: integer|Long expected";
                if (message.doc_string != null && message.hasOwnProperty("doc_string"))
                    if (!$util.isString(message.doc_string))
                        return "doc_string: string expected";
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
                if (object.ir_version != null)
                    message.ir_version = object.ir_version | 0;
                if (object.ir_version_prerelease != null)
                    message.ir_version_prerelease = String(object.ir_version_prerelease);
                if (object.ir_build_metadata != null)
                    message.ir_build_metadata = String(object.ir_build_metadata);
                if (object.domain != null)
                    message.domain = String(object.domain);
                if (object.opset_version != null)
                    if ($util.Long)
                        (message.opset_version = $util.Long.fromValue(object.opset_version)).unsigned = false;
                    else if (typeof object.opset_version === "string")
                        message.opset_version = parseInt(object.opset_version, 10);
                    else if (typeof object.opset_version === "number")
                        message.opset_version = object.opset_version;
                    else if (typeof object.opset_version === "object")
                        message.opset_version = new $util.LongBits(object.opset_version.low >>> 0, object.opset_version.high >>> 0).toNumber();
                if (object.doc_string != null)
                    message.doc_string = String(object.doc_string);
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
                    object.ir_version = 0;
                    object.ir_version_prerelease = "";
                    object.domain = "";
                    if ($util.Long) {
                        var long = new $util.Long(0, 0, false);
                        object.opset_version = options.longs === String ? long.toString() : options.longs === Number ? long.toNumber() : long;
                    } else
                        object.opset_version = options.longs === String ? "0" : 0;
                    object.doc_string = "";
                    object.ir_build_metadata = "";
                }
                if (message.magic != null && message.hasOwnProperty("magic"))
                    object.magic = message.magic;
                if (message.ir_version != null && message.hasOwnProperty("ir_version"))
                    object.ir_version = message.ir_version;
                if (message.ir_version_prerelease != null && message.hasOwnProperty("ir_version_prerelease"))
                    object.ir_version_prerelease = message.ir_version_prerelease;
                if (message.domain != null && message.hasOwnProperty("domain"))
                    object.domain = message.domain;
                if (message.opset_version != null && message.hasOwnProperty("opset_version"))
                    if (typeof message.opset_version === "number")
                        object.opset_version = options.longs === String ? String(message.opset_version) : message.opset_version;
                    else
                        object.opset_version = options.longs === String ? $util.Long.prototype.toString.call(message.opset_version) : options.longs === Number ? new $util.LongBits(message.opset_version.low >>> 0, message.opset_version.high >>> 0).toNumber() : message.opset_version;
                if (message.doc_string != null && message.hasOwnProperty("doc_string"))
                    object.doc_string = message.doc_string;
                if (message.ir_build_metadata != null && message.hasOwnProperty("ir_build_metadata"))
                    object.ir_build_metadata = message.ir_build_metadata;
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
