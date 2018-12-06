/*eslint-disable block-scoped-var, id-length, no-control-regex, no-magic-numbers, no-prototype-builtins, no-redeclare, no-shadow, no-var, sort-vars*/
(function($protobuf) {
    "use strict";

    var $Reader = $protobuf.Reader, $TextReader = $protobuf.TextReader, $util = $protobuf.util;
    
    var $root = $protobuf.roots.caffe || ($protobuf.roots.caffe = {});
    
    $root.caffe = (function() {
    
        var caffe = {};
    
        caffe.BlobShape = (function() {
    
            function BlobShape(properties) {
                this.dim = [];
                if (properties)
                    for (var keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                        if (properties[keys[i]] != null)
                            this[keys[i]] = properties[keys[i]];
            }
    
            BlobShape.prototype.dim = $util.emptyArray;
    
            BlobShape.create = function create(properties) {
                return new BlobShape(properties);
            };
    
            BlobShape.decode = function decode(reader, length) {
                if (!(reader instanceof $Reader))
                    reader = $Reader.create(reader);
                var end = length === undefined ? reader.len : reader.pos + length, message = new $root.caffe.BlobShape();
                while (reader.pos < end) {
                    var tag = reader.uint32();
                    switch (tag >>> 3) {
                    case 1:
                        if (!(message.dim && message.dim.length))
                            message.dim = [];
                        if ((tag & 7) === 2) {
                            var end2 = reader.uint32() + reader.pos;
                            while (reader.pos < end2)
                                message.dim.push(reader.int64());
                        } else
                            message.dim.push(reader.int64());
                        break;
                    default:
                        reader.skipType(tag & 7);
                        break;
                    }
                }
                return message;
            };
    
            BlobShape.decodeText = function decodeText(reader, block) {
                if (!(reader instanceof $TextReader))
                    reader = $TextReader.create(reader);
                var message = new $root.caffe.BlobShape();
                reader.start(block);
                while (!reader.end(block)) {
                    var tag = reader.tag();
                    switch (tag) {
                    case "dim":
                        if (!(message.dim && message.dim.length))
                            message.dim = [];
                        message.dim.push(reader.int64());
                        break;
                    default:
                        reader.handle(tag, message);
                        break;
                    }
                }
                return message;
            };
    
            BlobShape.verify = function verify(message) {
                if (typeof message !== "object" || message === null)
                    return "object expected";
                if (message.dim != null && message.hasOwnProperty("dim")) {
                    if (!Array.isArray(message.dim))
                        return "dim: array expected";
                    for (var i = 0; i < message.dim.length; ++i)
                        if (!$util.isInteger(message.dim[i]) && !(message.dim[i] && $util.isInteger(message.dim[i].low) && $util.isInteger(message.dim[i].high)))
                            return "dim: integer|Long[] expected";
                }
                return null;
            };
    
            BlobShape.fromObject = function fromObject(object) {
                if (object instanceof $root.caffe.BlobShape)
                    return object;
                var message = new $root.caffe.BlobShape();
                if (object.dim) {
                    if (!Array.isArray(object.dim))
                        throw TypeError(".caffe.BlobShape.dim: array expected");
                    message.dim = [];
                    for (var i = 0; i < object.dim.length; ++i)
                        if ($util.Long)
                            (message.dim[i] = $util.Long.fromValue(object.dim[i])).unsigned = false;
                        else if (typeof object.dim[i] === "string")
                            message.dim[i] = parseInt(object.dim[i], 10);
                        else if (typeof object.dim[i] === "number")
                            message.dim[i] = object.dim[i];
                        else if (typeof object.dim[i] === "object")
                            message.dim[i] = new $util.LongBits(object.dim[i].low >>> 0, object.dim[i].high >>> 0).toNumber();
                }
                return message;
            };
    
            BlobShape.toObject = function toObject(message, options) {
                if (!options)
                    options = {};
                var object = {};
                if (options.arrays || options.defaults)
                    object.dim = [];
                if (message.dim && message.dim.length) {
                    object.dim = [];
                    for (var j = 0; j < message.dim.length; ++j)
                        if (typeof message.dim[j] === "number")
                            object.dim[j] = options.longs === String ? String(message.dim[j]) : message.dim[j];
                        else
                            object.dim[j] = options.longs === String ? $util.Long.prototype.toString.call(message.dim[j]) : options.longs === Number ? new $util.LongBits(message.dim[j].low >>> 0, message.dim[j].high >>> 0).toNumber() : message.dim[j];
                }
                return object;
            };
    
            BlobShape.prototype.toJSON = function toJSON() {
                return this.constructor.toObject(this, $protobuf.util.toJSONOptions);
            };
    
            return BlobShape;
        })();
    
        caffe.BlobProto = (function() {
    
            function BlobProto(properties) {
                this.data = [];
                this.diff = [];
                this.double_data = [];
                this.double_diff = [];
                if (properties)
                    for (var keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                        if (properties[keys[i]] != null)
                            this[keys[i]] = properties[keys[i]];
            }
    
            BlobProto.prototype.shape = null;
            BlobProto.prototype.data = $util.emptyArray;
            BlobProto.prototype.diff = $util.emptyArray;
            BlobProto.prototype.double_data = $util.emptyArray;
            BlobProto.prototype.double_diff = $util.emptyArray;
            BlobProto.prototype.num = 0;
            BlobProto.prototype.channels = 0;
            BlobProto.prototype.height = 0;
            BlobProto.prototype.width = 0;
    
            BlobProto.create = function create(properties) {
                return new BlobProto(properties);
            };
    
            BlobProto.decode = function decode(reader, length) {
                if (!(reader instanceof $Reader))
                    reader = $Reader.create(reader);
                var end = length === undefined ? reader.len : reader.pos + length, message = new $root.caffe.BlobProto();
                while (reader.pos < end) {
                    var tag = reader.uint32();
                    switch (tag >>> 3) {
                    case 7:
                        message.shape = $root.caffe.BlobShape.decode(reader, reader.uint32());
                        break;
                    case 5:
                        if (!(message.data && message.data.length))
                            message.data = [];
                        if ((tag & 7) === 2) {
                            var end2 = reader.uint32() + reader.pos;
                            if (message.data.length == 0 && (end2 - reader.pos) > 1048576) {
                                var dataLength = end2 - reader.pos;
                                var dataView = new DataView(reader.buf.buffer, reader.buf.byteOffset + reader.pos, dataLength);
                                dataLength = dataLength >>> 2;
                                var data = new Float32Array(dataLength);
                                for (var i = 0; i < dataLength; i++) {
                                    data[i] = dataView.getFloat32(i << 2, true);
                                }
                                message.data = data;
                                reader.pos = end2;
                            }
                            else {
                                while (reader.pos < end2)
                                    message.data.push(reader.float());
                            }
                        } else
                            message.data.push(reader.float());
                        break;
                    case 6:
                        if (!(message.diff && message.diff.length))
                            message.diff = [];
                        if ((tag & 7) === 2) {
                            var end2 = reader.uint32() + reader.pos;
                            while (reader.pos < end2)
                                message.diff.push(reader.float());
                        } else
                            message.diff.push(reader.float());
                        break;
                    case 8:
                        if (!(message.double_data && message.double_data.length))
                            message.double_data = [];
                        if ((tag & 7) === 2) {
                            var end2 = reader.uint32() + reader.pos;
                            while (reader.pos < end2)
                                message.double_data.push(reader.double());
                        } else
                            message.double_data.push(reader.double());
                        break;
                    case 9:
                        if (!(message.double_diff && message.double_diff.length))
                            message.double_diff = [];
                        if ((tag & 7) === 2) {
                            var end2 = reader.uint32() + reader.pos;
                            while (reader.pos < end2)
                                message.double_diff.push(reader.double());
                        } else
                            message.double_diff.push(reader.double());
                        break;
                    case 1:
                        message.num = reader.int32();
                        break;
                    case 2:
                        message.channels = reader.int32();
                        break;
                    case 3:
                        message.height = reader.int32();
                        break;
                    case 4:
                        message.width = reader.int32();
                        break;
                    default:
                        reader.skipType(tag & 7);
                        break;
                    }
                }
                return message;
            };
    
            BlobProto.decodeText = function decodeText(reader, block) {
                if (!(reader instanceof $TextReader))
                    reader = $TextReader.create(reader);
                var message = new $root.caffe.BlobProto();
                reader.start(block);
                while (!reader.end(block)) {
                    var tag = reader.tag();
                    switch (tag) {
                    case "shape":
                        message.shape = $root.caffe.BlobShape.decodeText(reader, true);
                        break;
                    case "data":
                        if (!(message.data && message.data.length))
                            message.data = [];
                        message.data.push(reader.float());
                        break;
                    case "diff":
                        if (!(message.diff && message.diff.length))
                            message.diff = [];
                        message.diff.push(reader.float());
                        break;
                    case "double_data":
                        if (!(message.double_data && message.double_data.length))
                            message.double_data = [];
                        message.double_data.push(reader.double());
                        break;
                    case "double_diff":
                        if (!(message.double_diff && message.double_diff.length))
                            message.double_diff = [];
                        message.double_diff.push(reader.double());
                        break;
                    case "num":
                        message.num = reader.int32();
                        break;
                    case "channels":
                        message.channels = reader.int32();
                        break;
                    case "height":
                        message.height = reader.int32();
                        break;
                    case "width":
                        message.width = reader.int32();
                        break;
                    default:
                        reader.handle(tag, message);
                        break;
                    }
                }
                return message;
            };
    
            BlobProto.verify = function verify(message) {
                if (typeof message !== "object" || message === null)
                    return "object expected";
                if (message.shape != null && message.hasOwnProperty("shape")) {
                    var error = $root.caffe.BlobShape.verify(message.shape);
                    if (error)
                        return "shape." + error;
                }
                if (message.data != null && message.hasOwnProperty("data")) {
                    if (!Array.isArray(message.data))
                        return "data: array expected";
                    for (var i = 0; i < message.data.length; ++i)
                        if (typeof message.data[i] !== "number")
                            return "data: number[] expected";
                }
                if (message.diff != null && message.hasOwnProperty("diff")) {
                    if (!Array.isArray(message.diff))
                        return "diff: array expected";
                    for (var i = 0; i < message.diff.length; ++i)
                        if (typeof message.diff[i] !== "number")
                            return "diff: number[] expected";
                }
                if (message.double_data != null && message.hasOwnProperty("double_data")) {
                    if (!Array.isArray(message.double_data))
                        return "double_data: array expected";
                    for (var i = 0; i < message.double_data.length; ++i)
                        if (typeof message.double_data[i] !== "number")
                            return "double_data: number[] expected";
                }
                if (message.double_diff != null && message.hasOwnProperty("double_diff")) {
                    if (!Array.isArray(message.double_diff))
                        return "double_diff: array expected";
                    for (var i = 0; i < message.double_diff.length; ++i)
                        if (typeof message.double_diff[i] !== "number")
                            return "double_diff: number[] expected";
                }
                if (message.num != null && message.hasOwnProperty("num"))
                    if (!$util.isInteger(message.num))
                        return "num: integer expected";
                if (message.channels != null && message.hasOwnProperty("channels"))
                    if (!$util.isInteger(message.channels))
                        return "channels: integer expected";
                if (message.height != null && message.hasOwnProperty("height"))
                    if (!$util.isInteger(message.height))
                        return "height: integer expected";
                if (message.width != null && message.hasOwnProperty("width"))
                    if (!$util.isInteger(message.width))
                        return "width: integer expected";
                return null;
            };
    
            BlobProto.fromObject = function fromObject(object) {
                if (object instanceof $root.caffe.BlobProto)
                    return object;
                var message = new $root.caffe.BlobProto();
                if (object.shape != null) {
                    if (typeof object.shape !== "object")
                        throw TypeError(".caffe.BlobProto.shape: object expected");
                    message.shape = $root.caffe.BlobShape.fromObject(object.shape);
                }
                if (object.data) {
                    if (!Array.isArray(object.data))
                        throw TypeError(".caffe.BlobProto.data: array expected");
                    message.data = [];
                    for (var i = 0; i < object.data.length; ++i)
                        message.data[i] = Number(object.data[i]);
                }
                if (object.diff) {
                    if (!Array.isArray(object.diff))
                        throw TypeError(".caffe.BlobProto.diff: array expected");
                    message.diff = [];
                    for (var i = 0; i < object.diff.length; ++i)
                        message.diff[i] = Number(object.diff[i]);
                }
                if (object.double_data) {
                    if (!Array.isArray(object.double_data))
                        throw TypeError(".caffe.BlobProto.double_data: array expected");
                    message.double_data = [];
                    for (var i = 0; i < object.double_data.length; ++i)
                        message.double_data[i] = Number(object.double_data[i]);
                }
                if (object.double_diff) {
                    if (!Array.isArray(object.double_diff))
                        throw TypeError(".caffe.BlobProto.double_diff: array expected");
                    message.double_diff = [];
                    for (var i = 0; i < object.double_diff.length; ++i)
                        message.double_diff[i] = Number(object.double_diff[i]);
                }
                if (object.num != null)
                    message.num = object.num | 0;
                if (object.channels != null)
                    message.channels = object.channels | 0;
                if (object.height != null)
                    message.height = object.height | 0;
                if (object.width != null)
                    message.width = object.width | 0;
                return message;
            };
    
            BlobProto.toObject = function toObject(message, options) {
                if (!options)
                    options = {};
                var object = {};
                if (options.arrays || options.defaults) {
                    object.data = [];
                    object.diff = [];
                    object.double_data = [];
                    object.double_diff = [];
                }
                if (options.defaults) {
                    object.num = 0;
                    object.channels = 0;
                    object.height = 0;
                    object.width = 0;
                    object.shape = null;
                }
                if (message.num != null && message.hasOwnProperty("num"))
                    object.num = message.num;
                if (message.channels != null && message.hasOwnProperty("channels"))
                    object.channels = message.channels;
                if (message.height != null && message.hasOwnProperty("height"))
                    object.height = message.height;
                if (message.width != null && message.hasOwnProperty("width"))
                    object.width = message.width;
                if (message.data && message.data.length) {
                    object.data = [];
                    for (var j = 0; j < message.data.length; ++j)
                        object.data[j] = options.json && !isFinite(message.data[j]) ? String(message.data[j]) : message.data[j];
                }
                if (message.diff && message.diff.length) {
                    object.diff = [];
                    for (var j = 0; j < message.diff.length; ++j)
                        object.diff[j] = options.json && !isFinite(message.diff[j]) ? String(message.diff[j]) : message.diff[j];
                }
                if (message.shape != null && message.hasOwnProperty("shape"))
                    object.shape = $root.caffe.BlobShape.toObject(message.shape, options);
                if (message.double_data && message.double_data.length) {
                    object.double_data = [];
                    for (var j = 0; j < message.double_data.length; ++j)
                        object.double_data[j] = options.json && !isFinite(message.double_data[j]) ? String(message.double_data[j]) : message.double_data[j];
                }
                if (message.double_diff && message.double_diff.length) {
                    object.double_diff = [];
                    for (var j = 0; j < message.double_diff.length; ++j)
                        object.double_diff[j] = options.json && !isFinite(message.double_diff[j]) ? String(message.double_diff[j]) : message.double_diff[j];
                }
                return object;
            };
    
            BlobProto.prototype.toJSON = function toJSON() {
                return this.constructor.toObject(this, $protobuf.util.toJSONOptions);
            };
    
            return BlobProto;
        })();
    
        caffe.BlobProtoVector = (function() {
    
            function BlobProtoVector(properties) {
                this.blobs = [];
                if (properties)
                    for (var keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                        if (properties[keys[i]] != null)
                            this[keys[i]] = properties[keys[i]];
            }
    
            BlobProtoVector.prototype.blobs = $util.emptyArray;
    
            BlobProtoVector.create = function create(properties) {
                return new BlobProtoVector(properties);
            };
    
            BlobProtoVector.decode = function decode(reader, length) {
                if (!(reader instanceof $Reader))
                    reader = $Reader.create(reader);
                var end = length === undefined ? reader.len : reader.pos + length, message = new $root.caffe.BlobProtoVector();
                while (reader.pos < end) {
                    var tag = reader.uint32();
                    switch (tag >>> 3) {
                    case 1:
                        if (!(message.blobs && message.blobs.length))
                            message.blobs = [];
                        message.blobs.push($root.caffe.BlobProto.decode(reader, reader.uint32()));
                        break;
                    default:
                        reader.skipType(tag & 7);
                        break;
                    }
                }
                return message;
            };
    
            BlobProtoVector.decodeText = function decodeText(reader, block) {
                if (!(reader instanceof $TextReader))
                    reader = $TextReader.create(reader);
                var message = new $root.caffe.BlobProtoVector();
                reader.start(block);
                while (!reader.end(block)) {
                    var tag = reader.tag();
                    switch (tag) {
                    case "blobs":
                        if (!(message.blobs && message.blobs.length))
                            message.blobs = [];
                        message.blobs.push($root.caffe.BlobProto.decodeText(reader, true));
                        break;
                    default:
                        reader.handle(tag, message);
                        break;
                    }
                }
                return message;
            };
    
            BlobProtoVector.verify = function verify(message) {
                if (typeof message !== "object" || message === null)
                    return "object expected";
                if (message.blobs != null && message.hasOwnProperty("blobs")) {
                    if (!Array.isArray(message.blobs))
                        return "blobs: array expected";
                    for (var i = 0; i < message.blobs.length; ++i) {
                        var error = $root.caffe.BlobProto.verify(message.blobs[i]);
                        if (error)
                            return "blobs." + error;
                    }
                }
                return null;
            };
    
            BlobProtoVector.fromObject = function fromObject(object) {
                if (object instanceof $root.caffe.BlobProtoVector)
                    return object;
                var message = new $root.caffe.BlobProtoVector();
                if (object.blobs) {
                    if (!Array.isArray(object.blobs))
                        throw TypeError(".caffe.BlobProtoVector.blobs: array expected");
                    message.blobs = [];
                    for (var i = 0; i < object.blobs.length; ++i) {
                        if (typeof object.blobs[i] !== "object")
                            throw TypeError(".caffe.BlobProtoVector.blobs: object expected");
                        message.blobs[i] = $root.caffe.BlobProto.fromObject(object.blobs[i]);
                    }
                }
                return message;
            };
    
            BlobProtoVector.toObject = function toObject(message, options) {
                if (!options)
                    options = {};
                var object = {};
                if (options.arrays || options.defaults)
                    object.blobs = [];
                if (message.blobs && message.blobs.length) {
                    object.blobs = [];
                    for (var j = 0; j < message.blobs.length; ++j)
                        object.blobs[j] = $root.caffe.BlobProto.toObject(message.blobs[j], options);
                }
                return object;
            };
    
            BlobProtoVector.prototype.toJSON = function toJSON() {
                return this.constructor.toObject(this, $protobuf.util.toJSONOptions);
            };
    
            return BlobProtoVector;
        })();
    
        caffe.Datum = (function() {
    
            function Datum(properties) {
                this.float_data = [];
                if (properties)
                    for (var keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                        if (properties[keys[i]] != null)
                            this[keys[i]] = properties[keys[i]];
            }
    
            Datum.prototype.channels = 0;
            Datum.prototype.height = 0;
            Datum.prototype.width = 0;
            Datum.prototype.data = $util.newBuffer([]);
            Datum.prototype.label = 0;
            Datum.prototype.float_data = $util.emptyArray;
            Datum.prototype.encoded = false;
    
            Datum.create = function create(properties) {
                return new Datum(properties);
            };
    
            Datum.decode = function decode(reader, length) {
                if (!(reader instanceof $Reader))
                    reader = $Reader.create(reader);
                var end = length === undefined ? reader.len : reader.pos + length, message = new $root.caffe.Datum();
                while (reader.pos < end) {
                    var tag = reader.uint32();
                    switch (tag >>> 3) {
                    case 1:
                        message.channels = reader.int32();
                        break;
                    case 2:
                        message.height = reader.int32();
                        break;
                    case 3:
                        message.width = reader.int32();
                        break;
                    case 4:
                        message.data = reader.bytes();
                        break;
                    case 5:
                        message.label = reader.int32();
                        break;
                    case 6:
                        if (!(message.float_data && message.float_data.length))
                            message.float_data = [];
                        if ((tag & 7) === 2) {
                            var end2 = reader.uint32() + reader.pos;
                            while (reader.pos < end2)
                                message.float_data.push(reader.float());
                        } else
                            message.float_data.push(reader.float());
                        break;
                    case 7:
                        message.encoded = reader.bool();
                        break;
                    default:
                        reader.skipType(tag & 7);
                        break;
                    }
                }
                return message;
            };
    
            Datum.decodeText = function decodeText(reader, block) {
                if (!(reader instanceof $TextReader))
                    reader = $TextReader.create(reader);
                var message = new $root.caffe.Datum();
                reader.start(block);
                while (!reader.end(block)) {
                    var tag = reader.tag();
                    switch (tag) {
                    case "channels":
                        message.channels = reader.int32();
                        break;
                    case "height":
                        message.height = reader.int32();
                        break;
                    case "width":
                        message.width = reader.int32();
                        break;
                    case "data":
                        message.data = reader.bytes();
                        break;
                    case "label":
                        message.label = reader.int32();
                        break;
                    case "float_data":
                        if (!(message.float_data && message.float_data.length))
                            message.float_data = [];
                        message.float_data.push(reader.float());
                        break;
                    case "encoded":
                        message.encoded = reader.bool();
                        break;
                    default:
                        reader.handle(tag, message);
                        break;
                    }
                }
                return message;
            };
    
            Datum.verify = function verify(message) {
                if (typeof message !== "object" || message === null)
                    return "object expected";
                if (message.channels != null && message.hasOwnProperty("channels"))
                    if (!$util.isInteger(message.channels))
                        return "channels: integer expected";
                if (message.height != null && message.hasOwnProperty("height"))
                    if (!$util.isInteger(message.height))
                        return "height: integer expected";
                if (message.width != null && message.hasOwnProperty("width"))
                    if (!$util.isInteger(message.width))
                        return "width: integer expected";
                if (message.data != null && message.hasOwnProperty("data"))
                    if (!(message.data && typeof message.data.length === "number" || $util.isString(message.data)))
                        return "data: buffer expected";
                if (message.label != null && message.hasOwnProperty("label"))
                    if (!$util.isInteger(message.label))
                        return "label: integer expected";
                if (message.float_data != null && message.hasOwnProperty("float_data")) {
                    if (!Array.isArray(message.float_data))
                        return "float_data: array expected";
                    for (var i = 0; i < message.float_data.length; ++i)
                        if (typeof message.float_data[i] !== "number")
                            return "float_data: number[] expected";
                }
                if (message.encoded != null && message.hasOwnProperty("encoded"))
                    if (typeof message.encoded !== "boolean")
                        return "encoded: boolean expected";
                return null;
            };
    
            Datum.fromObject = function fromObject(object) {
                if (object instanceof $root.caffe.Datum)
                    return object;
                var message = new $root.caffe.Datum();
                if (object.channels != null)
                    message.channels = object.channels | 0;
                if (object.height != null)
                    message.height = object.height | 0;
                if (object.width != null)
                    message.width = object.width | 0;
                if (object.data != null)
                    if (typeof object.data === "string")
                        $util.base64.decode(object.data, message.data = $util.newBuffer($util.base64.length(object.data)), 0);
                    else if (object.data.length)
                        message.data = object.data;
                if (object.label != null)
                    message.label = object.label | 0;
                if (object.float_data) {
                    if (!Array.isArray(object.float_data))
                        throw TypeError(".caffe.Datum.float_data: array expected");
                    message.float_data = [];
                    for (var i = 0; i < object.float_data.length; ++i)
                        message.float_data[i] = Number(object.float_data[i]);
                }
                if (object.encoded != null)
                    message.encoded = Boolean(object.encoded);
                return message;
            };
    
            Datum.toObject = function toObject(message, options) {
                if (!options)
                    options = {};
                var object = {};
                if (options.arrays || options.defaults)
                    object.float_data = [];
                if (options.defaults) {
                    object.channels = 0;
                    object.height = 0;
                    object.width = 0;
                    if (options.bytes === String)
                        object.data = "";
                    else {
                        object.data = [];
                        if (options.bytes !== Array)
                            object.data = $util.newBuffer(object.data);
                    }
                    object.label = 0;
                    object.encoded = false;
                }
                if (message.channels != null && message.hasOwnProperty("channels"))
                    object.channels = message.channels;
                if (message.height != null && message.hasOwnProperty("height"))
                    object.height = message.height;
                if (message.width != null && message.hasOwnProperty("width"))
                    object.width = message.width;
                if (message.data != null && message.hasOwnProperty("data"))
                    object.data = options.bytes === String ? $util.base64.encode(message.data, 0, message.data.length) : options.bytes === Array ? Array.prototype.slice.call(message.data) : message.data;
                if (message.label != null && message.hasOwnProperty("label"))
                    object.label = message.label;
                if (message.float_data && message.float_data.length) {
                    object.float_data = [];
                    for (var j = 0; j < message.float_data.length; ++j)
                        object.float_data[j] = options.json && !isFinite(message.float_data[j]) ? String(message.float_data[j]) : message.float_data[j];
                }
                if (message.encoded != null && message.hasOwnProperty("encoded"))
                    object.encoded = message.encoded;
                return object;
            };
    
            Datum.prototype.toJSON = function toJSON() {
                return this.constructor.toObject(this, $protobuf.util.toJSONOptions);
            };
    
            return Datum;
        })();
    
        caffe.FillerParameter = (function() {
    
            function FillerParameter(properties) {
                if (properties)
                    for (var keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                        if (properties[keys[i]] != null)
                            this[keys[i]] = properties[keys[i]];
            }
    
            FillerParameter.prototype.type = "constant";
            FillerParameter.prototype.value = 0;
            FillerParameter.prototype.min = 0;
            FillerParameter.prototype.max = 1;
            FillerParameter.prototype.mean = 0;
            FillerParameter.prototype.std = 1;
            FillerParameter.prototype.sparse = -1;
            FillerParameter.prototype.variance_norm = 0;
    
            FillerParameter.create = function create(properties) {
                return new FillerParameter(properties);
            };
    
            FillerParameter.decode = function decode(reader, length) {
                if (!(reader instanceof $Reader))
                    reader = $Reader.create(reader);
                var end = length === undefined ? reader.len : reader.pos + length, message = new $root.caffe.FillerParameter();
                while (reader.pos < end) {
                    var tag = reader.uint32();
                    switch (tag >>> 3) {
                    case 1:
                        message.type = reader.string();
                        break;
                    case 2:
                        message.value = reader.float();
                        break;
                    case 3:
                        message.min = reader.float();
                        break;
                    case 4:
                        message.max = reader.float();
                        break;
                    case 5:
                        message.mean = reader.float();
                        break;
                    case 6:
                        message.std = reader.float();
                        break;
                    case 7:
                        message.sparse = reader.int32();
                        break;
                    case 8:
                        message.variance_norm = reader.int32();
                        break;
                    default:
                        reader.skipType(tag & 7);
                        break;
                    }
                }
                return message;
            };
    
            FillerParameter.decodeText = function decodeText(reader, block) {
                if (!(reader instanceof $TextReader))
                    reader = $TextReader.create(reader);
                var message = new $root.caffe.FillerParameter();
                reader.start(block);
                while (!reader.end(block)) {
                    var tag = reader.tag();
                    switch (tag) {
                    case "type":
                        message.type = reader.string();
                        break;
                    case "value":
                        message.value = reader.float();
                        break;
                    case "min":
                        message.min = reader.float();
                        break;
                    case "max":
                        message.max = reader.float();
                        break;
                    case "mean":
                        message.mean = reader.float();
                        break;
                    case "std":
                        message.std = reader.float();
                        break;
                    case "sparse":
                        message.sparse = reader.int32();
                        break;
                    case "variance_norm":
                        message.variance_norm = reader.enum($root.caffe.FillerParameter.VarianceNorm);
                        break;
                    default:
                        reader.handle(tag, message);
                        break;
                    }
                }
                return message;
            };
    
            FillerParameter.verify = function verify(message) {
                if (typeof message !== "object" || message === null)
                    return "object expected";
                if (message.type != null && message.hasOwnProperty("type"))
                    if (!$util.isString(message.type))
                        return "type: string expected";
                if (message.value != null && message.hasOwnProperty("value"))
                    if (typeof message.value !== "number")
                        return "value: number expected";
                if (message.min != null && message.hasOwnProperty("min"))
                    if (typeof message.min !== "number")
                        return "min: number expected";
                if (message.max != null && message.hasOwnProperty("max"))
                    if (typeof message.max !== "number")
                        return "max: number expected";
                if (message.mean != null && message.hasOwnProperty("mean"))
                    if (typeof message.mean !== "number")
                        return "mean: number expected";
                if (message.std != null && message.hasOwnProperty("std"))
                    if (typeof message.std !== "number")
                        return "std: number expected";
                if (message.sparse != null && message.hasOwnProperty("sparse"))
                    if (!$util.isInteger(message.sparse))
                        return "sparse: integer expected";
                if (message.variance_norm != null && message.hasOwnProperty("variance_norm"))
                    switch (message.variance_norm) {
                    default:
                        return "variance_norm: enum value expected";
                    case 0:
                    case 1:
                    case 2:
                        break;
                    }
                return null;
            };
    
            FillerParameter.fromObject = function fromObject(object) {
                if (object instanceof $root.caffe.FillerParameter)
                    return object;
                var message = new $root.caffe.FillerParameter();
                if (object.type != null)
                    message.type = String(object.type);
                if (object.value != null)
                    message.value = Number(object.value);
                if (object.min != null)
                    message.min = Number(object.min);
                if (object.max != null)
                    message.max = Number(object.max);
                if (object.mean != null)
                    message.mean = Number(object.mean);
                if (object.std != null)
                    message.std = Number(object.std);
                if (object.sparse != null)
                    message.sparse = object.sparse | 0;
                switch (object.variance_norm) {
                case "FAN_IN":
                case 0:
                    message.variance_norm = 0;
                    break;
                case "FAN_OUT":
                case 1:
                    message.variance_norm = 1;
                    break;
                case "AVERAGE":
                case 2:
                    message.variance_norm = 2;
                    break;
                }
                return message;
            };
    
            FillerParameter.toObject = function toObject(message, options) {
                if (!options)
                    options = {};
                var object = {};
                if (options.defaults) {
                    object.type = "constant";
                    object.value = 0;
                    object.min = 0;
                    object.max = 1;
                    object.mean = 0;
                    object.std = 1;
                    object.sparse = -1;
                    object.variance_norm = options.enums === String ? "FAN_IN" : 0;
                }
                if (message.type != null && message.hasOwnProperty("type"))
                    object.type = message.type;
                if (message.value != null && message.hasOwnProperty("value"))
                    object.value = options.json && !isFinite(message.value) ? String(message.value) : message.value;
                if (message.min != null && message.hasOwnProperty("min"))
                    object.min = options.json && !isFinite(message.min) ? String(message.min) : message.min;
                if (message.max != null && message.hasOwnProperty("max"))
                    object.max = options.json && !isFinite(message.max) ? String(message.max) : message.max;
                if (message.mean != null && message.hasOwnProperty("mean"))
                    object.mean = options.json && !isFinite(message.mean) ? String(message.mean) : message.mean;
                if (message.std != null && message.hasOwnProperty("std"))
                    object.std = options.json && !isFinite(message.std) ? String(message.std) : message.std;
                if (message.sparse != null && message.hasOwnProperty("sparse"))
                    object.sparse = message.sparse;
                if (message.variance_norm != null && message.hasOwnProperty("variance_norm"))
                    object.variance_norm = options.enums === String ? $root.caffe.FillerParameter.VarianceNorm[message.variance_norm] : message.variance_norm;
                return object;
            };
    
            FillerParameter.prototype.toJSON = function toJSON() {
                return this.constructor.toObject(this, $protobuf.util.toJSONOptions);
            };
    
            FillerParameter.VarianceNorm = (function() {
                var valuesById = {}, values = Object.create(valuesById);
                values[valuesById[0] = "FAN_IN"] = 0;
                values[valuesById[1] = "FAN_OUT"] = 1;
                values[valuesById[2] = "AVERAGE"] = 2;
                return values;
            })();
    
            return FillerParameter;
        })();
    
        caffe.NetParameter = (function() {
    
            function NetParameter(properties) {
                this.input = [];
                this.input_shape = [];
                this.input_dim = [];
                this.layer = [];
                this.layers = [];
                if (properties)
                    for (var keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                        if (properties[keys[i]] != null)
                            this[keys[i]] = properties[keys[i]];
            }
    
            NetParameter.prototype.name = "";
            NetParameter.prototype.input = $util.emptyArray;
            NetParameter.prototype.input_shape = $util.emptyArray;
            NetParameter.prototype.input_dim = $util.emptyArray;
            NetParameter.prototype.force_backward = false;
            NetParameter.prototype.state = null;
            NetParameter.prototype.debug_info = false;
            NetParameter.prototype.layer = $util.emptyArray;
            NetParameter.prototype.layers = $util.emptyArray;
    
            NetParameter.create = function create(properties) {
                return new NetParameter(properties);
            };
    
            NetParameter.decode = function decode(reader, length) {
                if (!(reader instanceof $Reader))
                    reader = $Reader.create(reader);
                var end = length === undefined ? reader.len : reader.pos + length, message = new $root.caffe.NetParameter();
                while (reader.pos < end) {
                    var tag = reader.uint32();
                    switch (tag >>> 3) {
                    case 1:
                        message.name = reader.string();
                        break;
                    case 3:
                        if (!(message.input && message.input.length))
                            message.input = [];
                        message.input.push(reader.string());
                        break;
                    case 8:
                        if (!(message.input_shape && message.input_shape.length))
                            message.input_shape = [];
                        message.input_shape.push($root.caffe.BlobShape.decode(reader, reader.uint32()));
                        break;
                    case 4:
                        if (!(message.input_dim && message.input_dim.length))
                            message.input_dim = [];
                        if ((tag & 7) === 2) {
                            var end2 = reader.uint32() + reader.pos;
                            while (reader.pos < end2)
                                message.input_dim.push(reader.int32());
                        } else
                            message.input_dim.push(reader.int32());
                        break;
                    case 5:
                        message.force_backward = reader.bool();
                        break;
                    case 6:
                        message.state = $root.caffe.NetState.decode(reader, reader.uint32());
                        break;
                    case 7:
                        message.debug_info = reader.bool();
                        break;
                    case 100:
                        if (!(message.layer && message.layer.length))
                            message.layer = [];
                        message.layer.push($root.caffe.LayerParameter.decode(reader, reader.uint32()));
                        break;
                    case 2:
                        if (!(message.layers && message.layers.length))
                            message.layers = [];
                        message.layers.push($root.caffe.V1LayerParameter.decode(reader, reader.uint32()));
                        break;
                    default:
                        reader.skipType(tag & 7);
                        break;
                    }
                }
                return message;
            };
    
            NetParameter.decodeText = function decodeText(reader, block) {
                if (!(reader instanceof $TextReader))
                    reader = $TextReader.create(reader);
                var message = new $root.caffe.NetParameter();
                reader.start(block);
                while (!reader.end(block)) {
                    var tag = reader.tag();
                    switch (tag) {
                    case "name":
                        message.name = reader.string();
                        break;
                    case "input":
                        if (!(message.input && message.input.length))
                            message.input = [];
                        message.input.push(reader.string());
                        break;
                    case "input_shape":
                        if (!(message.input_shape && message.input_shape.length))
                            message.input_shape = [];
                        message.input_shape.push($root.caffe.BlobShape.decodeText(reader, true));
                        break;
                    case "input_dim":
                        if (!(message.input_dim && message.input_dim.length))
                            message.input_dim = [];
                        message.input_dim.push(reader.int32());
                        break;
                    case "force_backward":
                        message.force_backward = reader.bool();
                        break;
                    case "state":
                        message.state = $root.caffe.NetState.decodeText(reader, true);
                        break;
                    case "debug_info":
                        message.debug_info = reader.bool();
                        break;
                    case "layer":
                        if (!(message.layer && message.layer.length))
                            message.layer = [];
                        message.layer.push($root.caffe.LayerParameter.decodeText(reader, true));
                        break;
                    case "layers":
                        if (!(message.layers && message.layers.length))
                            message.layers = [];
                        message.layers.push($root.caffe.V1LayerParameter.decodeText(reader, true));
                        break;
                    default:
                        reader.handle(tag, message);
                        break;
                    }
                }
                return message;
            };
    
            NetParameter.verify = function verify(message) {
                if (typeof message !== "object" || message === null)
                    return "object expected";
                if (message.name != null && message.hasOwnProperty("name"))
                    if (!$util.isString(message.name))
                        return "name: string expected";
                if (message.input != null && message.hasOwnProperty("input")) {
                    if (!Array.isArray(message.input))
                        return "input: array expected";
                    for (var i = 0; i < message.input.length; ++i)
                        if (!$util.isString(message.input[i]))
                            return "input: string[] expected";
                }
                if (message.input_shape != null && message.hasOwnProperty("input_shape")) {
                    if (!Array.isArray(message.input_shape))
                        return "input_shape: array expected";
                    for (var i = 0; i < message.input_shape.length; ++i) {
                        var error = $root.caffe.BlobShape.verify(message.input_shape[i]);
                        if (error)
                            return "input_shape." + error;
                    }
                }
                if (message.input_dim != null && message.hasOwnProperty("input_dim")) {
                    if (!Array.isArray(message.input_dim))
                        return "input_dim: array expected";
                    for (var i = 0; i < message.input_dim.length; ++i)
                        if (!$util.isInteger(message.input_dim[i]))
                            return "input_dim: integer[] expected";
                }
                if (message.force_backward != null && message.hasOwnProperty("force_backward"))
                    if (typeof message.force_backward !== "boolean")
                        return "force_backward: boolean expected";
                if (message.state != null && message.hasOwnProperty("state")) {
                    var error = $root.caffe.NetState.verify(message.state);
                    if (error)
                        return "state." + error;
                }
                if (message.debug_info != null && message.hasOwnProperty("debug_info"))
                    if (typeof message.debug_info !== "boolean")
                        return "debug_info: boolean expected";
                if (message.layer != null && message.hasOwnProperty("layer")) {
                    if (!Array.isArray(message.layer))
                        return "layer: array expected";
                    for (var i = 0; i < message.layer.length; ++i) {
                        var error = $root.caffe.LayerParameter.verify(message.layer[i]);
                        if (error)
                            return "layer." + error;
                    }
                }
                if (message.layers != null && message.hasOwnProperty("layers")) {
                    if (!Array.isArray(message.layers))
                        return "layers: array expected";
                    for (var i = 0; i < message.layers.length; ++i) {
                        var error = $root.caffe.V1LayerParameter.verify(message.layers[i]);
                        if (error)
                            return "layers." + error;
                    }
                }
                return null;
            };
    
            NetParameter.fromObject = function fromObject(object) {
                if (object instanceof $root.caffe.NetParameter)
                    return object;
                var message = new $root.caffe.NetParameter();
                if (object.name != null)
                    message.name = String(object.name);
                if (object.input) {
                    if (!Array.isArray(object.input))
                        throw TypeError(".caffe.NetParameter.input: array expected");
                    message.input = [];
                    for (var i = 0; i < object.input.length; ++i)
                        message.input[i] = String(object.input[i]);
                }
                if (object.input_shape) {
                    if (!Array.isArray(object.input_shape))
                        throw TypeError(".caffe.NetParameter.input_shape: array expected");
                    message.input_shape = [];
                    for (var i = 0; i < object.input_shape.length; ++i) {
                        if (typeof object.input_shape[i] !== "object")
                            throw TypeError(".caffe.NetParameter.input_shape: object expected");
                        message.input_shape[i] = $root.caffe.BlobShape.fromObject(object.input_shape[i]);
                    }
                }
                if (object.input_dim) {
                    if (!Array.isArray(object.input_dim))
                        throw TypeError(".caffe.NetParameter.input_dim: array expected");
                    message.input_dim = [];
                    for (var i = 0; i < object.input_dim.length; ++i)
                        message.input_dim[i] = object.input_dim[i] | 0;
                }
                if (object.force_backward != null)
                    message.force_backward = Boolean(object.force_backward);
                if (object.state != null) {
                    if (typeof object.state !== "object")
                        throw TypeError(".caffe.NetParameter.state: object expected");
                    message.state = $root.caffe.NetState.fromObject(object.state);
                }
                if (object.debug_info != null)
                    message.debug_info = Boolean(object.debug_info);
                if (object.layer) {
                    if (!Array.isArray(object.layer))
                        throw TypeError(".caffe.NetParameter.layer: array expected");
                    message.layer = [];
                    for (var i = 0; i < object.layer.length; ++i) {
                        if (typeof object.layer[i] !== "object")
                            throw TypeError(".caffe.NetParameter.layer: object expected");
                        message.layer[i] = $root.caffe.LayerParameter.fromObject(object.layer[i]);
                    }
                }
                if (object.layers) {
                    if (!Array.isArray(object.layers))
                        throw TypeError(".caffe.NetParameter.layers: array expected");
                    message.layers = [];
                    for (var i = 0; i < object.layers.length; ++i) {
                        if (typeof object.layers[i] !== "object")
                            throw TypeError(".caffe.NetParameter.layers: object expected");
                        message.layers[i] = $root.caffe.V1LayerParameter.fromObject(object.layers[i]);
                    }
                }
                return message;
            };
    
            NetParameter.toObject = function toObject(message, options) {
                if (!options)
                    options = {};
                var object = {};
                if (options.arrays || options.defaults) {
                    object.layers = [];
                    object.input = [];
                    object.input_dim = [];
                    object.input_shape = [];
                    object.layer = [];
                }
                if (options.defaults) {
                    object.name = "";
                    object.force_backward = false;
                    object.state = null;
                    object.debug_info = false;
                }
                if (message.name != null && message.hasOwnProperty("name"))
                    object.name = message.name;
                if (message.layers && message.layers.length) {
                    object.layers = [];
                    for (var j = 0; j < message.layers.length; ++j)
                        object.layers[j] = $root.caffe.V1LayerParameter.toObject(message.layers[j], options);
                }
                if (message.input && message.input.length) {
                    object.input = [];
                    for (var j = 0; j < message.input.length; ++j)
                        object.input[j] = message.input[j];
                }
                if (message.input_dim && message.input_dim.length) {
                    object.input_dim = [];
                    for (var j = 0; j < message.input_dim.length; ++j)
                        object.input_dim[j] = message.input_dim[j];
                }
                if (message.force_backward != null && message.hasOwnProperty("force_backward"))
                    object.force_backward = message.force_backward;
                if (message.state != null && message.hasOwnProperty("state"))
                    object.state = $root.caffe.NetState.toObject(message.state, options);
                if (message.debug_info != null && message.hasOwnProperty("debug_info"))
                    object.debug_info = message.debug_info;
                if (message.input_shape && message.input_shape.length) {
                    object.input_shape = [];
                    for (var j = 0; j < message.input_shape.length; ++j)
                        object.input_shape[j] = $root.caffe.BlobShape.toObject(message.input_shape[j], options);
                }
                if (message.layer && message.layer.length) {
                    object.layer = [];
                    for (var j = 0; j < message.layer.length; ++j)
                        object.layer[j] = $root.caffe.LayerParameter.toObject(message.layer[j], options);
                }
                return object;
            };
    
            NetParameter.prototype.toJSON = function toJSON() {
                return this.constructor.toObject(this, $protobuf.util.toJSONOptions);
            };
    
            return NetParameter;
        })();
    
        caffe.SolverParameter = (function() {
    
            function SolverParameter(properties) {
                this.test_net = [];
                this.test_net_param = [];
                this.test_state = [];
                this.test_iter = [];
                this.stepvalue = [];
                this.weights = [];
                if (properties)
                    for (var keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                        if (properties[keys[i]] != null)
                            this[keys[i]] = properties[keys[i]];
            }
    
            SolverParameter.prototype.net = "";
            SolverParameter.prototype.net_param = null;
            SolverParameter.prototype.train_net = "";
            SolverParameter.prototype.test_net = $util.emptyArray;
            SolverParameter.prototype.train_net_param = null;
            SolverParameter.prototype.test_net_param = $util.emptyArray;
            SolverParameter.prototype.train_state = null;
            SolverParameter.prototype.test_state = $util.emptyArray;
            SolverParameter.prototype.test_iter = $util.emptyArray;
            SolverParameter.prototype.test_interval = 0;
            SolverParameter.prototype.test_compute_loss = false;
            SolverParameter.prototype.test_initialization = true;
            SolverParameter.prototype.base_lr = 0;
            SolverParameter.prototype.display = 0;
            SolverParameter.prototype.average_loss = 1;
            SolverParameter.prototype.max_iter = 0;
            SolverParameter.prototype.iter_size = 1;
            SolverParameter.prototype.lr_policy = "";
            SolverParameter.prototype.gamma = 0;
            SolverParameter.prototype.power = 0;
            SolverParameter.prototype.momentum = 0;
            SolverParameter.prototype.weight_decay = 0;
            SolverParameter.prototype.regularization_type = "L2";
            SolverParameter.prototype.stepsize = 0;
            SolverParameter.prototype.stepvalue = $util.emptyArray;
            SolverParameter.prototype.clip_gradients = -1;
            SolverParameter.prototype.snapshot = 0;
            SolverParameter.prototype.snapshot_prefix = "";
            SolverParameter.prototype.snapshot_diff = false;
            SolverParameter.prototype.snapshot_format = 1;
            SolverParameter.prototype.solver_mode = 1;
            SolverParameter.prototype.device_id = 0;
            SolverParameter.prototype.random_seed = $util.Long ? $util.Long.fromBits(-1,-1,false) : -1;
            SolverParameter.prototype.type = "SGD";
            SolverParameter.prototype.delta = 1e-8;
            SolverParameter.prototype.momentum2 = 0.999;
            SolverParameter.prototype.rms_decay = 0.99;
            SolverParameter.prototype.debug_info = false;
            SolverParameter.prototype.snapshot_after_train = true;
            SolverParameter.prototype.solver_type = 0;
            SolverParameter.prototype.layer_wise_reduce = true;
            SolverParameter.prototype.weights = $util.emptyArray;
    
            SolverParameter.create = function create(properties) {
                return new SolverParameter(properties);
            };
    
            SolverParameter.decode = function decode(reader, length) {
                if (!(reader instanceof $Reader))
                    reader = $Reader.create(reader);
                var end = length === undefined ? reader.len : reader.pos + length, message = new $root.caffe.SolverParameter();
                while (reader.pos < end) {
                    var tag = reader.uint32();
                    switch (tag >>> 3) {
                    case 24:
                        message.net = reader.string();
                        break;
                    case 25:
                        message.net_param = $root.caffe.NetParameter.decode(reader, reader.uint32());
                        break;
                    case 1:
                        message.train_net = reader.string();
                        break;
                    case 2:
                        if (!(message.test_net && message.test_net.length))
                            message.test_net = [];
                        message.test_net.push(reader.string());
                        break;
                    case 21:
                        message.train_net_param = $root.caffe.NetParameter.decode(reader, reader.uint32());
                        break;
                    case 22:
                        if (!(message.test_net_param && message.test_net_param.length))
                            message.test_net_param = [];
                        message.test_net_param.push($root.caffe.NetParameter.decode(reader, reader.uint32()));
                        break;
                    case 26:
                        message.train_state = $root.caffe.NetState.decode(reader, reader.uint32());
                        break;
                    case 27:
                        if (!(message.test_state && message.test_state.length))
                            message.test_state = [];
                        message.test_state.push($root.caffe.NetState.decode(reader, reader.uint32()));
                        break;
                    case 3:
                        if (!(message.test_iter && message.test_iter.length))
                            message.test_iter = [];
                        if ((tag & 7) === 2) {
                            var end2 = reader.uint32() + reader.pos;
                            while (reader.pos < end2)
                                message.test_iter.push(reader.int32());
                        } else
                            message.test_iter.push(reader.int32());
                        break;
                    case 4:
                        message.test_interval = reader.int32();
                        break;
                    case 19:
                        message.test_compute_loss = reader.bool();
                        break;
                    case 32:
                        message.test_initialization = reader.bool();
                        break;
                    case 5:
                        message.base_lr = reader.float();
                        break;
                    case 6:
                        message.display = reader.int32();
                        break;
                    case 33:
                        message.average_loss = reader.int32();
                        break;
                    case 7:
                        message.max_iter = reader.int32();
                        break;
                    case 36:
                        message.iter_size = reader.int32();
                        break;
                    case 8:
                        message.lr_policy = reader.string();
                        break;
                    case 9:
                        message.gamma = reader.float();
                        break;
                    case 10:
                        message.power = reader.float();
                        break;
                    case 11:
                        message.momentum = reader.float();
                        break;
                    case 12:
                        message.weight_decay = reader.float();
                        break;
                    case 29:
                        message.regularization_type = reader.string();
                        break;
                    case 13:
                        message.stepsize = reader.int32();
                        break;
                    case 34:
                        if (!(message.stepvalue && message.stepvalue.length))
                            message.stepvalue = [];
                        if ((tag & 7) === 2) {
                            var end2 = reader.uint32() + reader.pos;
                            while (reader.pos < end2)
                                message.stepvalue.push(reader.int32());
                        } else
                            message.stepvalue.push(reader.int32());
                        break;
                    case 35:
                        message.clip_gradients = reader.float();
                        break;
                    case 14:
                        message.snapshot = reader.int32();
                        break;
                    case 15:
                        message.snapshot_prefix = reader.string();
                        break;
                    case 16:
                        message.snapshot_diff = reader.bool();
                        break;
                    case 37:
                        message.snapshot_format = reader.int32();
                        break;
                    case 17:
                        message.solver_mode = reader.int32();
                        break;
                    case 18:
                        message.device_id = reader.int32();
                        break;
                    case 20:
                        message.random_seed = reader.int64();
                        break;
                    case 40:
                        message.type = reader.string();
                        break;
                    case 31:
                        message.delta = reader.float();
                        break;
                    case 39:
                        message.momentum2 = reader.float();
                        break;
                    case 38:
                        message.rms_decay = reader.float();
                        break;
                    case 23:
                        message.debug_info = reader.bool();
                        break;
                    case 28:
                        message.snapshot_after_train = reader.bool();
                        break;
                    case 30:
                        message.solver_type = reader.int32();
                        break;
                    case 41:
                        message.layer_wise_reduce = reader.bool();
                        break;
                    case 42:
                        if (!(message.weights && message.weights.length))
                            message.weights = [];
                        message.weights.push(reader.string());
                        break;
                    default:
                        reader.skipType(tag & 7);
                        break;
                    }
                }
                return message;
            };
    
            SolverParameter.decodeText = function decodeText(reader, block) {
                if (!(reader instanceof $TextReader))
                    reader = $TextReader.create(reader);
                var message = new $root.caffe.SolverParameter();
                reader.start(block);
                while (!reader.end(block)) {
                    var tag = reader.tag();
                    switch (tag) {
                    case "net":
                        message.net = reader.string();
                        break;
                    case "net_param":
                        message.net_param = $root.caffe.NetParameter.decodeText(reader, true);
                        break;
                    case "train_net":
                        message.train_net = reader.string();
                        break;
                    case "test_net":
                        if (!(message.test_net && message.test_net.length))
                            message.test_net = [];
                        message.test_net.push(reader.string());
                        break;
                    case "train_net_param":
                        message.train_net_param = $root.caffe.NetParameter.decodeText(reader, true);
                        break;
                    case "test_net_param":
                        if (!(message.test_net_param && message.test_net_param.length))
                            message.test_net_param = [];
                        message.test_net_param.push($root.caffe.NetParameter.decodeText(reader, true));
                        break;
                    case "train_state":
                        message.train_state = $root.caffe.NetState.decodeText(reader, true);
                        break;
                    case "test_state":
                        if (!(message.test_state && message.test_state.length))
                            message.test_state = [];
                        message.test_state.push($root.caffe.NetState.decodeText(reader, true));
                        break;
                    case "test_iter":
                        if (!(message.test_iter && message.test_iter.length))
                            message.test_iter = [];
                        message.test_iter.push(reader.int32());
                        break;
                    case "test_interval":
                        message.test_interval = reader.int32();
                        break;
                    case "test_compute_loss":
                        message.test_compute_loss = reader.bool();
                        break;
                    case "test_initialization":
                        message.test_initialization = reader.bool();
                        break;
                    case "base_lr":
                        message.base_lr = reader.float();
                        break;
                    case "display":
                        message.display = reader.int32();
                        break;
                    case "average_loss":
                        message.average_loss = reader.int32();
                        break;
                    case "max_iter":
                        message.max_iter = reader.int32();
                        break;
                    case "iter_size":
                        message.iter_size = reader.int32();
                        break;
                    case "lr_policy":
                        message.lr_policy = reader.string();
                        break;
                    case "gamma":
                        message.gamma = reader.float();
                        break;
                    case "power":
                        message.power = reader.float();
                        break;
                    case "momentum":
                        message.momentum = reader.float();
                        break;
                    case "weight_decay":
                        message.weight_decay = reader.float();
                        break;
                    case "regularization_type":
                        message.regularization_type = reader.string();
                        break;
                    case "stepsize":
                        message.stepsize = reader.int32();
                        break;
                    case "stepvalue":
                        if (!(message.stepvalue && message.stepvalue.length))
                            message.stepvalue = [];
                        message.stepvalue.push(reader.int32());
                        break;
                    case "clip_gradients":
                        message.clip_gradients = reader.float();
                        break;
                    case "snapshot":
                        message.snapshot = reader.int32();
                        break;
                    case "snapshot_prefix":
                        message.snapshot_prefix = reader.string();
                        break;
                    case "snapshot_diff":
                        message.snapshot_diff = reader.bool();
                        break;
                    case "snapshot_format":
                        message.snapshot_format = reader.enum($root.caffe.SolverParameter.SnapshotFormat);
                        break;
                    case "solver_mode":
                        message.solver_mode = reader.enum($root.caffe.SolverParameter.SolverMode);
                        break;
                    case "device_id":
                        message.device_id = reader.int32();
                        break;
                    case "random_seed":
                        message.random_seed = reader.int64();
                        break;
                    case "type":
                        message.type = reader.string();
                        break;
                    case "delta":
                        message.delta = reader.float();
                        break;
                    case "momentum2":
                        message.momentum2 = reader.float();
                        break;
                    case "rms_decay":
                        message.rms_decay = reader.float();
                        break;
                    case "debug_info":
                        message.debug_info = reader.bool();
                        break;
                    case "snapshot_after_train":
                        message.snapshot_after_train = reader.bool();
                        break;
                    case "solver_type":
                        message.solver_type = reader.enum($root.caffe.SolverParameter.SolverType);
                        break;
                    case "layer_wise_reduce":
                        message.layer_wise_reduce = reader.bool();
                        break;
                    case "weights":
                        if (!(message.weights && message.weights.length))
                            message.weights = [];
                        message.weights.push(reader.string());
                        break;
                    default:
                        reader.handle(tag, message);
                        break;
                    }
                }
                return message;
            };
    
            SolverParameter.verify = function verify(message) {
                if (typeof message !== "object" || message === null)
                    return "object expected";
                if (message.net != null && message.hasOwnProperty("net"))
                    if (!$util.isString(message.net))
                        return "net: string expected";
                if (message.net_param != null && message.hasOwnProperty("net_param")) {
                    var error = $root.caffe.NetParameter.verify(message.net_param);
                    if (error)
                        return "net_param." + error;
                }
                if (message.train_net != null && message.hasOwnProperty("train_net"))
                    if (!$util.isString(message.train_net))
                        return "train_net: string expected";
                if (message.test_net != null && message.hasOwnProperty("test_net")) {
                    if (!Array.isArray(message.test_net))
                        return "test_net: array expected";
                    for (var i = 0; i < message.test_net.length; ++i)
                        if (!$util.isString(message.test_net[i]))
                            return "test_net: string[] expected";
                }
                if (message.train_net_param != null && message.hasOwnProperty("train_net_param")) {
                    var error = $root.caffe.NetParameter.verify(message.train_net_param);
                    if (error)
                        return "train_net_param." + error;
                }
                if (message.test_net_param != null && message.hasOwnProperty("test_net_param")) {
                    if (!Array.isArray(message.test_net_param))
                        return "test_net_param: array expected";
                    for (var i = 0; i < message.test_net_param.length; ++i) {
                        var error = $root.caffe.NetParameter.verify(message.test_net_param[i]);
                        if (error)
                            return "test_net_param." + error;
                    }
                }
                if (message.train_state != null && message.hasOwnProperty("train_state")) {
                    var error = $root.caffe.NetState.verify(message.train_state);
                    if (error)
                        return "train_state." + error;
                }
                if (message.test_state != null && message.hasOwnProperty("test_state")) {
                    if (!Array.isArray(message.test_state))
                        return "test_state: array expected";
                    for (var i = 0; i < message.test_state.length; ++i) {
                        var error = $root.caffe.NetState.verify(message.test_state[i]);
                        if (error)
                            return "test_state." + error;
                    }
                }
                if (message.test_iter != null && message.hasOwnProperty("test_iter")) {
                    if (!Array.isArray(message.test_iter))
                        return "test_iter: array expected";
                    for (var i = 0; i < message.test_iter.length; ++i)
                        if (!$util.isInteger(message.test_iter[i]))
                            return "test_iter: integer[] expected";
                }
                if (message.test_interval != null && message.hasOwnProperty("test_interval"))
                    if (!$util.isInteger(message.test_interval))
                        return "test_interval: integer expected";
                if (message.test_compute_loss != null && message.hasOwnProperty("test_compute_loss"))
                    if (typeof message.test_compute_loss !== "boolean")
                        return "test_compute_loss: boolean expected";
                if (message.test_initialization != null && message.hasOwnProperty("test_initialization"))
                    if (typeof message.test_initialization !== "boolean")
                        return "test_initialization: boolean expected";
                if (message.base_lr != null && message.hasOwnProperty("base_lr"))
                    if (typeof message.base_lr !== "number")
                        return "base_lr: number expected";
                if (message.display != null && message.hasOwnProperty("display"))
                    if (!$util.isInteger(message.display))
                        return "display: integer expected";
                if (message.average_loss != null && message.hasOwnProperty("average_loss"))
                    if (!$util.isInteger(message.average_loss))
                        return "average_loss: integer expected";
                if (message.max_iter != null && message.hasOwnProperty("max_iter"))
                    if (!$util.isInteger(message.max_iter))
                        return "max_iter: integer expected";
                if (message.iter_size != null && message.hasOwnProperty("iter_size"))
                    if (!$util.isInteger(message.iter_size))
                        return "iter_size: integer expected";
                if (message.lr_policy != null && message.hasOwnProperty("lr_policy"))
                    if (!$util.isString(message.lr_policy))
                        return "lr_policy: string expected";
                if (message.gamma != null && message.hasOwnProperty("gamma"))
                    if (typeof message.gamma !== "number")
                        return "gamma: number expected";
                if (message.power != null && message.hasOwnProperty("power"))
                    if (typeof message.power !== "number")
                        return "power: number expected";
                if (message.momentum != null && message.hasOwnProperty("momentum"))
                    if (typeof message.momentum !== "number")
                        return "momentum: number expected";
                if (message.weight_decay != null && message.hasOwnProperty("weight_decay"))
                    if (typeof message.weight_decay !== "number")
                        return "weight_decay: number expected";
                if (message.regularization_type != null && message.hasOwnProperty("regularization_type"))
                    if (!$util.isString(message.regularization_type))
                        return "regularization_type: string expected";
                if (message.stepsize != null && message.hasOwnProperty("stepsize"))
                    if (!$util.isInteger(message.stepsize))
                        return "stepsize: integer expected";
                if (message.stepvalue != null && message.hasOwnProperty("stepvalue")) {
                    if (!Array.isArray(message.stepvalue))
                        return "stepvalue: array expected";
                    for (var i = 0; i < message.stepvalue.length; ++i)
                        if (!$util.isInteger(message.stepvalue[i]))
                            return "stepvalue: integer[] expected";
                }
                if (message.clip_gradients != null && message.hasOwnProperty("clip_gradients"))
                    if (typeof message.clip_gradients !== "number")
                        return "clip_gradients: number expected";
                if (message.snapshot != null && message.hasOwnProperty("snapshot"))
                    if (!$util.isInteger(message.snapshot))
                        return "snapshot: integer expected";
                if (message.snapshot_prefix != null && message.hasOwnProperty("snapshot_prefix"))
                    if (!$util.isString(message.snapshot_prefix))
                        return "snapshot_prefix: string expected";
                if (message.snapshot_diff != null && message.hasOwnProperty("snapshot_diff"))
                    if (typeof message.snapshot_diff !== "boolean")
                        return "snapshot_diff: boolean expected";
                if (message.snapshot_format != null && message.hasOwnProperty("snapshot_format"))
                    switch (message.snapshot_format) {
                    default:
                        return "snapshot_format: enum value expected";
                    case 0:
                    case 1:
                        break;
                    }
                if (message.solver_mode != null && message.hasOwnProperty("solver_mode"))
                    switch (message.solver_mode) {
                    default:
                        return "solver_mode: enum value expected";
                    case 0:
                    case 1:
                        break;
                    }
                if (message.device_id != null && message.hasOwnProperty("device_id"))
                    if (!$util.isInteger(message.device_id))
                        return "device_id: integer expected";
                if (message.random_seed != null && message.hasOwnProperty("random_seed"))
                    if (!$util.isInteger(message.random_seed) && !(message.random_seed && $util.isInteger(message.random_seed.low) && $util.isInteger(message.random_seed.high)))
                        return "random_seed: integer|Long expected";
                if (message.type != null && message.hasOwnProperty("type"))
                    if (!$util.isString(message.type))
                        return "type: string expected";
                if (message.delta != null && message.hasOwnProperty("delta"))
                    if (typeof message.delta !== "number")
                        return "delta: number expected";
                if (message.momentum2 != null && message.hasOwnProperty("momentum2"))
                    if (typeof message.momentum2 !== "number")
                        return "momentum2: number expected";
                if (message.rms_decay != null && message.hasOwnProperty("rms_decay"))
                    if (typeof message.rms_decay !== "number")
                        return "rms_decay: number expected";
                if (message.debug_info != null && message.hasOwnProperty("debug_info"))
                    if (typeof message.debug_info !== "boolean")
                        return "debug_info: boolean expected";
                if (message.snapshot_after_train != null && message.hasOwnProperty("snapshot_after_train"))
                    if (typeof message.snapshot_after_train !== "boolean")
                        return "snapshot_after_train: boolean expected";
                if (message.solver_type != null && message.hasOwnProperty("solver_type"))
                    switch (message.solver_type) {
                    default:
                        return "solver_type: enum value expected";
                    case 0:
                    case 1:
                    case 2:
                    case 3:
                    case 4:
                    case 5:
                        break;
                    }
                if (message.layer_wise_reduce != null && message.hasOwnProperty("layer_wise_reduce"))
                    if (typeof message.layer_wise_reduce !== "boolean")
                        return "layer_wise_reduce: boolean expected";
                if (message.weights != null && message.hasOwnProperty("weights")) {
                    if (!Array.isArray(message.weights))
                        return "weights: array expected";
                    for (var i = 0; i < message.weights.length; ++i)
                        if (!$util.isString(message.weights[i]))
                            return "weights: string[] expected";
                }
                return null;
            };
    
            SolverParameter.fromObject = function fromObject(object) {
                if (object instanceof $root.caffe.SolverParameter)
                    return object;
                var message = new $root.caffe.SolverParameter();
                if (object.net != null)
                    message.net = String(object.net);
                if (object.net_param != null) {
                    if (typeof object.net_param !== "object")
                        throw TypeError(".caffe.SolverParameter.net_param: object expected");
                    message.net_param = $root.caffe.NetParameter.fromObject(object.net_param);
                }
                if (object.train_net != null)
                    message.train_net = String(object.train_net);
                if (object.test_net) {
                    if (!Array.isArray(object.test_net))
                        throw TypeError(".caffe.SolverParameter.test_net: array expected");
                    message.test_net = [];
                    for (var i = 0; i < object.test_net.length; ++i)
                        message.test_net[i] = String(object.test_net[i]);
                }
                if (object.train_net_param != null) {
                    if (typeof object.train_net_param !== "object")
                        throw TypeError(".caffe.SolverParameter.train_net_param: object expected");
                    message.train_net_param = $root.caffe.NetParameter.fromObject(object.train_net_param);
                }
                if (object.test_net_param) {
                    if (!Array.isArray(object.test_net_param))
                        throw TypeError(".caffe.SolverParameter.test_net_param: array expected");
                    message.test_net_param = [];
                    for (var i = 0; i < object.test_net_param.length; ++i) {
                        if (typeof object.test_net_param[i] !== "object")
                            throw TypeError(".caffe.SolverParameter.test_net_param: object expected");
                        message.test_net_param[i] = $root.caffe.NetParameter.fromObject(object.test_net_param[i]);
                    }
                }
                if (object.train_state != null) {
                    if (typeof object.train_state !== "object")
                        throw TypeError(".caffe.SolverParameter.train_state: object expected");
                    message.train_state = $root.caffe.NetState.fromObject(object.train_state);
                }
                if (object.test_state) {
                    if (!Array.isArray(object.test_state))
                        throw TypeError(".caffe.SolverParameter.test_state: array expected");
                    message.test_state = [];
                    for (var i = 0; i < object.test_state.length; ++i) {
                        if (typeof object.test_state[i] !== "object")
                            throw TypeError(".caffe.SolverParameter.test_state: object expected");
                        message.test_state[i] = $root.caffe.NetState.fromObject(object.test_state[i]);
                    }
                }
                if (object.test_iter) {
                    if (!Array.isArray(object.test_iter))
                        throw TypeError(".caffe.SolverParameter.test_iter: array expected");
                    message.test_iter = [];
                    for (var i = 0; i < object.test_iter.length; ++i)
                        message.test_iter[i] = object.test_iter[i] | 0;
                }
                if (object.test_interval != null)
                    message.test_interval = object.test_interval | 0;
                if (object.test_compute_loss != null)
                    message.test_compute_loss = Boolean(object.test_compute_loss);
                if (object.test_initialization != null)
                    message.test_initialization = Boolean(object.test_initialization);
                if (object.base_lr != null)
                    message.base_lr = Number(object.base_lr);
                if (object.display != null)
                    message.display = object.display | 0;
                if (object.average_loss != null)
                    message.average_loss = object.average_loss | 0;
                if (object.max_iter != null)
                    message.max_iter = object.max_iter | 0;
                if (object.iter_size != null)
                    message.iter_size = object.iter_size | 0;
                if (object.lr_policy != null)
                    message.lr_policy = String(object.lr_policy);
                if (object.gamma != null)
                    message.gamma = Number(object.gamma);
                if (object.power != null)
                    message.power = Number(object.power);
                if (object.momentum != null)
                    message.momentum = Number(object.momentum);
                if (object.weight_decay != null)
                    message.weight_decay = Number(object.weight_decay);
                if (object.regularization_type != null)
                    message.regularization_type = String(object.regularization_type);
                if (object.stepsize != null)
                    message.stepsize = object.stepsize | 0;
                if (object.stepvalue) {
                    if (!Array.isArray(object.stepvalue))
                        throw TypeError(".caffe.SolverParameter.stepvalue: array expected");
                    message.stepvalue = [];
                    for (var i = 0; i < object.stepvalue.length; ++i)
                        message.stepvalue[i] = object.stepvalue[i] | 0;
                }
                if (object.clip_gradients != null)
                    message.clip_gradients = Number(object.clip_gradients);
                if (object.snapshot != null)
                    message.snapshot = object.snapshot | 0;
                if (object.snapshot_prefix != null)
                    message.snapshot_prefix = String(object.snapshot_prefix);
                if (object.snapshot_diff != null)
                    message.snapshot_diff = Boolean(object.snapshot_diff);
                switch (object.snapshot_format) {
                case "HDF5":
                case 0:
                    message.snapshot_format = 0;
                    break;
                case "BINARYPROTO":
                case 1:
                    message.snapshot_format = 1;
                    break;
                }
                switch (object.solver_mode) {
                case "CPU":
                case 0:
                    message.solver_mode = 0;
                    break;
                case "GPU":
                case 1:
                    message.solver_mode = 1;
                    break;
                }
                if (object.device_id != null)
                    message.device_id = object.device_id | 0;
                if (object.random_seed != null)
                    if ($util.Long)
                        (message.random_seed = $util.Long.fromValue(object.random_seed)).unsigned = false;
                    else if (typeof object.random_seed === "string")
                        message.random_seed = parseInt(object.random_seed, 10);
                    else if (typeof object.random_seed === "number")
                        message.random_seed = object.random_seed;
                    else if (typeof object.random_seed === "object")
                        message.random_seed = new $util.LongBits(object.random_seed.low >>> 0, object.random_seed.high >>> 0).toNumber();
                if (object.type != null)
                    message.type = String(object.type);
                if (object.delta != null)
                    message.delta = Number(object.delta);
                if (object.momentum2 != null)
                    message.momentum2 = Number(object.momentum2);
                if (object.rms_decay != null)
                    message.rms_decay = Number(object.rms_decay);
                if (object.debug_info != null)
                    message.debug_info = Boolean(object.debug_info);
                if (object.snapshot_after_train != null)
                    message.snapshot_after_train = Boolean(object.snapshot_after_train);
                switch (object.solver_type) {
                case "SGD":
                case 0:
                    message.solver_type = 0;
                    break;
                case "NESTEROV":
                case 1:
                    message.solver_type = 1;
                    break;
                case "ADAGRAD":
                case 2:
                    message.solver_type = 2;
                    break;
                case "RMSPROP":
                case 3:
                    message.solver_type = 3;
                    break;
                case "ADADELTA":
                case 4:
                    message.solver_type = 4;
                    break;
                case "ADAM":
                case 5:
                    message.solver_type = 5;
                    break;
                }
                if (object.layer_wise_reduce != null)
                    message.layer_wise_reduce = Boolean(object.layer_wise_reduce);
                if (object.weights) {
                    if (!Array.isArray(object.weights))
                        throw TypeError(".caffe.SolverParameter.weights: array expected");
                    message.weights = [];
                    for (var i = 0; i < object.weights.length; ++i)
                        message.weights[i] = String(object.weights[i]);
                }
                return message;
            };
    
            SolverParameter.toObject = function toObject(message, options) {
                if (!options)
                    options = {};
                var object = {};
                if (options.arrays || options.defaults) {
                    object.test_net = [];
                    object.test_iter = [];
                    object.test_net_param = [];
                    object.test_state = [];
                    object.stepvalue = [];
                    object.weights = [];
                }
                if (options.defaults) {
                    object.train_net = "";
                    object.test_interval = 0;
                    object.base_lr = 0;
                    object.display = 0;
                    object.max_iter = 0;
                    object.lr_policy = "";
                    object.gamma = 0;
                    object.power = 0;
                    object.momentum = 0;
                    object.weight_decay = 0;
                    object.stepsize = 0;
                    object.snapshot = 0;
                    object.snapshot_prefix = "";
                    object.snapshot_diff = false;
                    object.solver_mode = options.enums === String ? "GPU" : 1;
                    object.device_id = 0;
                    object.test_compute_loss = false;
                    if ($util.Long) {
                        var long = new $util.Long(-1, -1, false);
                        object.random_seed = options.longs === String ? long.toString() : options.longs === Number ? long.toNumber() : long;
                    } else
                        object.random_seed = options.longs === String ? "-1" : -1;
                    object.train_net_param = null;
                    object.debug_info = false;
                    object.net = "";
                    object.net_param = null;
                    object.train_state = null;
                    object.snapshot_after_train = true;
                    object.regularization_type = "L2";
                    object.solver_type = options.enums === String ? "SGD" : 0;
                    object.delta = 1e-8;
                    object.test_initialization = true;
                    object.average_loss = 1;
                    object.clip_gradients = -1;
                    object.iter_size = 1;
                    object.snapshot_format = options.enums === String ? "BINARYPROTO" : 1;
                    object.rms_decay = 0.99;
                    object.momentum2 = 0.999;
                    object.type = "SGD";
                    object.layer_wise_reduce = true;
                }
                if (message.train_net != null && message.hasOwnProperty("train_net"))
                    object.train_net = message.train_net;
                if (message.test_net && message.test_net.length) {
                    object.test_net = [];
                    for (var j = 0; j < message.test_net.length; ++j)
                        object.test_net[j] = message.test_net[j];
                }
                if (message.test_iter && message.test_iter.length) {
                    object.test_iter = [];
                    for (var j = 0; j < message.test_iter.length; ++j)
                        object.test_iter[j] = message.test_iter[j];
                }
                if (message.test_interval != null && message.hasOwnProperty("test_interval"))
                    object.test_interval = message.test_interval;
                if (message.base_lr != null && message.hasOwnProperty("base_lr"))
                    object.base_lr = options.json && !isFinite(message.base_lr) ? String(message.base_lr) : message.base_lr;
                if (message.display != null && message.hasOwnProperty("display"))
                    object.display = message.display;
                if (message.max_iter != null && message.hasOwnProperty("max_iter"))
                    object.max_iter = message.max_iter;
                if (message.lr_policy != null && message.hasOwnProperty("lr_policy"))
                    object.lr_policy = message.lr_policy;
                if (message.gamma != null && message.hasOwnProperty("gamma"))
                    object.gamma = options.json && !isFinite(message.gamma) ? String(message.gamma) : message.gamma;
                if (message.power != null && message.hasOwnProperty("power"))
                    object.power = options.json && !isFinite(message.power) ? String(message.power) : message.power;
                if (message.momentum != null && message.hasOwnProperty("momentum"))
                    object.momentum = options.json && !isFinite(message.momentum) ? String(message.momentum) : message.momentum;
                if (message.weight_decay != null && message.hasOwnProperty("weight_decay"))
                    object.weight_decay = options.json && !isFinite(message.weight_decay) ? String(message.weight_decay) : message.weight_decay;
                if (message.stepsize != null && message.hasOwnProperty("stepsize"))
                    object.stepsize = message.stepsize;
                if (message.snapshot != null && message.hasOwnProperty("snapshot"))
                    object.snapshot = message.snapshot;
                if (message.snapshot_prefix != null && message.hasOwnProperty("snapshot_prefix"))
                    object.snapshot_prefix = message.snapshot_prefix;
                if (message.snapshot_diff != null && message.hasOwnProperty("snapshot_diff"))
                    object.snapshot_diff = message.snapshot_diff;
                if (message.solver_mode != null && message.hasOwnProperty("solver_mode"))
                    object.solver_mode = options.enums === String ? $root.caffe.SolverParameter.SolverMode[message.solver_mode] : message.solver_mode;
                if (message.device_id != null && message.hasOwnProperty("device_id"))
                    object.device_id = message.device_id;
                if (message.test_compute_loss != null && message.hasOwnProperty("test_compute_loss"))
                    object.test_compute_loss = message.test_compute_loss;
                if (message.random_seed != null && message.hasOwnProperty("random_seed"))
                    if (typeof message.random_seed === "number")
                        object.random_seed = options.longs === String ? String(message.random_seed) : message.random_seed;
                    else
                        object.random_seed = options.longs === String ? $util.Long.prototype.toString.call(message.random_seed) : options.longs === Number ? new $util.LongBits(message.random_seed.low >>> 0, message.random_seed.high >>> 0).toNumber() : message.random_seed;
                if (message.train_net_param != null && message.hasOwnProperty("train_net_param"))
                    object.train_net_param = $root.caffe.NetParameter.toObject(message.train_net_param, options);
                if (message.test_net_param && message.test_net_param.length) {
                    object.test_net_param = [];
                    for (var j = 0; j < message.test_net_param.length; ++j)
                        object.test_net_param[j] = $root.caffe.NetParameter.toObject(message.test_net_param[j], options);
                }
                if (message.debug_info != null && message.hasOwnProperty("debug_info"))
                    object.debug_info = message.debug_info;
                if (message.net != null && message.hasOwnProperty("net"))
                    object.net = message.net;
                if (message.net_param != null && message.hasOwnProperty("net_param"))
                    object.net_param = $root.caffe.NetParameter.toObject(message.net_param, options);
                if (message.train_state != null && message.hasOwnProperty("train_state"))
                    object.train_state = $root.caffe.NetState.toObject(message.train_state, options);
                if (message.test_state && message.test_state.length) {
                    object.test_state = [];
                    for (var j = 0; j < message.test_state.length; ++j)
                        object.test_state[j] = $root.caffe.NetState.toObject(message.test_state[j], options);
                }
                if (message.snapshot_after_train != null && message.hasOwnProperty("snapshot_after_train"))
                    object.snapshot_after_train = message.snapshot_after_train;
                if (message.regularization_type != null && message.hasOwnProperty("regularization_type"))
                    object.regularization_type = message.regularization_type;
                if (message.solver_type != null && message.hasOwnProperty("solver_type"))
                    object.solver_type = options.enums === String ? $root.caffe.SolverParameter.SolverType[message.solver_type] : message.solver_type;
                if (message.delta != null && message.hasOwnProperty("delta"))
                    object.delta = options.json && !isFinite(message.delta) ? String(message.delta) : message.delta;
                if (message.test_initialization != null && message.hasOwnProperty("test_initialization"))
                    object.test_initialization = message.test_initialization;
                if (message.average_loss != null && message.hasOwnProperty("average_loss"))
                    object.average_loss = message.average_loss;
                if (message.stepvalue && message.stepvalue.length) {
                    object.stepvalue = [];
                    for (var j = 0; j < message.stepvalue.length; ++j)
                        object.stepvalue[j] = message.stepvalue[j];
                }
                if (message.clip_gradients != null && message.hasOwnProperty("clip_gradients"))
                    object.clip_gradients = options.json && !isFinite(message.clip_gradients) ? String(message.clip_gradients) : message.clip_gradients;
                if (message.iter_size != null && message.hasOwnProperty("iter_size"))
                    object.iter_size = message.iter_size;
                if (message.snapshot_format != null && message.hasOwnProperty("snapshot_format"))
                    object.snapshot_format = options.enums === String ? $root.caffe.SolverParameter.SnapshotFormat[message.snapshot_format] : message.snapshot_format;
                if (message.rms_decay != null && message.hasOwnProperty("rms_decay"))
                    object.rms_decay = options.json && !isFinite(message.rms_decay) ? String(message.rms_decay) : message.rms_decay;
                if (message.momentum2 != null && message.hasOwnProperty("momentum2"))
                    object.momentum2 = options.json && !isFinite(message.momentum2) ? String(message.momentum2) : message.momentum2;
                if (message.type != null && message.hasOwnProperty("type"))
                    object.type = message.type;
                if (message.layer_wise_reduce != null && message.hasOwnProperty("layer_wise_reduce"))
                    object.layer_wise_reduce = message.layer_wise_reduce;
                if (message.weights && message.weights.length) {
                    object.weights = [];
                    for (var j = 0; j < message.weights.length; ++j)
                        object.weights[j] = message.weights[j];
                }
                return object;
            };
    
            SolverParameter.prototype.toJSON = function toJSON() {
                return this.constructor.toObject(this, $protobuf.util.toJSONOptions);
            };
    
            SolverParameter.SnapshotFormat = (function() {
                var valuesById = {}, values = Object.create(valuesById);
                values[valuesById[0] = "HDF5"] = 0;
                values[valuesById[1] = "BINARYPROTO"] = 1;
                return values;
            })();
    
            SolverParameter.SolverMode = (function() {
                var valuesById = {}, values = Object.create(valuesById);
                values[valuesById[0] = "CPU"] = 0;
                values[valuesById[1] = "GPU"] = 1;
                return values;
            })();
    
            SolverParameter.SolverType = (function() {
                var valuesById = {}, values = Object.create(valuesById);
                values[valuesById[0] = "SGD"] = 0;
                values[valuesById[1] = "NESTEROV"] = 1;
                values[valuesById[2] = "ADAGRAD"] = 2;
                values[valuesById[3] = "RMSPROP"] = 3;
                values[valuesById[4] = "ADADELTA"] = 4;
                values[valuesById[5] = "ADAM"] = 5;
                return values;
            })();
    
            return SolverParameter;
        })();
    
        caffe.SolverState = (function() {
    
            function SolverState(properties) {
                this.history = [];
                if (properties)
                    for (var keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                        if (properties[keys[i]] != null)
                            this[keys[i]] = properties[keys[i]];
            }
    
            SolverState.prototype.iter = 0;
            SolverState.prototype.learned_net = "";
            SolverState.prototype.history = $util.emptyArray;
            SolverState.prototype.current_step = 0;
    
            SolverState.create = function create(properties) {
                return new SolverState(properties);
            };
    
            SolverState.decode = function decode(reader, length) {
                if (!(reader instanceof $Reader))
                    reader = $Reader.create(reader);
                var end = length === undefined ? reader.len : reader.pos + length, message = new $root.caffe.SolverState();
                while (reader.pos < end) {
                    var tag = reader.uint32();
                    switch (tag >>> 3) {
                    case 1:
                        message.iter = reader.int32();
                        break;
                    case 2:
                        message.learned_net = reader.string();
                        break;
                    case 3:
                        if (!(message.history && message.history.length))
                            message.history = [];
                        message.history.push($root.caffe.BlobProto.decode(reader, reader.uint32()));
                        break;
                    case 4:
                        message.current_step = reader.int32();
                        break;
                    default:
                        reader.skipType(tag & 7);
                        break;
                    }
                }
                return message;
            };
    
            SolverState.decodeText = function decodeText(reader, block) {
                if (!(reader instanceof $TextReader))
                    reader = $TextReader.create(reader);
                var message = new $root.caffe.SolverState();
                reader.start(block);
                while (!reader.end(block)) {
                    var tag = reader.tag();
                    switch (tag) {
                    case "iter":
                        message.iter = reader.int32();
                        break;
                    case "learned_net":
                        message.learned_net = reader.string();
                        break;
                    case "history":
                        if (!(message.history && message.history.length))
                            message.history = [];
                        message.history.push($root.caffe.BlobProto.decodeText(reader, true));
                        break;
                    case "current_step":
                        message.current_step = reader.int32();
                        break;
                    default:
                        reader.handle(tag, message);
                        break;
                    }
                }
                return message;
            };
    
            SolverState.verify = function verify(message) {
                if (typeof message !== "object" || message === null)
                    return "object expected";
                if (message.iter != null && message.hasOwnProperty("iter"))
                    if (!$util.isInteger(message.iter))
                        return "iter: integer expected";
                if (message.learned_net != null && message.hasOwnProperty("learned_net"))
                    if (!$util.isString(message.learned_net))
                        return "learned_net: string expected";
                if (message.history != null && message.hasOwnProperty("history")) {
                    if (!Array.isArray(message.history))
                        return "history: array expected";
                    for (var i = 0; i < message.history.length; ++i) {
                        var error = $root.caffe.BlobProto.verify(message.history[i]);
                        if (error)
                            return "history." + error;
                    }
                }
                if (message.current_step != null && message.hasOwnProperty("current_step"))
                    if (!$util.isInteger(message.current_step))
                        return "current_step: integer expected";
                return null;
            };
    
            SolverState.fromObject = function fromObject(object) {
                if (object instanceof $root.caffe.SolverState)
                    return object;
                var message = new $root.caffe.SolverState();
                if (object.iter != null)
                    message.iter = object.iter | 0;
                if (object.learned_net != null)
                    message.learned_net = String(object.learned_net);
                if (object.history) {
                    if (!Array.isArray(object.history))
                        throw TypeError(".caffe.SolverState.history: array expected");
                    message.history = [];
                    for (var i = 0; i < object.history.length; ++i) {
                        if (typeof object.history[i] !== "object")
                            throw TypeError(".caffe.SolverState.history: object expected");
                        message.history[i] = $root.caffe.BlobProto.fromObject(object.history[i]);
                    }
                }
                if (object.current_step != null)
                    message.current_step = object.current_step | 0;
                return message;
            };
    
            SolverState.toObject = function toObject(message, options) {
                if (!options)
                    options = {};
                var object = {};
                if (options.arrays || options.defaults)
                    object.history = [];
                if (options.defaults) {
                    object.iter = 0;
                    object.learned_net = "";
                    object.current_step = 0;
                }
                if (message.iter != null && message.hasOwnProperty("iter"))
                    object.iter = message.iter;
                if (message.learned_net != null && message.hasOwnProperty("learned_net"))
                    object.learned_net = message.learned_net;
                if (message.history && message.history.length) {
                    object.history = [];
                    for (var j = 0; j < message.history.length; ++j)
                        object.history[j] = $root.caffe.BlobProto.toObject(message.history[j], options);
                }
                if (message.current_step != null && message.hasOwnProperty("current_step"))
                    object.current_step = message.current_step;
                return object;
            };
    
            SolverState.prototype.toJSON = function toJSON() {
                return this.constructor.toObject(this, $protobuf.util.toJSONOptions);
            };
    
            return SolverState;
        })();
    
        caffe.Phase = (function() {
            var valuesById = {}, values = Object.create(valuesById);
            values[valuesById[0] = "TRAIN"] = 0;
            values[valuesById[1] = "TEST"] = 1;
            return values;
        })();
    
        caffe.NetState = (function() {
    
            function NetState(properties) {
                this.stage = [];
                if (properties)
                    for (var keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                        if (properties[keys[i]] != null)
                            this[keys[i]] = properties[keys[i]];
            }
    
            NetState.prototype.phase = 1;
            NetState.prototype.level = 0;
            NetState.prototype.stage = $util.emptyArray;
    
            NetState.create = function create(properties) {
                return new NetState(properties);
            };
    
            NetState.decode = function decode(reader, length) {
                if (!(reader instanceof $Reader))
                    reader = $Reader.create(reader);
                var end = length === undefined ? reader.len : reader.pos + length, message = new $root.caffe.NetState();
                while (reader.pos < end) {
                    var tag = reader.uint32();
                    switch (tag >>> 3) {
                    case 1:
                        message.phase = reader.int32();
                        break;
                    case 2:
                        message.level = reader.int32();
                        break;
                    case 3:
                        if (!(message.stage && message.stage.length))
                            message.stage = [];
                        message.stage.push(reader.string());
                        break;
                    default:
                        reader.skipType(tag & 7);
                        break;
                    }
                }
                return message;
            };
    
            NetState.decodeText = function decodeText(reader, block) {
                if (!(reader instanceof $TextReader))
                    reader = $TextReader.create(reader);
                var message = new $root.caffe.NetState();
                reader.start(block);
                while (!reader.end(block)) {
                    var tag = reader.tag();
                    switch (tag) {
                    case "phase":
                        message.phase = reader.enum($root.caffe.Phase);
                        break;
                    case "level":
                        message.level = reader.int32();
                        break;
                    case "stage":
                        if (!(message.stage && message.stage.length))
                            message.stage = [];
                        message.stage.push(reader.string());
                        break;
                    default:
                        reader.handle(tag, message);
                        break;
                    }
                }
                return message;
            };
    
            NetState.verify = function verify(message) {
                if (typeof message !== "object" || message === null)
                    return "object expected";
                if (message.phase != null && message.hasOwnProperty("phase"))
                    switch (message.phase) {
                    default:
                        return "phase: enum value expected";
                    case 0:
                    case 1:
                        break;
                    }
                if (message.level != null && message.hasOwnProperty("level"))
                    if (!$util.isInteger(message.level))
                        return "level: integer expected";
                if (message.stage != null && message.hasOwnProperty("stage")) {
                    if (!Array.isArray(message.stage))
                        return "stage: array expected";
                    for (var i = 0; i < message.stage.length; ++i)
                        if (!$util.isString(message.stage[i]))
                            return "stage: string[] expected";
                }
                return null;
            };
    
            NetState.fromObject = function fromObject(object) {
                if (object instanceof $root.caffe.NetState)
                    return object;
                var message = new $root.caffe.NetState();
                switch (object.phase) {
                case "TRAIN":
                case 0:
                    message.phase = 0;
                    break;
                case "TEST":
                case 1:
                    message.phase = 1;
                    break;
                }
                if (object.level != null)
                    message.level = object.level | 0;
                if (object.stage) {
                    if (!Array.isArray(object.stage))
                        throw TypeError(".caffe.NetState.stage: array expected");
                    message.stage = [];
                    for (var i = 0; i < object.stage.length; ++i)
                        message.stage[i] = String(object.stage[i]);
                }
                return message;
            };
    
            NetState.toObject = function toObject(message, options) {
                if (!options)
                    options = {};
                var object = {};
                if (options.arrays || options.defaults)
                    object.stage = [];
                if (options.defaults) {
                    object.phase = options.enums === String ? "TEST" : 1;
                    object.level = 0;
                }
                if (message.phase != null && message.hasOwnProperty("phase"))
                    object.phase = options.enums === String ? $root.caffe.Phase[message.phase] : message.phase;
                if (message.level != null && message.hasOwnProperty("level"))
                    object.level = message.level;
                if (message.stage && message.stage.length) {
                    object.stage = [];
                    for (var j = 0; j < message.stage.length; ++j)
                        object.stage[j] = message.stage[j];
                }
                return object;
            };
    
            NetState.prototype.toJSON = function toJSON() {
                return this.constructor.toObject(this, $protobuf.util.toJSONOptions);
            };
    
            return NetState;
        })();
    
        caffe.NetStateRule = (function() {
    
            function NetStateRule(properties) {
                this.stage = [];
                this.not_stage = [];
                if (properties)
                    for (var keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                        if (properties[keys[i]] != null)
                            this[keys[i]] = properties[keys[i]];
            }
    
            NetStateRule.prototype.phase = 0;
            NetStateRule.prototype.min_level = 0;
            NetStateRule.prototype.max_level = 0;
            NetStateRule.prototype.stage = $util.emptyArray;
            NetStateRule.prototype.not_stage = $util.emptyArray;
    
            NetStateRule.create = function create(properties) {
                return new NetStateRule(properties);
            };
    
            NetStateRule.decode = function decode(reader, length) {
                if (!(reader instanceof $Reader))
                    reader = $Reader.create(reader);
                var end = length === undefined ? reader.len : reader.pos + length, message = new $root.caffe.NetStateRule();
                while (reader.pos < end) {
                    var tag = reader.uint32();
                    switch (tag >>> 3) {
                    case 1:
                        message.phase = reader.int32();
                        break;
                    case 2:
                        message.min_level = reader.int32();
                        break;
                    case 3:
                        message.max_level = reader.int32();
                        break;
                    case 4:
                        if (!(message.stage && message.stage.length))
                            message.stage = [];
                        message.stage.push(reader.string());
                        break;
                    case 5:
                        if (!(message.not_stage && message.not_stage.length))
                            message.not_stage = [];
                        message.not_stage.push(reader.string());
                        break;
                    default:
                        reader.skipType(tag & 7);
                        break;
                    }
                }
                return message;
            };
    
            NetStateRule.decodeText = function decodeText(reader, block) {
                if (!(reader instanceof $TextReader))
                    reader = $TextReader.create(reader);
                var message = new $root.caffe.NetStateRule();
                reader.start(block);
                while (!reader.end(block)) {
                    var tag = reader.tag();
                    switch (tag) {
                    case "phase":
                        message.phase = reader.enum($root.caffe.Phase);
                        break;
                    case "min_level":
                        message.min_level = reader.int32();
                        break;
                    case "max_level":
                        message.max_level = reader.int32();
                        break;
                    case "stage":
                        if (!(message.stage && message.stage.length))
                            message.stage = [];
                        message.stage.push(reader.string());
                        break;
                    case "not_stage":
                        if (!(message.not_stage && message.not_stage.length))
                            message.not_stage = [];
                        message.not_stage.push(reader.string());
                        break;
                    default:
                        reader.handle(tag, message);
                        break;
                    }
                }
                return message;
            };
    
            NetStateRule.verify = function verify(message) {
                if (typeof message !== "object" || message === null)
                    return "object expected";
                if (message.phase != null && message.hasOwnProperty("phase"))
                    switch (message.phase) {
                    default:
                        return "phase: enum value expected";
                    case 0:
                    case 1:
                        break;
                    }
                if (message.min_level != null && message.hasOwnProperty("min_level"))
                    if (!$util.isInteger(message.min_level))
                        return "min_level: integer expected";
                if (message.max_level != null && message.hasOwnProperty("max_level"))
                    if (!$util.isInteger(message.max_level))
                        return "max_level: integer expected";
                if (message.stage != null && message.hasOwnProperty("stage")) {
                    if (!Array.isArray(message.stage))
                        return "stage: array expected";
                    for (var i = 0; i < message.stage.length; ++i)
                        if (!$util.isString(message.stage[i]))
                            return "stage: string[] expected";
                }
                if (message.not_stage != null && message.hasOwnProperty("not_stage")) {
                    if (!Array.isArray(message.not_stage))
                        return "not_stage: array expected";
                    for (var i = 0; i < message.not_stage.length; ++i)
                        if (!$util.isString(message.not_stage[i]))
                            return "not_stage: string[] expected";
                }
                return null;
            };
    
            NetStateRule.fromObject = function fromObject(object) {
                if (object instanceof $root.caffe.NetStateRule)
                    return object;
                var message = new $root.caffe.NetStateRule();
                switch (object.phase) {
                case "TRAIN":
                case 0:
                    message.phase = 0;
                    break;
                case "TEST":
                case 1:
                    message.phase = 1;
                    break;
                }
                if (object.min_level != null)
                    message.min_level = object.min_level | 0;
                if (object.max_level != null)
                    message.max_level = object.max_level | 0;
                if (object.stage) {
                    if (!Array.isArray(object.stage))
                        throw TypeError(".caffe.NetStateRule.stage: array expected");
                    message.stage = [];
                    for (var i = 0; i < object.stage.length; ++i)
                        message.stage[i] = String(object.stage[i]);
                }
                if (object.not_stage) {
                    if (!Array.isArray(object.not_stage))
                        throw TypeError(".caffe.NetStateRule.not_stage: array expected");
                    message.not_stage = [];
                    for (var i = 0; i < object.not_stage.length; ++i)
                        message.not_stage[i] = String(object.not_stage[i]);
                }
                return message;
            };
    
            NetStateRule.toObject = function toObject(message, options) {
                if (!options)
                    options = {};
                var object = {};
                if (options.arrays || options.defaults) {
                    object.stage = [];
                    object.not_stage = [];
                }
                if (options.defaults) {
                    object.phase = options.enums === String ? "TRAIN" : 0;
                    object.min_level = 0;
                    object.max_level = 0;
                }
                if (message.phase != null && message.hasOwnProperty("phase"))
                    object.phase = options.enums === String ? $root.caffe.Phase[message.phase] : message.phase;
                if (message.min_level != null && message.hasOwnProperty("min_level"))
                    object.min_level = message.min_level;
                if (message.max_level != null && message.hasOwnProperty("max_level"))
                    object.max_level = message.max_level;
                if (message.stage && message.stage.length) {
                    object.stage = [];
                    for (var j = 0; j < message.stage.length; ++j)
                        object.stage[j] = message.stage[j];
                }
                if (message.not_stage && message.not_stage.length) {
                    object.not_stage = [];
                    for (var j = 0; j < message.not_stage.length; ++j)
                        object.not_stage[j] = message.not_stage[j];
                }
                return object;
            };
    
            NetStateRule.prototype.toJSON = function toJSON() {
                return this.constructor.toObject(this, $protobuf.util.toJSONOptions);
            };
    
            return NetStateRule;
        })();
    
        caffe.ParamSpec = (function() {
    
            function ParamSpec(properties) {
                if (properties)
                    for (var keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                        if (properties[keys[i]] != null)
                            this[keys[i]] = properties[keys[i]];
            }
    
            ParamSpec.prototype.name = "";
            ParamSpec.prototype.share_mode = 0;
            ParamSpec.prototype.lr_mult = 1;
            ParamSpec.prototype.decay_mult = 1;
    
            ParamSpec.create = function create(properties) {
                return new ParamSpec(properties);
            };
    
            ParamSpec.decode = function decode(reader, length) {
                if (!(reader instanceof $Reader))
                    reader = $Reader.create(reader);
                var end = length === undefined ? reader.len : reader.pos + length, message = new $root.caffe.ParamSpec();
                while (reader.pos < end) {
                    var tag = reader.uint32();
                    switch (tag >>> 3) {
                    case 1:
                        message.name = reader.string();
                        break;
                    case 2:
                        message.share_mode = reader.int32();
                        break;
                    case 3:
                        message.lr_mult = reader.float();
                        break;
                    case 4:
                        message.decay_mult = reader.float();
                        break;
                    default:
                        reader.skipType(tag & 7);
                        break;
                    }
                }
                return message;
            };
    
            ParamSpec.decodeText = function decodeText(reader, block) {
                if (!(reader instanceof $TextReader))
                    reader = $TextReader.create(reader);
                var message = new $root.caffe.ParamSpec();
                reader.start(block);
                while (!reader.end(block)) {
                    var tag = reader.tag();
                    switch (tag) {
                    case "name":
                        message.name = reader.string();
                        break;
                    case "share_mode":
                        message.share_mode = reader.enum($root.caffe.ParamSpec.DimCheckMode);
                        break;
                    case "lr_mult":
                        message.lr_mult = reader.float();
                        break;
                    case "decay_mult":
                        message.decay_mult = reader.float();
                        break;
                    default:
                        reader.handle(tag, message);
                        break;
                    }
                }
                return message;
            };
    
            ParamSpec.verify = function verify(message) {
                if (typeof message !== "object" || message === null)
                    return "object expected";
                if (message.name != null && message.hasOwnProperty("name"))
                    if (!$util.isString(message.name))
                        return "name: string expected";
                if (message.share_mode != null && message.hasOwnProperty("share_mode"))
                    switch (message.share_mode) {
                    default:
                        return "share_mode: enum value expected";
                    case 0:
                    case 1:
                        break;
                    }
                if (message.lr_mult != null && message.hasOwnProperty("lr_mult"))
                    if (typeof message.lr_mult !== "number")
                        return "lr_mult: number expected";
                if (message.decay_mult != null && message.hasOwnProperty("decay_mult"))
                    if (typeof message.decay_mult !== "number")
                        return "decay_mult: number expected";
                return null;
            };
    
            ParamSpec.fromObject = function fromObject(object) {
                if (object instanceof $root.caffe.ParamSpec)
                    return object;
                var message = new $root.caffe.ParamSpec();
                if (object.name != null)
                    message.name = String(object.name);
                switch (object.share_mode) {
                case "STRICT":
                case 0:
                    message.share_mode = 0;
                    break;
                case "PERMISSIVE":
                case 1:
                    message.share_mode = 1;
                    break;
                }
                if (object.lr_mult != null)
                    message.lr_mult = Number(object.lr_mult);
                if (object.decay_mult != null)
                    message.decay_mult = Number(object.decay_mult);
                return message;
            };
    
            ParamSpec.toObject = function toObject(message, options) {
                if (!options)
                    options = {};
                var object = {};
                if (options.defaults) {
                    object.name = "";
                    object.share_mode = options.enums === String ? "STRICT" : 0;
                    object.lr_mult = 1;
                    object.decay_mult = 1;
                }
                if (message.name != null && message.hasOwnProperty("name"))
                    object.name = message.name;
                if (message.share_mode != null && message.hasOwnProperty("share_mode"))
                    object.share_mode = options.enums === String ? $root.caffe.ParamSpec.DimCheckMode[message.share_mode] : message.share_mode;
                if (message.lr_mult != null && message.hasOwnProperty("lr_mult"))
                    object.lr_mult = options.json && !isFinite(message.lr_mult) ? String(message.lr_mult) : message.lr_mult;
                if (message.decay_mult != null && message.hasOwnProperty("decay_mult"))
                    object.decay_mult = options.json && !isFinite(message.decay_mult) ? String(message.decay_mult) : message.decay_mult;
                return object;
            };
    
            ParamSpec.prototype.toJSON = function toJSON() {
                return this.constructor.toObject(this, $protobuf.util.toJSONOptions);
            };
    
            ParamSpec.DimCheckMode = (function() {
                var valuesById = {}, values = Object.create(valuesById);
                values[valuesById[0] = "STRICT"] = 0;
                values[valuesById[1] = "PERMISSIVE"] = 1;
                return values;
            })();
    
            return ParamSpec;
        })();
    
        caffe.LayerParameter = (function() {
    
            function LayerParameter(properties) {
                this.bottom = [];
                this.top = [];
                this.loss_weight = [];
                this.param = [];
                this.blobs = [];
                this.propagate_down = [];
                this.include = [];
                this.exclude = [];
                if (properties)
                    for (var keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                        if (properties[keys[i]] != null)
                            this[keys[i]] = properties[keys[i]];
            }
    
            LayerParameter.prototype.name = "";
            LayerParameter.prototype.type = "";
            LayerParameter.prototype.bottom = $util.emptyArray;
            LayerParameter.prototype.top = $util.emptyArray;
            LayerParameter.prototype.phase = 0;
            LayerParameter.prototype.loss_weight = $util.emptyArray;
            LayerParameter.prototype.param = $util.emptyArray;
            LayerParameter.prototype.blobs = $util.emptyArray;
            LayerParameter.prototype.propagate_down = $util.emptyArray;
            LayerParameter.prototype.include = $util.emptyArray;
            LayerParameter.prototype.exclude = $util.emptyArray;
            LayerParameter.prototype.transform_param = null;
            LayerParameter.prototype.loss_param = null;
            LayerParameter.prototype.accuracy_param = null;
            LayerParameter.prototype.argmax_param = null;
            LayerParameter.prototype.batch_norm_param = null;
            LayerParameter.prototype.bias_param = null;
            LayerParameter.prototype.clip_param = null;
            LayerParameter.prototype.concat_param = null;
            LayerParameter.prototype.contrastive_loss_param = null;
            LayerParameter.prototype.convolution_param = null;
            LayerParameter.prototype.crop_param = null;
            LayerParameter.prototype.data_param = null;
            LayerParameter.prototype.dropout_param = null;
            LayerParameter.prototype.dummy_data_param = null;
            LayerParameter.prototype.eltwise_param = null;
            LayerParameter.prototype.elu_param = null;
            LayerParameter.prototype.embed_param = null;
            LayerParameter.prototype.exp_param = null;
            LayerParameter.prototype.flatten_param = null;
            LayerParameter.prototype.hdf5_data_param = null;
            LayerParameter.prototype.hdf5_output_param = null;
            LayerParameter.prototype.hinge_loss_param = null;
            LayerParameter.prototype.image_data_param = null;
            LayerParameter.prototype.infogain_loss_param = null;
            LayerParameter.prototype.inner_product_param = null;
            LayerParameter.prototype.input_param = null;
            LayerParameter.prototype.log_param = null;
            LayerParameter.prototype.lrn_param = null;
            LayerParameter.prototype.memory_data_param = null;
            LayerParameter.prototype.mvn_param = null;
            LayerParameter.prototype.parameter_param = null;
            LayerParameter.prototype.pooling_param = null;
            LayerParameter.prototype.power_param = null;
            LayerParameter.prototype.prelu_param = null;
            LayerParameter.prototype.python_param = null;
            LayerParameter.prototype.recurrent_param = null;
            LayerParameter.prototype.reduction_param = null;
            LayerParameter.prototype.relu_param = null;
            LayerParameter.prototype.reshape_param = null;
            LayerParameter.prototype.scale_param = null;
            LayerParameter.prototype.sigmoid_param = null;
            LayerParameter.prototype.softmax_param = null;
            LayerParameter.prototype.spp_param = null;
            LayerParameter.prototype.slice_param = null;
            LayerParameter.prototype.swish_param = null;
            LayerParameter.prototype.tanh_param = null;
            LayerParameter.prototype.threshold_param = null;
            LayerParameter.prototype.tile_param = null;
            LayerParameter.prototype.window_data_param = null;
    
            LayerParameter.create = function create(properties) {
                return new LayerParameter(properties);
            };
    
            LayerParameter.decode = function decode(reader, length) {
                if (!(reader instanceof $Reader))
                    reader = $Reader.create(reader);
                var end = length === undefined ? reader.len : reader.pos + length, message = new $root.caffe.LayerParameter();
                while (reader.pos < end) {
                    var tag = reader.uint32();
                    switch (tag >>> 3) {
                    case 1:
                        message.name = reader.string();
                        break;
                    case 2:
                        message.type = reader.string();
                        break;
                    case 3:
                        if (!(message.bottom && message.bottom.length))
                            message.bottom = [];
                        message.bottom.push(reader.string());
                        break;
                    case 4:
                        if (!(message.top && message.top.length))
                            message.top = [];
                        message.top.push(reader.string());
                        break;
                    case 10:
                        message.phase = reader.int32();
                        break;
                    case 5:
                        if (!(message.loss_weight && message.loss_weight.length))
                            message.loss_weight = [];
                        if ((tag & 7) === 2) {
                            var end2 = reader.uint32() + reader.pos;
                            while (reader.pos < end2)
                                message.loss_weight.push(reader.float());
                        } else
                            message.loss_weight.push(reader.float());
                        break;
                    case 6:
                        if (!(message.param && message.param.length))
                            message.param = [];
                        message.param.push($root.caffe.ParamSpec.decode(reader, reader.uint32()));
                        break;
                    case 7:
                        if (!(message.blobs && message.blobs.length))
                            message.blobs = [];
                        message.blobs.push($root.caffe.BlobProto.decode(reader, reader.uint32()));
                        break;
                    case 11:
                        if (!(message.propagate_down && message.propagate_down.length))
                            message.propagate_down = [];
                        if ((tag & 7) === 2) {
                            var end2 = reader.uint32() + reader.pos;
                            while (reader.pos < end2)
                                message.propagate_down.push(reader.bool());
                        } else
                            message.propagate_down.push(reader.bool());
                        break;
                    case 8:
                        if (!(message.include && message.include.length))
                            message.include = [];
                        message.include.push($root.caffe.NetStateRule.decode(reader, reader.uint32()));
                        break;
                    case 9:
                        if (!(message.exclude && message.exclude.length))
                            message.exclude = [];
                        message.exclude.push($root.caffe.NetStateRule.decode(reader, reader.uint32()));
                        break;
                    case 100:
                        message.transform_param = $root.caffe.TransformationParameter.decode(reader, reader.uint32());
                        break;
                    case 101:
                        message.loss_param = $root.caffe.LossParameter.decode(reader, reader.uint32());
                        break;
                    case 102:
                        message.accuracy_param = $root.caffe.AccuracyParameter.decode(reader, reader.uint32());
                        break;
                    case 103:
                        message.argmax_param = $root.caffe.ArgMaxParameter.decode(reader, reader.uint32());
                        break;
                    case 139:
                        message.batch_norm_param = $root.caffe.BatchNormParameter.decode(reader, reader.uint32());
                        break;
                    case 141:
                        message.bias_param = $root.caffe.BiasParameter.decode(reader, reader.uint32());
                        break;
                    case 148:
                        message.clip_param = $root.caffe.ClipParameter.decode(reader, reader.uint32());
                        break;
                    case 104:
                        message.concat_param = $root.caffe.ConcatParameter.decode(reader, reader.uint32());
                        break;
                    case 105:
                        message.contrastive_loss_param = $root.caffe.ContrastiveLossParameter.decode(reader, reader.uint32());
                        break;
                    case 106:
                        message.convolution_param = $root.caffe.ConvolutionParameter.decode(reader, reader.uint32());
                        break;
                    case 144:
                        message.crop_param = $root.caffe.CropParameter.decode(reader, reader.uint32());
                        break;
                    case 107:
                        message.data_param = $root.caffe.DataParameter.decode(reader, reader.uint32());
                        break;
                    case 108:
                        message.dropout_param = $root.caffe.DropoutParameter.decode(reader, reader.uint32());
                        break;
                    case 109:
                        message.dummy_data_param = $root.caffe.DummyDataParameter.decode(reader, reader.uint32());
                        break;
                    case 110:
                        message.eltwise_param = $root.caffe.EltwiseParameter.decode(reader, reader.uint32());
                        break;
                    case 140:
                        message.elu_param = $root.caffe.ELUParameter.decode(reader, reader.uint32());
                        break;
                    case 137:
                        message.embed_param = $root.caffe.EmbedParameter.decode(reader, reader.uint32());
                        break;
                    case 111:
                        message.exp_param = $root.caffe.ExpParameter.decode(reader, reader.uint32());
                        break;
                    case 135:
                        message.flatten_param = $root.caffe.FlattenParameter.decode(reader, reader.uint32());
                        break;
                    case 112:
                        message.hdf5_data_param = $root.caffe.HDF5DataParameter.decode(reader, reader.uint32());
                        break;
                    case 113:
                        message.hdf5_output_param = $root.caffe.HDF5OutputParameter.decode(reader, reader.uint32());
                        break;
                    case 114:
                        message.hinge_loss_param = $root.caffe.HingeLossParameter.decode(reader, reader.uint32());
                        break;
                    case 115:
                        message.image_data_param = $root.caffe.ImageDataParameter.decode(reader, reader.uint32());
                        break;
                    case 116:
                        message.infogain_loss_param = $root.caffe.InfogainLossParameter.decode(reader, reader.uint32());
                        break;
                    case 117:
                        message.inner_product_param = $root.caffe.InnerProductParameter.decode(reader, reader.uint32());
                        break;
                    case 143:
                        message.input_param = $root.caffe.InputParameter.decode(reader, reader.uint32());
                        break;
                    case 134:
                        message.log_param = $root.caffe.LogParameter.decode(reader, reader.uint32());
                        break;
                    case 118:
                        message.lrn_param = $root.caffe.LRNParameter.decode(reader, reader.uint32());
                        break;
                    case 119:
                        message.memory_data_param = $root.caffe.MemoryDataParameter.decode(reader, reader.uint32());
                        break;
                    case 120:
                        message.mvn_param = $root.caffe.MVNParameter.decode(reader, reader.uint32());
                        break;
                    case 145:
                        message.parameter_param = $root.caffe.ParameterParameter.decode(reader, reader.uint32());
                        break;
                    case 121:
                        message.pooling_param = $root.caffe.PoolingParameter.decode(reader, reader.uint32());
                        break;
                    case 122:
                        message.power_param = $root.caffe.PowerParameter.decode(reader, reader.uint32());
                        break;
                    case 131:
                        message.prelu_param = $root.caffe.PReLUParameter.decode(reader, reader.uint32());
                        break;
                    case 130:
                        message.python_param = $root.caffe.PythonParameter.decode(reader, reader.uint32());
                        break;
                    case 146:
                        message.recurrent_param = $root.caffe.RecurrentParameter.decode(reader, reader.uint32());
                        break;
                    case 136:
                        message.reduction_param = $root.caffe.ReductionParameter.decode(reader, reader.uint32());
                        break;
                    case 123:
                        message.relu_param = $root.caffe.ReLUParameter.decode(reader, reader.uint32());
                        break;
                    case 133:
                        message.reshape_param = $root.caffe.ReshapeParameter.decode(reader, reader.uint32());
                        break;
                    case 142:
                        message.scale_param = $root.caffe.ScaleParameter.decode(reader, reader.uint32());
                        break;
                    case 124:
                        message.sigmoid_param = $root.caffe.SigmoidParameter.decode(reader, reader.uint32());
                        break;
                    case 125:
                        message.softmax_param = $root.caffe.SoftmaxParameter.decode(reader, reader.uint32());
                        break;
                    case 132:
                        message.spp_param = $root.caffe.SPPParameter.decode(reader, reader.uint32());
                        break;
                    case 126:
                        message.slice_param = $root.caffe.SliceParameter.decode(reader, reader.uint32());
                        break;
                    case 147:
                        message.swish_param = $root.caffe.SwishParameter.decode(reader, reader.uint32());
                        break;
                    case 127:
                        message.tanh_param = $root.caffe.TanHParameter.decode(reader, reader.uint32());
                        break;
                    case 128:
                        message.threshold_param = $root.caffe.ThresholdParameter.decode(reader, reader.uint32());
                        break;
                    case 138:
                        message.tile_param = $root.caffe.TileParameter.decode(reader, reader.uint32());
                        break;
                    case 129:
                        message.window_data_param = $root.caffe.WindowDataParameter.decode(reader, reader.uint32());
                        break;
                    default:
                        reader.skipType(tag & 7);
                        break;
                    }
                }
                return message;
            };
    
            LayerParameter.decodeText = function decodeText(reader, block) {
                if (!(reader instanceof $TextReader))
                    reader = $TextReader.create(reader);
                var message = new $root.caffe.LayerParameter();
                reader.start(block);
                while (!reader.end(block)) {
                    var tag = reader.tag();
                    switch (tag) {
                    case "name":
                        message.name = reader.string();
                        break;
                    case "type":
                        message.type = reader.string();
                        break;
                    case "bottom":
                        if (!(message.bottom && message.bottom.length))
                            message.bottom = [];
                        message.bottom.push(reader.string());
                        break;
                    case "top":
                        if (!(message.top && message.top.length))
                            message.top = [];
                        message.top.push(reader.string());
                        break;
                    case "phase":
                        message.phase = reader.enum($root.caffe.Phase);
                        break;
                    case "loss_weight":
                        if (!(message.loss_weight && message.loss_weight.length))
                            message.loss_weight = [];
                        message.loss_weight.push(reader.float());
                        break;
                    case "param":
                        if (!(message.param && message.param.length))
                            message.param = [];
                        message.param.push($root.caffe.ParamSpec.decodeText(reader, true));
                        break;
                    case "blobs":
                        if (!(message.blobs && message.blobs.length))
                            message.blobs = [];
                        message.blobs.push($root.caffe.BlobProto.decodeText(reader, true));
                        break;
                    case "propagate_down":
                        if (!(message.propagate_down && message.propagate_down.length))
                            message.propagate_down = [];
                        message.propagate_down.push(reader.bool());
                        break;
                    case "include":
                        if (!(message.include && message.include.length))
                            message.include = [];
                        message.include.push($root.caffe.NetStateRule.decodeText(reader, true));
                        break;
                    case "exclude":
                        if (!(message.exclude && message.exclude.length))
                            message.exclude = [];
                        message.exclude.push($root.caffe.NetStateRule.decodeText(reader, true));
                        break;
                    case "transform_param":
                        message.transform_param = $root.caffe.TransformationParameter.decodeText(reader, true);
                        break;
                    case "loss_param":
                        message.loss_param = $root.caffe.LossParameter.decodeText(reader, true);
                        break;
                    case "accuracy_param":
                        message.accuracy_param = $root.caffe.AccuracyParameter.decodeText(reader, true);
                        break;
                    case "argmax_param":
                        message.argmax_param = $root.caffe.ArgMaxParameter.decodeText(reader, true);
                        break;
                    case "batch_norm_param":
                        message.batch_norm_param = $root.caffe.BatchNormParameter.decodeText(reader, true);
                        break;
                    case "bias_param":
                        message.bias_param = $root.caffe.BiasParameter.decodeText(reader, true);
                        break;
                    case "clip_param":
                        message.clip_param = $root.caffe.ClipParameter.decodeText(reader, true);
                        break;
                    case "concat_param":
                        message.concat_param = $root.caffe.ConcatParameter.decodeText(reader, true);
                        break;
                    case "contrastive_loss_param":
                        message.contrastive_loss_param = $root.caffe.ContrastiveLossParameter.decodeText(reader, true);
                        break;
                    case "convolution_param":
                        message.convolution_param = $root.caffe.ConvolutionParameter.decodeText(reader, true);
                        break;
                    case "crop_param":
                        message.crop_param = $root.caffe.CropParameter.decodeText(reader, true);
                        break;
                    case "data_param":
                        message.data_param = $root.caffe.DataParameter.decodeText(reader, true);
                        break;
                    case "dropout_param":
                        message.dropout_param = $root.caffe.DropoutParameter.decodeText(reader, true);
                        break;
                    case "dummy_data_param":
                        message.dummy_data_param = $root.caffe.DummyDataParameter.decodeText(reader, true);
                        break;
                    case "eltwise_param":
                        message.eltwise_param = $root.caffe.EltwiseParameter.decodeText(reader, true);
                        break;
                    case "elu_param":
                        message.elu_param = $root.caffe.ELUParameter.decodeText(reader, true);
                        break;
                    case "embed_param":
                        message.embed_param = $root.caffe.EmbedParameter.decodeText(reader, true);
                        break;
                    case "exp_param":
                        message.exp_param = $root.caffe.ExpParameter.decodeText(reader, true);
                        break;
                    case "flatten_param":
                        message.flatten_param = $root.caffe.FlattenParameter.decodeText(reader, true);
                        break;
                    case "hdf5_data_param":
                        message.hdf5_data_param = $root.caffe.HDF5DataParameter.decodeText(reader, true);
                        break;
                    case "hdf5_output_param":
                        message.hdf5_output_param = $root.caffe.HDF5OutputParameter.decodeText(reader, true);
                        break;
                    case "hinge_loss_param":
                        message.hinge_loss_param = $root.caffe.HingeLossParameter.decodeText(reader, true);
                        break;
                    case "image_data_param":
                        message.image_data_param = $root.caffe.ImageDataParameter.decodeText(reader, true);
                        break;
                    case "infogain_loss_param":
                        message.infogain_loss_param = $root.caffe.InfogainLossParameter.decodeText(reader, true);
                        break;
                    case "inner_product_param":
                        message.inner_product_param = $root.caffe.InnerProductParameter.decodeText(reader, true);
                        break;
                    case "input_param":
                        message.input_param = $root.caffe.InputParameter.decodeText(reader, true);
                        break;
                    case "log_param":
                        message.log_param = $root.caffe.LogParameter.decodeText(reader, true);
                        break;
                    case "lrn_param":
                        message.lrn_param = $root.caffe.LRNParameter.decodeText(reader, true);
                        break;
                    case "memory_data_param":
                        message.memory_data_param = $root.caffe.MemoryDataParameter.decodeText(reader, true);
                        break;
                    case "mvn_param":
                        message.mvn_param = $root.caffe.MVNParameter.decodeText(reader, true);
                        break;
                    case "parameter_param":
                        message.parameter_param = $root.caffe.ParameterParameter.decodeText(reader, true);
                        break;
                    case "pooling_param":
                        message.pooling_param = $root.caffe.PoolingParameter.decodeText(reader, true);
                        break;
                    case "power_param":
                        message.power_param = $root.caffe.PowerParameter.decodeText(reader, true);
                        break;
                    case "prelu_param":
                        message.prelu_param = $root.caffe.PReLUParameter.decodeText(reader, true);
                        break;
                    case "python_param":
                        message.python_param = $root.caffe.PythonParameter.decodeText(reader, true);
                        break;
                    case "recurrent_param":
                        message.recurrent_param = $root.caffe.RecurrentParameter.decodeText(reader, true);
                        break;
                    case "reduction_param":
                        message.reduction_param = $root.caffe.ReductionParameter.decodeText(reader, true);
                        break;
                    case "relu_param":
                        message.relu_param = $root.caffe.ReLUParameter.decodeText(reader, true);
                        break;
                    case "reshape_param":
                        message.reshape_param = $root.caffe.ReshapeParameter.decodeText(reader, true);
                        break;
                    case "scale_param":
                        message.scale_param = $root.caffe.ScaleParameter.decodeText(reader, true);
                        break;
                    case "sigmoid_param":
                        message.sigmoid_param = $root.caffe.SigmoidParameter.decodeText(reader, true);
                        break;
                    case "softmax_param":
                        message.softmax_param = $root.caffe.SoftmaxParameter.decodeText(reader, true);
                        break;
                    case "spp_param":
                        message.spp_param = $root.caffe.SPPParameter.decodeText(reader, true);
                        break;
                    case "slice_param":
                        message.slice_param = $root.caffe.SliceParameter.decodeText(reader, true);
                        break;
                    case "swish_param":
                        message.swish_param = $root.caffe.SwishParameter.decodeText(reader, true);
                        break;
                    case "tanh_param":
                        message.tanh_param = $root.caffe.TanHParameter.decodeText(reader, true);
                        break;
                    case "threshold_param":
                        message.threshold_param = $root.caffe.ThresholdParameter.decodeText(reader, true);
                        break;
                    case "tile_param":
                        message.tile_param = $root.caffe.TileParameter.decodeText(reader, true);
                        break;
                    case "window_data_param":
                        message.window_data_param = $root.caffe.WindowDataParameter.decodeText(reader, true);
                        break;
                    default:
                        reader.handle(tag, message);
                        break;
                    }
                }
                return message;
            };
    
            LayerParameter.verify = function verify(message) {
                if (typeof message !== "object" || message === null)
                    return "object expected";
                if (message.name != null && message.hasOwnProperty("name"))
                    if (!$util.isString(message.name))
                        return "name: string expected";
                if (message.type != null && message.hasOwnProperty("type"))
                    if (!$util.isString(message.type))
                        return "type: string expected";
                if (message.bottom != null && message.hasOwnProperty("bottom")) {
                    if (!Array.isArray(message.bottom))
                        return "bottom: array expected";
                    for (var i = 0; i < message.bottom.length; ++i)
                        if (!$util.isString(message.bottom[i]))
                            return "bottom: string[] expected";
                }
                if (message.top != null && message.hasOwnProperty("top")) {
                    if (!Array.isArray(message.top))
                        return "top: array expected";
                    for (var i = 0; i < message.top.length; ++i)
                        if (!$util.isString(message.top[i]))
                            return "top: string[] expected";
                }
                if (message.phase != null && message.hasOwnProperty("phase"))
                    switch (message.phase) {
                    default:
                        return "phase: enum value expected";
                    case 0:
                    case 1:
                        break;
                    }
                if (message.loss_weight != null && message.hasOwnProperty("loss_weight")) {
                    if (!Array.isArray(message.loss_weight))
                        return "loss_weight: array expected";
                    for (var i = 0; i < message.loss_weight.length; ++i)
                        if (typeof message.loss_weight[i] !== "number")
                            return "loss_weight: number[] expected";
                }
                if (message.param != null && message.hasOwnProperty("param")) {
                    if (!Array.isArray(message.param))
                        return "param: array expected";
                    for (var i = 0; i < message.param.length; ++i) {
                        var error = $root.caffe.ParamSpec.verify(message.param[i]);
                        if (error)
                            return "param." + error;
                    }
                }
                if (message.blobs != null && message.hasOwnProperty("blobs")) {
                    if (!Array.isArray(message.blobs))
                        return "blobs: array expected";
                    for (var i = 0; i < message.blobs.length; ++i) {
                        var error = $root.caffe.BlobProto.verify(message.blobs[i]);
                        if (error)
                            return "blobs." + error;
                    }
                }
                if (message.propagate_down != null && message.hasOwnProperty("propagate_down")) {
                    if (!Array.isArray(message.propagate_down))
                        return "propagate_down: array expected";
                    for (var i = 0; i < message.propagate_down.length; ++i)
                        if (typeof message.propagate_down[i] !== "boolean")
                            return "propagate_down: boolean[] expected";
                }
                if (message.include != null && message.hasOwnProperty("include")) {
                    if (!Array.isArray(message.include))
                        return "include: array expected";
                    for (var i = 0; i < message.include.length; ++i) {
                        var error = $root.caffe.NetStateRule.verify(message.include[i]);
                        if (error)
                            return "include." + error;
                    }
                }
                if (message.exclude != null && message.hasOwnProperty("exclude")) {
                    if (!Array.isArray(message.exclude))
                        return "exclude: array expected";
                    for (var i = 0; i < message.exclude.length; ++i) {
                        var error = $root.caffe.NetStateRule.verify(message.exclude[i]);
                        if (error)
                            return "exclude." + error;
                    }
                }
                if (message.transform_param != null && message.hasOwnProperty("transform_param")) {
                    var error = $root.caffe.TransformationParameter.verify(message.transform_param);
                    if (error)
                        return "transform_param." + error;
                }
                if (message.loss_param != null && message.hasOwnProperty("loss_param")) {
                    var error = $root.caffe.LossParameter.verify(message.loss_param);
                    if (error)
                        return "loss_param." + error;
                }
                if (message.accuracy_param != null && message.hasOwnProperty("accuracy_param")) {
                    var error = $root.caffe.AccuracyParameter.verify(message.accuracy_param);
                    if (error)
                        return "accuracy_param." + error;
                }
                if (message.argmax_param != null && message.hasOwnProperty("argmax_param")) {
                    var error = $root.caffe.ArgMaxParameter.verify(message.argmax_param);
                    if (error)
                        return "argmax_param." + error;
                }
                if (message.batch_norm_param != null && message.hasOwnProperty("batch_norm_param")) {
                    var error = $root.caffe.BatchNormParameter.verify(message.batch_norm_param);
                    if (error)
                        return "batch_norm_param." + error;
                }
                if (message.bias_param != null && message.hasOwnProperty("bias_param")) {
                    var error = $root.caffe.BiasParameter.verify(message.bias_param);
                    if (error)
                        return "bias_param." + error;
                }
                if (message.clip_param != null && message.hasOwnProperty("clip_param")) {
                    var error = $root.caffe.ClipParameter.verify(message.clip_param);
                    if (error)
                        return "clip_param." + error;
                }
                if (message.concat_param != null && message.hasOwnProperty("concat_param")) {
                    var error = $root.caffe.ConcatParameter.verify(message.concat_param);
                    if (error)
                        return "concat_param." + error;
                }
                if (message.contrastive_loss_param != null && message.hasOwnProperty("contrastive_loss_param")) {
                    var error = $root.caffe.ContrastiveLossParameter.verify(message.contrastive_loss_param);
                    if (error)
                        return "contrastive_loss_param." + error;
                }
                if (message.convolution_param != null && message.hasOwnProperty("convolution_param")) {
                    var error = $root.caffe.ConvolutionParameter.verify(message.convolution_param);
                    if (error)
                        return "convolution_param." + error;
                }
                if (message.crop_param != null && message.hasOwnProperty("crop_param")) {
                    var error = $root.caffe.CropParameter.verify(message.crop_param);
                    if (error)
                        return "crop_param." + error;
                }
                if (message.data_param != null && message.hasOwnProperty("data_param")) {
                    var error = $root.caffe.DataParameter.verify(message.data_param);
                    if (error)
                        return "data_param." + error;
                }
                if (message.dropout_param != null && message.hasOwnProperty("dropout_param")) {
                    var error = $root.caffe.DropoutParameter.verify(message.dropout_param);
                    if (error)
                        return "dropout_param." + error;
                }
                if (message.dummy_data_param != null && message.hasOwnProperty("dummy_data_param")) {
                    var error = $root.caffe.DummyDataParameter.verify(message.dummy_data_param);
                    if (error)
                        return "dummy_data_param." + error;
                }
                if (message.eltwise_param != null && message.hasOwnProperty("eltwise_param")) {
                    var error = $root.caffe.EltwiseParameter.verify(message.eltwise_param);
                    if (error)
                        return "eltwise_param." + error;
                }
                if (message.elu_param != null && message.hasOwnProperty("elu_param")) {
                    var error = $root.caffe.ELUParameter.verify(message.elu_param);
                    if (error)
                        return "elu_param." + error;
                }
                if (message.embed_param != null && message.hasOwnProperty("embed_param")) {
                    var error = $root.caffe.EmbedParameter.verify(message.embed_param);
                    if (error)
                        return "embed_param." + error;
                }
                if (message.exp_param != null && message.hasOwnProperty("exp_param")) {
                    var error = $root.caffe.ExpParameter.verify(message.exp_param);
                    if (error)
                        return "exp_param." + error;
                }
                if (message.flatten_param != null && message.hasOwnProperty("flatten_param")) {
                    var error = $root.caffe.FlattenParameter.verify(message.flatten_param);
                    if (error)
                        return "flatten_param." + error;
                }
                if (message.hdf5_data_param != null && message.hasOwnProperty("hdf5_data_param")) {
                    var error = $root.caffe.HDF5DataParameter.verify(message.hdf5_data_param);
                    if (error)
                        return "hdf5_data_param." + error;
                }
                if (message.hdf5_output_param != null && message.hasOwnProperty("hdf5_output_param")) {
                    var error = $root.caffe.HDF5OutputParameter.verify(message.hdf5_output_param);
                    if (error)
                        return "hdf5_output_param." + error;
                }
                if (message.hinge_loss_param != null && message.hasOwnProperty("hinge_loss_param")) {
                    var error = $root.caffe.HingeLossParameter.verify(message.hinge_loss_param);
                    if (error)
                        return "hinge_loss_param." + error;
                }
                if (message.image_data_param != null && message.hasOwnProperty("image_data_param")) {
                    var error = $root.caffe.ImageDataParameter.verify(message.image_data_param);
                    if (error)
                        return "image_data_param." + error;
                }
                if (message.infogain_loss_param != null && message.hasOwnProperty("infogain_loss_param")) {
                    var error = $root.caffe.InfogainLossParameter.verify(message.infogain_loss_param);
                    if (error)
                        return "infogain_loss_param." + error;
                }
                if (message.inner_product_param != null && message.hasOwnProperty("inner_product_param")) {
                    var error = $root.caffe.InnerProductParameter.verify(message.inner_product_param);
                    if (error)
                        return "inner_product_param." + error;
                }
                if (message.input_param != null && message.hasOwnProperty("input_param")) {
                    var error = $root.caffe.InputParameter.verify(message.input_param);
                    if (error)
                        return "input_param." + error;
                }
                if (message.log_param != null && message.hasOwnProperty("log_param")) {
                    var error = $root.caffe.LogParameter.verify(message.log_param);
                    if (error)
                        return "log_param." + error;
                }
                if (message.lrn_param != null && message.hasOwnProperty("lrn_param")) {
                    var error = $root.caffe.LRNParameter.verify(message.lrn_param);
                    if (error)
                        return "lrn_param." + error;
                }
                if (message.memory_data_param != null && message.hasOwnProperty("memory_data_param")) {
                    var error = $root.caffe.MemoryDataParameter.verify(message.memory_data_param);
                    if (error)
                        return "memory_data_param." + error;
                }
                if (message.mvn_param != null && message.hasOwnProperty("mvn_param")) {
                    var error = $root.caffe.MVNParameter.verify(message.mvn_param);
                    if (error)
                        return "mvn_param." + error;
                }
                if (message.parameter_param != null && message.hasOwnProperty("parameter_param")) {
                    var error = $root.caffe.ParameterParameter.verify(message.parameter_param);
                    if (error)
                        return "parameter_param." + error;
                }
                if (message.pooling_param != null && message.hasOwnProperty("pooling_param")) {
                    var error = $root.caffe.PoolingParameter.verify(message.pooling_param);
                    if (error)
                        return "pooling_param." + error;
                }
                if (message.power_param != null && message.hasOwnProperty("power_param")) {
                    var error = $root.caffe.PowerParameter.verify(message.power_param);
                    if (error)
                        return "power_param." + error;
                }
                if (message.prelu_param != null && message.hasOwnProperty("prelu_param")) {
                    var error = $root.caffe.PReLUParameter.verify(message.prelu_param);
                    if (error)
                        return "prelu_param." + error;
                }
                if (message.python_param != null && message.hasOwnProperty("python_param")) {
                    var error = $root.caffe.PythonParameter.verify(message.python_param);
                    if (error)
                        return "python_param." + error;
                }
                if (message.recurrent_param != null && message.hasOwnProperty("recurrent_param")) {
                    var error = $root.caffe.RecurrentParameter.verify(message.recurrent_param);
                    if (error)
                        return "recurrent_param." + error;
                }
                if (message.reduction_param != null && message.hasOwnProperty("reduction_param")) {
                    var error = $root.caffe.ReductionParameter.verify(message.reduction_param);
                    if (error)
                        return "reduction_param." + error;
                }
                if (message.relu_param != null && message.hasOwnProperty("relu_param")) {
                    var error = $root.caffe.ReLUParameter.verify(message.relu_param);
                    if (error)
                        return "relu_param." + error;
                }
                if (message.reshape_param != null && message.hasOwnProperty("reshape_param")) {
                    var error = $root.caffe.ReshapeParameter.verify(message.reshape_param);
                    if (error)
                        return "reshape_param." + error;
                }
                if (message.scale_param != null && message.hasOwnProperty("scale_param")) {
                    var error = $root.caffe.ScaleParameter.verify(message.scale_param);
                    if (error)
                        return "scale_param." + error;
                }
                if (message.sigmoid_param != null && message.hasOwnProperty("sigmoid_param")) {
                    var error = $root.caffe.SigmoidParameter.verify(message.sigmoid_param);
                    if (error)
                        return "sigmoid_param." + error;
                }
                if (message.softmax_param != null && message.hasOwnProperty("softmax_param")) {
                    var error = $root.caffe.SoftmaxParameter.verify(message.softmax_param);
                    if (error)
                        return "softmax_param." + error;
                }
                if (message.spp_param != null && message.hasOwnProperty("spp_param")) {
                    var error = $root.caffe.SPPParameter.verify(message.spp_param);
                    if (error)
                        return "spp_param." + error;
                }
                if (message.slice_param != null && message.hasOwnProperty("slice_param")) {
                    var error = $root.caffe.SliceParameter.verify(message.slice_param);
                    if (error)
                        return "slice_param." + error;
                }
                if (message.swish_param != null && message.hasOwnProperty("swish_param")) {
                    var error = $root.caffe.SwishParameter.verify(message.swish_param);
                    if (error)
                        return "swish_param." + error;
                }
                if (message.tanh_param != null && message.hasOwnProperty("tanh_param")) {
                    var error = $root.caffe.TanHParameter.verify(message.tanh_param);
                    if (error)
                        return "tanh_param." + error;
                }
                if (message.threshold_param != null && message.hasOwnProperty("threshold_param")) {
                    var error = $root.caffe.ThresholdParameter.verify(message.threshold_param);
                    if (error)
                        return "threshold_param." + error;
                }
                if (message.tile_param != null && message.hasOwnProperty("tile_param")) {
                    var error = $root.caffe.TileParameter.verify(message.tile_param);
                    if (error)
                        return "tile_param." + error;
                }
                if (message.window_data_param != null && message.hasOwnProperty("window_data_param")) {
                    var error = $root.caffe.WindowDataParameter.verify(message.window_data_param);
                    if (error)
                        return "window_data_param." + error;
                }
                return null;
            };
    
            LayerParameter.fromObject = function fromObject(object) {
                if (object instanceof $root.caffe.LayerParameter)
                    return object;
                var message = new $root.caffe.LayerParameter();
                if (object.name != null)
                    message.name = String(object.name);
                if (object.type != null)
                    message.type = String(object.type);
                if (object.bottom) {
                    if (!Array.isArray(object.bottom))
                        throw TypeError(".caffe.LayerParameter.bottom: array expected");
                    message.bottom = [];
                    for (var i = 0; i < object.bottom.length; ++i)
                        message.bottom[i] = String(object.bottom[i]);
                }
                if (object.top) {
                    if (!Array.isArray(object.top))
                        throw TypeError(".caffe.LayerParameter.top: array expected");
                    message.top = [];
                    for (var i = 0; i < object.top.length; ++i)
                        message.top[i] = String(object.top[i]);
                }
                switch (object.phase) {
                case "TRAIN":
                case 0:
                    message.phase = 0;
                    break;
                case "TEST":
                case 1:
                    message.phase = 1;
                    break;
                }
                if (object.loss_weight) {
                    if (!Array.isArray(object.loss_weight))
                        throw TypeError(".caffe.LayerParameter.loss_weight: array expected");
                    message.loss_weight = [];
                    for (var i = 0; i < object.loss_weight.length; ++i)
                        message.loss_weight[i] = Number(object.loss_weight[i]);
                }
                if (object.param) {
                    if (!Array.isArray(object.param))
                        throw TypeError(".caffe.LayerParameter.param: array expected");
                    message.param = [];
                    for (var i = 0; i < object.param.length; ++i) {
                        if (typeof object.param[i] !== "object")
                            throw TypeError(".caffe.LayerParameter.param: object expected");
                        message.param[i] = $root.caffe.ParamSpec.fromObject(object.param[i]);
                    }
                }
                if (object.blobs) {
                    if (!Array.isArray(object.blobs))
                        throw TypeError(".caffe.LayerParameter.blobs: array expected");
                    message.blobs = [];
                    for (var i = 0; i < object.blobs.length; ++i) {
                        if (typeof object.blobs[i] !== "object")
                            throw TypeError(".caffe.LayerParameter.blobs: object expected");
                        message.blobs[i] = $root.caffe.BlobProto.fromObject(object.blobs[i]);
                    }
                }
                if (object.propagate_down) {
                    if (!Array.isArray(object.propagate_down))
                        throw TypeError(".caffe.LayerParameter.propagate_down: array expected");
                    message.propagate_down = [];
                    for (var i = 0; i < object.propagate_down.length; ++i)
                        message.propagate_down[i] = Boolean(object.propagate_down[i]);
                }
                if (object.include) {
                    if (!Array.isArray(object.include))
                        throw TypeError(".caffe.LayerParameter.include: array expected");
                    message.include = [];
                    for (var i = 0; i < object.include.length; ++i) {
                        if (typeof object.include[i] !== "object")
                            throw TypeError(".caffe.LayerParameter.include: object expected");
                        message.include[i] = $root.caffe.NetStateRule.fromObject(object.include[i]);
                    }
                }
                if (object.exclude) {
                    if (!Array.isArray(object.exclude))
                        throw TypeError(".caffe.LayerParameter.exclude: array expected");
                    message.exclude = [];
                    for (var i = 0; i < object.exclude.length; ++i) {
                        if (typeof object.exclude[i] !== "object")
                            throw TypeError(".caffe.LayerParameter.exclude: object expected");
                        message.exclude[i] = $root.caffe.NetStateRule.fromObject(object.exclude[i]);
                    }
                }
                if (object.transform_param != null) {
                    if (typeof object.transform_param !== "object")
                        throw TypeError(".caffe.LayerParameter.transform_param: object expected");
                    message.transform_param = $root.caffe.TransformationParameter.fromObject(object.transform_param);
                }
                if (object.loss_param != null) {
                    if (typeof object.loss_param !== "object")
                        throw TypeError(".caffe.LayerParameter.loss_param: object expected");
                    message.loss_param = $root.caffe.LossParameter.fromObject(object.loss_param);
                }
                if (object.accuracy_param != null) {
                    if (typeof object.accuracy_param !== "object")
                        throw TypeError(".caffe.LayerParameter.accuracy_param: object expected");
                    message.accuracy_param = $root.caffe.AccuracyParameter.fromObject(object.accuracy_param);
                }
                if (object.argmax_param != null) {
                    if (typeof object.argmax_param !== "object")
                        throw TypeError(".caffe.LayerParameter.argmax_param: object expected");
                    message.argmax_param = $root.caffe.ArgMaxParameter.fromObject(object.argmax_param);
                }
                if (object.batch_norm_param != null) {
                    if (typeof object.batch_norm_param !== "object")
                        throw TypeError(".caffe.LayerParameter.batch_norm_param: object expected");
                    message.batch_norm_param = $root.caffe.BatchNormParameter.fromObject(object.batch_norm_param);
                }
                if (object.bias_param != null) {
                    if (typeof object.bias_param !== "object")
                        throw TypeError(".caffe.LayerParameter.bias_param: object expected");
                    message.bias_param = $root.caffe.BiasParameter.fromObject(object.bias_param);
                }
                if (object.clip_param != null) {
                    if (typeof object.clip_param !== "object")
                        throw TypeError(".caffe.LayerParameter.clip_param: object expected");
                    message.clip_param = $root.caffe.ClipParameter.fromObject(object.clip_param);
                }
                if (object.concat_param != null) {
                    if (typeof object.concat_param !== "object")
                        throw TypeError(".caffe.LayerParameter.concat_param: object expected");
                    message.concat_param = $root.caffe.ConcatParameter.fromObject(object.concat_param);
                }
                if (object.contrastive_loss_param != null) {
                    if (typeof object.contrastive_loss_param !== "object")
                        throw TypeError(".caffe.LayerParameter.contrastive_loss_param: object expected");
                    message.contrastive_loss_param = $root.caffe.ContrastiveLossParameter.fromObject(object.contrastive_loss_param);
                }
                if (object.convolution_param != null) {
                    if (typeof object.convolution_param !== "object")
                        throw TypeError(".caffe.LayerParameter.convolution_param: object expected");
                    message.convolution_param = $root.caffe.ConvolutionParameter.fromObject(object.convolution_param);
                }
                if (object.crop_param != null) {
                    if (typeof object.crop_param !== "object")
                        throw TypeError(".caffe.LayerParameter.crop_param: object expected");
                    message.crop_param = $root.caffe.CropParameter.fromObject(object.crop_param);
                }
                if (object.data_param != null) {
                    if (typeof object.data_param !== "object")
                        throw TypeError(".caffe.LayerParameter.data_param: object expected");
                    message.data_param = $root.caffe.DataParameter.fromObject(object.data_param);
                }
                if (object.dropout_param != null) {
                    if (typeof object.dropout_param !== "object")
                        throw TypeError(".caffe.LayerParameter.dropout_param: object expected");
                    message.dropout_param = $root.caffe.DropoutParameter.fromObject(object.dropout_param);
                }
                if (object.dummy_data_param != null) {
                    if (typeof object.dummy_data_param !== "object")
                        throw TypeError(".caffe.LayerParameter.dummy_data_param: object expected");
                    message.dummy_data_param = $root.caffe.DummyDataParameter.fromObject(object.dummy_data_param);
                }
                if (object.eltwise_param != null) {
                    if (typeof object.eltwise_param !== "object")
                        throw TypeError(".caffe.LayerParameter.eltwise_param: object expected");
                    message.eltwise_param = $root.caffe.EltwiseParameter.fromObject(object.eltwise_param);
                }
                if (object.elu_param != null) {
                    if (typeof object.elu_param !== "object")
                        throw TypeError(".caffe.LayerParameter.elu_param: object expected");
                    message.elu_param = $root.caffe.ELUParameter.fromObject(object.elu_param);
                }
                if (object.embed_param != null) {
                    if (typeof object.embed_param !== "object")
                        throw TypeError(".caffe.LayerParameter.embed_param: object expected");
                    message.embed_param = $root.caffe.EmbedParameter.fromObject(object.embed_param);
                }
                if (object.exp_param != null) {
                    if (typeof object.exp_param !== "object")
                        throw TypeError(".caffe.LayerParameter.exp_param: object expected");
                    message.exp_param = $root.caffe.ExpParameter.fromObject(object.exp_param);
                }
                if (object.flatten_param != null) {
                    if (typeof object.flatten_param !== "object")
                        throw TypeError(".caffe.LayerParameter.flatten_param: object expected");
                    message.flatten_param = $root.caffe.FlattenParameter.fromObject(object.flatten_param);
                }
                if (object.hdf5_data_param != null) {
                    if (typeof object.hdf5_data_param !== "object")
                        throw TypeError(".caffe.LayerParameter.hdf5_data_param: object expected");
                    message.hdf5_data_param = $root.caffe.HDF5DataParameter.fromObject(object.hdf5_data_param);
                }
                if (object.hdf5_output_param != null) {
                    if (typeof object.hdf5_output_param !== "object")
                        throw TypeError(".caffe.LayerParameter.hdf5_output_param: object expected");
                    message.hdf5_output_param = $root.caffe.HDF5OutputParameter.fromObject(object.hdf5_output_param);
                }
                if (object.hinge_loss_param != null) {
                    if (typeof object.hinge_loss_param !== "object")
                        throw TypeError(".caffe.LayerParameter.hinge_loss_param: object expected");
                    message.hinge_loss_param = $root.caffe.HingeLossParameter.fromObject(object.hinge_loss_param);
                }
                if (object.image_data_param != null) {
                    if (typeof object.image_data_param !== "object")
                        throw TypeError(".caffe.LayerParameter.image_data_param: object expected");
                    message.image_data_param = $root.caffe.ImageDataParameter.fromObject(object.image_data_param);
                }
                if (object.infogain_loss_param != null) {
                    if (typeof object.infogain_loss_param !== "object")
                        throw TypeError(".caffe.LayerParameter.infogain_loss_param: object expected");
                    message.infogain_loss_param = $root.caffe.InfogainLossParameter.fromObject(object.infogain_loss_param);
                }
                if (object.inner_product_param != null) {
                    if (typeof object.inner_product_param !== "object")
                        throw TypeError(".caffe.LayerParameter.inner_product_param: object expected");
                    message.inner_product_param = $root.caffe.InnerProductParameter.fromObject(object.inner_product_param);
                }
                if (object.input_param != null) {
                    if (typeof object.input_param !== "object")
                        throw TypeError(".caffe.LayerParameter.input_param: object expected");
                    message.input_param = $root.caffe.InputParameter.fromObject(object.input_param);
                }
                if (object.log_param != null) {
                    if (typeof object.log_param !== "object")
                        throw TypeError(".caffe.LayerParameter.log_param: object expected");
                    message.log_param = $root.caffe.LogParameter.fromObject(object.log_param);
                }
                if (object.lrn_param != null) {
                    if (typeof object.lrn_param !== "object")
                        throw TypeError(".caffe.LayerParameter.lrn_param: object expected");
                    message.lrn_param = $root.caffe.LRNParameter.fromObject(object.lrn_param);
                }
                if (object.memory_data_param != null) {
                    if (typeof object.memory_data_param !== "object")
                        throw TypeError(".caffe.LayerParameter.memory_data_param: object expected");
                    message.memory_data_param = $root.caffe.MemoryDataParameter.fromObject(object.memory_data_param);
                }
                if (object.mvn_param != null) {
                    if (typeof object.mvn_param !== "object")
                        throw TypeError(".caffe.LayerParameter.mvn_param: object expected");
                    message.mvn_param = $root.caffe.MVNParameter.fromObject(object.mvn_param);
                }
                if (object.parameter_param != null) {
                    if (typeof object.parameter_param !== "object")
                        throw TypeError(".caffe.LayerParameter.parameter_param: object expected");
                    message.parameter_param = $root.caffe.ParameterParameter.fromObject(object.parameter_param);
                }
                if (object.pooling_param != null) {
                    if (typeof object.pooling_param !== "object")
                        throw TypeError(".caffe.LayerParameter.pooling_param: object expected");
                    message.pooling_param = $root.caffe.PoolingParameter.fromObject(object.pooling_param);
                }
                if (object.power_param != null) {
                    if (typeof object.power_param !== "object")
                        throw TypeError(".caffe.LayerParameter.power_param: object expected");
                    message.power_param = $root.caffe.PowerParameter.fromObject(object.power_param);
                }
                if (object.prelu_param != null) {
                    if (typeof object.prelu_param !== "object")
                        throw TypeError(".caffe.LayerParameter.prelu_param: object expected");
                    message.prelu_param = $root.caffe.PReLUParameter.fromObject(object.prelu_param);
                }
                if (object.python_param != null) {
                    if (typeof object.python_param !== "object")
                        throw TypeError(".caffe.LayerParameter.python_param: object expected");
                    message.python_param = $root.caffe.PythonParameter.fromObject(object.python_param);
                }
                if (object.recurrent_param != null) {
                    if (typeof object.recurrent_param !== "object")
                        throw TypeError(".caffe.LayerParameter.recurrent_param: object expected");
                    message.recurrent_param = $root.caffe.RecurrentParameter.fromObject(object.recurrent_param);
                }
                if (object.reduction_param != null) {
                    if (typeof object.reduction_param !== "object")
                        throw TypeError(".caffe.LayerParameter.reduction_param: object expected");
                    message.reduction_param = $root.caffe.ReductionParameter.fromObject(object.reduction_param);
                }
                if (object.relu_param != null) {
                    if (typeof object.relu_param !== "object")
                        throw TypeError(".caffe.LayerParameter.relu_param: object expected");
                    message.relu_param = $root.caffe.ReLUParameter.fromObject(object.relu_param);
                }
                if (object.reshape_param != null) {
                    if (typeof object.reshape_param !== "object")
                        throw TypeError(".caffe.LayerParameter.reshape_param: object expected");
                    message.reshape_param = $root.caffe.ReshapeParameter.fromObject(object.reshape_param);
                }
                if (object.scale_param != null) {
                    if (typeof object.scale_param !== "object")
                        throw TypeError(".caffe.LayerParameter.scale_param: object expected");
                    message.scale_param = $root.caffe.ScaleParameter.fromObject(object.scale_param);
                }
                if (object.sigmoid_param != null) {
                    if (typeof object.sigmoid_param !== "object")
                        throw TypeError(".caffe.LayerParameter.sigmoid_param: object expected");
                    message.sigmoid_param = $root.caffe.SigmoidParameter.fromObject(object.sigmoid_param);
                }
                if (object.softmax_param != null) {
                    if (typeof object.softmax_param !== "object")
                        throw TypeError(".caffe.LayerParameter.softmax_param: object expected");
                    message.softmax_param = $root.caffe.SoftmaxParameter.fromObject(object.softmax_param);
                }
                if (object.spp_param != null) {
                    if (typeof object.spp_param !== "object")
                        throw TypeError(".caffe.LayerParameter.spp_param: object expected");
                    message.spp_param = $root.caffe.SPPParameter.fromObject(object.spp_param);
                }
                if (object.slice_param != null) {
                    if (typeof object.slice_param !== "object")
                        throw TypeError(".caffe.LayerParameter.slice_param: object expected");
                    message.slice_param = $root.caffe.SliceParameter.fromObject(object.slice_param);
                }
                if (object.swish_param != null) {
                    if (typeof object.swish_param !== "object")
                        throw TypeError(".caffe.LayerParameter.swish_param: object expected");
                    message.swish_param = $root.caffe.SwishParameter.fromObject(object.swish_param);
                }
                if (object.tanh_param != null) {
                    if (typeof object.tanh_param !== "object")
                        throw TypeError(".caffe.LayerParameter.tanh_param: object expected");
                    message.tanh_param = $root.caffe.TanHParameter.fromObject(object.tanh_param);
                }
                if (object.threshold_param != null) {
                    if (typeof object.threshold_param !== "object")
                        throw TypeError(".caffe.LayerParameter.threshold_param: object expected");
                    message.threshold_param = $root.caffe.ThresholdParameter.fromObject(object.threshold_param);
                }
                if (object.tile_param != null) {
                    if (typeof object.tile_param !== "object")
                        throw TypeError(".caffe.LayerParameter.tile_param: object expected");
                    message.tile_param = $root.caffe.TileParameter.fromObject(object.tile_param);
                }
                if (object.window_data_param != null) {
                    if (typeof object.window_data_param !== "object")
                        throw TypeError(".caffe.LayerParameter.window_data_param: object expected");
                    message.window_data_param = $root.caffe.WindowDataParameter.fromObject(object.window_data_param);
                }
                return message;
            };
    
            LayerParameter.toObject = function toObject(message, options) {
                if (!options)
                    options = {};
                var object = {};
                if (options.arrays || options.defaults) {
                    object.bottom = [];
                    object.top = [];
                    object.loss_weight = [];
                    object.param = [];
                    object.blobs = [];
                    object.include = [];
                    object.exclude = [];
                    object.propagate_down = [];
                }
                if (options.defaults) {
                    object.name = "";
                    object.type = "";
                    object.phase = options.enums === String ? "TRAIN" : 0;
                    object.transform_param = null;
                    object.loss_param = null;
                    object.accuracy_param = null;
                    object.argmax_param = null;
                    object.concat_param = null;
                    object.contrastive_loss_param = null;
                    object.convolution_param = null;
                    object.data_param = null;
                    object.dropout_param = null;
                    object.dummy_data_param = null;
                    object.eltwise_param = null;
                    object.exp_param = null;
                    object.hdf5_data_param = null;
                    object.hdf5_output_param = null;
                    object.hinge_loss_param = null;
                    object.image_data_param = null;
                    object.infogain_loss_param = null;
                    object.inner_product_param = null;
                    object.lrn_param = null;
                    object.memory_data_param = null;
                    object.mvn_param = null;
                    object.pooling_param = null;
                    object.power_param = null;
                    object.relu_param = null;
                    object.sigmoid_param = null;
                    object.softmax_param = null;
                    object.slice_param = null;
                    object.tanh_param = null;
                    object.threshold_param = null;
                    object.window_data_param = null;
                    object.python_param = null;
                    object.prelu_param = null;
                    object.spp_param = null;
                    object.reshape_param = null;
                    object.log_param = null;
                    object.flatten_param = null;
                    object.reduction_param = null;
                    object.embed_param = null;
                    object.tile_param = null;
                    object.batch_norm_param = null;
                    object.elu_param = null;
                    object.bias_param = null;
                    object.scale_param = null;
                    object.input_param = null;
                    object.crop_param = null;
                    object.parameter_param = null;
                    object.recurrent_param = null;
                    object.swish_param = null;
                    object.clip_param = null;
                }
                if (message.name != null && message.hasOwnProperty("name"))
                    object.name = message.name;
                if (message.type != null && message.hasOwnProperty("type"))
                    object.type = message.type;
                if (message.bottom && message.bottom.length) {
                    object.bottom = [];
                    for (var j = 0; j < message.bottom.length; ++j)
                        object.bottom[j] = message.bottom[j];
                }
                if (message.top && message.top.length) {
                    object.top = [];
                    for (var j = 0; j < message.top.length; ++j)
                        object.top[j] = message.top[j];
                }
                if (message.loss_weight && message.loss_weight.length) {
                    object.loss_weight = [];
                    for (var j = 0; j < message.loss_weight.length; ++j)
                        object.loss_weight[j] = options.json && !isFinite(message.loss_weight[j]) ? String(message.loss_weight[j]) : message.loss_weight[j];
                }
                if (message.param && message.param.length) {
                    object.param = [];
                    for (var j = 0; j < message.param.length; ++j)
                        object.param[j] = $root.caffe.ParamSpec.toObject(message.param[j], options);
                }
                if (message.blobs && message.blobs.length) {
                    object.blobs = [];
                    for (var j = 0; j < message.blobs.length; ++j)
                        object.blobs[j] = $root.caffe.BlobProto.toObject(message.blobs[j], options);
                }
                if (message.include && message.include.length) {
                    object.include = [];
                    for (var j = 0; j < message.include.length; ++j)
                        object.include[j] = $root.caffe.NetStateRule.toObject(message.include[j], options);
                }
                if (message.exclude && message.exclude.length) {
                    object.exclude = [];
                    for (var j = 0; j < message.exclude.length; ++j)
                        object.exclude[j] = $root.caffe.NetStateRule.toObject(message.exclude[j], options);
                }
                if (message.phase != null && message.hasOwnProperty("phase"))
                    object.phase = options.enums === String ? $root.caffe.Phase[message.phase] : message.phase;
                if (message.propagate_down && message.propagate_down.length) {
                    object.propagate_down = [];
                    for (var j = 0; j < message.propagate_down.length; ++j)
                        object.propagate_down[j] = message.propagate_down[j];
                }
                if (message.transform_param != null && message.hasOwnProperty("transform_param"))
                    object.transform_param = $root.caffe.TransformationParameter.toObject(message.transform_param, options);
                if (message.loss_param != null && message.hasOwnProperty("loss_param"))
                    object.loss_param = $root.caffe.LossParameter.toObject(message.loss_param, options);
                if (message.accuracy_param != null && message.hasOwnProperty("accuracy_param"))
                    object.accuracy_param = $root.caffe.AccuracyParameter.toObject(message.accuracy_param, options);
                if (message.argmax_param != null && message.hasOwnProperty("argmax_param"))
                    object.argmax_param = $root.caffe.ArgMaxParameter.toObject(message.argmax_param, options);
                if (message.concat_param != null && message.hasOwnProperty("concat_param"))
                    object.concat_param = $root.caffe.ConcatParameter.toObject(message.concat_param, options);
                if (message.contrastive_loss_param != null && message.hasOwnProperty("contrastive_loss_param"))
                    object.contrastive_loss_param = $root.caffe.ContrastiveLossParameter.toObject(message.contrastive_loss_param, options);
                if (message.convolution_param != null && message.hasOwnProperty("convolution_param"))
                    object.convolution_param = $root.caffe.ConvolutionParameter.toObject(message.convolution_param, options);
                if (message.data_param != null && message.hasOwnProperty("data_param"))
                    object.data_param = $root.caffe.DataParameter.toObject(message.data_param, options);
                if (message.dropout_param != null && message.hasOwnProperty("dropout_param"))
                    object.dropout_param = $root.caffe.DropoutParameter.toObject(message.dropout_param, options);
                if (message.dummy_data_param != null && message.hasOwnProperty("dummy_data_param"))
                    object.dummy_data_param = $root.caffe.DummyDataParameter.toObject(message.dummy_data_param, options);
                if (message.eltwise_param != null && message.hasOwnProperty("eltwise_param"))
                    object.eltwise_param = $root.caffe.EltwiseParameter.toObject(message.eltwise_param, options);
                if (message.exp_param != null && message.hasOwnProperty("exp_param"))
                    object.exp_param = $root.caffe.ExpParameter.toObject(message.exp_param, options);
                if (message.hdf5_data_param != null && message.hasOwnProperty("hdf5_data_param"))
                    object.hdf5_data_param = $root.caffe.HDF5DataParameter.toObject(message.hdf5_data_param, options);
                if (message.hdf5_output_param != null && message.hasOwnProperty("hdf5_output_param"))
                    object.hdf5_output_param = $root.caffe.HDF5OutputParameter.toObject(message.hdf5_output_param, options);
                if (message.hinge_loss_param != null && message.hasOwnProperty("hinge_loss_param"))
                    object.hinge_loss_param = $root.caffe.HingeLossParameter.toObject(message.hinge_loss_param, options);
                if (message.image_data_param != null && message.hasOwnProperty("image_data_param"))
                    object.image_data_param = $root.caffe.ImageDataParameter.toObject(message.image_data_param, options);
                if (message.infogain_loss_param != null && message.hasOwnProperty("infogain_loss_param"))
                    object.infogain_loss_param = $root.caffe.InfogainLossParameter.toObject(message.infogain_loss_param, options);
                if (message.inner_product_param != null && message.hasOwnProperty("inner_product_param"))
                    object.inner_product_param = $root.caffe.InnerProductParameter.toObject(message.inner_product_param, options);
                if (message.lrn_param != null && message.hasOwnProperty("lrn_param"))
                    object.lrn_param = $root.caffe.LRNParameter.toObject(message.lrn_param, options);
                if (message.memory_data_param != null && message.hasOwnProperty("memory_data_param"))
                    object.memory_data_param = $root.caffe.MemoryDataParameter.toObject(message.memory_data_param, options);
                if (message.mvn_param != null && message.hasOwnProperty("mvn_param"))
                    object.mvn_param = $root.caffe.MVNParameter.toObject(message.mvn_param, options);
                if (message.pooling_param != null && message.hasOwnProperty("pooling_param"))
                    object.pooling_param = $root.caffe.PoolingParameter.toObject(message.pooling_param, options);
                if (message.power_param != null && message.hasOwnProperty("power_param"))
                    object.power_param = $root.caffe.PowerParameter.toObject(message.power_param, options);
                if (message.relu_param != null && message.hasOwnProperty("relu_param"))
                    object.relu_param = $root.caffe.ReLUParameter.toObject(message.relu_param, options);
                if (message.sigmoid_param != null && message.hasOwnProperty("sigmoid_param"))
                    object.sigmoid_param = $root.caffe.SigmoidParameter.toObject(message.sigmoid_param, options);
                if (message.softmax_param != null && message.hasOwnProperty("softmax_param"))
                    object.softmax_param = $root.caffe.SoftmaxParameter.toObject(message.softmax_param, options);
                if (message.slice_param != null && message.hasOwnProperty("slice_param"))
                    object.slice_param = $root.caffe.SliceParameter.toObject(message.slice_param, options);
                if (message.tanh_param != null && message.hasOwnProperty("tanh_param"))
                    object.tanh_param = $root.caffe.TanHParameter.toObject(message.tanh_param, options);
                if (message.threshold_param != null && message.hasOwnProperty("threshold_param"))
                    object.threshold_param = $root.caffe.ThresholdParameter.toObject(message.threshold_param, options);
                if (message.window_data_param != null && message.hasOwnProperty("window_data_param"))
                    object.window_data_param = $root.caffe.WindowDataParameter.toObject(message.window_data_param, options);
                if (message.python_param != null && message.hasOwnProperty("python_param"))
                    object.python_param = $root.caffe.PythonParameter.toObject(message.python_param, options);
                if (message.prelu_param != null && message.hasOwnProperty("prelu_param"))
                    object.prelu_param = $root.caffe.PReLUParameter.toObject(message.prelu_param, options);
                if (message.spp_param != null && message.hasOwnProperty("spp_param"))
                    object.spp_param = $root.caffe.SPPParameter.toObject(message.spp_param, options);
                if (message.reshape_param != null && message.hasOwnProperty("reshape_param"))
                    object.reshape_param = $root.caffe.ReshapeParameter.toObject(message.reshape_param, options);
                if (message.log_param != null && message.hasOwnProperty("log_param"))
                    object.log_param = $root.caffe.LogParameter.toObject(message.log_param, options);
                if (message.flatten_param != null && message.hasOwnProperty("flatten_param"))
                    object.flatten_param = $root.caffe.FlattenParameter.toObject(message.flatten_param, options);
                if (message.reduction_param != null && message.hasOwnProperty("reduction_param"))
                    object.reduction_param = $root.caffe.ReductionParameter.toObject(message.reduction_param, options);
                if (message.embed_param != null && message.hasOwnProperty("embed_param"))
                    object.embed_param = $root.caffe.EmbedParameter.toObject(message.embed_param, options);
                if (message.tile_param != null && message.hasOwnProperty("tile_param"))
                    object.tile_param = $root.caffe.TileParameter.toObject(message.tile_param, options);
                if (message.batch_norm_param != null && message.hasOwnProperty("batch_norm_param"))
                    object.batch_norm_param = $root.caffe.BatchNormParameter.toObject(message.batch_norm_param, options);
                if (message.elu_param != null && message.hasOwnProperty("elu_param"))
                    object.elu_param = $root.caffe.ELUParameter.toObject(message.elu_param, options);
                if (message.bias_param != null && message.hasOwnProperty("bias_param"))
                    object.bias_param = $root.caffe.BiasParameter.toObject(message.bias_param, options);
                if (message.scale_param != null && message.hasOwnProperty("scale_param"))
                    object.scale_param = $root.caffe.ScaleParameter.toObject(message.scale_param, options);
                if (message.input_param != null && message.hasOwnProperty("input_param"))
                    object.input_param = $root.caffe.InputParameter.toObject(message.input_param, options);
                if (message.crop_param != null && message.hasOwnProperty("crop_param"))
                    object.crop_param = $root.caffe.CropParameter.toObject(message.crop_param, options);
                if (message.parameter_param != null && message.hasOwnProperty("parameter_param"))
                    object.parameter_param = $root.caffe.ParameterParameter.toObject(message.parameter_param, options);
                if (message.recurrent_param != null && message.hasOwnProperty("recurrent_param"))
                    object.recurrent_param = $root.caffe.RecurrentParameter.toObject(message.recurrent_param, options);
                if (message.swish_param != null && message.hasOwnProperty("swish_param"))
                    object.swish_param = $root.caffe.SwishParameter.toObject(message.swish_param, options);
                if (message.clip_param != null && message.hasOwnProperty("clip_param"))
                    object.clip_param = $root.caffe.ClipParameter.toObject(message.clip_param, options);
                return object;
            };
    
            LayerParameter.prototype.toJSON = function toJSON() {
                return this.constructor.toObject(this, $protobuf.util.toJSONOptions);
            };
    
            return LayerParameter;
        })();
    
        caffe.TransformationParameter = (function() {
    
            function TransformationParameter(properties) {
                this.mean_value = [];
                if (properties)
                    for (var keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                        if (properties[keys[i]] != null)
                            this[keys[i]] = properties[keys[i]];
            }
    
            TransformationParameter.prototype.scale = 1;
            TransformationParameter.prototype.mirror = false;
            TransformationParameter.prototype.crop_size = 0;
            TransformationParameter.prototype.mean_file = "";
            TransformationParameter.prototype.mean_value = $util.emptyArray;
            TransformationParameter.prototype.force_color = false;
            TransformationParameter.prototype.force_gray = false;
    
            TransformationParameter.create = function create(properties) {
                return new TransformationParameter(properties);
            };
    
            TransformationParameter.decode = function decode(reader, length) {
                if (!(reader instanceof $Reader))
                    reader = $Reader.create(reader);
                var end = length === undefined ? reader.len : reader.pos + length, message = new $root.caffe.TransformationParameter();
                while (reader.pos < end) {
                    var tag = reader.uint32();
                    switch (tag >>> 3) {
                    case 1:
                        message.scale = reader.float();
                        break;
                    case 2:
                        message.mirror = reader.bool();
                        break;
                    case 3:
                        message.crop_size = reader.uint32();
                        break;
                    case 4:
                        message.mean_file = reader.string();
                        break;
                    case 5:
                        if (!(message.mean_value && message.mean_value.length))
                            message.mean_value = [];
                        if ((tag & 7) === 2) {
                            var end2 = reader.uint32() + reader.pos;
                            while (reader.pos < end2)
                                message.mean_value.push(reader.float());
                        } else
                            message.mean_value.push(reader.float());
                        break;
                    case 6:
                        message.force_color = reader.bool();
                        break;
                    case 7:
                        message.force_gray = reader.bool();
                        break;
                    default:
                        reader.skipType(tag & 7);
                        break;
                    }
                }
                return message;
            };
    
            TransformationParameter.decodeText = function decodeText(reader, block) {
                if (!(reader instanceof $TextReader))
                    reader = $TextReader.create(reader);
                var message = new $root.caffe.TransformationParameter();
                reader.start(block);
                while (!reader.end(block)) {
                    var tag = reader.tag();
                    switch (tag) {
                    case "scale":
                        message.scale = reader.float();
                        break;
                    case "mirror":
                        message.mirror = reader.bool();
                        break;
                    case "crop_size":
                        message.crop_size = reader.uint32();
                        break;
                    case "mean_file":
                        message.mean_file = reader.string();
                        break;
                    case "mean_value":
                        if (!(message.mean_value && message.mean_value.length))
                            message.mean_value = [];
                        message.mean_value.push(reader.float());
                        break;
                    case "force_color":
                        message.force_color = reader.bool();
                        break;
                    case "force_gray":
                        message.force_gray = reader.bool();
                        break;
                    default:
                        reader.handle(tag, message);
                        break;
                    }
                }
                return message;
            };
    
            TransformationParameter.verify = function verify(message) {
                if (typeof message !== "object" || message === null)
                    return "object expected";
                if (message.scale != null && message.hasOwnProperty("scale"))
                    if (typeof message.scale !== "number")
                        return "scale: number expected";
                if (message.mirror != null && message.hasOwnProperty("mirror"))
                    if (typeof message.mirror !== "boolean")
                        return "mirror: boolean expected";
                if (message.crop_size != null && message.hasOwnProperty("crop_size"))
                    if (!$util.isInteger(message.crop_size))
                        return "crop_size: integer expected";
                if (message.mean_file != null && message.hasOwnProperty("mean_file"))
                    if (!$util.isString(message.mean_file))
                        return "mean_file: string expected";
                if (message.mean_value != null && message.hasOwnProperty("mean_value")) {
                    if (!Array.isArray(message.mean_value))
                        return "mean_value: array expected";
                    for (var i = 0; i < message.mean_value.length; ++i)
                        if (typeof message.mean_value[i] !== "number")
                            return "mean_value: number[] expected";
                }
                if (message.force_color != null && message.hasOwnProperty("force_color"))
                    if (typeof message.force_color !== "boolean")
                        return "force_color: boolean expected";
                if (message.force_gray != null && message.hasOwnProperty("force_gray"))
                    if (typeof message.force_gray !== "boolean")
                        return "force_gray: boolean expected";
                return null;
            };
    
            TransformationParameter.fromObject = function fromObject(object) {
                if (object instanceof $root.caffe.TransformationParameter)
                    return object;
                var message = new $root.caffe.TransformationParameter();
                if (object.scale != null)
                    message.scale = Number(object.scale);
                if (object.mirror != null)
                    message.mirror = Boolean(object.mirror);
                if (object.crop_size != null)
                    message.crop_size = object.crop_size >>> 0;
                if (object.mean_file != null)
                    message.mean_file = String(object.mean_file);
                if (object.mean_value) {
                    if (!Array.isArray(object.mean_value))
                        throw TypeError(".caffe.TransformationParameter.mean_value: array expected");
                    message.mean_value = [];
                    for (var i = 0; i < object.mean_value.length; ++i)
                        message.mean_value[i] = Number(object.mean_value[i]);
                }
                if (object.force_color != null)
                    message.force_color = Boolean(object.force_color);
                if (object.force_gray != null)
                    message.force_gray = Boolean(object.force_gray);
                return message;
            };
    
            TransformationParameter.toObject = function toObject(message, options) {
                if (!options)
                    options = {};
                var object = {};
                if (options.arrays || options.defaults)
                    object.mean_value = [];
                if (options.defaults) {
                    object.scale = 1;
                    object.mirror = false;
                    object.crop_size = 0;
                    object.mean_file = "";
                    object.force_color = false;
                    object.force_gray = false;
                }
                if (message.scale != null && message.hasOwnProperty("scale"))
                    object.scale = options.json && !isFinite(message.scale) ? String(message.scale) : message.scale;
                if (message.mirror != null && message.hasOwnProperty("mirror"))
                    object.mirror = message.mirror;
                if (message.crop_size != null && message.hasOwnProperty("crop_size"))
                    object.crop_size = message.crop_size;
                if (message.mean_file != null && message.hasOwnProperty("mean_file"))
                    object.mean_file = message.mean_file;
                if (message.mean_value && message.mean_value.length) {
                    object.mean_value = [];
                    for (var j = 0; j < message.mean_value.length; ++j)
                        object.mean_value[j] = options.json && !isFinite(message.mean_value[j]) ? String(message.mean_value[j]) : message.mean_value[j];
                }
                if (message.force_color != null && message.hasOwnProperty("force_color"))
                    object.force_color = message.force_color;
                if (message.force_gray != null && message.hasOwnProperty("force_gray"))
                    object.force_gray = message.force_gray;
                return object;
            };
    
            TransformationParameter.prototype.toJSON = function toJSON() {
                return this.constructor.toObject(this, $protobuf.util.toJSONOptions);
            };
    
            return TransformationParameter;
        })();
    
        caffe.LossParameter = (function() {
    
            function LossParameter(properties) {
                if (properties)
                    for (var keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                        if (properties[keys[i]] != null)
                            this[keys[i]] = properties[keys[i]];
            }
    
            LossParameter.prototype.ignore_label = 0;
            LossParameter.prototype.normalization = 1;
            LossParameter.prototype.normalize = false;
    
            LossParameter.create = function create(properties) {
                return new LossParameter(properties);
            };
    
            LossParameter.decode = function decode(reader, length) {
                if (!(reader instanceof $Reader))
                    reader = $Reader.create(reader);
                var end = length === undefined ? reader.len : reader.pos + length, message = new $root.caffe.LossParameter();
                while (reader.pos < end) {
                    var tag = reader.uint32();
                    switch (tag >>> 3) {
                    case 1:
                        message.ignore_label = reader.int32();
                        break;
                    case 3:
                        message.normalization = reader.int32();
                        break;
                    case 2:
                        message.normalize = reader.bool();
                        break;
                    default:
                        reader.skipType(tag & 7);
                        break;
                    }
                }
                return message;
            };
    
            LossParameter.decodeText = function decodeText(reader, block) {
                if (!(reader instanceof $TextReader))
                    reader = $TextReader.create(reader);
                var message = new $root.caffe.LossParameter();
                reader.start(block);
                while (!reader.end(block)) {
                    var tag = reader.tag();
                    switch (tag) {
                    case "ignore_label":
                        message.ignore_label = reader.int32();
                        break;
                    case "normalization":
                        message.normalization = reader.enum($root.caffe.LossParameter.NormalizationMode);
                        break;
                    case "normalize":
                        message.normalize = reader.bool();
                        break;
                    default:
                        reader.handle(tag, message);
                        break;
                    }
                }
                return message;
            };
    
            LossParameter.verify = function verify(message) {
                if (typeof message !== "object" || message === null)
                    return "object expected";
                if (message.ignore_label != null && message.hasOwnProperty("ignore_label"))
                    if (!$util.isInteger(message.ignore_label))
                        return "ignore_label: integer expected";
                if (message.normalization != null && message.hasOwnProperty("normalization"))
                    switch (message.normalization) {
                    default:
                        return "normalization: enum value expected";
                    case 0:
                    case 1:
                    case 2:
                    case 3:
                        break;
                    }
                if (message.normalize != null && message.hasOwnProperty("normalize"))
                    if (typeof message.normalize !== "boolean")
                        return "normalize: boolean expected";
                return null;
            };
    
            LossParameter.fromObject = function fromObject(object) {
                if (object instanceof $root.caffe.LossParameter)
                    return object;
                var message = new $root.caffe.LossParameter();
                if (object.ignore_label != null)
                    message.ignore_label = object.ignore_label | 0;
                switch (object.normalization) {
                case "FULL":
                case 0:
                    message.normalization = 0;
                    break;
                case "VALID":
                case 1:
                    message.normalization = 1;
                    break;
                case "BATCH_SIZE":
                case 2:
                    message.normalization = 2;
                    break;
                case "NONE":
                case 3:
                    message.normalization = 3;
                    break;
                }
                if (object.normalize != null)
                    message.normalize = Boolean(object.normalize);
                return message;
            };
    
            LossParameter.toObject = function toObject(message, options) {
                if (!options)
                    options = {};
                var object = {};
                if (options.defaults) {
                    object.ignore_label = 0;
                    object.normalize = false;
                    object.normalization = options.enums === String ? "VALID" : 1;
                }
                if (message.ignore_label != null && message.hasOwnProperty("ignore_label"))
                    object.ignore_label = message.ignore_label;
                if (message.normalize != null && message.hasOwnProperty("normalize"))
                    object.normalize = message.normalize;
                if (message.normalization != null && message.hasOwnProperty("normalization"))
                    object.normalization = options.enums === String ? $root.caffe.LossParameter.NormalizationMode[message.normalization] : message.normalization;
                return object;
            };
    
            LossParameter.prototype.toJSON = function toJSON() {
                return this.constructor.toObject(this, $protobuf.util.toJSONOptions);
            };
    
            LossParameter.NormalizationMode = (function() {
                var valuesById = {}, values = Object.create(valuesById);
                values[valuesById[0] = "FULL"] = 0;
                values[valuesById[1] = "VALID"] = 1;
                values[valuesById[2] = "BATCH_SIZE"] = 2;
                values[valuesById[3] = "NONE"] = 3;
                return values;
            })();
    
            return LossParameter;
        })();
    
        caffe.AccuracyParameter = (function() {
    
            function AccuracyParameter(properties) {
                if (properties)
                    for (var keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                        if (properties[keys[i]] != null)
                            this[keys[i]] = properties[keys[i]];
            }
    
            AccuracyParameter.prototype.top_k = 1;
            AccuracyParameter.prototype.axis = 1;
            AccuracyParameter.prototype.ignore_label = 0;
    
            AccuracyParameter.create = function create(properties) {
                return new AccuracyParameter(properties);
            };
    
            AccuracyParameter.decode = function decode(reader, length) {
                if (!(reader instanceof $Reader))
                    reader = $Reader.create(reader);
                var end = length === undefined ? reader.len : reader.pos + length, message = new $root.caffe.AccuracyParameter();
                while (reader.pos < end) {
                    var tag = reader.uint32();
                    switch (tag >>> 3) {
                    case 1:
                        message.top_k = reader.uint32();
                        break;
                    case 2:
                        message.axis = reader.int32();
                        break;
                    case 3:
                        message.ignore_label = reader.int32();
                        break;
                    default:
                        reader.skipType(tag & 7);
                        break;
                    }
                }
                return message;
            };
    
            AccuracyParameter.decodeText = function decodeText(reader, block) {
                if (!(reader instanceof $TextReader))
                    reader = $TextReader.create(reader);
                var message = new $root.caffe.AccuracyParameter();
                reader.start(block);
                while (!reader.end(block)) {
                    var tag = reader.tag();
                    switch (tag) {
                    case "top_k":
                        message.top_k = reader.uint32();
                        break;
                    case "axis":
                        message.axis = reader.int32();
                        break;
                    case "ignore_label":
                        message.ignore_label = reader.int32();
                        break;
                    default:
                        reader.handle(tag, message);
                        break;
                    }
                }
                return message;
            };
    
            AccuracyParameter.verify = function verify(message) {
                if (typeof message !== "object" || message === null)
                    return "object expected";
                if (message.top_k != null && message.hasOwnProperty("top_k"))
                    if (!$util.isInteger(message.top_k))
                        return "top_k: integer expected";
                if (message.axis != null && message.hasOwnProperty("axis"))
                    if (!$util.isInteger(message.axis))
                        return "axis: integer expected";
                if (message.ignore_label != null && message.hasOwnProperty("ignore_label"))
                    if (!$util.isInteger(message.ignore_label))
                        return "ignore_label: integer expected";
                return null;
            };
    
            AccuracyParameter.fromObject = function fromObject(object) {
                if (object instanceof $root.caffe.AccuracyParameter)
                    return object;
                var message = new $root.caffe.AccuracyParameter();
                if (object.top_k != null)
                    message.top_k = object.top_k >>> 0;
                if (object.axis != null)
                    message.axis = object.axis | 0;
                if (object.ignore_label != null)
                    message.ignore_label = object.ignore_label | 0;
                return message;
            };
    
            AccuracyParameter.toObject = function toObject(message, options) {
                if (!options)
                    options = {};
                var object = {};
                if (options.defaults) {
                    object.top_k = 1;
                    object.axis = 1;
                    object.ignore_label = 0;
                }
                if (message.top_k != null && message.hasOwnProperty("top_k"))
                    object.top_k = message.top_k;
                if (message.axis != null && message.hasOwnProperty("axis"))
                    object.axis = message.axis;
                if (message.ignore_label != null && message.hasOwnProperty("ignore_label"))
                    object.ignore_label = message.ignore_label;
                return object;
            };
    
            AccuracyParameter.prototype.toJSON = function toJSON() {
                return this.constructor.toObject(this, $protobuf.util.toJSONOptions);
            };
    
            return AccuracyParameter;
        })();
    
        caffe.ArgMaxParameter = (function() {
    
            function ArgMaxParameter(properties) {
                if (properties)
                    for (var keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                        if (properties[keys[i]] != null)
                            this[keys[i]] = properties[keys[i]];
            }
    
            ArgMaxParameter.prototype.out_max_val = false;
            ArgMaxParameter.prototype.top_k = 1;
            ArgMaxParameter.prototype.axis = 0;
    
            ArgMaxParameter.create = function create(properties) {
                return new ArgMaxParameter(properties);
            };
    
            ArgMaxParameter.decode = function decode(reader, length) {
                if (!(reader instanceof $Reader))
                    reader = $Reader.create(reader);
                var end = length === undefined ? reader.len : reader.pos + length, message = new $root.caffe.ArgMaxParameter();
                while (reader.pos < end) {
                    var tag = reader.uint32();
                    switch (tag >>> 3) {
                    case 1:
                        message.out_max_val = reader.bool();
                        break;
                    case 2:
                        message.top_k = reader.uint32();
                        break;
                    case 3:
                        message.axis = reader.int32();
                        break;
                    default:
                        reader.skipType(tag & 7);
                        break;
                    }
                }
                return message;
            };
    
            ArgMaxParameter.decodeText = function decodeText(reader, block) {
                if (!(reader instanceof $TextReader))
                    reader = $TextReader.create(reader);
                var message = new $root.caffe.ArgMaxParameter();
                reader.start(block);
                while (!reader.end(block)) {
                    var tag = reader.tag();
                    switch (tag) {
                    case "out_max_val":
                        message.out_max_val = reader.bool();
                        break;
                    case "top_k":
                        message.top_k = reader.uint32();
                        break;
                    case "axis":
                        message.axis = reader.int32();
                        break;
                    default:
                        reader.handle(tag, message);
                        break;
                    }
                }
                return message;
            };
    
            ArgMaxParameter.verify = function verify(message) {
                if (typeof message !== "object" || message === null)
                    return "object expected";
                if (message.out_max_val != null && message.hasOwnProperty("out_max_val"))
                    if (typeof message.out_max_val !== "boolean")
                        return "out_max_val: boolean expected";
                if (message.top_k != null && message.hasOwnProperty("top_k"))
                    if (!$util.isInteger(message.top_k))
                        return "top_k: integer expected";
                if (message.axis != null && message.hasOwnProperty("axis"))
                    if (!$util.isInteger(message.axis))
                        return "axis: integer expected";
                return null;
            };
    
            ArgMaxParameter.fromObject = function fromObject(object) {
                if (object instanceof $root.caffe.ArgMaxParameter)
                    return object;
                var message = new $root.caffe.ArgMaxParameter();
                if (object.out_max_val != null)
                    message.out_max_val = Boolean(object.out_max_val);
                if (object.top_k != null)
                    message.top_k = object.top_k >>> 0;
                if (object.axis != null)
                    message.axis = object.axis | 0;
                return message;
            };
    
            ArgMaxParameter.toObject = function toObject(message, options) {
                if (!options)
                    options = {};
                var object = {};
                if (options.defaults) {
                    object.out_max_val = false;
                    object.top_k = 1;
                    object.axis = 0;
                }
                if (message.out_max_val != null && message.hasOwnProperty("out_max_val"))
                    object.out_max_val = message.out_max_val;
                if (message.top_k != null && message.hasOwnProperty("top_k"))
                    object.top_k = message.top_k;
                if (message.axis != null && message.hasOwnProperty("axis"))
                    object.axis = message.axis;
                return object;
            };
    
            ArgMaxParameter.prototype.toJSON = function toJSON() {
                return this.constructor.toObject(this, $protobuf.util.toJSONOptions);
            };
    
            return ArgMaxParameter;
        })();
    
        caffe.ClipParameter = (function() {
    
            function ClipParameter(properties) {
                if (properties)
                    for (var keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                        if (properties[keys[i]] != null)
                            this[keys[i]] = properties[keys[i]];
            }
    
            ClipParameter.prototype.min = 0;
            ClipParameter.prototype.max = 0;
    
            ClipParameter.create = function create(properties) {
                return new ClipParameter(properties);
            };
    
            ClipParameter.decode = function decode(reader, length) {
                if (!(reader instanceof $Reader))
                    reader = $Reader.create(reader);
                var end = length === undefined ? reader.len : reader.pos + length, message = new $root.caffe.ClipParameter();
                while (reader.pos < end) {
                    var tag = reader.uint32();
                    switch (tag >>> 3) {
                    case 1:
                        message.min = reader.float();
                        break;
                    case 2:
                        message.max = reader.float();
                        break;
                    default:
                        reader.skipType(tag & 7);
                        break;
                    }
                }
                if (!message.hasOwnProperty("min"))
                    throw $util.ProtocolError("missing required 'min'", { instance: message });
                if (!message.hasOwnProperty("max"))
                    throw $util.ProtocolError("missing required 'max'", { instance: message });
                return message;
            };
    
            ClipParameter.decodeText = function decodeText(reader, block) {
                if (!(reader instanceof $TextReader))
                    reader = $TextReader.create(reader);
                var message = new $root.caffe.ClipParameter();
                reader.start(block);
                while (!reader.end(block)) {
                    var tag = reader.tag();
                    switch (tag) {
                    case "min":
                        message.min = reader.float();
                        break;
                    case "max":
                        message.max = reader.float();
                        break;
                    default:
                        reader.handle(tag, message);
                        break;
                    }
                }
                if (!message.hasOwnProperty("min"))
                    throw $util.ProtocolError("missing required 'min'", { instance: message });
                if (!message.hasOwnProperty("max"))
                    throw $util.ProtocolError("missing required 'max'", { instance: message });
                return message;
            };
    
            ClipParameter.verify = function verify(message) {
                if (typeof message !== "object" || message === null)
                    return "object expected";
                if (typeof message.min !== "number")
                    return "min: number expected";
                if (typeof message.max !== "number")
                    return "max: number expected";
                return null;
            };
    
            ClipParameter.fromObject = function fromObject(object) {
                if (object instanceof $root.caffe.ClipParameter)
                    return object;
                var message = new $root.caffe.ClipParameter();
                if (object.min != null)
                    message.min = Number(object.min);
                if (object.max != null)
                    message.max = Number(object.max);
                return message;
            };
    
            ClipParameter.toObject = function toObject(message, options) {
                if (!options)
                    options = {};
                var object = {};
                if (options.defaults) {
                    object.min = 0;
                    object.max = 0;
                }
                if (message.min != null && message.hasOwnProperty("min"))
                    object.min = options.json && !isFinite(message.min) ? String(message.min) : message.min;
                if (message.max != null && message.hasOwnProperty("max"))
                    object.max = options.json && !isFinite(message.max) ? String(message.max) : message.max;
                return object;
            };
    
            ClipParameter.prototype.toJSON = function toJSON() {
                return this.constructor.toObject(this, $protobuf.util.toJSONOptions);
            };
    
            return ClipParameter;
        })();
    
        caffe.ConcatParameter = (function() {
    
            function ConcatParameter(properties) {
                if (properties)
                    for (var keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                        if (properties[keys[i]] != null)
                            this[keys[i]] = properties[keys[i]];
            }
    
            ConcatParameter.prototype.axis = 1;
            ConcatParameter.prototype.concat_dim = 1;
    
            ConcatParameter.create = function create(properties) {
                return new ConcatParameter(properties);
            };
    
            ConcatParameter.decode = function decode(reader, length) {
                if (!(reader instanceof $Reader))
                    reader = $Reader.create(reader);
                var end = length === undefined ? reader.len : reader.pos + length, message = new $root.caffe.ConcatParameter();
                while (reader.pos < end) {
                    var tag = reader.uint32();
                    switch (tag >>> 3) {
                    case 2:
                        message.axis = reader.int32();
                        break;
                    case 1:
                        message.concat_dim = reader.uint32();
                        break;
                    default:
                        reader.skipType(tag & 7);
                        break;
                    }
                }
                return message;
            };
    
            ConcatParameter.decodeText = function decodeText(reader, block) {
                if (!(reader instanceof $TextReader))
                    reader = $TextReader.create(reader);
                var message = new $root.caffe.ConcatParameter();
                reader.start(block);
                while (!reader.end(block)) {
                    var tag = reader.tag();
                    switch (tag) {
                    case "axis":
                        message.axis = reader.int32();
                        break;
                    case "concat_dim":
                        message.concat_dim = reader.uint32();
                        break;
                    default:
                        reader.handle(tag, message);
                        break;
                    }
                }
                return message;
            };
    
            ConcatParameter.verify = function verify(message) {
                if (typeof message !== "object" || message === null)
                    return "object expected";
                if (message.axis != null && message.hasOwnProperty("axis"))
                    if (!$util.isInteger(message.axis))
                        return "axis: integer expected";
                if (message.concat_dim != null && message.hasOwnProperty("concat_dim"))
                    if (!$util.isInteger(message.concat_dim))
                        return "concat_dim: integer expected";
                return null;
            };
    
            ConcatParameter.fromObject = function fromObject(object) {
                if (object instanceof $root.caffe.ConcatParameter)
                    return object;
                var message = new $root.caffe.ConcatParameter();
                if (object.axis != null)
                    message.axis = object.axis | 0;
                if (object.concat_dim != null)
                    message.concat_dim = object.concat_dim >>> 0;
                return message;
            };
    
            ConcatParameter.toObject = function toObject(message, options) {
                if (!options)
                    options = {};
                var object = {};
                if (options.defaults) {
                    object.concat_dim = 1;
                    object.axis = 1;
                }
                if (message.concat_dim != null && message.hasOwnProperty("concat_dim"))
                    object.concat_dim = message.concat_dim;
                if (message.axis != null && message.hasOwnProperty("axis"))
                    object.axis = message.axis;
                return object;
            };
    
            ConcatParameter.prototype.toJSON = function toJSON() {
                return this.constructor.toObject(this, $protobuf.util.toJSONOptions);
            };
    
            return ConcatParameter;
        })();
    
        caffe.BatchNormParameter = (function() {
    
            function BatchNormParameter(properties) {
                if (properties)
                    for (var keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                        if (properties[keys[i]] != null)
                            this[keys[i]] = properties[keys[i]];
            }
    
            BatchNormParameter.prototype.use_global_stats = false;
            BatchNormParameter.prototype.moving_average_fraction = 0.999;
            BatchNormParameter.prototype.eps = 0.00001;
    
            BatchNormParameter.create = function create(properties) {
                return new BatchNormParameter(properties);
            };
    
            BatchNormParameter.decode = function decode(reader, length) {
                if (!(reader instanceof $Reader))
                    reader = $Reader.create(reader);
                var end = length === undefined ? reader.len : reader.pos + length, message = new $root.caffe.BatchNormParameter();
                while (reader.pos < end) {
                    var tag = reader.uint32();
                    switch (tag >>> 3) {
                    case 1:
                        message.use_global_stats = reader.bool();
                        break;
                    case 2:
                        message.moving_average_fraction = reader.float();
                        break;
                    case 3:
                        message.eps = reader.float();
                        break;
                    default:
                        reader.skipType(tag & 7);
                        break;
                    }
                }
                return message;
            };
    
            BatchNormParameter.decodeText = function decodeText(reader, block) {
                if (!(reader instanceof $TextReader))
                    reader = $TextReader.create(reader);
                var message = new $root.caffe.BatchNormParameter();
                reader.start(block);
                while (!reader.end(block)) {
                    var tag = reader.tag();
                    switch (tag) {
                    case "use_global_stats":
                        message.use_global_stats = reader.bool();
                        break;
                    case "moving_average_fraction":
                        message.moving_average_fraction = reader.float();
                        break;
                    case "eps":
                        message.eps = reader.float();
                        break;
                    default:
                        reader.handle(tag, message);
                        break;
                    }
                }
                return message;
            };
    
            BatchNormParameter.verify = function verify(message) {
                if (typeof message !== "object" || message === null)
                    return "object expected";
                if (message.use_global_stats != null && message.hasOwnProperty("use_global_stats"))
                    if (typeof message.use_global_stats !== "boolean")
                        return "use_global_stats: boolean expected";
                if (message.moving_average_fraction != null && message.hasOwnProperty("moving_average_fraction"))
                    if (typeof message.moving_average_fraction !== "number")
                        return "moving_average_fraction: number expected";
                if (message.eps != null && message.hasOwnProperty("eps"))
                    if (typeof message.eps !== "number")
                        return "eps: number expected";
                return null;
            };
    
            BatchNormParameter.fromObject = function fromObject(object) {
                if (object instanceof $root.caffe.BatchNormParameter)
                    return object;
                var message = new $root.caffe.BatchNormParameter();
                if (object.use_global_stats != null)
                    message.use_global_stats = Boolean(object.use_global_stats);
                if (object.moving_average_fraction != null)
                    message.moving_average_fraction = Number(object.moving_average_fraction);
                if (object.eps != null)
                    message.eps = Number(object.eps);
                return message;
            };
    
            BatchNormParameter.toObject = function toObject(message, options) {
                if (!options)
                    options = {};
                var object = {};
                if (options.defaults) {
                    object.use_global_stats = false;
                    object.moving_average_fraction = 0.999;
                    object.eps = 0.00001;
                }
                if (message.use_global_stats != null && message.hasOwnProperty("use_global_stats"))
                    object.use_global_stats = message.use_global_stats;
                if (message.moving_average_fraction != null && message.hasOwnProperty("moving_average_fraction"))
                    object.moving_average_fraction = options.json && !isFinite(message.moving_average_fraction) ? String(message.moving_average_fraction) : message.moving_average_fraction;
                if (message.eps != null && message.hasOwnProperty("eps"))
                    object.eps = options.json && !isFinite(message.eps) ? String(message.eps) : message.eps;
                return object;
            };
    
            BatchNormParameter.prototype.toJSON = function toJSON() {
                return this.constructor.toObject(this, $protobuf.util.toJSONOptions);
            };
    
            return BatchNormParameter;
        })();
    
        caffe.BiasParameter = (function() {
    
            function BiasParameter(properties) {
                if (properties)
                    for (var keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                        if (properties[keys[i]] != null)
                            this[keys[i]] = properties[keys[i]];
            }
    
            BiasParameter.prototype.axis = 1;
            BiasParameter.prototype.num_axes = 1;
            BiasParameter.prototype.filler = null;
    
            BiasParameter.create = function create(properties) {
                return new BiasParameter(properties);
            };
    
            BiasParameter.decode = function decode(reader, length) {
                if (!(reader instanceof $Reader))
                    reader = $Reader.create(reader);
                var end = length === undefined ? reader.len : reader.pos + length, message = new $root.caffe.BiasParameter();
                while (reader.pos < end) {
                    var tag = reader.uint32();
                    switch (tag >>> 3) {
                    case 1:
                        message.axis = reader.int32();
                        break;
                    case 2:
                        message.num_axes = reader.int32();
                        break;
                    case 3:
                        message.filler = $root.caffe.FillerParameter.decode(reader, reader.uint32());
                        break;
                    default:
                        reader.skipType(tag & 7);
                        break;
                    }
                }
                return message;
            };
    
            BiasParameter.decodeText = function decodeText(reader, block) {
                if (!(reader instanceof $TextReader))
                    reader = $TextReader.create(reader);
                var message = new $root.caffe.BiasParameter();
                reader.start(block);
                while (!reader.end(block)) {
                    var tag = reader.tag();
                    switch (tag) {
                    case "axis":
                        message.axis = reader.int32();
                        break;
                    case "num_axes":
                        message.num_axes = reader.int32();
                        break;
                    case "filler":
                        message.filler = $root.caffe.FillerParameter.decodeText(reader, true);
                        break;
                    default:
                        reader.handle(tag, message);
                        break;
                    }
                }
                return message;
            };
    
            BiasParameter.verify = function verify(message) {
                if (typeof message !== "object" || message === null)
                    return "object expected";
                if (message.axis != null && message.hasOwnProperty("axis"))
                    if (!$util.isInteger(message.axis))
                        return "axis: integer expected";
                if (message.num_axes != null && message.hasOwnProperty("num_axes"))
                    if (!$util.isInteger(message.num_axes))
                        return "num_axes: integer expected";
                if (message.filler != null && message.hasOwnProperty("filler")) {
                    var error = $root.caffe.FillerParameter.verify(message.filler);
                    if (error)
                        return "filler." + error;
                }
                return null;
            };
    
            BiasParameter.fromObject = function fromObject(object) {
                if (object instanceof $root.caffe.BiasParameter)
                    return object;
                var message = new $root.caffe.BiasParameter();
                if (object.axis != null)
                    message.axis = object.axis | 0;
                if (object.num_axes != null)
                    message.num_axes = object.num_axes | 0;
                if (object.filler != null) {
                    if (typeof object.filler !== "object")
                        throw TypeError(".caffe.BiasParameter.filler: object expected");
                    message.filler = $root.caffe.FillerParameter.fromObject(object.filler);
                }
                return message;
            };
    
            BiasParameter.toObject = function toObject(message, options) {
                if (!options)
                    options = {};
                var object = {};
                if (options.defaults) {
                    object.axis = 1;
                    object.num_axes = 1;
                    object.filler = null;
                }
                if (message.axis != null && message.hasOwnProperty("axis"))
                    object.axis = message.axis;
                if (message.num_axes != null && message.hasOwnProperty("num_axes"))
                    object.num_axes = message.num_axes;
                if (message.filler != null && message.hasOwnProperty("filler"))
                    object.filler = $root.caffe.FillerParameter.toObject(message.filler, options);
                return object;
            };
    
            BiasParameter.prototype.toJSON = function toJSON() {
                return this.constructor.toObject(this, $protobuf.util.toJSONOptions);
            };
    
            return BiasParameter;
        })();
    
        caffe.ContrastiveLossParameter = (function() {
    
            function ContrastiveLossParameter(properties) {
                if (properties)
                    for (var keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                        if (properties[keys[i]] != null)
                            this[keys[i]] = properties[keys[i]];
            }
    
            ContrastiveLossParameter.prototype.margin = 1;
            ContrastiveLossParameter.prototype.legacy_version = false;
    
            ContrastiveLossParameter.create = function create(properties) {
                return new ContrastiveLossParameter(properties);
            };
    
            ContrastiveLossParameter.decode = function decode(reader, length) {
                if (!(reader instanceof $Reader))
                    reader = $Reader.create(reader);
                var end = length === undefined ? reader.len : reader.pos + length, message = new $root.caffe.ContrastiveLossParameter();
                while (reader.pos < end) {
                    var tag = reader.uint32();
                    switch (tag >>> 3) {
                    case 1:
                        message.margin = reader.float();
                        break;
                    case 2:
                        message.legacy_version = reader.bool();
                        break;
                    default:
                        reader.skipType(tag & 7);
                        break;
                    }
                }
                return message;
            };
    
            ContrastiveLossParameter.decodeText = function decodeText(reader, block) {
                if (!(reader instanceof $TextReader))
                    reader = $TextReader.create(reader);
                var message = new $root.caffe.ContrastiveLossParameter();
                reader.start(block);
                while (!reader.end(block)) {
                    var tag = reader.tag();
                    switch (tag) {
                    case "margin":
                        message.margin = reader.float();
                        break;
                    case "legacy_version":
                        message.legacy_version = reader.bool();
                        break;
                    default:
                        reader.handle(tag, message);
                        break;
                    }
                }
                return message;
            };
    
            ContrastiveLossParameter.verify = function verify(message) {
                if (typeof message !== "object" || message === null)
                    return "object expected";
                if (message.margin != null && message.hasOwnProperty("margin"))
                    if (typeof message.margin !== "number")
                        return "margin: number expected";
                if (message.legacy_version != null && message.hasOwnProperty("legacy_version"))
                    if (typeof message.legacy_version !== "boolean")
                        return "legacy_version: boolean expected";
                return null;
            };
    
            ContrastiveLossParameter.fromObject = function fromObject(object) {
                if (object instanceof $root.caffe.ContrastiveLossParameter)
                    return object;
                var message = new $root.caffe.ContrastiveLossParameter();
                if (object.margin != null)
                    message.margin = Number(object.margin);
                if (object.legacy_version != null)
                    message.legacy_version = Boolean(object.legacy_version);
                return message;
            };
    
            ContrastiveLossParameter.toObject = function toObject(message, options) {
                if (!options)
                    options = {};
                var object = {};
                if (options.defaults) {
                    object.margin = 1;
                    object.legacy_version = false;
                }
                if (message.margin != null && message.hasOwnProperty("margin"))
                    object.margin = options.json && !isFinite(message.margin) ? String(message.margin) : message.margin;
                if (message.legacy_version != null && message.hasOwnProperty("legacy_version"))
                    object.legacy_version = message.legacy_version;
                return object;
            };
    
            ContrastiveLossParameter.prototype.toJSON = function toJSON() {
                return this.constructor.toObject(this, $protobuf.util.toJSONOptions);
            };
    
            return ContrastiveLossParameter;
        })();
    
        caffe.ConvolutionParameter = (function() {
    
            function ConvolutionParameter(properties) {
                this.pad = [];
                this.kernel_size = [];
                this.stride = [];
                this.dilation = [];
                if (properties)
                    for (var keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                        if (properties[keys[i]] != null)
                            this[keys[i]] = properties[keys[i]];
            }
    
            ConvolutionParameter.prototype.num_output = 0;
            ConvolutionParameter.prototype.bias_term = true;
            ConvolutionParameter.prototype.pad = $util.emptyArray;
            ConvolutionParameter.prototype.kernel_size = $util.emptyArray;
            ConvolutionParameter.prototype.stride = $util.emptyArray;
            ConvolutionParameter.prototype.dilation = $util.emptyArray;
            ConvolutionParameter.prototype.pad_h = 0;
            ConvolutionParameter.prototype.pad_w = 0;
            ConvolutionParameter.prototype.kernel_h = 0;
            ConvolutionParameter.prototype.kernel_w = 0;
            ConvolutionParameter.prototype.stride_h = 0;
            ConvolutionParameter.prototype.stride_w = 0;
            ConvolutionParameter.prototype.group = 1;
            ConvolutionParameter.prototype.weight_filler = null;
            ConvolutionParameter.prototype.bias_filler = null;
            ConvolutionParameter.prototype.engine = 0;
            ConvolutionParameter.prototype.axis = 1;
            ConvolutionParameter.prototype.force_nd_im2col = false;
    
            ConvolutionParameter.create = function create(properties) {
                return new ConvolutionParameter(properties);
            };
    
            ConvolutionParameter.decode = function decode(reader, length) {
                if (!(reader instanceof $Reader))
                    reader = $Reader.create(reader);
                var end = length === undefined ? reader.len : reader.pos + length, message = new $root.caffe.ConvolutionParameter();
                while (reader.pos < end) {
                    var tag = reader.uint32();
                    switch (tag >>> 3) {
                    case 1:
                        message.num_output = reader.uint32();
                        break;
                    case 2:
                        message.bias_term = reader.bool();
                        break;
                    case 3:
                        if (!(message.pad && message.pad.length))
                            message.pad = [];
                        if ((tag & 7) === 2) {
                            var end2 = reader.uint32() + reader.pos;
                            while (reader.pos < end2)
                                message.pad.push(reader.uint32());
                        } else
                            message.pad.push(reader.uint32());
                        break;
                    case 4:
                        if (!(message.kernel_size && message.kernel_size.length))
                            message.kernel_size = [];
                        if ((tag & 7) === 2) {
                            var end2 = reader.uint32() + reader.pos;
                            while (reader.pos < end2)
                                message.kernel_size.push(reader.uint32());
                        } else
                            message.kernel_size.push(reader.uint32());
                        break;
                    case 6:
                        if (!(message.stride && message.stride.length))
                            message.stride = [];
                        if ((tag & 7) === 2) {
                            var end2 = reader.uint32() + reader.pos;
                            while (reader.pos < end2)
                                message.stride.push(reader.uint32());
                        } else
                            message.stride.push(reader.uint32());
                        break;
                    case 18:
                        if (!(message.dilation && message.dilation.length))
                            message.dilation = [];
                        if ((tag & 7) === 2) {
                            var end2 = reader.uint32() + reader.pos;
                            while (reader.pos < end2)
                                message.dilation.push(reader.uint32());
                        } else
                            message.dilation.push(reader.uint32());
                        break;
                    case 9:
                        message.pad_h = reader.uint32();
                        break;
                    case 10:
                        message.pad_w = reader.uint32();
                        break;
                    case 11:
                        message.kernel_h = reader.uint32();
                        break;
                    case 12:
                        message.kernel_w = reader.uint32();
                        break;
                    case 13:
                        message.stride_h = reader.uint32();
                        break;
                    case 14:
                        message.stride_w = reader.uint32();
                        break;
                    case 5:
                        message.group = reader.uint32();
                        break;
                    case 7:
                        message.weight_filler = $root.caffe.FillerParameter.decode(reader, reader.uint32());
                        break;
                    case 8:
                        message.bias_filler = $root.caffe.FillerParameter.decode(reader, reader.uint32());
                        break;
                    case 15:
                        message.engine = reader.int32();
                        break;
                    case 16:
                        message.axis = reader.int32();
                        break;
                    case 17:
                        message.force_nd_im2col = reader.bool();
                        break;
                    default:
                        reader.skipType(tag & 7);
                        break;
                    }
                }
                return message;
            };
    
            ConvolutionParameter.decodeText = function decodeText(reader, block) {
                if (!(reader instanceof $TextReader))
                    reader = $TextReader.create(reader);
                var message = new $root.caffe.ConvolutionParameter();
                reader.start(block);
                while (!reader.end(block)) {
                    var tag = reader.tag();
                    switch (tag) {
                    case "num_output":
                        message.num_output = reader.uint32();
                        break;
                    case "bias_term":
                        message.bias_term = reader.bool();
                        break;
                    case "pad":
                        if (!(message.pad && message.pad.length))
                            message.pad = [];
                        message.pad.push(reader.uint32());
                        break;
                    case "kernel_size":
                        if (!(message.kernel_size && message.kernel_size.length))
                            message.kernel_size = [];
                        message.kernel_size.push(reader.uint32());
                        break;
                    case "stride":
                        if (!(message.stride && message.stride.length))
                            message.stride = [];
                        message.stride.push(reader.uint32());
                        break;
                    case "dilation":
                        if (!(message.dilation && message.dilation.length))
                            message.dilation = [];
                        message.dilation.push(reader.uint32());
                        break;
                    case "pad_h":
                        message.pad_h = reader.uint32();
                        break;
                    case "pad_w":
                        message.pad_w = reader.uint32();
                        break;
                    case "kernel_h":
                        message.kernel_h = reader.uint32();
                        break;
                    case "kernel_w":
                        message.kernel_w = reader.uint32();
                        break;
                    case "stride_h":
                        message.stride_h = reader.uint32();
                        break;
                    case "stride_w":
                        message.stride_w = reader.uint32();
                        break;
                    case "group":
                        message.group = reader.uint32();
                        break;
                    case "weight_filler":
                        message.weight_filler = $root.caffe.FillerParameter.decodeText(reader, true);
                        break;
                    case "bias_filler":
                        message.bias_filler = $root.caffe.FillerParameter.decodeText(reader, true);
                        break;
                    case "engine":
                        message.engine = reader.enum($root.caffe.ConvolutionParameter.Engine);
                        break;
                    case "axis":
                        message.axis = reader.int32();
                        break;
                    case "force_nd_im2col":
                        message.force_nd_im2col = reader.bool();
                        break;
                    default:
                        reader.handle(tag, message);
                        break;
                    }
                }
                return message;
            };
    
            ConvolutionParameter.verify = function verify(message) {
                if (typeof message !== "object" || message === null)
                    return "object expected";
                if (message.num_output != null && message.hasOwnProperty("num_output"))
                    if (!$util.isInteger(message.num_output))
                        return "num_output: integer expected";
                if (message.bias_term != null && message.hasOwnProperty("bias_term"))
                    if (typeof message.bias_term !== "boolean")
                        return "bias_term: boolean expected";
                if (message.pad != null && message.hasOwnProperty("pad")) {
                    if (!Array.isArray(message.pad))
                        return "pad: array expected";
                    for (var i = 0; i < message.pad.length; ++i)
                        if (!$util.isInteger(message.pad[i]))
                            return "pad: integer[] expected";
                }
                if (message.kernel_size != null && message.hasOwnProperty("kernel_size")) {
                    if (!Array.isArray(message.kernel_size))
                        return "kernel_size: array expected";
                    for (var i = 0; i < message.kernel_size.length; ++i)
                        if (!$util.isInteger(message.kernel_size[i]))
                            return "kernel_size: integer[] expected";
                }
                if (message.stride != null && message.hasOwnProperty("stride")) {
                    if (!Array.isArray(message.stride))
                        return "stride: array expected";
                    for (var i = 0; i < message.stride.length; ++i)
                        if (!$util.isInteger(message.stride[i]))
                            return "stride: integer[] expected";
                }
                if (message.dilation != null && message.hasOwnProperty("dilation")) {
                    if (!Array.isArray(message.dilation))
                        return "dilation: array expected";
                    for (var i = 0; i < message.dilation.length; ++i)
                        if (!$util.isInteger(message.dilation[i]))
                            return "dilation: integer[] expected";
                }
                if (message.pad_h != null && message.hasOwnProperty("pad_h"))
                    if (!$util.isInteger(message.pad_h))
                        return "pad_h: integer expected";
                if (message.pad_w != null && message.hasOwnProperty("pad_w"))
                    if (!$util.isInteger(message.pad_w))
                        return "pad_w: integer expected";
                if (message.kernel_h != null && message.hasOwnProperty("kernel_h"))
                    if (!$util.isInteger(message.kernel_h))
                        return "kernel_h: integer expected";
                if (message.kernel_w != null && message.hasOwnProperty("kernel_w"))
                    if (!$util.isInteger(message.kernel_w))
                        return "kernel_w: integer expected";
                if (message.stride_h != null && message.hasOwnProperty("stride_h"))
                    if (!$util.isInteger(message.stride_h))
                        return "stride_h: integer expected";
                if (message.stride_w != null && message.hasOwnProperty("stride_w"))
                    if (!$util.isInteger(message.stride_w))
                        return "stride_w: integer expected";
                if (message.group != null && message.hasOwnProperty("group"))
                    if (!$util.isInteger(message.group))
                        return "group: integer expected";
                if (message.weight_filler != null && message.hasOwnProperty("weight_filler")) {
                    var error = $root.caffe.FillerParameter.verify(message.weight_filler);
                    if (error)
                        return "weight_filler." + error;
                }
                if (message.bias_filler != null && message.hasOwnProperty("bias_filler")) {
                    var error = $root.caffe.FillerParameter.verify(message.bias_filler);
                    if (error)
                        return "bias_filler." + error;
                }
                if (message.engine != null && message.hasOwnProperty("engine"))
                    switch (message.engine) {
                    default:
                        return "engine: enum value expected";
                    case 0:
                    case 1:
                    case 2:
                        break;
                    }
                if (message.axis != null && message.hasOwnProperty("axis"))
                    if (!$util.isInteger(message.axis))
                        return "axis: integer expected";
                if (message.force_nd_im2col != null && message.hasOwnProperty("force_nd_im2col"))
                    if (typeof message.force_nd_im2col !== "boolean")
                        return "force_nd_im2col: boolean expected";
                return null;
            };
    
            ConvolutionParameter.fromObject = function fromObject(object) {
                if (object instanceof $root.caffe.ConvolutionParameter)
                    return object;
                var message = new $root.caffe.ConvolutionParameter();
                if (object.num_output != null)
                    message.num_output = object.num_output >>> 0;
                if (object.bias_term != null)
                    message.bias_term = Boolean(object.bias_term);
                if (object.pad) {
                    if (!Array.isArray(object.pad))
                        throw TypeError(".caffe.ConvolutionParameter.pad: array expected");
                    message.pad = [];
                    for (var i = 0; i < object.pad.length; ++i)
                        message.pad[i] = object.pad[i] >>> 0;
                }
                if (object.kernel_size) {
                    if (!Array.isArray(object.kernel_size))
                        throw TypeError(".caffe.ConvolutionParameter.kernel_size: array expected");
                    message.kernel_size = [];
                    for (var i = 0; i < object.kernel_size.length; ++i)
                        message.kernel_size[i] = object.kernel_size[i] >>> 0;
                }
                if (object.stride) {
                    if (!Array.isArray(object.stride))
                        throw TypeError(".caffe.ConvolutionParameter.stride: array expected");
                    message.stride = [];
                    for (var i = 0; i < object.stride.length; ++i)
                        message.stride[i] = object.stride[i] >>> 0;
                }
                if (object.dilation) {
                    if (!Array.isArray(object.dilation))
                        throw TypeError(".caffe.ConvolutionParameter.dilation: array expected");
                    message.dilation = [];
                    for (var i = 0; i < object.dilation.length; ++i)
                        message.dilation[i] = object.dilation[i] >>> 0;
                }
                if (object.pad_h != null)
                    message.pad_h = object.pad_h >>> 0;
                if (object.pad_w != null)
                    message.pad_w = object.pad_w >>> 0;
                if (object.kernel_h != null)
                    message.kernel_h = object.kernel_h >>> 0;
                if (object.kernel_w != null)
                    message.kernel_w = object.kernel_w >>> 0;
                if (object.stride_h != null)
                    message.stride_h = object.stride_h >>> 0;
                if (object.stride_w != null)
                    message.stride_w = object.stride_w >>> 0;
                if (object.group != null)
                    message.group = object.group >>> 0;
                if (object.weight_filler != null) {
                    if (typeof object.weight_filler !== "object")
                        throw TypeError(".caffe.ConvolutionParameter.weight_filler: object expected");
                    message.weight_filler = $root.caffe.FillerParameter.fromObject(object.weight_filler);
                }
                if (object.bias_filler != null) {
                    if (typeof object.bias_filler !== "object")
                        throw TypeError(".caffe.ConvolutionParameter.bias_filler: object expected");
                    message.bias_filler = $root.caffe.FillerParameter.fromObject(object.bias_filler);
                }
                switch (object.engine) {
                case "DEFAULT":
                case 0:
                    message.engine = 0;
                    break;
                case "CAFFE":
                case 1:
                    message.engine = 1;
                    break;
                case "CUDNN":
                case 2:
                    message.engine = 2;
                    break;
                }
                if (object.axis != null)
                    message.axis = object.axis | 0;
                if (object.force_nd_im2col != null)
                    message.force_nd_im2col = Boolean(object.force_nd_im2col);
                return message;
            };
    
            ConvolutionParameter.toObject = function toObject(message, options) {
                if (!options)
                    options = {};
                var object = {};
                if (options.arrays || options.defaults) {
                    object.pad = [];
                    object.kernel_size = [];
                    object.stride = [];
                    object.dilation = [];
                }
                if (options.defaults) {
                    object.num_output = 0;
                    object.bias_term = true;
                    object.group = 1;
                    object.weight_filler = null;
                    object.bias_filler = null;
                    object.pad_h = 0;
                    object.pad_w = 0;
                    object.kernel_h = 0;
                    object.kernel_w = 0;
                    object.stride_h = 0;
                    object.stride_w = 0;
                    object.engine = options.enums === String ? "DEFAULT" : 0;
                    object.axis = 1;
                    object.force_nd_im2col = false;
                }
                if (message.num_output != null && message.hasOwnProperty("num_output"))
                    object.num_output = message.num_output;
                if (message.bias_term != null && message.hasOwnProperty("bias_term"))
                    object.bias_term = message.bias_term;
                if (message.pad && message.pad.length) {
                    object.pad = [];
                    for (var j = 0; j < message.pad.length; ++j)
                        object.pad[j] = message.pad[j];
                }
                if (message.kernel_size && message.kernel_size.length) {
                    object.kernel_size = [];
                    for (var j = 0; j < message.kernel_size.length; ++j)
                        object.kernel_size[j] = message.kernel_size[j];
                }
                if (message.group != null && message.hasOwnProperty("group"))
                    object.group = message.group;
                if (message.stride && message.stride.length) {
                    object.stride = [];
                    for (var j = 0; j < message.stride.length; ++j)
                        object.stride[j] = message.stride[j];
                }
                if (message.weight_filler != null && message.hasOwnProperty("weight_filler"))
                    object.weight_filler = $root.caffe.FillerParameter.toObject(message.weight_filler, options);
                if (message.bias_filler != null && message.hasOwnProperty("bias_filler"))
                    object.bias_filler = $root.caffe.FillerParameter.toObject(message.bias_filler, options);
                if (message.pad_h != null && message.hasOwnProperty("pad_h"))
                    object.pad_h = message.pad_h;
                if (message.pad_w != null && message.hasOwnProperty("pad_w"))
                    object.pad_w = message.pad_w;
                if (message.kernel_h != null && message.hasOwnProperty("kernel_h"))
                    object.kernel_h = message.kernel_h;
                if (message.kernel_w != null && message.hasOwnProperty("kernel_w"))
                    object.kernel_w = message.kernel_w;
                if (message.stride_h != null && message.hasOwnProperty("stride_h"))
                    object.stride_h = message.stride_h;
                if (message.stride_w != null && message.hasOwnProperty("stride_w"))
                    object.stride_w = message.stride_w;
                if (message.engine != null && message.hasOwnProperty("engine"))
                    object.engine = options.enums === String ? $root.caffe.ConvolutionParameter.Engine[message.engine] : message.engine;
                if (message.axis != null && message.hasOwnProperty("axis"))
                    object.axis = message.axis;
                if (message.force_nd_im2col != null && message.hasOwnProperty("force_nd_im2col"))
                    object.force_nd_im2col = message.force_nd_im2col;
                if (message.dilation && message.dilation.length) {
                    object.dilation = [];
                    for (var j = 0; j < message.dilation.length; ++j)
                        object.dilation[j] = message.dilation[j];
                }
                return object;
            };
    
            ConvolutionParameter.prototype.toJSON = function toJSON() {
                return this.constructor.toObject(this, $protobuf.util.toJSONOptions);
            };
    
            ConvolutionParameter.Engine = (function() {
                var valuesById = {}, values = Object.create(valuesById);
                values[valuesById[0] = "DEFAULT"] = 0;
                values[valuesById[1] = "CAFFE"] = 1;
                values[valuesById[2] = "CUDNN"] = 2;
                return values;
            })();
    
            return ConvolutionParameter;
        })();
    
        caffe.CropParameter = (function() {
    
            function CropParameter(properties) {
                this.offset = [];
                if (properties)
                    for (var keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                        if (properties[keys[i]] != null)
                            this[keys[i]] = properties[keys[i]];
            }
    
            CropParameter.prototype.axis = 2;
            CropParameter.prototype.offset = $util.emptyArray;
    
            CropParameter.create = function create(properties) {
                return new CropParameter(properties);
            };
    
            CropParameter.decode = function decode(reader, length) {
                if (!(reader instanceof $Reader))
                    reader = $Reader.create(reader);
                var end = length === undefined ? reader.len : reader.pos + length, message = new $root.caffe.CropParameter();
                while (reader.pos < end) {
                    var tag = reader.uint32();
                    switch (tag >>> 3) {
                    case 1:
                        message.axis = reader.int32();
                        break;
                    case 2:
                        if (!(message.offset && message.offset.length))
                            message.offset = [];
                        if ((tag & 7) === 2) {
                            var end2 = reader.uint32() + reader.pos;
                            while (reader.pos < end2)
                                message.offset.push(reader.uint32());
                        } else
                            message.offset.push(reader.uint32());
                        break;
                    default:
                        reader.skipType(tag & 7);
                        break;
                    }
                }
                return message;
            };
    
            CropParameter.decodeText = function decodeText(reader, block) {
                if (!(reader instanceof $TextReader))
                    reader = $TextReader.create(reader);
                var message = new $root.caffe.CropParameter();
                reader.start(block);
                while (!reader.end(block)) {
                    var tag = reader.tag();
                    switch (tag) {
                    case "axis":
                        message.axis = reader.int32();
                        break;
                    case "offset":
                        if (!(message.offset && message.offset.length))
                            message.offset = [];
                        message.offset.push(reader.uint32());
                        break;
                    default:
                        reader.handle(tag, message);
                        break;
                    }
                }
                return message;
            };
    
            CropParameter.verify = function verify(message) {
                if (typeof message !== "object" || message === null)
                    return "object expected";
                if (message.axis != null && message.hasOwnProperty("axis"))
                    if (!$util.isInteger(message.axis))
                        return "axis: integer expected";
                if (message.offset != null && message.hasOwnProperty("offset")) {
                    if (!Array.isArray(message.offset))
                        return "offset: array expected";
                    for (var i = 0; i < message.offset.length; ++i)
                        if (!$util.isInteger(message.offset[i]))
                            return "offset: integer[] expected";
                }
                return null;
            };
    
            CropParameter.fromObject = function fromObject(object) {
                if (object instanceof $root.caffe.CropParameter)
                    return object;
                var message = new $root.caffe.CropParameter();
                if (object.axis != null)
                    message.axis = object.axis | 0;
                if (object.offset) {
                    if (!Array.isArray(object.offset))
                        throw TypeError(".caffe.CropParameter.offset: array expected");
                    message.offset = [];
                    for (var i = 0; i < object.offset.length; ++i)
                        message.offset[i] = object.offset[i] >>> 0;
                }
                return message;
            };
    
            CropParameter.toObject = function toObject(message, options) {
                if (!options)
                    options = {};
                var object = {};
                if (options.arrays || options.defaults)
                    object.offset = [];
                if (options.defaults)
                    object.axis = 2;
                if (message.axis != null && message.hasOwnProperty("axis"))
                    object.axis = message.axis;
                if (message.offset && message.offset.length) {
                    object.offset = [];
                    for (var j = 0; j < message.offset.length; ++j)
                        object.offset[j] = message.offset[j];
                }
                return object;
            };
    
            CropParameter.prototype.toJSON = function toJSON() {
                return this.constructor.toObject(this, $protobuf.util.toJSONOptions);
            };
    
            return CropParameter;
        })();
    
        caffe.DataParameter = (function() {
    
            function DataParameter(properties) {
                if (properties)
                    for (var keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                        if (properties[keys[i]] != null)
                            this[keys[i]] = properties[keys[i]];
            }
    
            DataParameter.prototype.source = "";
            DataParameter.prototype.batch_size = 0;
            DataParameter.prototype.rand_skip = 0;
            DataParameter.prototype.backend = 0;
            DataParameter.prototype.scale = 1;
            DataParameter.prototype.mean_file = "";
            DataParameter.prototype.crop_size = 0;
            DataParameter.prototype.mirror = false;
            DataParameter.prototype.force_encoded_color = false;
            DataParameter.prototype.prefetch = 4;
    
            DataParameter.create = function create(properties) {
                return new DataParameter(properties);
            };
    
            DataParameter.decode = function decode(reader, length) {
                if (!(reader instanceof $Reader))
                    reader = $Reader.create(reader);
                var end = length === undefined ? reader.len : reader.pos + length, message = new $root.caffe.DataParameter();
                while (reader.pos < end) {
                    var tag = reader.uint32();
                    switch (tag >>> 3) {
                    case 1:
                        message.source = reader.string();
                        break;
                    case 4:
                        message.batch_size = reader.uint32();
                        break;
                    case 7:
                        message.rand_skip = reader.uint32();
                        break;
                    case 8:
                        message.backend = reader.int32();
                        break;
                    case 2:
                        message.scale = reader.float();
                        break;
                    case 3:
                        message.mean_file = reader.string();
                        break;
                    case 5:
                        message.crop_size = reader.uint32();
                        break;
                    case 6:
                        message.mirror = reader.bool();
                        break;
                    case 9:
                        message.force_encoded_color = reader.bool();
                        break;
                    case 10:
                        message.prefetch = reader.uint32();
                        break;
                    default:
                        reader.skipType(tag & 7);
                        break;
                    }
                }
                return message;
            };
    
            DataParameter.decodeText = function decodeText(reader, block) {
                if (!(reader instanceof $TextReader))
                    reader = $TextReader.create(reader);
                var message = new $root.caffe.DataParameter();
                reader.start(block);
                while (!reader.end(block)) {
                    var tag = reader.tag();
                    switch (tag) {
                    case "source":
                        message.source = reader.string();
                        break;
                    case "batch_size":
                        message.batch_size = reader.uint32();
                        break;
                    case "rand_skip":
                        message.rand_skip = reader.uint32();
                        break;
                    case "backend":
                        message.backend = reader.enum($root.caffe.DataParameter.DB);
                        break;
                    case "scale":
                        message.scale = reader.float();
                        break;
                    case "mean_file":
                        message.mean_file = reader.string();
                        break;
                    case "crop_size":
                        message.crop_size = reader.uint32();
                        break;
                    case "mirror":
                        message.mirror = reader.bool();
                        break;
                    case "force_encoded_color":
                        message.force_encoded_color = reader.bool();
                        break;
                    case "prefetch":
                        message.prefetch = reader.uint32();
                        break;
                    default:
                        reader.handle(tag, message);
                        break;
                    }
                }
                return message;
            };
    
            DataParameter.verify = function verify(message) {
                if (typeof message !== "object" || message === null)
                    return "object expected";
                if (message.source != null && message.hasOwnProperty("source"))
                    if (!$util.isString(message.source))
                        return "source: string expected";
                if (message.batch_size != null && message.hasOwnProperty("batch_size"))
                    if (!$util.isInteger(message.batch_size))
                        return "batch_size: integer expected";
                if (message.rand_skip != null && message.hasOwnProperty("rand_skip"))
                    if (!$util.isInteger(message.rand_skip))
                        return "rand_skip: integer expected";
                if (message.backend != null && message.hasOwnProperty("backend"))
                    switch (message.backend) {
                    default:
                        return "backend: enum value expected";
                    case 0:
                    case 1:
                        break;
                    }
                if (message.scale != null && message.hasOwnProperty("scale"))
                    if (typeof message.scale !== "number")
                        return "scale: number expected";
                if (message.mean_file != null && message.hasOwnProperty("mean_file"))
                    if (!$util.isString(message.mean_file))
                        return "mean_file: string expected";
                if (message.crop_size != null && message.hasOwnProperty("crop_size"))
                    if (!$util.isInteger(message.crop_size))
                        return "crop_size: integer expected";
                if (message.mirror != null && message.hasOwnProperty("mirror"))
                    if (typeof message.mirror !== "boolean")
                        return "mirror: boolean expected";
                if (message.force_encoded_color != null && message.hasOwnProperty("force_encoded_color"))
                    if (typeof message.force_encoded_color !== "boolean")
                        return "force_encoded_color: boolean expected";
                if (message.prefetch != null && message.hasOwnProperty("prefetch"))
                    if (!$util.isInteger(message.prefetch))
                        return "prefetch: integer expected";
                return null;
            };
    
            DataParameter.fromObject = function fromObject(object) {
                if (object instanceof $root.caffe.DataParameter)
                    return object;
                var message = new $root.caffe.DataParameter();
                if (object.source != null)
                    message.source = String(object.source);
                if (object.batch_size != null)
                    message.batch_size = object.batch_size >>> 0;
                if (object.rand_skip != null)
                    message.rand_skip = object.rand_skip >>> 0;
                switch (object.backend) {
                case "LEVELDB":
                case 0:
                    message.backend = 0;
                    break;
                case "LMDB":
                case 1:
                    message.backend = 1;
                    break;
                }
                if (object.scale != null)
                    message.scale = Number(object.scale);
                if (object.mean_file != null)
                    message.mean_file = String(object.mean_file);
                if (object.crop_size != null)
                    message.crop_size = object.crop_size >>> 0;
                if (object.mirror != null)
                    message.mirror = Boolean(object.mirror);
                if (object.force_encoded_color != null)
                    message.force_encoded_color = Boolean(object.force_encoded_color);
                if (object.prefetch != null)
                    message.prefetch = object.prefetch >>> 0;
                return message;
            };
    
            DataParameter.toObject = function toObject(message, options) {
                if (!options)
                    options = {};
                var object = {};
                if (options.defaults) {
                    object.source = "";
                    object.scale = 1;
                    object.mean_file = "";
                    object.batch_size = 0;
                    object.crop_size = 0;
                    object.mirror = false;
                    object.rand_skip = 0;
                    object.backend = options.enums === String ? "LEVELDB" : 0;
                    object.force_encoded_color = false;
                    object.prefetch = 4;
                }
                if (message.source != null && message.hasOwnProperty("source"))
                    object.source = message.source;
                if (message.scale != null && message.hasOwnProperty("scale"))
                    object.scale = options.json && !isFinite(message.scale) ? String(message.scale) : message.scale;
                if (message.mean_file != null && message.hasOwnProperty("mean_file"))
                    object.mean_file = message.mean_file;
                if (message.batch_size != null && message.hasOwnProperty("batch_size"))
                    object.batch_size = message.batch_size;
                if (message.crop_size != null && message.hasOwnProperty("crop_size"))
                    object.crop_size = message.crop_size;
                if (message.mirror != null && message.hasOwnProperty("mirror"))
                    object.mirror = message.mirror;
                if (message.rand_skip != null && message.hasOwnProperty("rand_skip"))
                    object.rand_skip = message.rand_skip;
                if (message.backend != null && message.hasOwnProperty("backend"))
                    object.backend = options.enums === String ? $root.caffe.DataParameter.DB[message.backend] : message.backend;
                if (message.force_encoded_color != null && message.hasOwnProperty("force_encoded_color"))
                    object.force_encoded_color = message.force_encoded_color;
                if (message.prefetch != null && message.hasOwnProperty("prefetch"))
                    object.prefetch = message.prefetch;
                return object;
            };
    
            DataParameter.prototype.toJSON = function toJSON() {
                return this.constructor.toObject(this, $protobuf.util.toJSONOptions);
            };
    
            DataParameter.DB = (function() {
                var valuesById = {}, values = Object.create(valuesById);
                values[valuesById[0] = "LEVELDB"] = 0;
                values[valuesById[1] = "LMDB"] = 1;
                return values;
            })();
    
            return DataParameter;
        })();
    
        caffe.DropoutParameter = (function() {
    
            function DropoutParameter(properties) {
                if (properties)
                    for (var keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                        if (properties[keys[i]] != null)
                            this[keys[i]] = properties[keys[i]];
            }
    
            DropoutParameter.prototype.dropout_ratio = 0.5;
    
            DropoutParameter.create = function create(properties) {
                return new DropoutParameter(properties);
            };
    
            DropoutParameter.decode = function decode(reader, length) {
                if (!(reader instanceof $Reader))
                    reader = $Reader.create(reader);
                var end = length === undefined ? reader.len : reader.pos + length, message = new $root.caffe.DropoutParameter();
                while (reader.pos < end) {
                    var tag = reader.uint32();
                    switch (tag >>> 3) {
                    case 1:
                        message.dropout_ratio = reader.float();
                        break;
                    default:
                        reader.skipType(tag & 7);
                        break;
                    }
                }
                return message;
            };
    
            DropoutParameter.decodeText = function decodeText(reader, block) {
                if (!(reader instanceof $TextReader))
                    reader = $TextReader.create(reader);
                var message = new $root.caffe.DropoutParameter();
                reader.start(block);
                while (!reader.end(block)) {
                    var tag = reader.tag();
                    switch (tag) {
                    case "dropout_ratio":
                        message.dropout_ratio = reader.float();
                        break;
                    default:
                        reader.handle(tag, message);
                        break;
                    }
                }
                return message;
            };
    
            DropoutParameter.verify = function verify(message) {
                if (typeof message !== "object" || message === null)
                    return "object expected";
                if (message.dropout_ratio != null && message.hasOwnProperty("dropout_ratio"))
                    if (typeof message.dropout_ratio !== "number")
                        return "dropout_ratio: number expected";
                return null;
            };
    
            DropoutParameter.fromObject = function fromObject(object) {
                if (object instanceof $root.caffe.DropoutParameter)
                    return object;
                var message = new $root.caffe.DropoutParameter();
                if (object.dropout_ratio != null)
                    message.dropout_ratio = Number(object.dropout_ratio);
                return message;
            };
    
            DropoutParameter.toObject = function toObject(message, options) {
                if (!options)
                    options = {};
                var object = {};
                if (options.defaults)
                    object.dropout_ratio = 0.5;
                if (message.dropout_ratio != null && message.hasOwnProperty("dropout_ratio"))
                    object.dropout_ratio = options.json && !isFinite(message.dropout_ratio) ? String(message.dropout_ratio) : message.dropout_ratio;
                return object;
            };
    
            DropoutParameter.prototype.toJSON = function toJSON() {
                return this.constructor.toObject(this, $protobuf.util.toJSONOptions);
            };
    
            return DropoutParameter;
        })();
    
        caffe.DummyDataParameter = (function() {
    
            function DummyDataParameter(properties) {
                this.data_filler = [];
                this.shape = [];
                this.num = [];
                this.channels = [];
                this.height = [];
                this.width = [];
                if (properties)
                    for (var keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                        if (properties[keys[i]] != null)
                            this[keys[i]] = properties[keys[i]];
            }
    
            DummyDataParameter.prototype.data_filler = $util.emptyArray;
            DummyDataParameter.prototype.shape = $util.emptyArray;
            DummyDataParameter.prototype.num = $util.emptyArray;
            DummyDataParameter.prototype.channels = $util.emptyArray;
            DummyDataParameter.prototype.height = $util.emptyArray;
            DummyDataParameter.prototype.width = $util.emptyArray;
    
            DummyDataParameter.create = function create(properties) {
                return new DummyDataParameter(properties);
            };
    
            DummyDataParameter.decode = function decode(reader, length) {
                if (!(reader instanceof $Reader))
                    reader = $Reader.create(reader);
                var end = length === undefined ? reader.len : reader.pos + length, message = new $root.caffe.DummyDataParameter();
                while (reader.pos < end) {
                    var tag = reader.uint32();
                    switch (tag >>> 3) {
                    case 1:
                        if (!(message.data_filler && message.data_filler.length))
                            message.data_filler = [];
                        message.data_filler.push($root.caffe.FillerParameter.decode(reader, reader.uint32()));
                        break;
                    case 6:
                        if (!(message.shape && message.shape.length))
                            message.shape = [];
                        message.shape.push($root.caffe.BlobShape.decode(reader, reader.uint32()));
                        break;
                    case 2:
                        if (!(message.num && message.num.length))
                            message.num = [];
                        if ((tag & 7) === 2) {
                            var end2 = reader.uint32() + reader.pos;
                            while (reader.pos < end2)
                                message.num.push(reader.uint32());
                        } else
                            message.num.push(reader.uint32());
                        break;
                    case 3:
                        if (!(message.channels && message.channels.length))
                            message.channels = [];
                        if ((tag & 7) === 2) {
                            var end2 = reader.uint32() + reader.pos;
                            while (reader.pos < end2)
                                message.channels.push(reader.uint32());
                        } else
                            message.channels.push(reader.uint32());
                        break;
                    case 4:
                        if (!(message.height && message.height.length))
                            message.height = [];
                        if ((tag & 7) === 2) {
                            var end2 = reader.uint32() + reader.pos;
                            while (reader.pos < end2)
                                message.height.push(reader.uint32());
                        } else
                            message.height.push(reader.uint32());
                        break;
                    case 5:
                        if (!(message.width && message.width.length))
                            message.width = [];
                        if ((tag & 7) === 2) {
                            var end2 = reader.uint32() + reader.pos;
                            while (reader.pos < end2)
                                message.width.push(reader.uint32());
                        } else
                            message.width.push(reader.uint32());
                        break;
                    default:
                        reader.skipType(tag & 7);
                        break;
                    }
                }
                return message;
            };
    
            DummyDataParameter.decodeText = function decodeText(reader, block) {
                if (!(reader instanceof $TextReader))
                    reader = $TextReader.create(reader);
                var message = new $root.caffe.DummyDataParameter();
                reader.start(block);
                while (!reader.end(block)) {
                    var tag = reader.tag();
                    switch (tag) {
                    case "data_filler":
                        if (!(message.data_filler && message.data_filler.length))
                            message.data_filler = [];
                        message.data_filler.push($root.caffe.FillerParameter.decodeText(reader, true));
                        break;
                    case "shape":
                        if (!(message.shape && message.shape.length))
                            message.shape = [];
                        message.shape.push($root.caffe.BlobShape.decodeText(reader, true));
                        break;
                    case "num":
                        if (!(message.num && message.num.length))
                            message.num = [];
                        message.num.push(reader.uint32());
                        break;
                    case "channels":
                        if (!(message.channels && message.channels.length))
                            message.channels = [];
                        message.channels.push(reader.uint32());
                        break;
                    case "height":
                        if (!(message.height && message.height.length))
                            message.height = [];
                        message.height.push(reader.uint32());
                        break;
                    case "width":
                        if (!(message.width && message.width.length))
                            message.width = [];
                        message.width.push(reader.uint32());
                        break;
                    default:
                        reader.handle(tag, message);
                        break;
                    }
                }
                return message;
            };
    
            DummyDataParameter.verify = function verify(message) {
                if (typeof message !== "object" || message === null)
                    return "object expected";
                if (message.data_filler != null && message.hasOwnProperty("data_filler")) {
                    if (!Array.isArray(message.data_filler))
                        return "data_filler: array expected";
                    for (var i = 0; i < message.data_filler.length; ++i) {
                        var error = $root.caffe.FillerParameter.verify(message.data_filler[i]);
                        if (error)
                            return "data_filler." + error;
                    }
                }
                if (message.shape != null && message.hasOwnProperty("shape")) {
                    if (!Array.isArray(message.shape))
                        return "shape: array expected";
                    for (var i = 0; i < message.shape.length; ++i) {
                        var error = $root.caffe.BlobShape.verify(message.shape[i]);
                        if (error)
                            return "shape." + error;
                    }
                }
                if (message.num != null && message.hasOwnProperty("num")) {
                    if (!Array.isArray(message.num))
                        return "num: array expected";
                    for (var i = 0; i < message.num.length; ++i)
                        if (!$util.isInteger(message.num[i]))
                            return "num: integer[] expected";
                }
                if (message.channels != null && message.hasOwnProperty("channels")) {
                    if (!Array.isArray(message.channels))
                        return "channels: array expected";
                    for (var i = 0; i < message.channels.length; ++i)
                        if (!$util.isInteger(message.channels[i]))
                            return "channels: integer[] expected";
                }
                if (message.height != null && message.hasOwnProperty("height")) {
                    if (!Array.isArray(message.height))
                        return "height: array expected";
                    for (var i = 0; i < message.height.length; ++i)
                        if (!$util.isInteger(message.height[i]))
                            return "height: integer[] expected";
                }
                if (message.width != null && message.hasOwnProperty("width")) {
                    if (!Array.isArray(message.width))
                        return "width: array expected";
                    for (var i = 0; i < message.width.length; ++i)
                        if (!$util.isInteger(message.width[i]))
                            return "width: integer[] expected";
                }
                return null;
            };
    
            DummyDataParameter.fromObject = function fromObject(object) {
                if (object instanceof $root.caffe.DummyDataParameter)
                    return object;
                var message = new $root.caffe.DummyDataParameter();
                if (object.data_filler) {
                    if (!Array.isArray(object.data_filler))
                        throw TypeError(".caffe.DummyDataParameter.data_filler: array expected");
                    message.data_filler = [];
                    for (var i = 0; i < object.data_filler.length; ++i) {
                        if (typeof object.data_filler[i] !== "object")
                            throw TypeError(".caffe.DummyDataParameter.data_filler: object expected");
                        message.data_filler[i] = $root.caffe.FillerParameter.fromObject(object.data_filler[i]);
                    }
                }
                if (object.shape) {
                    if (!Array.isArray(object.shape))
                        throw TypeError(".caffe.DummyDataParameter.shape: array expected");
                    message.shape = [];
                    for (var i = 0; i < object.shape.length; ++i) {
                        if (typeof object.shape[i] !== "object")
                            throw TypeError(".caffe.DummyDataParameter.shape: object expected");
                        message.shape[i] = $root.caffe.BlobShape.fromObject(object.shape[i]);
                    }
                }
                if (object.num) {
                    if (!Array.isArray(object.num))
                        throw TypeError(".caffe.DummyDataParameter.num: array expected");
                    message.num = [];
                    for (var i = 0; i < object.num.length; ++i)
                        message.num[i] = object.num[i] >>> 0;
                }
                if (object.channels) {
                    if (!Array.isArray(object.channels))
                        throw TypeError(".caffe.DummyDataParameter.channels: array expected");
                    message.channels = [];
                    for (var i = 0; i < object.channels.length; ++i)
                        message.channels[i] = object.channels[i] >>> 0;
                }
                if (object.height) {
                    if (!Array.isArray(object.height))
                        throw TypeError(".caffe.DummyDataParameter.height: array expected");
                    message.height = [];
                    for (var i = 0; i < object.height.length; ++i)
                        message.height[i] = object.height[i] >>> 0;
                }
                if (object.width) {
                    if (!Array.isArray(object.width))
                        throw TypeError(".caffe.DummyDataParameter.width: array expected");
                    message.width = [];
                    for (var i = 0; i < object.width.length; ++i)
                        message.width[i] = object.width[i] >>> 0;
                }
                return message;
            };
    
            DummyDataParameter.toObject = function toObject(message, options) {
                if (!options)
                    options = {};
                var object = {};
                if (options.arrays || options.defaults) {
                    object.data_filler = [];
                    object.num = [];
                    object.channels = [];
                    object.height = [];
                    object.width = [];
                    object.shape = [];
                }
                if (message.data_filler && message.data_filler.length) {
                    object.data_filler = [];
                    for (var j = 0; j < message.data_filler.length; ++j)
                        object.data_filler[j] = $root.caffe.FillerParameter.toObject(message.data_filler[j], options);
                }
                if (message.num && message.num.length) {
                    object.num = [];
                    for (var j = 0; j < message.num.length; ++j)
                        object.num[j] = message.num[j];
                }
                if (message.channels && message.channels.length) {
                    object.channels = [];
                    for (var j = 0; j < message.channels.length; ++j)
                        object.channels[j] = message.channels[j];
                }
                if (message.height && message.height.length) {
                    object.height = [];
                    for (var j = 0; j < message.height.length; ++j)
                        object.height[j] = message.height[j];
                }
                if (message.width && message.width.length) {
                    object.width = [];
                    for (var j = 0; j < message.width.length; ++j)
                        object.width[j] = message.width[j];
                }
                if (message.shape && message.shape.length) {
                    object.shape = [];
                    for (var j = 0; j < message.shape.length; ++j)
                        object.shape[j] = $root.caffe.BlobShape.toObject(message.shape[j], options);
                }
                return object;
            };
    
            DummyDataParameter.prototype.toJSON = function toJSON() {
                return this.constructor.toObject(this, $protobuf.util.toJSONOptions);
            };
    
            return DummyDataParameter;
        })();
    
        caffe.EltwiseParameter = (function() {
    
            function EltwiseParameter(properties) {
                this.coeff = [];
                if (properties)
                    for (var keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                        if (properties[keys[i]] != null)
                            this[keys[i]] = properties[keys[i]];
            }
    
            EltwiseParameter.prototype.operation = 1;
            EltwiseParameter.prototype.coeff = $util.emptyArray;
            EltwiseParameter.prototype.stable_prod_grad = true;
    
            EltwiseParameter.create = function create(properties) {
                return new EltwiseParameter(properties);
            };
    
            EltwiseParameter.decode = function decode(reader, length) {
                if (!(reader instanceof $Reader))
                    reader = $Reader.create(reader);
                var end = length === undefined ? reader.len : reader.pos + length, message = new $root.caffe.EltwiseParameter();
                while (reader.pos < end) {
                    var tag = reader.uint32();
                    switch (tag >>> 3) {
                    case 1:
                        message.operation = reader.int32();
                        break;
                    case 2:
                        if (!(message.coeff && message.coeff.length))
                            message.coeff = [];
                        if ((tag & 7) === 2) {
                            var end2 = reader.uint32() + reader.pos;
                            while (reader.pos < end2)
                                message.coeff.push(reader.float());
                        } else
                            message.coeff.push(reader.float());
                        break;
                    case 3:
                        message.stable_prod_grad = reader.bool();
                        break;
                    default:
                        reader.skipType(tag & 7);
                        break;
                    }
                }
                return message;
            };
    
            EltwiseParameter.decodeText = function decodeText(reader, block) {
                if (!(reader instanceof $TextReader))
                    reader = $TextReader.create(reader);
                var message = new $root.caffe.EltwiseParameter();
                reader.start(block);
                while (!reader.end(block)) {
                    var tag = reader.tag();
                    switch (tag) {
                    case "operation":
                        message.operation = reader.enum($root.caffe.EltwiseParameter.EltwiseOp);
                        break;
                    case "coeff":
                        if (!(message.coeff && message.coeff.length))
                            message.coeff = [];
                        message.coeff.push(reader.float());
                        break;
                    case "stable_prod_grad":
                        message.stable_prod_grad = reader.bool();
                        break;
                    default:
                        reader.handle(tag, message);
                        break;
                    }
                }
                return message;
            };
    
            EltwiseParameter.verify = function verify(message) {
                if (typeof message !== "object" || message === null)
                    return "object expected";
                if (message.operation != null && message.hasOwnProperty("operation"))
                    switch (message.operation) {
                    default:
                        return "operation: enum value expected";
                    case 0:
                    case 1:
                    case 2:
                        break;
                    }
                if (message.coeff != null && message.hasOwnProperty("coeff")) {
                    if (!Array.isArray(message.coeff))
                        return "coeff: array expected";
                    for (var i = 0; i < message.coeff.length; ++i)
                        if (typeof message.coeff[i] !== "number")
                            return "coeff: number[] expected";
                }
                if (message.stable_prod_grad != null && message.hasOwnProperty("stable_prod_grad"))
                    if (typeof message.stable_prod_grad !== "boolean")
                        return "stable_prod_grad: boolean expected";
                return null;
            };
    
            EltwiseParameter.fromObject = function fromObject(object) {
                if (object instanceof $root.caffe.EltwiseParameter)
                    return object;
                var message = new $root.caffe.EltwiseParameter();
                switch (object.operation) {
                case "PROD":
                case 0:
                    message.operation = 0;
                    break;
                case "SUM":
                case 1:
                    message.operation = 1;
                    break;
                case "MAX":
                case 2:
                    message.operation = 2;
                    break;
                }
                if (object.coeff) {
                    if (!Array.isArray(object.coeff))
                        throw TypeError(".caffe.EltwiseParameter.coeff: array expected");
                    message.coeff = [];
                    for (var i = 0; i < object.coeff.length; ++i)
                        message.coeff[i] = Number(object.coeff[i]);
                }
                if (object.stable_prod_grad != null)
                    message.stable_prod_grad = Boolean(object.stable_prod_grad);
                return message;
            };
    
            EltwiseParameter.toObject = function toObject(message, options) {
                if (!options)
                    options = {};
                var object = {};
                if (options.arrays || options.defaults)
                    object.coeff = [];
                if (options.defaults) {
                    object.operation = options.enums === String ? "SUM" : 1;
                    object.stable_prod_grad = true;
                }
                if (message.operation != null && message.hasOwnProperty("operation"))
                    object.operation = options.enums === String ? $root.caffe.EltwiseParameter.EltwiseOp[message.operation] : message.operation;
                if (message.coeff && message.coeff.length) {
                    object.coeff = [];
                    for (var j = 0; j < message.coeff.length; ++j)
                        object.coeff[j] = options.json && !isFinite(message.coeff[j]) ? String(message.coeff[j]) : message.coeff[j];
                }
                if (message.stable_prod_grad != null && message.hasOwnProperty("stable_prod_grad"))
                    object.stable_prod_grad = message.stable_prod_grad;
                return object;
            };
    
            EltwiseParameter.prototype.toJSON = function toJSON() {
                return this.constructor.toObject(this, $protobuf.util.toJSONOptions);
            };
    
            EltwiseParameter.EltwiseOp = (function() {
                var valuesById = {}, values = Object.create(valuesById);
                values[valuesById[0] = "PROD"] = 0;
                values[valuesById[1] = "SUM"] = 1;
                values[valuesById[2] = "MAX"] = 2;
                return values;
            })();
    
            return EltwiseParameter;
        })();
    
        caffe.ELUParameter = (function() {
    
            function ELUParameter(properties) {
                if (properties)
                    for (var keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                        if (properties[keys[i]] != null)
                            this[keys[i]] = properties[keys[i]];
            }
    
            ELUParameter.prototype.alpha = 1;
    
            ELUParameter.create = function create(properties) {
                return new ELUParameter(properties);
            };
    
            ELUParameter.decode = function decode(reader, length) {
                if (!(reader instanceof $Reader))
                    reader = $Reader.create(reader);
                var end = length === undefined ? reader.len : reader.pos + length, message = new $root.caffe.ELUParameter();
                while (reader.pos < end) {
                    var tag = reader.uint32();
                    switch (tag >>> 3) {
                    case 1:
                        message.alpha = reader.float();
                        break;
                    default:
                        reader.skipType(tag & 7);
                        break;
                    }
                }
                return message;
            };
    
            ELUParameter.decodeText = function decodeText(reader, block) {
                if (!(reader instanceof $TextReader))
                    reader = $TextReader.create(reader);
                var message = new $root.caffe.ELUParameter();
                reader.start(block);
                while (!reader.end(block)) {
                    var tag = reader.tag();
                    switch (tag) {
                    case "alpha":
                        message.alpha = reader.float();
                        break;
                    default:
                        reader.handle(tag, message);
                        break;
                    }
                }
                return message;
            };
    
            ELUParameter.verify = function verify(message) {
                if (typeof message !== "object" || message === null)
                    return "object expected";
                if (message.alpha != null && message.hasOwnProperty("alpha"))
                    if (typeof message.alpha !== "number")
                        return "alpha: number expected";
                return null;
            };
    
            ELUParameter.fromObject = function fromObject(object) {
                if (object instanceof $root.caffe.ELUParameter)
                    return object;
                var message = new $root.caffe.ELUParameter();
                if (object.alpha != null)
                    message.alpha = Number(object.alpha);
                return message;
            };
    
            ELUParameter.toObject = function toObject(message, options) {
                if (!options)
                    options = {};
                var object = {};
                if (options.defaults)
                    object.alpha = 1;
                if (message.alpha != null && message.hasOwnProperty("alpha"))
                    object.alpha = options.json && !isFinite(message.alpha) ? String(message.alpha) : message.alpha;
                return object;
            };
    
            ELUParameter.prototype.toJSON = function toJSON() {
                return this.constructor.toObject(this, $protobuf.util.toJSONOptions);
            };
    
            return ELUParameter;
        })();
    
        caffe.EmbedParameter = (function() {
    
            function EmbedParameter(properties) {
                if (properties)
                    for (var keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                        if (properties[keys[i]] != null)
                            this[keys[i]] = properties[keys[i]];
            }
    
            EmbedParameter.prototype.num_output = 0;
            EmbedParameter.prototype.input_dim = 0;
            EmbedParameter.prototype.bias_term = true;
            EmbedParameter.prototype.weight_filler = null;
            EmbedParameter.prototype.bias_filler = null;
    
            EmbedParameter.create = function create(properties) {
                return new EmbedParameter(properties);
            };
    
            EmbedParameter.decode = function decode(reader, length) {
                if (!(reader instanceof $Reader))
                    reader = $Reader.create(reader);
                var end = length === undefined ? reader.len : reader.pos + length, message = new $root.caffe.EmbedParameter();
                while (reader.pos < end) {
                    var tag = reader.uint32();
                    switch (tag >>> 3) {
                    case 1:
                        message.num_output = reader.uint32();
                        break;
                    case 2:
                        message.input_dim = reader.uint32();
                        break;
                    case 3:
                        message.bias_term = reader.bool();
                        break;
                    case 4:
                        message.weight_filler = $root.caffe.FillerParameter.decode(reader, reader.uint32());
                        break;
                    case 5:
                        message.bias_filler = $root.caffe.FillerParameter.decode(reader, reader.uint32());
                        break;
                    default:
                        reader.skipType(tag & 7);
                        break;
                    }
                }
                return message;
            };
    
            EmbedParameter.decodeText = function decodeText(reader, block) {
                if (!(reader instanceof $TextReader))
                    reader = $TextReader.create(reader);
                var message = new $root.caffe.EmbedParameter();
                reader.start(block);
                while (!reader.end(block)) {
                    var tag = reader.tag();
                    switch (tag) {
                    case "num_output":
                        message.num_output = reader.uint32();
                        break;
                    case "input_dim":
                        message.input_dim = reader.uint32();
                        break;
                    case "bias_term":
                        message.bias_term = reader.bool();
                        break;
                    case "weight_filler":
                        message.weight_filler = $root.caffe.FillerParameter.decodeText(reader, true);
                        break;
                    case "bias_filler":
                        message.bias_filler = $root.caffe.FillerParameter.decodeText(reader, true);
                        break;
                    default:
                        reader.handle(tag, message);
                        break;
                    }
                }
                return message;
            };
    
            EmbedParameter.verify = function verify(message) {
                if (typeof message !== "object" || message === null)
                    return "object expected";
                if (message.num_output != null && message.hasOwnProperty("num_output"))
                    if (!$util.isInteger(message.num_output))
                        return "num_output: integer expected";
                if (message.input_dim != null && message.hasOwnProperty("input_dim"))
                    if (!$util.isInteger(message.input_dim))
                        return "input_dim: integer expected";
                if (message.bias_term != null && message.hasOwnProperty("bias_term"))
                    if (typeof message.bias_term !== "boolean")
                        return "bias_term: boolean expected";
                if (message.weight_filler != null && message.hasOwnProperty("weight_filler")) {
                    var error = $root.caffe.FillerParameter.verify(message.weight_filler);
                    if (error)
                        return "weight_filler." + error;
                }
                if (message.bias_filler != null && message.hasOwnProperty("bias_filler")) {
                    var error = $root.caffe.FillerParameter.verify(message.bias_filler);
                    if (error)
                        return "bias_filler." + error;
                }
                return null;
            };
    
            EmbedParameter.fromObject = function fromObject(object) {
                if (object instanceof $root.caffe.EmbedParameter)
                    return object;
                var message = new $root.caffe.EmbedParameter();
                if (object.num_output != null)
                    message.num_output = object.num_output >>> 0;
                if (object.input_dim != null)
                    message.input_dim = object.input_dim >>> 0;
                if (object.bias_term != null)
                    message.bias_term = Boolean(object.bias_term);
                if (object.weight_filler != null) {
                    if (typeof object.weight_filler !== "object")
                        throw TypeError(".caffe.EmbedParameter.weight_filler: object expected");
                    message.weight_filler = $root.caffe.FillerParameter.fromObject(object.weight_filler);
                }
                if (object.bias_filler != null) {
                    if (typeof object.bias_filler !== "object")
                        throw TypeError(".caffe.EmbedParameter.bias_filler: object expected");
                    message.bias_filler = $root.caffe.FillerParameter.fromObject(object.bias_filler);
                }
                return message;
            };
    
            EmbedParameter.toObject = function toObject(message, options) {
                if (!options)
                    options = {};
                var object = {};
                if (options.defaults) {
                    object.num_output = 0;
                    object.input_dim = 0;
                    object.bias_term = true;
                    object.weight_filler = null;
                    object.bias_filler = null;
                }
                if (message.num_output != null && message.hasOwnProperty("num_output"))
                    object.num_output = message.num_output;
                if (message.input_dim != null && message.hasOwnProperty("input_dim"))
                    object.input_dim = message.input_dim;
                if (message.bias_term != null && message.hasOwnProperty("bias_term"))
                    object.bias_term = message.bias_term;
                if (message.weight_filler != null && message.hasOwnProperty("weight_filler"))
                    object.weight_filler = $root.caffe.FillerParameter.toObject(message.weight_filler, options);
                if (message.bias_filler != null && message.hasOwnProperty("bias_filler"))
                    object.bias_filler = $root.caffe.FillerParameter.toObject(message.bias_filler, options);
                return object;
            };
    
            EmbedParameter.prototype.toJSON = function toJSON() {
                return this.constructor.toObject(this, $protobuf.util.toJSONOptions);
            };
    
            return EmbedParameter;
        })();
    
        caffe.ExpParameter = (function() {
    
            function ExpParameter(properties) {
                if (properties)
                    for (var keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                        if (properties[keys[i]] != null)
                            this[keys[i]] = properties[keys[i]];
            }
    
            ExpParameter.prototype.base = -1;
            ExpParameter.prototype.scale = 1;
            ExpParameter.prototype.shift = 0;
    
            ExpParameter.create = function create(properties) {
                return new ExpParameter(properties);
            };
    
            ExpParameter.decode = function decode(reader, length) {
                if (!(reader instanceof $Reader))
                    reader = $Reader.create(reader);
                var end = length === undefined ? reader.len : reader.pos + length, message = new $root.caffe.ExpParameter();
                while (reader.pos < end) {
                    var tag = reader.uint32();
                    switch (tag >>> 3) {
                    case 1:
                        message.base = reader.float();
                        break;
                    case 2:
                        message.scale = reader.float();
                        break;
                    case 3:
                        message.shift = reader.float();
                        break;
                    default:
                        reader.skipType(tag & 7);
                        break;
                    }
                }
                return message;
            };
    
            ExpParameter.decodeText = function decodeText(reader, block) {
                if (!(reader instanceof $TextReader))
                    reader = $TextReader.create(reader);
                var message = new $root.caffe.ExpParameter();
                reader.start(block);
                while (!reader.end(block)) {
                    var tag = reader.tag();
                    switch (tag) {
                    case "base":
                        message.base = reader.float();
                        break;
                    case "scale":
                        message.scale = reader.float();
                        break;
                    case "shift":
                        message.shift = reader.float();
                        break;
                    default:
                        reader.handle(tag, message);
                        break;
                    }
                }
                return message;
            };
    
            ExpParameter.verify = function verify(message) {
                if (typeof message !== "object" || message === null)
                    return "object expected";
                if (message.base != null && message.hasOwnProperty("base"))
                    if (typeof message.base !== "number")
                        return "base: number expected";
                if (message.scale != null && message.hasOwnProperty("scale"))
                    if (typeof message.scale !== "number")
                        return "scale: number expected";
                if (message.shift != null && message.hasOwnProperty("shift"))
                    if (typeof message.shift !== "number")
                        return "shift: number expected";
                return null;
            };
    
            ExpParameter.fromObject = function fromObject(object) {
                if (object instanceof $root.caffe.ExpParameter)
                    return object;
                var message = new $root.caffe.ExpParameter();
                if (object.base != null)
                    message.base = Number(object.base);
                if (object.scale != null)
                    message.scale = Number(object.scale);
                if (object.shift != null)
                    message.shift = Number(object.shift);
                return message;
            };
    
            ExpParameter.toObject = function toObject(message, options) {
                if (!options)
                    options = {};
                var object = {};
                if (options.defaults) {
                    object.base = -1;
                    object.scale = 1;
                    object.shift = 0;
                }
                if (message.base != null && message.hasOwnProperty("base"))
                    object.base = options.json && !isFinite(message.base) ? String(message.base) : message.base;
                if (message.scale != null && message.hasOwnProperty("scale"))
                    object.scale = options.json && !isFinite(message.scale) ? String(message.scale) : message.scale;
                if (message.shift != null && message.hasOwnProperty("shift"))
                    object.shift = options.json && !isFinite(message.shift) ? String(message.shift) : message.shift;
                return object;
            };
    
            ExpParameter.prototype.toJSON = function toJSON() {
                return this.constructor.toObject(this, $protobuf.util.toJSONOptions);
            };
    
            return ExpParameter;
        })();
    
        caffe.FlattenParameter = (function() {
    
            function FlattenParameter(properties) {
                if (properties)
                    for (var keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                        if (properties[keys[i]] != null)
                            this[keys[i]] = properties[keys[i]];
            }
    
            FlattenParameter.prototype.axis = 1;
            FlattenParameter.prototype.end_axis = -1;
    
            FlattenParameter.create = function create(properties) {
                return new FlattenParameter(properties);
            };
    
            FlattenParameter.decode = function decode(reader, length) {
                if (!(reader instanceof $Reader))
                    reader = $Reader.create(reader);
                var end = length === undefined ? reader.len : reader.pos + length, message = new $root.caffe.FlattenParameter();
                while (reader.pos < end) {
                    var tag = reader.uint32();
                    switch (tag >>> 3) {
                    case 1:
                        message.axis = reader.int32();
                        break;
                    case 2:
                        message.end_axis = reader.int32();
                        break;
                    default:
                        reader.skipType(tag & 7);
                        break;
                    }
                }
                return message;
            };
    
            FlattenParameter.decodeText = function decodeText(reader, block) {
                if (!(reader instanceof $TextReader))
                    reader = $TextReader.create(reader);
                var message = new $root.caffe.FlattenParameter();
                reader.start(block);
                while (!reader.end(block)) {
                    var tag = reader.tag();
                    switch (tag) {
                    case "axis":
                        message.axis = reader.int32();
                        break;
                    case "end_axis":
                        message.end_axis = reader.int32();
                        break;
                    default:
                        reader.handle(tag, message);
                        break;
                    }
                }
                return message;
            };
    
            FlattenParameter.verify = function verify(message) {
                if (typeof message !== "object" || message === null)
                    return "object expected";
                if (message.axis != null && message.hasOwnProperty("axis"))
                    if (!$util.isInteger(message.axis))
                        return "axis: integer expected";
                if (message.end_axis != null && message.hasOwnProperty("end_axis"))
                    if (!$util.isInteger(message.end_axis))
                        return "end_axis: integer expected";
                return null;
            };
    
            FlattenParameter.fromObject = function fromObject(object) {
                if (object instanceof $root.caffe.FlattenParameter)
                    return object;
                var message = new $root.caffe.FlattenParameter();
                if (object.axis != null)
                    message.axis = object.axis | 0;
                if (object.end_axis != null)
                    message.end_axis = object.end_axis | 0;
                return message;
            };
    
            FlattenParameter.toObject = function toObject(message, options) {
                if (!options)
                    options = {};
                var object = {};
                if (options.defaults) {
                    object.axis = 1;
                    object.end_axis = -1;
                }
                if (message.axis != null && message.hasOwnProperty("axis"))
                    object.axis = message.axis;
                if (message.end_axis != null && message.hasOwnProperty("end_axis"))
                    object.end_axis = message.end_axis;
                return object;
            };
    
            FlattenParameter.prototype.toJSON = function toJSON() {
                return this.constructor.toObject(this, $protobuf.util.toJSONOptions);
            };
    
            return FlattenParameter;
        })();
    
        caffe.HDF5DataParameter = (function() {
    
            function HDF5DataParameter(properties) {
                if (properties)
                    for (var keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                        if (properties[keys[i]] != null)
                            this[keys[i]] = properties[keys[i]];
            }
    
            HDF5DataParameter.prototype.source = "";
            HDF5DataParameter.prototype.batch_size = 0;
            HDF5DataParameter.prototype.shuffle = false;
    
            HDF5DataParameter.create = function create(properties) {
                return new HDF5DataParameter(properties);
            };
    
            HDF5DataParameter.decode = function decode(reader, length) {
                if (!(reader instanceof $Reader))
                    reader = $Reader.create(reader);
                var end = length === undefined ? reader.len : reader.pos + length, message = new $root.caffe.HDF5DataParameter();
                while (reader.pos < end) {
                    var tag = reader.uint32();
                    switch (tag >>> 3) {
                    case 1:
                        message.source = reader.string();
                        break;
                    case 2:
                        message.batch_size = reader.uint32();
                        break;
                    case 3:
                        message.shuffle = reader.bool();
                        break;
                    default:
                        reader.skipType(tag & 7);
                        break;
                    }
                }
                return message;
            };
    
            HDF5DataParameter.decodeText = function decodeText(reader, block) {
                if (!(reader instanceof $TextReader))
                    reader = $TextReader.create(reader);
                var message = new $root.caffe.HDF5DataParameter();
                reader.start(block);
                while (!reader.end(block)) {
                    var tag = reader.tag();
                    switch (tag) {
                    case "source":
                        message.source = reader.string();
                        break;
                    case "batch_size":
                        message.batch_size = reader.uint32();
                        break;
                    case "shuffle":
                        message.shuffle = reader.bool();
                        break;
                    default:
                        reader.handle(tag, message);
                        break;
                    }
                }
                return message;
            };
    
            HDF5DataParameter.verify = function verify(message) {
                if (typeof message !== "object" || message === null)
                    return "object expected";
                if (message.source != null && message.hasOwnProperty("source"))
                    if (!$util.isString(message.source))
                        return "source: string expected";
                if (message.batch_size != null && message.hasOwnProperty("batch_size"))
                    if (!$util.isInteger(message.batch_size))
                        return "batch_size: integer expected";
                if (message.shuffle != null && message.hasOwnProperty("shuffle"))
                    if (typeof message.shuffle !== "boolean")
                        return "shuffle: boolean expected";
                return null;
            };
    
            HDF5DataParameter.fromObject = function fromObject(object) {
                if (object instanceof $root.caffe.HDF5DataParameter)
                    return object;
                var message = new $root.caffe.HDF5DataParameter();
                if (object.source != null)
                    message.source = String(object.source);
                if (object.batch_size != null)
                    message.batch_size = object.batch_size >>> 0;
                if (object.shuffle != null)
                    message.shuffle = Boolean(object.shuffle);
                return message;
            };
    
            HDF5DataParameter.toObject = function toObject(message, options) {
                if (!options)
                    options = {};
                var object = {};
                if (options.defaults) {
                    object.source = "";
                    object.batch_size = 0;
                    object.shuffle = false;
                }
                if (message.source != null && message.hasOwnProperty("source"))
                    object.source = message.source;
                if (message.batch_size != null && message.hasOwnProperty("batch_size"))
                    object.batch_size = message.batch_size;
                if (message.shuffle != null && message.hasOwnProperty("shuffle"))
                    object.shuffle = message.shuffle;
                return object;
            };
    
            HDF5DataParameter.prototype.toJSON = function toJSON() {
                return this.constructor.toObject(this, $protobuf.util.toJSONOptions);
            };
    
            return HDF5DataParameter;
        })();
    
        caffe.HDF5OutputParameter = (function() {
    
            function HDF5OutputParameter(properties) {
                if (properties)
                    for (var keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                        if (properties[keys[i]] != null)
                            this[keys[i]] = properties[keys[i]];
            }
    
            HDF5OutputParameter.prototype.file_name = "";
    
            HDF5OutputParameter.create = function create(properties) {
                return new HDF5OutputParameter(properties);
            };
    
            HDF5OutputParameter.decode = function decode(reader, length) {
                if (!(reader instanceof $Reader))
                    reader = $Reader.create(reader);
                var end = length === undefined ? reader.len : reader.pos + length, message = new $root.caffe.HDF5OutputParameter();
                while (reader.pos < end) {
                    var tag = reader.uint32();
                    switch (tag >>> 3) {
                    case 1:
                        message.file_name = reader.string();
                        break;
                    default:
                        reader.skipType(tag & 7);
                        break;
                    }
                }
                return message;
            };
    
            HDF5OutputParameter.decodeText = function decodeText(reader, block) {
                if (!(reader instanceof $TextReader))
                    reader = $TextReader.create(reader);
                var message = new $root.caffe.HDF5OutputParameter();
                reader.start(block);
                while (!reader.end(block)) {
                    var tag = reader.tag();
                    switch (tag) {
                    case "file_name":
                        message.file_name = reader.string();
                        break;
                    default:
                        reader.handle(tag, message);
                        break;
                    }
                }
                return message;
            };
    
            HDF5OutputParameter.verify = function verify(message) {
                if (typeof message !== "object" || message === null)
                    return "object expected";
                if (message.file_name != null && message.hasOwnProperty("file_name"))
                    if (!$util.isString(message.file_name))
                        return "file_name: string expected";
                return null;
            };
    
            HDF5OutputParameter.fromObject = function fromObject(object) {
                if (object instanceof $root.caffe.HDF5OutputParameter)
                    return object;
                var message = new $root.caffe.HDF5OutputParameter();
                if (object.file_name != null)
                    message.file_name = String(object.file_name);
                return message;
            };
    
            HDF5OutputParameter.toObject = function toObject(message, options) {
                if (!options)
                    options = {};
                var object = {};
                if (options.defaults)
                    object.file_name = "";
                if (message.file_name != null && message.hasOwnProperty("file_name"))
                    object.file_name = message.file_name;
                return object;
            };
    
            HDF5OutputParameter.prototype.toJSON = function toJSON() {
                return this.constructor.toObject(this, $protobuf.util.toJSONOptions);
            };
    
            return HDF5OutputParameter;
        })();
    
        caffe.HingeLossParameter = (function() {
    
            function HingeLossParameter(properties) {
                if (properties)
                    for (var keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                        if (properties[keys[i]] != null)
                            this[keys[i]] = properties[keys[i]];
            }
    
            HingeLossParameter.prototype.norm = 1;
    
            HingeLossParameter.create = function create(properties) {
                return new HingeLossParameter(properties);
            };
    
            HingeLossParameter.decode = function decode(reader, length) {
                if (!(reader instanceof $Reader))
                    reader = $Reader.create(reader);
                var end = length === undefined ? reader.len : reader.pos + length, message = new $root.caffe.HingeLossParameter();
                while (reader.pos < end) {
                    var tag = reader.uint32();
                    switch (tag >>> 3) {
                    case 1:
                        message.norm = reader.int32();
                        break;
                    default:
                        reader.skipType(tag & 7);
                        break;
                    }
                }
                return message;
            };
    
            HingeLossParameter.decodeText = function decodeText(reader, block) {
                if (!(reader instanceof $TextReader))
                    reader = $TextReader.create(reader);
                var message = new $root.caffe.HingeLossParameter();
                reader.start(block);
                while (!reader.end(block)) {
                    var tag = reader.tag();
                    switch (tag) {
                    case "norm":
                        message.norm = reader.enum($root.caffe.HingeLossParameter.Norm);
                        break;
                    default:
                        reader.handle(tag, message);
                        break;
                    }
                }
                return message;
            };
    
            HingeLossParameter.verify = function verify(message) {
                if (typeof message !== "object" || message === null)
                    return "object expected";
                if (message.norm != null && message.hasOwnProperty("norm"))
                    switch (message.norm) {
                    default:
                        return "norm: enum value expected";
                    case 1:
                    case 2:
                        break;
                    }
                return null;
            };
    
            HingeLossParameter.fromObject = function fromObject(object) {
                if (object instanceof $root.caffe.HingeLossParameter)
                    return object;
                var message = new $root.caffe.HingeLossParameter();
                switch (object.norm) {
                case "L1":
                case 1:
                    message.norm = 1;
                    break;
                case "L2":
                case 2:
                    message.norm = 2;
                    break;
                }
                return message;
            };
    
            HingeLossParameter.toObject = function toObject(message, options) {
                if (!options)
                    options = {};
                var object = {};
                if (options.defaults)
                    object.norm = options.enums === String ? "L1" : 1;
                if (message.norm != null && message.hasOwnProperty("norm"))
                    object.norm = options.enums === String ? $root.caffe.HingeLossParameter.Norm[message.norm] : message.norm;
                return object;
            };
    
            HingeLossParameter.prototype.toJSON = function toJSON() {
                return this.constructor.toObject(this, $protobuf.util.toJSONOptions);
            };
    
            HingeLossParameter.Norm = (function() {
                var valuesById = {}, values = Object.create(valuesById);
                values[valuesById[1] = "L1"] = 1;
                values[valuesById[2] = "L2"] = 2;
                return values;
            })();
    
            return HingeLossParameter;
        })();
    
        caffe.ImageDataParameter = (function() {
    
            function ImageDataParameter(properties) {
                if (properties)
                    for (var keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                        if (properties[keys[i]] != null)
                            this[keys[i]] = properties[keys[i]];
            }
    
            ImageDataParameter.prototype.source = "";
            ImageDataParameter.prototype.batch_size = 1;
            ImageDataParameter.prototype.rand_skip = 0;
            ImageDataParameter.prototype.shuffle = false;
            ImageDataParameter.prototype.new_height = 0;
            ImageDataParameter.prototype.new_width = 0;
            ImageDataParameter.prototype.is_color = true;
            ImageDataParameter.prototype.scale = 1;
            ImageDataParameter.prototype.mean_file = "";
            ImageDataParameter.prototype.crop_size = 0;
            ImageDataParameter.prototype.mirror = false;
            ImageDataParameter.prototype.root_folder = "";
    
            ImageDataParameter.create = function create(properties) {
                return new ImageDataParameter(properties);
            };
    
            ImageDataParameter.decode = function decode(reader, length) {
                if (!(reader instanceof $Reader))
                    reader = $Reader.create(reader);
                var end = length === undefined ? reader.len : reader.pos + length, message = new $root.caffe.ImageDataParameter();
                while (reader.pos < end) {
                    var tag = reader.uint32();
                    switch (tag >>> 3) {
                    case 1:
                        message.source = reader.string();
                        break;
                    case 4:
                        message.batch_size = reader.uint32();
                        break;
                    case 7:
                        message.rand_skip = reader.uint32();
                        break;
                    case 8:
                        message.shuffle = reader.bool();
                        break;
                    case 9:
                        message.new_height = reader.uint32();
                        break;
                    case 10:
                        message.new_width = reader.uint32();
                        break;
                    case 11:
                        message.is_color = reader.bool();
                        break;
                    case 2:
                        message.scale = reader.float();
                        break;
                    case 3:
                        message.mean_file = reader.string();
                        break;
                    case 5:
                        message.crop_size = reader.uint32();
                        break;
                    case 6:
                        message.mirror = reader.bool();
                        break;
                    case 12:
                        message.root_folder = reader.string();
                        break;
                    default:
                        reader.skipType(tag & 7);
                        break;
                    }
                }
                return message;
            };
    
            ImageDataParameter.decodeText = function decodeText(reader, block) {
                if (!(reader instanceof $TextReader))
                    reader = $TextReader.create(reader);
                var message = new $root.caffe.ImageDataParameter();
                reader.start(block);
                while (!reader.end(block)) {
                    var tag = reader.tag();
                    switch (tag) {
                    case "source":
                        message.source = reader.string();
                        break;
                    case "batch_size":
                        message.batch_size = reader.uint32();
                        break;
                    case "rand_skip":
                        message.rand_skip = reader.uint32();
                        break;
                    case "shuffle":
                        message.shuffle = reader.bool();
                        break;
                    case "new_height":
                        message.new_height = reader.uint32();
                        break;
                    case "new_width":
                        message.new_width = reader.uint32();
                        break;
                    case "is_color":
                        message.is_color = reader.bool();
                        break;
                    case "scale":
                        message.scale = reader.float();
                        break;
                    case "mean_file":
                        message.mean_file = reader.string();
                        break;
                    case "crop_size":
                        message.crop_size = reader.uint32();
                        break;
                    case "mirror":
                        message.mirror = reader.bool();
                        break;
                    case "root_folder":
                        message.root_folder = reader.string();
                        break;
                    default:
                        reader.handle(tag, message);
                        break;
                    }
                }
                return message;
            };
    
            ImageDataParameter.verify = function verify(message) {
                if (typeof message !== "object" || message === null)
                    return "object expected";
                if (message.source != null && message.hasOwnProperty("source"))
                    if (!$util.isString(message.source))
                        return "source: string expected";
                if (message.batch_size != null && message.hasOwnProperty("batch_size"))
                    if (!$util.isInteger(message.batch_size))
                        return "batch_size: integer expected";
                if (message.rand_skip != null && message.hasOwnProperty("rand_skip"))
                    if (!$util.isInteger(message.rand_skip))
                        return "rand_skip: integer expected";
                if (message.shuffle != null && message.hasOwnProperty("shuffle"))
                    if (typeof message.shuffle !== "boolean")
                        return "shuffle: boolean expected";
                if (message.new_height != null && message.hasOwnProperty("new_height"))
                    if (!$util.isInteger(message.new_height))
                        return "new_height: integer expected";
                if (message.new_width != null && message.hasOwnProperty("new_width"))
                    if (!$util.isInteger(message.new_width))
                        return "new_width: integer expected";
                if (message.is_color != null && message.hasOwnProperty("is_color"))
                    if (typeof message.is_color !== "boolean")
                        return "is_color: boolean expected";
                if (message.scale != null && message.hasOwnProperty("scale"))
                    if (typeof message.scale !== "number")
                        return "scale: number expected";
                if (message.mean_file != null && message.hasOwnProperty("mean_file"))
                    if (!$util.isString(message.mean_file))
                        return "mean_file: string expected";
                if (message.crop_size != null && message.hasOwnProperty("crop_size"))
                    if (!$util.isInteger(message.crop_size))
                        return "crop_size: integer expected";
                if (message.mirror != null && message.hasOwnProperty("mirror"))
                    if (typeof message.mirror !== "boolean")
                        return "mirror: boolean expected";
                if (message.root_folder != null && message.hasOwnProperty("root_folder"))
                    if (!$util.isString(message.root_folder))
                        return "root_folder: string expected";
                return null;
            };
    
            ImageDataParameter.fromObject = function fromObject(object) {
                if (object instanceof $root.caffe.ImageDataParameter)
                    return object;
                var message = new $root.caffe.ImageDataParameter();
                if (object.source != null)
                    message.source = String(object.source);
                if (object.batch_size != null)
                    message.batch_size = object.batch_size >>> 0;
                if (object.rand_skip != null)
                    message.rand_skip = object.rand_skip >>> 0;
                if (object.shuffle != null)
                    message.shuffle = Boolean(object.shuffle);
                if (object.new_height != null)
                    message.new_height = object.new_height >>> 0;
                if (object.new_width != null)
                    message.new_width = object.new_width >>> 0;
                if (object.is_color != null)
                    message.is_color = Boolean(object.is_color);
                if (object.scale != null)
                    message.scale = Number(object.scale);
                if (object.mean_file != null)
                    message.mean_file = String(object.mean_file);
                if (object.crop_size != null)
                    message.crop_size = object.crop_size >>> 0;
                if (object.mirror != null)
                    message.mirror = Boolean(object.mirror);
                if (object.root_folder != null)
                    message.root_folder = String(object.root_folder);
                return message;
            };
    
            ImageDataParameter.toObject = function toObject(message, options) {
                if (!options)
                    options = {};
                var object = {};
                if (options.defaults) {
                    object.source = "";
                    object.scale = 1;
                    object.mean_file = "";
                    object.batch_size = 1;
                    object.crop_size = 0;
                    object.mirror = false;
                    object.rand_skip = 0;
                    object.shuffle = false;
                    object.new_height = 0;
                    object.new_width = 0;
                    object.is_color = true;
                    object.root_folder = "";
                }
                if (message.source != null && message.hasOwnProperty("source"))
                    object.source = message.source;
                if (message.scale != null && message.hasOwnProperty("scale"))
                    object.scale = options.json && !isFinite(message.scale) ? String(message.scale) : message.scale;
                if (message.mean_file != null && message.hasOwnProperty("mean_file"))
                    object.mean_file = message.mean_file;
                if (message.batch_size != null && message.hasOwnProperty("batch_size"))
                    object.batch_size = message.batch_size;
                if (message.crop_size != null && message.hasOwnProperty("crop_size"))
                    object.crop_size = message.crop_size;
                if (message.mirror != null && message.hasOwnProperty("mirror"))
                    object.mirror = message.mirror;
                if (message.rand_skip != null && message.hasOwnProperty("rand_skip"))
                    object.rand_skip = message.rand_skip;
                if (message.shuffle != null && message.hasOwnProperty("shuffle"))
                    object.shuffle = message.shuffle;
                if (message.new_height != null && message.hasOwnProperty("new_height"))
                    object.new_height = message.new_height;
                if (message.new_width != null && message.hasOwnProperty("new_width"))
                    object.new_width = message.new_width;
                if (message.is_color != null && message.hasOwnProperty("is_color"))
                    object.is_color = message.is_color;
                if (message.root_folder != null && message.hasOwnProperty("root_folder"))
                    object.root_folder = message.root_folder;
                return object;
            };
    
            ImageDataParameter.prototype.toJSON = function toJSON() {
                return this.constructor.toObject(this, $protobuf.util.toJSONOptions);
            };
    
            return ImageDataParameter;
        })();
    
        caffe.InfogainLossParameter = (function() {
    
            function InfogainLossParameter(properties) {
                if (properties)
                    for (var keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                        if (properties[keys[i]] != null)
                            this[keys[i]] = properties[keys[i]];
            }
    
            InfogainLossParameter.prototype.source = "";
            InfogainLossParameter.prototype.axis = 1;
    
            InfogainLossParameter.create = function create(properties) {
                return new InfogainLossParameter(properties);
            };
    
            InfogainLossParameter.decode = function decode(reader, length) {
                if (!(reader instanceof $Reader))
                    reader = $Reader.create(reader);
                var end = length === undefined ? reader.len : reader.pos + length, message = new $root.caffe.InfogainLossParameter();
                while (reader.pos < end) {
                    var tag = reader.uint32();
                    switch (tag >>> 3) {
                    case 1:
                        message.source = reader.string();
                        break;
                    case 2:
                        message.axis = reader.int32();
                        break;
                    default:
                        reader.skipType(tag & 7);
                        break;
                    }
                }
                return message;
            };
    
            InfogainLossParameter.decodeText = function decodeText(reader, block) {
                if (!(reader instanceof $TextReader))
                    reader = $TextReader.create(reader);
                var message = new $root.caffe.InfogainLossParameter();
                reader.start(block);
                while (!reader.end(block)) {
                    var tag = reader.tag();
                    switch (tag) {
                    case "source":
                        message.source = reader.string();
                        break;
                    case "axis":
                        message.axis = reader.int32();
                        break;
                    default:
                        reader.handle(tag, message);
                        break;
                    }
                }
                return message;
            };
    
            InfogainLossParameter.verify = function verify(message) {
                if (typeof message !== "object" || message === null)
                    return "object expected";
                if (message.source != null && message.hasOwnProperty("source"))
                    if (!$util.isString(message.source))
                        return "source: string expected";
                if (message.axis != null && message.hasOwnProperty("axis"))
                    if (!$util.isInteger(message.axis))
                        return "axis: integer expected";
                return null;
            };
    
            InfogainLossParameter.fromObject = function fromObject(object) {
                if (object instanceof $root.caffe.InfogainLossParameter)
                    return object;
                var message = new $root.caffe.InfogainLossParameter();
                if (object.source != null)
                    message.source = String(object.source);
                if (object.axis != null)
                    message.axis = object.axis | 0;
                return message;
            };
    
            InfogainLossParameter.toObject = function toObject(message, options) {
                if (!options)
                    options = {};
                var object = {};
                if (options.defaults) {
                    object.source = "";
                    object.axis = 1;
                }
                if (message.source != null && message.hasOwnProperty("source"))
                    object.source = message.source;
                if (message.axis != null && message.hasOwnProperty("axis"))
                    object.axis = message.axis;
                return object;
            };
    
            InfogainLossParameter.prototype.toJSON = function toJSON() {
                return this.constructor.toObject(this, $protobuf.util.toJSONOptions);
            };
    
            return InfogainLossParameter;
        })();
    
        caffe.InnerProductParameter = (function() {
    
            function InnerProductParameter(properties) {
                if (properties)
                    for (var keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                        if (properties[keys[i]] != null)
                            this[keys[i]] = properties[keys[i]];
            }
    
            InnerProductParameter.prototype.num_output = 0;
            InnerProductParameter.prototype.bias_term = true;
            InnerProductParameter.prototype.weight_filler = null;
            InnerProductParameter.prototype.bias_filler = null;
            InnerProductParameter.prototype.axis = 1;
            InnerProductParameter.prototype.transpose = false;
    
            InnerProductParameter.create = function create(properties) {
                return new InnerProductParameter(properties);
            };
    
            InnerProductParameter.decode = function decode(reader, length) {
                if (!(reader instanceof $Reader))
                    reader = $Reader.create(reader);
                var end = length === undefined ? reader.len : reader.pos + length, message = new $root.caffe.InnerProductParameter();
                while (reader.pos < end) {
                    var tag = reader.uint32();
                    switch (tag >>> 3) {
                    case 1:
                        message.num_output = reader.uint32();
                        break;
                    case 2:
                        message.bias_term = reader.bool();
                        break;
                    case 3:
                        message.weight_filler = $root.caffe.FillerParameter.decode(reader, reader.uint32());
                        break;
                    case 4:
                        message.bias_filler = $root.caffe.FillerParameter.decode(reader, reader.uint32());
                        break;
                    case 5:
                        message.axis = reader.int32();
                        break;
                    case 6:
                        message.transpose = reader.bool();
                        break;
                    default:
                        reader.skipType(tag & 7);
                        break;
                    }
                }
                return message;
            };
    
            InnerProductParameter.decodeText = function decodeText(reader, block) {
                if (!(reader instanceof $TextReader))
                    reader = $TextReader.create(reader);
                var message = new $root.caffe.InnerProductParameter();
                reader.start(block);
                while (!reader.end(block)) {
                    var tag = reader.tag();
                    switch (tag) {
                    case "num_output":
                        message.num_output = reader.uint32();
                        break;
                    case "bias_term":
                        message.bias_term = reader.bool();
                        break;
                    case "weight_filler":
                        message.weight_filler = $root.caffe.FillerParameter.decodeText(reader, true);
                        break;
                    case "bias_filler":
                        message.bias_filler = $root.caffe.FillerParameter.decodeText(reader, true);
                        break;
                    case "axis":
                        message.axis = reader.int32();
                        break;
                    case "transpose":
                        message.transpose = reader.bool();
                        break;
                    default:
                        reader.handle(tag, message);
                        break;
                    }
                }
                return message;
            };
    
            InnerProductParameter.verify = function verify(message) {
                if (typeof message !== "object" || message === null)
                    return "object expected";
                if (message.num_output != null && message.hasOwnProperty("num_output"))
                    if (!$util.isInteger(message.num_output))
                        return "num_output: integer expected";
                if (message.bias_term != null && message.hasOwnProperty("bias_term"))
                    if (typeof message.bias_term !== "boolean")
                        return "bias_term: boolean expected";
                if (message.weight_filler != null && message.hasOwnProperty("weight_filler")) {
                    var error = $root.caffe.FillerParameter.verify(message.weight_filler);
                    if (error)
                        return "weight_filler." + error;
                }
                if (message.bias_filler != null && message.hasOwnProperty("bias_filler")) {
                    var error = $root.caffe.FillerParameter.verify(message.bias_filler);
                    if (error)
                        return "bias_filler." + error;
                }
                if (message.axis != null && message.hasOwnProperty("axis"))
                    if (!$util.isInteger(message.axis))
                        return "axis: integer expected";
                if (message.transpose != null && message.hasOwnProperty("transpose"))
                    if (typeof message.transpose !== "boolean")
                        return "transpose: boolean expected";
                return null;
            };
    
            InnerProductParameter.fromObject = function fromObject(object) {
                if (object instanceof $root.caffe.InnerProductParameter)
                    return object;
                var message = new $root.caffe.InnerProductParameter();
                if (object.num_output != null)
                    message.num_output = object.num_output >>> 0;
                if (object.bias_term != null)
                    message.bias_term = Boolean(object.bias_term);
                if (object.weight_filler != null) {
                    if (typeof object.weight_filler !== "object")
                        throw TypeError(".caffe.InnerProductParameter.weight_filler: object expected");
                    message.weight_filler = $root.caffe.FillerParameter.fromObject(object.weight_filler);
                }
                if (object.bias_filler != null) {
                    if (typeof object.bias_filler !== "object")
                        throw TypeError(".caffe.InnerProductParameter.bias_filler: object expected");
                    message.bias_filler = $root.caffe.FillerParameter.fromObject(object.bias_filler);
                }
                if (object.axis != null)
                    message.axis = object.axis | 0;
                if (object.transpose != null)
                    message.transpose = Boolean(object.transpose);
                return message;
            };
    
            InnerProductParameter.toObject = function toObject(message, options) {
                if (!options)
                    options = {};
                var object = {};
                if (options.defaults) {
                    object.num_output = 0;
                    object.bias_term = true;
                    object.weight_filler = null;
                    object.bias_filler = null;
                    object.axis = 1;
                    object.transpose = false;
                }
                if (message.num_output != null && message.hasOwnProperty("num_output"))
                    object.num_output = message.num_output;
                if (message.bias_term != null && message.hasOwnProperty("bias_term"))
                    object.bias_term = message.bias_term;
                if (message.weight_filler != null && message.hasOwnProperty("weight_filler"))
                    object.weight_filler = $root.caffe.FillerParameter.toObject(message.weight_filler, options);
                if (message.bias_filler != null && message.hasOwnProperty("bias_filler"))
                    object.bias_filler = $root.caffe.FillerParameter.toObject(message.bias_filler, options);
                if (message.axis != null && message.hasOwnProperty("axis"))
                    object.axis = message.axis;
                if (message.transpose != null && message.hasOwnProperty("transpose"))
                    object.transpose = message.transpose;
                return object;
            };
    
            InnerProductParameter.prototype.toJSON = function toJSON() {
                return this.constructor.toObject(this, $protobuf.util.toJSONOptions);
            };
    
            return InnerProductParameter;
        })();
    
        caffe.InputParameter = (function() {
    
            function InputParameter(properties) {
                this.shape = [];
                if (properties)
                    for (var keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                        if (properties[keys[i]] != null)
                            this[keys[i]] = properties[keys[i]];
            }
    
            InputParameter.prototype.shape = $util.emptyArray;
    
            InputParameter.create = function create(properties) {
                return new InputParameter(properties);
            };
    
            InputParameter.decode = function decode(reader, length) {
                if (!(reader instanceof $Reader))
                    reader = $Reader.create(reader);
                var end = length === undefined ? reader.len : reader.pos + length, message = new $root.caffe.InputParameter();
                while (reader.pos < end) {
                    var tag = reader.uint32();
                    switch (tag >>> 3) {
                    case 1:
                        if (!(message.shape && message.shape.length))
                            message.shape = [];
                        message.shape.push($root.caffe.BlobShape.decode(reader, reader.uint32()));
                        break;
                    default:
                        reader.skipType(tag & 7);
                        break;
                    }
                }
                return message;
            };
    
            InputParameter.decodeText = function decodeText(reader, block) {
                if (!(reader instanceof $TextReader))
                    reader = $TextReader.create(reader);
                var message = new $root.caffe.InputParameter();
                reader.start(block);
                while (!reader.end(block)) {
                    var tag = reader.tag();
                    switch (tag) {
                    case "shape":
                        if (!(message.shape && message.shape.length))
                            message.shape = [];
                        message.shape.push($root.caffe.BlobShape.decodeText(reader, true));
                        break;
                    default:
                        reader.handle(tag, message);
                        break;
                    }
                }
                return message;
            };
    
            InputParameter.verify = function verify(message) {
                if (typeof message !== "object" || message === null)
                    return "object expected";
                if (message.shape != null && message.hasOwnProperty("shape")) {
                    if (!Array.isArray(message.shape))
                        return "shape: array expected";
                    for (var i = 0; i < message.shape.length; ++i) {
                        var error = $root.caffe.BlobShape.verify(message.shape[i]);
                        if (error)
                            return "shape." + error;
                    }
                }
                return null;
            };
    
            InputParameter.fromObject = function fromObject(object) {
                if (object instanceof $root.caffe.InputParameter)
                    return object;
                var message = new $root.caffe.InputParameter();
                if (object.shape) {
                    if (!Array.isArray(object.shape))
                        throw TypeError(".caffe.InputParameter.shape: array expected");
                    message.shape = [];
                    for (var i = 0; i < object.shape.length; ++i) {
                        if (typeof object.shape[i] !== "object")
                            throw TypeError(".caffe.InputParameter.shape: object expected");
                        message.shape[i] = $root.caffe.BlobShape.fromObject(object.shape[i]);
                    }
                }
                return message;
            };
    
            InputParameter.toObject = function toObject(message, options) {
                if (!options)
                    options = {};
                var object = {};
                if (options.arrays || options.defaults)
                    object.shape = [];
                if (message.shape && message.shape.length) {
                    object.shape = [];
                    for (var j = 0; j < message.shape.length; ++j)
                        object.shape[j] = $root.caffe.BlobShape.toObject(message.shape[j], options);
                }
                return object;
            };
    
            InputParameter.prototype.toJSON = function toJSON() {
                return this.constructor.toObject(this, $protobuf.util.toJSONOptions);
            };
    
            return InputParameter;
        })();
    
        caffe.LogParameter = (function() {
    
            function LogParameter(properties) {
                if (properties)
                    for (var keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                        if (properties[keys[i]] != null)
                            this[keys[i]] = properties[keys[i]];
            }
    
            LogParameter.prototype.base = -1;
            LogParameter.prototype.scale = 1;
            LogParameter.prototype.shift = 0;
    
            LogParameter.create = function create(properties) {
                return new LogParameter(properties);
            };
    
            LogParameter.decode = function decode(reader, length) {
                if (!(reader instanceof $Reader))
                    reader = $Reader.create(reader);
                var end = length === undefined ? reader.len : reader.pos + length, message = new $root.caffe.LogParameter();
                while (reader.pos < end) {
                    var tag = reader.uint32();
                    switch (tag >>> 3) {
                    case 1:
                        message.base = reader.float();
                        break;
                    case 2:
                        message.scale = reader.float();
                        break;
                    case 3:
                        message.shift = reader.float();
                        break;
                    default:
                        reader.skipType(tag & 7);
                        break;
                    }
                }
                return message;
            };
    
            LogParameter.decodeText = function decodeText(reader, block) {
                if (!(reader instanceof $TextReader))
                    reader = $TextReader.create(reader);
                var message = new $root.caffe.LogParameter();
                reader.start(block);
                while (!reader.end(block)) {
                    var tag = reader.tag();
                    switch (tag) {
                    case "base":
                        message.base = reader.float();
                        break;
                    case "scale":
                        message.scale = reader.float();
                        break;
                    case "shift":
                        message.shift = reader.float();
                        break;
                    default:
                        reader.handle(tag, message);
                        break;
                    }
                }
                return message;
            };
    
            LogParameter.verify = function verify(message) {
                if (typeof message !== "object" || message === null)
                    return "object expected";
                if (message.base != null && message.hasOwnProperty("base"))
                    if (typeof message.base !== "number")
                        return "base: number expected";
                if (message.scale != null && message.hasOwnProperty("scale"))
                    if (typeof message.scale !== "number")
                        return "scale: number expected";
                if (message.shift != null && message.hasOwnProperty("shift"))
                    if (typeof message.shift !== "number")
                        return "shift: number expected";
                return null;
            };
    
            LogParameter.fromObject = function fromObject(object) {
                if (object instanceof $root.caffe.LogParameter)
                    return object;
                var message = new $root.caffe.LogParameter();
                if (object.base != null)
                    message.base = Number(object.base);
                if (object.scale != null)
                    message.scale = Number(object.scale);
                if (object.shift != null)
                    message.shift = Number(object.shift);
                return message;
            };
    
            LogParameter.toObject = function toObject(message, options) {
                if (!options)
                    options = {};
                var object = {};
                if (options.defaults) {
                    object.base = -1;
                    object.scale = 1;
                    object.shift = 0;
                }
                if (message.base != null && message.hasOwnProperty("base"))
                    object.base = options.json && !isFinite(message.base) ? String(message.base) : message.base;
                if (message.scale != null && message.hasOwnProperty("scale"))
                    object.scale = options.json && !isFinite(message.scale) ? String(message.scale) : message.scale;
                if (message.shift != null && message.hasOwnProperty("shift"))
                    object.shift = options.json && !isFinite(message.shift) ? String(message.shift) : message.shift;
                return object;
            };
    
            LogParameter.prototype.toJSON = function toJSON() {
                return this.constructor.toObject(this, $protobuf.util.toJSONOptions);
            };
    
            return LogParameter;
        })();
    
        caffe.LRNParameter = (function() {
    
            function LRNParameter(properties) {
                if (properties)
                    for (var keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                        if (properties[keys[i]] != null)
                            this[keys[i]] = properties[keys[i]];
            }
    
            LRNParameter.prototype.local_size = 5;
            LRNParameter.prototype.alpha = 1;
            LRNParameter.prototype.beta = 0.75;
            LRNParameter.prototype.norm_region = 0;
            LRNParameter.prototype.k = 1;
            LRNParameter.prototype.engine = 0;
    
            LRNParameter.create = function create(properties) {
                return new LRNParameter(properties);
            };
    
            LRNParameter.decode = function decode(reader, length) {
                if (!(reader instanceof $Reader))
                    reader = $Reader.create(reader);
                var end = length === undefined ? reader.len : reader.pos + length, message = new $root.caffe.LRNParameter();
                while (reader.pos < end) {
                    var tag = reader.uint32();
                    switch (tag >>> 3) {
                    case 1:
                        message.local_size = reader.uint32();
                        break;
                    case 2:
                        message.alpha = reader.float();
                        break;
                    case 3:
                        message.beta = reader.float();
                        break;
                    case 4:
                        message.norm_region = reader.int32();
                        break;
                    case 5:
                        message.k = reader.float();
                        break;
                    case 6:
                        message.engine = reader.int32();
                        break;
                    default:
                        reader.skipType(tag & 7);
                        break;
                    }
                }
                return message;
            };
    
            LRNParameter.decodeText = function decodeText(reader, block) {
                if (!(reader instanceof $TextReader))
                    reader = $TextReader.create(reader);
                var message = new $root.caffe.LRNParameter();
                reader.start(block);
                while (!reader.end(block)) {
                    var tag = reader.tag();
                    switch (tag) {
                    case "local_size":
                        message.local_size = reader.uint32();
                        break;
                    case "alpha":
                        message.alpha = reader.float();
                        break;
                    case "beta":
                        message.beta = reader.float();
                        break;
                    case "norm_region":
                        message.norm_region = reader.enum($root.caffe.LRNParameter.NormRegion);
                        break;
                    case "k":
                        message.k = reader.float();
                        break;
                    case "engine":
                        message.engine = reader.enum($root.caffe.LRNParameter.Engine);
                        break;
                    default:
                        reader.handle(tag, message);
                        break;
                    }
                }
                return message;
            };
    
            LRNParameter.verify = function verify(message) {
                if (typeof message !== "object" || message === null)
                    return "object expected";
                if (message.local_size != null && message.hasOwnProperty("local_size"))
                    if (!$util.isInteger(message.local_size))
                        return "local_size: integer expected";
                if (message.alpha != null && message.hasOwnProperty("alpha"))
                    if (typeof message.alpha !== "number")
                        return "alpha: number expected";
                if (message.beta != null && message.hasOwnProperty("beta"))
                    if (typeof message.beta !== "number")
                        return "beta: number expected";
                if (message.norm_region != null && message.hasOwnProperty("norm_region"))
                    switch (message.norm_region) {
                    default:
                        return "norm_region: enum value expected";
                    case 0:
                    case 1:
                        break;
                    }
                if (message.k != null && message.hasOwnProperty("k"))
                    if (typeof message.k !== "number")
                        return "k: number expected";
                if (message.engine != null && message.hasOwnProperty("engine"))
                    switch (message.engine) {
                    default:
                        return "engine: enum value expected";
                    case 0:
                    case 1:
                    case 2:
                        break;
                    }
                return null;
            };
    
            LRNParameter.fromObject = function fromObject(object) {
                if (object instanceof $root.caffe.LRNParameter)
                    return object;
                var message = new $root.caffe.LRNParameter();
                if (object.local_size != null)
                    message.local_size = object.local_size >>> 0;
                if (object.alpha != null)
                    message.alpha = Number(object.alpha);
                if (object.beta != null)
                    message.beta = Number(object.beta);
                switch (object.norm_region) {
                case "ACROSS_CHANNELS":
                case 0:
                    message.norm_region = 0;
                    break;
                case "WITHIN_CHANNEL":
                case 1:
                    message.norm_region = 1;
                    break;
                }
                if (object.k != null)
                    message.k = Number(object.k);
                switch (object.engine) {
                case "DEFAULT":
                case 0:
                    message.engine = 0;
                    break;
                case "CAFFE":
                case 1:
                    message.engine = 1;
                    break;
                case "CUDNN":
                case 2:
                    message.engine = 2;
                    break;
                }
                return message;
            };
    
            LRNParameter.toObject = function toObject(message, options) {
                if (!options)
                    options = {};
                var object = {};
                if (options.defaults) {
                    object.local_size = 5;
                    object.alpha = 1;
                    object.beta = 0.75;
                    object.norm_region = options.enums === String ? "ACROSS_CHANNELS" : 0;
                    object.k = 1;
                    object.engine = options.enums === String ? "DEFAULT" : 0;
                }
                if (message.local_size != null && message.hasOwnProperty("local_size"))
                    object.local_size = message.local_size;
                if (message.alpha != null && message.hasOwnProperty("alpha"))
                    object.alpha = options.json && !isFinite(message.alpha) ? String(message.alpha) : message.alpha;
                if (message.beta != null && message.hasOwnProperty("beta"))
                    object.beta = options.json && !isFinite(message.beta) ? String(message.beta) : message.beta;
                if (message.norm_region != null && message.hasOwnProperty("norm_region"))
                    object.norm_region = options.enums === String ? $root.caffe.LRNParameter.NormRegion[message.norm_region] : message.norm_region;
                if (message.k != null && message.hasOwnProperty("k"))
                    object.k = options.json && !isFinite(message.k) ? String(message.k) : message.k;
                if (message.engine != null && message.hasOwnProperty("engine"))
                    object.engine = options.enums === String ? $root.caffe.LRNParameter.Engine[message.engine] : message.engine;
                return object;
            };
    
            LRNParameter.prototype.toJSON = function toJSON() {
                return this.constructor.toObject(this, $protobuf.util.toJSONOptions);
            };
    
            LRNParameter.NormRegion = (function() {
                var valuesById = {}, values = Object.create(valuesById);
                values[valuesById[0] = "ACROSS_CHANNELS"] = 0;
                values[valuesById[1] = "WITHIN_CHANNEL"] = 1;
                return values;
            })();
    
            LRNParameter.Engine = (function() {
                var valuesById = {}, values = Object.create(valuesById);
                values[valuesById[0] = "DEFAULT"] = 0;
                values[valuesById[1] = "CAFFE"] = 1;
                values[valuesById[2] = "CUDNN"] = 2;
                return values;
            })();
    
            return LRNParameter;
        })();
    
        caffe.MemoryDataParameter = (function() {
    
            function MemoryDataParameter(properties) {
                if (properties)
                    for (var keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                        if (properties[keys[i]] != null)
                            this[keys[i]] = properties[keys[i]];
            }
    
            MemoryDataParameter.prototype.batch_size = 0;
            MemoryDataParameter.prototype.channels = 0;
            MemoryDataParameter.prototype.height = 0;
            MemoryDataParameter.prototype.width = 0;
    
            MemoryDataParameter.create = function create(properties) {
                return new MemoryDataParameter(properties);
            };
    
            MemoryDataParameter.decode = function decode(reader, length) {
                if (!(reader instanceof $Reader))
                    reader = $Reader.create(reader);
                var end = length === undefined ? reader.len : reader.pos + length, message = new $root.caffe.MemoryDataParameter();
                while (reader.pos < end) {
                    var tag = reader.uint32();
                    switch (tag >>> 3) {
                    case 1:
                        message.batch_size = reader.uint32();
                        break;
                    case 2:
                        message.channels = reader.uint32();
                        break;
                    case 3:
                        message.height = reader.uint32();
                        break;
                    case 4:
                        message.width = reader.uint32();
                        break;
                    default:
                        reader.skipType(tag & 7);
                        break;
                    }
                }
                return message;
            };
    
            MemoryDataParameter.decodeText = function decodeText(reader, block) {
                if (!(reader instanceof $TextReader))
                    reader = $TextReader.create(reader);
                var message = new $root.caffe.MemoryDataParameter();
                reader.start(block);
                while (!reader.end(block)) {
                    var tag = reader.tag();
                    switch (tag) {
                    case "batch_size":
                        message.batch_size = reader.uint32();
                        break;
                    case "channels":
                        message.channels = reader.uint32();
                        break;
                    case "height":
                        message.height = reader.uint32();
                        break;
                    case "width":
                        message.width = reader.uint32();
                        break;
                    default:
                        reader.handle(tag, message);
                        break;
                    }
                }
                return message;
            };
    
            MemoryDataParameter.verify = function verify(message) {
                if (typeof message !== "object" || message === null)
                    return "object expected";
                if (message.batch_size != null && message.hasOwnProperty("batch_size"))
                    if (!$util.isInteger(message.batch_size))
                        return "batch_size: integer expected";
                if (message.channels != null && message.hasOwnProperty("channels"))
                    if (!$util.isInteger(message.channels))
                        return "channels: integer expected";
                if (message.height != null && message.hasOwnProperty("height"))
                    if (!$util.isInteger(message.height))
                        return "height: integer expected";
                if (message.width != null && message.hasOwnProperty("width"))
                    if (!$util.isInteger(message.width))
                        return "width: integer expected";
                return null;
            };
    
            MemoryDataParameter.fromObject = function fromObject(object) {
                if (object instanceof $root.caffe.MemoryDataParameter)
                    return object;
                var message = new $root.caffe.MemoryDataParameter();
                if (object.batch_size != null)
                    message.batch_size = object.batch_size >>> 0;
                if (object.channels != null)
                    message.channels = object.channels >>> 0;
                if (object.height != null)
                    message.height = object.height >>> 0;
                if (object.width != null)
                    message.width = object.width >>> 0;
                return message;
            };
    
            MemoryDataParameter.toObject = function toObject(message, options) {
                if (!options)
                    options = {};
                var object = {};
                if (options.defaults) {
                    object.batch_size = 0;
                    object.channels = 0;
                    object.height = 0;
                    object.width = 0;
                }
                if (message.batch_size != null && message.hasOwnProperty("batch_size"))
                    object.batch_size = message.batch_size;
                if (message.channels != null && message.hasOwnProperty("channels"))
                    object.channels = message.channels;
                if (message.height != null && message.hasOwnProperty("height"))
                    object.height = message.height;
                if (message.width != null && message.hasOwnProperty("width"))
                    object.width = message.width;
                return object;
            };
    
            MemoryDataParameter.prototype.toJSON = function toJSON() {
                return this.constructor.toObject(this, $protobuf.util.toJSONOptions);
            };
    
            return MemoryDataParameter;
        })();
    
        caffe.MVNParameter = (function() {
    
            function MVNParameter(properties) {
                if (properties)
                    for (var keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                        if (properties[keys[i]] != null)
                            this[keys[i]] = properties[keys[i]];
            }
    
            MVNParameter.prototype.normalize_variance = true;
            MVNParameter.prototype.across_channels = false;
            MVNParameter.prototype.eps = 1e-9;
    
            MVNParameter.create = function create(properties) {
                return new MVNParameter(properties);
            };
    
            MVNParameter.decode = function decode(reader, length) {
                if (!(reader instanceof $Reader))
                    reader = $Reader.create(reader);
                var end = length === undefined ? reader.len : reader.pos + length, message = new $root.caffe.MVNParameter();
                while (reader.pos < end) {
                    var tag = reader.uint32();
                    switch (tag >>> 3) {
                    case 1:
                        message.normalize_variance = reader.bool();
                        break;
                    case 2:
                        message.across_channels = reader.bool();
                        break;
                    case 3:
                        message.eps = reader.float();
                        break;
                    default:
                        reader.skipType(tag & 7);
                        break;
                    }
                }
                return message;
            };
    
            MVNParameter.decodeText = function decodeText(reader, block) {
                if (!(reader instanceof $TextReader))
                    reader = $TextReader.create(reader);
                var message = new $root.caffe.MVNParameter();
                reader.start(block);
                while (!reader.end(block)) {
                    var tag = reader.tag();
                    switch (tag) {
                    case "normalize_variance":
                        message.normalize_variance = reader.bool();
                        break;
                    case "across_channels":
                        message.across_channels = reader.bool();
                        break;
                    case "eps":
                        message.eps = reader.float();
                        break;
                    default:
                        reader.handle(tag, message);
                        break;
                    }
                }
                return message;
            };
    
            MVNParameter.verify = function verify(message) {
                if (typeof message !== "object" || message === null)
                    return "object expected";
                if (message.normalize_variance != null && message.hasOwnProperty("normalize_variance"))
                    if (typeof message.normalize_variance !== "boolean")
                        return "normalize_variance: boolean expected";
                if (message.across_channels != null && message.hasOwnProperty("across_channels"))
                    if (typeof message.across_channels !== "boolean")
                        return "across_channels: boolean expected";
                if (message.eps != null && message.hasOwnProperty("eps"))
                    if (typeof message.eps !== "number")
                        return "eps: number expected";
                return null;
            };
    
            MVNParameter.fromObject = function fromObject(object) {
                if (object instanceof $root.caffe.MVNParameter)
                    return object;
                var message = new $root.caffe.MVNParameter();
                if (object.normalize_variance != null)
                    message.normalize_variance = Boolean(object.normalize_variance);
                if (object.across_channels != null)
                    message.across_channels = Boolean(object.across_channels);
                if (object.eps != null)
                    message.eps = Number(object.eps);
                return message;
            };
    
            MVNParameter.toObject = function toObject(message, options) {
                if (!options)
                    options = {};
                var object = {};
                if (options.defaults) {
                    object.normalize_variance = true;
                    object.across_channels = false;
                    object.eps = 1e-9;
                }
                if (message.normalize_variance != null && message.hasOwnProperty("normalize_variance"))
                    object.normalize_variance = message.normalize_variance;
                if (message.across_channels != null && message.hasOwnProperty("across_channels"))
                    object.across_channels = message.across_channels;
                if (message.eps != null && message.hasOwnProperty("eps"))
                    object.eps = options.json && !isFinite(message.eps) ? String(message.eps) : message.eps;
                return object;
            };
    
            MVNParameter.prototype.toJSON = function toJSON() {
                return this.constructor.toObject(this, $protobuf.util.toJSONOptions);
            };
    
            return MVNParameter;
        })();
    
        caffe.ParameterParameter = (function() {
    
            function ParameterParameter(properties) {
                if (properties)
                    for (var keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                        if (properties[keys[i]] != null)
                            this[keys[i]] = properties[keys[i]];
            }
    
            ParameterParameter.prototype.shape = null;
    
            ParameterParameter.create = function create(properties) {
                return new ParameterParameter(properties);
            };
    
            ParameterParameter.decode = function decode(reader, length) {
                if (!(reader instanceof $Reader))
                    reader = $Reader.create(reader);
                var end = length === undefined ? reader.len : reader.pos + length, message = new $root.caffe.ParameterParameter();
                while (reader.pos < end) {
                    var tag = reader.uint32();
                    switch (tag >>> 3) {
                    case 1:
                        message.shape = $root.caffe.BlobShape.decode(reader, reader.uint32());
                        break;
                    default:
                        reader.skipType(tag & 7);
                        break;
                    }
                }
                return message;
            };
    
            ParameterParameter.decodeText = function decodeText(reader, block) {
                if (!(reader instanceof $TextReader))
                    reader = $TextReader.create(reader);
                var message = new $root.caffe.ParameterParameter();
                reader.start(block);
                while (!reader.end(block)) {
                    var tag = reader.tag();
                    switch (tag) {
                    case "shape":
                        message.shape = $root.caffe.BlobShape.decodeText(reader, true);
                        break;
                    default:
                        reader.handle(tag, message);
                        break;
                    }
                }
                return message;
            };
    
            ParameterParameter.verify = function verify(message) {
                if (typeof message !== "object" || message === null)
                    return "object expected";
                if (message.shape != null && message.hasOwnProperty("shape")) {
                    var error = $root.caffe.BlobShape.verify(message.shape);
                    if (error)
                        return "shape." + error;
                }
                return null;
            };
    
            ParameterParameter.fromObject = function fromObject(object) {
                if (object instanceof $root.caffe.ParameterParameter)
                    return object;
                var message = new $root.caffe.ParameterParameter();
                if (object.shape != null) {
                    if (typeof object.shape !== "object")
                        throw TypeError(".caffe.ParameterParameter.shape: object expected");
                    message.shape = $root.caffe.BlobShape.fromObject(object.shape);
                }
                return message;
            };
    
            ParameterParameter.toObject = function toObject(message, options) {
                if (!options)
                    options = {};
                var object = {};
                if (options.defaults)
                    object.shape = null;
                if (message.shape != null && message.hasOwnProperty("shape"))
                    object.shape = $root.caffe.BlobShape.toObject(message.shape, options);
                return object;
            };
    
            ParameterParameter.prototype.toJSON = function toJSON() {
                return this.constructor.toObject(this, $protobuf.util.toJSONOptions);
            };
    
            return ParameterParameter;
        })();
    
        caffe.PoolingParameter = (function() {
    
            function PoolingParameter(properties) {
                if (properties)
                    for (var keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                        if (properties[keys[i]] != null)
                            this[keys[i]] = properties[keys[i]];
            }
    
            PoolingParameter.prototype.pool = 0;
            PoolingParameter.prototype.pad = 0;
            PoolingParameter.prototype.pad_h = 0;
            PoolingParameter.prototype.pad_w = 0;
            PoolingParameter.prototype.kernel_size = 0;
            PoolingParameter.prototype.kernel_h = 0;
            PoolingParameter.prototype.kernel_w = 0;
            PoolingParameter.prototype.stride = 1;
            PoolingParameter.prototype.stride_h = 0;
            PoolingParameter.prototype.stride_w = 0;
            PoolingParameter.prototype.engine = 0;
            PoolingParameter.prototype.global_pooling = false;
            PoolingParameter.prototype.round_mode = 0;
    
            PoolingParameter.create = function create(properties) {
                return new PoolingParameter(properties);
            };
    
            PoolingParameter.decode = function decode(reader, length) {
                if (!(reader instanceof $Reader))
                    reader = $Reader.create(reader);
                var end = length === undefined ? reader.len : reader.pos + length, message = new $root.caffe.PoolingParameter();
                while (reader.pos < end) {
                    var tag = reader.uint32();
                    switch (tag >>> 3) {
                    case 1:
                        message.pool = reader.int32();
                        break;
                    case 4:
                        message.pad = reader.uint32();
                        break;
                    case 9:
                        message.pad_h = reader.uint32();
                        break;
                    case 10:
                        message.pad_w = reader.uint32();
                        break;
                    case 2:
                        message.kernel_size = reader.uint32();
                        break;
                    case 5:
                        message.kernel_h = reader.uint32();
                        break;
                    case 6:
                        message.kernel_w = reader.uint32();
                        break;
                    case 3:
                        message.stride = reader.uint32();
                        break;
                    case 7:
                        message.stride_h = reader.uint32();
                        break;
                    case 8:
                        message.stride_w = reader.uint32();
                        break;
                    case 11:
                        message.engine = reader.int32();
                        break;
                    case 12:
                        message.global_pooling = reader.bool();
                        break;
                    case 13:
                        message.round_mode = reader.int32();
                        break;
                    default:
                        reader.skipType(tag & 7);
                        break;
                    }
                }
                return message;
            };
    
            PoolingParameter.decodeText = function decodeText(reader, block) {
                if (!(reader instanceof $TextReader))
                    reader = $TextReader.create(reader);
                var message = new $root.caffe.PoolingParameter();
                reader.start(block);
                while (!reader.end(block)) {
                    var tag = reader.tag();
                    switch (tag) {
                    case "pool":
                        message.pool = reader.enum($root.caffe.PoolingParameter.PoolMethod);
                        break;
                    case "pad":
                        message.pad = reader.uint32();
                        break;
                    case "pad_h":
                        message.pad_h = reader.uint32();
                        break;
                    case "pad_w":
                        message.pad_w = reader.uint32();
                        break;
                    case "kernel_size":
                        message.kernel_size = reader.uint32();
                        break;
                    case "kernel_h":
                        message.kernel_h = reader.uint32();
                        break;
                    case "kernel_w":
                        message.kernel_w = reader.uint32();
                        break;
                    case "stride":
                        message.stride = reader.uint32();
                        break;
                    case "stride_h":
                        message.stride_h = reader.uint32();
                        break;
                    case "stride_w":
                        message.stride_w = reader.uint32();
                        break;
                    case "engine":
                        message.engine = reader.enum($root.caffe.PoolingParameter.Engine);
                        break;
                    case "global_pooling":
                        message.global_pooling = reader.bool();
                        break;
                    case "round_mode":
                        message.round_mode = reader.enum($root.caffe.PoolingParameter.RoundMode);
                        break;
                    default:
                        reader.handle(tag, message);
                        break;
                    }
                }
                return message;
            };
    
            PoolingParameter.verify = function verify(message) {
                if (typeof message !== "object" || message === null)
                    return "object expected";
                if (message.pool != null && message.hasOwnProperty("pool"))
                    switch (message.pool) {
                    default:
                        return "pool: enum value expected";
                    case 0:
                    case 1:
                    case 2:
                        break;
                    }
                if (message.pad != null && message.hasOwnProperty("pad"))
                    if (!$util.isInteger(message.pad))
                        return "pad: integer expected";
                if (message.pad_h != null && message.hasOwnProperty("pad_h"))
                    if (!$util.isInteger(message.pad_h))
                        return "pad_h: integer expected";
                if (message.pad_w != null && message.hasOwnProperty("pad_w"))
                    if (!$util.isInteger(message.pad_w))
                        return "pad_w: integer expected";
                if (message.kernel_size != null && message.hasOwnProperty("kernel_size"))
                    if (!$util.isInteger(message.kernel_size))
                        return "kernel_size: integer expected";
                if (message.kernel_h != null && message.hasOwnProperty("kernel_h"))
                    if (!$util.isInteger(message.kernel_h))
                        return "kernel_h: integer expected";
                if (message.kernel_w != null && message.hasOwnProperty("kernel_w"))
                    if (!$util.isInteger(message.kernel_w))
                        return "kernel_w: integer expected";
                if (message.stride != null && message.hasOwnProperty("stride"))
                    if (!$util.isInteger(message.stride))
                        return "stride: integer expected";
                if (message.stride_h != null && message.hasOwnProperty("stride_h"))
                    if (!$util.isInteger(message.stride_h))
                        return "stride_h: integer expected";
                if (message.stride_w != null && message.hasOwnProperty("stride_w"))
                    if (!$util.isInteger(message.stride_w))
                        return "stride_w: integer expected";
                if (message.engine != null && message.hasOwnProperty("engine"))
                    switch (message.engine) {
                    default:
                        return "engine: enum value expected";
                    case 0:
                    case 1:
                    case 2:
                        break;
                    }
                if (message.global_pooling != null && message.hasOwnProperty("global_pooling"))
                    if (typeof message.global_pooling !== "boolean")
                        return "global_pooling: boolean expected";
                if (message.round_mode != null && message.hasOwnProperty("round_mode"))
                    switch (message.round_mode) {
                    default:
                        return "round_mode: enum value expected";
                    case 0:
                    case 1:
                        break;
                    }
                return null;
            };
    
            PoolingParameter.fromObject = function fromObject(object) {
                if (object instanceof $root.caffe.PoolingParameter)
                    return object;
                var message = new $root.caffe.PoolingParameter();
                switch (object.pool) {
                case "MAX":
                case 0:
                    message.pool = 0;
                    break;
                case "AVE":
                case 1:
                    message.pool = 1;
                    break;
                case "STOCHASTIC":
                case 2:
                    message.pool = 2;
                    break;
                }
                if (object.pad != null)
                    message.pad = object.pad >>> 0;
                if (object.pad_h != null)
                    message.pad_h = object.pad_h >>> 0;
                if (object.pad_w != null)
                    message.pad_w = object.pad_w >>> 0;
                if (object.kernel_size != null)
                    message.kernel_size = object.kernel_size >>> 0;
                if (object.kernel_h != null)
                    message.kernel_h = object.kernel_h >>> 0;
                if (object.kernel_w != null)
                    message.kernel_w = object.kernel_w >>> 0;
                if (object.stride != null)
                    message.stride = object.stride >>> 0;
                if (object.stride_h != null)
                    message.stride_h = object.stride_h >>> 0;
                if (object.stride_w != null)
                    message.stride_w = object.stride_w >>> 0;
                switch (object.engine) {
                case "DEFAULT":
                case 0:
                    message.engine = 0;
                    break;
                case "CAFFE":
                case 1:
                    message.engine = 1;
                    break;
                case "CUDNN":
                case 2:
                    message.engine = 2;
                    break;
                }
                if (object.global_pooling != null)
                    message.global_pooling = Boolean(object.global_pooling);
                switch (object.round_mode) {
                case "CEIL":
                case 0:
                    message.round_mode = 0;
                    break;
                case "FLOOR":
                case 1:
                    message.round_mode = 1;
                    break;
                }
                return message;
            };
    
            PoolingParameter.toObject = function toObject(message, options) {
                if (!options)
                    options = {};
                var object = {};
                if (options.defaults) {
                    object.pool = options.enums === String ? "MAX" : 0;
                    object.kernel_size = 0;
                    object.stride = 1;
                    object.pad = 0;
                    object.kernel_h = 0;
                    object.kernel_w = 0;
                    object.stride_h = 0;
                    object.stride_w = 0;
                    object.pad_h = 0;
                    object.pad_w = 0;
                    object.engine = options.enums === String ? "DEFAULT" : 0;
                    object.global_pooling = false;
                    object.round_mode = options.enums === String ? "CEIL" : 0;
                }
                if (message.pool != null && message.hasOwnProperty("pool"))
                    object.pool = options.enums === String ? $root.caffe.PoolingParameter.PoolMethod[message.pool] : message.pool;
                if (message.kernel_size != null && message.hasOwnProperty("kernel_size"))
                    object.kernel_size = message.kernel_size;
                if (message.stride != null && message.hasOwnProperty("stride"))
                    object.stride = message.stride;
                if (message.pad != null && message.hasOwnProperty("pad"))
                    object.pad = message.pad;
                if (message.kernel_h != null && message.hasOwnProperty("kernel_h"))
                    object.kernel_h = message.kernel_h;
                if (message.kernel_w != null && message.hasOwnProperty("kernel_w"))
                    object.kernel_w = message.kernel_w;
                if (message.stride_h != null && message.hasOwnProperty("stride_h"))
                    object.stride_h = message.stride_h;
                if (message.stride_w != null && message.hasOwnProperty("stride_w"))
                    object.stride_w = message.stride_w;
                if (message.pad_h != null && message.hasOwnProperty("pad_h"))
                    object.pad_h = message.pad_h;
                if (message.pad_w != null && message.hasOwnProperty("pad_w"))
                    object.pad_w = message.pad_w;
                if (message.engine != null && message.hasOwnProperty("engine"))
                    object.engine = options.enums === String ? $root.caffe.PoolingParameter.Engine[message.engine] : message.engine;
                if (message.global_pooling != null && message.hasOwnProperty("global_pooling"))
                    object.global_pooling = message.global_pooling;
                if (message.round_mode != null && message.hasOwnProperty("round_mode"))
                    object.round_mode = options.enums === String ? $root.caffe.PoolingParameter.RoundMode[message.round_mode] : message.round_mode;
                return object;
            };
    
            PoolingParameter.prototype.toJSON = function toJSON() {
                return this.constructor.toObject(this, $protobuf.util.toJSONOptions);
            };
    
            PoolingParameter.PoolMethod = (function() {
                var valuesById = {}, values = Object.create(valuesById);
                values[valuesById[0] = "MAX"] = 0;
                values[valuesById[1] = "AVE"] = 1;
                values[valuesById[2] = "STOCHASTIC"] = 2;
                return values;
            })();
    
            PoolingParameter.Engine = (function() {
                var valuesById = {}, values = Object.create(valuesById);
                values[valuesById[0] = "DEFAULT"] = 0;
                values[valuesById[1] = "CAFFE"] = 1;
                values[valuesById[2] = "CUDNN"] = 2;
                return values;
            })();
    
            PoolingParameter.RoundMode = (function() {
                var valuesById = {}, values = Object.create(valuesById);
                values[valuesById[0] = "CEIL"] = 0;
                values[valuesById[1] = "FLOOR"] = 1;
                return values;
            })();
    
            return PoolingParameter;
        })();
    
        caffe.PowerParameter = (function() {
    
            function PowerParameter(properties) {
                if (properties)
                    for (var keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                        if (properties[keys[i]] != null)
                            this[keys[i]] = properties[keys[i]];
            }
    
            PowerParameter.prototype.power = 1;
            PowerParameter.prototype.scale = 1;
            PowerParameter.prototype.shift = 0;
    
            PowerParameter.create = function create(properties) {
                return new PowerParameter(properties);
            };
    
            PowerParameter.decode = function decode(reader, length) {
                if (!(reader instanceof $Reader))
                    reader = $Reader.create(reader);
                var end = length === undefined ? reader.len : reader.pos + length, message = new $root.caffe.PowerParameter();
                while (reader.pos < end) {
                    var tag = reader.uint32();
                    switch (tag >>> 3) {
                    case 1:
                        message.power = reader.float();
                        break;
                    case 2:
                        message.scale = reader.float();
                        break;
                    case 3:
                        message.shift = reader.float();
                        break;
                    default:
                        reader.skipType(tag & 7);
                        break;
                    }
                }
                return message;
            };
    
            PowerParameter.decodeText = function decodeText(reader, block) {
                if (!(reader instanceof $TextReader))
                    reader = $TextReader.create(reader);
                var message = new $root.caffe.PowerParameter();
                reader.start(block);
                while (!reader.end(block)) {
                    var tag = reader.tag();
                    switch (tag) {
                    case "power":
                        message.power = reader.float();
                        break;
                    case "scale":
                        message.scale = reader.float();
                        break;
                    case "shift":
                        message.shift = reader.float();
                        break;
                    default:
                        reader.handle(tag, message);
                        break;
                    }
                }
                return message;
            };
    
            PowerParameter.verify = function verify(message) {
                if (typeof message !== "object" || message === null)
                    return "object expected";
                if (message.power != null && message.hasOwnProperty("power"))
                    if (typeof message.power !== "number")
                        return "power: number expected";
                if (message.scale != null && message.hasOwnProperty("scale"))
                    if (typeof message.scale !== "number")
                        return "scale: number expected";
                if (message.shift != null && message.hasOwnProperty("shift"))
                    if (typeof message.shift !== "number")
                        return "shift: number expected";
                return null;
            };
    
            PowerParameter.fromObject = function fromObject(object) {
                if (object instanceof $root.caffe.PowerParameter)
                    return object;
                var message = new $root.caffe.PowerParameter();
                if (object.power != null)
                    message.power = Number(object.power);
                if (object.scale != null)
                    message.scale = Number(object.scale);
                if (object.shift != null)
                    message.shift = Number(object.shift);
                return message;
            };
    
            PowerParameter.toObject = function toObject(message, options) {
                if (!options)
                    options = {};
                var object = {};
                if (options.defaults) {
                    object.power = 1;
                    object.scale = 1;
                    object.shift = 0;
                }
                if (message.power != null && message.hasOwnProperty("power"))
                    object.power = options.json && !isFinite(message.power) ? String(message.power) : message.power;
                if (message.scale != null && message.hasOwnProperty("scale"))
                    object.scale = options.json && !isFinite(message.scale) ? String(message.scale) : message.scale;
                if (message.shift != null && message.hasOwnProperty("shift"))
                    object.shift = options.json && !isFinite(message.shift) ? String(message.shift) : message.shift;
                return object;
            };
    
            PowerParameter.prototype.toJSON = function toJSON() {
                return this.constructor.toObject(this, $protobuf.util.toJSONOptions);
            };
    
            return PowerParameter;
        })();
    
        caffe.PythonParameter = (function() {
    
            function PythonParameter(properties) {
                if (properties)
                    for (var keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                        if (properties[keys[i]] != null)
                            this[keys[i]] = properties[keys[i]];
            }
    
            PythonParameter.prototype.module = "";
            PythonParameter.prototype.layer = "";
            PythonParameter.prototype.param_str = "";
            PythonParameter.prototype.share_in_parallel = false;
    
            PythonParameter.create = function create(properties) {
                return new PythonParameter(properties);
            };
    
            PythonParameter.decode = function decode(reader, length) {
                if (!(reader instanceof $Reader))
                    reader = $Reader.create(reader);
                var end = length === undefined ? reader.len : reader.pos + length, message = new $root.caffe.PythonParameter();
                while (reader.pos < end) {
                    var tag = reader.uint32();
                    switch (tag >>> 3) {
                    case 1:
                        message.module = reader.string();
                        break;
                    case 2:
                        message.layer = reader.string();
                        break;
                    case 3:
                        message.param_str = reader.string();
                        break;
                    case 4:
                        message.share_in_parallel = reader.bool();
                        break;
                    default:
                        reader.skipType(tag & 7);
                        break;
                    }
                }
                return message;
            };
    
            PythonParameter.decodeText = function decodeText(reader, block) {
                if (!(reader instanceof $TextReader))
                    reader = $TextReader.create(reader);
                var message = new $root.caffe.PythonParameter();
                reader.start(block);
                while (!reader.end(block)) {
                    var tag = reader.tag();
                    switch (tag) {
                    case "module":
                        message.module = reader.string();
                        break;
                    case "layer":
                        message.layer = reader.string();
                        break;
                    case "param_str":
                        message.param_str = reader.string();
                        break;
                    case "share_in_parallel":
                        message.share_in_parallel = reader.bool();
                        break;
                    default:
                        reader.handle(tag, message);
                        break;
                    }
                }
                return message;
            };
    
            PythonParameter.verify = function verify(message) {
                if (typeof message !== "object" || message === null)
                    return "object expected";
                if (message.module != null && message.hasOwnProperty("module"))
                    if (!$util.isString(message.module))
                        return "module: string expected";
                if (message.layer != null && message.hasOwnProperty("layer"))
                    if (!$util.isString(message.layer))
                        return "layer: string expected";
                if (message.param_str != null && message.hasOwnProperty("param_str"))
                    if (!$util.isString(message.param_str))
                        return "param_str: string expected";
                if (message.share_in_parallel != null && message.hasOwnProperty("share_in_parallel"))
                    if (typeof message.share_in_parallel !== "boolean")
                        return "share_in_parallel: boolean expected";
                return null;
            };
    
            PythonParameter.fromObject = function fromObject(object) {
                if (object instanceof $root.caffe.PythonParameter)
                    return object;
                var message = new $root.caffe.PythonParameter();
                if (object.module != null)
                    message.module = String(object.module);
                if (object.layer != null)
                    message.layer = String(object.layer);
                if (object.param_str != null)
                    message.param_str = String(object.param_str);
                if (object.share_in_parallel != null)
                    message.share_in_parallel = Boolean(object.share_in_parallel);
                return message;
            };
    
            PythonParameter.toObject = function toObject(message, options) {
                if (!options)
                    options = {};
                var object = {};
                if (options.defaults) {
                    object.module = "";
                    object.layer = "";
                    object.param_str = "";
                    object.share_in_parallel = false;
                }
                if (message.module != null && message.hasOwnProperty("module"))
                    object.module = message.module;
                if (message.layer != null && message.hasOwnProperty("layer"))
                    object.layer = message.layer;
                if (message.param_str != null && message.hasOwnProperty("param_str"))
                    object.param_str = message.param_str;
                if (message.share_in_parallel != null && message.hasOwnProperty("share_in_parallel"))
                    object.share_in_parallel = message.share_in_parallel;
                return object;
            };
    
            PythonParameter.prototype.toJSON = function toJSON() {
                return this.constructor.toObject(this, $protobuf.util.toJSONOptions);
            };
    
            return PythonParameter;
        })();
    
        caffe.RecurrentParameter = (function() {
    
            function RecurrentParameter(properties) {
                if (properties)
                    for (var keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                        if (properties[keys[i]] != null)
                            this[keys[i]] = properties[keys[i]];
            }
    
            RecurrentParameter.prototype.num_output = 0;
            RecurrentParameter.prototype.weight_filler = null;
            RecurrentParameter.prototype.bias_filler = null;
            RecurrentParameter.prototype.debug_info = false;
            RecurrentParameter.prototype.expose_hidden = false;
    
            RecurrentParameter.create = function create(properties) {
                return new RecurrentParameter(properties);
            };
    
            RecurrentParameter.decode = function decode(reader, length) {
                if (!(reader instanceof $Reader))
                    reader = $Reader.create(reader);
                var end = length === undefined ? reader.len : reader.pos + length, message = new $root.caffe.RecurrentParameter();
                while (reader.pos < end) {
                    var tag = reader.uint32();
                    switch (tag >>> 3) {
                    case 1:
                        message.num_output = reader.uint32();
                        break;
                    case 2:
                        message.weight_filler = $root.caffe.FillerParameter.decode(reader, reader.uint32());
                        break;
                    case 3:
                        message.bias_filler = $root.caffe.FillerParameter.decode(reader, reader.uint32());
                        break;
                    case 4:
                        message.debug_info = reader.bool();
                        break;
                    case 5:
                        message.expose_hidden = reader.bool();
                        break;
                    default:
                        reader.skipType(tag & 7);
                        break;
                    }
                }
                return message;
            };
    
            RecurrentParameter.decodeText = function decodeText(reader, block) {
                if (!(reader instanceof $TextReader))
                    reader = $TextReader.create(reader);
                var message = new $root.caffe.RecurrentParameter();
                reader.start(block);
                while (!reader.end(block)) {
                    var tag = reader.tag();
                    switch (tag) {
                    case "num_output":
                        message.num_output = reader.uint32();
                        break;
                    case "weight_filler":
                        message.weight_filler = $root.caffe.FillerParameter.decodeText(reader, true);
                        break;
                    case "bias_filler":
                        message.bias_filler = $root.caffe.FillerParameter.decodeText(reader, true);
                        break;
                    case "debug_info":
                        message.debug_info = reader.bool();
                        break;
                    case "expose_hidden":
                        message.expose_hidden = reader.bool();
                        break;
                    default:
                        reader.handle(tag, message);
                        break;
                    }
                }
                return message;
            };
    
            RecurrentParameter.verify = function verify(message) {
                if (typeof message !== "object" || message === null)
                    return "object expected";
                if (message.num_output != null && message.hasOwnProperty("num_output"))
                    if (!$util.isInteger(message.num_output))
                        return "num_output: integer expected";
                if (message.weight_filler != null && message.hasOwnProperty("weight_filler")) {
                    var error = $root.caffe.FillerParameter.verify(message.weight_filler);
                    if (error)
                        return "weight_filler." + error;
                }
                if (message.bias_filler != null && message.hasOwnProperty("bias_filler")) {
                    var error = $root.caffe.FillerParameter.verify(message.bias_filler);
                    if (error)
                        return "bias_filler." + error;
                }
                if (message.debug_info != null && message.hasOwnProperty("debug_info"))
                    if (typeof message.debug_info !== "boolean")
                        return "debug_info: boolean expected";
                if (message.expose_hidden != null && message.hasOwnProperty("expose_hidden"))
                    if (typeof message.expose_hidden !== "boolean")
                        return "expose_hidden: boolean expected";
                return null;
            };
    
            RecurrentParameter.fromObject = function fromObject(object) {
                if (object instanceof $root.caffe.RecurrentParameter)
                    return object;
                var message = new $root.caffe.RecurrentParameter();
                if (object.num_output != null)
                    message.num_output = object.num_output >>> 0;
                if (object.weight_filler != null) {
                    if (typeof object.weight_filler !== "object")
                        throw TypeError(".caffe.RecurrentParameter.weight_filler: object expected");
                    message.weight_filler = $root.caffe.FillerParameter.fromObject(object.weight_filler);
                }
                if (object.bias_filler != null) {
                    if (typeof object.bias_filler !== "object")
                        throw TypeError(".caffe.RecurrentParameter.bias_filler: object expected");
                    message.bias_filler = $root.caffe.FillerParameter.fromObject(object.bias_filler);
                }
                if (object.debug_info != null)
                    message.debug_info = Boolean(object.debug_info);
                if (object.expose_hidden != null)
                    message.expose_hidden = Boolean(object.expose_hidden);
                return message;
            };
    
            RecurrentParameter.toObject = function toObject(message, options) {
                if (!options)
                    options = {};
                var object = {};
                if (options.defaults) {
                    object.num_output = 0;
                    object.weight_filler = null;
                    object.bias_filler = null;
                    object.debug_info = false;
                    object.expose_hidden = false;
                }
                if (message.num_output != null && message.hasOwnProperty("num_output"))
                    object.num_output = message.num_output;
                if (message.weight_filler != null && message.hasOwnProperty("weight_filler"))
                    object.weight_filler = $root.caffe.FillerParameter.toObject(message.weight_filler, options);
                if (message.bias_filler != null && message.hasOwnProperty("bias_filler"))
                    object.bias_filler = $root.caffe.FillerParameter.toObject(message.bias_filler, options);
                if (message.debug_info != null && message.hasOwnProperty("debug_info"))
                    object.debug_info = message.debug_info;
                if (message.expose_hidden != null && message.hasOwnProperty("expose_hidden"))
                    object.expose_hidden = message.expose_hidden;
                return object;
            };
    
            RecurrentParameter.prototype.toJSON = function toJSON() {
                return this.constructor.toObject(this, $protobuf.util.toJSONOptions);
            };
    
            return RecurrentParameter;
        })();
    
        caffe.ReductionParameter = (function() {
    
            function ReductionParameter(properties) {
                if (properties)
                    for (var keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                        if (properties[keys[i]] != null)
                            this[keys[i]] = properties[keys[i]];
            }
    
            ReductionParameter.prototype.operation = 1;
            ReductionParameter.prototype.axis = 0;
            ReductionParameter.prototype.coeff = 1;
    
            ReductionParameter.create = function create(properties) {
                return new ReductionParameter(properties);
            };
    
            ReductionParameter.decode = function decode(reader, length) {
                if (!(reader instanceof $Reader))
                    reader = $Reader.create(reader);
                var end = length === undefined ? reader.len : reader.pos + length, message = new $root.caffe.ReductionParameter();
                while (reader.pos < end) {
                    var tag = reader.uint32();
                    switch (tag >>> 3) {
                    case 1:
                        message.operation = reader.int32();
                        break;
                    case 2:
                        message.axis = reader.int32();
                        break;
                    case 3:
                        message.coeff = reader.float();
                        break;
                    default:
                        reader.skipType(tag & 7);
                        break;
                    }
                }
                return message;
            };
    
            ReductionParameter.decodeText = function decodeText(reader, block) {
                if (!(reader instanceof $TextReader))
                    reader = $TextReader.create(reader);
                var message = new $root.caffe.ReductionParameter();
                reader.start(block);
                while (!reader.end(block)) {
                    var tag = reader.tag();
                    switch (tag) {
                    case "operation":
                        message.operation = reader.enum($root.caffe.ReductionParameter.ReductionOp);
                        break;
                    case "axis":
                        message.axis = reader.int32();
                        break;
                    case "coeff":
                        message.coeff = reader.float();
                        break;
                    default:
                        reader.handle(tag, message);
                        break;
                    }
                }
                return message;
            };
    
            ReductionParameter.verify = function verify(message) {
                if (typeof message !== "object" || message === null)
                    return "object expected";
                if (message.operation != null && message.hasOwnProperty("operation"))
                    switch (message.operation) {
                    default:
                        return "operation: enum value expected";
                    case 1:
                    case 2:
                    case 3:
                    case 4:
                        break;
                    }
                if (message.axis != null && message.hasOwnProperty("axis"))
                    if (!$util.isInteger(message.axis))
                        return "axis: integer expected";
                if (message.coeff != null && message.hasOwnProperty("coeff"))
                    if (typeof message.coeff !== "number")
                        return "coeff: number expected";
                return null;
            };
    
            ReductionParameter.fromObject = function fromObject(object) {
                if (object instanceof $root.caffe.ReductionParameter)
                    return object;
                var message = new $root.caffe.ReductionParameter();
                switch (object.operation) {
                case "SUM":
                case 1:
                    message.operation = 1;
                    break;
                case "ASUM":
                case 2:
                    message.operation = 2;
                    break;
                case "SUMSQ":
                case 3:
                    message.operation = 3;
                    break;
                case "MEAN":
                case 4:
                    message.operation = 4;
                    break;
                }
                if (object.axis != null)
                    message.axis = object.axis | 0;
                if (object.coeff != null)
                    message.coeff = Number(object.coeff);
                return message;
            };
    
            ReductionParameter.toObject = function toObject(message, options) {
                if (!options)
                    options = {};
                var object = {};
                if (options.defaults) {
                    object.operation = options.enums === String ? "SUM" : 1;
                    object.axis = 0;
                    object.coeff = 1;
                }
                if (message.operation != null && message.hasOwnProperty("operation"))
                    object.operation = options.enums === String ? $root.caffe.ReductionParameter.ReductionOp[message.operation] : message.operation;
                if (message.axis != null && message.hasOwnProperty("axis"))
                    object.axis = message.axis;
                if (message.coeff != null && message.hasOwnProperty("coeff"))
                    object.coeff = options.json && !isFinite(message.coeff) ? String(message.coeff) : message.coeff;
                return object;
            };
    
            ReductionParameter.prototype.toJSON = function toJSON() {
                return this.constructor.toObject(this, $protobuf.util.toJSONOptions);
            };
    
            ReductionParameter.ReductionOp = (function() {
                var valuesById = {}, values = Object.create(valuesById);
                values[valuesById[1] = "SUM"] = 1;
                values[valuesById[2] = "ASUM"] = 2;
                values[valuesById[3] = "SUMSQ"] = 3;
                values[valuesById[4] = "MEAN"] = 4;
                return values;
            })();
    
            return ReductionParameter;
        })();
    
        caffe.ReLUParameter = (function() {
    
            function ReLUParameter(properties) {
                if (properties)
                    for (var keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                        if (properties[keys[i]] != null)
                            this[keys[i]] = properties[keys[i]];
            }
    
            ReLUParameter.prototype.negative_slope = 0;
            ReLUParameter.prototype.engine = 0;
    
            ReLUParameter.create = function create(properties) {
                return new ReLUParameter(properties);
            };
    
            ReLUParameter.decode = function decode(reader, length) {
                if (!(reader instanceof $Reader))
                    reader = $Reader.create(reader);
                var end = length === undefined ? reader.len : reader.pos + length, message = new $root.caffe.ReLUParameter();
                while (reader.pos < end) {
                    var tag = reader.uint32();
                    switch (tag >>> 3) {
                    case 1:
                        message.negative_slope = reader.float();
                        break;
                    case 2:
                        message.engine = reader.int32();
                        break;
                    default:
                        reader.skipType(tag & 7);
                        break;
                    }
                }
                return message;
            };
    
            ReLUParameter.decodeText = function decodeText(reader, block) {
                if (!(reader instanceof $TextReader))
                    reader = $TextReader.create(reader);
                var message = new $root.caffe.ReLUParameter();
                reader.start(block);
                while (!reader.end(block)) {
                    var tag = reader.tag();
                    switch (tag) {
                    case "negative_slope":
                        message.negative_slope = reader.float();
                        break;
                    case "engine":
                        message.engine = reader.enum($root.caffe.ReLUParameter.Engine);
                        break;
                    default:
                        reader.handle(tag, message);
                        break;
                    }
                }
                return message;
            };
    
            ReLUParameter.verify = function verify(message) {
                if (typeof message !== "object" || message === null)
                    return "object expected";
                if (message.negative_slope != null && message.hasOwnProperty("negative_slope"))
                    if (typeof message.negative_slope !== "number")
                        return "negative_slope: number expected";
                if (message.engine != null && message.hasOwnProperty("engine"))
                    switch (message.engine) {
                    default:
                        return "engine: enum value expected";
                    case 0:
                    case 1:
                    case 2:
                        break;
                    }
                return null;
            };
    
            ReLUParameter.fromObject = function fromObject(object) {
                if (object instanceof $root.caffe.ReLUParameter)
                    return object;
                var message = new $root.caffe.ReLUParameter();
                if (object.negative_slope != null)
                    message.negative_slope = Number(object.negative_slope);
                switch (object.engine) {
                case "DEFAULT":
                case 0:
                    message.engine = 0;
                    break;
                case "CAFFE":
                case 1:
                    message.engine = 1;
                    break;
                case "CUDNN":
                case 2:
                    message.engine = 2;
                    break;
                }
                return message;
            };
    
            ReLUParameter.toObject = function toObject(message, options) {
                if (!options)
                    options = {};
                var object = {};
                if (options.defaults) {
                    object.negative_slope = 0;
                    object.engine = options.enums === String ? "DEFAULT" : 0;
                }
                if (message.negative_slope != null && message.hasOwnProperty("negative_slope"))
                    object.negative_slope = options.json && !isFinite(message.negative_slope) ? String(message.negative_slope) : message.negative_slope;
                if (message.engine != null && message.hasOwnProperty("engine"))
                    object.engine = options.enums === String ? $root.caffe.ReLUParameter.Engine[message.engine] : message.engine;
                return object;
            };
    
            ReLUParameter.prototype.toJSON = function toJSON() {
                return this.constructor.toObject(this, $protobuf.util.toJSONOptions);
            };
    
            ReLUParameter.Engine = (function() {
                var valuesById = {}, values = Object.create(valuesById);
                values[valuesById[0] = "DEFAULT"] = 0;
                values[valuesById[1] = "CAFFE"] = 1;
                values[valuesById[2] = "CUDNN"] = 2;
                return values;
            })();
    
            return ReLUParameter;
        })();
    
        caffe.ReshapeParameter = (function() {
    
            function ReshapeParameter(properties) {
                if (properties)
                    for (var keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                        if (properties[keys[i]] != null)
                            this[keys[i]] = properties[keys[i]];
            }
    
            ReshapeParameter.prototype.shape = null;
            ReshapeParameter.prototype.axis = 0;
            ReshapeParameter.prototype.num_axes = -1;
    
            ReshapeParameter.create = function create(properties) {
                return new ReshapeParameter(properties);
            };
    
            ReshapeParameter.decode = function decode(reader, length) {
                if (!(reader instanceof $Reader))
                    reader = $Reader.create(reader);
                var end = length === undefined ? reader.len : reader.pos + length, message = new $root.caffe.ReshapeParameter();
                while (reader.pos < end) {
                    var tag = reader.uint32();
                    switch (tag >>> 3) {
                    case 1:
                        message.shape = $root.caffe.BlobShape.decode(reader, reader.uint32());
                        break;
                    case 2:
                        message.axis = reader.int32();
                        break;
                    case 3:
                        message.num_axes = reader.int32();
                        break;
                    default:
                        reader.skipType(tag & 7);
                        break;
                    }
                }
                return message;
            };
    
            ReshapeParameter.decodeText = function decodeText(reader, block) {
                if (!(reader instanceof $TextReader))
                    reader = $TextReader.create(reader);
                var message = new $root.caffe.ReshapeParameter();
                reader.start(block);
                while (!reader.end(block)) {
                    var tag = reader.tag();
                    switch (tag) {
                    case "shape":
                        message.shape = $root.caffe.BlobShape.decodeText(reader, true);
                        break;
                    case "axis":
                        message.axis = reader.int32();
                        break;
                    case "num_axes":
                        message.num_axes = reader.int32();
                        break;
                    default:
                        reader.handle(tag, message);
                        break;
                    }
                }
                return message;
            };
    
            ReshapeParameter.verify = function verify(message) {
                if (typeof message !== "object" || message === null)
                    return "object expected";
                if (message.shape != null && message.hasOwnProperty("shape")) {
                    var error = $root.caffe.BlobShape.verify(message.shape);
                    if (error)
                        return "shape." + error;
                }
                if (message.axis != null && message.hasOwnProperty("axis"))
                    if (!$util.isInteger(message.axis))
                        return "axis: integer expected";
                if (message.num_axes != null && message.hasOwnProperty("num_axes"))
                    if (!$util.isInteger(message.num_axes))
                        return "num_axes: integer expected";
                return null;
            };
    
            ReshapeParameter.fromObject = function fromObject(object) {
                if (object instanceof $root.caffe.ReshapeParameter)
                    return object;
                var message = new $root.caffe.ReshapeParameter();
                if (object.shape != null) {
                    if (typeof object.shape !== "object")
                        throw TypeError(".caffe.ReshapeParameter.shape: object expected");
                    message.shape = $root.caffe.BlobShape.fromObject(object.shape);
                }
                if (object.axis != null)
                    message.axis = object.axis | 0;
                if (object.num_axes != null)
                    message.num_axes = object.num_axes | 0;
                return message;
            };
    
            ReshapeParameter.toObject = function toObject(message, options) {
                if (!options)
                    options = {};
                var object = {};
                if (options.defaults) {
                    object.shape = null;
                    object.axis = 0;
                    object.num_axes = -1;
                }
                if (message.shape != null && message.hasOwnProperty("shape"))
                    object.shape = $root.caffe.BlobShape.toObject(message.shape, options);
                if (message.axis != null && message.hasOwnProperty("axis"))
                    object.axis = message.axis;
                if (message.num_axes != null && message.hasOwnProperty("num_axes"))
                    object.num_axes = message.num_axes;
                return object;
            };
    
            ReshapeParameter.prototype.toJSON = function toJSON() {
                return this.constructor.toObject(this, $protobuf.util.toJSONOptions);
            };
    
            return ReshapeParameter;
        })();
    
        caffe.ScaleParameter = (function() {
    
            function ScaleParameter(properties) {
                if (properties)
                    for (var keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                        if (properties[keys[i]] != null)
                            this[keys[i]] = properties[keys[i]];
            }
    
            ScaleParameter.prototype.axis = 1;
            ScaleParameter.prototype.num_axes = 1;
            ScaleParameter.prototype.filler = null;
            ScaleParameter.prototype.bias_term = false;
            ScaleParameter.prototype.bias_filler = null;
    
            ScaleParameter.create = function create(properties) {
                return new ScaleParameter(properties);
            };
    
            ScaleParameter.decode = function decode(reader, length) {
                if (!(reader instanceof $Reader))
                    reader = $Reader.create(reader);
                var end = length === undefined ? reader.len : reader.pos + length, message = new $root.caffe.ScaleParameter();
                while (reader.pos < end) {
                    var tag = reader.uint32();
                    switch (tag >>> 3) {
                    case 1:
                        message.axis = reader.int32();
                        break;
                    case 2:
                        message.num_axes = reader.int32();
                        break;
                    case 3:
                        message.filler = $root.caffe.FillerParameter.decode(reader, reader.uint32());
                        break;
                    case 4:
                        message.bias_term = reader.bool();
                        break;
                    case 5:
                        message.bias_filler = $root.caffe.FillerParameter.decode(reader, reader.uint32());
                        break;
                    default:
                        reader.skipType(tag & 7);
                        break;
                    }
                }
                return message;
            };
    
            ScaleParameter.decodeText = function decodeText(reader, block) {
                if (!(reader instanceof $TextReader))
                    reader = $TextReader.create(reader);
                var message = new $root.caffe.ScaleParameter();
                reader.start(block);
                while (!reader.end(block)) {
                    var tag = reader.tag();
                    switch (tag) {
                    case "axis":
                        message.axis = reader.int32();
                        break;
                    case "num_axes":
                        message.num_axes = reader.int32();
                        break;
                    case "filler":
                        message.filler = $root.caffe.FillerParameter.decodeText(reader, true);
                        break;
                    case "bias_term":
                        message.bias_term = reader.bool();
                        break;
                    case "bias_filler":
                        message.bias_filler = $root.caffe.FillerParameter.decodeText(reader, true);
                        break;
                    default:
                        reader.handle(tag, message);
                        break;
                    }
                }
                return message;
            };
    
            ScaleParameter.verify = function verify(message) {
                if (typeof message !== "object" || message === null)
                    return "object expected";
                if (message.axis != null && message.hasOwnProperty("axis"))
                    if (!$util.isInteger(message.axis))
                        return "axis: integer expected";
                if (message.num_axes != null && message.hasOwnProperty("num_axes"))
                    if (!$util.isInteger(message.num_axes))
                        return "num_axes: integer expected";
                if (message.filler != null && message.hasOwnProperty("filler")) {
                    var error = $root.caffe.FillerParameter.verify(message.filler);
                    if (error)
                        return "filler." + error;
                }
                if (message.bias_term != null && message.hasOwnProperty("bias_term"))
                    if (typeof message.bias_term !== "boolean")
                        return "bias_term: boolean expected";
                if (message.bias_filler != null && message.hasOwnProperty("bias_filler")) {
                    var error = $root.caffe.FillerParameter.verify(message.bias_filler);
                    if (error)
                        return "bias_filler." + error;
                }
                return null;
            };
    
            ScaleParameter.fromObject = function fromObject(object) {
                if (object instanceof $root.caffe.ScaleParameter)
                    return object;
                var message = new $root.caffe.ScaleParameter();
                if (object.axis != null)
                    message.axis = object.axis | 0;
                if (object.num_axes != null)
                    message.num_axes = object.num_axes | 0;
                if (object.filler != null) {
                    if (typeof object.filler !== "object")
                        throw TypeError(".caffe.ScaleParameter.filler: object expected");
                    message.filler = $root.caffe.FillerParameter.fromObject(object.filler);
                }
                if (object.bias_term != null)
                    message.bias_term = Boolean(object.bias_term);
                if (object.bias_filler != null) {
                    if (typeof object.bias_filler !== "object")
                        throw TypeError(".caffe.ScaleParameter.bias_filler: object expected");
                    message.bias_filler = $root.caffe.FillerParameter.fromObject(object.bias_filler);
                }
                return message;
            };
    
            ScaleParameter.toObject = function toObject(message, options) {
                if (!options)
                    options = {};
                var object = {};
                if (options.defaults) {
                    object.axis = 1;
                    object.num_axes = 1;
                    object.filler = null;
                    object.bias_term = false;
                    object.bias_filler = null;
                }
                if (message.axis != null && message.hasOwnProperty("axis"))
                    object.axis = message.axis;
                if (message.num_axes != null && message.hasOwnProperty("num_axes"))
                    object.num_axes = message.num_axes;
                if (message.filler != null && message.hasOwnProperty("filler"))
                    object.filler = $root.caffe.FillerParameter.toObject(message.filler, options);
                if (message.bias_term != null && message.hasOwnProperty("bias_term"))
                    object.bias_term = message.bias_term;
                if (message.bias_filler != null && message.hasOwnProperty("bias_filler"))
                    object.bias_filler = $root.caffe.FillerParameter.toObject(message.bias_filler, options);
                return object;
            };
    
            ScaleParameter.prototype.toJSON = function toJSON() {
                return this.constructor.toObject(this, $protobuf.util.toJSONOptions);
            };
    
            return ScaleParameter;
        })();
    
        caffe.SigmoidParameter = (function() {
    
            function SigmoidParameter(properties) {
                if (properties)
                    for (var keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                        if (properties[keys[i]] != null)
                            this[keys[i]] = properties[keys[i]];
            }
    
            SigmoidParameter.prototype.engine = 0;
    
            SigmoidParameter.create = function create(properties) {
                return new SigmoidParameter(properties);
            };
    
            SigmoidParameter.decode = function decode(reader, length) {
                if (!(reader instanceof $Reader))
                    reader = $Reader.create(reader);
                var end = length === undefined ? reader.len : reader.pos + length, message = new $root.caffe.SigmoidParameter();
                while (reader.pos < end) {
                    var tag = reader.uint32();
                    switch (tag >>> 3) {
                    case 1:
                        message.engine = reader.int32();
                        break;
                    default:
                        reader.skipType(tag & 7);
                        break;
                    }
                }
                return message;
            };
    
            SigmoidParameter.decodeText = function decodeText(reader, block) {
                if (!(reader instanceof $TextReader))
                    reader = $TextReader.create(reader);
                var message = new $root.caffe.SigmoidParameter();
                reader.start(block);
                while (!reader.end(block)) {
                    var tag = reader.tag();
                    switch (tag) {
                    case "engine":
                        message.engine = reader.enum($root.caffe.SigmoidParameter.Engine);
                        break;
                    default:
                        reader.handle(tag, message);
                        break;
                    }
                }
                return message;
            };
    
            SigmoidParameter.verify = function verify(message) {
                if (typeof message !== "object" || message === null)
                    return "object expected";
                if (message.engine != null && message.hasOwnProperty("engine"))
                    switch (message.engine) {
                    default:
                        return "engine: enum value expected";
                    case 0:
                    case 1:
                    case 2:
                        break;
                    }
                return null;
            };
    
            SigmoidParameter.fromObject = function fromObject(object) {
                if (object instanceof $root.caffe.SigmoidParameter)
                    return object;
                var message = new $root.caffe.SigmoidParameter();
                switch (object.engine) {
                case "DEFAULT":
                case 0:
                    message.engine = 0;
                    break;
                case "CAFFE":
                case 1:
                    message.engine = 1;
                    break;
                case "CUDNN":
                case 2:
                    message.engine = 2;
                    break;
                }
                return message;
            };
    
            SigmoidParameter.toObject = function toObject(message, options) {
                if (!options)
                    options = {};
                var object = {};
                if (options.defaults)
                    object.engine = options.enums === String ? "DEFAULT" : 0;
                if (message.engine != null && message.hasOwnProperty("engine"))
                    object.engine = options.enums === String ? $root.caffe.SigmoidParameter.Engine[message.engine] : message.engine;
                return object;
            };
    
            SigmoidParameter.prototype.toJSON = function toJSON() {
                return this.constructor.toObject(this, $protobuf.util.toJSONOptions);
            };
    
            SigmoidParameter.Engine = (function() {
                var valuesById = {}, values = Object.create(valuesById);
                values[valuesById[0] = "DEFAULT"] = 0;
                values[valuesById[1] = "CAFFE"] = 1;
                values[valuesById[2] = "CUDNN"] = 2;
                return values;
            })();
    
            return SigmoidParameter;
        })();
    
        caffe.SliceParameter = (function() {
    
            function SliceParameter(properties) {
                this.slice_point = [];
                if (properties)
                    for (var keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                        if (properties[keys[i]] != null)
                            this[keys[i]] = properties[keys[i]];
            }
    
            SliceParameter.prototype.axis = 1;
            SliceParameter.prototype.slice_point = $util.emptyArray;
            SliceParameter.prototype.slice_dim = 1;
    
            SliceParameter.create = function create(properties) {
                return new SliceParameter(properties);
            };
    
            SliceParameter.decode = function decode(reader, length) {
                if (!(reader instanceof $Reader))
                    reader = $Reader.create(reader);
                var end = length === undefined ? reader.len : reader.pos + length, message = new $root.caffe.SliceParameter();
                while (reader.pos < end) {
                    var tag = reader.uint32();
                    switch (tag >>> 3) {
                    case 3:
                        message.axis = reader.int32();
                        break;
                    case 2:
                        if (!(message.slice_point && message.slice_point.length))
                            message.slice_point = [];
                        if ((tag & 7) === 2) {
                            var end2 = reader.uint32() + reader.pos;
                            while (reader.pos < end2)
                                message.slice_point.push(reader.uint32());
                        } else
                            message.slice_point.push(reader.uint32());
                        break;
                    case 1:
                        message.slice_dim = reader.uint32();
                        break;
                    default:
                        reader.skipType(tag & 7);
                        break;
                    }
                }
                return message;
            };
    
            SliceParameter.decodeText = function decodeText(reader, block) {
                if (!(reader instanceof $TextReader))
                    reader = $TextReader.create(reader);
                var message = new $root.caffe.SliceParameter();
                reader.start(block);
                while (!reader.end(block)) {
                    var tag = reader.tag();
                    switch (tag) {
                    case "axis":
                        message.axis = reader.int32();
                        break;
                    case "slice_point":
                        if (!(message.slice_point && message.slice_point.length))
                            message.slice_point = [];
                        message.slice_point.push(reader.uint32());
                        break;
                    case "slice_dim":
                        message.slice_dim = reader.uint32();
                        break;
                    default:
                        reader.handle(tag, message);
                        break;
                    }
                }
                return message;
            };
    
            SliceParameter.verify = function verify(message) {
                if (typeof message !== "object" || message === null)
                    return "object expected";
                if (message.axis != null && message.hasOwnProperty("axis"))
                    if (!$util.isInteger(message.axis))
                        return "axis: integer expected";
                if (message.slice_point != null && message.hasOwnProperty("slice_point")) {
                    if (!Array.isArray(message.slice_point))
                        return "slice_point: array expected";
                    for (var i = 0; i < message.slice_point.length; ++i)
                        if (!$util.isInteger(message.slice_point[i]))
                            return "slice_point: integer[] expected";
                }
                if (message.slice_dim != null && message.hasOwnProperty("slice_dim"))
                    if (!$util.isInteger(message.slice_dim))
                        return "slice_dim: integer expected";
                return null;
            };
    
            SliceParameter.fromObject = function fromObject(object) {
                if (object instanceof $root.caffe.SliceParameter)
                    return object;
                var message = new $root.caffe.SliceParameter();
                if (object.axis != null)
                    message.axis = object.axis | 0;
                if (object.slice_point) {
                    if (!Array.isArray(object.slice_point))
                        throw TypeError(".caffe.SliceParameter.slice_point: array expected");
                    message.slice_point = [];
                    for (var i = 0; i < object.slice_point.length; ++i)
                        message.slice_point[i] = object.slice_point[i] >>> 0;
                }
                if (object.slice_dim != null)
                    message.slice_dim = object.slice_dim >>> 0;
                return message;
            };
    
            SliceParameter.toObject = function toObject(message, options) {
                if (!options)
                    options = {};
                var object = {};
                if (options.arrays || options.defaults)
                    object.slice_point = [];
                if (options.defaults) {
                    object.slice_dim = 1;
                    object.axis = 1;
                }
                if (message.slice_dim != null && message.hasOwnProperty("slice_dim"))
                    object.slice_dim = message.slice_dim;
                if (message.slice_point && message.slice_point.length) {
                    object.slice_point = [];
                    for (var j = 0; j < message.slice_point.length; ++j)
                        object.slice_point[j] = message.slice_point[j];
                }
                if (message.axis != null && message.hasOwnProperty("axis"))
                    object.axis = message.axis;
                return object;
            };
    
            SliceParameter.prototype.toJSON = function toJSON() {
                return this.constructor.toObject(this, $protobuf.util.toJSONOptions);
            };
    
            return SliceParameter;
        })();
    
        caffe.SoftmaxParameter = (function() {
    
            function SoftmaxParameter(properties) {
                if (properties)
                    for (var keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                        if (properties[keys[i]] != null)
                            this[keys[i]] = properties[keys[i]];
            }
    
            SoftmaxParameter.prototype.engine = 0;
            SoftmaxParameter.prototype.axis = 1;
    
            SoftmaxParameter.create = function create(properties) {
                return new SoftmaxParameter(properties);
            };
    
            SoftmaxParameter.decode = function decode(reader, length) {
                if (!(reader instanceof $Reader))
                    reader = $Reader.create(reader);
                var end = length === undefined ? reader.len : reader.pos + length, message = new $root.caffe.SoftmaxParameter();
                while (reader.pos < end) {
                    var tag = reader.uint32();
                    switch (tag >>> 3) {
                    case 1:
                        message.engine = reader.int32();
                        break;
                    case 2:
                        message.axis = reader.int32();
                        break;
                    default:
                        reader.skipType(tag & 7);
                        break;
                    }
                }
                return message;
            };
    
            SoftmaxParameter.decodeText = function decodeText(reader, block) {
                if (!(reader instanceof $TextReader))
                    reader = $TextReader.create(reader);
                var message = new $root.caffe.SoftmaxParameter();
                reader.start(block);
                while (!reader.end(block)) {
                    var tag = reader.tag();
                    switch (tag) {
                    case "engine":
                        message.engine = reader.enum($root.caffe.SoftmaxParameter.Engine);
                        break;
                    case "axis":
                        message.axis = reader.int32();
                        break;
                    default:
                        reader.handle(tag, message);
                        break;
                    }
                }
                return message;
            };
    
            SoftmaxParameter.verify = function verify(message) {
                if (typeof message !== "object" || message === null)
                    return "object expected";
                if (message.engine != null && message.hasOwnProperty("engine"))
                    switch (message.engine) {
                    default:
                        return "engine: enum value expected";
                    case 0:
                    case 1:
                    case 2:
                        break;
                    }
                if (message.axis != null && message.hasOwnProperty("axis"))
                    if (!$util.isInteger(message.axis))
                        return "axis: integer expected";
                return null;
            };
    
            SoftmaxParameter.fromObject = function fromObject(object) {
                if (object instanceof $root.caffe.SoftmaxParameter)
                    return object;
                var message = new $root.caffe.SoftmaxParameter();
                switch (object.engine) {
                case "DEFAULT":
                case 0:
                    message.engine = 0;
                    break;
                case "CAFFE":
                case 1:
                    message.engine = 1;
                    break;
                case "CUDNN":
                case 2:
                    message.engine = 2;
                    break;
                }
                if (object.axis != null)
                    message.axis = object.axis | 0;
                return message;
            };
    
            SoftmaxParameter.toObject = function toObject(message, options) {
                if (!options)
                    options = {};
                var object = {};
                if (options.defaults) {
                    object.engine = options.enums === String ? "DEFAULT" : 0;
                    object.axis = 1;
                }
                if (message.engine != null && message.hasOwnProperty("engine"))
                    object.engine = options.enums === String ? $root.caffe.SoftmaxParameter.Engine[message.engine] : message.engine;
                if (message.axis != null && message.hasOwnProperty("axis"))
                    object.axis = message.axis;
                return object;
            };
    
            SoftmaxParameter.prototype.toJSON = function toJSON() {
                return this.constructor.toObject(this, $protobuf.util.toJSONOptions);
            };
    
            SoftmaxParameter.Engine = (function() {
                var valuesById = {}, values = Object.create(valuesById);
                values[valuesById[0] = "DEFAULT"] = 0;
                values[valuesById[1] = "CAFFE"] = 1;
                values[valuesById[2] = "CUDNN"] = 2;
                return values;
            })();
    
            return SoftmaxParameter;
        })();
    
        caffe.SwishParameter = (function() {
    
            function SwishParameter(properties) {
                if (properties)
                    for (var keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                        if (properties[keys[i]] != null)
                            this[keys[i]] = properties[keys[i]];
            }
    
            SwishParameter.prototype.beta = 1;
    
            SwishParameter.create = function create(properties) {
                return new SwishParameter(properties);
            };
    
            SwishParameter.decode = function decode(reader, length) {
                if (!(reader instanceof $Reader))
                    reader = $Reader.create(reader);
                var end = length === undefined ? reader.len : reader.pos + length, message = new $root.caffe.SwishParameter();
                while (reader.pos < end) {
                    var tag = reader.uint32();
                    switch (tag >>> 3) {
                    case 1:
                        message.beta = reader.float();
                        break;
                    default:
                        reader.skipType(tag & 7);
                        break;
                    }
                }
                return message;
            };
    
            SwishParameter.decodeText = function decodeText(reader, block) {
                if (!(reader instanceof $TextReader))
                    reader = $TextReader.create(reader);
                var message = new $root.caffe.SwishParameter();
                reader.start(block);
                while (!reader.end(block)) {
                    var tag = reader.tag();
                    switch (tag) {
                    case "beta":
                        message.beta = reader.float();
                        break;
                    default:
                        reader.handle(tag, message);
                        break;
                    }
                }
                return message;
            };
    
            SwishParameter.verify = function verify(message) {
                if (typeof message !== "object" || message === null)
                    return "object expected";
                if (message.beta != null && message.hasOwnProperty("beta"))
                    if (typeof message.beta !== "number")
                        return "beta: number expected";
                return null;
            };
    
            SwishParameter.fromObject = function fromObject(object) {
                if (object instanceof $root.caffe.SwishParameter)
                    return object;
                var message = new $root.caffe.SwishParameter();
                if (object.beta != null)
                    message.beta = Number(object.beta);
                return message;
            };
    
            SwishParameter.toObject = function toObject(message, options) {
                if (!options)
                    options = {};
                var object = {};
                if (options.defaults)
                    object.beta = 1;
                if (message.beta != null && message.hasOwnProperty("beta"))
                    object.beta = options.json && !isFinite(message.beta) ? String(message.beta) : message.beta;
                return object;
            };
    
            SwishParameter.prototype.toJSON = function toJSON() {
                return this.constructor.toObject(this, $protobuf.util.toJSONOptions);
            };
    
            return SwishParameter;
        })();
    
        caffe.TanHParameter = (function() {
    
            function TanHParameter(properties) {
                if (properties)
                    for (var keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                        if (properties[keys[i]] != null)
                            this[keys[i]] = properties[keys[i]];
            }
    
            TanHParameter.prototype.engine = 0;
    
            TanHParameter.create = function create(properties) {
                return new TanHParameter(properties);
            };
    
            TanHParameter.decode = function decode(reader, length) {
                if (!(reader instanceof $Reader))
                    reader = $Reader.create(reader);
                var end = length === undefined ? reader.len : reader.pos + length, message = new $root.caffe.TanHParameter();
                while (reader.pos < end) {
                    var tag = reader.uint32();
                    switch (tag >>> 3) {
                    case 1:
                        message.engine = reader.int32();
                        break;
                    default:
                        reader.skipType(tag & 7);
                        break;
                    }
                }
                return message;
            };
    
            TanHParameter.decodeText = function decodeText(reader, block) {
                if (!(reader instanceof $TextReader))
                    reader = $TextReader.create(reader);
                var message = new $root.caffe.TanHParameter();
                reader.start(block);
                while (!reader.end(block)) {
                    var tag = reader.tag();
                    switch (tag) {
                    case "engine":
                        message.engine = reader.enum($root.caffe.TanHParameter.Engine);
                        break;
                    default:
                        reader.handle(tag, message);
                        break;
                    }
                }
                return message;
            };
    
            TanHParameter.verify = function verify(message) {
                if (typeof message !== "object" || message === null)
                    return "object expected";
                if (message.engine != null && message.hasOwnProperty("engine"))
                    switch (message.engine) {
                    default:
                        return "engine: enum value expected";
                    case 0:
                    case 1:
                    case 2:
                        break;
                    }
                return null;
            };
    
            TanHParameter.fromObject = function fromObject(object) {
                if (object instanceof $root.caffe.TanHParameter)
                    return object;
                var message = new $root.caffe.TanHParameter();
                switch (object.engine) {
                case "DEFAULT":
                case 0:
                    message.engine = 0;
                    break;
                case "CAFFE":
                case 1:
                    message.engine = 1;
                    break;
                case "CUDNN":
                case 2:
                    message.engine = 2;
                    break;
                }
                return message;
            };
    
            TanHParameter.toObject = function toObject(message, options) {
                if (!options)
                    options = {};
                var object = {};
                if (options.defaults)
                    object.engine = options.enums === String ? "DEFAULT" : 0;
                if (message.engine != null && message.hasOwnProperty("engine"))
                    object.engine = options.enums === String ? $root.caffe.TanHParameter.Engine[message.engine] : message.engine;
                return object;
            };
    
            TanHParameter.prototype.toJSON = function toJSON() {
                return this.constructor.toObject(this, $protobuf.util.toJSONOptions);
            };
    
            TanHParameter.Engine = (function() {
                var valuesById = {}, values = Object.create(valuesById);
                values[valuesById[0] = "DEFAULT"] = 0;
                values[valuesById[1] = "CAFFE"] = 1;
                values[valuesById[2] = "CUDNN"] = 2;
                return values;
            })();
    
            return TanHParameter;
        })();
    
        caffe.TileParameter = (function() {
    
            function TileParameter(properties) {
                if (properties)
                    for (var keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                        if (properties[keys[i]] != null)
                            this[keys[i]] = properties[keys[i]];
            }
    
            TileParameter.prototype.axis = 1;
            TileParameter.prototype.tiles = 0;
    
            TileParameter.create = function create(properties) {
                return new TileParameter(properties);
            };
    
            TileParameter.decode = function decode(reader, length) {
                if (!(reader instanceof $Reader))
                    reader = $Reader.create(reader);
                var end = length === undefined ? reader.len : reader.pos + length, message = new $root.caffe.TileParameter();
                while (reader.pos < end) {
                    var tag = reader.uint32();
                    switch (tag >>> 3) {
                    case 1:
                        message.axis = reader.int32();
                        break;
                    case 2:
                        message.tiles = reader.int32();
                        break;
                    default:
                        reader.skipType(tag & 7);
                        break;
                    }
                }
                return message;
            };
    
            TileParameter.decodeText = function decodeText(reader, block) {
                if (!(reader instanceof $TextReader))
                    reader = $TextReader.create(reader);
                var message = new $root.caffe.TileParameter();
                reader.start(block);
                while (!reader.end(block)) {
                    var tag = reader.tag();
                    switch (tag) {
                    case "axis":
                        message.axis = reader.int32();
                        break;
                    case "tiles":
                        message.tiles = reader.int32();
                        break;
                    default:
                        reader.handle(tag, message);
                        break;
                    }
                }
                return message;
            };
    
            TileParameter.verify = function verify(message) {
                if (typeof message !== "object" || message === null)
                    return "object expected";
                if (message.axis != null && message.hasOwnProperty("axis"))
                    if (!$util.isInteger(message.axis))
                        return "axis: integer expected";
                if (message.tiles != null && message.hasOwnProperty("tiles"))
                    if (!$util.isInteger(message.tiles))
                        return "tiles: integer expected";
                return null;
            };
    
            TileParameter.fromObject = function fromObject(object) {
                if (object instanceof $root.caffe.TileParameter)
                    return object;
                var message = new $root.caffe.TileParameter();
                if (object.axis != null)
                    message.axis = object.axis | 0;
                if (object.tiles != null)
                    message.tiles = object.tiles | 0;
                return message;
            };
    
            TileParameter.toObject = function toObject(message, options) {
                if (!options)
                    options = {};
                var object = {};
                if (options.defaults) {
                    object.axis = 1;
                    object.tiles = 0;
                }
                if (message.axis != null && message.hasOwnProperty("axis"))
                    object.axis = message.axis;
                if (message.tiles != null && message.hasOwnProperty("tiles"))
                    object.tiles = message.tiles;
                return object;
            };
    
            TileParameter.prototype.toJSON = function toJSON() {
                return this.constructor.toObject(this, $protobuf.util.toJSONOptions);
            };
    
            return TileParameter;
        })();
    
        caffe.ThresholdParameter = (function() {
    
            function ThresholdParameter(properties) {
                if (properties)
                    for (var keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                        if (properties[keys[i]] != null)
                            this[keys[i]] = properties[keys[i]];
            }
    
            ThresholdParameter.prototype.threshold = 0;
    
            ThresholdParameter.create = function create(properties) {
                return new ThresholdParameter(properties);
            };
    
            ThresholdParameter.decode = function decode(reader, length) {
                if (!(reader instanceof $Reader))
                    reader = $Reader.create(reader);
                var end = length === undefined ? reader.len : reader.pos + length, message = new $root.caffe.ThresholdParameter();
                while (reader.pos < end) {
                    var tag = reader.uint32();
                    switch (tag >>> 3) {
                    case 1:
                        message.threshold = reader.float();
                        break;
                    default:
                        reader.skipType(tag & 7);
                        break;
                    }
                }
                return message;
            };
    
            ThresholdParameter.decodeText = function decodeText(reader, block) {
                if (!(reader instanceof $TextReader))
                    reader = $TextReader.create(reader);
                var message = new $root.caffe.ThresholdParameter();
                reader.start(block);
                while (!reader.end(block)) {
                    var tag = reader.tag();
                    switch (tag) {
                    case "threshold":
                        message.threshold = reader.float();
                        break;
                    default:
                        reader.handle(tag, message);
                        break;
                    }
                }
                return message;
            };
    
            ThresholdParameter.verify = function verify(message) {
                if (typeof message !== "object" || message === null)
                    return "object expected";
                if (message.threshold != null && message.hasOwnProperty("threshold"))
                    if (typeof message.threshold !== "number")
                        return "threshold: number expected";
                return null;
            };
    
            ThresholdParameter.fromObject = function fromObject(object) {
                if (object instanceof $root.caffe.ThresholdParameter)
                    return object;
                var message = new $root.caffe.ThresholdParameter();
                if (object.threshold != null)
                    message.threshold = Number(object.threshold);
                return message;
            };
    
            ThresholdParameter.toObject = function toObject(message, options) {
                if (!options)
                    options = {};
                var object = {};
                if (options.defaults)
                    object.threshold = 0;
                if (message.threshold != null && message.hasOwnProperty("threshold"))
                    object.threshold = options.json && !isFinite(message.threshold) ? String(message.threshold) : message.threshold;
                return object;
            };
    
            ThresholdParameter.prototype.toJSON = function toJSON() {
                return this.constructor.toObject(this, $protobuf.util.toJSONOptions);
            };
    
            return ThresholdParameter;
        })();
    
        caffe.WindowDataParameter = (function() {
    
            function WindowDataParameter(properties) {
                if (properties)
                    for (var keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                        if (properties[keys[i]] != null)
                            this[keys[i]] = properties[keys[i]];
            }
    
            WindowDataParameter.prototype.source = "";
            WindowDataParameter.prototype.scale = 1;
            WindowDataParameter.prototype.mean_file = "";
            WindowDataParameter.prototype.batch_size = 0;
            WindowDataParameter.prototype.crop_size = 0;
            WindowDataParameter.prototype.mirror = false;
            WindowDataParameter.prototype.fg_threshold = 0.5;
            WindowDataParameter.prototype.bg_threshold = 0.5;
            WindowDataParameter.prototype.fg_fraction = 0.25;
            WindowDataParameter.prototype.context_pad = 0;
            WindowDataParameter.prototype.crop_mode = "warp";
            WindowDataParameter.prototype.cache_images = false;
            WindowDataParameter.prototype.root_folder = "";
    
            WindowDataParameter.create = function create(properties) {
                return new WindowDataParameter(properties);
            };
    
            WindowDataParameter.decode = function decode(reader, length) {
                if (!(reader instanceof $Reader))
                    reader = $Reader.create(reader);
                var end = length === undefined ? reader.len : reader.pos + length, message = new $root.caffe.WindowDataParameter();
                while (reader.pos < end) {
                    var tag = reader.uint32();
                    switch (tag >>> 3) {
                    case 1:
                        message.source = reader.string();
                        break;
                    case 2:
                        message.scale = reader.float();
                        break;
                    case 3:
                        message.mean_file = reader.string();
                        break;
                    case 4:
                        message.batch_size = reader.uint32();
                        break;
                    case 5:
                        message.crop_size = reader.uint32();
                        break;
                    case 6:
                        message.mirror = reader.bool();
                        break;
                    case 7:
                        message.fg_threshold = reader.float();
                        break;
                    case 8:
                        message.bg_threshold = reader.float();
                        break;
                    case 9:
                        message.fg_fraction = reader.float();
                        break;
                    case 10:
                        message.context_pad = reader.uint32();
                        break;
                    case 11:
                        message.crop_mode = reader.string();
                        break;
                    case 12:
                        message.cache_images = reader.bool();
                        break;
                    case 13:
                        message.root_folder = reader.string();
                        break;
                    default:
                        reader.skipType(tag & 7);
                        break;
                    }
                }
                return message;
            };
    
            WindowDataParameter.decodeText = function decodeText(reader, block) {
                if (!(reader instanceof $TextReader))
                    reader = $TextReader.create(reader);
                var message = new $root.caffe.WindowDataParameter();
                reader.start(block);
                while (!reader.end(block)) {
                    var tag = reader.tag();
                    switch (tag) {
                    case "source":
                        message.source = reader.string();
                        break;
                    case "scale":
                        message.scale = reader.float();
                        break;
                    case "mean_file":
                        message.mean_file = reader.string();
                        break;
                    case "batch_size":
                        message.batch_size = reader.uint32();
                        break;
                    case "crop_size":
                        message.crop_size = reader.uint32();
                        break;
                    case "mirror":
                        message.mirror = reader.bool();
                        break;
                    case "fg_threshold":
                        message.fg_threshold = reader.float();
                        break;
                    case "bg_threshold":
                        message.bg_threshold = reader.float();
                        break;
                    case "fg_fraction":
                        message.fg_fraction = reader.float();
                        break;
                    case "context_pad":
                        message.context_pad = reader.uint32();
                        break;
                    case "crop_mode":
                        message.crop_mode = reader.string();
                        break;
                    case "cache_images":
                        message.cache_images = reader.bool();
                        break;
                    case "root_folder":
                        message.root_folder = reader.string();
                        break;
                    default:
                        reader.handle(tag, message);
                        break;
                    }
                }
                return message;
            };
    
            WindowDataParameter.verify = function verify(message) {
                if (typeof message !== "object" || message === null)
                    return "object expected";
                if (message.source != null && message.hasOwnProperty("source"))
                    if (!$util.isString(message.source))
                        return "source: string expected";
                if (message.scale != null && message.hasOwnProperty("scale"))
                    if (typeof message.scale !== "number")
                        return "scale: number expected";
                if (message.mean_file != null && message.hasOwnProperty("mean_file"))
                    if (!$util.isString(message.mean_file))
                        return "mean_file: string expected";
                if (message.batch_size != null && message.hasOwnProperty("batch_size"))
                    if (!$util.isInteger(message.batch_size))
                        return "batch_size: integer expected";
                if (message.crop_size != null && message.hasOwnProperty("crop_size"))
                    if (!$util.isInteger(message.crop_size))
                        return "crop_size: integer expected";
                if (message.mirror != null && message.hasOwnProperty("mirror"))
                    if (typeof message.mirror !== "boolean")
                        return "mirror: boolean expected";
                if (message.fg_threshold != null && message.hasOwnProperty("fg_threshold"))
                    if (typeof message.fg_threshold !== "number")
                        return "fg_threshold: number expected";
                if (message.bg_threshold != null && message.hasOwnProperty("bg_threshold"))
                    if (typeof message.bg_threshold !== "number")
                        return "bg_threshold: number expected";
                if (message.fg_fraction != null && message.hasOwnProperty("fg_fraction"))
                    if (typeof message.fg_fraction !== "number")
                        return "fg_fraction: number expected";
                if (message.context_pad != null && message.hasOwnProperty("context_pad"))
                    if (!$util.isInteger(message.context_pad))
                        return "context_pad: integer expected";
                if (message.crop_mode != null && message.hasOwnProperty("crop_mode"))
                    if (!$util.isString(message.crop_mode))
                        return "crop_mode: string expected";
                if (message.cache_images != null && message.hasOwnProperty("cache_images"))
                    if (typeof message.cache_images !== "boolean")
                        return "cache_images: boolean expected";
                if (message.root_folder != null && message.hasOwnProperty("root_folder"))
                    if (!$util.isString(message.root_folder))
                        return "root_folder: string expected";
                return null;
            };
    
            WindowDataParameter.fromObject = function fromObject(object) {
                if (object instanceof $root.caffe.WindowDataParameter)
                    return object;
                var message = new $root.caffe.WindowDataParameter();
                if (object.source != null)
                    message.source = String(object.source);
                if (object.scale != null)
                    message.scale = Number(object.scale);
                if (object.mean_file != null)
                    message.mean_file = String(object.mean_file);
                if (object.batch_size != null)
                    message.batch_size = object.batch_size >>> 0;
                if (object.crop_size != null)
                    message.crop_size = object.crop_size >>> 0;
                if (object.mirror != null)
                    message.mirror = Boolean(object.mirror);
                if (object.fg_threshold != null)
                    message.fg_threshold = Number(object.fg_threshold);
                if (object.bg_threshold != null)
                    message.bg_threshold = Number(object.bg_threshold);
                if (object.fg_fraction != null)
                    message.fg_fraction = Number(object.fg_fraction);
                if (object.context_pad != null)
                    message.context_pad = object.context_pad >>> 0;
                if (object.crop_mode != null)
                    message.crop_mode = String(object.crop_mode);
                if (object.cache_images != null)
                    message.cache_images = Boolean(object.cache_images);
                if (object.root_folder != null)
                    message.root_folder = String(object.root_folder);
                return message;
            };
    
            WindowDataParameter.toObject = function toObject(message, options) {
                if (!options)
                    options = {};
                var object = {};
                if (options.defaults) {
                    object.source = "";
                    object.scale = 1;
                    object.mean_file = "";
                    object.batch_size = 0;
                    object.crop_size = 0;
                    object.mirror = false;
                    object.fg_threshold = 0.5;
                    object.bg_threshold = 0.5;
                    object.fg_fraction = 0.25;
                    object.context_pad = 0;
                    object.crop_mode = "warp";
                    object.cache_images = false;
                    object.root_folder = "";
                }
                if (message.source != null && message.hasOwnProperty("source"))
                    object.source = message.source;
                if (message.scale != null && message.hasOwnProperty("scale"))
                    object.scale = options.json && !isFinite(message.scale) ? String(message.scale) : message.scale;
                if (message.mean_file != null && message.hasOwnProperty("mean_file"))
                    object.mean_file = message.mean_file;
                if (message.batch_size != null && message.hasOwnProperty("batch_size"))
                    object.batch_size = message.batch_size;
                if (message.crop_size != null && message.hasOwnProperty("crop_size"))
                    object.crop_size = message.crop_size;
                if (message.mirror != null && message.hasOwnProperty("mirror"))
                    object.mirror = message.mirror;
                if (message.fg_threshold != null && message.hasOwnProperty("fg_threshold"))
                    object.fg_threshold = options.json && !isFinite(message.fg_threshold) ? String(message.fg_threshold) : message.fg_threshold;
                if (message.bg_threshold != null && message.hasOwnProperty("bg_threshold"))
                    object.bg_threshold = options.json && !isFinite(message.bg_threshold) ? String(message.bg_threshold) : message.bg_threshold;
                if (message.fg_fraction != null && message.hasOwnProperty("fg_fraction"))
                    object.fg_fraction = options.json && !isFinite(message.fg_fraction) ? String(message.fg_fraction) : message.fg_fraction;
                if (message.context_pad != null && message.hasOwnProperty("context_pad"))
                    object.context_pad = message.context_pad;
                if (message.crop_mode != null && message.hasOwnProperty("crop_mode"))
                    object.crop_mode = message.crop_mode;
                if (message.cache_images != null && message.hasOwnProperty("cache_images"))
                    object.cache_images = message.cache_images;
                if (message.root_folder != null && message.hasOwnProperty("root_folder"))
                    object.root_folder = message.root_folder;
                return object;
            };
    
            WindowDataParameter.prototype.toJSON = function toJSON() {
                return this.constructor.toObject(this, $protobuf.util.toJSONOptions);
            };
    
            return WindowDataParameter;
        })();
    
        caffe.SPPParameter = (function() {
    
            function SPPParameter(properties) {
                if (properties)
                    for (var keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                        if (properties[keys[i]] != null)
                            this[keys[i]] = properties[keys[i]];
            }
    
            SPPParameter.prototype.pyramid_height = 0;
            SPPParameter.prototype.pool = 0;
            SPPParameter.prototype.engine = 0;
    
            SPPParameter.create = function create(properties) {
                return new SPPParameter(properties);
            };
    
            SPPParameter.decode = function decode(reader, length) {
                if (!(reader instanceof $Reader))
                    reader = $Reader.create(reader);
                var end = length === undefined ? reader.len : reader.pos + length, message = new $root.caffe.SPPParameter();
                while (reader.pos < end) {
                    var tag = reader.uint32();
                    switch (tag >>> 3) {
                    case 1:
                        message.pyramid_height = reader.uint32();
                        break;
                    case 2:
                        message.pool = reader.int32();
                        break;
                    case 6:
                        message.engine = reader.int32();
                        break;
                    default:
                        reader.skipType(tag & 7);
                        break;
                    }
                }
                return message;
            };
    
            SPPParameter.decodeText = function decodeText(reader, block) {
                if (!(reader instanceof $TextReader))
                    reader = $TextReader.create(reader);
                var message = new $root.caffe.SPPParameter();
                reader.start(block);
                while (!reader.end(block)) {
                    var tag = reader.tag();
                    switch (tag) {
                    case "pyramid_height":
                        message.pyramid_height = reader.uint32();
                        break;
                    case "pool":
                        message.pool = reader.enum($root.caffe.SPPParameter.PoolMethod);
                        break;
                    case "engine":
                        message.engine = reader.enum($root.caffe.SPPParameter.Engine);
                        break;
                    default:
                        reader.handle(tag, message);
                        break;
                    }
                }
                return message;
            };
    
            SPPParameter.verify = function verify(message) {
                if (typeof message !== "object" || message === null)
                    return "object expected";
                if (message.pyramid_height != null && message.hasOwnProperty("pyramid_height"))
                    if (!$util.isInteger(message.pyramid_height))
                        return "pyramid_height: integer expected";
                if (message.pool != null && message.hasOwnProperty("pool"))
                    switch (message.pool) {
                    default:
                        return "pool: enum value expected";
                    case 0:
                    case 1:
                    case 2:
                        break;
                    }
                if (message.engine != null && message.hasOwnProperty("engine"))
                    switch (message.engine) {
                    default:
                        return "engine: enum value expected";
                    case 0:
                    case 1:
                    case 2:
                        break;
                    }
                return null;
            };
    
            SPPParameter.fromObject = function fromObject(object) {
                if (object instanceof $root.caffe.SPPParameter)
                    return object;
                var message = new $root.caffe.SPPParameter();
                if (object.pyramid_height != null)
                    message.pyramid_height = object.pyramid_height >>> 0;
                switch (object.pool) {
                case "MAX":
                case 0:
                    message.pool = 0;
                    break;
                case "AVE":
                case 1:
                    message.pool = 1;
                    break;
                case "STOCHASTIC":
                case 2:
                    message.pool = 2;
                    break;
                }
                switch (object.engine) {
                case "DEFAULT":
                case 0:
                    message.engine = 0;
                    break;
                case "CAFFE":
                case 1:
                    message.engine = 1;
                    break;
                case "CUDNN":
                case 2:
                    message.engine = 2;
                    break;
                }
                return message;
            };
    
            SPPParameter.toObject = function toObject(message, options) {
                if (!options)
                    options = {};
                var object = {};
                if (options.defaults) {
                    object.pyramid_height = 0;
                    object.pool = options.enums === String ? "MAX" : 0;
                    object.engine = options.enums === String ? "DEFAULT" : 0;
                }
                if (message.pyramid_height != null && message.hasOwnProperty("pyramid_height"))
                    object.pyramid_height = message.pyramid_height;
                if (message.pool != null && message.hasOwnProperty("pool"))
                    object.pool = options.enums === String ? $root.caffe.SPPParameter.PoolMethod[message.pool] : message.pool;
                if (message.engine != null && message.hasOwnProperty("engine"))
                    object.engine = options.enums === String ? $root.caffe.SPPParameter.Engine[message.engine] : message.engine;
                return object;
            };
    
            SPPParameter.prototype.toJSON = function toJSON() {
                return this.constructor.toObject(this, $protobuf.util.toJSONOptions);
            };
    
            SPPParameter.PoolMethod = (function() {
                var valuesById = {}, values = Object.create(valuesById);
                values[valuesById[0] = "MAX"] = 0;
                values[valuesById[1] = "AVE"] = 1;
                values[valuesById[2] = "STOCHASTIC"] = 2;
                return values;
            })();
    
            SPPParameter.Engine = (function() {
                var valuesById = {}, values = Object.create(valuesById);
                values[valuesById[0] = "DEFAULT"] = 0;
                values[valuesById[1] = "CAFFE"] = 1;
                values[valuesById[2] = "CUDNN"] = 2;
                return values;
            })();
    
            return SPPParameter;
        })();
    
        caffe.V1LayerParameter = (function() {
    
            function V1LayerParameter(properties) {
                this.bottom = [];
                this.top = [];
                this.include = [];
                this.exclude = [];
                this.blobs = [];
                this.param = [];
                this.blob_share_mode = [];
                this.blobs_lr = [];
                this.weight_decay = [];
                this.loss_weight = [];
                if (properties)
                    for (var keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                        if (properties[keys[i]] != null)
                            this[keys[i]] = properties[keys[i]];
            }
    
            V1LayerParameter.prototype.bottom = $util.emptyArray;
            V1LayerParameter.prototype.top = $util.emptyArray;
            V1LayerParameter.prototype.name = "";
            V1LayerParameter.prototype.include = $util.emptyArray;
            V1LayerParameter.prototype.exclude = $util.emptyArray;
            V1LayerParameter.prototype.type = 0;
            V1LayerParameter.prototype.blobs = $util.emptyArray;
            V1LayerParameter.prototype.param = $util.emptyArray;
            V1LayerParameter.prototype.blob_share_mode = $util.emptyArray;
            V1LayerParameter.prototype.blobs_lr = $util.emptyArray;
            V1LayerParameter.prototype.weight_decay = $util.emptyArray;
            V1LayerParameter.prototype.loss_weight = $util.emptyArray;
            V1LayerParameter.prototype.accuracy_param = null;
            V1LayerParameter.prototype.argmax_param = null;
            V1LayerParameter.prototype.concat_param = null;
            V1LayerParameter.prototype.contrastive_loss_param = null;
            V1LayerParameter.prototype.convolution_param = null;
            V1LayerParameter.prototype.data_param = null;
            V1LayerParameter.prototype.dropout_param = null;
            V1LayerParameter.prototype.dummy_data_param = null;
            V1LayerParameter.prototype.eltwise_param = null;
            V1LayerParameter.prototype.exp_param = null;
            V1LayerParameter.prototype.hdf5_data_param = null;
            V1LayerParameter.prototype.hdf5_output_param = null;
            V1LayerParameter.prototype.hinge_loss_param = null;
            V1LayerParameter.prototype.image_data_param = null;
            V1LayerParameter.prototype.infogain_loss_param = null;
            V1LayerParameter.prototype.inner_product_param = null;
            V1LayerParameter.prototype.lrn_param = null;
            V1LayerParameter.prototype.memory_data_param = null;
            V1LayerParameter.prototype.mvn_param = null;
            V1LayerParameter.prototype.pooling_param = null;
            V1LayerParameter.prototype.power_param = null;
            V1LayerParameter.prototype.relu_param = null;
            V1LayerParameter.prototype.sigmoid_param = null;
            V1LayerParameter.prototype.softmax_param = null;
            V1LayerParameter.prototype.slice_param = null;
            V1LayerParameter.prototype.tanh_param = null;
            V1LayerParameter.prototype.threshold_param = null;
            V1LayerParameter.prototype.window_data_param = null;
            V1LayerParameter.prototype.transform_param = null;
            V1LayerParameter.prototype.loss_param = null;
            V1LayerParameter.prototype.layer = null;
    
            V1LayerParameter.create = function create(properties) {
                return new V1LayerParameter(properties);
            };
    
            V1LayerParameter.decode = function decode(reader, length) {
                if (!(reader instanceof $Reader))
                    reader = $Reader.create(reader);
                var end = length === undefined ? reader.len : reader.pos + length, message = new $root.caffe.V1LayerParameter();
                while (reader.pos < end) {
                    var tag = reader.uint32();
                    switch (tag >>> 3) {
                    case 2:
                        if (!(message.bottom && message.bottom.length))
                            message.bottom = [];
                        message.bottom.push(reader.string());
                        break;
                    case 3:
                        if (!(message.top && message.top.length))
                            message.top = [];
                        message.top.push(reader.string());
                        break;
                    case 4:
                        message.name = reader.string();
                        break;
                    case 32:
                        if (!(message.include && message.include.length))
                            message.include = [];
                        message.include.push($root.caffe.NetStateRule.decode(reader, reader.uint32()));
                        break;
                    case 33:
                        if (!(message.exclude && message.exclude.length))
                            message.exclude = [];
                        message.exclude.push($root.caffe.NetStateRule.decode(reader, reader.uint32()));
                        break;
                    case 5:
                        message.type = reader.int32();
                        break;
                    case 6:
                        if (!(message.blobs && message.blobs.length))
                            message.blobs = [];
                        message.blobs.push($root.caffe.BlobProto.decode(reader, reader.uint32()));
                        break;
                    case 1001:
                        if (!(message.param && message.param.length))
                            message.param = [];
                        message.param.push(reader.string());
                        break;
                    case 1002:
                        if (!(message.blob_share_mode && message.blob_share_mode.length))
                            message.blob_share_mode = [];
                        if ((tag & 7) === 2) {
                            var end2 = reader.uint32() + reader.pos;
                            while (reader.pos < end2)
                                message.blob_share_mode.push(reader.int32());
                        } else
                            message.blob_share_mode.push(reader.int32());
                        break;
                    case 7:
                        if (!(message.blobs_lr && message.blobs_lr.length))
                            message.blobs_lr = [];
                        if ((tag & 7) === 2) {
                            var end2 = reader.uint32() + reader.pos;
                            while (reader.pos < end2)
                                message.blobs_lr.push(reader.float());
                        } else
                            message.blobs_lr.push(reader.float());
                        break;
                    case 8:
                        if (!(message.weight_decay && message.weight_decay.length))
                            message.weight_decay = [];
                        if ((tag & 7) === 2) {
                            var end2 = reader.uint32() + reader.pos;
                            while (reader.pos < end2)
                                message.weight_decay.push(reader.float());
                        } else
                            message.weight_decay.push(reader.float());
                        break;
                    case 35:
                        if (!(message.loss_weight && message.loss_weight.length))
                            message.loss_weight = [];
                        if ((tag & 7) === 2) {
                            var end2 = reader.uint32() + reader.pos;
                            while (reader.pos < end2)
                                message.loss_weight.push(reader.float());
                        } else
                            message.loss_weight.push(reader.float());
                        break;
                    case 27:
                        message.accuracy_param = $root.caffe.AccuracyParameter.decode(reader, reader.uint32());
                        break;
                    case 23:
                        message.argmax_param = $root.caffe.ArgMaxParameter.decode(reader, reader.uint32());
                        break;
                    case 9:
                        message.concat_param = $root.caffe.ConcatParameter.decode(reader, reader.uint32());
                        break;
                    case 40:
                        message.contrastive_loss_param = $root.caffe.ContrastiveLossParameter.decode(reader, reader.uint32());
                        break;
                    case 10:
                        message.convolution_param = $root.caffe.ConvolutionParameter.decode(reader, reader.uint32());
                        break;
                    case 11:
                        message.data_param = $root.caffe.DataParameter.decode(reader, reader.uint32());
                        break;
                    case 12:
                        message.dropout_param = $root.caffe.DropoutParameter.decode(reader, reader.uint32());
                        break;
                    case 26:
                        message.dummy_data_param = $root.caffe.DummyDataParameter.decode(reader, reader.uint32());
                        break;
                    case 24:
                        message.eltwise_param = $root.caffe.EltwiseParameter.decode(reader, reader.uint32());
                        break;
                    case 41:
                        message.exp_param = $root.caffe.ExpParameter.decode(reader, reader.uint32());
                        break;
                    case 13:
                        message.hdf5_data_param = $root.caffe.HDF5DataParameter.decode(reader, reader.uint32());
                        break;
                    case 14:
                        message.hdf5_output_param = $root.caffe.HDF5OutputParameter.decode(reader, reader.uint32());
                        break;
                    case 29:
                        message.hinge_loss_param = $root.caffe.HingeLossParameter.decode(reader, reader.uint32());
                        break;
                    case 15:
                        message.image_data_param = $root.caffe.ImageDataParameter.decode(reader, reader.uint32());
                        break;
                    case 16:
                        message.infogain_loss_param = $root.caffe.InfogainLossParameter.decode(reader, reader.uint32());
                        break;
                    case 17:
                        message.inner_product_param = $root.caffe.InnerProductParameter.decode(reader, reader.uint32());
                        break;
                    case 18:
                        message.lrn_param = $root.caffe.LRNParameter.decode(reader, reader.uint32());
                        break;
                    case 22:
                        message.memory_data_param = $root.caffe.MemoryDataParameter.decode(reader, reader.uint32());
                        break;
                    case 34:
                        message.mvn_param = $root.caffe.MVNParameter.decode(reader, reader.uint32());
                        break;
                    case 19:
                        message.pooling_param = $root.caffe.PoolingParameter.decode(reader, reader.uint32());
                        break;
                    case 21:
                        message.power_param = $root.caffe.PowerParameter.decode(reader, reader.uint32());
                        break;
                    case 30:
                        message.relu_param = $root.caffe.ReLUParameter.decode(reader, reader.uint32());
                        break;
                    case 38:
                        message.sigmoid_param = $root.caffe.SigmoidParameter.decode(reader, reader.uint32());
                        break;
                    case 39:
                        message.softmax_param = $root.caffe.SoftmaxParameter.decode(reader, reader.uint32());
                        break;
                    case 31:
                        message.slice_param = $root.caffe.SliceParameter.decode(reader, reader.uint32());
                        break;
                    case 37:
                        message.tanh_param = $root.caffe.TanHParameter.decode(reader, reader.uint32());
                        break;
                    case 25:
                        message.threshold_param = $root.caffe.ThresholdParameter.decode(reader, reader.uint32());
                        break;
                    case 20:
                        message.window_data_param = $root.caffe.WindowDataParameter.decode(reader, reader.uint32());
                        break;
                    case 36:
                        message.transform_param = $root.caffe.TransformationParameter.decode(reader, reader.uint32());
                        break;
                    case 42:
                        message.loss_param = $root.caffe.LossParameter.decode(reader, reader.uint32());
                        break;
                    case 1:
                        message.layer = $root.caffe.V0LayerParameter.decode(reader, reader.uint32());
                        break;
                    default:
                        reader.skipType(tag & 7);
                        break;
                    }
                }
                return message;
            };
    
            V1LayerParameter.decodeText = function decodeText(reader, block) {
                if (!(reader instanceof $TextReader))
                    reader = $TextReader.create(reader);
                var message = new $root.caffe.V1LayerParameter();
                reader.start(block);
                while (!reader.end(block)) {
                    var tag = reader.tag();
                    switch (tag) {
                    case "bottom":
                        if (!(message.bottom && message.bottom.length))
                            message.bottom = [];
                        message.bottom.push(reader.string());
                        break;
                    case "top":
                        if (!(message.top && message.top.length))
                            message.top = [];
                        message.top.push(reader.string());
                        break;
                    case "name":
                        message.name = reader.string();
                        break;
                    case "include":
                        if (!(message.include && message.include.length))
                            message.include = [];
                        message.include.push($root.caffe.NetStateRule.decodeText(reader, true));
                        break;
                    case "exclude":
                        if (!(message.exclude && message.exclude.length))
                            message.exclude = [];
                        message.exclude.push($root.caffe.NetStateRule.decodeText(reader, true));
                        break;
                    case "type":
                        message.type = reader.enum($root.caffe.V1LayerParameter.LayerType);
                        break;
                    case "blobs":
                        if (!(message.blobs && message.blobs.length))
                            message.blobs = [];
                        message.blobs.push($root.caffe.BlobProto.decodeText(reader, true));
                        break;
                    case "param":
                        if (!(message.param && message.param.length))
                            message.param = [];
                        message.param.push(reader.string());
                        break;
                    case "blob_share_mode":
                        if (!(message.blob_share_mode && message.blob_share_mode.length))
                            message.blob_share_mode = [];
                        message.blob_share_mode.push(reader.enum($root.caffe.V1LayerParameter.DimCheckMode));
                        break;
                    case "blobs_lr":
                        if (!(message.blobs_lr && message.blobs_lr.length))
                            message.blobs_lr = [];
                        message.blobs_lr.push(reader.float());
                        break;
                    case "weight_decay":
                        if (!(message.weight_decay && message.weight_decay.length))
                            message.weight_decay = [];
                        message.weight_decay.push(reader.float());
                        break;
                    case "loss_weight":
                        if (!(message.loss_weight && message.loss_weight.length))
                            message.loss_weight = [];
                        message.loss_weight.push(reader.float());
                        break;
                    case "accuracy_param":
                        message.accuracy_param = $root.caffe.AccuracyParameter.decodeText(reader, true);
                        break;
                    case "argmax_param":
                        message.argmax_param = $root.caffe.ArgMaxParameter.decodeText(reader, true);
                        break;
                    case "concat_param":
                        message.concat_param = $root.caffe.ConcatParameter.decodeText(reader, true);
                        break;
                    case "contrastive_loss_param":
                        message.contrastive_loss_param = $root.caffe.ContrastiveLossParameter.decodeText(reader, true);
                        break;
                    case "convolution_param":
                        message.convolution_param = $root.caffe.ConvolutionParameter.decodeText(reader, true);
                        break;
                    case "data_param":
                        message.data_param = $root.caffe.DataParameter.decodeText(reader, true);
                        break;
                    case "dropout_param":
                        message.dropout_param = $root.caffe.DropoutParameter.decodeText(reader, true);
                        break;
                    case "dummy_data_param":
                        message.dummy_data_param = $root.caffe.DummyDataParameter.decodeText(reader, true);
                        break;
                    case "eltwise_param":
                        message.eltwise_param = $root.caffe.EltwiseParameter.decodeText(reader, true);
                        break;
                    case "exp_param":
                        message.exp_param = $root.caffe.ExpParameter.decodeText(reader, true);
                        break;
                    case "hdf5_data_param":
                        message.hdf5_data_param = $root.caffe.HDF5DataParameter.decodeText(reader, true);
                        break;
                    case "hdf5_output_param":
                        message.hdf5_output_param = $root.caffe.HDF5OutputParameter.decodeText(reader, true);
                        break;
                    case "hinge_loss_param":
                        message.hinge_loss_param = $root.caffe.HingeLossParameter.decodeText(reader, true);
                        break;
                    case "image_data_param":
                        message.image_data_param = $root.caffe.ImageDataParameter.decodeText(reader, true);
                        break;
                    case "infogain_loss_param":
                        message.infogain_loss_param = $root.caffe.InfogainLossParameter.decodeText(reader, true);
                        break;
                    case "inner_product_param":
                        message.inner_product_param = $root.caffe.InnerProductParameter.decodeText(reader, true);
                        break;
                    case "lrn_param":
                        message.lrn_param = $root.caffe.LRNParameter.decodeText(reader, true);
                        break;
                    case "memory_data_param":
                        message.memory_data_param = $root.caffe.MemoryDataParameter.decodeText(reader, true);
                        break;
                    case "mvn_param":
                        message.mvn_param = $root.caffe.MVNParameter.decodeText(reader, true);
                        break;
                    case "pooling_param":
                        message.pooling_param = $root.caffe.PoolingParameter.decodeText(reader, true);
                        break;
                    case "power_param":
                        message.power_param = $root.caffe.PowerParameter.decodeText(reader, true);
                        break;
                    case "relu_param":
                        message.relu_param = $root.caffe.ReLUParameter.decodeText(reader, true);
                        break;
                    case "sigmoid_param":
                        message.sigmoid_param = $root.caffe.SigmoidParameter.decodeText(reader, true);
                        break;
                    case "softmax_param":
                        message.softmax_param = $root.caffe.SoftmaxParameter.decodeText(reader, true);
                        break;
                    case "slice_param":
                        message.slice_param = $root.caffe.SliceParameter.decodeText(reader, true);
                        break;
                    case "tanh_param":
                        message.tanh_param = $root.caffe.TanHParameter.decodeText(reader, true);
                        break;
                    case "threshold_param":
                        message.threshold_param = $root.caffe.ThresholdParameter.decodeText(reader, true);
                        break;
                    case "window_data_param":
                        message.window_data_param = $root.caffe.WindowDataParameter.decodeText(reader, true);
                        break;
                    case "transform_param":
                        message.transform_param = $root.caffe.TransformationParameter.decodeText(reader, true);
                        break;
                    case "loss_param":
                        message.loss_param = $root.caffe.LossParameter.decodeText(reader, true);
                        break;
                    case "layer":
                        message.layer = $root.caffe.V0LayerParameter.decodeText(reader, true);
                        break;
                    default:
                        reader.handle(tag, message);
                        break;
                    }
                }
                return message;
            };
    
            V1LayerParameter.verify = function verify(message) {
                if (typeof message !== "object" || message === null)
                    return "object expected";
                if (message.bottom != null && message.hasOwnProperty("bottom")) {
                    if (!Array.isArray(message.bottom))
                        return "bottom: array expected";
                    for (var i = 0; i < message.bottom.length; ++i)
                        if (!$util.isString(message.bottom[i]))
                            return "bottom: string[] expected";
                }
                if (message.top != null && message.hasOwnProperty("top")) {
                    if (!Array.isArray(message.top))
                        return "top: array expected";
                    for (var i = 0; i < message.top.length; ++i)
                        if (!$util.isString(message.top[i]))
                            return "top: string[] expected";
                }
                if (message.name != null && message.hasOwnProperty("name"))
                    if (!$util.isString(message.name))
                        return "name: string expected";
                if (message.include != null && message.hasOwnProperty("include")) {
                    if (!Array.isArray(message.include))
                        return "include: array expected";
                    for (var i = 0; i < message.include.length; ++i) {
                        var error = $root.caffe.NetStateRule.verify(message.include[i]);
                        if (error)
                            return "include." + error;
                    }
                }
                if (message.exclude != null && message.hasOwnProperty("exclude")) {
                    if (!Array.isArray(message.exclude))
                        return "exclude: array expected";
                    for (var i = 0; i < message.exclude.length; ++i) {
                        var error = $root.caffe.NetStateRule.verify(message.exclude[i]);
                        if (error)
                            return "exclude." + error;
                    }
                }
                if (message.type != null && message.hasOwnProperty("type"))
                    switch (message.type) {
                    default:
                        return "type: enum value expected";
                    case 0:
                    case 35:
                    case 1:
                    case 30:
                    case 2:
                    case 3:
                    case 37:
                    case 4:
                    case 5:
                    case 39:
                    case 6:
                    case 32:
                    case 7:
                    case 25:
                    case 38:
                    case 8:
                    case 9:
                    case 10:
                    case 28:
                    case 11:
                    case 12:
                    case 13:
                    case 14:
                    case 15:
                    case 29:
                    case 16:
                    case 34:
                    case 17:
                    case 26:
                    case 18:
                    case 19:
                    case 27:
                    case 36:
                    case 20:
                    case 21:
                    case 22:
                    case 33:
                    case 23:
                    case 24:
                    case 31:
                        break;
                    }
                if (message.blobs != null && message.hasOwnProperty("blobs")) {
                    if (!Array.isArray(message.blobs))
                        return "blobs: array expected";
                    for (var i = 0; i < message.blobs.length; ++i) {
                        var error = $root.caffe.BlobProto.verify(message.blobs[i]);
                        if (error)
                            return "blobs." + error;
                    }
                }
                if (message.param != null && message.hasOwnProperty("param")) {
                    if (!Array.isArray(message.param))
                        return "param: array expected";
                    for (var i = 0; i < message.param.length; ++i)
                        if (!$util.isString(message.param[i]))
                            return "param: string[] expected";
                }
                if (message.blob_share_mode != null && message.hasOwnProperty("blob_share_mode")) {
                    if (!Array.isArray(message.blob_share_mode))
                        return "blob_share_mode: array expected";
                    for (var i = 0; i < message.blob_share_mode.length; ++i)
                        switch (message.blob_share_mode[i]) {
                        default:
                            return "blob_share_mode: enum value[] expected";
                        case 0:
                        case 1:
                            break;
                        }
                }
                if (message.blobs_lr != null && message.hasOwnProperty("blobs_lr")) {
                    if (!Array.isArray(message.blobs_lr))
                        return "blobs_lr: array expected";
                    for (var i = 0; i < message.blobs_lr.length; ++i)
                        if (typeof message.blobs_lr[i] !== "number")
                            return "blobs_lr: number[] expected";
                }
                if (message.weight_decay != null && message.hasOwnProperty("weight_decay")) {
                    if (!Array.isArray(message.weight_decay))
                        return "weight_decay: array expected";
                    for (var i = 0; i < message.weight_decay.length; ++i)
                        if (typeof message.weight_decay[i] !== "number")
                            return "weight_decay: number[] expected";
                }
                if (message.loss_weight != null && message.hasOwnProperty("loss_weight")) {
                    if (!Array.isArray(message.loss_weight))
                        return "loss_weight: array expected";
                    for (var i = 0; i < message.loss_weight.length; ++i)
                        if (typeof message.loss_weight[i] !== "number")
                            return "loss_weight: number[] expected";
                }
                if (message.accuracy_param != null && message.hasOwnProperty("accuracy_param")) {
                    var error = $root.caffe.AccuracyParameter.verify(message.accuracy_param);
                    if (error)
                        return "accuracy_param." + error;
                }
                if (message.argmax_param != null && message.hasOwnProperty("argmax_param")) {
                    var error = $root.caffe.ArgMaxParameter.verify(message.argmax_param);
                    if (error)
                        return "argmax_param." + error;
                }
                if (message.concat_param != null && message.hasOwnProperty("concat_param")) {
                    var error = $root.caffe.ConcatParameter.verify(message.concat_param);
                    if (error)
                        return "concat_param." + error;
                }
                if (message.contrastive_loss_param != null && message.hasOwnProperty("contrastive_loss_param")) {
                    var error = $root.caffe.ContrastiveLossParameter.verify(message.contrastive_loss_param);
                    if (error)
                        return "contrastive_loss_param." + error;
                }
                if (message.convolution_param != null && message.hasOwnProperty("convolution_param")) {
                    var error = $root.caffe.ConvolutionParameter.verify(message.convolution_param);
                    if (error)
                        return "convolution_param." + error;
                }
                if (message.data_param != null && message.hasOwnProperty("data_param")) {
                    var error = $root.caffe.DataParameter.verify(message.data_param);
                    if (error)
                        return "data_param." + error;
                }
                if (message.dropout_param != null && message.hasOwnProperty("dropout_param")) {
                    var error = $root.caffe.DropoutParameter.verify(message.dropout_param);
                    if (error)
                        return "dropout_param." + error;
                }
                if (message.dummy_data_param != null && message.hasOwnProperty("dummy_data_param")) {
                    var error = $root.caffe.DummyDataParameter.verify(message.dummy_data_param);
                    if (error)
                        return "dummy_data_param." + error;
                }
                if (message.eltwise_param != null && message.hasOwnProperty("eltwise_param")) {
                    var error = $root.caffe.EltwiseParameter.verify(message.eltwise_param);
                    if (error)
                        return "eltwise_param." + error;
                }
                if (message.exp_param != null && message.hasOwnProperty("exp_param")) {
                    var error = $root.caffe.ExpParameter.verify(message.exp_param);
                    if (error)
                        return "exp_param." + error;
                }
                if (message.hdf5_data_param != null && message.hasOwnProperty("hdf5_data_param")) {
                    var error = $root.caffe.HDF5DataParameter.verify(message.hdf5_data_param);
                    if (error)
                        return "hdf5_data_param." + error;
                }
                if (message.hdf5_output_param != null && message.hasOwnProperty("hdf5_output_param")) {
                    var error = $root.caffe.HDF5OutputParameter.verify(message.hdf5_output_param);
                    if (error)
                        return "hdf5_output_param." + error;
                }
                if (message.hinge_loss_param != null && message.hasOwnProperty("hinge_loss_param")) {
                    var error = $root.caffe.HingeLossParameter.verify(message.hinge_loss_param);
                    if (error)
                        return "hinge_loss_param." + error;
                }
                if (message.image_data_param != null && message.hasOwnProperty("image_data_param")) {
                    var error = $root.caffe.ImageDataParameter.verify(message.image_data_param);
                    if (error)
                        return "image_data_param." + error;
                }
                if (message.infogain_loss_param != null && message.hasOwnProperty("infogain_loss_param")) {
                    var error = $root.caffe.InfogainLossParameter.verify(message.infogain_loss_param);
                    if (error)
                        return "infogain_loss_param." + error;
                }
                if (message.inner_product_param != null && message.hasOwnProperty("inner_product_param")) {
                    var error = $root.caffe.InnerProductParameter.verify(message.inner_product_param);
                    if (error)
                        return "inner_product_param." + error;
                }
                if (message.lrn_param != null && message.hasOwnProperty("lrn_param")) {
                    var error = $root.caffe.LRNParameter.verify(message.lrn_param);
                    if (error)
                        return "lrn_param." + error;
                }
                if (message.memory_data_param != null && message.hasOwnProperty("memory_data_param")) {
                    var error = $root.caffe.MemoryDataParameter.verify(message.memory_data_param);
                    if (error)
                        return "memory_data_param." + error;
                }
                if (message.mvn_param != null && message.hasOwnProperty("mvn_param")) {
                    var error = $root.caffe.MVNParameter.verify(message.mvn_param);
                    if (error)
                        return "mvn_param." + error;
                }
                if (message.pooling_param != null && message.hasOwnProperty("pooling_param")) {
                    var error = $root.caffe.PoolingParameter.verify(message.pooling_param);
                    if (error)
                        return "pooling_param." + error;
                }
                if (message.power_param != null && message.hasOwnProperty("power_param")) {
                    var error = $root.caffe.PowerParameter.verify(message.power_param);
                    if (error)
                        return "power_param." + error;
                }
                if (message.relu_param != null && message.hasOwnProperty("relu_param")) {
                    var error = $root.caffe.ReLUParameter.verify(message.relu_param);
                    if (error)
                        return "relu_param." + error;
                }
                if (message.sigmoid_param != null && message.hasOwnProperty("sigmoid_param")) {
                    var error = $root.caffe.SigmoidParameter.verify(message.sigmoid_param);
                    if (error)
                        return "sigmoid_param." + error;
                }
                if (message.softmax_param != null && message.hasOwnProperty("softmax_param")) {
                    var error = $root.caffe.SoftmaxParameter.verify(message.softmax_param);
                    if (error)
                        return "softmax_param." + error;
                }
                if (message.slice_param != null && message.hasOwnProperty("slice_param")) {
                    var error = $root.caffe.SliceParameter.verify(message.slice_param);
                    if (error)
                        return "slice_param." + error;
                }
                if (message.tanh_param != null && message.hasOwnProperty("tanh_param")) {
                    var error = $root.caffe.TanHParameter.verify(message.tanh_param);
                    if (error)
                        return "tanh_param." + error;
                }
                if (message.threshold_param != null && message.hasOwnProperty("threshold_param")) {
                    var error = $root.caffe.ThresholdParameter.verify(message.threshold_param);
                    if (error)
                        return "threshold_param." + error;
                }
                if (message.window_data_param != null && message.hasOwnProperty("window_data_param")) {
                    var error = $root.caffe.WindowDataParameter.verify(message.window_data_param);
                    if (error)
                        return "window_data_param." + error;
                }
                if (message.transform_param != null && message.hasOwnProperty("transform_param")) {
                    var error = $root.caffe.TransformationParameter.verify(message.transform_param);
                    if (error)
                        return "transform_param." + error;
                }
                if (message.loss_param != null && message.hasOwnProperty("loss_param")) {
                    var error = $root.caffe.LossParameter.verify(message.loss_param);
                    if (error)
                        return "loss_param." + error;
                }
                if (message.layer != null && message.hasOwnProperty("layer")) {
                    var error = $root.caffe.V0LayerParameter.verify(message.layer);
                    if (error)
                        return "layer." + error;
                }
                return null;
            };
    
            V1LayerParameter.fromObject = function fromObject(object) {
                if (object instanceof $root.caffe.V1LayerParameter)
                    return object;
                var message = new $root.caffe.V1LayerParameter();
                if (object.bottom) {
                    if (!Array.isArray(object.bottom))
                        throw TypeError(".caffe.V1LayerParameter.bottom: array expected");
                    message.bottom = [];
                    for (var i = 0; i < object.bottom.length; ++i)
                        message.bottom[i] = String(object.bottom[i]);
                }
                if (object.top) {
                    if (!Array.isArray(object.top))
                        throw TypeError(".caffe.V1LayerParameter.top: array expected");
                    message.top = [];
                    for (var i = 0; i < object.top.length; ++i)
                        message.top[i] = String(object.top[i]);
                }
                if (object.name != null)
                    message.name = String(object.name);
                if (object.include) {
                    if (!Array.isArray(object.include))
                        throw TypeError(".caffe.V1LayerParameter.include: array expected");
                    message.include = [];
                    for (var i = 0; i < object.include.length; ++i) {
                        if (typeof object.include[i] !== "object")
                            throw TypeError(".caffe.V1LayerParameter.include: object expected");
                        message.include[i] = $root.caffe.NetStateRule.fromObject(object.include[i]);
                    }
                }
                if (object.exclude) {
                    if (!Array.isArray(object.exclude))
                        throw TypeError(".caffe.V1LayerParameter.exclude: array expected");
                    message.exclude = [];
                    for (var i = 0; i < object.exclude.length; ++i) {
                        if (typeof object.exclude[i] !== "object")
                            throw TypeError(".caffe.V1LayerParameter.exclude: object expected");
                        message.exclude[i] = $root.caffe.NetStateRule.fromObject(object.exclude[i]);
                    }
                }
                switch (object.type) {
                case "NONE":
                case 0:
                    message.type = 0;
                    break;
                case "ABSVAL":
                case 35:
                    message.type = 35;
                    break;
                case "ACCURACY":
                case 1:
                    message.type = 1;
                    break;
                case "ARGMAX":
                case 30:
                    message.type = 30;
                    break;
                case "BNLL":
                case 2:
                    message.type = 2;
                    break;
                case "CONCAT":
                case 3:
                    message.type = 3;
                    break;
                case "CONTRASTIVE_LOSS":
                case 37:
                    message.type = 37;
                    break;
                case "CONVOLUTION":
                case 4:
                    message.type = 4;
                    break;
                case "DATA":
                case 5:
                    message.type = 5;
                    break;
                case "DECONVOLUTION":
                case 39:
                    message.type = 39;
                    break;
                case "DROPOUT":
                case 6:
                    message.type = 6;
                    break;
                case "DUMMY_DATA":
                case 32:
                    message.type = 32;
                    break;
                case "EUCLIDEAN_LOSS":
                case 7:
                    message.type = 7;
                    break;
                case "ELTWISE":
                case 25:
                    message.type = 25;
                    break;
                case "EXP":
                case 38:
                    message.type = 38;
                    break;
                case "FLATTEN":
                case 8:
                    message.type = 8;
                    break;
                case "HDF5_DATA":
                case 9:
                    message.type = 9;
                    break;
                case "HDF5_OUTPUT":
                case 10:
                    message.type = 10;
                    break;
                case "HINGE_LOSS":
                case 28:
                    message.type = 28;
                    break;
                case "IM2COL":
                case 11:
                    message.type = 11;
                    break;
                case "IMAGE_DATA":
                case 12:
                    message.type = 12;
                    break;
                case "INFOGAIN_LOSS":
                case 13:
                    message.type = 13;
                    break;
                case "INNER_PRODUCT":
                case 14:
                    message.type = 14;
                    break;
                case "LRN":
                case 15:
                    message.type = 15;
                    break;
                case "MEMORY_DATA":
                case 29:
                    message.type = 29;
                    break;
                case "MULTINOMIAL_LOGISTIC_LOSS":
                case 16:
                    message.type = 16;
                    break;
                case "MVN":
                case 34:
                    message.type = 34;
                    break;
                case "POOLING":
                case 17:
                    message.type = 17;
                    break;
                case "POWER":
                case 26:
                    message.type = 26;
                    break;
                case "RELU":
                case 18:
                    message.type = 18;
                    break;
                case "SIGMOID":
                case 19:
                    message.type = 19;
                    break;
                case "SIGMOID_CROSS_ENTROPY_LOSS":
                case 27:
                    message.type = 27;
                    break;
                case "SILENCE":
                case 36:
                    message.type = 36;
                    break;
                case "SOFTMAX":
                case 20:
                    message.type = 20;
                    break;
                case "SOFTMAX_LOSS":
                case 21:
                    message.type = 21;
                    break;
                case "SPLIT":
                case 22:
                    message.type = 22;
                    break;
                case "SLICE":
                case 33:
                    message.type = 33;
                    break;
                case "TANH":
                case 23:
                    message.type = 23;
                    break;
                case "WINDOW_DATA":
                case 24:
                    message.type = 24;
                    break;
                case "THRESHOLD":
                case 31:
                    message.type = 31;
                    break;
                }
                if (object.blobs) {
                    if (!Array.isArray(object.blobs))
                        throw TypeError(".caffe.V1LayerParameter.blobs: array expected");
                    message.blobs = [];
                    for (var i = 0; i < object.blobs.length; ++i) {
                        if (typeof object.blobs[i] !== "object")
                            throw TypeError(".caffe.V1LayerParameter.blobs: object expected");
                        message.blobs[i] = $root.caffe.BlobProto.fromObject(object.blobs[i]);
                    }
                }
                if (object.param) {
                    if (!Array.isArray(object.param))
                        throw TypeError(".caffe.V1LayerParameter.param: array expected");
                    message.param = [];
                    for (var i = 0; i < object.param.length; ++i)
                        message.param[i] = String(object.param[i]);
                }
                if (object.blob_share_mode) {
                    if (!Array.isArray(object.blob_share_mode))
                        throw TypeError(".caffe.V1LayerParameter.blob_share_mode: array expected");
                    message.blob_share_mode = [];
                    for (var i = 0; i < object.blob_share_mode.length; ++i)
                        switch (object.blob_share_mode[i]) {
                        default:
                        case "STRICT":
                        case 0:
                            message.blob_share_mode[i] = 0;
                            break;
                        case "PERMISSIVE":
                        case 1:
                            message.blob_share_mode[i] = 1;
                            break;
                        }
                }
                if (object.blobs_lr) {
                    if (!Array.isArray(object.blobs_lr))
                        throw TypeError(".caffe.V1LayerParameter.blobs_lr: array expected");
                    message.blobs_lr = [];
                    for (var i = 0; i < object.blobs_lr.length; ++i)
                        message.blobs_lr[i] = Number(object.blobs_lr[i]);
                }
                if (object.weight_decay) {
                    if (!Array.isArray(object.weight_decay))
                        throw TypeError(".caffe.V1LayerParameter.weight_decay: array expected");
                    message.weight_decay = [];
                    for (var i = 0; i < object.weight_decay.length; ++i)
                        message.weight_decay[i] = Number(object.weight_decay[i]);
                }
                if (object.loss_weight) {
                    if (!Array.isArray(object.loss_weight))
                        throw TypeError(".caffe.V1LayerParameter.loss_weight: array expected");
                    message.loss_weight = [];
                    for (var i = 0; i < object.loss_weight.length; ++i)
                        message.loss_weight[i] = Number(object.loss_weight[i]);
                }
                if (object.accuracy_param != null) {
                    if (typeof object.accuracy_param !== "object")
                        throw TypeError(".caffe.V1LayerParameter.accuracy_param: object expected");
                    message.accuracy_param = $root.caffe.AccuracyParameter.fromObject(object.accuracy_param);
                }
                if (object.argmax_param != null) {
                    if (typeof object.argmax_param !== "object")
                        throw TypeError(".caffe.V1LayerParameter.argmax_param: object expected");
                    message.argmax_param = $root.caffe.ArgMaxParameter.fromObject(object.argmax_param);
                }
                if (object.concat_param != null) {
                    if (typeof object.concat_param !== "object")
                        throw TypeError(".caffe.V1LayerParameter.concat_param: object expected");
                    message.concat_param = $root.caffe.ConcatParameter.fromObject(object.concat_param);
                }
                if (object.contrastive_loss_param != null) {
                    if (typeof object.contrastive_loss_param !== "object")
                        throw TypeError(".caffe.V1LayerParameter.contrastive_loss_param: object expected");
                    message.contrastive_loss_param = $root.caffe.ContrastiveLossParameter.fromObject(object.contrastive_loss_param);
                }
                if (object.convolution_param != null) {
                    if (typeof object.convolution_param !== "object")
                        throw TypeError(".caffe.V1LayerParameter.convolution_param: object expected");
                    message.convolution_param = $root.caffe.ConvolutionParameter.fromObject(object.convolution_param);
                }
                if (object.data_param != null) {
                    if (typeof object.data_param !== "object")
                        throw TypeError(".caffe.V1LayerParameter.data_param: object expected");
                    message.data_param = $root.caffe.DataParameter.fromObject(object.data_param);
                }
                if (object.dropout_param != null) {
                    if (typeof object.dropout_param !== "object")
                        throw TypeError(".caffe.V1LayerParameter.dropout_param: object expected");
                    message.dropout_param = $root.caffe.DropoutParameter.fromObject(object.dropout_param);
                }
                if (object.dummy_data_param != null) {
                    if (typeof object.dummy_data_param !== "object")
                        throw TypeError(".caffe.V1LayerParameter.dummy_data_param: object expected");
                    message.dummy_data_param = $root.caffe.DummyDataParameter.fromObject(object.dummy_data_param);
                }
                if (object.eltwise_param != null) {
                    if (typeof object.eltwise_param !== "object")
                        throw TypeError(".caffe.V1LayerParameter.eltwise_param: object expected");
                    message.eltwise_param = $root.caffe.EltwiseParameter.fromObject(object.eltwise_param);
                }
                if (object.exp_param != null) {
                    if (typeof object.exp_param !== "object")
                        throw TypeError(".caffe.V1LayerParameter.exp_param: object expected");
                    message.exp_param = $root.caffe.ExpParameter.fromObject(object.exp_param);
                }
                if (object.hdf5_data_param != null) {
                    if (typeof object.hdf5_data_param !== "object")
                        throw TypeError(".caffe.V1LayerParameter.hdf5_data_param: object expected");
                    message.hdf5_data_param = $root.caffe.HDF5DataParameter.fromObject(object.hdf5_data_param);
                }
                if (object.hdf5_output_param != null) {
                    if (typeof object.hdf5_output_param !== "object")
                        throw TypeError(".caffe.V1LayerParameter.hdf5_output_param: object expected");
                    message.hdf5_output_param = $root.caffe.HDF5OutputParameter.fromObject(object.hdf5_output_param);
                }
                if (object.hinge_loss_param != null) {
                    if (typeof object.hinge_loss_param !== "object")
                        throw TypeError(".caffe.V1LayerParameter.hinge_loss_param: object expected");
                    message.hinge_loss_param = $root.caffe.HingeLossParameter.fromObject(object.hinge_loss_param);
                }
                if (object.image_data_param != null) {
                    if (typeof object.image_data_param !== "object")
                        throw TypeError(".caffe.V1LayerParameter.image_data_param: object expected");
                    message.image_data_param = $root.caffe.ImageDataParameter.fromObject(object.image_data_param);
                }
                if (object.infogain_loss_param != null) {
                    if (typeof object.infogain_loss_param !== "object")
                        throw TypeError(".caffe.V1LayerParameter.infogain_loss_param: object expected");
                    message.infogain_loss_param = $root.caffe.InfogainLossParameter.fromObject(object.infogain_loss_param);
                }
                if (object.inner_product_param != null) {
                    if (typeof object.inner_product_param !== "object")
                        throw TypeError(".caffe.V1LayerParameter.inner_product_param: object expected");
                    message.inner_product_param = $root.caffe.InnerProductParameter.fromObject(object.inner_product_param);
                }
                if (object.lrn_param != null) {
                    if (typeof object.lrn_param !== "object")
                        throw TypeError(".caffe.V1LayerParameter.lrn_param: object expected");
                    message.lrn_param = $root.caffe.LRNParameter.fromObject(object.lrn_param);
                }
                if (object.memory_data_param != null) {
                    if (typeof object.memory_data_param !== "object")
                        throw TypeError(".caffe.V1LayerParameter.memory_data_param: object expected");
                    message.memory_data_param = $root.caffe.MemoryDataParameter.fromObject(object.memory_data_param);
                }
                if (object.mvn_param != null) {
                    if (typeof object.mvn_param !== "object")
                        throw TypeError(".caffe.V1LayerParameter.mvn_param: object expected");
                    message.mvn_param = $root.caffe.MVNParameter.fromObject(object.mvn_param);
                }
                if (object.pooling_param != null) {
                    if (typeof object.pooling_param !== "object")
                        throw TypeError(".caffe.V1LayerParameter.pooling_param: object expected");
                    message.pooling_param = $root.caffe.PoolingParameter.fromObject(object.pooling_param);
                }
                if (object.power_param != null) {
                    if (typeof object.power_param !== "object")
                        throw TypeError(".caffe.V1LayerParameter.power_param: object expected");
                    message.power_param = $root.caffe.PowerParameter.fromObject(object.power_param);
                }
                if (object.relu_param != null) {
                    if (typeof object.relu_param !== "object")
                        throw TypeError(".caffe.V1LayerParameter.relu_param: object expected");
                    message.relu_param = $root.caffe.ReLUParameter.fromObject(object.relu_param);
                }
                if (object.sigmoid_param != null) {
                    if (typeof object.sigmoid_param !== "object")
                        throw TypeError(".caffe.V1LayerParameter.sigmoid_param: object expected");
                    message.sigmoid_param = $root.caffe.SigmoidParameter.fromObject(object.sigmoid_param);
                }
                if (object.softmax_param != null) {
                    if (typeof object.softmax_param !== "object")
                        throw TypeError(".caffe.V1LayerParameter.softmax_param: object expected");
                    message.softmax_param = $root.caffe.SoftmaxParameter.fromObject(object.softmax_param);
                }
                if (object.slice_param != null) {
                    if (typeof object.slice_param !== "object")
                        throw TypeError(".caffe.V1LayerParameter.slice_param: object expected");
                    message.slice_param = $root.caffe.SliceParameter.fromObject(object.slice_param);
                }
                if (object.tanh_param != null) {
                    if (typeof object.tanh_param !== "object")
                        throw TypeError(".caffe.V1LayerParameter.tanh_param: object expected");
                    message.tanh_param = $root.caffe.TanHParameter.fromObject(object.tanh_param);
                }
                if (object.threshold_param != null) {
                    if (typeof object.threshold_param !== "object")
                        throw TypeError(".caffe.V1LayerParameter.threshold_param: object expected");
                    message.threshold_param = $root.caffe.ThresholdParameter.fromObject(object.threshold_param);
                }
                if (object.window_data_param != null) {
                    if (typeof object.window_data_param !== "object")
                        throw TypeError(".caffe.V1LayerParameter.window_data_param: object expected");
                    message.window_data_param = $root.caffe.WindowDataParameter.fromObject(object.window_data_param);
                }
                if (object.transform_param != null) {
                    if (typeof object.transform_param !== "object")
                        throw TypeError(".caffe.V1LayerParameter.transform_param: object expected");
                    message.transform_param = $root.caffe.TransformationParameter.fromObject(object.transform_param);
                }
                if (object.loss_param != null) {
                    if (typeof object.loss_param !== "object")
                        throw TypeError(".caffe.V1LayerParameter.loss_param: object expected");
                    message.loss_param = $root.caffe.LossParameter.fromObject(object.loss_param);
                }
                if (object.layer != null) {
                    if (typeof object.layer !== "object")
                        throw TypeError(".caffe.V1LayerParameter.layer: object expected");
                    message.layer = $root.caffe.V0LayerParameter.fromObject(object.layer);
                }
                return message;
            };
    
            V1LayerParameter.toObject = function toObject(message, options) {
                if (!options)
                    options = {};
                var object = {};
                if (options.arrays || options.defaults) {
                    object.bottom = [];
                    object.top = [];
                    object.blobs = [];
                    object.blobs_lr = [];
                    object.weight_decay = [];
                    object.include = [];
                    object.exclude = [];
                    object.loss_weight = [];
                    object.param = [];
                    object.blob_share_mode = [];
                }
                if (options.defaults) {
                    object.layer = null;
                    object.name = "";
                    object.type = options.enums === String ? "NONE" : 0;
                    object.concat_param = null;
                    object.convolution_param = null;
                    object.data_param = null;
                    object.dropout_param = null;
                    object.hdf5_data_param = null;
                    object.hdf5_output_param = null;
                    object.image_data_param = null;
                    object.infogain_loss_param = null;
                    object.inner_product_param = null;
                    object.lrn_param = null;
                    object.pooling_param = null;
                    object.window_data_param = null;
                    object.power_param = null;
                    object.memory_data_param = null;
                    object.argmax_param = null;
                    object.eltwise_param = null;
                    object.threshold_param = null;
                    object.dummy_data_param = null;
                    object.accuracy_param = null;
                    object.hinge_loss_param = null;
                    object.relu_param = null;
                    object.slice_param = null;
                    object.mvn_param = null;
                    object.transform_param = null;
                    object.tanh_param = null;
                    object.sigmoid_param = null;
                    object.softmax_param = null;
                    object.contrastive_loss_param = null;
                    object.exp_param = null;
                    object.loss_param = null;
                }
                if (message.layer != null && message.hasOwnProperty("layer"))
                    object.layer = $root.caffe.V0LayerParameter.toObject(message.layer, options);
                if (message.bottom && message.bottom.length) {
                    object.bottom = [];
                    for (var j = 0; j < message.bottom.length; ++j)
                        object.bottom[j] = message.bottom[j];
                }
                if (message.top && message.top.length) {
                    object.top = [];
                    for (var j = 0; j < message.top.length; ++j)
                        object.top[j] = message.top[j];
                }
                if (message.name != null && message.hasOwnProperty("name"))
                    object.name = message.name;
                if (message.type != null && message.hasOwnProperty("type"))
                    object.type = options.enums === String ? $root.caffe.V1LayerParameter.LayerType[message.type] : message.type;
                if (message.blobs && message.blobs.length) {
                    object.blobs = [];
                    for (var j = 0; j < message.blobs.length; ++j)
                        object.blobs[j] = $root.caffe.BlobProto.toObject(message.blobs[j], options);
                }
                if (message.blobs_lr && message.blobs_lr.length) {
                    object.blobs_lr = [];
                    for (var j = 0; j < message.blobs_lr.length; ++j)
                        object.blobs_lr[j] = options.json && !isFinite(message.blobs_lr[j]) ? String(message.blobs_lr[j]) : message.blobs_lr[j];
                }
                if (message.weight_decay && message.weight_decay.length) {
                    object.weight_decay = [];
                    for (var j = 0; j < message.weight_decay.length; ++j)
                        object.weight_decay[j] = options.json && !isFinite(message.weight_decay[j]) ? String(message.weight_decay[j]) : message.weight_decay[j];
                }
                if (message.concat_param != null && message.hasOwnProperty("concat_param"))
                    object.concat_param = $root.caffe.ConcatParameter.toObject(message.concat_param, options);
                if (message.convolution_param != null && message.hasOwnProperty("convolution_param"))
                    object.convolution_param = $root.caffe.ConvolutionParameter.toObject(message.convolution_param, options);
                if (message.data_param != null && message.hasOwnProperty("data_param"))
                    object.data_param = $root.caffe.DataParameter.toObject(message.data_param, options);
                if (message.dropout_param != null && message.hasOwnProperty("dropout_param"))
                    object.dropout_param = $root.caffe.DropoutParameter.toObject(message.dropout_param, options);
                if (message.hdf5_data_param != null && message.hasOwnProperty("hdf5_data_param"))
                    object.hdf5_data_param = $root.caffe.HDF5DataParameter.toObject(message.hdf5_data_param, options);
                if (message.hdf5_output_param != null && message.hasOwnProperty("hdf5_output_param"))
                    object.hdf5_output_param = $root.caffe.HDF5OutputParameter.toObject(message.hdf5_output_param, options);
                if (message.image_data_param != null && message.hasOwnProperty("image_data_param"))
                    object.image_data_param = $root.caffe.ImageDataParameter.toObject(message.image_data_param, options);
                if (message.infogain_loss_param != null && message.hasOwnProperty("infogain_loss_param"))
                    object.infogain_loss_param = $root.caffe.InfogainLossParameter.toObject(message.infogain_loss_param, options);
                if (message.inner_product_param != null && message.hasOwnProperty("inner_product_param"))
                    object.inner_product_param = $root.caffe.InnerProductParameter.toObject(message.inner_product_param, options);
                if (message.lrn_param != null && message.hasOwnProperty("lrn_param"))
                    object.lrn_param = $root.caffe.LRNParameter.toObject(message.lrn_param, options);
                if (message.pooling_param != null && message.hasOwnProperty("pooling_param"))
                    object.pooling_param = $root.caffe.PoolingParameter.toObject(message.pooling_param, options);
                if (message.window_data_param != null && message.hasOwnProperty("window_data_param"))
                    object.window_data_param = $root.caffe.WindowDataParameter.toObject(message.window_data_param, options);
                if (message.power_param != null && message.hasOwnProperty("power_param"))
                    object.power_param = $root.caffe.PowerParameter.toObject(message.power_param, options);
                if (message.memory_data_param != null && message.hasOwnProperty("memory_data_param"))
                    object.memory_data_param = $root.caffe.MemoryDataParameter.toObject(message.memory_data_param, options);
                if (message.argmax_param != null && message.hasOwnProperty("argmax_param"))
                    object.argmax_param = $root.caffe.ArgMaxParameter.toObject(message.argmax_param, options);
                if (message.eltwise_param != null && message.hasOwnProperty("eltwise_param"))
                    object.eltwise_param = $root.caffe.EltwiseParameter.toObject(message.eltwise_param, options);
                if (message.threshold_param != null && message.hasOwnProperty("threshold_param"))
                    object.threshold_param = $root.caffe.ThresholdParameter.toObject(message.threshold_param, options);
                if (message.dummy_data_param != null && message.hasOwnProperty("dummy_data_param"))
                    object.dummy_data_param = $root.caffe.DummyDataParameter.toObject(message.dummy_data_param, options);
                if (message.accuracy_param != null && message.hasOwnProperty("accuracy_param"))
                    object.accuracy_param = $root.caffe.AccuracyParameter.toObject(message.accuracy_param, options);
                if (message.hinge_loss_param != null && message.hasOwnProperty("hinge_loss_param"))
                    object.hinge_loss_param = $root.caffe.HingeLossParameter.toObject(message.hinge_loss_param, options);
                if (message.relu_param != null && message.hasOwnProperty("relu_param"))
                    object.relu_param = $root.caffe.ReLUParameter.toObject(message.relu_param, options);
                if (message.slice_param != null && message.hasOwnProperty("slice_param"))
                    object.slice_param = $root.caffe.SliceParameter.toObject(message.slice_param, options);
                if (message.include && message.include.length) {
                    object.include = [];
                    for (var j = 0; j < message.include.length; ++j)
                        object.include[j] = $root.caffe.NetStateRule.toObject(message.include[j], options);
                }
                if (message.exclude && message.exclude.length) {
                    object.exclude = [];
                    for (var j = 0; j < message.exclude.length; ++j)
                        object.exclude[j] = $root.caffe.NetStateRule.toObject(message.exclude[j], options);
                }
                if (message.mvn_param != null && message.hasOwnProperty("mvn_param"))
                    object.mvn_param = $root.caffe.MVNParameter.toObject(message.mvn_param, options);
                if (message.loss_weight && message.loss_weight.length) {
                    object.loss_weight = [];
                    for (var j = 0; j < message.loss_weight.length; ++j)
                        object.loss_weight[j] = options.json && !isFinite(message.loss_weight[j]) ? String(message.loss_weight[j]) : message.loss_weight[j];
                }
                if (message.transform_param != null && message.hasOwnProperty("transform_param"))
                    object.transform_param = $root.caffe.TransformationParameter.toObject(message.transform_param, options);
                if (message.tanh_param != null && message.hasOwnProperty("tanh_param"))
                    object.tanh_param = $root.caffe.TanHParameter.toObject(message.tanh_param, options);
                if (message.sigmoid_param != null && message.hasOwnProperty("sigmoid_param"))
                    object.sigmoid_param = $root.caffe.SigmoidParameter.toObject(message.sigmoid_param, options);
                if (message.softmax_param != null && message.hasOwnProperty("softmax_param"))
                    object.softmax_param = $root.caffe.SoftmaxParameter.toObject(message.softmax_param, options);
                if (message.contrastive_loss_param != null && message.hasOwnProperty("contrastive_loss_param"))
                    object.contrastive_loss_param = $root.caffe.ContrastiveLossParameter.toObject(message.contrastive_loss_param, options);
                if (message.exp_param != null && message.hasOwnProperty("exp_param"))
                    object.exp_param = $root.caffe.ExpParameter.toObject(message.exp_param, options);
                if (message.loss_param != null && message.hasOwnProperty("loss_param"))
                    object.loss_param = $root.caffe.LossParameter.toObject(message.loss_param, options);
                if (message.param && message.param.length) {
                    object.param = [];
                    for (var j = 0; j < message.param.length; ++j)
                        object.param[j] = message.param[j];
                }
                if (message.blob_share_mode && message.blob_share_mode.length) {
                    object.blob_share_mode = [];
                    for (var j = 0; j < message.blob_share_mode.length; ++j)
                        object.blob_share_mode[j] = options.enums === String ? $root.caffe.V1LayerParameter.DimCheckMode[message.blob_share_mode[j]] : message.blob_share_mode[j];
                }
                return object;
            };
    
            V1LayerParameter.prototype.toJSON = function toJSON() {
                return this.constructor.toObject(this, $protobuf.util.toJSONOptions);
            };
    
            V1LayerParameter.LayerType = (function() {
                var valuesById = {}, values = Object.create(valuesById);
                values[valuesById[0] = "NONE"] = 0;
                values[valuesById[35] = "ABSVAL"] = 35;
                values[valuesById[1] = "ACCURACY"] = 1;
                values[valuesById[30] = "ARGMAX"] = 30;
                values[valuesById[2] = "BNLL"] = 2;
                values[valuesById[3] = "CONCAT"] = 3;
                values[valuesById[37] = "CONTRASTIVE_LOSS"] = 37;
                values[valuesById[4] = "CONVOLUTION"] = 4;
                values[valuesById[5] = "DATA"] = 5;
                values[valuesById[39] = "DECONVOLUTION"] = 39;
                values[valuesById[6] = "DROPOUT"] = 6;
                values[valuesById[32] = "DUMMY_DATA"] = 32;
                values[valuesById[7] = "EUCLIDEAN_LOSS"] = 7;
                values[valuesById[25] = "ELTWISE"] = 25;
                values[valuesById[38] = "EXP"] = 38;
                values[valuesById[8] = "FLATTEN"] = 8;
                values[valuesById[9] = "HDF5_DATA"] = 9;
                values[valuesById[10] = "HDF5_OUTPUT"] = 10;
                values[valuesById[28] = "HINGE_LOSS"] = 28;
                values[valuesById[11] = "IM2COL"] = 11;
                values[valuesById[12] = "IMAGE_DATA"] = 12;
                values[valuesById[13] = "INFOGAIN_LOSS"] = 13;
                values[valuesById[14] = "INNER_PRODUCT"] = 14;
                values[valuesById[15] = "LRN"] = 15;
                values[valuesById[29] = "MEMORY_DATA"] = 29;
                values[valuesById[16] = "MULTINOMIAL_LOGISTIC_LOSS"] = 16;
                values[valuesById[34] = "MVN"] = 34;
                values[valuesById[17] = "POOLING"] = 17;
                values[valuesById[26] = "POWER"] = 26;
                values[valuesById[18] = "RELU"] = 18;
                values[valuesById[19] = "SIGMOID"] = 19;
                values[valuesById[27] = "SIGMOID_CROSS_ENTROPY_LOSS"] = 27;
                values[valuesById[36] = "SILENCE"] = 36;
                values[valuesById[20] = "SOFTMAX"] = 20;
                values[valuesById[21] = "SOFTMAX_LOSS"] = 21;
                values[valuesById[22] = "SPLIT"] = 22;
                values[valuesById[33] = "SLICE"] = 33;
                values[valuesById[23] = "TANH"] = 23;
                values[valuesById[24] = "WINDOW_DATA"] = 24;
                values[valuesById[31] = "THRESHOLD"] = 31;
                return values;
            })();
    
            V1LayerParameter.DimCheckMode = (function() {
                var valuesById = {}, values = Object.create(valuesById);
                values[valuesById[0] = "STRICT"] = 0;
                values[valuesById[1] = "PERMISSIVE"] = 1;
                return values;
            })();
    
            return V1LayerParameter;
        })();
    
        caffe.V0LayerParameter = (function() {
    
            function V0LayerParameter(properties) {
                this.blobs = [];
                this.blobs_lr = [];
                this.weight_decay = [];
                if (properties)
                    for (var keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                        if (properties[keys[i]] != null)
                            this[keys[i]] = properties[keys[i]];
            }
    
            V0LayerParameter.prototype.name = "";
            V0LayerParameter.prototype.type = "";
            V0LayerParameter.prototype.num_output = 0;
            V0LayerParameter.prototype.biasterm = true;
            V0LayerParameter.prototype.weight_filler = null;
            V0LayerParameter.prototype.bias_filler = null;
            V0LayerParameter.prototype.pad = 0;
            V0LayerParameter.prototype.kernelsize = 0;
            V0LayerParameter.prototype.group = 1;
            V0LayerParameter.prototype.stride = 1;
            V0LayerParameter.prototype.pool = 0;
            V0LayerParameter.prototype.dropout_ratio = 0.5;
            V0LayerParameter.prototype.local_size = 5;
            V0LayerParameter.prototype.alpha = 1;
            V0LayerParameter.prototype.beta = 0.75;
            V0LayerParameter.prototype.k = 1;
            V0LayerParameter.prototype.source = "";
            V0LayerParameter.prototype.scale = 1;
            V0LayerParameter.prototype.meanfile = "";
            V0LayerParameter.prototype.batchsize = 0;
            V0LayerParameter.prototype.cropsize = 0;
            V0LayerParameter.prototype.mirror = false;
            V0LayerParameter.prototype.blobs = $util.emptyArray;
            V0LayerParameter.prototype.blobs_lr = $util.emptyArray;
            V0LayerParameter.prototype.weight_decay = $util.emptyArray;
            V0LayerParameter.prototype.rand_skip = 0;
            V0LayerParameter.prototype.det_fg_threshold = 0.5;
            V0LayerParameter.prototype.det_bg_threshold = 0.5;
            V0LayerParameter.prototype.det_fg_fraction = 0.25;
            V0LayerParameter.prototype.det_context_pad = 0;
            V0LayerParameter.prototype.det_crop_mode = "warp";
            V0LayerParameter.prototype.new_num = 0;
            V0LayerParameter.prototype.new_channels = 0;
            V0LayerParameter.prototype.new_height = 0;
            V0LayerParameter.prototype.new_width = 0;
            V0LayerParameter.prototype.shuffle_images = false;
            V0LayerParameter.prototype.concat_dim = 1;
            V0LayerParameter.prototype.hdf5_output_param = null;
    
            V0LayerParameter.create = function create(properties) {
                return new V0LayerParameter(properties);
            };
    
            V0LayerParameter.decode = function decode(reader, length) {
                if (!(reader instanceof $Reader))
                    reader = $Reader.create(reader);
                var end = length === undefined ? reader.len : reader.pos + length, message = new $root.caffe.V0LayerParameter();
                while (reader.pos < end) {
                    var tag = reader.uint32();
                    switch (tag >>> 3) {
                    case 1:
                        message.name = reader.string();
                        break;
                    case 2:
                        message.type = reader.string();
                        break;
                    case 3:
                        message.num_output = reader.uint32();
                        break;
                    case 4:
                        message.biasterm = reader.bool();
                        break;
                    case 5:
                        message.weight_filler = $root.caffe.FillerParameter.decode(reader, reader.uint32());
                        break;
                    case 6:
                        message.bias_filler = $root.caffe.FillerParameter.decode(reader, reader.uint32());
                        break;
                    case 7:
                        message.pad = reader.uint32();
                        break;
                    case 8:
                        message.kernelsize = reader.uint32();
                        break;
                    case 9:
                        message.group = reader.uint32();
                        break;
                    case 10:
                        message.stride = reader.uint32();
                        break;
                    case 11:
                        message.pool = reader.int32();
                        break;
                    case 12:
                        message.dropout_ratio = reader.float();
                        break;
                    case 13:
                        message.local_size = reader.uint32();
                        break;
                    case 14:
                        message.alpha = reader.float();
                        break;
                    case 15:
                        message.beta = reader.float();
                        break;
                    case 22:
                        message.k = reader.float();
                        break;
                    case 16:
                        message.source = reader.string();
                        break;
                    case 17:
                        message.scale = reader.float();
                        break;
                    case 18:
                        message.meanfile = reader.string();
                        break;
                    case 19:
                        message.batchsize = reader.uint32();
                        break;
                    case 20:
                        message.cropsize = reader.uint32();
                        break;
                    case 21:
                        message.mirror = reader.bool();
                        break;
                    case 50:
                        if (!(message.blobs && message.blobs.length))
                            message.blobs = [];
                        message.blobs.push($root.caffe.BlobProto.decode(reader, reader.uint32()));
                        break;
                    case 51:
                        if (!(message.blobs_lr && message.blobs_lr.length))
                            message.blobs_lr = [];
                        if ((tag & 7) === 2) {
                            var end2 = reader.uint32() + reader.pos;
                            while (reader.pos < end2)
                                message.blobs_lr.push(reader.float());
                        } else
                            message.blobs_lr.push(reader.float());
                        break;
                    case 52:
                        if (!(message.weight_decay && message.weight_decay.length))
                            message.weight_decay = [];
                        if ((tag & 7) === 2) {
                            var end2 = reader.uint32() + reader.pos;
                            while (reader.pos < end2)
                                message.weight_decay.push(reader.float());
                        } else
                            message.weight_decay.push(reader.float());
                        break;
                    case 53:
                        message.rand_skip = reader.uint32();
                        break;
                    case 54:
                        message.det_fg_threshold = reader.float();
                        break;
                    case 55:
                        message.det_bg_threshold = reader.float();
                        break;
                    case 56:
                        message.det_fg_fraction = reader.float();
                        break;
                    case 58:
                        message.det_context_pad = reader.uint32();
                        break;
                    case 59:
                        message.det_crop_mode = reader.string();
                        break;
                    case 60:
                        message.new_num = reader.int32();
                        break;
                    case 61:
                        message.new_channels = reader.int32();
                        break;
                    case 62:
                        message.new_height = reader.int32();
                        break;
                    case 63:
                        message.new_width = reader.int32();
                        break;
                    case 64:
                        message.shuffle_images = reader.bool();
                        break;
                    case 65:
                        message.concat_dim = reader.uint32();
                        break;
                    case 1001:
                        message.hdf5_output_param = $root.caffe.HDF5OutputParameter.decode(reader, reader.uint32());
                        break;
                    default:
                        reader.skipType(tag & 7);
                        break;
                    }
                }
                return message;
            };
    
            V0LayerParameter.decodeText = function decodeText(reader, block) {
                if (!(reader instanceof $TextReader))
                    reader = $TextReader.create(reader);
                var message = new $root.caffe.V0LayerParameter();
                reader.start(block);
                while (!reader.end(block)) {
                    var tag = reader.tag();
                    switch (tag) {
                    case "name":
                        message.name = reader.string();
                        break;
                    case "type":
                        message.type = reader.string();
                        break;
                    case "num_output":
                        message.num_output = reader.uint32();
                        break;
                    case "biasterm":
                        message.biasterm = reader.bool();
                        break;
                    case "weight_filler":
                        message.weight_filler = $root.caffe.FillerParameter.decodeText(reader, true);
                        break;
                    case "bias_filler":
                        message.bias_filler = $root.caffe.FillerParameter.decodeText(reader, true);
                        break;
                    case "pad":
                        message.pad = reader.uint32();
                        break;
                    case "kernelsize":
                        message.kernelsize = reader.uint32();
                        break;
                    case "group":
                        message.group = reader.uint32();
                        break;
                    case "stride":
                        message.stride = reader.uint32();
                        break;
                    case "pool":
                        message.pool = reader.enum($root.caffe.V0LayerParameter.PoolMethod);
                        break;
                    case "dropout_ratio":
                        message.dropout_ratio = reader.float();
                        break;
                    case "local_size":
                        message.local_size = reader.uint32();
                        break;
                    case "alpha":
                        message.alpha = reader.float();
                        break;
                    case "beta":
                        message.beta = reader.float();
                        break;
                    case "k":
                        message.k = reader.float();
                        break;
                    case "source":
                        message.source = reader.string();
                        break;
                    case "scale":
                        message.scale = reader.float();
                        break;
                    case "meanfile":
                        message.meanfile = reader.string();
                        break;
                    case "batchsize":
                        message.batchsize = reader.uint32();
                        break;
                    case "cropsize":
                        message.cropsize = reader.uint32();
                        break;
                    case "mirror":
                        message.mirror = reader.bool();
                        break;
                    case "blobs":
                        if (!(message.blobs && message.blobs.length))
                            message.blobs = [];
                        message.blobs.push($root.caffe.BlobProto.decodeText(reader, true));
                        break;
                    case "blobs_lr":
                        if (!(message.blobs_lr && message.blobs_lr.length))
                            message.blobs_lr = [];
                        message.blobs_lr.push(reader.float());
                        break;
                    case "weight_decay":
                        if (!(message.weight_decay && message.weight_decay.length))
                            message.weight_decay = [];
                        message.weight_decay.push(reader.float());
                        break;
                    case "rand_skip":
                        message.rand_skip = reader.uint32();
                        break;
                    case "det_fg_threshold":
                        message.det_fg_threshold = reader.float();
                        break;
                    case "det_bg_threshold":
                        message.det_bg_threshold = reader.float();
                        break;
                    case "det_fg_fraction":
                        message.det_fg_fraction = reader.float();
                        break;
                    case "det_context_pad":
                        message.det_context_pad = reader.uint32();
                        break;
                    case "det_crop_mode":
                        message.det_crop_mode = reader.string();
                        break;
                    case "new_num":
                        message.new_num = reader.int32();
                        break;
                    case "new_channels":
                        message.new_channels = reader.int32();
                        break;
                    case "new_height":
                        message.new_height = reader.int32();
                        break;
                    case "new_width":
                        message.new_width = reader.int32();
                        break;
                    case "shuffle_images":
                        message.shuffle_images = reader.bool();
                        break;
                    case "concat_dim":
                        message.concat_dim = reader.uint32();
                        break;
                    case "hdf5_output_param":
                        message.hdf5_output_param = $root.caffe.HDF5OutputParameter.decodeText(reader, true);
                        break;
                    default:
                        reader.handle(tag, message);
                        break;
                    }
                }
                return message;
            };
    
            V0LayerParameter.verify = function verify(message) {
                if (typeof message !== "object" || message === null)
                    return "object expected";
                if (message.name != null && message.hasOwnProperty("name"))
                    if (!$util.isString(message.name))
                        return "name: string expected";
                if (message.type != null && message.hasOwnProperty("type"))
                    if (!$util.isString(message.type))
                        return "type: string expected";
                if (message.num_output != null && message.hasOwnProperty("num_output"))
                    if (!$util.isInteger(message.num_output))
                        return "num_output: integer expected";
                if (message.biasterm != null && message.hasOwnProperty("biasterm"))
                    if (typeof message.biasterm !== "boolean")
                        return "biasterm: boolean expected";
                if (message.weight_filler != null && message.hasOwnProperty("weight_filler")) {
                    var error = $root.caffe.FillerParameter.verify(message.weight_filler);
                    if (error)
                        return "weight_filler." + error;
                }
                if (message.bias_filler != null && message.hasOwnProperty("bias_filler")) {
                    var error = $root.caffe.FillerParameter.verify(message.bias_filler);
                    if (error)
                        return "bias_filler." + error;
                }
                if (message.pad != null && message.hasOwnProperty("pad"))
                    if (!$util.isInteger(message.pad))
                        return "pad: integer expected";
                if (message.kernelsize != null && message.hasOwnProperty("kernelsize"))
                    if (!$util.isInteger(message.kernelsize))
                        return "kernelsize: integer expected";
                if (message.group != null && message.hasOwnProperty("group"))
                    if (!$util.isInteger(message.group))
                        return "group: integer expected";
                if (message.stride != null && message.hasOwnProperty("stride"))
                    if (!$util.isInteger(message.stride))
                        return "stride: integer expected";
                if (message.pool != null && message.hasOwnProperty("pool"))
                    switch (message.pool) {
                    default:
                        return "pool: enum value expected";
                    case 0:
                    case 1:
                    case 2:
                        break;
                    }
                if (message.dropout_ratio != null && message.hasOwnProperty("dropout_ratio"))
                    if (typeof message.dropout_ratio !== "number")
                        return "dropout_ratio: number expected";
                if (message.local_size != null && message.hasOwnProperty("local_size"))
                    if (!$util.isInteger(message.local_size))
                        return "local_size: integer expected";
                if (message.alpha != null && message.hasOwnProperty("alpha"))
                    if (typeof message.alpha !== "number")
                        return "alpha: number expected";
                if (message.beta != null && message.hasOwnProperty("beta"))
                    if (typeof message.beta !== "number")
                        return "beta: number expected";
                if (message.k != null && message.hasOwnProperty("k"))
                    if (typeof message.k !== "number")
                        return "k: number expected";
                if (message.source != null && message.hasOwnProperty("source"))
                    if (!$util.isString(message.source))
                        return "source: string expected";
                if (message.scale != null && message.hasOwnProperty("scale"))
                    if (typeof message.scale !== "number")
                        return "scale: number expected";
                if (message.meanfile != null && message.hasOwnProperty("meanfile"))
                    if (!$util.isString(message.meanfile))
                        return "meanfile: string expected";
                if (message.batchsize != null && message.hasOwnProperty("batchsize"))
                    if (!$util.isInteger(message.batchsize))
                        return "batchsize: integer expected";
                if (message.cropsize != null && message.hasOwnProperty("cropsize"))
                    if (!$util.isInteger(message.cropsize))
                        return "cropsize: integer expected";
                if (message.mirror != null && message.hasOwnProperty("mirror"))
                    if (typeof message.mirror !== "boolean")
                        return "mirror: boolean expected";
                if (message.blobs != null && message.hasOwnProperty("blobs")) {
                    if (!Array.isArray(message.blobs))
                        return "blobs: array expected";
                    for (var i = 0; i < message.blobs.length; ++i) {
                        var error = $root.caffe.BlobProto.verify(message.blobs[i]);
                        if (error)
                            return "blobs." + error;
                    }
                }
                if (message.blobs_lr != null && message.hasOwnProperty("blobs_lr")) {
                    if (!Array.isArray(message.blobs_lr))
                        return "blobs_lr: array expected";
                    for (var i = 0; i < message.blobs_lr.length; ++i)
                        if (typeof message.blobs_lr[i] !== "number")
                            return "blobs_lr: number[] expected";
                }
                if (message.weight_decay != null && message.hasOwnProperty("weight_decay")) {
                    if (!Array.isArray(message.weight_decay))
                        return "weight_decay: array expected";
                    for (var i = 0; i < message.weight_decay.length; ++i)
                        if (typeof message.weight_decay[i] !== "number")
                            return "weight_decay: number[] expected";
                }
                if (message.rand_skip != null && message.hasOwnProperty("rand_skip"))
                    if (!$util.isInteger(message.rand_skip))
                        return "rand_skip: integer expected";
                if (message.det_fg_threshold != null && message.hasOwnProperty("det_fg_threshold"))
                    if (typeof message.det_fg_threshold !== "number")
                        return "det_fg_threshold: number expected";
                if (message.det_bg_threshold != null && message.hasOwnProperty("det_bg_threshold"))
                    if (typeof message.det_bg_threshold !== "number")
                        return "det_bg_threshold: number expected";
                if (message.det_fg_fraction != null && message.hasOwnProperty("det_fg_fraction"))
                    if (typeof message.det_fg_fraction !== "number")
                        return "det_fg_fraction: number expected";
                if (message.det_context_pad != null && message.hasOwnProperty("det_context_pad"))
                    if (!$util.isInteger(message.det_context_pad))
                        return "det_context_pad: integer expected";
                if (message.det_crop_mode != null && message.hasOwnProperty("det_crop_mode"))
                    if (!$util.isString(message.det_crop_mode))
                        return "det_crop_mode: string expected";
                if (message.new_num != null && message.hasOwnProperty("new_num"))
                    if (!$util.isInteger(message.new_num))
                        return "new_num: integer expected";
                if (message.new_channels != null && message.hasOwnProperty("new_channels"))
                    if (!$util.isInteger(message.new_channels))
                        return "new_channels: integer expected";
                if (message.new_height != null && message.hasOwnProperty("new_height"))
                    if (!$util.isInteger(message.new_height))
                        return "new_height: integer expected";
                if (message.new_width != null && message.hasOwnProperty("new_width"))
                    if (!$util.isInteger(message.new_width))
                        return "new_width: integer expected";
                if (message.shuffle_images != null && message.hasOwnProperty("shuffle_images"))
                    if (typeof message.shuffle_images !== "boolean")
                        return "shuffle_images: boolean expected";
                if (message.concat_dim != null && message.hasOwnProperty("concat_dim"))
                    if (!$util.isInteger(message.concat_dim))
                        return "concat_dim: integer expected";
                if (message.hdf5_output_param != null && message.hasOwnProperty("hdf5_output_param")) {
                    var error = $root.caffe.HDF5OutputParameter.verify(message.hdf5_output_param);
                    if (error)
                        return "hdf5_output_param." + error;
                }
                return null;
            };
    
            V0LayerParameter.fromObject = function fromObject(object) {
                if (object instanceof $root.caffe.V0LayerParameter)
                    return object;
                var message = new $root.caffe.V0LayerParameter();
                if (object.name != null)
                    message.name = String(object.name);
                if (object.type != null)
                    message.type = String(object.type);
                if (object.num_output != null)
                    message.num_output = object.num_output >>> 0;
                if (object.biasterm != null)
                    message.biasterm = Boolean(object.biasterm);
                if (object.weight_filler != null) {
                    if (typeof object.weight_filler !== "object")
                        throw TypeError(".caffe.V0LayerParameter.weight_filler: object expected");
                    message.weight_filler = $root.caffe.FillerParameter.fromObject(object.weight_filler);
                }
                if (object.bias_filler != null) {
                    if (typeof object.bias_filler !== "object")
                        throw TypeError(".caffe.V0LayerParameter.bias_filler: object expected");
                    message.bias_filler = $root.caffe.FillerParameter.fromObject(object.bias_filler);
                }
                if (object.pad != null)
                    message.pad = object.pad >>> 0;
                if (object.kernelsize != null)
                    message.kernelsize = object.kernelsize >>> 0;
                if (object.group != null)
                    message.group = object.group >>> 0;
                if (object.stride != null)
                    message.stride = object.stride >>> 0;
                switch (object.pool) {
                case "MAX":
                case 0:
                    message.pool = 0;
                    break;
                case "AVE":
                case 1:
                    message.pool = 1;
                    break;
                case "STOCHASTIC":
                case 2:
                    message.pool = 2;
                    break;
                }
                if (object.dropout_ratio != null)
                    message.dropout_ratio = Number(object.dropout_ratio);
                if (object.local_size != null)
                    message.local_size = object.local_size >>> 0;
                if (object.alpha != null)
                    message.alpha = Number(object.alpha);
                if (object.beta != null)
                    message.beta = Number(object.beta);
                if (object.k != null)
                    message.k = Number(object.k);
                if (object.source != null)
                    message.source = String(object.source);
                if (object.scale != null)
                    message.scale = Number(object.scale);
                if (object.meanfile != null)
                    message.meanfile = String(object.meanfile);
                if (object.batchsize != null)
                    message.batchsize = object.batchsize >>> 0;
                if (object.cropsize != null)
                    message.cropsize = object.cropsize >>> 0;
                if (object.mirror != null)
                    message.mirror = Boolean(object.mirror);
                if (object.blobs) {
                    if (!Array.isArray(object.blobs))
                        throw TypeError(".caffe.V0LayerParameter.blobs: array expected");
                    message.blobs = [];
                    for (var i = 0; i < object.blobs.length; ++i) {
                        if (typeof object.blobs[i] !== "object")
                            throw TypeError(".caffe.V0LayerParameter.blobs: object expected");
                        message.blobs[i] = $root.caffe.BlobProto.fromObject(object.blobs[i]);
                    }
                }
                if (object.blobs_lr) {
                    if (!Array.isArray(object.blobs_lr))
                        throw TypeError(".caffe.V0LayerParameter.blobs_lr: array expected");
                    message.blobs_lr = [];
                    for (var i = 0; i < object.blobs_lr.length; ++i)
                        message.blobs_lr[i] = Number(object.blobs_lr[i]);
                }
                if (object.weight_decay) {
                    if (!Array.isArray(object.weight_decay))
                        throw TypeError(".caffe.V0LayerParameter.weight_decay: array expected");
                    message.weight_decay = [];
                    for (var i = 0; i < object.weight_decay.length; ++i)
                        message.weight_decay[i] = Number(object.weight_decay[i]);
                }
                if (object.rand_skip != null)
                    message.rand_skip = object.rand_skip >>> 0;
                if (object.det_fg_threshold != null)
                    message.det_fg_threshold = Number(object.det_fg_threshold);
                if (object.det_bg_threshold != null)
                    message.det_bg_threshold = Number(object.det_bg_threshold);
                if (object.det_fg_fraction != null)
                    message.det_fg_fraction = Number(object.det_fg_fraction);
                if (object.det_context_pad != null)
                    message.det_context_pad = object.det_context_pad >>> 0;
                if (object.det_crop_mode != null)
                    message.det_crop_mode = String(object.det_crop_mode);
                if (object.new_num != null)
                    message.new_num = object.new_num | 0;
                if (object.new_channels != null)
                    message.new_channels = object.new_channels | 0;
                if (object.new_height != null)
                    message.new_height = object.new_height | 0;
                if (object.new_width != null)
                    message.new_width = object.new_width | 0;
                if (object.shuffle_images != null)
                    message.shuffle_images = Boolean(object.shuffle_images);
                if (object.concat_dim != null)
                    message.concat_dim = object.concat_dim >>> 0;
                if (object.hdf5_output_param != null) {
                    if (typeof object.hdf5_output_param !== "object")
                        throw TypeError(".caffe.V0LayerParameter.hdf5_output_param: object expected");
                    message.hdf5_output_param = $root.caffe.HDF5OutputParameter.fromObject(object.hdf5_output_param);
                }
                return message;
            };
    
            V0LayerParameter.toObject = function toObject(message, options) {
                if (!options)
                    options = {};
                var object = {};
                if (options.arrays || options.defaults) {
                    object.blobs = [];
                    object.blobs_lr = [];
                    object.weight_decay = [];
                }
                if (options.defaults) {
                    object.name = "";
                    object.type = "";
                    object.num_output = 0;
                    object.biasterm = true;
                    object.weight_filler = null;
                    object.bias_filler = null;
                    object.pad = 0;
                    object.kernelsize = 0;
                    object.group = 1;
                    object.stride = 1;
                    object.pool = options.enums === String ? "MAX" : 0;
                    object.dropout_ratio = 0.5;
                    object.local_size = 5;
                    object.alpha = 1;
                    object.beta = 0.75;
                    object.source = "";
                    object.scale = 1;
                    object.meanfile = "";
                    object.batchsize = 0;
                    object.cropsize = 0;
                    object.mirror = false;
                    object.k = 1;
                    object.rand_skip = 0;
                    object.det_fg_threshold = 0.5;
                    object.det_bg_threshold = 0.5;
                    object.det_fg_fraction = 0.25;
                    object.det_context_pad = 0;
                    object.det_crop_mode = "warp";
                    object.new_num = 0;
                    object.new_channels = 0;
                    object.new_height = 0;
                    object.new_width = 0;
                    object.shuffle_images = false;
                    object.concat_dim = 1;
                    object.hdf5_output_param = null;
                }
                if (message.name != null && message.hasOwnProperty("name"))
                    object.name = message.name;
                if (message.type != null && message.hasOwnProperty("type"))
                    object.type = message.type;
                if (message.num_output != null && message.hasOwnProperty("num_output"))
                    object.num_output = message.num_output;
                if (message.biasterm != null && message.hasOwnProperty("biasterm"))
                    object.biasterm = message.biasterm;
                if (message.weight_filler != null && message.hasOwnProperty("weight_filler"))
                    object.weight_filler = $root.caffe.FillerParameter.toObject(message.weight_filler, options);
                if (message.bias_filler != null && message.hasOwnProperty("bias_filler"))
                    object.bias_filler = $root.caffe.FillerParameter.toObject(message.bias_filler, options);
                if (message.pad != null && message.hasOwnProperty("pad"))
                    object.pad = message.pad;
                if (message.kernelsize != null && message.hasOwnProperty("kernelsize"))
                    object.kernelsize = message.kernelsize;
                if (message.group != null && message.hasOwnProperty("group"))
                    object.group = message.group;
                if (message.stride != null && message.hasOwnProperty("stride"))
                    object.stride = message.stride;
                if (message.pool != null && message.hasOwnProperty("pool"))
                    object.pool = options.enums === String ? $root.caffe.V0LayerParameter.PoolMethod[message.pool] : message.pool;
                if (message.dropout_ratio != null && message.hasOwnProperty("dropout_ratio"))
                    object.dropout_ratio = options.json && !isFinite(message.dropout_ratio) ? String(message.dropout_ratio) : message.dropout_ratio;
                if (message.local_size != null && message.hasOwnProperty("local_size"))
                    object.local_size = message.local_size;
                if (message.alpha != null && message.hasOwnProperty("alpha"))
                    object.alpha = options.json && !isFinite(message.alpha) ? String(message.alpha) : message.alpha;
                if (message.beta != null && message.hasOwnProperty("beta"))
                    object.beta = options.json && !isFinite(message.beta) ? String(message.beta) : message.beta;
                if (message.source != null && message.hasOwnProperty("source"))
                    object.source = message.source;
                if (message.scale != null && message.hasOwnProperty("scale"))
                    object.scale = options.json && !isFinite(message.scale) ? String(message.scale) : message.scale;
                if (message.meanfile != null && message.hasOwnProperty("meanfile"))
                    object.meanfile = message.meanfile;
                if (message.batchsize != null && message.hasOwnProperty("batchsize"))
                    object.batchsize = message.batchsize;
                if (message.cropsize != null && message.hasOwnProperty("cropsize"))
                    object.cropsize = message.cropsize;
                if (message.mirror != null && message.hasOwnProperty("mirror"))
                    object.mirror = message.mirror;
                if (message.k != null && message.hasOwnProperty("k"))
                    object.k = options.json && !isFinite(message.k) ? String(message.k) : message.k;
                if (message.blobs && message.blobs.length) {
                    object.blobs = [];
                    for (var j = 0; j < message.blobs.length; ++j)
                        object.blobs[j] = $root.caffe.BlobProto.toObject(message.blobs[j], options);
                }
                if (message.blobs_lr && message.blobs_lr.length) {
                    object.blobs_lr = [];
                    for (var j = 0; j < message.blobs_lr.length; ++j)
                        object.blobs_lr[j] = options.json && !isFinite(message.blobs_lr[j]) ? String(message.blobs_lr[j]) : message.blobs_lr[j];
                }
                if (message.weight_decay && message.weight_decay.length) {
                    object.weight_decay = [];
                    for (var j = 0; j < message.weight_decay.length; ++j)
                        object.weight_decay[j] = options.json && !isFinite(message.weight_decay[j]) ? String(message.weight_decay[j]) : message.weight_decay[j];
                }
                if (message.rand_skip != null && message.hasOwnProperty("rand_skip"))
                    object.rand_skip = message.rand_skip;
                if (message.det_fg_threshold != null && message.hasOwnProperty("det_fg_threshold"))
                    object.det_fg_threshold = options.json && !isFinite(message.det_fg_threshold) ? String(message.det_fg_threshold) : message.det_fg_threshold;
                if (message.det_bg_threshold != null && message.hasOwnProperty("det_bg_threshold"))
                    object.det_bg_threshold = options.json && !isFinite(message.det_bg_threshold) ? String(message.det_bg_threshold) : message.det_bg_threshold;
                if (message.det_fg_fraction != null && message.hasOwnProperty("det_fg_fraction"))
                    object.det_fg_fraction = options.json && !isFinite(message.det_fg_fraction) ? String(message.det_fg_fraction) : message.det_fg_fraction;
                if (message.det_context_pad != null && message.hasOwnProperty("det_context_pad"))
                    object.det_context_pad = message.det_context_pad;
                if (message.det_crop_mode != null && message.hasOwnProperty("det_crop_mode"))
                    object.det_crop_mode = message.det_crop_mode;
                if (message.new_num != null && message.hasOwnProperty("new_num"))
                    object.new_num = message.new_num;
                if (message.new_channels != null && message.hasOwnProperty("new_channels"))
                    object.new_channels = message.new_channels;
                if (message.new_height != null && message.hasOwnProperty("new_height"))
                    object.new_height = message.new_height;
                if (message.new_width != null && message.hasOwnProperty("new_width"))
                    object.new_width = message.new_width;
                if (message.shuffle_images != null && message.hasOwnProperty("shuffle_images"))
                    object.shuffle_images = message.shuffle_images;
                if (message.concat_dim != null && message.hasOwnProperty("concat_dim"))
                    object.concat_dim = message.concat_dim;
                if (message.hdf5_output_param != null && message.hasOwnProperty("hdf5_output_param"))
                    object.hdf5_output_param = $root.caffe.HDF5OutputParameter.toObject(message.hdf5_output_param, options);
                return object;
            };
    
            V0LayerParameter.prototype.toJSON = function toJSON() {
                return this.constructor.toObject(this, $protobuf.util.toJSONOptions);
            };
    
            V0LayerParameter.PoolMethod = (function() {
                var valuesById = {}, values = Object.create(valuesById);
                values[valuesById[0] = "MAX"] = 0;
                values[valuesById[1] = "AVE"] = 1;
                values[valuesById[2] = "STOCHASTIC"] = 2;
                return values;
            })();
    
            return V0LayerParameter;
        })();
    
        caffe.PReLUParameter = (function() {
    
            function PReLUParameter(properties) {
                if (properties)
                    for (var keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                        if (properties[keys[i]] != null)
                            this[keys[i]] = properties[keys[i]];
            }
    
            PReLUParameter.prototype.filler = null;
            PReLUParameter.prototype.channel_shared = false;
    
            PReLUParameter.create = function create(properties) {
                return new PReLUParameter(properties);
            };
    
            PReLUParameter.decode = function decode(reader, length) {
                if (!(reader instanceof $Reader))
                    reader = $Reader.create(reader);
                var end = length === undefined ? reader.len : reader.pos + length, message = new $root.caffe.PReLUParameter();
                while (reader.pos < end) {
                    var tag = reader.uint32();
                    switch (tag >>> 3) {
                    case 1:
                        message.filler = $root.caffe.FillerParameter.decode(reader, reader.uint32());
                        break;
                    case 2:
                        message.channel_shared = reader.bool();
                        break;
                    default:
                        reader.skipType(tag & 7);
                        break;
                    }
                }
                return message;
            };
    
            PReLUParameter.decodeText = function decodeText(reader, block) {
                if (!(reader instanceof $TextReader))
                    reader = $TextReader.create(reader);
                var message = new $root.caffe.PReLUParameter();
                reader.start(block);
                while (!reader.end(block)) {
                    var tag = reader.tag();
                    switch (tag) {
                    case "filler":
                        message.filler = $root.caffe.FillerParameter.decodeText(reader, true);
                        break;
                    case "channel_shared":
                        message.channel_shared = reader.bool();
                        break;
                    default:
                        reader.handle(tag, message);
                        break;
                    }
                }
                return message;
            };
    
            PReLUParameter.verify = function verify(message) {
                if (typeof message !== "object" || message === null)
                    return "object expected";
                if (message.filler != null && message.hasOwnProperty("filler")) {
                    var error = $root.caffe.FillerParameter.verify(message.filler);
                    if (error)
                        return "filler." + error;
                }
                if (message.channel_shared != null && message.hasOwnProperty("channel_shared"))
                    if (typeof message.channel_shared !== "boolean")
                        return "channel_shared: boolean expected";
                return null;
            };
    
            PReLUParameter.fromObject = function fromObject(object) {
                if (object instanceof $root.caffe.PReLUParameter)
                    return object;
                var message = new $root.caffe.PReLUParameter();
                if (object.filler != null) {
                    if (typeof object.filler !== "object")
                        throw TypeError(".caffe.PReLUParameter.filler: object expected");
                    message.filler = $root.caffe.FillerParameter.fromObject(object.filler);
                }
                if (object.channel_shared != null)
                    message.channel_shared = Boolean(object.channel_shared);
                return message;
            };
    
            PReLUParameter.toObject = function toObject(message, options) {
                if (!options)
                    options = {};
                var object = {};
                if (options.defaults) {
                    object.filler = null;
                    object.channel_shared = false;
                }
                if (message.filler != null && message.hasOwnProperty("filler"))
                    object.filler = $root.caffe.FillerParameter.toObject(message.filler, options);
                if (message.channel_shared != null && message.hasOwnProperty("channel_shared"))
                    object.channel_shared = message.channel_shared;
                return object;
            };
    
            PReLUParameter.prototype.toJSON = function toJSON() {
                return this.constructor.toObject(this, $protobuf.util.toJSONOptions);
            };
    
            return PReLUParameter;
        })();
    
        return caffe;
    })();

    return $root;
})(protobuf);
