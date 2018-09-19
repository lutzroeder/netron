/*eslint-disable block-scoped-var, id-length, no-control-regex, no-magic-numbers, no-prototype-builtins, no-redeclare, no-shadow, no-var, sort-vars*/
(function($protobuf) {
    "use strict";

    var $Reader = $protobuf.Reader, $util = $protobuf.util;
    
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
                this.doubleData = [];
                this.doubleDiff = [];
                if (properties)
                    for (var keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                        if (properties[keys[i]] != null)
                            this[keys[i]] = properties[keys[i]];
            }
    
            BlobProto.prototype.shape = null;
            BlobProto.prototype.data = $util.emptyArray;
            BlobProto.prototype.diff = $util.emptyArray;
            BlobProto.prototype.doubleData = $util.emptyArray;
            BlobProto.prototype.doubleDiff = $util.emptyArray;
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
                        if (!(message.doubleData && message.doubleData.length))
                            message.doubleData = [];
                        if ((tag & 7) === 2) {
                            var end2 = reader.uint32() + reader.pos;
                            while (reader.pos < end2)
                                message.doubleData.push(reader.double());
                        } else
                            message.doubleData.push(reader.double());
                        break;
                    case 9:
                        if (!(message.doubleDiff && message.doubleDiff.length))
                            message.doubleDiff = [];
                        if ((tag & 7) === 2) {
                            var end2 = reader.uint32() + reader.pos;
                            while (reader.pos < end2)
                                message.doubleDiff.push(reader.double());
                        } else
                            message.doubleDiff.push(reader.double());
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
                if (message.doubleData != null && message.hasOwnProperty("doubleData")) {
                    if (!Array.isArray(message.doubleData))
                        return "doubleData: array expected";
                    for (var i = 0; i < message.doubleData.length; ++i)
                        if (typeof message.doubleData[i] !== "number")
                            return "doubleData: number[] expected";
                }
                if (message.doubleDiff != null && message.hasOwnProperty("doubleDiff")) {
                    if (!Array.isArray(message.doubleDiff))
                        return "doubleDiff: array expected";
                    for (var i = 0; i < message.doubleDiff.length; ++i)
                        if (typeof message.doubleDiff[i] !== "number")
                            return "doubleDiff: number[] expected";
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
                if (object.doubleData) {
                    if (!Array.isArray(object.doubleData))
                        throw TypeError(".caffe.BlobProto.doubleData: array expected");
                    message.doubleData = [];
                    for (var i = 0; i < object.doubleData.length; ++i)
                        message.doubleData[i] = Number(object.doubleData[i]);
                }
                if (object.doubleDiff) {
                    if (!Array.isArray(object.doubleDiff))
                        throw TypeError(".caffe.BlobProto.doubleDiff: array expected");
                    message.doubleDiff = [];
                    for (var i = 0; i < object.doubleDiff.length; ++i)
                        message.doubleDiff[i] = Number(object.doubleDiff[i]);
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
                    object.doubleData = [];
                    object.doubleDiff = [];
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
                if (message.doubleData && message.doubleData.length) {
                    object.doubleData = [];
                    for (var j = 0; j < message.doubleData.length; ++j)
                        object.doubleData[j] = options.json && !isFinite(message.doubleData[j]) ? String(message.doubleData[j]) : message.doubleData[j];
                }
                if (message.doubleDiff && message.doubleDiff.length) {
                    object.doubleDiff = [];
                    for (var j = 0; j < message.doubleDiff.length; ++j)
                        object.doubleDiff[j] = options.json && !isFinite(message.doubleDiff[j]) ? String(message.doubleDiff[j]) : message.doubleDiff[j];
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
                this.floatData = [];
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
            Datum.prototype.floatData = $util.emptyArray;
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
                        if (!(message.floatData && message.floatData.length))
                            message.floatData = [];
                        if ((tag & 7) === 2) {
                            var end2 = reader.uint32() + reader.pos;
                            while (reader.pos < end2)
                                message.floatData.push(reader.float());
                        } else
                            message.floatData.push(reader.float());
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
                if (message.floatData != null && message.hasOwnProperty("floatData")) {
                    if (!Array.isArray(message.floatData))
                        return "floatData: array expected";
                    for (var i = 0; i < message.floatData.length; ++i)
                        if (typeof message.floatData[i] !== "number")
                            return "floatData: number[] expected";
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
                if (object.floatData) {
                    if (!Array.isArray(object.floatData))
                        throw TypeError(".caffe.Datum.floatData: array expected");
                    message.floatData = [];
                    for (var i = 0; i < object.floatData.length; ++i)
                        message.floatData[i] = Number(object.floatData[i]);
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
                    object.floatData = [];
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
                if (message.floatData && message.floatData.length) {
                    object.floatData = [];
                    for (var j = 0; j < message.floatData.length; ++j)
                        object.floatData[j] = options.json && !isFinite(message.floatData[j]) ? String(message.floatData[j]) : message.floatData[j];
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
            FillerParameter.prototype.varianceNorm = 0;
    
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
                        message.varianceNorm = reader.int32();
                        break;
                    default:
                        reader.skipType(tag & 7);
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
                if (message.varianceNorm != null && message.hasOwnProperty("varianceNorm"))
                    switch (message.varianceNorm) {
                    default:
                        return "varianceNorm: enum value expected";
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
                switch (object.varianceNorm) {
                case "FAN_IN":
                case 0:
                    message.varianceNorm = 0;
                    break;
                case "FAN_OUT":
                case 1:
                    message.varianceNorm = 1;
                    break;
                case "AVERAGE":
                case 2:
                    message.varianceNorm = 2;
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
                    object.varianceNorm = options.enums === String ? "FAN_IN" : 0;
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
                if (message.varianceNorm != null && message.hasOwnProperty("varianceNorm"))
                    object.varianceNorm = options.enums === String ? $root.caffe.FillerParameter.VarianceNorm[message.varianceNorm] : message.varianceNorm;
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
                this.inputShape = [];
                this.inputDim = [];
                this.layer = [];
                this.layers = [];
                if (properties)
                    for (var keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                        if (properties[keys[i]] != null)
                            this[keys[i]] = properties[keys[i]];
            }
    
            NetParameter.prototype.name = "";
            NetParameter.prototype.input = $util.emptyArray;
            NetParameter.prototype.inputShape = $util.emptyArray;
            NetParameter.prototype.inputDim = $util.emptyArray;
            NetParameter.prototype.forceBackward = false;
            NetParameter.prototype.state = null;
            NetParameter.prototype.debugInfo = false;
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
                        if (!(message.inputShape && message.inputShape.length))
                            message.inputShape = [];
                        message.inputShape.push($root.caffe.BlobShape.decode(reader, reader.uint32()));
                        break;
                    case 4:
                        if (!(message.inputDim && message.inputDim.length))
                            message.inputDim = [];
                        if ((tag & 7) === 2) {
                            var end2 = reader.uint32() + reader.pos;
                            while (reader.pos < end2)
                                message.inputDim.push(reader.int32());
                        } else
                            message.inputDim.push(reader.int32());
                        break;
                    case 5:
                        message.forceBackward = reader.bool();
                        break;
                    case 6:
                        message.state = $root.caffe.NetState.decode(reader, reader.uint32());
                        break;
                    case 7:
                        message.debugInfo = reader.bool();
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
                if (message.inputShape != null && message.hasOwnProperty("inputShape")) {
                    if (!Array.isArray(message.inputShape))
                        return "inputShape: array expected";
                    for (var i = 0; i < message.inputShape.length; ++i) {
                        var error = $root.caffe.BlobShape.verify(message.inputShape[i]);
                        if (error)
                            return "inputShape." + error;
                    }
                }
                if (message.inputDim != null && message.hasOwnProperty("inputDim")) {
                    if (!Array.isArray(message.inputDim))
                        return "inputDim: array expected";
                    for (var i = 0; i < message.inputDim.length; ++i)
                        if (!$util.isInteger(message.inputDim[i]))
                            return "inputDim: integer[] expected";
                }
                if (message.forceBackward != null && message.hasOwnProperty("forceBackward"))
                    if (typeof message.forceBackward !== "boolean")
                        return "forceBackward: boolean expected";
                if (message.state != null && message.hasOwnProperty("state")) {
                    var error = $root.caffe.NetState.verify(message.state);
                    if (error)
                        return "state." + error;
                }
                if (message.debugInfo != null && message.hasOwnProperty("debugInfo"))
                    if (typeof message.debugInfo !== "boolean")
                        return "debugInfo: boolean expected";
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
                if (object.inputShape) {
                    if (!Array.isArray(object.inputShape))
                        throw TypeError(".caffe.NetParameter.inputShape: array expected");
                    message.inputShape = [];
                    for (var i = 0; i < object.inputShape.length; ++i) {
                        if (typeof object.inputShape[i] !== "object")
                            throw TypeError(".caffe.NetParameter.inputShape: object expected");
                        message.inputShape[i] = $root.caffe.BlobShape.fromObject(object.inputShape[i]);
                    }
                }
                if (object.inputDim) {
                    if (!Array.isArray(object.inputDim))
                        throw TypeError(".caffe.NetParameter.inputDim: array expected");
                    message.inputDim = [];
                    for (var i = 0; i < object.inputDim.length; ++i)
                        message.inputDim[i] = object.inputDim[i] | 0;
                }
                if (object.forceBackward != null)
                    message.forceBackward = Boolean(object.forceBackward);
                if (object.state != null) {
                    if (typeof object.state !== "object")
                        throw TypeError(".caffe.NetParameter.state: object expected");
                    message.state = $root.caffe.NetState.fromObject(object.state);
                }
                if (object.debugInfo != null)
                    message.debugInfo = Boolean(object.debugInfo);
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
                    object.inputDim = [];
                    object.inputShape = [];
                    object.layer = [];
                }
                if (options.defaults) {
                    object.name = "";
                    object.forceBackward = false;
                    object.state = null;
                    object.debugInfo = false;
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
                if (message.inputDim && message.inputDim.length) {
                    object.inputDim = [];
                    for (var j = 0; j < message.inputDim.length; ++j)
                        object.inputDim[j] = message.inputDim[j];
                }
                if (message.forceBackward != null && message.hasOwnProperty("forceBackward"))
                    object.forceBackward = message.forceBackward;
                if (message.state != null && message.hasOwnProperty("state"))
                    object.state = $root.caffe.NetState.toObject(message.state, options);
                if (message.debugInfo != null && message.hasOwnProperty("debugInfo"))
                    object.debugInfo = message.debugInfo;
                if (message.inputShape && message.inputShape.length) {
                    object.inputShape = [];
                    for (var j = 0; j < message.inputShape.length; ++j)
                        object.inputShape[j] = $root.caffe.BlobShape.toObject(message.inputShape[j], options);
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
                this.testNet = [];
                this.testNetParam = [];
                this.testState = [];
                this.testIter = [];
                this.stepvalue = [];
                this.weights = [];
                if (properties)
                    for (var keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                        if (properties[keys[i]] != null)
                            this[keys[i]] = properties[keys[i]];
            }
    
            SolverParameter.prototype.net = "";
            SolverParameter.prototype.netParam = null;
            SolverParameter.prototype.trainNet = "";
            SolverParameter.prototype.testNet = $util.emptyArray;
            SolverParameter.prototype.trainNetParam = null;
            SolverParameter.prototype.testNetParam = $util.emptyArray;
            SolverParameter.prototype.trainState = null;
            SolverParameter.prototype.testState = $util.emptyArray;
            SolverParameter.prototype.testIter = $util.emptyArray;
            SolverParameter.prototype.testInterval = 0;
            SolverParameter.prototype.testComputeLoss = false;
            SolverParameter.prototype.testInitialization = true;
            SolverParameter.prototype.baseLr = 0;
            SolverParameter.prototype.display = 0;
            SolverParameter.prototype.averageLoss = 1;
            SolverParameter.prototype.maxIter = 0;
            SolverParameter.prototype.iterSize = 1;
            SolverParameter.prototype.lrPolicy = "";
            SolverParameter.prototype.gamma = 0;
            SolverParameter.prototype.power = 0;
            SolverParameter.prototype.momentum = 0;
            SolverParameter.prototype.weightDecay = 0;
            SolverParameter.prototype.regularizationType = "L2";
            SolverParameter.prototype.stepsize = 0;
            SolverParameter.prototype.stepvalue = $util.emptyArray;
            SolverParameter.prototype.clipGradients = -1;
            SolverParameter.prototype.snapshot = 0;
            SolverParameter.prototype.snapshotPrefix = "";
            SolverParameter.prototype.snapshotDiff = false;
            SolverParameter.prototype.snapshotFormat = 1;
            SolverParameter.prototype.solverMode = 1;
            SolverParameter.prototype.deviceId = 0;
            SolverParameter.prototype.randomSeed = $util.Long ? $util.Long.fromBits(-1,-1,false) : -1;
            SolverParameter.prototype.type = "SGD";
            SolverParameter.prototype.delta = 1e-8;
            SolverParameter.prototype.momentum2 = 0.999;
            SolverParameter.prototype.rmsDecay = 0.99;
            SolverParameter.prototype.debugInfo = false;
            SolverParameter.prototype.snapshotAfterTrain = true;
            SolverParameter.prototype.solverType = 0;
            SolverParameter.prototype.layerWiseReduce = true;
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
                        message.netParam = $root.caffe.NetParameter.decode(reader, reader.uint32());
                        break;
                    case 1:
                        message.trainNet = reader.string();
                        break;
                    case 2:
                        if (!(message.testNet && message.testNet.length))
                            message.testNet = [];
                        message.testNet.push(reader.string());
                        break;
                    case 21:
                        message.trainNetParam = $root.caffe.NetParameter.decode(reader, reader.uint32());
                        break;
                    case 22:
                        if (!(message.testNetParam && message.testNetParam.length))
                            message.testNetParam = [];
                        message.testNetParam.push($root.caffe.NetParameter.decode(reader, reader.uint32()));
                        break;
                    case 26:
                        message.trainState = $root.caffe.NetState.decode(reader, reader.uint32());
                        break;
                    case 27:
                        if (!(message.testState && message.testState.length))
                            message.testState = [];
                        message.testState.push($root.caffe.NetState.decode(reader, reader.uint32()));
                        break;
                    case 3:
                        if (!(message.testIter && message.testIter.length))
                            message.testIter = [];
                        if ((tag & 7) === 2) {
                            var end2 = reader.uint32() + reader.pos;
                            while (reader.pos < end2)
                                message.testIter.push(reader.int32());
                        } else
                            message.testIter.push(reader.int32());
                        break;
                    case 4:
                        message.testInterval = reader.int32();
                        break;
                    case 19:
                        message.testComputeLoss = reader.bool();
                        break;
                    case 32:
                        message.testInitialization = reader.bool();
                        break;
                    case 5:
                        message.baseLr = reader.float();
                        break;
                    case 6:
                        message.display = reader.int32();
                        break;
                    case 33:
                        message.averageLoss = reader.int32();
                        break;
                    case 7:
                        message.maxIter = reader.int32();
                        break;
                    case 36:
                        message.iterSize = reader.int32();
                        break;
                    case 8:
                        message.lrPolicy = reader.string();
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
                        message.weightDecay = reader.float();
                        break;
                    case 29:
                        message.regularizationType = reader.string();
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
                        message.clipGradients = reader.float();
                        break;
                    case 14:
                        message.snapshot = reader.int32();
                        break;
                    case 15:
                        message.snapshotPrefix = reader.string();
                        break;
                    case 16:
                        message.snapshotDiff = reader.bool();
                        break;
                    case 37:
                        message.snapshotFormat = reader.int32();
                        break;
                    case 17:
                        message.solverMode = reader.int32();
                        break;
                    case 18:
                        message.deviceId = reader.int32();
                        break;
                    case 20:
                        message.randomSeed = reader.int64();
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
                        message.rmsDecay = reader.float();
                        break;
                    case 23:
                        message.debugInfo = reader.bool();
                        break;
                    case 28:
                        message.snapshotAfterTrain = reader.bool();
                        break;
                    case 30:
                        message.solverType = reader.int32();
                        break;
                    case 41:
                        message.layerWiseReduce = reader.bool();
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
    
            SolverParameter.verify = function verify(message) {
                if (typeof message !== "object" || message === null)
                    return "object expected";
                if (message.net != null && message.hasOwnProperty("net"))
                    if (!$util.isString(message.net))
                        return "net: string expected";
                if (message.netParam != null && message.hasOwnProperty("netParam")) {
                    var error = $root.caffe.NetParameter.verify(message.netParam);
                    if (error)
                        return "netParam." + error;
                }
                if (message.trainNet != null && message.hasOwnProperty("trainNet"))
                    if (!$util.isString(message.trainNet))
                        return "trainNet: string expected";
                if (message.testNet != null && message.hasOwnProperty("testNet")) {
                    if (!Array.isArray(message.testNet))
                        return "testNet: array expected";
                    for (var i = 0; i < message.testNet.length; ++i)
                        if (!$util.isString(message.testNet[i]))
                            return "testNet: string[] expected";
                }
                if (message.trainNetParam != null && message.hasOwnProperty("trainNetParam")) {
                    var error = $root.caffe.NetParameter.verify(message.trainNetParam);
                    if (error)
                        return "trainNetParam." + error;
                }
                if (message.testNetParam != null && message.hasOwnProperty("testNetParam")) {
                    if (!Array.isArray(message.testNetParam))
                        return "testNetParam: array expected";
                    for (var i = 0; i < message.testNetParam.length; ++i) {
                        var error = $root.caffe.NetParameter.verify(message.testNetParam[i]);
                        if (error)
                            return "testNetParam." + error;
                    }
                }
                if (message.trainState != null && message.hasOwnProperty("trainState")) {
                    var error = $root.caffe.NetState.verify(message.trainState);
                    if (error)
                        return "trainState." + error;
                }
                if (message.testState != null && message.hasOwnProperty("testState")) {
                    if (!Array.isArray(message.testState))
                        return "testState: array expected";
                    for (var i = 0; i < message.testState.length; ++i) {
                        var error = $root.caffe.NetState.verify(message.testState[i]);
                        if (error)
                            return "testState." + error;
                    }
                }
                if (message.testIter != null && message.hasOwnProperty("testIter")) {
                    if (!Array.isArray(message.testIter))
                        return "testIter: array expected";
                    for (var i = 0; i < message.testIter.length; ++i)
                        if (!$util.isInteger(message.testIter[i]))
                            return "testIter: integer[] expected";
                }
                if (message.testInterval != null && message.hasOwnProperty("testInterval"))
                    if (!$util.isInteger(message.testInterval))
                        return "testInterval: integer expected";
                if (message.testComputeLoss != null && message.hasOwnProperty("testComputeLoss"))
                    if (typeof message.testComputeLoss !== "boolean")
                        return "testComputeLoss: boolean expected";
                if (message.testInitialization != null && message.hasOwnProperty("testInitialization"))
                    if (typeof message.testInitialization !== "boolean")
                        return "testInitialization: boolean expected";
                if (message.baseLr != null && message.hasOwnProperty("baseLr"))
                    if (typeof message.baseLr !== "number")
                        return "baseLr: number expected";
                if (message.display != null && message.hasOwnProperty("display"))
                    if (!$util.isInteger(message.display))
                        return "display: integer expected";
                if (message.averageLoss != null && message.hasOwnProperty("averageLoss"))
                    if (!$util.isInteger(message.averageLoss))
                        return "averageLoss: integer expected";
                if (message.maxIter != null && message.hasOwnProperty("maxIter"))
                    if (!$util.isInteger(message.maxIter))
                        return "maxIter: integer expected";
                if (message.iterSize != null && message.hasOwnProperty("iterSize"))
                    if (!$util.isInteger(message.iterSize))
                        return "iterSize: integer expected";
                if (message.lrPolicy != null && message.hasOwnProperty("lrPolicy"))
                    if (!$util.isString(message.lrPolicy))
                        return "lrPolicy: string expected";
                if (message.gamma != null && message.hasOwnProperty("gamma"))
                    if (typeof message.gamma !== "number")
                        return "gamma: number expected";
                if (message.power != null && message.hasOwnProperty("power"))
                    if (typeof message.power !== "number")
                        return "power: number expected";
                if (message.momentum != null && message.hasOwnProperty("momentum"))
                    if (typeof message.momentum !== "number")
                        return "momentum: number expected";
                if (message.weightDecay != null && message.hasOwnProperty("weightDecay"))
                    if (typeof message.weightDecay !== "number")
                        return "weightDecay: number expected";
                if (message.regularizationType != null && message.hasOwnProperty("regularizationType"))
                    if (!$util.isString(message.regularizationType))
                        return "regularizationType: string expected";
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
                if (message.clipGradients != null && message.hasOwnProperty("clipGradients"))
                    if (typeof message.clipGradients !== "number")
                        return "clipGradients: number expected";
                if (message.snapshot != null && message.hasOwnProperty("snapshot"))
                    if (!$util.isInteger(message.snapshot))
                        return "snapshot: integer expected";
                if (message.snapshotPrefix != null && message.hasOwnProperty("snapshotPrefix"))
                    if (!$util.isString(message.snapshotPrefix))
                        return "snapshotPrefix: string expected";
                if (message.snapshotDiff != null && message.hasOwnProperty("snapshotDiff"))
                    if (typeof message.snapshotDiff !== "boolean")
                        return "snapshotDiff: boolean expected";
                if (message.snapshotFormat != null && message.hasOwnProperty("snapshotFormat"))
                    switch (message.snapshotFormat) {
                    default:
                        return "snapshotFormat: enum value expected";
                    case 0:
                    case 1:
                        break;
                    }
                if (message.solverMode != null && message.hasOwnProperty("solverMode"))
                    switch (message.solverMode) {
                    default:
                        return "solverMode: enum value expected";
                    case 0:
                    case 1:
                        break;
                    }
                if (message.deviceId != null && message.hasOwnProperty("deviceId"))
                    if (!$util.isInteger(message.deviceId))
                        return "deviceId: integer expected";
                if (message.randomSeed != null && message.hasOwnProperty("randomSeed"))
                    if (!$util.isInteger(message.randomSeed) && !(message.randomSeed && $util.isInteger(message.randomSeed.low) && $util.isInteger(message.randomSeed.high)))
                        return "randomSeed: integer|Long expected";
                if (message.type != null && message.hasOwnProperty("type"))
                    if (!$util.isString(message.type))
                        return "type: string expected";
                if (message.delta != null && message.hasOwnProperty("delta"))
                    if (typeof message.delta !== "number")
                        return "delta: number expected";
                if (message.momentum2 != null && message.hasOwnProperty("momentum2"))
                    if (typeof message.momentum2 !== "number")
                        return "momentum2: number expected";
                if (message.rmsDecay != null && message.hasOwnProperty("rmsDecay"))
                    if (typeof message.rmsDecay !== "number")
                        return "rmsDecay: number expected";
                if (message.debugInfo != null && message.hasOwnProperty("debugInfo"))
                    if (typeof message.debugInfo !== "boolean")
                        return "debugInfo: boolean expected";
                if (message.snapshotAfterTrain != null && message.hasOwnProperty("snapshotAfterTrain"))
                    if (typeof message.snapshotAfterTrain !== "boolean")
                        return "snapshotAfterTrain: boolean expected";
                if (message.solverType != null && message.hasOwnProperty("solverType"))
                    switch (message.solverType) {
                    default:
                        return "solverType: enum value expected";
                    case 0:
                    case 1:
                    case 2:
                    case 3:
                    case 4:
                    case 5:
                        break;
                    }
                if (message.layerWiseReduce != null && message.hasOwnProperty("layerWiseReduce"))
                    if (typeof message.layerWiseReduce !== "boolean")
                        return "layerWiseReduce: boolean expected";
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
                if (object.netParam != null) {
                    if (typeof object.netParam !== "object")
                        throw TypeError(".caffe.SolverParameter.netParam: object expected");
                    message.netParam = $root.caffe.NetParameter.fromObject(object.netParam);
                }
                if (object.trainNet != null)
                    message.trainNet = String(object.trainNet);
                if (object.testNet) {
                    if (!Array.isArray(object.testNet))
                        throw TypeError(".caffe.SolverParameter.testNet: array expected");
                    message.testNet = [];
                    for (var i = 0; i < object.testNet.length; ++i)
                        message.testNet[i] = String(object.testNet[i]);
                }
                if (object.trainNetParam != null) {
                    if (typeof object.trainNetParam !== "object")
                        throw TypeError(".caffe.SolverParameter.trainNetParam: object expected");
                    message.trainNetParam = $root.caffe.NetParameter.fromObject(object.trainNetParam);
                }
                if (object.testNetParam) {
                    if (!Array.isArray(object.testNetParam))
                        throw TypeError(".caffe.SolverParameter.testNetParam: array expected");
                    message.testNetParam = [];
                    for (var i = 0; i < object.testNetParam.length; ++i) {
                        if (typeof object.testNetParam[i] !== "object")
                            throw TypeError(".caffe.SolverParameter.testNetParam: object expected");
                        message.testNetParam[i] = $root.caffe.NetParameter.fromObject(object.testNetParam[i]);
                    }
                }
                if (object.trainState != null) {
                    if (typeof object.trainState !== "object")
                        throw TypeError(".caffe.SolverParameter.trainState: object expected");
                    message.trainState = $root.caffe.NetState.fromObject(object.trainState);
                }
                if (object.testState) {
                    if (!Array.isArray(object.testState))
                        throw TypeError(".caffe.SolverParameter.testState: array expected");
                    message.testState = [];
                    for (var i = 0; i < object.testState.length; ++i) {
                        if (typeof object.testState[i] !== "object")
                            throw TypeError(".caffe.SolverParameter.testState: object expected");
                        message.testState[i] = $root.caffe.NetState.fromObject(object.testState[i]);
                    }
                }
                if (object.testIter) {
                    if (!Array.isArray(object.testIter))
                        throw TypeError(".caffe.SolverParameter.testIter: array expected");
                    message.testIter = [];
                    for (var i = 0; i < object.testIter.length; ++i)
                        message.testIter[i] = object.testIter[i] | 0;
                }
                if (object.testInterval != null)
                    message.testInterval = object.testInterval | 0;
                if (object.testComputeLoss != null)
                    message.testComputeLoss = Boolean(object.testComputeLoss);
                if (object.testInitialization != null)
                    message.testInitialization = Boolean(object.testInitialization);
                if (object.baseLr != null)
                    message.baseLr = Number(object.baseLr);
                if (object.display != null)
                    message.display = object.display | 0;
                if (object.averageLoss != null)
                    message.averageLoss = object.averageLoss | 0;
                if (object.maxIter != null)
                    message.maxIter = object.maxIter | 0;
                if (object.iterSize != null)
                    message.iterSize = object.iterSize | 0;
                if (object.lrPolicy != null)
                    message.lrPolicy = String(object.lrPolicy);
                if (object.gamma != null)
                    message.gamma = Number(object.gamma);
                if (object.power != null)
                    message.power = Number(object.power);
                if (object.momentum != null)
                    message.momentum = Number(object.momentum);
                if (object.weightDecay != null)
                    message.weightDecay = Number(object.weightDecay);
                if (object.regularizationType != null)
                    message.regularizationType = String(object.regularizationType);
                if (object.stepsize != null)
                    message.stepsize = object.stepsize | 0;
                if (object.stepvalue) {
                    if (!Array.isArray(object.stepvalue))
                        throw TypeError(".caffe.SolverParameter.stepvalue: array expected");
                    message.stepvalue = [];
                    for (var i = 0; i < object.stepvalue.length; ++i)
                        message.stepvalue[i] = object.stepvalue[i] | 0;
                }
                if (object.clipGradients != null)
                    message.clipGradients = Number(object.clipGradients);
                if (object.snapshot != null)
                    message.snapshot = object.snapshot | 0;
                if (object.snapshotPrefix != null)
                    message.snapshotPrefix = String(object.snapshotPrefix);
                if (object.snapshotDiff != null)
                    message.snapshotDiff = Boolean(object.snapshotDiff);
                switch (object.snapshotFormat) {
                case "HDF5":
                case 0:
                    message.snapshotFormat = 0;
                    break;
                case "BINARYPROTO":
                case 1:
                    message.snapshotFormat = 1;
                    break;
                }
                switch (object.solverMode) {
                case "CPU":
                case 0:
                    message.solverMode = 0;
                    break;
                case "GPU":
                case 1:
                    message.solverMode = 1;
                    break;
                }
                if (object.deviceId != null)
                    message.deviceId = object.deviceId | 0;
                if (object.randomSeed != null)
                    if ($util.Long)
                        (message.randomSeed = $util.Long.fromValue(object.randomSeed)).unsigned = false;
                    else if (typeof object.randomSeed === "string")
                        message.randomSeed = parseInt(object.randomSeed, 10);
                    else if (typeof object.randomSeed === "number")
                        message.randomSeed = object.randomSeed;
                    else if (typeof object.randomSeed === "object")
                        message.randomSeed = new $util.LongBits(object.randomSeed.low >>> 0, object.randomSeed.high >>> 0).toNumber();
                if (object.type != null)
                    message.type = String(object.type);
                if (object.delta != null)
                    message.delta = Number(object.delta);
                if (object.momentum2 != null)
                    message.momentum2 = Number(object.momentum2);
                if (object.rmsDecay != null)
                    message.rmsDecay = Number(object.rmsDecay);
                if (object.debugInfo != null)
                    message.debugInfo = Boolean(object.debugInfo);
                if (object.snapshotAfterTrain != null)
                    message.snapshotAfterTrain = Boolean(object.snapshotAfterTrain);
                switch (object.solverType) {
                case "SGD":
                case 0:
                    message.solverType = 0;
                    break;
                case "NESTEROV":
                case 1:
                    message.solverType = 1;
                    break;
                case "ADAGRAD":
                case 2:
                    message.solverType = 2;
                    break;
                case "RMSPROP":
                case 3:
                    message.solverType = 3;
                    break;
                case "ADADELTA":
                case 4:
                    message.solverType = 4;
                    break;
                case "ADAM":
                case 5:
                    message.solverType = 5;
                    break;
                }
                if (object.layerWiseReduce != null)
                    message.layerWiseReduce = Boolean(object.layerWiseReduce);
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
                    object.testNet = [];
                    object.testIter = [];
                    object.testNetParam = [];
                    object.testState = [];
                    object.stepvalue = [];
                    object.weights = [];
                }
                if (options.defaults) {
                    object.trainNet = "";
                    object.testInterval = 0;
                    object.baseLr = 0;
                    object.display = 0;
                    object.maxIter = 0;
                    object.lrPolicy = "";
                    object.gamma = 0;
                    object.power = 0;
                    object.momentum = 0;
                    object.weightDecay = 0;
                    object.stepsize = 0;
                    object.snapshot = 0;
                    object.snapshotPrefix = "";
                    object.snapshotDiff = false;
                    object.solverMode = options.enums === String ? "GPU" : 1;
                    object.deviceId = 0;
                    object.testComputeLoss = false;
                    if ($util.Long) {
                        var long = new $util.Long(-1, -1, false);
                        object.randomSeed = options.longs === String ? long.toString() : options.longs === Number ? long.toNumber() : long;
                    } else
                        object.randomSeed = options.longs === String ? "-1" : -1;
                    object.trainNetParam = null;
                    object.debugInfo = false;
                    object.net = "";
                    object.netParam = null;
                    object.trainState = null;
                    object.snapshotAfterTrain = true;
                    object.regularizationType = "L2";
                    object.solverType = options.enums === String ? "SGD" : 0;
                    object.delta = 1e-8;
                    object.testInitialization = true;
                    object.averageLoss = 1;
                    object.clipGradients = -1;
                    object.iterSize = 1;
                    object.snapshotFormat = options.enums === String ? "BINARYPROTO" : 1;
                    object.rmsDecay = 0.99;
                    object.momentum2 = 0.999;
                    object.type = "SGD";
                    object.layerWiseReduce = true;
                }
                if (message.trainNet != null && message.hasOwnProperty("trainNet"))
                    object.trainNet = message.trainNet;
                if (message.testNet && message.testNet.length) {
                    object.testNet = [];
                    for (var j = 0; j < message.testNet.length; ++j)
                        object.testNet[j] = message.testNet[j];
                }
                if (message.testIter && message.testIter.length) {
                    object.testIter = [];
                    for (var j = 0; j < message.testIter.length; ++j)
                        object.testIter[j] = message.testIter[j];
                }
                if (message.testInterval != null && message.hasOwnProperty("testInterval"))
                    object.testInterval = message.testInterval;
                if (message.baseLr != null && message.hasOwnProperty("baseLr"))
                    object.baseLr = options.json && !isFinite(message.baseLr) ? String(message.baseLr) : message.baseLr;
                if (message.display != null && message.hasOwnProperty("display"))
                    object.display = message.display;
                if (message.maxIter != null && message.hasOwnProperty("maxIter"))
                    object.maxIter = message.maxIter;
                if (message.lrPolicy != null && message.hasOwnProperty("lrPolicy"))
                    object.lrPolicy = message.lrPolicy;
                if (message.gamma != null && message.hasOwnProperty("gamma"))
                    object.gamma = options.json && !isFinite(message.gamma) ? String(message.gamma) : message.gamma;
                if (message.power != null && message.hasOwnProperty("power"))
                    object.power = options.json && !isFinite(message.power) ? String(message.power) : message.power;
                if (message.momentum != null && message.hasOwnProperty("momentum"))
                    object.momentum = options.json && !isFinite(message.momentum) ? String(message.momentum) : message.momentum;
                if (message.weightDecay != null && message.hasOwnProperty("weightDecay"))
                    object.weightDecay = options.json && !isFinite(message.weightDecay) ? String(message.weightDecay) : message.weightDecay;
                if (message.stepsize != null && message.hasOwnProperty("stepsize"))
                    object.stepsize = message.stepsize;
                if (message.snapshot != null && message.hasOwnProperty("snapshot"))
                    object.snapshot = message.snapshot;
                if (message.snapshotPrefix != null && message.hasOwnProperty("snapshotPrefix"))
                    object.snapshotPrefix = message.snapshotPrefix;
                if (message.snapshotDiff != null && message.hasOwnProperty("snapshotDiff"))
                    object.snapshotDiff = message.snapshotDiff;
                if (message.solverMode != null && message.hasOwnProperty("solverMode"))
                    object.solverMode = options.enums === String ? $root.caffe.SolverParameter.SolverMode[message.solverMode] : message.solverMode;
                if (message.deviceId != null && message.hasOwnProperty("deviceId"))
                    object.deviceId = message.deviceId;
                if (message.testComputeLoss != null && message.hasOwnProperty("testComputeLoss"))
                    object.testComputeLoss = message.testComputeLoss;
                if (message.randomSeed != null && message.hasOwnProperty("randomSeed"))
                    if (typeof message.randomSeed === "number")
                        object.randomSeed = options.longs === String ? String(message.randomSeed) : message.randomSeed;
                    else
                        object.randomSeed = options.longs === String ? $util.Long.prototype.toString.call(message.randomSeed) : options.longs === Number ? new $util.LongBits(message.randomSeed.low >>> 0, message.randomSeed.high >>> 0).toNumber() : message.randomSeed;
                if (message.trainNetParam != null && message.hasOwnProperty("trainNetParam"))
                    object.trainNetParam = $root.caffe.NetParameter.toObject(message.trainNetParam, options);
                if (message.testNetParam && message.testNetParam.length) {
                    object.testNetParam = [];
                    for (var j = 0; j < message.testNetParam.length; ++j)
                        object.testNetParam[j] = $root.caffe.NetParameter.toObject(message.testNetParam[j], options);
                }
                if (message.debugInfo != null && message.hasOwnProperty("debugInfo"))
                    object.debugInfo = message.debugInfo;
                if (message.net != null && message.hasOwnProperty("net"))
                    object.net = message.net;
                if (message.netParam != null && message.hasOwnProperty("netParam"))
                    object.netParam = $root.caffe.NetParameter.toObject(message.netParam, options);
                if (message.trainState != null && message.hasOwnProperty("trainState"))
                    object.trainState = $root.caffe.NetState.toObject(message.trainState, options);
                if (message.testState && message.testState.length) {
                    object.testState = [];
                    for (var j = 0; j < message.testState.length; ++j)
                        object.testState[j] = $root.caffe.NetState.toObject(message.testState[j], options);
                }
                if (message.snapshotAfterTrain != null && message.hasOwnProperty("snapshotAfterTrain"))
                    object.snapshotAfterTrain = message.snapshotAfterTrain;
                if (message.regularizationType != null && message.hasOwnProperty("regularizationType"))
                    object.regularizationType = message.regularizationType;
                if (message.solverType != null && message.hasOwnProperty("solverType"))
                    object.solverType = options.enums === String ? $root.caffe.SolverParameter.SolverType[message.solverType] : message.solverType;
                if (message.delta != null && message.hasOwnProperty("delta"))
                    object.delta = options.json && !isFinite(message.delta) ? String(message.delta) : message.delta;
                if (message.testInitialization != null && message.hasOwnProperty("testInitialization"))
                    object.testInitialization = message.testInitialization;
                if (message.averageLoss != null && message.hasOwnProperty("averageLoss"))
                    object.averageLoss = message.averageLoss;
                if (message.stepvalue && message.stepvalue.length) {
                    object.stepvalue = [];
                    for (var j = 0; j < message.stepvalue.length; ++j)
                        object.stepvalue[j] = message.stepvalue[j];
                }
                if (message.clipGradients != null && message.hasOwnProperty("clipGradients"))
                    object.clipGradients = options.json && !isFinite(message.clipGradients) ? String(message.clipGradients) : message.clipGradients;
                if (message.iterSize != null && message.hasOwnProperty("iterSize"))
                    object.iterSize = message.iterSize;
                if (message.snapshotFormat != null && message.hasOwnProperty("snapshotFormat"))
                    object.snapshotFormat = options.enums === String ? $root.caffe.SolverParameter.SnapshotFormat[message.snapshotFormat] : message.snapshotFormat;
                if (message.rmsDecay != null && message.hasOwnProperty("rmsDecay"))
                    object.rmsDecay = options.json && !isFinite(message.rmsDecay) ? String(message.rmsDecay) : message.rmsDecay;
                if (message.momentum2 != null && message.hasOwnProperty("momentum2"))
                    object.momentum2 = options.json && !isFinite(message.momentum2) ? String(message.momentum2) : message.momentum2;
                if (message.type != null && message.hasOwnProperty("type"))
                    object.type = message.type;
                if (message.layerWiseReduce != null && message.hasOwnProperty("layerWiseReduce"))
                    object.layerWiseReduce = message.layerWiseReduce;
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
            SolverState.prototype.learnedNet = "";
            SolverState.prototype.history = $util.emptyArray;
            SolverState.prototype.currentStep = 0;
    
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
                        message.learnedNet = reader.string();
                        break;
                    case 3:
                        if (!(message.history && message.history.length))
                            message.history = [];
                        message.history.push($root.caffe.BlobProto.decode(reader, reader.uint32()));
                        break;
                    case 4:
                        message.currentStep = reader.int32();
                        break;
                    default:
                        reader.skipType(tag & 7);
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
                if (message.learnedNet != null && message.hasOwnProperty("learnedNet"))
                    if (!$util.isString(message.learnedNet))
                        return "learnedNet: string expected";
                if (message.history != null && message.hasOwnProperty("history")) {
                    if (!Array.isArray(message.history))
                        return "history: array expected";
                    for (var i = 0; i < message.history.length; ++i) {
                        var error = $root.caffe.BlobProto.verify(message.history[i]);
                        if (error)
                            return "history." + error;
                    }
                }
                if (message.currentStep != null && message.hasOwnProperty("currentStep"))
                    if (!$util.isInteger(message.currentStep))
                        return "currentStep: integer expected";
                return null;
            };
    
            SolverState.fromObject = function fromObject(object) {
                if (object instanceof $root.caffe.SolverState)
                    return object;
                var message = new $root.caffe.SolverState();
                if (object.iter != null)
                    message.iter = object.iter | 0;
                if (object.learnedNet != null)
                    message.learnedNet = String(object.learnedNet);
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
                if (object.currentStep != null)
                    message.currentStep = object.currentStep | 0;
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
                    object.learnedNet = "";
                    object.currentStep = 0;
                }
                if (message.iter != null && message.hasOwnProperty("iter"))
                    object.iter = message.iter;
                if (message.learnedNet != null && message.hasOwnProperty("learnedNet"))
                    object.learnedNet = message.learnedNet;
                if (message.history && message.history.length) {
                    object.history = [];
                    for (var j = 0; j < message.history.length; ++j)
                        object.history[j] = $root.caffe.BlobProto.toObject(message.history[j], options);
                }
                if (message.currentStep != null && message.hasOwnProperty("currentStep"))
                    object.currentStep = message.currentStep;
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
                this.notStage = [];
                if (properties)
                    for (var keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                        if (properties[keys[i]] != null)
                            this[keys[i]] = properties[keys[i]];
            }
    
            NetStateRule.prototype.phase = 0;
            NetStateRule.prototype.minLevel = 0;
            NetStateRule.prototype.maxLevel = 0;
            NetStateRule.prototype.stage = $util.emptyArray;
            NetStateRule.prototype.notStage = $util.emptyArray;
    
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
                        message.minLevel = reader.int32();
                        break;
                    case 3:
                        message.maxLevel = reader.int32();
                        break;
                    case 4:
                        if (!(message.stage && message.stage.length))
                            message.stage = [];
                        message.stage.push(reader.string());
                        break;
                    case 5:
                        if (!(message.notStage && message.notStage.length))
                            message.notStage = [];
                        message.notStage.push(reader.string());
                        break;
                    default:
                        reader.skipType(tag & 7);
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
                if (message.minLevel != null && message.hasOwnProperty("minLevel"))
                    if (!$util.isInteger(message.minLevel))
                        return "minLevel: integer expected";
                if (message.maxLevel != null && message.hasOwnProperty("maxLevel"))
                    if (!$util.isInteger(message.maxLevel))
                        return "maxLevel: integer expected";
                if (message.stage != null && message.hasOwnProperty("stage")) {
                    if (!Array.isArray(message.stage))
                        return "stage: array expected";
                    for (var i = 0; i < message.stage.length; ++i)
                        if (!$util.isString(message.stage[i]))
                            return "stage: string[] expected";
                }
                if (message.notStage != null && message.hasOwnProperty("notStage")) {
                    if (!Array.isArray(message.notStage))
                        return "notStage: array expected";
                    for (var i = 0; i < message.notStage.length; ++i)
                        if (!$util.isString(message.notStage[i]))
                            return "notStage: string[] expected";
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
                if (object.minLevel != null)
                    message.minLevel = object.minLevel | 0;
                if (object.maxLevel != null)
                    message.maxLevel = object.maxLevel | 0;
                if (object.stage) {
                    if (!Array.isArray(object.stage))
                        throw TypeError(".caffe.NetStateRule.stage: array expected");
                    message.stage = [];
                    for (var i = 0; i < object.stage.length; ++i)
                        message.stage[i] = String(object.stage[i]);
                }
                if (object.notStage) {
                    if (!Array.isArray(object.notStage))
                        throw TypeError(".caffe.NetStateRule.notStage: array expected");
                    message.notStage = [];
                    for (var i = 0; i < object.notStage.length; ++i)
                        message.notStage[i] = String(object.notStage[i]);
                }
                return message;
            };
    
            NetStateRule.toObject = function toObject(message, options) {
                if (!options)
                    options = {};
                var object = {};
                if (options.arrays || options.defaults) {
                    object.stage = [];
                    object.notStage = [];
                }
                if (options.defaults) {
                    object.phase = options.enums === String ? "TRAIN" : 0;
                    object.minLevel = 0;
                    object.maxLevel = 0;
                }
                if (message.phase != null && message.hasOwnProperty("phase"))
                    object.phase = options.enums === String ? $root.caffe.Phase[message.phase] : message.phase;
                if (message.minLevel != null && message.hasOwnProperty("minLevel"))
                    object.minLevel = message.minLevel;
                if (message.maxLevel != null && message.hasOwnProperty("maxLevel"))
                    object.maxLevel = message.maxLevel;
                if (message.stage && message.stage.length) {
                    object.stage = [];
                    for (var j = 0; j < message.stage.length; ++j)
                        object.stage[j] = message.stage[j];
                }
                if (message.notStage && message.notStage.length) {
                    object.notStage = [];
                    for (var j = 0; j < message.notStage.length; ++j)
                        object.notStage[j] = message.notStage[j];
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
            ParamSpec.prototype.shareMode = 0;
            ParamSpec.prototype.lrMult = 1;
            ParamSpec.prototype.decayMult = 1;
    
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
                        message.shareMode = reader.int32();
                        break;
                    case 3:
                        message.lrMult = reader.float();
                        break;
                    case 4:
                        message.decayMult = reader.float();
                        break;
                    default:
                        reader.skipType(tag & 7);
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
                if (message.shareMode != null && message.hasOwnProperty("shareMode"))
                    switch (message.shareMode) {
                    default:
                        return "shareMode: enum value expected";
                    case 0:
                    case 1:
                        break;
                    }
                if (message.lrMult != null && message.hasOwnProperty("lrMult"))
                    if (typeof message.lrMult !== "number")
                        return "lrMult: number expected";
                if (message.decayMult != null && message.hasOwnProperty("decayMult"))
                    if (typeof message.decayMult !== "number")
                        return "decayMult: number expected";
                return null;
            };
    
            ParamSpec.fromObject = function fromObject(object) {
                if (object instanceof $root.caffe.ParamSpec)
                    return object;
                var message = new $root.caffe.ParamSpec();
                if (object.name != null)
                    message.name = String(object.name);
                switch (object.shareMode) {
                case "STRICT":
                case 0:
                    message.shareMode = 0;
                    break;
                case "PERMISSIVE":
                case 1:
                    message.shareMode = 1;
                    break;
                }
                if (object.lrMult != null)
                    message.lrMult = Number(object.lrMult);
                if (object.decayMult != null)
                    message.decayMult = Number(object.decayMult);
                return message;
            };
    
            ParamSpec.toObject = function toObject(message, options) {
                if (!options)
                    options = {};
                var object = {};
                if (options.defaults) {
                    object.name = "";
                    object.shareMode = options.enums === String ? "STRICT" : 0;
                    object.lrMult = 1;
                    object.decayMult = 1;
                }
                if (message.name != null && message.hasOwnProperty("name"))
                    object.name = message.name;
                if (message.shareMode != null && message.hasOwnProperty("shareMode"))
                    object.shareMode = options.enums === String ? $root.caffe.ParamSpec.DimCheckMode[message.shareMode] : message.shareMode;
                if (message.lrMult != null && message.hasOwnProperty("lrMult"))
                    object.lrMult = options.json && !isFinite(message.lrMult) ? String(message.lrMult) : message.lrMult;
                if (message.decayMult != null && message.hasOwnProperty("decayMult"))
                    object.decayMult = options.json && !isFinite(message.decayMult) ? String(message.decayMult) : message.decayMult;
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
                this.lossWeight = [];
                this.param = [];
                this.blobs = [];
                this.propagateDown = [];
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
            LayerParameter.prototype.lossWeight = $util.emptyArray;
            LayerParameter.prototype.param = $util.emptyArray;
            LayerParameter.prototype.blobs = $util.emptyArray;
            LayerParameter.prototype.propagateDown = $util.emptyArray;
            LayerParameter.prototype.include = $util.emptyArray;
            LayerParameter.prototype.exclude = $util.emptyArray;
            LayerParameter.prototype.transformParam = null;
            LayerParameter.prototype.lossParam = null;
            LayerParameter.prototype.accuracyParam = null;
            LayerParameter.prototype.argmaxParam = null;
            LayerParameter.prototype.batchNormParam = null;
            LayerParameter.prototype.biasParam = null;
            LayerParameter.prototype.clipParam = null;
            LayerParameter.prototype.concatParam = null;
            LayerParameter.prototype.contrastiveLossParam = null;
            LayerParameter.prototype.convolutionParam = null;
            LayerParameter.prototype.cropParam = null;
            LayerParameter.prototype.dataParam = null;
            LayerParameter.prototype.dropoutParam = null;
            LayerParameter.prototype.dummyDataParam = null;
            LayerParameter.prototype.eltwiseParam = null;
            LayerParameter.prototype.eluParam = null;
            LayerParameter.prototype.embedParam = null;
            LayerParameter.prototype.expParam = null;
            LayerParameter.prototype.flattenParam = null;
            LayerParameter.prototype.hdf5DataParam = null;
            LayerParameter.prototype.hdf5OutputParam = null;
            LayerParameter.prototype.hingeLossParam = null;
            LayerParameter.prototype.imageDataParam = null;
            LayerParameter.prototype.infogainLossParam = null;
            LayerParameter.prototype.innerProductParam = null;
            LayerParameter.prototype.inputParam = null;
            LayerParameter.prototype.logParam = null;
            LayerParameter.prototype.lrnParam = null;
            LayerParameter.prototype.memoryDataParam = null;
            LayerParameter.prototype.mvnParam = null;
            LayerParameter.prototype.parameterParam = null;
            LayerParameter.prototype.poolingParam = null;
            LayerParameter.prototype.powerParam = null;
            LayerParameter.prototype.preluParam = null;
            LayerParameter.prototype.pythonParam = null;
            LayerParameter.prototype.recurrentParam = null;
            LayerParameter.prototype.reductionParam = null;
            LayerParameter.prototype.reluParam = null;
            LayerParameter.prototype.reshapeParam = null;
            LayerParameter.prototype.scaleParam = null;
            LayerParameter.prototype.sigmoidParam = null;
            LayerParameter.prototype.softmaxParam = null;
            LayerParameter.prototype.sppParam = null;
            LayerParameter.prototype.sliceParam = null;
            LayerParameter.prototype.swishParam = null;
            LayerParameter.prototype.tanhParam = null;
            LayerParameter.prototype.thresholdParam = null;
            LayerParameter.prototype.tileParam = null;
            LayerParameter.prototype.windowDataParam = null;
    
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
                        if (!(message.lossWeight && message.lossWeight.length))
                            message.lossWeight = [];
                        if ((tag & 7) === 2) {
                            var end2 = reader.uint32() + reader.pos;
                            while (reader.pos < end2)
                                message.lossWeight.push(reader.float());
                        } else
                            message.lossWeight.push(reader.float());
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
                        if (!(message.propagateDown && message.propagateDown.length))
                            message.propagateDown = [];
                        if ((tag & 7) === 2) {
                            var end2 = reader.uint32() + reader.pos;
                            while (reader.pos < end2)
                                message.propagateDown.push(reader.bool());
                        } else
                            message.propagateDown.push(reader.bool());
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
                        message.transformParam = $root.caffe.TransformationParameter.decode(reader, reader.uint32());
                        break;
                    case 101:
                        message.lossParam = $root.caffe.LossParameter.decode(reader, reader.uint32());
                        break;
                    case 102:
                        message.accuracyParam = $root.caffe.AccuracyParameter.decode(reader, reader.uint32());
                        break;
                    case 103:
                        message.argmaxParam = $root.caffe.ArgMaxParameter.decode(reader, reader.uint32());
                        break;
                    case 139:
                        message.batchNormParam = $root.caffe.BatchNormParameter.decode(reader, reader.uint32());
                        break;
                    case 141:
                        message.biasParam = $root.caffe.BiasParameter.decode(reader, reader.uint32());
                        break;
                    case 148:
                        message.clipParam = $root.caffe.ClipParameter.decode(reader, reader.uint32());
                        break;
                    case 104:
                        message.concatParam = $root.caffe.ConcatParameter.decode(reader, reader.uint32());
                        break;
                    case 105:
                        message.contrastiveLossParam = $root.caffe.ContrastiveLossParameter.decode(reader, reader.uint32());
                        break;
                    case 106:
                        message.convolutionParam = $root.caffe.ConvolutionParameter.decode(reader, reader.uint32());
                        break;
                    case 144:
                        message.cropParam = $root.caffe.CropParameter.decode(reader, reader.uint32());
                        break;
                    case 107:
                        message.dataParam = $root.caffe.DataParameter.decode(reader, reader.uint32());
                        break;
                    case 108:
                        message.dropoutParam = $root.caffe.DropoutParameter.decode(reader, reader.uint32());
                        break;
                    case 109:
                        message.dummyDataParam = $root.caffe.DummyDataParameter.decode(reader, reader.uint32());
                        break;
                    case 110:
                        message.eltwiseParam = $root.caffe.EltwiseParameter.decode(reader, reader.uint32());
                        break;
                    case 140:
                        message.eluParam = $root.caffe.ELUParameter.decode(reader, reader.uint32());
                        break;
                    case 137:
                        message.embedParam = $root.caffe.EmbedParameter.decode(reader, reader.uint32());
                        break;
                    case 111:
                        message.expParam = $root.caffe.ExpParameter.decode(reader, reader.uint32());
                        break;
                    case 135:
                        message.flattenParam = $root.caffe.FlattenParameter.decode(reader, reader.uint32());
                        break;
                    case 112:
                        message.hdf5DataParam = $root.caffe.HDF5DataParameter.decode(reader, reader.uint32());
                        break;
                    case 113:
                        message.hdf5OutputParam = $root.caffe.HDF5OutputParameter.decode(reader, reader.uint32());
                        break;
                    case 114:
                        message.hingeLossParam = $root.caffe.HingeLossParameter.decode(reader, reader.uint32());
                        break;
                    case 115:
                        message.imageDataParam = $root.caffe.ImageDataParameter.decode(reader, reader.uint32());
                        break;
                    case 116:
                        message.infogainLossParam = $root.caffe.InfogainLossParameter.decode(reader, reader.uint32());
                        break;
                    case 117:
                        message.innerProductParam = $root.caffe.InnerProductParameter.decode(reader, reader.uint32());
                        break;
                    case 143:
                        message.inputParam = $root.caffe.InputParameter.decode(reader, reader.uint32());
                        break;
                    case 134:
                        message.logParam = $root.caffe.LogParameter.decode(reader, reader.uint32());
                        break;
                    case 118:
                        message.lrnParam = $root.caffe.LRNParameter.decode(reader, reader.uint32());
                        break;
                    case 119:
                        message.memoryDataParam = $root.caffe.MemoryDataParameter.decode(reader, reader.uint32());
                        break;
                    case 120:
                        message.mvnParam = $root.caffe.MVNParameter.decode(reader, reader.uint32());
                        break;
                    case 145:
                        message.parameterParam = $root.caffe.ParameterParameter.decode(reader, reader.uint32());
                        break;
                    case 121:
                        message.poolingParam = $root.caffe.PoolingParameter.decode(reader, reader.uint32());
                        break;
                    case 122:
                        message.powerParam = $root.caffe.PowerParameter.decode(reader, reader.uint32());
                        break;
                    case 131:
                        message.preluParam = $root.caffe.PReLUParameter.decode(reader, reader.uint32());
                        break;
                    case 130:
                        message.pythonParam = $root.caffe.PythonParameter.decode(reader, reader.uint32());
                        break;
                    case 146:
                        message.recurrentParam = $root.caffe.RecurrentParameter.decode(reader, reader.uint32());
                        break;
                    case 136:
                        message.reductionParam = $root.caffe.ReductionParameter.decode(reader, reader.uint32());
                        break;
                    case 123:
                        message.reluParam = $root.caffe.ReLUParameter.decode(reader, reader.uint32());
                        break;
                    case 133:
                        message.reshapeParam = $root.caffe.ReshapeParameter.decode(reader, reader.uint32());
                        break;
                    case 142:
                        message.scaleParam = $root.caffe.ScaleParameter.decode(reader, reader.uint32());
                        break;
                    case 124:
                        message.sigmoidParam = $root.caffe.SigmoidParameter.decode(reader, reader.uint32());
                        break;
                    case 125:
                        message.softmaxParam = $root.caffe.SoftmaxParameter.decode(reader, reader.uint32());
                        break;
                    case 132:
                        message.sppParam = $root.caffe.SPPParameter.decode(reader, reader.uint32());
                        break;
                    case 126:
                        message.sliceParam = $root.caffe.SliceParameter.decode(reader, reader.uint32());
                        break;
                    case 147:
                        message.swishParam = $root.caffe.SwishParameter.decode(reader, reader.uint32());
                        break;
                    case 127:
                        message.tanhParam = $root.caffe.TanHParameter.decode(reader, reader.uint32());
                        break;
                    case 128:
                        message.thresholdParam = $root.caffe.ThresholdParameter.decode(reader, reader.uint32());
                        break;
                    case 138:
                        message.tileParam = $root.caffe.TileParameter.decode(reader, reader.uint32());
                        break;
                    case 129:
                        message.windowDataParam = $root.caffe.WindowDataParameter.decode(reader, reader.uint32());
                        break;
                    default:
                        reader.skipType(tag & 7);
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
                if (message.lossWeight != null && message.hasOwnProperty("lossWeight")) {
                    if (!Array.isArray(message.lossWeight))
                        return "lossWeight: array expected";
                    for (var i = 0; i < message.lossWeight.length; ++i)
                        if (typeof message.lossWeight[i] !== "number")
                            return "lossWeight: number[] expected";
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
                if (message.propagateDown != null && message.hasOwnProperty("propagateDown")) {
                    if (!Array.isArray(message.propagateDown))
                        return "propagateDown: array expected";
                    for (var i = 0; i < message.propagateDown.length; ++i)
                        if (typeof message.propagateDown[i] !== "boolean")
                            return "propagateDown: boolean[] expected";
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
                if (message.transformParam != null && message.hasOwnProperty("transformParam")) {
                    var error = $root.caffe.TransformationParameter.verify(message.transformParam);
                    if (error)
                        return "transformParam." + error;
                }
                if (message.lossParam != null && message.hasOwnProperty("lossParam")) {
                    var error = $root.caffe.LossParameter.verify(message.lossParam);
                    if (error)
                        return "lossParam." + error;
                }
                if (message.accuracyParam != null && message.hasOwnProperty("accuracyParam")) {
                    var error = $root.caffe.AccuracyParameter.verify(message.accuracyParam);
                    if (error)
                        return "accuracyParam." + error;
                }
                if (message.argmaxParam != null && message.hasOwnProperty("argmaxParam")) {
                    var error = $root.caffe.ArgMaxParameter.verify(message.argmaxParam);
                    if (error)
                        return "argmaxParam." + error;
                }
                if (message.batchNormParam != null && message.hasOwnProperty("batchNormParam")) {
                    var error = $root.caffe.BatchNormParameter.verify(message.batchNormParam);
                    if (error)
                        return "batchNormParam." + error;
                }
                if (message.biasParam != null && message.hasOwnProperty("biasParam")) {
                    var error = $root.caffe.BiasParameter.verify(message.biasParam);
                    if (error)
                        return "biasParam." + error;
                }
                if (message.clipParam != null && message.hasOwnProperty("clipParam")) {
                    var error = $root.caffe.ClipParameter.verify(message.clipParam);
                    if (error)
                        return "clipParam." + error;
                }
                if (message.concatParam != null && message.hasOwnProperty("concatParam")) {
                    var error = $root.caffe.ConcatParameter.verify(message.concatParam);
                    if (error)
                        return "concatParam." + error;
                }
                if (message.contrastiveLossParam != null && message.hasOwnProperty("contrastiveLossParam")) {
                    var error = $root.caffe.ContrastiveLossParameter.verify(message.contrastiveLossParam);
                    if (error)
                        return "contrastiveLossParam." + error;
                }
                if (message.convolutionParam != null && message.hasOwnProperty("convolutionParam")) {
                    var error = $root.caffe.ConvolutionParameter.verify(message.convolutionParam);
                    if (error)
                        return "convolutionParam." + error;
                }
                if (message.cropParam != null && message.hasOwnProperty("cropParam")) {
                    var error = $root.caffe.CropParameter.verify(message.cropParam);
                    if (error)
                        return "cropParam." + error;
                }
                if (message.dataParam != null && message.hasOwnProperty("dataParam")) {
                    var error = $root.caffe.DataParameter.verify(message.dataParam);
                    if (error)
                        return "dataParam." + error;
                }
                if (message.dropoutParam != null && message.hasOwnProperty("dropoutParam")) {
                    var error = $root.caffe.DropoutParameter.verify(message.dropoutParam);
                    if (error)
                        return "dropoutParam." + error;
                }
                if (message.dummyDataParam != null && message.hasOwnProperty("dummyDataParam")) {
                    var error = $root.caffe.DummyDataParameter.verify(message.dummyDataParam);
                    if (error)
                        return "dummyDataParam." + error;
                }
                if (message.eltwiseParam != null && message.hasOwnProperty("eltwiseParam")) {
                    var error = $root.caffe.EltwiseParameter.verify(message.eltwiseParam);
                    if (error)
                        return "eltwiseParam." + error;
                }
                if (message.eluParam != null && message.hasOwnProperty("eluParam")) {
                    var error = $root.caffe.ELUParameter.verify(message.eluParam);
                    if (error)
                        return "eluParam." + error;
                }
                if (message.embedParam != null && message.hasOwnProperty("embedParam")) {
                    var error = $root.caffe.EmbedParameter.verify(message.embedParam);
                    if (error)
                        return "embedParam." + error;
                }
                if (message.expParam != null && message.hasOwnProperty("expParam")) {
                    var error = $root.caffe.ExpParameter.verify(message.expParam);
                    if (error)
                        return "expParam." + error;
                }
                if (message.flattenParam != null && message.hasOwnProperty("flattenParam")) {
                    var error = $root.caffe.FlattenParameter.verify(message.flattenParam);
                    if (error)
                        return "flattenParam." + error;
                }
                if (message.hdf5DataParam != null && message.hasOwnProperty("hdf5DataParam")) {
                    var error = $root.caffe.HDF5DataParameter.verify(message.hdf5DataParam);
                    if (error)
                        return "hdf5DataParam." + error;
                }
                if (message.hdf5OutputParam != null && message.hasOwnProperty("hdf5OutputParam")) {
                    var error = $root.caffe.HDF5OutputParameter.verify(message.hdf5OutputParam);
                    if (error)
                        return "hdf5OutputParam." + error;
                }
                if (message.hingeLossParam != null && message.hasOwnProperty("hingeLossParam")) {
                    var error = $root.caffe.HingeLossParameter.verify(message.hingeLossParam);
                    if (error)
                        return "hingeLossParam." + error;
                }
                if (message.imageDataParam != null && message.hasOwnProperty("imageDataParam")) {
                    var error = $root.caffe.ImageDataParameter.verify(message.imageDataParam);
                    if (error)
                        return "imageDataParam." + error;
                }
                if (message.infogainLossParam != null && message.hasOwnProperty("infogainLossParam")) {
                    var error = $root.caffe.InfogainLossParameter.verify(message.infogainLossParam);
                    if (error)
                        return "infogainLossParam." + error;
                }
                if (message.innerProductParam != null && message.hasOwnProperty("innerProductParam")) {
                    var error = $root.caffe.InnerProductParameter.verify(message.innerProductParam);
                    if (error)
                        return "innerProductParam." + error;
                }
                if (message.inputParam != null && message.hasOwnProperty("inputParam")) {
                    var error = $root.caffe.InputParameter.verify(message.inputParam);
                    if (error)
                        return "inputParam." + error;
                }
                if (message.logParam != null && message.hasOwnProperty("logParam")) {
                    var error = $root.caffe.LogParameter.verify(message.logParam);
                    if (error)
                        return "logParam." + error;
                }
                if (message.lrnParam != null && message.hasOwnProperty("lrnParam")) {
                    var error = $root.caffe.LRNParameter.verify(message.lrnParam);
                    if (error)
                        return "lrnParam." + error;
                }
                if (message.memoryDataParam != null && message.hasOwnProperty("memoryDataParam")) {
                    var error = $root.caffe.MemoryDataParameter.verify(message.memoryDataParam);
                    if (error)
                        return "memoryDataParam." + error;
                }
                if (message.mvnParam != null && message.hasOwnProperty("mvnParam")) {
                    var error = $root.caffe.MVNParameter.verify(message.mvnParam);
                    if (error)
                        return "mvnParam." + error;
                }
                if (message.parameterParam != null && message.hasOwnProperty("parameterParam")) {
                    var error = $root.caffe.ParameterParameter.verify(message.parameterParam);
                    if (error)
                        return "parameterParam." + error;
                }
                if (message.poolingParam != null && message.hasOwnProperty("poolingParam")) {
                    var error = $root.caffe.PoolingParameter.verify(message.poolingParam);
                    if (error)
                        return "poolingParam." + error;
                }
                if (message.powerParam != null && message.hasOwnProperty("powerParam")) {
                    var error = $root.caffe.PowerParameter.verify(message.powerParam);
                    if (error)
                        return "powerParam." + error;
                }
                if (message.preluParam != null && message.hasOwnProperty("preluParam")) {
                    var error = $root.caffe.PReLUParameter.verify(message.preluParam);
                    if (error)
                        return "preluParam." + error;
                }
                if (message.pythonParam != null && message.hasOwnProperty("pythonParam")) {
                    var error = $root.caffe.PythonParameter.verify(message.pythonParam);
                    if (error)
                        return "pythonParam." + error;
                }
                if (message.recurrentParam != null && message.hasOwnProperty("recurrentParam")) {
                    var error = $root.caffe.RecurrentParameter.verify(message.recurrentParam);
                    if (error)
                        return "recurrentParam." + error;
                }
                if (message.reductionParam != null && message.hasOwnProperty("reductionParam")) {
                    var error = $root.caffe.ReductionParameter.verify(message.reductionParam);
                    if (error)
                        return "reductionParam." + error;
                }
                if (message.reluParam != null && message.hasOwnProperty("reluParam")) {
                    var error = $root.caffe.ReLUParameter.verify(message.reluParam);
                    if (error)
                        return "reluParam." + error;
                }
                if (message.reshapeParam != null && message.hasOwnProperty("reshapeParam")) {
                    var error = $root.caffe.ReshapeParameter.verify(message.reshapeParam);
                    if (error)
                        return "reshapeParam." + error;
                }
                if (message.scaleParam != null && message.hasOwnProperty("scaleParam")) {
                    var error = $root.caffe.ScaleParameter.verify(message.scaleParam);
                    if (error)
                        return "scaleParam." + error;
                }
                if (message.sigmoidParam != null && message.hasOwnProperty("sigmoidParam")) {
                    var error = $root.caffe.SigmoidParameter.verify(message.sigmoidParam);
                    if (error)
                        return "sigmoidParam." + error;
                }
                if (message.softmaxParam != null && message.hasOwnProperty("softmaxParam")) {
                    var error = $root.caffe.SoftmaxParameter.verify(message.softmaxParam);
                    if (error)
                        return "softmaxParam." + error;
                }
                if (message.sppParam != null && message.hasOwnProperty("sppParam")) {
                    var error = $root.caffe.SPPParameter.verify(message.sppParam);
                    if (error)
                        return "sppParam." + error;
                }
                if (message.sliceParam != null && message.hasOwnProperty("sliceParam")) {
                    var error = $root.caffe.SliceParameter.verify(message.sliceParam);
                    if (error)
                        return "sliceParam." + error;
                }
                if (message.swishParam != null && message.hasOwnProperty("swishParam")) {
                    var error = $root.caffe.SwishParameter.verify(message.swishParam);
                    if (error)
                        return "swishParam." + error;
                }
                if (message.tanhParam != null && message.hasOwnProperty("tanhParam")) {
                    var error = $root.caffe.TanHParameter.verify(message.tanhParam);
                    if (error)
                        return "tanhParam." + error;
                }
                if (message.thresholdParam != null && message.hasOwnProperty("thresholdParam")) {
                    var error = $root.caffe.ThresholdParameter.verify(message.thresholdParam);
                    if (error)
                        return "thresholdParam." + error;
                }
                if (message.tileParam != null && message.hasOwnProperty("tileParam")) {
                    var error = $root.caffe.TileParameter.verify(message.tileParam);
                    if (error)
                        return "tileParam." + error;
                }
                if (message.windowDataParam != null && message.hasOwnProperty("windowDataParam")) {
                    var error = $root.caffe.WindowDataParameter.verify(message.windowDataParam);
                    if (error)
                        return "windowDataParam." + error;
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
                if (object.lossWeight) {
                    if (!Array.isArray(object.lossWeight))
                        throw TypeError(".caffe.LayerParameter.lossWeight: array expected");
                    message.lossWeight = [];
                    for (var i = 0; i < object.lossWeight.length; ++i)
                        message.lossWeight[i] = Number(object.lossWeight[i]);
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
                if (object.propagateDown) {
                    if (!Array.isArray(object.propagateDown))
                        throw TypeError(".caffe.LayerParameter.propagateDown: array expected");
                    message.propagateDown = [];
                    for (var i = 0; i < object.propagateDown.length; ++i)
                        message.propagateDown[i] = Boolean(object.propagateDown[i]);
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
                if (object.transformParam != null) {
                    if (typeof object.transformParam !== "object")
                        throw TypeError(".caffe.LayerParameter.transformParam: object expected");
                    message.transformParam = $root.caffe.TransformationParameter.fromObject(object.transformParam);
                }
                if (object.lossParam != null) {
                    if (typeof object.lossParam !== "object")
                        throw TypeError(".caffe.LayerParameter.lossParam: object expected");
                    message.lossParam = $root.caffe.LossParameter.fromObject(object.lossParam);
                }
                if (object.accuracyParam != null) {
                    if (typeof object.accuracyParam !== "object")
                        throw TypeError(".caffe.LayerParameter.accuracyParam: object expected");
                    message.accuracyParam = $root.caffe.AccuracyParameter.fromObject(object.accuracyParam);
                }
                if (object.argmaxParam != null) {
                    if (typeof object.argmaxParam !== "object")
                        throw TypeError(".caffe.LayerParameter.argmaxParam: object expected");
                    message.argmaxParam = $root.caffe.ArgMaxParameter.fromObject(object.argmaxParam);
                }
                if (object.batchNormParam != null) {
                    if (typeof object.batchNormParam !== "object")
                        throw TypeError(".caffe.LayerParameter.batchNormParam: object expected");
                    message.batchNormParam = $root.caffe.BatchNormParameter.fromObject(object.batchNormParam);
                }
                if (object.biasParam != null) {
                    if (typeof object.biasParam !== "object")
                        throw TypeError(".caffe.LayerParameter.biasParam: object expected");
                    message.biasParam = $root.caffe.BiasParameter.fromObject(object.biasParam);
                }
                if (object.clipParam != null) {
                    if (typeof object.clipParam !== "object")
                        throw TypeError(".caffe.LayerParameter.clipParam: object expected");
                    message.clipParam = $root.caffe.ClipParameter.fromObject(object.clipParam);
                }
                if (object.concatParam != null) {
                    if (typeof object.concatParam !== "object")
                        throw TypeError(".caffe.LayerParameter.concatParam: object expected");
                    message.concatParam = $root.caffe.ConcatParameter.fromObject(object.concatParam);
                }
                if (object.contrastiveLossParam != null) {
                    if (typeof object.contrastiveLossParam !== "object")
                        throw TypeError(".caffe.LayerParameter.contrastiveLossParam: object expected");
                    message.contrastiveLossParam = $root.caffe.ContrastiveLossParameter.fromObject(object.contrastiveLossParam);
                }
                if (object.convolutionParam != null) {
                    if (typeof object.convolutionParam !== "object")
                        throw TypeError(".caffe.LayerParameter.convolutionParam: object expected");
                    message.convolutionParam = $root.caffe.ConvolutionParameter.fromObject(object.convolutionParam);
                }
                if (object.cropParam != null) {
                    if (typeof object.cropParam !== "object")
                        throw TypeError(".caffe.LayerParameter.cropParam: object expected");
                    message.cropParam = $root.caffe.CropParameter.fromObject(object.cropParam);
                }
                if (object.dataParam != null) {
                    if (typeof object.dataParam !== "object")
                        throw TypeError(".caffe.LayerParameter.dataParam: object expected");
                    message.dataParam = $root.caffe.DataParameter.fromObject(object.dataParam);
                }
                if (object.dropoutParam != null) {
                    if (typeof object.dropoutParam !== "object")
                        throw TypeError(".caffe.LayerParameter.dropoutParam: object expected");
                    message.dropoutParam = $root.caffe.DropoutParameter.fromObject(object.dropoutParam);
                }
                if (object.dummyDataParam != null) {
                    if (typeof object.dummyDataParam !== "object")
                        throw TypeError(".caffe.LayerParameter.dummyDataParam: object expected");
                    message.dummyDataParam = $root.caffe.DummyDataParameter.fromObject(object.dummyDataParam);
                }
                if (object.eltwiseParam != null) {
                    if (typeof object.eltwiseParam !== "object")
                        throw TypeError(".caffe.LayerParameter.eltwiseParam: object expected");
                    message.eltwiseParam = $root.caffe.EltwiseParameter.fromObject(object.eltwiseParam);
                }
                if (object.eluParam != null) {
                    if (typeof object.eluParam !== "object")
                        throw TypeError(".caffe.LayerParameter.eluParam: object expected");
                    message.eluParam = $root.caffe.ELUParameter.fromObject(object.eluParam);
                }
                if (object.embedParam != null) {
                    if (typeof object.embedParam !== "object")
                        throw TypeError(".caffe.LayerParameter.embedParam: object expected");
                    message.embedParam = $root.caffe.EmbedParameter.fromObject(object.embedParam);
                }
                if (object.expParam != null) {
                    if (typeof object.expParam !== "object")
                        throw TypeError(".caffe.LayerParameter.expParam: object expected");
                    message.expParam = $root.caffe.ExpParameter.fromObject(object.expParam);
                }
                if (object.flattenParam != null) {
                    if (typeof object.flattenParam !== "object")
                        throw TypeError(".caffe.LayerParameter.flattenParam: object expected");
                    message.flattenParam = $root.caffe.FlattenParameter.fromObject(object.flattenParam);
                }
                if (object.hdf5DataParam != null) {
                    if (typeof object.hdf5DataParam !== "object")
                        throw TypeError(".caffe.LayerParameter.hdf5DataParam: object expected");
                    message.hdf5DataParam = $root.caffe.HDF5DataParameter.fromObject(object.hdf5DataParam);
                }
                if (object.hdf5OutputParam != null) {
                    if (typeof object.hdf5OutputParam !== "object")
                        throw TypeError(".caffe.LayerParameter.hdf5OutputParam: object expected");
                    message.hdf5OutputParam = $root.caffe.HDF5OutputParameter.fromObject(object.hdf5OutputParam);
                }
                if (object.hingeLossParam != null) {
                    if (typeof object.hingeLossParam !== "object")
                        throw TypeError(".caffe.LayerParameter.hingeLossParam: object expected");
                    message.hingeLossParam = $root.caffe.HingeLossParameter.fromObject(object.hingeLossParam);
                }
                if (object.imageDataParam != null) {
                    if (typeof object.imageDataParam !== "object")
                        throw TypeError(".caffe.LayerParameter.imageDataParam: object expected");
                    message.imageDataParam = $root.caffe.ImageDataParameter.fromObject(object.imageDataParam);
                }
                if (object.infogainLossParam != null) {
                    if (typeof object.infogainLossParam !== "object")
                        throw TypeError(".caffe.LayerParameter.infogainLossParam: object expected");
                    message.infogainLossParam = $root.caffe.InfogainLossParameter.fromObject(object.infogainLossParam);
                }
                if (object.innerProductParam != null) {
                    if (typeof object.innerProductParam !== "object")
                        throw TypeError(".caffe.LayerParameter.innerProductParam: object expected");
                    message.innerProductParam = $root.caffe.InnerProductParameter.fromObject(object.innerProductParam);
                }
                if (object.inputParam != null) {
                    if (typeof object.inputParam !== "object")
                        throw TypeError(".caffe.LayerParameter.inputParam: object expected");
                    message.inputParam = $root.caffe.InputParameter.fromObject(object.inputParam);
                }
                if (object.logParam != null) {
                    if (typeof object.logParam !== "object")
                        throw TypeError(".caffe.LayerParameter.logParam: object expected");
                    message.logParam = $root.caffe.LogParameter.fromObject(object.logParam);
                }
                if (object.lrnParam != null) {
                    if (typeof object.lrnParam !== "object")
                        throw TypeError(".caffe.LayerParameter.lrnParam: object expected");
                    message.lrnParam = $root.caffe.LRNParameter.fromObject(object.lrnParam);
                }
                if (object.memoryDataParam != null) {
                    if (typeof object.memoryDataParam !== "object")
                        throw TypeError(".caffe.LayerParameter.memoryDataParam: object expected");
                    message.memoryDataParam = $root.caffe.MemoryDataParameter.fromObject(object.memoryDataParam);
                }
                if (object.mvnParam != null) {
                    if (typeof object.mvnParam !== "object")
                        throw TypeError(".caffe.LayerParameter.mvnParam: object expected");
                    message.mvnParam = $root.caffe.MVNParameter.fromObject(object.mvnParam);
                }
                if (object.parameterParam != null) {
                    if (typeof object.parameterParam !== "object")
                        throw TypeError(".caffe.LayerParameter.parameterParam: object expected");
                    message.parameterParam = $root.caffe.ParameterParameter.fromObject(object.parameterParam);
                }
                if (object.poolingParam != null) {
                    if (typeof object.poolingParam !== "object")
                        throw TypeError(".caffe.LayerParameter.poolingParam: object expected");
                    message.poolingParam = $root.caffe.PoolingParameter.fromObject(object.poolingParam);
                }
                if (object.powerParam != null) {
                    if (typeof object.powerParam !== "object")
                        throw TypeError(".caffe.LayerParameter.powerParam: object expected");
                    message.powerParam = $root.caffe.PowerParameter.fromObject(object.powerParam);
                }
                if (object.preluParam != null) {
                    if (typeof object.preluParam !== "object")
                        throw TypeError(".caffe.LayerParameter.preluParam: object expected");
                    message.preluParam = $root.caffe.PReLUParameter.fromObject(object.preluParam);
                }
                if (object.pythonParam != null) {
                    if (typeof object.pythonParam !== "object")
                        throw TypeError(".caffe.LayerParameter.pythonParam: object expected");
                    message.pythonParam = $root.caffe.PythonParameter.fromObject(object.pythonParam);
                }
                if (object.recurrentParam != null) {
                    if (typeof object.recurrentParam !== "object")
                        throw TypeError(".caffe.LayerParameter.recurrentParam: object expected");
                    message.recurrentParam = $root.caffe.RecurrentParameter.fromObject(object.recurrentParam);
                }
                if (object.reductionParam != null) {
                    if (typeof object.reductionParam !== "object")
                        throw TypeError(".caffe.LayerParameter.reductionParam: object expected");
                    message.reductionParam = $root.caffe.ReductionParameter.fromObject(object.reductionParam);
                }
                if (object.reluParam != null) {
                    if (typeof object.reluParam !== "object")
                        throw TypeError(".caffe.LayerParameter.reluParam: object expected");
                    message.reluParam = $root.caffe.ReLUParameter.fromObject(object.reluParam);
                }
                if (object.reshapeParam != null) {
                    if (typeof object.reshapeParam !== "object")
                        throw TypeError(".caffe.LayerParameter.reshapeParam: object expected");
                    message.reshapeParam = $root.caffe.ReshapeParameter.fromObject(object.reshapeParam);
                }
                if (object.scaleParam != null) {
                    if (typeof object.scaleParam !== "object")
                        throw TypeError(".caffe.LayerParameter.scaleParam: object expected");
                    message.scaleParam = $root.caffe.ScaleParameter.fromObject(object.scaleParam);
                }
                if (object.sigmoidParam != null) {
                    if (typeof object.sigmoidParam !== "object")
                        throw TypeError(".caffe.LayerParameter.sigmoidParam: object expected");
                    message.sigmoidParam = $root.caffe.SigmoidParameter.fromObject(object.sigmoidParam);
                }
                if (object.softmaxParam != null) {
                    if (typeof object.softmaxParam !== "object")
                        throw TypeError(".caffe.LayerParameter.softmaxParam: object expected");
                    message.softmaxParam = $root.caffe.SoftmaxParameter.fromObject(object.softmaxParam);
                }
                if (object.sppParam != null) {
                    if (typeof object.sppParam !== "object")
                        throw TypeError(".caffe.LayerParameter.sppParam: object expected");
                    message.sppParam = $root.caffe.SPPParameter.fromObject(object.sppParam);
                }
                if (object.sliceParam != null) {
                    if (typeof object.sliceParam !== "object")
                        throw TypeError(".caffe.LayerParameter.sliceParam: object expected");
                    message.sliceParam = $root.caffe.SliceParameter.fromObject(object.sliceParam);
                }
                if (object.swishParam != null) {
                    if (typeof object.swishParam !== "object")
                        throw TypeError(".caffe.LayerParameter.swishParam: object expected");
                    message.swishParam = $root.caffe.SwishParameter.fromObject(object.swishParam);
                }
                if (object.tanhParam != null) {
                    if (typeof object.tanhParam !== "object")
                        throw TypeError(".caffe.LayerParameter.tanhParam: object expected");
                    message.tanhParam = $root.caffe.TanHParameter.fromObject(object.tanhParam);
                }
                if (object.thresholdParam != null) {
                    if (typeof object.thresholdParam !== "object")
                        throw TypeError(".caffe.LayerParameter.thresholdParam: object expected");
                    message.thresholdParam = $root.caffe.ThresholdParameter.fromObject(object.thresholdParam);
                }
                if (object.tileParam != null) {
                    if (typeof object.tileParam !== "object")
                        throw TypeError(".caffe.LayerParameter.tileParam: object expected");
                    message.tileParam = $root.caffe.TileParameter.fromObject(object.tileParam);
                }
                if (object.windowDataParam != null) {
                    if (typeof object.windowDataParam !== "object")
                        throw TypeError(".caffe.LayerParameter.windowDataParam: object expected");
                    message.windowDataParam = $root.caffe.WindowDataParameter.fromObject(object.windowDataParam);
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
                    object.lossWeight = [];
                    object.param = [];
                    object.blobs = [];
                    object.include = [];
                    object.exclude = [];
                    object.propagateDown = [];
                }
                if (options.defaults) {
                    object.name = "";
                    object.type = "";
                    object.phase = options.enums === String ? "TRAIN" : 0;
                    object.transformParam = null;
                    object.lossParam = null;
                    object.accuracyParam = null;
                    object.argmaxParam = null;
                    object.concatParam = null;
                    object.contrastiveLossParam = null;
                    object.convolutionParam = null;
                    object.dataParam = null;
                    object.dropoutParam = null;
                    object.dummyDataParam = null;
                    object.eltwiseParam = null;
                    object.expParam = null;
                    object.hdf5DataParam = null;
                    object.hdf5OutputParam = null;
                    object.hingeLossParam = null;
                    object.imageDataParam = null;
                    object.infogainLossParam = null;
                    object.innerProductParam = null;
                    object.lrnParam = null;
                    object.memoryDataParam = null;
                    object.mvnParam = null;
                    object.poolingParam = null;
                    object.powerParam = null;
                    object.reluParam = null;
                    object.sigmoidParam = null;
                    object.softmaxParam = null;
                    object.sliceParam = null;
                    object.tanhParam = null;
                    object.thresholdParam = null;
                    object.windowDataParam = null;
                    object.pythonParam = null;
                    object.preluParam = null;
                    object.sppParam = null;
                    object.reshapeParam = null;
                    object.logParam = null;
                    object.flattenParam = null;
                    object.reductionParam = null;
                    object.embedParam = null;
                    object.tileParam = null;
                    object.batchNormParam = null;
                    object.eluParam = null;
                    object.biasParam = null;
                    object.scaleParam = null;
                    object.inputParam = null;
                    object.cropParam = null;
                    object.parameterParam = null;
                    object.recurrentParam = null;
                    object.swishParam = null;
                    object.clipParam = null;
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
                if (message.lossWeight && message.lossWeight.length) {
                    object.lossWeight = [];
                    for (var j = 0; j < message.lossWeight.length; ++j)
                        object.lossWeight[j] = options.json && !isFinite(message.lossWeight[j]) ? String(message.lossWeight[j]) : message.lossWeight[j];
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
                if (message.propagateDown && message.propagateDown.length) {
                    object.propagateDown = [];
                    for (var j = 0; j < message.propagateDown.length; ++j)
                        object.propagateDown[j] = message.propagateDown[j];
                }
                if (message.transformParam != null && message.hasOwnProperty("transformParam"))
                    object.transformParam = $root.caffe.TransformationParameter.toObject(message.transformParam, options);
                if (message.lossParam != null && message.hasOwnProperty("lossParam"))
                    object.lossParam = $root.caffe.LossParameter.toObject(message.lossParam, options);
                if (message.accuracyParam != null && message.hasOwnProperty("accuracyParam"))
                    object.accuracyParam = $root.caffe.AccuracyParameter.toObject(message.accuracyParam, options);
                if (message.argmaxParam != null && message.hasOwnProperty("argmaxParam"))
                    object.argmaxParam = $root.caffe.ArgMaxParameter.toObject(message.argmaxParam, options);
                if (message.concatParam != null && message.hasOwnProperty("concatParam"))
                    object.concatParam = $root.caffe.ConcatParameter.toObject(message.concatParam, options);
                if (message.contrastiveLossParam != null && message.hasOwnProperty("contrastiveLossParam"))
                    object.contrastiveLossParam = $root.caffe.ContrastiveLossParameter.toObject(message.contrastiveLossParam, options);
                if (message.convolutionParam != null && message.hasOwnProperty("convolutionParam"))
                    object.convolutionParam = $root.caffe.ConvolutionParameter.toObject(message.convolutionParam, options);
                if (message.dataParam != null && message.hasOwnProperty("dataParam"))
                    object.dataParam = $root.caffe.DataParameter.toObject(message.dataParam, options);
                if (message.dropoutParam != null && message.hasOwnProperty("dropoutParam"))
                    object.dropoutParam = $root.caffe.DropoutParameter.toObject(message.dropoutParam, options);
                if (message.dummyDataParam != null && message.hasOwnProperty("dummyDataParam"))
                    object.dummyDataParam = $root.caffe.DummyDataParameter.toObject(message.dummyDataParam, options);
                if (message.eltwiseParam != null && message.hasOwnProperty("eltwiseParam"))
                    object.eltwiseParam = $root.caffe.EltwiseParameter.toObject(message.eltwiseParam, options);
                if (message.expParam != null && message.hasOwnProperty("expParam"))
                    object.expParam = $root.caffe.ExpParameter.toObject(message.expParam, options);
                if (message.hdf5DataParam != null && message.hasOwnProperty("hdf5DataParam"))
                    object.hdf5DataParam = $root.caffe.HDF5DataParameter.toObject(message.hdf5DataParam, options);
                if (message.hdf5OutputParam != null && message.hasOwnProperty("hdf5OutputParam"))
                    object.hdf5OutputParam = $root.caffe.HDF5OutputParameter.toObject(message.hdf5OutputParam, options);
                if (message.hingeLossParam != null && message.hasOwnProperty("hingeLossParam"))
                    object.hingeLossParam = $root.caffe.HingeLossParameter.toObject(message.hingeLossParam, options);
                if (message.imageDataParam != null && message.hasOwnProperty("imageDataParam"))
                    object.imageDataParam = $root.caffe.ImageDataParameter.toObject(message.imageDataParam, options);
                if (message.infogainLossParam != null && message.hasOwnProperty("infogainLossParam"))
                    object.infogainLossParam = $root.caffe.InfogainLossParameter.toObject(message.infogainLossParam, options);
                if (message.innerProductParam != null && message.hasOwnProperty("innerProductParam"))
                    object.innerProductParam = $root.caffe.InnerProductParameter.toObject(message.innerProductParam, options);
                if (message.lrnParam != null && message.hasOwnProperty("lrnParam"))
                    object.lrnParam = $root.caffe.LRNParameter.toObject(message.lrnParam, options);
                if (message.memoryDataParam != null && message.hasOwnProperty("memoryDataParam"))
                    object.memoryDataParam = $root.caffe.MemoryDataParameter.toObject(message.memoryDataParam, options);
                if (message.mvnParam != null && message.hasOwnProperty("mvnParam"))
                    object.mvnParam = $root.caffe.MVNParameter.toObject(message.mvnParam, options);
                if (message.poolingParam != null && message.hasOwnProperty("poolingParam"))
                    object.poolingParam = $root.caffe.PoolingParameter.toObject(message.poolingParam, options);
                if (message.powerParam != null && message.hasOwnProperty("powerParam"))
                    object.powerParam = $root.caffe.PowerParameter.toObject(message.powerParam, options);
                if (message.reluParam != null && message.hasOwnProperty("reluParam"))
                    object.reluParam = $root.caffe.ReLUParameter.toObject(message.reluParam, options);
                if (message.sigmoidParam != null && message.hasOwnProperty("sigmoidParam"))
                    object.sigmoidParam = $root.caffe.SigmoidParameter.toObject(message.sigmoidParam, options);
                if (message.softmaxParam != null && message.hasOwnProperty("softmaxParam"))
                    object.softmaxParam = $root.caffe.SoftmaxParameter.toObject(message.softmaxParam, options);
                if (message.sliceParam != null && message.hasOwnProperty("sliceParam"))
                    object.sliceParam = $root.caffe.SliceParameter.toObject(message.sliceParam, options);
                if (message.tanhParam != null && message.hasOwnProperty("tanhParam"))
                    object.tanhParam = $root.caffe.TanHParameter.toObject(message.tanhParam, options);
                if (message.thresholdParam != null && message.hasOwnProperty("thresholdParam"))
                    object.thresholdParam = $root.caffe.ThresholdParameter.toObject(message.thresholdParam, options);
                if (message.windowDataParam != null && message.hasOwnProperty("windowDataParam"))
                    object.windowDataParam = $root.caffe.WindowDataParameter.toObject(message.windowDataParam, options);
                if (message.pythonParam != null && message.hasOwnProperty("pythonParam"))
                    object.pythonParam = $root.caffe.PythonParameter.toObject(message.pythonParam, options);
                if (message.preluParam != null && message.hasOwnProperty("preluParam"))
                    object.preluParam = $root.caffe.PReLUParameter.toObject(message.preluParam, options);
                if (message.sppParam != null && message.hasOwnProperty("sppParam"))
                    object.sppParam = $root.caffe.SPPParameter.toObject(message.sppParam, options);
                if (message.reshapeParam != null && message.hasOwnProperty("reshapeParam"))
                    object.reshapeParam = $root.caffe.ReshapeParameter.toObject(message.reshapeParam, options);
                if (message.logParam != null && message.hasOwnProperty("logParam"))
                    object.logParam = $root.caffe.LogParameter.toObject(message.logParam, options);
                if (message.flattenParam != null && message.hasOwnProperty("flattenParam"))
                    object.flattenParam = $root.caffe.FlattenParameter.toObject(message.flattenParam, options);
                if (message.reductionParam != null && message.hasOwnProperty("reductionParam"))
                    object.reductionParam = $root.caffe.ReductionParameter.toObject(message.reductionParam, options);
                if (message.embedParam != null && message.hasOwnProperty("embedParam"))
                    object.embedParam = $root.caffe.EmbedParameter.toObject(message.embedParam, options);
                if (message.tileParam != null && message.hasOwnProperty("tileParam"))
                    object.tileParam = $root.caffe.TileParameter.toObject(message.tileParam, options);
                if (message.batchNormParam != null && message.hasOwnProperty("batchNormParam"))
                    object.batchNormParam = $root.caffe.BatchNormParameter.toObject(message.batchNormParam, options);
                if (message.eluParam != null && message.hasOwnProperty("eluParam"))
                    object.eluParam = $root.caffe.ELUParameter.toObject(message.eluParam, options);
                if (message.biasParam != null && message.hasOwnProperty("biasParam"))
                    object.biasParam = $root.caffe.BiasParameter.toObject(message.biasParam, options);
                if (message.scaleParam != null && message.hasOwnProperty("scaleParam"))
                    object.scaleParam = $root.caffe.ScaleParameter.toObject(message.scaleParam, options);
                if (message.inputParam != null && message.hasOwnProperty("inputParam"))
                    object.inputParam = $root.caffe.InputParameter.toObject(message.inputParam, options);
                if (message.cropParam != null && message.hasOwnProperty("cropParam"))
                    object.cropParam = $root.caffe.CropParameter.toObject(message.cropParam, options);
                if (message.parameterParam != null && message.hasOwnProperty("parameterParam"))
                    object.parameterParam = $root.caffe.ParameterParameter.toObject(message.parameterParam, options);
                if (message.recurrentParam != null && message.hasOwnProperty("recurrentParam"))
                    object.recurrentParam = $root.caffe.RecurrentParameter.toObject(message.recurrentParam, options);
                if (message.swishParam != null && message.hasOwnProperty("swishParam"))
                    object.swishParam = $root.caffe.SwishParameter.toObject(message.swishParam, options);
                if (message.clipParam != null && message.hasOwnProperty("clipParam"))
                    object.clipParam = $root.caffe.ClipParameter.toObject(message.clipParam, options);
                return object;
            };
    
            LayerParameter.prototype.toJSON = function toJSON() {
                return this.constructor.toObject(this, $protobuf.util.toJSONOptions);
            };
    
            return LayerParameter;
        })();
    
        caffe.TransformationParameter = (function() {
    
            function TransformationParameter(properties) {
                this.meanValue = [];
                if (properties)
                    for (var keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                        if (properties[keys[i]] != null)
                            this[keys[i]] = properties[keys[i]];
            }
    
            TransformationParameter.prototype.scale = 1;
            TransformationParameter.prototype.mirror = false;
            TransformationParameter.prototype.cropSize = 0;
            TransformationParameter.prototype.meanFile = "";
            TransformationParameter.prototype.meanValue = $util.emptyArray;
            TransformationParameter.prototype.forceColor = false;
            TransformationParameter.prototype.forceGray = false;
    
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
                        message.cropSize = reader.uint32();
                        break;
                    case 4:
                        message.meanFile = reader.string();
                        break;
                    case 5:
                        if (!(message.meanValue && message.meanValue.length))
                            message.meanValue = [];
                        if ((tag & 7) === 2) {
                            var end2 = reader.uint32() + reader.pos;
                            while (reader.pos < end2)
                                message.meanValue.push(reader.float());
                        } else
                            message.meanValue.push(reader.float());
                        break;
                    case 6:
                        message.forceColor = reader.bool();
                        break;
                    case 7:
                        message.forceGray = reader.bool();
                        break;
                    default:
                        reader.skipType(tag & 7);
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
                if (message.cropSize != null && message.hasOwnProperty("cropSize"))
                    if (!$util.isInteger(message.cropSize))
                        return "cropSize: integer expected";
                if (message.meanFile != null && message.hasOwnProperty("meanFile"))
                    if (!$util.isString(message.meanFile))
                        return "meanFile: string expected";
                if (message.meanValue != null && message.hasOwnProperty("meanValue")) {
                    if (!Array.isArray(message.meanValue))
                        return "meanValue: array expected";
                    for (var i = 0; i < message.meanValue.length; ++i)
                        if (typeof message.meanValue[i] !== "number")
                            return "meanValue: number[] expected";
                }
                if (message.forceColor != null && message.hasOwnProperty("forceColor"))
                    if (typeof message.forceColor !== "boolean")
                        return "forceColor: boolean expected";
                if (message.forceGray != null && message.hasOwnProperty("forceGray"))
                    if (typeof message.forceGray !== "boolean")
                        return "forceGray: boolean expected";
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
                if (object.cropSize != null)
                    message.cropSize = object.cropSize >>> 0;
                if (object.meanFile != null)
                    message.meanFile = String(object.meanFile);
                if (object.meanValue) {
                    if (!Array.isArray(object.meanValue))
                        throw TypeError(".caffe.TransformationParameter.meanValue: array expected");
                    message.meanValue = [];
                    for (var i = 0; i < object.meanValue.length; ++i)
                        message.meanValue[i] = Number(object.meanValue[i]);
                }
                if (object.forceColor != null)
                    message.forceColor = Boolean(object.forceColor);
                if (object.forceGray != null)
                    message.forceGray = Boolean(object.forceGray);
                return message;
            };
    
            TransformationParameter.toObject = function toObject(message, options) {
                if (!options)
                    options = {};
                var object = {};
                if (options.arrays || options.defaults)
                    object.meanValue = [];
                if (options.defaults) {
                    object.scale = 1;
                    object.mirror = false;
                    object.cropSize = 0;
                    object.meanFile = "";
                    object.forceColor = false;
                    object.forceGray = false;
                }
                if (message.scale != null && message.hasOwnProperty("scale"))
                    object.scale = options.json && !isFinite(message.scale) ? String(message.scale) : message.scale;
                if (message.mirror != null && message.hasOwnProperty("mirror"))
                    object.mirror = message.mirror;
                if (message.cropSize != null && message.hasOwnProperty("cropSize"))
                    object.cropSize = message.cropSize;
                if (message.meanFile != null && message.hasOwnProperty("meanFile"))
                    object.meanFile = message.meanFile;
                if (message.meanValue && message.meanValue.length) {
                    object.meanValue = [];
                    for (var j = 0; j < message.meanValue.length; ++j)
                        object.meanValue[j] = options.json && !isFinite(message.meanValue[j]) ? String(message.meanValue[j]) : message.meanValue[j];
                }
                if (message.forceColor != null && message.hasOwnProperty("forceColor"))
                    object.forceColor = message.forceColor;
                if (message.forceGray != null && message.hasOwnProperty("forceGray"))
                    object.forceGray = message.forceGray;
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
    
            LossParameter.prototype.ignoreLabel = 0;
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
                        message.ignoreLabel = reader.int32();
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
    
            LossParameter.verify = function verify(message) {
                if (typeof message !== "object" || message === null)
                    return "object expected";
                if (message.ignoreLabel != null && message.hasOwnProperty("ignoreLabel"))
                    if (!$util.isInteger(message.ignoreLabel))
                        return "ignoreLabel: integer expected";
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
                if (object.ignoreLabel != null)
                    message.ignoreLabel = object.ignoreLabel | 0;
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
                    object.ignoreLabel = 0;
                    object.normalize = false;
                    object.normalization = options.enums === String ? "VALID" : 1;
                }
                if (message.ignoreLabel != null && message.hasOwnProperty("ignoreLabel"))
                    object.ignoreLabel = message.ignoreLabel;
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
    
            AccuracyParameter.prototype.topK = 1;
            AccuracyParameter.prototype.axis = 1;
            AccuracyParameter.prototype.ignoreLabel = 0;
    
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
                        message.topK = reader.uint32();
                        break;
                    case 2:
                        message.axis = reader.int32();
                        break;
                    case 3:
                        message.ignoreLabel = reader.int32();
                        break;
                    default:
                        reader.skipType(tag & 7);
                        break;
                    }
                }
                return message;
            };
    
            AccuracyParameter.verify = function verify(message) {
                if (typeof message !== "object" || message === null)
                    return "object expected";
                if (message.topK != null && message.hasOwnProperty("topK"))
                    if (!$util.isInteger(message.topK))
                        return "topK: integer expected";
                if (message.axis != null && message.hasOwnProperty("axis"))
                    if (!$util.isInteger(message.axis))
                        return "axis: integer expected";
                if (message.ignoreLabel != null && message.hasOwnProperty("ignoreLabel"))
                    if (!$util.isInteger(message.ignoreLabel))
                        return "ignoreLabel: integer expected";
                return null;
            };
    
            AccuracyParameter.fromObject = function fromObject(object) {
                if (object instanceof $root.caffe.AccuracyParameter)
                    return object;
                var message = new $root.caffe.AccuracyParameter();
                if (object.topK != null)
                    message.topK = object.topK >>> 0;
                if (object.axis != null)
                    message.axis = object.axis | 0;
                if (object.ignoreLabel != null)
                    message.ignoreLabel = object.ignoreLabel | 0;
                return message;
            };
    
            AccuracyParameter.toObject = function toObject(message, options) {
                if (!options)
                    options = {};
                var object = {};
                if (options.defaults) {
                    object.topK = 1;
                    object.axis = 1;
                    object.ignoreLabel = 0;
                }
                if (message.topK != null && message.hasOwnProperty("topK"))
                    object.topK = message.topK;
                if (message.axis != null && message.hasOwnProperty("axis"))
                    object.axis = message.axis;
                if (message.ignoreLabel != null && message.hasOwnProperty("ignoreLabel"))
                    object.ignoreLabel = message.ignoreLabel;
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
    
            ArgMaxParameter.prototype.outMaxVal = false;
            ArgMaxParameter.prototype.topK = 1;
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
                        message.outMaxVal = reader.bool();
                        break;
                    case 2:
                        message.topK = reader.uint32();
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
    
            ArgMaxParameter.verify = function verify(message) {
                if (typeof message !== "object" || message === null)
                    return "object expected";
                if (message.outMaxVal != null && message.hasOwnProperty("outMaxVal"))
                    if (typeof message.outMaxVal !== "boolean")
                        return "outMaxVal: boolean expected";
                if (message.topK != null && message.hasOwnProperty("topK"))
                    if (!$util.isInteger(message.topK))
                        return "topK: integer expected";
                if (message.axis != null && message.hasOwnProperty("axis"))
                    if (!$util.isInteger(message.axis))
                        return "axis: integer expected";
                return null;
            };
    
            ArgMaxParameter.fromObject = function fromObject(object) {
                if (object instanceof $root.caffe.ArgMaxParameter)
                    return object;
                var message = new $root.caffe.ArgMaxParameter();
                if (object.outMaxVal != null)
                    message.outMaxVal = Boolean(object.outMaxVal);
                if (object.topK != null)
                    message.topK = object.topK >>> 0;
                if (object.axis != null)
                    message.axis = object.axis | 0;
                return message;
            };
    
            ArgMaxParameter.toObject = function toObject(message, options) {
                if (!options)
                    options = {};
                var object = {};
                if (options.defaults) {
                    object.outMaxVal = false;
                    object.topK = 1;
                    object.axis = 0;
                }
                if (message.outMaxVal != null && message.hasOwnProperty("outMaxVal"))
                    object.outMaxVal = message.outMaxVal;
                if (message.topK != null && message.hasOwnProperty("topK"))
                    object.topK = message.topK;
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
            ConcatParameter.prototype.concatDim = 1;
    
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
                        message.concatDim = reader.uint32();
                        break;
                    default:
                        reader.skipType(tag & 7);
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
                if (message.concatDim != null && message.hasOwnProperty("concatDim"))
                    if (!$util.isInteger(message.concatDim))
                        return "concatDim: integer expected";
                return null;
            };
    
            ConcatParameter.fromObject = function fromObject(object) {
                if (object instanceof $root.caffe.ConcatParameter)
                    return object;
                var message = new $root.caffe.ConcatParameter();
                if (object.axis != null)
                    message.axis = object.axis | 0;
                if (object.concatDim != null)
                    message.concatDim = object.concatDim >>> 0;
                return message;
            };
    
            ConcatParameter.toObject = function toObject(message, options) {
                if (!options)
                    options = {};
                var object = {};
                if (options.defaults) {
                    object.concatDim = 1;
                    object.axis = 1;
                }
                if (message.concatDim != null && message.hasOwnProperty("concatDim"))
                    object.concatDim = message.concatDim;
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
    
            BatchNormParameter.prototype.useGlobalStats = false;
            BatchNormParameter.prototype.movingAverageFraction = 0.999;
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
                        message.useGlobalStats = reader.bool();
                        break;
                    case 2:
                        message.movingAverageFraction = reader.float();
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
    
            BatchNormParameter.verify = function verify(message) {
                if (typeof message !== "object" || message === null)
                    return "object expected";
                if (message.useGlobalStats != null && message.hasOwnProperty("useGlobalStats"))
                    if (typeof message.useGlobalStats !== "boolean")
                        return "useGlobalStats: boolean expected";
                if (message.movingAverageFraction != null && message.hasOwnProperty("movingAverageFraction"))
                    if (typeof message.movingAverageFraction !== "number")
                        return "movingAverageFraction: number expected";
                if (message.eps != null && message.hasOwnProperty("eps"))
                    if (typeof message.eps !== "number")
                        return "eps: number expected";
                return null;
            };
    
            BatchNormParameter.fromObject = function fromObject(object) {
                if (object instanceof $root.caffe.BatchNormParameter)
                    return object;
                var message = new $root.caffe.BatchNormParameter();
                if (object.useGlobalStats != null)
                    message.useGlobalStats = Boolean(object.useGlobalStats);
                if (object.movingAverageFraction != null)
                    message.movingAverageFraction = Number(object.movingAverageFraction);
                if (object.eps != null)
                    message.eps = Number(object.eps);
                return message;
            };
    
            BatchNormParameter.toObject = function toObject(message, options) {
                if (!options)
                    options = {};
                var object = {};
                if (options.defaults) {
                    object.useGlobalStats = false;
                    object.movingAverageFraction = 0.999;
                    object.eps = 0.00001;
                }
                if (message.useGlobalStats != null && message.hasOwnProperty("useGlobalStats"))
                    object.useGlobalStats = message.useGlobalStats;
                if (message.movingAverageFraction != null && message.hasOwnProperty("movingAverageFraction"))
                    object.movingAverageFraction = options.json && !isFinite(message.movingAverageFraction) ? String(message.movingAverageFraction) : message.movingAverageFraction;
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
            BiasParameter.prototype.numAxes = 1;
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
                        message.numAxes = reader.int32();
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
    
            BiasParameter.verify = function verify(message) {
                if (typeof message !== "object" || message === null)
                    return "object expected";
                if (message.axis != null && message.hasOwnProperty("axis"))
                    if (!$util.isInteger(message.axis))
                        return "axis: integer expected";
                if (message.numAxes != null && message.hasOwnProperty("numAxes"))
                    if (!$util.isInteger(message.numAxes))
                        return "numAxes: integer expected";
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
                if (object.numAxes != null)
                    message.numAxes = object.numAxes | 0;
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
                    object.numAxes = 1;
                    object.filler = null;
                }
                if (message.axis != null && message.hasOwnProperty("axis"))
                    object.axis = message.axis;
                if (message.numAxes != null && message.hasOwnProperty("numAxes"))
                    object.numAxes = message.numAxes;
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
            ContrastiveLossParameter.prototype.legacyVersion = false;
    
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
                        message.legacyVersion = reader.bool();
                        break;
                    default:
                        reader.skipType(tag & 7);
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
                if (message.legacyVersion != null && message.hasOwnProperty("legacyVersion"))
                    if (typeof message.legacyVersion !== "boolean")
                        return "legacyVersion: boolean expected";
                return null;
            };
    
            ContrastiveLossParameter.fromObject = function fromObject(object) {
                if (object instanceof $root.caffe.ContrastiveLossParameter)
                    return object;
                var message = new $root.caffe.ContrastiveLossParameter();
                if (object.margin != null)
                    message.margin = Number(object.margin);
                if (object.legacyVersion != null)
                    message.legacyVersion = Boolean(object.legacyVersion);
                return message;
            };
    
            ContrastiveLossParameter.toObject = function toObject(message, options) {
                if (!options)
                    options = {};
                var object = {};
                if (options.defaults) {
                    object.margin = 1;
                    object.legacyVersion = false;
                }
                if (message.margin != null && message.hasOwnProperty("margin"))
                    object.margin = options.json && !isFinite(message.margin) ? String(message.margin) : message.margin;
                if (message.legacyVersion != null && message.hasOwnProperty("legacyVersion"))
                    object.legacyVersion = message.legacyVersion;
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
                this.kernelSize = [];
                this.stride = [];
                this.dilation = [];
                if (properties)
                    for (var keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                        if (properties[keys[i]] != null)
                            this[keys[i]] = properties[keys[i]];
            }
    
            ConvolutionParameter.prototype.numOutput = 0;
            ConvolutionParameter.prototype.biasTerm = true;
            ConvolutionParameter.prototype.pad = $util.emptyArray;
            ConvolutionParameter.prototype.kernelSize = $util.emptyArray;
            ConvolutionParameter.prototype.stride = $util.emptyArray;
            ConvolutionParameter.prototype.dilation = $util.emptyArray;
            ConvolutionParameter.prototype.padH = 0;
            ConvolutionParameter.prototype.padW = 0;
            ConvolutionParameter.prototype.kernelH = 0;
            ConvolutionParameter.prototype.kernelW = 0;
            ConvolutionParameter.prototype.strideH = 0;
            ConvolutionParameter.prototype.strideW = 0;
            ConvolutionParameter.prototype.group = 1;
            ConvolutionParameter.prototype.weightFiller = null;
            ConvolutionParameter.prototype.biasFiller = null;
            ConvolutionParameter.prototype.engine = 0;
            ConvolutionParameter.prototype.axis = 1;
            ConvolutionParameter.prototype.forceNdIm2col = false;
    
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
                        message.numOutput = reader.uint32();
                        break;
                    case 2:
                        message.biasTerm = reader.bool();
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
                        if (!(message.kernelSize && message.kernelSize.length))
                            message.kernelSize = [];
                        if ((tag & 7) === 2) {
                            var end2 = reader.uint32() + reader.pos;
                            while (reader.pos < end2)
                                message.kernelSize.push(reader.uint32());
                        } else
                            message.kernelSize.push(reader.uint32());
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
                        message.padH = reader.uint32();
                        break;
                    case 10:
                        message.padW = reader.uint32();
                        break;
                    case 11:
                        message.kernelH = reader.uint32();
                        break;
                    case 12:
                        message.kernelW = reader.uint32();
                        break;
                    case 13:
                        message.strideH = reader.uint32();
                        break;
                    case 14:
                        message.strideW = reader.uint32();
                        break;
                    case 5:
                        message.group = reader.uint32();
                        break;
                    case 7:
                        message.weightFiller = $root.caffe.FillerParameter.decode(reader, reader.uint32());
                        break;
                    case 8:
                        message.biasFiller = $root.caffe.FillerParameter.decode(reader, reader.uint32());
                        break;
                    case 15:
                        message.engine = reader.int32();
                        break;
                    case 16:
                        message.axis = reader.int32();
                        break;
                    case 17:
                        message.forceNdIm2col = reader.bool();
                        break;
                    default:
                        reader.skipType(tag & 7);
                        break;
                    }
                }
                return message;
            };
    
            ConvolutionParameter.verify = function verify(message) {
                if (typeof message !== "object" || message === null)
                    return "object expected";
                if (message.numOutput != null && message.hasOwnProperty("numOutput"))
                    if (!$util.isInteger(message.numOutput))
                        return "numOutput: integer expected";
                if (message.biasTerm != null && message.hasOwnProperty("biasTerm"))
                    if (typeof message.biasTerm !== "boolean")
                        return "biasTerm: boolean expected";
                if (message.pad != null && message.hasOwnProperty("pad")) {
                    if (!Array.isArray(message.pad))
                        return "pad: array expected";
                    for (var i = 0; i < message.pad.length; ++i)
                        if (!$util.isInteger(message.pad[i]))
                            return "pad: integer[] expected";
                }
                if (message.kernelSize != null && message.hasOwnProperty("kernelSize")) {
                    if (!Array.isArray(message.kernelSize))
                        return "kernelSize: array expected";
                    for (var i = 0; i < message.kernelSize.length; ++i)
                        if (!$util.isInteger(message.kernelSize[i]))
                            return "kernelSize: integer[] expected";
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
                if (message.padH != null && message.hasOwnProperty("padH"))
                    if (!$util.isInteger(message.padH))
                        return "padH: integer expected";
                if (message.padW != null && message.hasOwnProperty("padW"))
                    if (!$util.isInteger(message.padW))
                        return "padW: integer expected";
                if (message.kernelH != null && message.hasOwnProperty("kernelH"))
                    if (!$util.isInteger(message.kernelH))
                        return "kernelH: integer expected";
                if (message.kernelW != null && message.hasOwnProperty("kernelW"))
                    if (!$util.isInteger(message.kernelW))
                        return "kernelW: integer expected";
                if (message.strideH != null && message.hasOwnProperty("strideH"))
                    if (!$util.isInteger(message.strideH))
                        return "strideH: integer expected";
                if (message.strideW != null && message.hasOwnProperty("strideW"))
                    if (!$util.isInteger(message.strideW))
                        return "strideW: integer expected";
                if (message.group != null && message.hasOwnProperty("group"))
                    if (!$util.isInteger(message.group))
                        return "group: integer expected";
                if (message.weightFiller != null && message.hasOwnProperty("weightFiller")) {
                    var error = $root.caffe.FillerParameter.verify(message.weightFiller);
                    if (error)
                        return "weightFiller." + error;
                }
                if (message.biasFiller != null && message.hasOwnProperty("biasFiller")) {
                    var error = $root.caffe.FillerParameter.verify(message.biasFiller);
                    if (error)
                        return "biasFiller." + error;
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
                if (message.forceNdIm2col != null && message.hasOwnProperty("forceNdIm2col"))
                    if (typeof message.forceNdIm2col !== "boolean")
                        return "forceNdIm2col: boolean expected";
                return null;
            };
    
            ConvolutionParameter.fromObject = function fromObject(object) {
                if (object instanceof $root.caffe.ConvolutionParameter)
                    return object;
                var message = new $root.caffe.ConvolutionParameter();
                if (object.numOutput != null)
                    message.numOutput = object.numOutput >>> 0;
                if (object.biasTerm != null)
                    message.biasTerm = Boolean(object.biasTerm);
                if (object.pad) {
                    if (!Array.isArray(object.pad))
                        throw TypeError(".caffe.ConvolutionParameter.pad: array expected");
                    message.pad = [];
                    for (var i = 0; i < object.pad.length; ++i)
                        message.pad[i] = object.pad[i] >>> 0;
                }
                if (object.kernelSize) {
                    if (!Array.isArray(object.kernelSize))
                        throw TypeError(".caffe.ConvolutionParameter.kernelSize: array expected");
                    message.kernelSize = [];
                    for (var i = 0; i < object.kernelSize.length; ++i)
                        message.kernelSize[i] = object.kernelSize[i] >>> 0;
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
                if (object.padH != null)
                    message.padH = object.padH >>> 0;
                if (object.padW != null)
                    message.padW = object.padW >>> 0;
                if (object.kernelH != null)
                    message.kernelH = object.kernelH >>> 0;
                if (object.kernelW != null)
                    message.kernelW = object.kernelW >>> 0;
                if (object.strideH != null)
                    message.strideH = object.strideH >>> 0;
                if (object.strideW != null)
                    message.strideW = object.strideW >>> 0;
                if (object.group != null)
                    message.group = object.group >>> 0;
                if (object.weightFiller != null) {
                    if (typeof object.weightFiller !== "object")
                        throw TypeError(".caffe.ConvolutionParameter.weightFiller: object expected");
                    message.weightFiller = $root.caffe.FillerParameter.fromObject(object.weightFiller);
                }
                if (object.biasFiller != null) {
                    if (typeof object.biasFiller !== "object")
                        throw TypeError(".caffe.ConvolutionParameter.biasFiller: object expected");
                    message.biasFiller = $root.caffe.FillerParameter.fromObject(object.biasFiller);
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
                if (object.forceNdIm2col != null)
                    message.forceNdIm2col = Boolean(object.forceNdIm2col);
                return message;
            };
    
            ConvolutionParameter.toObject = function toObject(message, options) {
                if (!options)
                    options = {};
                var object = {};
                if (options.arrays || options.defaults) {
                    object.pad = [];
                    object.kernelSize = [];
                    object.stride = [];
                    object.dilation = [];
                }
                if (options.defaults) {
                    object.numOutput = 0;
                    object.biasTerm = true;
                    object.group = 1;
                    object.weightFiller = null;
                    object.biasFiller = null;
                    object.padH = 0;
                    object.padW = 0;
                    object.kernelH = 0;
                    object.kernelW = 0;
                    object.strideH = 0;
                    object.strideW = 0;
                    object.engine = options.enums === String ? "DEFAULT" : 0;
                    object.axis = 1;
                    object.forceNdIm2col = false;
                }
                if (message.numOutput != null && message.hasOwnProperty("numOutput"))
                    object.numOutput = message.numOutput;
                if (message.biasTerm != null && message.hasOwnProperty("biasTerm"))
                    object.biasTerm = message.biasTerm;
                if (message.pad && message.pad.length) {
                    object.pad = [];
                    for (var j = 0; j < message.pad.length; ++j)
                        object.pad[j] = message.pad[j];
                }
                if (message.kernelSize && message.kernelSize.length) {
                    object.kernelSize = [];
                    for (var j = 0; j < message.kernelSize.length; ++j)
                        object.kernelSize[j] = message.kernelSize[j];
                }
                if (message.group != null && message.hasOwnProperty("group"))
                    object.group = message.group;
                if (message.stride && message.stride.length) {
                    object.stride = [];
                    for (var j = 0; j < message.stride.length; ++j)
                        object.stride[j] = message.stride[j];
                }
                if (message.weightFiller != null && message.hasOwnProperty("weightFiller"))
                    object.weightFiller = $root.caffe.FillerParameter.toObject(message.weightFiller, options);
                if (message.biasFiller != null && message.hasOwnProperty("biasFiller"))
                    object.biasFiller = $root.caffe.FillerParameter.toObject(message.biasFiller, options);
                if (message.padH != null && message.hasOwnProperty("padH"))
                    object.padH = message.padH;
                if (message.padW != null && message.hasOwnProperty("padW"))
                    object.padW = message.padW;
                if (message.kernelH != null && message.hasOwnProperty("kernelH"))
                    object.kernelH = message.kernelH;
                if (message.kernelW != null && message.hasOwnProperty("kernelW"))
                    object.kernelW = message.kernelW;
                if (message.strideH != null && message.hasOwnProperty("strideH"))
                    object.strideH = message.strideH;
                if (message.strideW != null && message.hasOwnProperty("strideW"))
                    object.strideW = message.strideW;
                if (message.engine != null && message.hasOwnProperty("engine"))
                    object.engine = options.enums === String ? $root.caffe.ConvolutionParameter.Engine[message.engine] : message.engine;
                if (message.axis != null && message.hasOwnProperty("axis"))
                    object.axis = message.axis;
                if (message.forceNdIm2col != null && message.hasOwnProperty("forceNdIm2col"))
                    object.forceNdIm2col = message.forceNdIm2col;
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
            DataParameter.prototype.batchSize = 0;
            DataParameter.prototype.randSkip = 0;
            DataParameter.prototype.backend = 0;
            DataParameter.prototype.scale = 1;
            DataParameter.prototype.meanFile = "";
            DataParameter.prototype.cropSize = 0;
            DataParameter.prototype.mirror = false;
            DataParameter.prototype.forceEncodedColor = false;
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
                        message.batchSize = reader.uint32();
                        break;
                    case 7:
                        message.randSkip = reader.uint32();
                        break;
                    case 8:
                        message.backend = reader.int32();
                        break;
                    case 2:
                        message.scale = reader.float();
                        break;
                    case 3:
                        message.meanFile = reader.string();
                        break;
                    case 5:
                        message.cropSize = reader.uint32();
                        break;
                    case 6:
                        message.mirror = reader.bool();
                        break;
                    case 9:
                        message.forceEncodedColor = reader.bool();
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
    
            DataParameter.verify = function verify(message) {
                if (typeof message !== "object" || message === null)
                    return "object expected";
                if (message.source != null && message.hasOwnProperty("source"))
                    if (!$util.isString(message.source))
                        return "source: string expected";
                if (message.batchSize != null && message.hasOwnProperty("batchSize"))
                    if (!$util.isInteger(message.batchSize))
                        return "batchSize: integer expected";
                if (message.randSkip != null && message.hasOwnProperty("randSkip"))
                    if (!$util.isInteger(message.randSkip))
                        return "randSkip: integer expected";
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
                if (message.meanFile != null && message.hasOwnProperty("meanFile"))
                    if (!$util.isString(message.meanFile))
                        return "meanFile: string expected";
                if (message.cropSize != null && message.hasOwnProperty("cropSize"))
                    if (!$util.isInteger(message.cropSize))
                        return "cropSize: integer expected";
                if (message.mirror != null && message.hasOwnProperty("mirror"))
                    if (typeof message.mirror !== "boolean")
                        return "mirror: boolean expected";
                if (message.forceEncodedColor != null && message.hasOwnProperty("forceEncodedColor"))
                    if (typeof message.forceEncodedColor !== "boolean")
                        return "forceEncodedColor: boolean expected";
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
                if (object.batchSize != null)
                    message.batchSize = object.batchSize >>> 0;
                if (object.randSkip != null)
                    message.randSkip = object.randSkip >>> 0;
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
                if (object.meanFile != null)
                    message.meanFile = String(object.meanFile);
                if (object.cropSize != null)
                    message.cropSize = object.cropSize >>> 0;
                if (object.mirror != null)
                    message.mirror = Boolean(object.mirror);
                if (object.forceEncodedColor != null)
                    message.forceEncodedColor = Boolean(object.forceEncodedColor);
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
                    object.meanFile = "";
                    object.batchSize = 0;
                    object.cropSize = 0;
                    object.mirror = false;
                    object.randSkip = 0;
                    object.backend = options.enums === String ? "LEVELDB" : 0;
                    object.forceEncodedColor = false;
                    object.prefetch = 4;
                }
                if (message.source != null && message.hasOwnProperty("source"))
                    object.source = message.source;
                if (message.scale != null && message.hasOwnProperty("scale"))
                    object.scale = options.json && !isFinite(message.scale) ? String(message.scale) : message.scale;
                if (message.meanFile != null && message.hasOwnProperty("meanFile"))
                    object.meanFile = message.meanFile;
                if (message.batchSize != null && message.hasOwnProperty("batchSize"))
                    object.batchSize = message.batchSize;
                if (message.cropSize != null && message.hasOwnProperty("cropSize"))
                    object.cropSize = message.cropSize;
                if (message.mirror != null && message.hasOwnProperty("mirror"))
                    object.mirror = message.mirror;
                if (message.randSkip != null && message.hasOwnProperty("randSkip"))
                    object.randSkip = message.randSkip;
                if (message.backend != null && message.hasOwnProperty("backend"))
                    object.backend = options.enums === String ? $root.caffe.DataParameter.DB[message.backend] : message.backend;
                if (message.forceEncodedColor != null && message.hasOwnProperty("forceEncodedColor"))
                    object.forceEncodedColor = message.forceEncodedColor;
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
    
            DropoutParameter.prototype.dropoutRatio = 0.5;
    
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
                        message.dropoutRatio = reader.float();
                        break;
                    default:
                        reader.skipType(tag & 7);
                        break;
                    }
                }
                return message;
            };
    
            DropoutParameter.verify = function verify(message) {
                if (typeof message !== "object" || message === null)
                    return "object expected";
                if (message.dropoutRatio != null && message.hasOwnProperty("dropoutRatio"))
                    if (typeof message.dropoutRatio !== "number")
                        return "dropoutRatio: number expected";
                return null;
            };
    
            DropoutParameter.fromObject = function fromObject(object) {
                if (object instanceof $root.caffe.DropoutParameter)
                    return object;
                var message = new $root.caffe.DropoutParameter();
                if (object.dropoutRatio != null)
                    message.dropoutRatio = Number(object.dropoutRatio);
                return message;
            };
    
            DropoutParameter.toObject = function toObject(message, options) {
                if (!options)
                    options = {};
                var object = {};
                if (options.defaults)
                    object.dropoutRatio = 0.5;
                if (message.dropoutRatio != null && message.hasOwnProperty("dropoutRatio"))
                    object.dropoutRatio = options.json && !isFinite(message.dropoutRatio) ? String(message.dropoutRatio) : message.dropoutRatio;
                return object;
            };
    
            DropoutParameter.prototype.toJSON = function toJSON() {
                return this.constructor.toObject(this, $protobuf.util.toJSONOptions);
            };
    
            return DropoutParameter;
        })();
    
        caffe.DummyDataParameter = (function() {
    
            function DummyDataParameter(properties) {
                this.dataFiller = [];
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
    
            DummyDataParameter.prototype.dataFiller = $util.emptyArray;
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
                        if (!(message.dataFiller && message.dataFiller.length))
                            message.dataFiller = [];
                        message.dataFiller.push($root.caffe.FillerParameter.decode(reader, reader.uint32()));
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
    
            DummyDataParameter.verify = function verify(message) {
                if (typeof message !== "object" || message === null)
                    return "object expected";
                if (message.dataFiller != null && message.hasOwnProperty("dataFiller")) {
                    if (!Array.isArray(message.dataFiller))
                        return "dataFiller: array expected";
                    for (var i = 0; i < message.dataFiller.length; ++i) {
                        var error = $root.caffe.FillerParameter.verify(message.dataFiller[i]);
                        if (error)
                            return "dataFiller." + error;
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
                if (object.dataFiller) {
                    if (!Array.isArray(object.dataFiller))
                        throw TypeError(".caffe.DummyDataParameter.dataFiller: array expected");
                    message.dataFiller = [];
                    for (var i = 0; i < object.dataFiller.length; ++i) {
                        if (typeof object.dataFiller[i] !== "object")
                            throw TypeError(".caffe.DummyDataParameter.dataFiller: object expected");
                        message.dataFiller[i] = $root.caffe.FillerParameter.fromObject(object.dataFiller[i]);
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
                    object.dataFiller = [];
                    object.num = [];
                    object.channels = [];
                    object.height = [];
                    object.width = [];
                    object.shape = [];
                }
                if (message.dataFiller && message.dataFiller.length) {
                    object.dataFiller = [];
                    for (var j = 0; j < message.dataFiller.length; ++j)
                        object.dataFiller[j] = $root.caffe.FillerParameter.toObject(message.dataFiller[j], options);
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
            EltwiseParameter.prototype.stableProdGrad = true;
    
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
                        message.stableProdGrad = reader.bool();
                        break;
                    default:
                        reader.skipType(tag & 7);
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
                if (message.stableProdGrad != null && message.hasOwnProperty("stableProdGrad"))
                    if (typeof message.stableProdGrad !== "boolean")
                        return "stableProdGrad: boolean expected";
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
                if (object.stableProdGrad != null)
                    message.stableProdGrad = Boolean(object.stableProdGrad);
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
                    object.stableProdGrad = true;
                }
                if (message.operation != null && message.hasOwnProperty("operation"))
                    object.operation = options.enums === String ? $root.caffe.EltwiseParameter.EltwiseOp[message.operation] : message.operation;
                if (message.coeff && message.coeff.length) {
                    object.coeff = [];
                    for (var j = 0; j < message.coeff.length; ++j)
                        object.coeff[j] = options.json && !isFinite(message.coeff[j]) ? String(message.coeff[j]) : message.coeff[j];
                }
                if (message.stableProdGrad != null && message.hasOwnProperty("stableProdGrad"))
                    object.stableProdGrad = message.stableProdGrad;
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
    
            EmbedParameter.prototype.numOutput = 0;
            EmbedParameter.prototype.inputDim = 0;
            EmbedParameter.prototype.biasTerm = true;
            EmbedParameter.prototype.weightFiller = null;
            EmbedParameter.prototype.biasFiller = null;
    
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
                        message.numOutput = reader.uint32();
                        break;
                    case 2:
                        message.inputDim = reader.uint32();
                        break;
                    case 3:
                        message.biasTerm = reader.bool();
                        break;
                    case 4:
                        message.weightFiller = $root.caffe.FillerParameter.decode(reader, reader.uint32());
                        break;
                    case 5:
                        message.biasFiller = $root.caffe.FillerParameter.decode(reader, reader.uint32());
                        break;
                    default:
                        reader.skipType(tag & 7);
                        break;
                    }
                }
                return message;
            };
    
            EmbedParameter.verify = function verify(message) {
                if (typeof message !== "object" || message === null)
                    return "object expected";
                if (message.numOutput != null && message.hasOwnProperty("numOutput"))
                    if (!$util.isInteger(message.numOutput))
                        return "numOutput: integer expected";
                if (message.inputDim != null && message.hasOwnProperty("inputDim"))
                    if (!$util.isInteger(message.inputDim))
                        return "inputDim: integer expected";
                if (message.biasTerm != null && message.hasOwnProperty("biasTerm"))
                    if (typeof message.biasTerm !== "boolean")
                        return "biasTerm: boolean expected";
                if (message.weightFiller != null && message.hasOwnProperty("weightFiller")) {
                    var error = $root.caffe.FillerParameter.verify(message.weightFiller);
                    if (error)
                        return "weightFiller." + error;
                }
                if (message.biasFiller != null && message.hasOwnProperty("biasFiller")) {
                    var error = $root.caffe.FillerParameter.verify(message.biasFiller);
                    if (error)
                        return "biasFiller." + error;
                }
                return null;
            };
    
            EmbedParameter.fromObject = function fromObject(object) {
                if (object instanceof $root.caffe.EmbedParameter)
                    return object;
                var message = new $root.caffe.EmbedParameter();
                if (object.numOutput != null)
                    message.numOutput = object.numOutput >>> 0;
                if (object.inputDim != null)
                    message.inputDim = object.inputDim >>> 0;
                if (object.biasTerm != null)
                    message.biasTerm = Boolean(object.biasTerm);
                if (object.weightFiller != null) {
                    if (typeof object.weightFiller !== "object")
                        throw TypeError(".caffe.EmbedParameter.weightFiller: object expected");
                    message.weightFiller = $root.caffe.FillerParameter.fromObject(object.weightFiller);
                }
                if (object.biasFiller != null) {
                    if (typeof object.biasFiller !== "object")
                        throw TypeError(".caffe.EmbedParameter.biasFiller: object expected");
                    message.biasFiller = $root.caffe.FillerParameter.fromObject(object.biasFiller);
                }
                return message;
            };
    
            EmbedParameter.toObject = function toObject(message, options) {
                if (!options)
                    options = {};
                var object = {};
                if (options.defaults) {
                    object.numOutput = 0;
                    object.inputDim = 0;
                    object.biasTerm = true;
                    object.weightFiller = null;
                    object.biasFiller = null;
                }
                if (message.numOutput != null && message.hasOwnProperty("numOutput"))
                    object.numOutput = message.numOutput;
                if (message.inputDim != null && message.hasOwnProperty("inputDim"))
                    object.inputDim = message.inputDim;
                if (message.biasTerm != null && message.hasOwnProperty("biasTerm"))
                    object.biasTerm = message.biasTerm;
                if (message.weightFiller != null && message.hasOwnProperty("weightFiller"))
                    object.weightFiller = $root.caffe.FillerParameter.toObject(message.weightFiller, options);
                if (message.biasFiller != null && message.hasOwnProperty("biasFiller"))
                    object.biasFiller = $root.caffe.FillerParameter.toObject(message.biasFiller, options);
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
            FlattenParameter.prototype.endAxis = -1;
    
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
                        message.endAxis = reader.int32();
                        break;
                    default:
                        reader.skipType(tag & 7);
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
                if (message.endAxis != null && message.hasOwnProperty("endAxis"))
                    if (!$util.isInteger(message.endAxis))
                        return "endAxis: integer expected";
                return null;
            };
    
            FlattenParameter.fromObject = function fromObject(object) {
                if (object instanceof $root.caffe.FlattenParameter)
                    return object;
                var message = new $root.caffe.FlattenParameter();
                if (object.axis != null)
                    message.axis = object.axis | 0;
                if (object.endAxis != null)
                    message.endAxis = object.endAxis | 0;
                return message;
            };
    
            FlattenParameter.toObject = function toObject(message, options) {
                if (!options)
                    options = {};
                var object = {};
                if (options.defaults) {
                    object.axis = 1;
                    object.endAxis = -1;
                }
                if (message.axis != null && message.hasOwnProperty("axis"))
                    object.axis = message.axis;
                if (message.endAxis != null && message.hasOwnProperty("endAxis"))
                    object.endAxis = message.endAxis;
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
            HDF5DataParameter.prototype.batchSize = 0;
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
                        message.batchSize = reader.uint32();
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
    
            HDF5DataParameter.verify = function verify(message) {
                if (typeof message !== "object" || message === null)
                    return "object expected";
                if (message.source != null && message.hasOwnProperty("source"))
                    if (!$util.isString(message.source))
                        return "source: string expected";
                if (message.batchSize != null && message.hasOwnProperty("batchSize"))
                    if (!$util.isInteger(message.batchSize))
                        return "batchSize: integer expected";
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
                if (object.batchSize != null)
                    message.batchSize = object.batchSize >>> 0;
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
                    object.batchSize = 0;
                    object.shuffle = false;
                }
                if (message.source != null && message.hasOwnProperty("source"))
                    object.source = message.source;
                if (message.batchSize != null && message.hasOwnProperty("batchSize"))
                    object.batchSize = message.batchSize;
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
    
            HDF5OutputParameter.prototype.fileName = "";
    
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
                        message.fileName = reader.string();
                        break;
                    default:
                        reader.skipType(tag & 7);
                        break;
                    }
                }
                return message;
            };
    
            HDF5OutputParameter.verify = function verify(message) {
                if (typeof message !== "object" || message === null)
                    return "object expected";
                if (message.fileName != null && message.hasOwnProperty("fileName"))
                    if (!$util.isString(message.fileName))
                        return "fileName: string expected";
                return null;
            };
    
            HDF5OutputParameter.fromObject = function fromObject(object) {
                if (object instanceof $root.caffe.HDF5OutputParameter)
                    return object;
                var message = new $root.caffe.HDF5OutputParameter();
                if (object.fileName != null)
                    message.fileName = String(object.fileName);
                return message;
            };
    
            HDF5OutputParameter.toObject = function toObject(message, options) {
                if (!options)
                    options = {};
                var object = {};
                if (options.defaults)
                    object.fileName = "";
                if (message.fileName != null && message.hasOwnProperty("fileName"))
                    object.fileName = message.fileName;
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
            ImageDataParameter.prototype.batchSize = 1;
            ImageDataParameter.prototype.randSkip = 0;
            ImageDataParameter.prototype.shuffle = false;
            ImageDataParameter.prototype.newHeight = 0;
            ImageDataParameter.prototype.newWidth = 0;
            ImageDataParameter.prototype.isColor = true;
            ImageDataParameter.prototype.scale = 1;
            ImageDataParameter.prototype.meanFile = "";
            ImageDataParameter.prototype.cropSize = 0;
            ImageDataParameter.prototype.mirror = false;
            ImageDataParameter.prototype.rootFolder = "";
    
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
                        message.batchSize = reader.uint32();
                        break;
                    case 7:
                        message.randSkip = reader.uint32();
                        break;
                    case 8:
                        message.shuffle = reader.bool();
                        break;
                    case 9:
                        message.newHeight = reader.uint32();
                        break;
                    case 10:
                        message.newWidth = reader.uint32();
                        break;
                    case 11:
                        message.isColor = reader.bool();
                        break;
                    case 2:
                        message.scale = reader.float();
                        break;
                    case 3:
                        message.meanFile = reader.string();
                        break;
                    case 5:
                        message.cropSize = reader.uint32();
                        break;
                    case 6:
                        message.mirror = reader.bool();
                        break;
                    case 12:
                        message.rootFolder = reader.string();
                        break;
                    default:
                        reader.skipType(tag & 7);
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
                if (message.batchSize != null && message.hasOwnProperty("batchSize"))
                    if (!$util.isInteger(message.batchSize))
                        return "batchSize: integer expected";
                if (message.randSkip != null && message.hasOwnProperty("randSkip"))
                    if (!$util.isInteger(message.randSkip))
                        return "randSkip: integer expected";
                if (message.shuffle != null && message.hasOwnProperty("shuffle"))
                    if (typeof message.shuffle !== "boolean")
                        return "shuffle: boolean expected";
                if (message.newHeight != null && message.hasOwnProperty("newHeight"))
                    if (!$util.isInteger(message.newHeight))
                        return "newHeight: integer expected";
                if (message.newWidth != null && message.hasOwnProperty("newWidth"))
                    if (!$util.isInteger(message.newWidth))
                        return "newWidth: integer expected";
                if (message.isColor != null && message.hasOwnProperty("isColor"))
                    if (typeof message.isColor !== "boolean")
                        return "isColor: boolean expected";
                if (message.scale != null && message.hasOwnProperty("scale"))
                    if (typeof message.scale !== "number")
                        return "scale: number expected";
                if (message.meanFile != null && message.hasOwnProperty("meanFile"))
                    if (!$util.isString(message.meanFile))
                        return "meanFile: string expected";
                if (message.cropSize != null && message.hasOwnProperty("cropSize"))
                    if (!$util.isInteger(message.cropSize))
                        return "cropSize: integer expected";
                if (message.mirror != null && message.hasOwnProperty("mirror"))
                    if (typeof message.mirror !== "boolean")
                        return "mirror: boolean expected";
                if (message.rootFolder != null && message.hasOwnProperty("rootFolder"))
                    if (!$util.isString(message.rootFolder))
                        return "rootFolder: string expected";
                return null;
            };
    
            ImageDataParameter.fromObject = function fromObject(object) {
                if (object instanceof $root.caffe.ImageDataParameter)
                    return object;
                var message = new $root.caffe.ImageDataParameter();
                if (object.source != null)
                    message.source = String(object.source);
                if (object.batchSize != null)
                    message.batchSize = object.batchSize >>> 0;
                if (object.randSkip != null)
                    message.randSkip = object.randSkip >>> 0;
                if (object.shuffle != null)
                    message.shuffle = Boolean(object.shuffle);
                if (object.newHeight != null)
                    message.newHeight = object.newHeight >>> 0;
                if (object.newWidth != null)
                    message.newWidth = object.newWidth >>> 0;
                if (object.isColor != null)
                    message.isColor = Boolean(object.isColor);
                if (object.scale != null)
                    message.scale = Number(object.scale);
                if (object.meanFile != null)
                    message.meanFile = String(object.meanFile);
                if (object.cropSize != null)
                    message.cropSize = object.cropSize >>> 0;
                if (object.mirror != null)
                    message.mirror = Boolean(object.mirror);
                if (object.rootFolder != null)
                    message.rootFolder = String(object.rootFolder);
                return message;
            };
    
            ImageDataParameter.toObject = function toObject(message, options) {
                if (!options)
                    options = {};
                var object = {};
                if (options.defaults) {
                    object.source = "";
                    object.scale = 1;
                    object.meanFile = "";
                    object.batchSize = 1;
                    object.cropSize = 0;
                    object.mirror = false;
                    object.randSkip = 0;
                    object.shuffle = false;
                    object.newHeight = 0;
                    object.newWidth = 0;
                    object.isColor = true;
                    object.rootFolder = "";
                }
                if (message.source != null && message.hasOwnProperty("source"))
                    object.source = message.source;
                if (message.scale != null && message.hasOwnProperty("scale"))
                    object.scale = options.json && !isFinite(message.scale) ? String(message.scale) : message.scale;
                if (message.meanFile != null && message.hasOwnProperty("meanFile"))
                    object.meanFile = message.meanFile;
                if (message.batchSize != null && message.hasOwnProperty("batchSize"))
                    object.batchSize = message.batchSize;
                if (message.cropSize != null && message.hasOwnProperty("cropSize"))
                    object.cropSize = message.cropSize;
                if (message.mirror != null && message.hasOwnProperty("mirror"))
                    object.mirror = message.mirror;
                if (message.randSkip != null && message.hasOwnProperty("randSkip"))
                    object.randSkip = message.randSkip;
                if (message.shuffle != null && message.hasOwnProperty("shuffle"))
                    object.shuffle = message.shuffle;
                if (message.newHeight != null && message.hasOwnProperty("newHeight"))
                    object.newHeight = message.newHeight;
                if (message.newWidth != null && message.hasOwnProperty("newWidth"))
                    object.newWidth = message.newWidth;
                if (message.isColor != null && message.hasOwnProperty("isColor"))
                    object.isColor = message.isColor;
                if (message.rootFolder != null && message.hasOwnProperty("rootFolder"))
                    object.rootFolder = message.rootFolder;
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
    
            InnerProductParameter.prototype.numOutput = 0;
            InnerProductParameter.prototype.biasTerm = true;
            InnerProductParameter.prototype.weightFiller = null;
            InnerProductParameter.prototype.biasFiller = null;
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
                        message.numOutput = reader.uint32();
                        break;
                    case 2:
                        message.biasTerm = reader.bool();
                        break;
                    case 3:
                        message.weightFiller = $root.caffe.FillerParameter.decode(reader, reader.uint32());
                        break;
                    case 4:
                        message.biasFiller = $root.caffe.FillerParameter.decode(reader, reader.uint32());
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
    
            InnerProductParameter.verify = function verify(message) {
                if (typeof message !== "object" || message === null)
                    return "object expected";
                if (message.numOutput != null && message.hasOwnProperty("numOutput"))
                    if (!$util.isInteger(message.numOutput))
                        return "numOutput: integer expected";
                if (message.biasTerm != null && message.hasOwnProperty("biasTerm"))
                    if (typeof message.biasTerm !== "boolean")
                        return "biasTerm: boolean expected";
                if (message.weightFiller != null && message.hasOwnProperty("weightFiller")) {
                    var error = $root.caffe.FillerParameter.verify(message.weightFiller);
                    if (error)
                        return "weightFiller." + error;
                }
                if (message.biasFiller != null && message.hasOwnProperty("biasFiller")) {
                    var error = $root.caffe.FillerParameter.verify(message.biasFiller);
                    if (error)
                        return "biasFiller." + error;
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
                if (object.numOutput != null)
                    message.numOutput = object.numOutput >>> 0;
                if (object.biasTerm != null)
                    message.biasTerm = Boolean(object.biasTerm);
                if (object.weightFiller != null) {
                    if (typeof object.weightFiller !== "object")
                        throw TypeError(".caffe.InnerProductParameter.weightFiller: object expected");
                    message.weightFiller = $root.caffe.FillerParameter.fromObject(object.weightFiller);
                }
                if (object.biasFiller != null) {
                    if (typeof object.biasFiller !== "object")
                        throw TypeError(".caffe.InnerProductParameter.biasFiller: object expected");
                    message.biasFiller = $root.caffe.FillerParameter.fromObject(object.biasFiller);
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
                    object.numOutput = 0;
                    object.biasTerm = true;
                    object.weightFiller = null;
                    object.biasFiller = null;
                    object.axis = 1;
                    object.transpose = false;
                }
                if (message.numOutput != null && message.hasOwnProperty("numOutput"))
                    object.numOutput = message.numOutput;
                if (message.biasTerm != null && message.hasOwnProperty("biasTerm"))
                    object.biasTerm = message.biasTerm;
                if (message.weightFiller != null && message.hasOwnProperty("weightFiller"))
                    object.weightFiller = $root.caffe.FillerParameter.toObject(message.weightFiller, options);
                if (message.biasFiller != null && message.hasOwnProperty("biasFiller"))
                    object.biasFiller = $root.caffe.FillerParameter.toObject(message.biasFiller, options);
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
    
            LRNParameter.prototype.localSize = 5;
            LRNParameter.prototype.alpha = 1;
            LRNParameter.prototype.beta = 0.75;
            LRNParameter.prototype.normRegion = 0;
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
                        message.localSize = reader.uint32();
                        break;
                    case 2:
                        message.alpha = reader.float();
                        break;
                    case 3:
                        message.beta = reader.float();
                        break;
                    case 4:
                        message.normRegion = reader.int32();
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
    
            LRNParameter.verify = function verify(message) {
                if (typeof message !== "object" || message === null)
                    return "object expected";
                if (message.localSize != null && message.hasOwnProperty("localSize"))
                    if (!$util.isInteger(message.localSize))
                        return "localSize: integer expected";
                if (message.alpha != null && message.hasOwnProperty("alpha"))
                    if (typeof message.alpha !== "number")
                        return "alpha: number expected";
                if (message.beta != null && message.hasOwnProperty("beta"))
                    if (typeof message.beta !== "number")
                        return "beta: number expected";
                if (message.normRegion != null && message.hasOwnProperty("normRegion"))
                    switch (message.normRegion) {
                    default:
                        return "normRegion: enum value expected";
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
                if (object.localSize != null)
                    message.localSize = object.localSize >>> 0;
                if (object.alpha != null)
                    message.alpha = Number(object.alpha);
                if (object.beta != null)
                    message.beta = Number(object.beta);
                switch (object.normRegion) {
                case "ACROSS_CHANNELS":
                case 0:
                    message.normRegion = 0;
                    break;
                case "WITHIN_CHANNEL":
                case 1:
                    message.normRegion = 1;
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
                    object.localSize = 5;
                    object.alpha = 1;
                    object.beta = 0.75;
                    object.normRegion = options.enums === String ? "ACROSS_CHANNELS" : 0;
                    object.k = 1;
                    object.engine = options.enums === String ? "DEFAULT" : 0;
                }
                if (message.localSize != null && message.hasOwnProperty("localSize"))
                    object.localSize = message.localSize;
                if (message.alpha != null && message.hasOwnProperty("alpha"))
                    object.alpha = options.json && !isFinite(message.alpha) ? String(message.alpha) : message.alpha;
                if (message.beta != null && message.hasOwnProperty("beta"))
                    object.beta = options.json && !isFinite(message.beta) ? String(message.beta) : message.beta;
                if (message.normRegion != null && message.hasOwnProperty("normRegion"))
                    object.normRegion = options.enums === String ? $root.caffe.LRNParameter.NormRegion[message.normRegion] : message.normRegion;
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
    
            MemoryDataParameter.prototype.batchSize = 0;
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
                        message.batchSize = reader.uint32();
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
    
            MemoryDataParameter.verify = function verify(message) {
                if (typeof message !== "object" || message === null)
                    return "object expected";
                if (message.batchSize != null && message.hasOwnProperty("batchSize"))
                    if (!$util.isInteger(message.batchSize))
                        return "batchSize: integer expected";
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
                if (object.batchSize != null)
                    message.batchSize = object.batchSize >>> 0;
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
                    object.batchSize = 0;
                    object.channels = 0;
                    object.height = 0;
                    object.width = 0;
                }
                if (message.batchSize != null && message.hasOwnProperty("batchSize"))
                    object.batchSize = message.batchSize;
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
    
            MVNParameter.prototype.normalizeVariance = true;
            MVNParameter.prototype.acrossChannels = false;
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
                        message.normalizeVariance = reader.bool();
                        break;
                    case 2:
                        message.acrossChannels = reader.bool();
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
    
            MVNParameter.verify = function verify(message) {
                if (typeof message !== "object" || message === null)
                    return "object expected";
                if (message.normalizeVariance != null && message.hasOwnProperty("normalizeVariance"))
                    if (typeof message.normalizeVariance !== "boolean")
                        return "normalizeVariance: boolean expected";
                if (message.acrossChannels != null && message.hasOwnProperty("acrossChannels"))
                    if (typeof message.acrossChannels !== "boolean")
                        return "acrossChannels: boolean expected";
                if (message.eps != null && message.hasOwnProperty("eps"))
                    if (typeof message.eps !== "number")
                        return "eps: number expected";
                return null;
            };
    
            MVNParameter.fromObject = function fromObject(object) {
                if (object instanceof $root.caffe.MVNParameter)
                    return object;
                var message = new $root.caffe.MVNParameter();
                if (object.normalizeVariance != null)
                    message.normalizeVariance = Boolean(object.normalizeVariance);
                if (object.acrossChannels != null)
                    message.acrossChannels = Boolean(object.acrossChannels);
                if (object.eps != null)
                    message.eps = Number(object.eps);
                return message;
            };
    
            MVNParameter.toObject = function toObject(message, options) {
                if (!options)
                    options = {};
                var object = {};
                if (options.defaults) {
                    object.normalizeVariance = true;
                    object.acrossChannels = false;
                    object.eps = 1e-9;
                }
                if (message.normalizeVariance != null && message.hasOwnProperty("normalizeVariance"))
                    object.normalizeVariance = message.normalizeVariance;
                if (message.acrossChannels != null && message.hasOwnProperty("acrossChannels"))
                    object.acrossChannels = message.acrossChannels;
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
            PoolingParameter.prototype.padH = 0;
            PoolingParameter.prototype.padW = 0;
            PoolingParameter.prototype.kernelSize = 0;
            PoolingParameter.prototype.kernelH = 0;
            PoolingParameter.prototype.kernelW = 0;
            PoolingParameter.prototype.stride = 1;
            PoolingParameter.prototype.strideH = 0;
            PoolingParameter.prototype.strideW = 0;
            PoolingParameter.prototype.engine = 0;
            PoolingParameter.prototype.globalPooling = false;
            PoolingParameter.prototype.roundMode = 0;
    
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
                        message.padH = reader.uint32();
                        break;
                    case 10:
                        message.padW = reader.uint32();
                        break;
                    case 2:
                        message.kernelSize = reader.uint32();
                        break;
                    case 5:
                        message.kernelH = reader.uint32();
                        break;
                    case 6:
                        message.kernelW = reader.uint32();
                        break;
                    case 3:
                        message.stride = reader.uint32();
                        break;
                    case 7:
                        message.strideH = reader.uint32();
                        break;
                    case 8:
                        message.strideW = reader.uint32();
                        break;
                    case 11:
                        message.engine = reader.int32();
                        break;
                    case 12:
                        message.globalPooling = reader.bool();
                        break;
                    case 13:
                        message.roundMode = reader.int32();
                        break;
                    default:
                        reader.skipType(tag & 7);
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
                if (message.padH != null && message.hasOwnProperty("padH"))
                    if (!$util.isInteger(message.padH))
                        return "padH: integer expected";
                if (message.padW != null && message.hasOwnProperty("padW"))
                    if (!$util.isInteger(message.padW))
                        return "padW: integer expected";
                if (message.kernelSize != null && message.hasOwnProperty("kernelSize"))
                    if (!$util.isInteger(message.kernelSize))
                        return "kernelSize: integer expected";
                if (message.kernelH != null && message.hasOwnProperty("kernelH"))
                    if (!$util.isInteger(message.kernelH))
                        return "kernelH: integer expected";
                if (message.kernelW != null && message.hasOwnProperty("kernelW"))
                    if (!$util.isInteger(message.kernelW))
                        return "kernelW: integer expected";
                if (message.stride != null && message.hasOwnProperty("stride"))
                    if (!$util.isInteger(message.stride))
                        return "stride: integer expected";
                if (message.strideH != null && message.hasOwnProperty("strideH"))
                    if (!$util.isInteger(message.strideH))
                        return "strideH: integer expected";
                if (message.strideW != null && message.hasOwnProperty("strideW"))
                    if (!$util.isInteger(message.strideW))
                        return "strideW: integer expected";
                if (message.engine != null && message.hasOwnProperty("engine"))
                    switch (message.engine) {
                    default:
                        return "engine: enum value expected";
                    case 0:
                    case 1:
                    case 2:
                        break;
                    }
                if (message.globalPooling != null && message.hasOwnProperty("globalPooling"))
                    if (typeof message.globalPooling !== "boolean")
                        return "globalPooling: boolean expected";
                if (message.roundMode != null && message.hasOwnProperty("roundMode"))
                    switch (message.roundMode) {
                    default:
                        return "roundMode: enum value expected";
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
                if (object.padH != null)
                    message.padH = object.padH >>> 0;
                if (object.padW != null)
                    message.padW = object.padW >>> 0;
                if (object.kernelSize != null)
                    message.kernelSize = object.kernelSize >>> 0;
                if (object.kernelH != null)
                    message.kernelH = object.kernelH >>> 0;
                if (object.kernelW != null)
                    message.kernelW = object.kernelW >>> 0;
                if (object.stride != null)
                    message.stride = object.stride >>> 0;
                if (object.strideH != null)
                    message.strideH = object.strideH >>> 0;
                if (object.strideW != null)
                    message.strideW = object.strideW >>> 0;
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
                if (object.globalPooling != null)
                    message.globalPooling = Boolean(object.globalPooling);
                switch (object.roundMode) {
                case "CEIL":
                case 0:
                    message.roundMode = 0;
                    break;
                case "FLOOR":
                case 1:
                    message.roundMode = 1;
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
                    object.kernelSize = 0;
                    object.stride = 1;
                    object.pad = 0;
                    object.kernelH = 0;
                    object.kernelW = 0;
                    object.strideH = 0;
                    object.strideW = 0;
                    object.padH = 0;
                    object.padW = 0;
                    object.engine = options.enums === String ? "DEFAULT" : 0;
                    object.globalPooling = false;
                    object.roundMode = options.enums === String ? "CEIL" : 0;
                }
                if (message.pool != null && message.hasOwnProperty("pool"))
                    object.pool = options.enums === String ? $root.caffe.PoolingParameter.PoolMethod[message.pool] : message.pool;
                if (message.kernelSize != null && message.hasOwnProperty("kernelSize"))
                    object.kernelSize = message.kernelSize;
                if (message.stride != null && message.hasOwnProperty("stride"))
                    object.stride = message.stride;
                if (message.pad != null && message.hasOwnProperty("pad"))
                    object.pad = message.pad;
                if (message.kernelH != null && message.hasOwnProperty("kernelH"))
                    object.kernelH = message.kernelH;
                if (message.kernelW != null && message.hasOwnProperty("kernelW"))
                    object.kernelW = message.kernelW;
                if (message.strideH != null && message.hasOwnProperty("strideH"))
                    object.strideH = message.strideH;
                if (message.strideW != null && message.hasOwnProperty("strideW"))
                    object.strideW = message.strideW;
                if (message.padH != null && message.hasOwnProperty("padH"))
                    object.padH = message.padH;
                if (message.padW != null && message.hasOwnProperty("padW"))
                    object.padW = message.padW;
                if (message.engine != null && message.hasOwnProperty("engine"))
                    object.engine = options.enums === String ? $root.caffe.PoolingParameter.Engine[message.engine] : message.engine;
                if (message.globalPooling != null && message.hasOwnProperty("globalPooling"))
                    object.globalPooling = message.globalPooling;
                if (message.roundMode != null && message.hasOwnProperty("roundMode"))
                    object.roundMode = options.enums === String ? $root.caffe.PoolingParameter.RoundMode[message.roundMode] : message.roundMode;
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
            PythonParameter.prototype.paramStr = "";
            PythonParameter.prototype.shareInParallel = false;
    
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
                        message.paramStr = reader.string();
                        break;
                    case 4:
                        message.shareInParallel = reader.bool();
                        break;
                    default:
                        reader.skipType(tag & 7);
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
                if (message.paramStr != null && message.hasOwnProperty("paramStr"))
                    if (!$util.isString(message.paramStr))
                        return "paramStr: string expected";
                if (message.shareInParallel != null && message.hasOwnProperty("shareInParallel"))
                    if (typeof message.shareInParallel !== "boolean")
                        return "shareInParallel: boolean expected";
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
                if (object.paramStr != null)
                    message.paramStr = String(object.paramStr);
                if (object.shareInParallel != null)
                    message.shareInParallel = Boolean(object.shareInParallel);
                return message;
            };
    
            PythonParameter.toObject = function toObject(message, options) {
                if (!options)
                    options = {};
                var object = {};
                if (options.defaults) {
                    object.module = "";
                    object.layer = "";
                    object.paramStr = "";
                    object.shareInParallel = false;
                }
                if (message.module != null && message.hasOwnProperty("module"))
                    object.module = message.module;
                if (message.layer != null && message.hasOwnProperty("layer"))
                    object.layer = message.layer;
                if (message.paramStr != null && message.hasOwnProperty("paramStr"))
                    object.paramStr = message.paramStr;
                if (message.shareInParallel != null && message.hasOwnProperty("shareInParallel"))
                    object.shareInParallel = message.shareInParallel;
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
    
            RecurrentParameter.prototype.numOutput = 0;
            RecurrentParameter.prototype.weightFiller = null;
            RecurrentParameter.prototype.biasFiller = null;
            RecurrentParameter.prototype.debugInfo = false;
            RecurrentParameter.prototype.exposeHidden = false;
    
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
                        message.numOutput = reader.uint32();
                        break;
                    case 2:
                        message.weightFiller = $root.caffe.FillerParameter.decode(reader, reader.uint32());
                        break;
                    case 3:
                        message.biasFiller = $root.caffe.FillerParameter.decode(reader, reader.uint32());
                        break;
                    case 4:
                        message.debugInfo = reader.bool();
                        break;
                    case 5:
                        message.exposeHidden = reader.bool();
                        break;
                    default:
                        reader.skipType(tag & 7);
                        break;
                    }
                }
                return message;
            };
    
            RecurrentParameter.verify = function verify(message) {
                if (typeof message !== "object" || message === null)
                    return "object expected";
                if (message.numOutput != null && message.hasOwnProperty("numOutput"))
                    if (!$util.isInteger(message.numOutput))
                        return "numOutput: integer expected";
                if (message.weightFiller != null && message.hasOwnProperty("weightFiller")) {
                    var error = $root.caffe.FillerParameter.verify(message.weightFiller);
                    if (error)
                        return "weightFiller." + error;
                }
                if (message.biasFiller != null && message.hasOwnProperty("biasFiller")) {
                    var error = $root.caffe.FillerParameter.verify(message.biasFiller);
                    if (error)
                        return "biasFiller." + error;
                }
                if (message.debugInfo != null && message.hasOwnProperty("debugInfo"))
                    if (typeof message.debugInfo !== "boolean")
                        return "debugInfo: boolean expected";
                if (message.exposeHidden != null && message.hasOwnProperty("exposeHidden"))
                    if (typeof message.exposeHidden !== "boolean")
                        return "exposeHidden: boolean expected";
                return null;
            };
    
            RecurrentParameter.fromObject = function fromObject(object) {
                if (object instanceof $root.caffe.RecurrentParameter)
                    return object;
                var message = new $root.caffe.RecurrentParameter();
                if (object.numOutput != null)
                    message.numOutput = object.numOutput >>> 0;
                if (object.weightFiller != null) {
                    if (typeof object.weightFiller !== "object")
                        throw TypeError(".caffe.RecurrentParameter.weightFiller: object expected");
                    message.weightFiller = $root.caffe.FillerParameter.fromObject(object.weightFiller);
                }
                if (object.biasFiller != null) {
                    if (typeof object.biasFiller !== "object")
                        throw TypeError(".caffe.RecurrentParameter.biasFiller: object expected");
                    message.biasFiller = $root.caffe.FillerParameter.fromObject(object.biasFiller);
                }
                if (object.debugInfo != null)
                    message.debugInfo = Boolean(object.debugInfo);
                if (object.exposeHidden != null)
                    message.exposeHidden = Boolean(object.exposeHidden);
                return message;
            };
    
            RecurrentParameter.toObject = function toObject(message, options) {
                if (!options)
                    options = {};
                var object = {};
                if (options.defaults) {
                    object.numOutput = 0;
                    object.weightFiller = null;
                    object.biasFiller = null;
                    object.debugInfo = false;
                    object.exposeHidden = false;
                }
                if (message.numOutput != null && message.hasOwnProperty("numOutput"))
                    object.numOutput = message.numOutput;
                if (message.weightFiller != null && message.hasOwnProperty("weightFiller"))
                    object.weightFiller = $root.caffe.FillerParameter.toObject(message.weightFiller, options);
                if (message.biasFiller != null && message.hasOwnProperty("biasFiller"))
                    object.biasFiller = $root.caffe.FillerParameter.toObject(message.biasFiller, options);
                if (message.debugInfo != null && message.hasOwnProperty("debugInfo"))
                    object.debugInfo = message.debugInfo;
                if (message.exposeHidden != null && message.hasOwnProperty("exposeHidden"))
                    object.exposeHidden = message.exposeHidden;
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
    
            ReLUParameter.prototype.negativeSlope = 0;
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
                        message.negativeSlope = reader.float();
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
    
            ReLUParameter.verify = function verify(message) {
                if (typeof message !== "object" || message === null)
                    return "object expected";
                if (message.negativeSlope != null && message.hasOwnProperty("negativeSlope"))
                    if (typeof message.negativeSlope !== "number")
                        return "negativeSlope: number expected";
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
                if (object.negativeSlope != null)
                    message.negativeSlope = Number(object.negativeSlope);
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
                    object.negativeSlope = 0;
                    object.engine = options.enums === String ? "DEFAULT" : 0;
                }
                if (message.negativeSlope != null && message.hasOwnProperty("negativeSlope"))
                    object.negativeSlope = options.json && !isFinite(message.negativeSlope) ? String(message.negativeSlope) : message.negativeSlope;
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
            ReshapeParameter.prototype.numAxes = -1;
    
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
                        message.numAxes = reader.int32();
                        break;
                    default:
                        reader.skipType(tag & 7);
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
                if (message.numAxes != null && message.hasOwnProperty("numAxes"))
                    if (!$util.isInteger(message.numAxes))
                        return "numAxes: integer expected";
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
                if (object.numAxes != null)
                    message.numAxes = object.numAxes | 0;
                return message;
            };
    
            ReshapeParameter.toObject = function toObject(message, options) {
                if (!options)
                    options = {};
                var object = {};
                if (options.defaults) {
                    object.shape = null;
                    object.axis = 0;
                    object.numAxes = -1;
                }
                if (message.shape != null && message.hasOwnProperty("shape"))
                    object.shape = $root.caffe.BlobShape.toObject(message.shape, options);
                if (message.axis != null && message.hasOwnProperty("axis"))
                    object.axis = message.axis;
                if (message.numAxes != null && message.hasOwnProperty("numAxes"))
                    object.numAxes = message.numAxes;
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
            ScaleParameter.prototype.numAxes = 1;
            ScaleParameter.prototype.filler = null;
            ScaleParameter.prototype.biasTerm = false;
            ScaleParameter.prototype.biasFiller = null;
    
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
                        message.numAxes = reader.int32();
                        break;
                    case 3:
                        message.filler = $root.caffe.FillerParameter.decode(reader, reader.uint32());
                        break;
                    case 4:
                        message.biasTerm = reader.bool();
                        break;
                    case 5:
                        message.biasFiller = $root.caffe.FillerParameter.decode(reader, reader.uint32());
                        break;
                    default:
                        reader.skipType(tag & 7);
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
                if (message.numAxes != null && message.hasOwnProperty("numAxes"))
                    if (!$util.isInteger(message.numAxes))
                        return "numAxes: integer expected";
                if (message.filler != null && message.hasOwnProperty("filler")) {
                    var error = $root.caffe.FillerParameter.verify(message.filler);
                    if (error)
                        return "filler." + error;
                }
                if (message.biasTerm != null && message.hasOwnProperty("biasTerm"))
                    if (typeof message.biasTerm !== "boolean")
                        return "biasTerm: boolean expected";
                if (message.biasFiller != null && message.hasOwnProperty("biasFiller")) {
                    var error = $root.caffe.FillerParameter.verify(message.biasFiller);
                    if (error)
                        return "biasFiller." + error;
                }
                return null;
            };
    
            ScaleParameter.fromObject = function fromObject(object) {
                if (object instanceof $root.caffe.ScaleParameter)
                    return object;
                var message = new $root.caffe.ScaleParameter();
                if (object.axis != null)
                    message.axis = object.axis | 0;
                if (object.numAxes != null)
                    message.numAxes = object.numAxes | 0;
                if (object.filler != null) {
                    if (typeof object.filler !== "object")
                        throw TypeError(".caffe.ScaleParameter.filler: object expected");
                    message.filler = $root.caffe.FillerParameter.fromObject(object.filler);
                }
                if (object.biasTerm != null)
                    message.biasTerm = Boolean(object.biasTerm);
                if (object.biasFiller != null) {
                    if (typeof object.biasFiller !== "object")
                        throw TypeError(".caffe.ScaleParameter.biasFiller: object expected");
                    message.biasFiller = $root.caffe.FillerParameter.fromObject(object.biasFiller);
                }
                return message;
            };
    
            ScaleParameter.toObject = function toObject(message, options) {
                if (!options)
                    options = {};
                var object = {};
                if (options.defaults) {
                    object.axis = 1;
                    object.numAxes = 1;
                    object.filler = null;
                    object.biasTerm = false;
                    object.biasFiller = null;
                }
                if (message.axis != null && message.hasOwnProperty("axis"))
                    object.axis = message.axis;
                if (message.numAxes != null && message.hasOwnProperty("numAxes"))
                    object.numAxes = message.numAxes;
                if (message.filler != null && message.hasOwnProperty("filler"))
                    object.filler = $root.caffe.FillerParameter.toObject(message.filler, options);
                if (message.biasTerm != null && message.hasOwnProperty("biasTerm"))
                    object.biasTerm = message.biasTerm;
                if (message.biasFiller != null && message.hasOwnProperty("biasFiller"))
                    object.biasFiller = $root.caffe.FillerParameter.toObject(message.biasFiller, options);
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
                this.slicePoint = [];
                if (properties)
                    for (var keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                        if (properties[keys[i]] != null)
                            this[keys[i]] = properties[keys[i]];
            }
    
            SliceParameter.prototype.axis = 1;
            SliceParameter.prototype.slicePoint = $util.emptyArray;
            SliceParameter.prototype.sliceDim = 1;
    
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
                        if (!(message.slicePoint && message.slicePoint.length))
                            message.slicePoint = [];
                        if ((tag & 7) === 2) {
                            var end2 = reader.uint32() + reader.pos;
                            while (reader.pos < end2)
                                message.slicePoint.push(reader.uint32());
                        } else
                            message.slicePoint.push(reader.uint32());
                        break;
                    case 1:
                        message.sliceDim = reader.uint32();
                        break;
                    default:
                        reader.skipType(tag & 7);
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
                if (message.slicePoint != null && message.hasOwnProperty("slicePoint")) {
                    if (!Array.isArray(message.slicePoint))
                        return "slicePoint: array expected";
                    for (var i = 0; i < message.slicePoint.length; ++i)
                        if (!$util.isInteger(message.slicePoint[i]))
                            return "slicePoint: integer[] expected";
                }
                if (message.sliceDim != null && message.hasOwnProperty("sliceDim"))
                    if (!$util.isInteger(message.sliceDim))
                        return "sliceDim: integer expected";
                return null;
            };
    
            SliceParameter.fromObject = function fromObject(object) {
                if (object instanceof $root.caffe.SliceParameter)
                    return object;
                var message = new $root.caffe.SliceParameter();
                if (object.axis != null)
                    message.axis = object.axis | 0;
                if (object.slicePoint) {
                    if (!Array.isArray(object.slicePoint))
                        throw TypeError(".caffe.SliceParameter.slicePoint: array expected");
                    message.slicePoint = [];
                    for (var i = 0; i < object.slicePoint.length; ++i)
                        message.slicePoint[i] = object.slicePoint[i] >>> 0;
                }
                if (object.sliceDim != null)
                    message.sliceDim = object.sliceDim >>> 0;
                return message;
            };
    
            SliceParameter.toObject = function toObject(message, options) {
                if (!options)
                    options = {};
                var object = {};
                if (options.arrays || options.defaults)
                    object.slicePoint = [];
                if (options.defaults) {
                    object.sliceDim = 1;
                    object.axis = 1;
                }
                if (message.sliceDim != null && message.hasOwnProperty("sliceDim"))
                    object.sliceDim = message.sliceDim;
                if (message.slicePoint && message.slicePoint.length) {
                    object.slicePoint = [];
                    for (var j = 0; j < message.slicePoint.length; ++j)
                        object.slicePoint[j] = message.slicePoint[j];
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
            WindowDataParameter.prototype.meanFile = "";
            WindowDataParameter.prototype.batchSize = 0;
            WindowDataParameter.prototype.cropSize = 0;
            WindowDataParameter.prototype.mirror = false;
            WindowDataParameter.prototype.fgThreshold = 0.5;
            WindowDataParameter.prototype.bgThreshold = 0.5;
            WindowDataParameter.prototype.fgFraction = 0.25;
            WindowDataParameter.prototype.contextPad = 0;
            WindowDataParameter.prototype.cropMode = "warp";
            WindowDataParameter.prototype.cacheImages = false;
            WindowDataParameter.prototype.rootFolder = "";
    
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
                        message.meanFile = reader.string();
                        break;
                    case 4:
                        message.batchSize = reader.uint32();
                        break;
                    case 5:
                        message.cropSize = reader.uint32();
                        break;
                    case 6:
                        message.mirror = reader.bool();
                        break;
                    case 7:
                        message.fgThreshold = reader.float();
                        break;
                    case 8:
                        message.bgThreshold = reader.float();
                        break;
                    case 9:
                        message.fgFraction = reader.float();
                        break;
                    case 10:
                        message.contextPad = reader.uint32();
                        break;
                    case 11:
                        message.cropMode = reader.string();
                        break;
                    case 12:
                        message.cacheImages = reader.bool();
                        break;
                    case 13:
                        message.rootFolder = reader.string();
                        break;
                    default:
                        reader.skipType(tag & 7);
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
                if (message.meanFile != null && message.hasOwnProperty("meanFile"))
                    if (!$util.isString(message.meanFile))
                        return "meanFile: string expected";
                if (message.batchSize != null && message.hasOwnProperty("batchSize"))
                    if (!$util.isInteger(message.batchSize))
                        return "batchSize: integer expected";
                if (message.cropSize != null && message.hasOwnProperty("cropSize"))
                    if (!$util.isInteger(message.cropSize))
                        return "cropSize: integer expected";
                if (message.mirror != null && message.hasOwnProperty("mirror"))
                    if (typeof message.mirror !== "boolean")
                        return "mirror: boolean expected";
                if (message.fgThreshold != null && message.hasOwnProperty("fgThreshold"))
                    if (typeof message.fgThreshold !== "number")
                        return "fgThreshold: number expected";
                if (message.bgThreshold != null && message.hasOwnProperty("bgThreshold"))
                    if (typeof message.bgThreshold !== "number")
                        return "bgThreshold: number expected";
                if (message.fgFraction != null && message.hasOwnProperty("fgFraction"))
                    if (typeof message.fgFraction !== "number")
                        return "fgFraction: number expected";
                if (message.contextPad != null && message.hasOwnProperty("contextPad"))
                    if (!$util.isInteger(message.contextPad))
                        return "contextPad: integer expected";
                if (message.cropMode != null && message.hasOwnProperty("cropMode"))
                    if (!$util.isString(message.cropMode))
                        return "cropMode: string expected";
                if (message.cacheImages != null && message.hasOwnProperty("cacheImages"))
                    if (typeof message.cacheImages !== "boolean")
                        return "cacheImages: boolean expected";
                if (message.rootFolder != null && message.hasOwnProperty("rootFolder"))
                    if (!$util.isString(message.rootFolder))
                        return "rootFolder: string expected";
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
                if (object.meanFile != null)
                    message.meanFile = String(object.meanFile);
                if (object.batchSize != null)
                    message.batchSize = object.batchSize >>> 0;
                if (object.cropSize != null)
                    message.cropSize = object.cropSize >>> 0;
                if (object.mirror != null)
                    message.mirror = Boolean(object.mirror);
                if (object.fgThreshold != null)
                    message.fgThreshold = Number(object.fgThreshold);
                if (object.bgThreshold != null)
                    message.bgThreshold = Number(object.bgThreshold);
                if (object.fgFraction != null)
                    message.fgFraction = Number(object.fgFraction);
                if (object.contextPad != null)
                    message.contextPad = object.contextPad >>> 0;
                if (object.cropMode != null)
                    message.cropMode = String(object.cropMode);
                if (object.cacheImages != null)
                    message.cacheImages = Boolean(object.cacheImages);
                if (object.rootFolder != null)
                    message.rootFolder = String(object.rootFolder);
                return message;
            };
    
            WindowDataParameter.toObject = function toObject(message, options) {
                if (!options)
                    options = {};
                var object = {};
                if (options.defaults) {
                    object.source = "";
                    object.scale = 1;
                    object.meanFile = "";
                    object.batchSize = 0;
                    object.cropSize = 0;
                    object.mirror = false;
                    object.fgThreshold = 0.5;
                    object.bgThreshold = 0.5;
                    object.fgFraction = 0.25;
                    object.contextPad = 0;
                    object.cropMode = "warp";
                    object.cacheImages = false;
                    object.rootFolder = "";
                }
                if (message.source != null && message.hasOwnProperty("source"))
                    object.source = message.source;
                if (message.scale != null && message.hasOwnProperty("scale"))
                    object.scale = options.json && !isFinite(message.scale) ? String(message.scale) : message.scale;
                if (message.meanFile != null && message.hasOwnProperty("meanFile"))
                    object.meanFile = message.meanFile;
                if (message.batchSize != null && message.hasOwnProperty("batchSize"))
                    object.batchSize = message.batchSize;
                if (message.cropSize != null && message.hasOwnProperty("cropSize"))
                    object.cropSize = message.cropSize;
                if (message.mirror != null && message.hasOwnProperty("mirror"))
                    object.mirror = message.mirror;
                if (message.fgThreshold != null && message.hasOwnProperty("fgThreshold"))
                    object.fgThreshold = options.json && !isFinite(message.fgThreshold) ? String(message.fgThreshold) : message.fgThreshold;
                if (message.bgThreshold != null && message.hasOwnProperty("bgThreshold"))
                    object.bgThreshold = options.json && !isFinite(message.bgThreshold) ? String(message.bgThreshold) : message.bgThreshold;
                if (message.fgFraction != null && message.hasOwnProperty("fgFraction"))
                    object.fgFraction = options.json && !isFinite(message.fgFraction) ? String(message.fgFraction) : message.fgFraction;
                if (message.contextPad != null && message.hasOwnProperty("contextPad"))
                    object.contextPad = message.contextPad;
                if (message.cropMode != null && message.hasOwnProperty("cropMode"))
                    object.cropMode = message.cropMode;
                if (message.cacheImages != null && message.hasOwnProperty("cacheImages"))
                    object.cacheImages = message.cacheImages;
                if (message.rootFolder != null && message.hasOwnProperty("rootFolder"))
                    object.rootFolder = message.rootFolder;
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
    
            SPPParameter.prototype.pyramidHeight = 0;
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
                        message.pyramidHeight = reader.uint32();
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
    
            SPPParameter.verify = function verify(message) {
                if (typeof message !== "object" || message === null)
                    return "object expected";
                if (message.pyramidHeight != null && message.hasOwnProperty("pyramidHeight"))
                    if (!$util.isInteger(message.pyramidHeight))
                        return "pyramidHeight: integer expected";
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
                if (object.pyramidHeight != null)
                    message.pyramidHeight = object.pyramidHeight >>> 0;
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
                    object.pyramidHeight = 0;
                    object.pool = options.enums === String ? "MAX" : 0;
                    object.engine = options.enums === String ? "DEFAULT" : 0;
                }
                if (message.pyramidHeight != null && message.hasOwnProperty("pyramidHeight"))
                    object.pyramidHeight = message.pyramidHeight;
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
                this.blobShareMode = [];
                this.blobsLr = [];
                this.weightDecay = [];
                this.lossWeight = [];
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
            V1LayerParameter.prototype.blobShareMode = $util.emptyArray;
            V1LayerParameter.prototype.blobsLr = $util.emptyArray;
            V1LayerParameter.prototype.weightDecay = $util.emptyArray;
            V1LayerParameter.prototype.lossWeight = $util.emptyArray;
            V1LayerParameter.prototype.accuracyParam = null;
            V1LayerParameter.prototype.argmaxParam = null;
            V1LayerParameter.prototype.concatParam = null;
            V1LayerParameter.prototype.contrastiveLossParam = null;
            V1LayerParameter.prototype.convolutionParam = null;
            V1LayerParameter.prototype.dataParam = null;
            V1LayerParameter.prototype.dropoutParam = null;
            V1LayerParameter.prototype.dummyDataParam = null;
            V1LayerParameter.prototype.eltwiseParam = null;
            V1LayerParameter.prototype.expParam = null;
            V1LayerParameter.prototype.hdf5DataParam = null;
            V1LayerParameter.prototype.hdf5OutputParam = null;
            V1LayerParameter.prototype.hingeLossParam = null;
            V1LayerParameter.prototype.imageDataParam = null;
            V1LayerParameter.prototype.infogainLossParam = null;
            V1LayerParameter.prototype.innerProductParam = null;
            V1LayerParameter.prototype.lrnParam = null;
            V1LayerParameter.prototype.memoryDataParam = null;
            V1LayerParameter.prototype.mvnParam = null;
            V1LayerParameter.prototype.poolingParam = null;
            V1LayerParameter.prototype.powerParam = null;
            V1LayerParameter.prototype.reluParam = null;
            V1LayerParameter.prototype.sigmoidParam = null;
            V1LayerParameter.prototype.softmaxParam = null;
            V1LayerParameter.prototype.sliceParam = null;
            V1LayerParameter.prototype.tanhParam = null;
            V1LayerParameter.prototype.thresholdParam = null;
            V1LayerParameter.prototype.windowDataParam = null;
            V1LayerParameter.prototype.transformParam = null;
            V1LayerParameter.prototype.lossParam = null;
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
                        if (!(message.blobShareMode && message.blobShareMode.length))
                            message.blobShareMode = [];
                        if ((tag & 7) === 2) {
                            var end2 = reader.uint32() + reader.pos;
                            while (reader.pos < end2)
                                message.blobShareMode.push(reader.int32());
                        } else
                            message.blobShareMode.push(reader.int32());
                        break;
                    case 7:
                        if (!(message.blobsLr && message.blobsLr.length))
                            message.blobsLr = [];
                        if ((tag & 7) === 2) {
                            var end2 = reader.uint32() + reader.pos;
                            while (reader.pos < end2)
                                message.blobsLr.push(reader.float());
                        } else
                            message.blobsLr.push(reader.float());
                        break;
                    case 8:
                        if (!(message.weightDecay && message.weightDecay.length))
                            message.weightDecay = [];
                        if ((tag & 7) === 2) {
                            var end2 = reader.uint32() + reader.pos;
                            while (reader.pos < end2)
                                message.weightDecay.push(reader.float());
                        } else
                            message.weightDecay.push(reader.float());
                        break;
                    case 35:
                        if (!(message.lossWeight && message.lossWeight.length))
                            message.lossWeight = [];
                        if ((tag & 7) === 2) {
                            var end2 = reader.uint32() + reader.pos;
                            while (reader.pos < end2)
                                message.lossWeight.push(reader.float());
                        } else
                            message.lossWeight.push(reader.float());
                        break;
                    case 27:
                        message.accuracyParam = $root.caffe.AccuracyParameter.decode(reader, reader.uint32());
                        break;
                    case 23:
                        message.argmaxParam = $root.caffe.ArgMaxParameter.decode(reader, reader.uint32());
                        break;
                    case 9:
                        message.concatParam = $root.caffe.ConcatParameter.decode(reader, reader.uint32());
                        break;
                    case 40:
                        message.contrastiveLossParam = $root.caffe.ContrastiveLossParameter.decode(reader, reader.uint32());
                        break;
                    case 10:
                        message.convolutionParam = $root.caffe.ConvolutionParameter.decode(reader, reader.uint32());
                        break;
                    case 11:
                        message.dataParam = $root.caffe.DataParameter.decode(reader, reader.uint32());
                        break;
                    case 12:
                        message.dropoutParam = $root.caffe.DropoutParameter.decode(reader, reader.uint32());
                        break;
                    case 26:
                        message.dummyDataParam = $root.caffe.DummyDataParameter.decode(reader, reader.uint32());
                        break;
                    case 24:
                        message.eltwiseParam = $root.caffe.EltwiseParameter.decode(reader, reader.uint32());
                        break;
                    case 41:
                        message.expParam = $root.caffe.ExpParameter.decode(reader, reader.uint32());
                        break;
                    case 13:
                        message.hdf5DataParam = $root.caffe.HDF5DataParameter.decode(reader, reader.uint32());
                        break;
                    case 14:
                        message.hdf5OutputParam = $root.caffe.HDF5OutputParameter.decode(reader, reader.uint32());
                        break;
                    case 29:
                        message.hingeLossParam = $root.caffe.HingeLossParameter.decode(reader, reader.uint32());
                        break;
                    case 15:
                        message.imageDataParam = $root.caffe.ImageDataParameter.decode(reader, reader.uint32());
                        break;
                    case 16:
                        message.infogainLossParam = $root.caffe.InfogainLossParameter.decode(reader, reader.uint32());
                        break;
                    case 17:
                        message.innerProductParam = $root.caffe.InnerProductParameter.decode(reader, reader.uint32());
                        break;
                    case 18:
                        message.lrnParam = $root.caffe.LRNParameter.decode(reader, reader.uint32());
                        break;
                    case 22:
                        message.memoryDataParam = $root.caffe.MemoryDataParameter.decode(reader, reader.uint32());
                        break;
                    case 34:
                        message.mvnParam = $root.caffe.MVNParameter.decode(reader, reader.uint32());
                        break;
                    case 19:
                        message.poolingParam = $root.caffe.PoolingParameter.decode(reader, reader.uint32());
                        break;
                    case 21:
                        message.powerParam = $root.caffe.PowerParameter.decode(reader, reader.uint32());
                        break;
                    case 30:
                        message.reluParam = $root.caffe.ReLUParameter.decode(reader, reader.uint32());
                        break;
                    case 38:
                        message.sigmoidParam = $root.caffe.SigmoidParameter.decode(reader, reader.uint32());
                        break;
                    case 39:
                        message.softmaxParam = $root.caffe.SoftmaxParameter.decode(reader, reader.uint32());
                        break;
                    case 31:
                        message.sliceParam = $root.caffe.SliceParameter.decode(reader, reader.uint32());
                        break;
                    case 37:
                        message.tanhParam = $root.caffe.TanHParameter.decode(reader, reader.uint32());
                        break;
                    case 25:
                        message.thresholdParam = $root.caffe.ThresholdParameter.decode(reader, reader.uint32());
                        break;
                    case 20:
                        message.windowDataParam = $root.caffe.WindowDataParameter.decode(reader, reader.uint32());
                        break;
                    case 36:
                        message.transformParam = $root.caffe.TransformationParameter.decode(reader, reader.uint32());
                        break;
                    case 42:
                        message.lossParam = $root.caffe.LossParameter.decode(reader, reader.uint32());
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
                if (message.blobShareMode != null && message.hasOwnProperty("blobShareMode")) {
                    if (!Array.isArray(message.blobShareMode))
                        return "blobShareMode: array expected";
                    for (var i = 0; i < message.blobShareMode.length; ++i)
                        switch (message.blobShareMode[i]) {
                        default:
                            return "blobShareMode: enum value[] expected";
                        case 0:
                        case 1:
                            break;
                        }
                }
                if (message.blobsLr != null && message.hasOwnProperty("blobsLr")) {
                    if (!Array.isArray(message.blobsLr))
                        return "blobsLr: array expected";
                    for (var i = 0; i < message.blobsLr.length; ++i)
                        if (typeof message.blobsLr[i] !== "number")
                            return "blobsLr: number[] expected";
                }
                if (message.weightDecay != null && message.hasOwnProperty("weightDecay")) {
                    if (!Array.isArray(message.weightDecay))
                        return "weightDecay: array expected";
                    for (var i = 0; i < message.weightDecay.length; ++i)
                        if (typeof message.weightDecay[i] !== "number")
                            return "weightDecay: number[] expected";
                }
                if (message.lossWeight != null && message.hasOwnProperty("lossWeight")) {
                    if (!Array.isArray(message.lossWeight))
                        return "lossWeight: array expected";
                    for (var i = 0; i < message.lossWeight.length; ++i)
                        if (typeof message.lossWeight[i] !== "number")
                            return "lossWeight: number[] expected";
                }
                if (message.accuracyParam != null && message.hasOwnProperty("accuracyParam")) {
                    var error = $root.caffe.AccuracyParameter.verify(message.accuracyParam);
                    if (error)
                        return "accuracyParam." + error;
                }
                if (message.argmaxParam != null && message.hasOwnProperty("argmaxParam")) {
                    var error = $root.caffe.ArgMaxParameter.verify(message.argmaxParam);
                    if (error)
                        return "argmaxParam." + error;
                }
                if (message.concatParam != null && message.hasOwnProperty("concatParam")) {
                    var error = $root.caffe.ConcatParameter.verify(message.concatParam);
                    if (error)
                        return "concatParam." + error;
                }
                if (message.contrastiveLossParam != null && message.hasOwnProperty("contrastiveLossParam")) {
                    var error = $root.caffe.ContrastiveLossParameter.verify(message.contrastiveLossParam);
                    if (error)
                        return "contrastiveLossParam." + error;
                }
                if (message.convolutionParam != null && message.hasOwnProperty("convolutionParam")) {
                    var error = $root.caffe.ConvolutionParameter.verify(message.convolutionParam);
                    if (error)
                        return "convolutionParam." + error;
                }
                if (message.dataParam != null && message.hasOwnProperty("dataParam")) {
                    var error = $root.caffe.DataParameter.verify(message.dataParam);
                    if (error)
                        return "dataParam." + error;
                }
                if (message.dropoutParam != null && message.hasOwnProperty("dropoutParam")) {
                    var error = $root.caffe.DropoutParameter.verify(message.dropoutParam);
                    if (error)
                        return "dropoutParam." + error;
                }
                if (message.dummyDataParam != null && message.hasOwnProperty("dummyDataParam")) {
                    var error = $root.caffe.DummyDataParameter.verify(message.dummyDataParam);
                    if (error)
                        return "dummyDataParam." + error;
                }
                if (message.eltwiseParam != null && message.hasOwnProperty("eltwiseParam")) {
                    var error = $root.caffe.EltwiseParameter.verify(message.eltwiseParam);
                    if (error)
                        return "eltwiseParam." + error;
                }
                if (message.expParam != null && message.hasOwnProperty("expParam")) {
                    var error = $root.caffe.ExpParameter.verify(message.expParam);
                    if (error)
                        return "expParam." + error;
                }
                if (message.hdf5DataParam != null && message.hasOwnProperty("hdf5DataParam")) {
                    var error = $root.caffe.HDF5DataParameter.verify(message.hdf5DataParam);
                    if (error)
                        return "hdf5DataParam." + error;
                }
                if (message.hdf5OutputParam != null && message.hasOwnProperty("hdf5OutputParam")) {
                    var error = $root.caffe.HDF5OutputParameter.verify(message.hdf5OutputParam);
                    if (error)
                        return "hdf5OutputParam." + error;
                }
                if (message.hingeLossParam != null && message.hasOwnProperty("hingeLossParam")) {
                    var error = $root.caffe.HingeLossParameter.verify(message.hingeLossParam);
                    if (error)
                        return "hingeLossParam." + error;
                }
                if (message.imageDataParam != null && message.hasOwnProperty("imageDataParam")) {
                    var error = $root.caffe.ImageDataParameter.verify(message.imageDataParam);
                    if (error)
                        return "imageDataParam." + error;
                }
                if (message.infogainLossParam != null && message.hasOwnProperty("infogainLossParam")) {
                    var error = $root.caffe.InfogainLossParameter.verify(message.infogainLossParam);
                    if (error)
                        return "infogainLossParam." + error;
                }
                if (message.innerProductParam != null && message.hasOwnProperty("innerProductParam")) {
                    var error = $root.caffe.InnerProductParameter.verify(message.innerProductParam);
                    if (error)
                        return "innerProductParam." + error;
                }
                if (message.lrnParam != null && message.hasOwnProperty("lrnParam")) {
                    var error = $root.caffe.LRNParameter.verify(message.lrnParam);
                    if (error)
                        return "lrnParam." + error;
                }
                if (message.memoryDataParam != null && message.hasOwnProperty("memoryDataParam")) {
                    var error = $root.caffe.MemoryDataParameter.verify(message.memoryDataParam);
                    if (error)
                        return "memoryDataParam." + error;
                }
                if (message.mvnParam != null && message.hasOwnProperty("mvnParam")) {
                    var error = $root.caffe.MVNParameter.verify(message.mvnParam);
                    if (error)
                        return "mvnParam." + error;
                }
                if (message.poolingParam != null && message.hasOwnProperty("poolingParam")) {
                    var error = $root.caffe.PoolingParameter.verify(message.poolingParam);
                    if (error)
                        return "poolingParam." + error;
                }
                if (message.powerParam != null && message.hasOwnProperty("powerParam")) {
                    var error = $root.caffe.PowerParameter.verify(message.powerParam);
                    if (error)
                        return "powerParam." + error;
                }
                if (message.reluParam != null && message.hasOwnProperty("reluParam")) {
                    var error = $root.caffe.ReLUParameter.verify(message.reluParam);
                    if (error)
                        return "reluParam." + error;
                }
                if (message.sigmoidParam != null && message.hasOwnProperty("sigmoidParam")) {
                    var error = $root.caffe.SigmoidParameter.verify(message.sigmoidParam);
                    if (error)
                        return "sigmoidParam." + error;
                }
                if (message.softmaxParam != null && message.hasOwnProperty("softmaxParam")) {
                    var error = $root.caffe.SoftmaxParameter.verify(message.softmaxParam);
                    if (error)
                        return "softmaxParam." + error;
                }
                if (message.sliceParam != null && message.hasOwnProperty("sliceParam")) {
                    var error = $root.caffe.SliceParameter.verify(message.sliceParam);
                    if (error)
                        return "sliceParam." + error;
                }
                if (message.tanhParam != null && message.hasOwnProperty("tanhParam")) {
                    var error = $root.caffe.TanHParameter.verify(message.tanhParam);
                    if (error)
                        return "tanhParam." + error;
                }
                if (message.thresholdParam != null && message.hasOwnProperty("thresholdParam")) {
                    var error = $root.caffe.ThresholdParameter.verify(message.thresholdParam);
                    if (error)
                        return "thresholdParam." + error;
                }
                if (message.windowDataParam != null && message.hasOwnProperty("windowDataParam")) {
                    var error = $root.caffe.WindowDataParameter.verify(message.windowDataParam);
                    if (error)
                        return "windowDataParam." + error;
                }
                if (message.transformParam != null && message.hasOwnProperty("transformParam")) {
                    var error = $root.caffe.TransformationParameter.verify(message.transformParam);
                    if (error)
                        return "transformParam." + error;
                }
                if (message.lossParam != null && message.hasOwnProperty("lossParam")) {
                    var error = $root.caffe.LossParameter.verify(message.lossParam);
                    if (error)
                        return "lossParam." + error;
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
                if (object.blobShareMode) {
                    if (!Array.isArray(object.blobShareMode))
                        throw TypeError(".caffe.V1LayerParameter.blobShareMode: array expected");
                    message.blobShareMode = [];
                    for (var i = 0; i < object.blobShareMode.length; ++i)
                        switch (object.blobShareMode[i]) {
                        default:
                        case "STRICT":
                        case 0:
                            message.blobShareMode[i] = 0;
                            break;
                        case "PERMISSIVE":
                        case 1:
                            message.blobShareMode[i] = 1;
                            break;
                        }
                }
                if (object.blobsLr) {
                    if (!Array.isArray(object.blobsLr))
                        throw TypeError(".caffe.V1LayerParameter.blobsLr: array expected");
                    message.blobsLr = [];
                    for (var i = 0; i < object.blobsLr.length; ++i)
                        message.blobsLr[i] = Number(object.blobsLr[i]);
                }
                if (object.weightDecay) {
                    if (!Array.isArray(object.weightDecay))
                        throw TypeError(".caffe.V1LayerParameter.weightDecay: array expected");
                    message.weightDecay = [];
                    for (var i = 0; i < object.weightDecay.length; ++i)
                        message.weightDecay[i] = Number(object.weightDecay[i]);
                }
                if (object.lossWeight) {
                    if (!Array.isArray(object.lossWeight))
                        throw TypeError(".caffe.V1LayerParameter.lossWeight: array expected");
                    message.lossWeight = [];
                    for (var i = 0; i < object.lossWeight.length; ++i)
                        message.lossWeight[i] = Number(object.lossWeight[i]);
                }
                if (object.accuracyParam != null) {
                    if (typeof object.accuracyParam !== "object")
                        throw TypeError(".caffe.V1LayerParameter.accuracyParam: object expected");
                    message.accuracyParam = $root.caffe.AccuracyParameter.fromObject(object.accuracyParam);
                }
                if (object.argmaxParam != null) {
                    if (typeof object.argmaxParam !== "object")
                        throw TypeError(".caffe.V1LayerParameter.argmaxParam: object expected");
                    message.argmaxParam = $root.caffe.ArgMaxParameter.fromObject(object.argmaxParam);
                }
                if (object.concatParam != null) {
                    if (typeof object.concatParam !== "object")
                        throw TypeError(".caffe.V1LayerParameter.concatParam: object expected");
                    message.concatParam = $root.caffe.ConcatParameter.fromObject(object.concatParam);
                }
                if (object.contrastiveLossParam != null) {
                    if (typeof object.contrastiveLossParam !== "object")
                        throw TypeError(".caffe.V1LayerParameter.contrastiveLossParam: object expected");
                    message.contrastiveLossParam = $root.caffe.ContrastiveLossParameter.fromObject(object.contrastiveLossParam);
                }
                if (object.convolutionParam != null) {
                    if (typeof object.convolutionParam !== "object")
                        throw TypeError(".caffe.V1LayerParameter.convolutionParam: object expected");
                    message.convolutionParam = $root.caffe.ConvolutionParameter.fromObject(object.convolutionParam);
                }
                if (object.dataParam != null) {
                    if (typeof object.dataParam !== "object")
                        throw TypeError(".caffe.V1LayerParameter.dataParam: object expected");
                    message.dataParam = $root.caffe.DataParameter.fromObject(object.dataParam);
                }
                if (object.dropoutParam != null) {
                    if (typeof object.dropoutParam !== "object")
                        throw TypeError(".caffe.V1LayerParameter.dropoutParam: object expected");
                    message.dropoutParam = $root.caffe.DropoutParameter.fromObject(object.dropoutParam);
                }
                if (object.dummyDataParam != null) {
                    if (typeof object.dummyDataParam !== "object")
                        throw TypeError(".caffe.V1LayerParameter.dummyDataParam: object expected");
                    message.dummyDataParam = $root.caffe.DummyDataParameter.fromObject(object.dummyDataParam);
                }
                if (object.eltwiseParam != null) {
                    if (typeof object.eltwiseParam !== "object")
                        throw TypeError(".caffe.V1LayerParameter.eltwiseParam: object expected");
                    message.eltwiseParam = $root.caffe.EltwiseParameter.fromObject(object.eltwiseParam);
                }
                if (object.expParam != null) {
                    if (typeof object.expParam !== "object")
                        throw TypeError(".caffe.V1LayerParameter.expParam: object expected");
                    message.expParam = $root.caffe.ExpParameter.fromObject(object.expParam);
                }
                if (object.hdf5DataParam != null) {
                    if (typeof object.hdf5DataParam !== "object")
                        throw TypeError(".caffe.V1LayerParameter.hdf5DataParam: object expected");
                    message.hdf5DataParam = $root.caffe.HDF5DataParameter.fromObject(object.hdf5DataParam);
                }
                if (object.hdf5OutputParam != null) {
                    if (typeof object.hdf5OutputParam !== "object")
                        throw TypeError(".caffe.V1LayerParameter.hdf5OutputParam: object expected");
                    message.hdf5OutputParam = $root.caffe.HDF5OutputParameter.fromObject(object.hdf5OutputParam);
                }
                if (object.hingeLossParam != null) {
                    if (typeof object.hingeLossParam !== "object")
                        throw TypeError(".caffe.V1LayerParameter.hingeLossParam: object expected");
                    message.hingeLossParam = $root.caffe.HingeLossParameter.fromObject(object.hingeLossParam);
                }
                if (object.imageDataParam != null) {
                    if (typeof object.imageDataParam !== "object")
                        throw TypeError(".caffe.V1LayerParameter.imageDataParam: object expected");
                    message.imageDataParam = $root.caffe.ImageDataParameter.fromObject(object.imageDataParam);
                }
                if (object.infogainLossParam != null) {
                    if (typeof object.infogainLossParam !== "object")
                        throw TypeError(".caffe.V1LayerParameter.infogainLossParam: object expected");
                    message.infogainLossParam = $root.caffe.InfogainLossParameter.fromObject(object.infogainLossParam);
                }
                if (object.innerProductParam != null) {
                    if (typeof object.innerProductParam !== "object")
                        throw TypeError(".caffe.V1LayerParameter.innerProductParam: object expected");
                    message.innerProductParam = $root.caffe.InnerProductParameter.fromObject(object.innerProductParam);
                }
                if (object.lrnParam != null) {
                    if (typeof object.lrnParam !== "object")
                        throw TypeError(".caffe.V1LayerParameter.lrnParam: object expected");
                    message.lrnParam = $root.caffe.LRNParameter.fromObject(object.lrnParam);
                }
                if (object.memoryDataParam != null) {
                    if (typeof object.memoryDataParam !== "object")
                        throw TypeError(".caffe.V1LayerParameter.memoryDataParam: object expected");
                    message.memoryDataParam = $root.caffe.MemoryDataParameter.fromObject(object.memoryDataParam);
                }
                if (object.mvnParam != null) {
                    if (typeof object.mvnParam !== "object")
                        throw TypeError(".caffe.V1LayerParameter.mvnParam: object expected");
                    message.mvnParam = $root.caffe.MVNParameter.fromObject(object.mvnParam);
                }
                if (object.poolingParam != null) {
                    if (typeof object.poolingParam !== "object")
                        throw TypeError(".caffe.V1LayerParameter.poolingParam: object expected");
                    message.poolingParam = $root.caffe.PoolingParameter.fromObject(object.poolingParam);
                }
                if (object.powerParam != null) {
                    if (typeof object.powerParam !== "object")
                        throw TypeError(".caffe.V1LayerParameter.powerParam: object expected");
                    message.powerParam = $root.caffe.PowerParameter.fromObject(object.powerParam);
                }
                if (object.reluParam != null) {
                    if (typeof object.reluParam !== "object")
                        throw TypeError(".caffe.V1LayerParameter.reluParam: object expected");
                    message.reluParam = $root.caffe.ReLUParameter.fromObject(object.reluParam);
                }
                if (object.sigmoidParam != null) {
                    if (typeof object.sigmoidParam !== "object")
                        throw TypeError(".caffe.V1LayerParameter.sigmoidParam: object expected");
                    message.sigmoidParam = $root.caffe.SigmoidParameter.fromObject(object.sigmoidParam);
                }
                if (object.softmaxParam != null) {
                    if (typeof object.softmaxParam !== "object")
                        throw TypeError(".caffe.V1LayerParameter.softmaxParam: object expected");
                    message.softmaxParam = $root.caffe.SoftmaxParameter.fromObject(object.softmaxParam);
                }
                if (object.sliceParam != null) {
                    if (typeof object.sliceParam !== "object")
                        throw TypeError(".caffe.V1LayerParameter.sliceParam: object expected");
                    message.sliceParam = $root.caffe.SliceParameter.fromObject(object.sliceParam);
                }
                if (object.tanhParam != null) {
                    if (typeof object.tanhParam !== "object")
                        throw TypeError(".caffe.V1LayerParameter.tanhParam: object expected");
                    message.tanhParam = $root.caffe.TanHParameter.fromObject(object.tanhParam);
                }
                if (object.thresholdParam != null) {
                    if (typeof object.thresholdParam !== "object")
                        throw TypeError(".caffe.V1LayerParameter.thresholdParam: object expected");
                    message.thresholdParam = $root.caffe.ThresholdParameter.fromObject(object.thresholdParam);
                }
                if (object.windowDataParam != null) {
                    if (typeof object.windowDataParam !== "object")
                        throw TypeError(".caffe.V1LayerParameter.windowDataParam: object expected");
                    message.windowDataParam = $root.caffe.WindowDataParameter.fromObject(object.windowDataParam);
                }
                if (object.transformParam != null) {
                    if (typeof object.transformParam !== "object")
                        throw TypeError(".caffe.V1LayerParameter.transformParam: object expected");
                    message.transformParam = $root.caffe.TransformationParameter.fromObject(object.transformParam);
                }
                if (object.lossParam != null) {
                    if (typeof object.lossParam !== "object")
                        throw TypeError(".caffe.V1LayerParameter.lossParam: object expected");
                    message.lossParam = $root.caffe.LossParameter.fromObject(object.lossParam);
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
                    object.blobsLr = [];
                    object.weightDecay = [];
                    object.include = [];
                    object.exclude = [];
                    object.lossWeight = [];
                    object.param = [];
                    object.blobShareMode = [];
                }
                if (options.defaults) {
                    object.layer = null;
                    object.name = "";
                    object.type = options.enums === String ? "NONE" : 0;
                    object.concatParam = null;
                    object.convolutionParam = null;
                    object.dataParam = null;
                    object.dropoutParam = null;
                    object.hdf5DataParam = null;
                    object.hdf5OutputParam = null;
                    object.imageDataParam = null;
                    object.infogainLossParam = null;
                    object.innerProductParam = null;
                    object.lrnParam = null;
                    object.poolingParam = null;
                    object.windowDataParam = null;
                    object.powerParam = null;
                    object.memoryDataParam = null;
                    object.argmaxParam = null;
                    object.eltwiseParam = null;
                    object.thresholdParam = null;
                    object.dummyDataParam = null;
                    object.accuracyParam = null;
                    object.hingeLossParam = null;
                    object.reluParam = null;
                    object.sliceParam = null;
                    object.mvnParam = null;
                    object.transformParam = null;
                    object.tanhParam = null;
                    object.sigmoidParam = null;
                    object.softmaxParam = null;
                    object.contrastiveLossParam = null;
                    object.expParam = null;
                    object.lossParam = null;
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
                if (message.blobsLr && message.blobsLr.length) {
                    object.blobsLr = [];
                    for (var j = 0; j < message.blobsLr.length; ++j)
                        object.blobsLr[j] = options.json && !isFinite(message.blobsLr[j]) ? String(message.blobsLr[j]) : message.blobsLr[j];
                }
                if (message.weightDecay && message.weightDecay.length) {
                    object.weightDecay = [];
                    for (var j = 0; j < message.weightDecay.length; ++j)
                        object.weightDecay[j] = options.json && !isFinite(message.weightDecay[j]) ? String(message.weightDecay[j]) : message.weightDecay[j];
                }
                if (message.concatParam != null && message.hasOwnProperty("concatParam"))
                    object.concatParam = $root.caffe.ConcatParameter.toObject(message.concatParam, options);
                if (message.convolutionParam != null && message.hasOwnProperty("convolutionParam"))
                    object.convolutionParam = $root.caffe.ConvolutionParameter.toObject(message.convolutionParam, options);
                if (message.dataParam != null && message.hasOwnProperty("dataParam"))
                    object.dataParam = $root.caffe.DataParameter.toObject(message.dataParam, options);
                if (message.dropoutParam != null && message.hasOwnProperty("dropoutParam"))
                    object.dropoutParam = $root.caffe.DropoutParameter.toObject(message.dropoutParam, options);
                if (message.hdf5DataParam != null && message.hasOwnProperty("hdf5DataParam"))
                    object.hdf5DataParam = $root.caffe.HDF5DataParameter.toObject(message.hdf5DataParam, options);
                if (message.hdf5OutputParam != null && message.hasOwnProperty("hdf5OutputParam"))
                    object.hdf5OutputParam = $root.caffe.HDF5OutputParameter.toObject(message.hdf5OutputParam, options);
                if (message.imageDataParam != null && message.hasOwnProperty("imageDataParam"))
                    object.imageDataParam = $root.caffe.ImageDataParameter.toObject(message.imageDataParam, options);
                if (message.infogainLossParam != null && message.hasOwnProperty("infogainLossParam"))
                    object.infogainLossParam = $root.caffe.InfogainLossParameter.toObject(message.infogainLossParam, options);
                if (message.innerProductParam != null && message.hasOwnProperty("innerProductParam"))
                    object.innerProductParam = $root.caffe.InnerProductParameter.toObject(message.innerProductParam, options);
                if (message.lrnParam != null && message.hasOwnProperty("lrnParam"))
                    object.lrnParam = $root.caffe.LRNParameter.toObject(message.lrnParam, options);
                if (message.poolingParam != null && message.hasOwnProperty("poolingParam"))
                    object.poolingParam = $root.caffe.PoolingParameter.toObject(message.poolingParam, options);
                if (message.windowDataParam != null && message.hasOwnProperty("windowDataParam"))
                    object.windowDataParam = $root.caffe.WindowDataParameter.toObject(message.windowDataParam, options);
                if (message.powerParam != null && message.hasOwnProperty("powerParam"))
                    object.powerParam = $root.caffe.PowerParameter.toObject(message.powerParam, options);
                if (message.memoryDataParam != null && message.hasOwnProperty("memoryDataParam"))
                    object.memoryDataParam = $root.caffe.MemoryDataParameter.toObject(message.memoryDataParam, options);
                if (message.argmaxParam != null && message.hasOwnProperty("argmaxParam"))
                    object.argmaxParam = $root.caffe.ArgMaxParameter.toObject(message.argmaxParam, options);
                if (message.eltwiseParam != null && message.hasOwnProperty("eltwiseParam"))
                    object.eltwiseParam = $root.caffe.EltwiseParameter.toObject(message.eltwiseParam, options);
                if (message.thresholdParam != null && message.hasOwnProperty("thresholdParam"))
                    object.thresholdParam = $root.caffe.ThresholdParameter.toObject(message.thresholdParam, options);
                if (message.dummyDataParam != null && message.hasOwnProperty("dummyDataParam"))
                    object.dummyDataParam = $root.caffe.DummyDataParameter.toObject(message.dummyDataParam, options);
                if (message.accuracyParam != null && message.hasOwnProperty("accuracyParam"))
                    object.accuracyParam = $root.caffe.AccuracyParameter.toObject(message.accuracyParam, options);
                if (message.hingeLossParam != null && message.hasOwnProperty("hingeLossParam"))
                    object.hingeLossParam = $root.caffe.HingeLossParameter.toObject(message.hingeLossParam, options);
                if (message.reluParam != null && message.hasOwnProperty("reluParam"))
                    object.reluParam = $root.caffe.ReLUParameter.toObject(message.reluParam, options);
                if (message.sliceParam != null && message.hasOwnProperty("sliceParam"))
                    object.sliceParam = $root.caffe.SliceParameter.toObject(message.sliceParam, options);
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
                if (message.mvnParam != null && message.hasOwnProperty("mvnParam"))
                    object.mvnParam = $root.caffe.MVNParameter.toObject(message.mvnParam, options);
                if (message.lossWeight && message.lossWeight.length) {
                    object.lossWeight = [];
                    for (var j = 0; j < message.lossWeight.length; ++j)
                        object.lossWeight[j] = options.json && !isFinite(message.lossWeight[j]) ? String(message.lossWeight[j]) : message.lossWeight[j];
                }
                if (message.transformParam != null && message.hasOwnProperty("transformParam"))
                    object.transformParam = $root.caffe.TransformationParameter.toObject(message.transformParam, options);
                if (message.tanhParam != null && message.hasOwnProperty("tanhParam"))
                    object.tanhParam = $root.caffe.TanHParameter.toObject(message.tanhParam, options);
                if (message.sigmoidParam != null && message.hasOwnProperty("sigmoidParam"))
                    object.sigmoidParam = $root.caffe.SigmoidParameter.toObject(message.sigmoidParam, options);
                if (message.softmaxParam != null && message.hasOwnProperty("softmaxParam"))
                    object.softmaxParam = $root.caffe.SoftmaxParameter.toObject(message.softmaxParam, options);
                if (message.contrastiveLossParam != null && message.hasOwnProperty("contrastiveLossParam"))
                    object.contrastiveLossParam = $root.caffe.ContrastiveLossParameter.toObject(message.contrastiveLossParam, options);
                if (message.expParam != null && message.hasOwnProperty("expParam"))
                    object.expParam = $root.caffe.ExpParameter.toObject(message.expParam, options);
                if (message.lossParam != null && message.hasOwnProperty("lossParam"))
                    object.lossParam = $root.caffe.LossParameter.toObject(message.lossParam, options);
                if (message.param && message.param.length) {
                    object.param = [];
                    for (var j = 0; j < message.param.length; ++j)
                        object.param[j] = message.param[j];
                }
                if (message.blobShareMode && message.blobShareMode.length) {
                    object.blobShareMode = [];
                    for (var j = 0; j < message.blobShareMode.length; ++j)
                        object.blobShareMode[j] = options.enums === String ? $root.caffe.V1LayerParameter.DimCheckMode[message.blobShareMode[j]] : message.blobShareMode[j];
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
                this.blobsLr = [];
                this.weightDecay = [];
                if (properties)
                    for (var keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                        if (properties[keys[i]] != null)
                            this[keys[i]] = properties[keys[i]];
            }
    
            V0LayerParameter.prototype.name = "";
            V0LayerParameter.prototype.type = "";
            V0LayerParameter.prototype.numOutput = 0;
            V0LayerParameter.prototype.biasterm = true;
            V0LayerParameter.prototype.weightFiller = null;
            V0LayerParameter.prototype.biasFiller = null;
            V0LayerParameter.prototype.pad = 0;
            V0LayerParameter.prototype.kernelsize = 0;
            V0LayerParameter.prototype.group = 1;
            V0LayerParameter.prototype.stride = 1;
            V0LayerParameter.prototype.pool = 0;
            V0LayerParameter.prototype.dropoutRatio = 0.5;
            V0LayerParameter.prototype.localSize = 5;
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
            V0LayerParameter.prototype.blobsLr = $util.emptyArray;
            V0LayerParameter.prototype.weightDecay = $util.emptyArray;
            V0LayerParameter.prototype.randSkip = 0;
            V0LayerParameter.prototype.detFgThreshold = 0.5;
            V0LayerParameter.prototype.detBgThreshold = 0.5;
            V0LayerParameter.prototype.detFgFraction = 0.25;
            V0LayerParameter.prototype.detContextPad = 0;
            V0LayerParameter.prototype.detCropMode = "warp";
            V0LayerParameter.prototype.newNum = 0;
            V0LayerParameter.prototype.newChannels = 0;
            V0LayerParameter.prototype.newHeight = 0;
            V0LayerParameter.prototype.newWidth = 0;
            V0LayerParameter.prototype.shuffleImages = false;
            V0LayerParameter.prototype.concatDim = 1;
            V0LayerParameter.prototype.hdf5OutputParam = null;
    
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
                        message.numOutput = reader.uint32();
                        break;
                    case 4:
                        message.biasterm = reader.bool();
                        break;
                    case 5:
                        message.weightFiller = $root.caffe.FillerParameter.decode(reader, reader.uint32());
                        break;
                    case 6:
                        message.biasFiller = $root.caffe.FillerParameter.decode(reader, reader.uint32());
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
                        message.dropoutRatio = reader.float();
                        break;
                    case 13:
                        message.localSize = reader.uint32();
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
                        if (!(message.blobsLr && message.blobsLr.length))
                            message.blobsLr = [];
                        if ((tag & 7) === 2) {
                            var end2 = reader.uint32() + reader.pos;
                            while (reader.pos < end2)
                                message.blobsLr.push(reader.float());
                        } else
                            message.blobsLr.push(reader.float());
                        break;
                    case 52:
                        if (!(message.weightDecay && message.weightDecay.length))
                            message.weightDecay = [];
                        if ((tag & 7) === 2) {
                            var end2 = reader.uint32() + reader.pos;
                            while (reader.pos < end2)
                                message.weightDecay.push(reader.float());
                        } else
                            message.weightDecay.push(reader.float());
                        break;
                    case 53:
                        message.randSkip = reader.uint32();
                        break;
                    case 54:
                        message.detFgThreshold = reader.float();
                        break;
                    case 55:
                        message.detBgThreshold = reader.float();
                        break;
                    case 56:
                        message.detFgFraction = reader.float();
                        break;
                    case 58:
                        message.detContextPad = reader.uint32();
                        break;
                    case 59:
                        message.detCropMode = reader.string();
                        break;
                    case 60:
                        message.newNum = reader.int32();
                        break;
                    case 61:
                        message.newChannels = reader.int32();
                        break;
                    case 62:
                        message.newHeight = reader.int32();
                        break;
                    case 63:
                        message.newWidth = reader.int32();
                        break;
                    case 64:
                        message.shuffleImages = reader.bool();
                        break;
                    case 65:
                        message.concatDim = reader.uint32();
                        break;
                    case 1001:
                        message.hdf5OutputParam = $root.caffe.HDF5OutputParameter.decode(reader, reader.uint32());
                        break;
                    default:
                        reader.skipType(tag & 7);
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
                if (message.numOutput != null && message.hasOwnProperty("numOutput"))
                    if (!$util.isInteger(message.numOutput))
                        return "numOutput: integer expected";
                if (message.biasterm != null && message.hasOwnProperty("biasterm"))
                    if (typeof message.biasterm !== "boolean")
                        return "biasterm: boolean expected";
                if (message.weightFiller != null && message.hasOwnProperty("weightFiller")) {
                    var error = $root.caffe.FillerParameter.verify(message.weightFiller);
                    if (error)
                        return "weightFiller." + error;
                }
                if (message.biasFiller != null && message.hasOwnProperty("biasFiller")) {
                    var error = $root.caffe.FillerParameter.verify(message.biasFiller);
                    if (error)
                        return "biasFiller." + error;
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
                if (message.dropoutRatio != null && message.hasOwnProperty("dropoutRatio"))
                    if (typeof message.dropoutRatio !== "number")
                        return "dropoutRatio: number expected";
                if (message.localSize != null && message.hasOwnProperty("localSize"))
                    if (!$util.isInteger(message.localSize))
                        return "localSize: integer expected";
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
                if (message.blobsLr != null && message.hasOwnProperty("blobsLr")) {
                    if (!Array.isArray(message.blobsLr))
                        return "blobsLr: array expected";
                    for (var i = 0; i < message.blobsLr.length; ++i)
                        if (typeof message.blobsLr[i] !== "number")
                            return "blobsLr: number[] expected";
                }
                if (message.weightDecay != null && message.hasOwnProperty("weightDecay")) {
                    if (!Array.isArray(message.weightDecay))
                        return "weightDecay: array expected";
                    for (var i = 0; i < message.weightDecay.length; ++i)
                        if (typeof message.weightDecay[i] !== "number")
                            return "weightDecay: number[] expected";
                }
                if (message.randSkip != null && message.hasOwnProperty("randSkip"))
                    if (!$util.isInteger(message.randSkip))
                        return "randSkip: integer expected";
                if (message.detFgThreshold != null && message.hasOwnProperty("detFgThreshold"))
                    if (typeof message.detFgThreshold !== "number")
                        return "detFgThreshold: number expected";
                if (message.detBgThreshold != null && message.hasOwnProperty("detBgThreshold"))
                    if (typeof message.detBgThreshold !== "number")
                        return "detBgThreshold: number expected";
                if (message.detFgFraction != null && message.hasOwnProperty("detFgFraction"))
                    if (typeof message.detFgFraction !== "number")
                        return "detFgFraction: number expected";
                if (message.detContextPad != null && message.hasOwnProperty("detContextPad"))
                    if (!$util.isInteger(message.detContextPad))
                        return "detContextPad: integer expected";
                if (message.detCropMode != null && message.hasOwnProperty("detCropMode"))
                    if (!$util.isString(message.detCropMode))
                        return "detCropMode: string expected";
                if (message.newNum != null && message.hasOwnProperty("newNum"))
                    if (!$util.isInteger(message.newNum))
                        return "newNum: integer expected";
                if (message.newChannels != null && message.hasOwnProperty("newChannels"))
                    if (!$util.isInteger(message.newChannels))
                        return "newChannels: integer expected";
                if (message.newHeight != null && message.hasOwnProperty("newHeight"))
                    if (!$util.isInteger(message.newHeight))
                        return "newHeight: integer expected";
                if (message.newWidth != null && message.hasOwnProperty("newWidth"))
                    if (!$util.isInteger(message.newWidth))
                        return "newWidth: integer expected";
                if (message.shuffleImages != null && message.hasOwnProperty("shuffleImages"))
                    if (typeof message.shuffleImages !== "boolean")
                        return "shuffleImages: boolean expected";
                if (message.concatDim != null && message.hasOwnProperty("concatDim"))
                    if (!$util.isInteger(message.concatDim))
                        return "concatDim: integer expected";
                if (message.hdf5OutputParam != null && message.hasOwnProperty("hdf5OutputParam")) {
                    var error = $root.caffe.HDF5OutputParameter.verify(message.hdf5OutputParam);
                    if (error)
                        return "hdf5OutputParam." + error;
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
                if (object.numOutput != null)
                    message.numOutput = object.numOutput >>> 0;
                if (object.biasterm != null)
                    message.biasterm = Boolean(object.biasterm);
                if (object.weightFiller != null) {
                    if (typeof object.weightFiller !== "object")
                        throw TypeError(".caffe.V0LayerParameter.weightFiller: object expected");
                    message.weightFiller = $root.caffe.FillerParameter.fromObject(object.weightFiller);
                }
                if (object.biasFiller != null) {
                    if (typeof object.biasFiller !== "object")
                        throw TypeError(".caffe.V0LayerParameter.biasFiller: object expected");
                    message.biasFiller = $root.caffe.FillerParameter.fromObject(object.biasFiller);
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
                if (object.dropoutRatio != null)
                    message.dropoutRatio = Number(object.dropoutRatio);
                if (object.localSize != null)
                    message.localSize = object.localSize >>> 0;
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
                if (object.blobsLr) {
                    if (!Array.isArray(object.blobsLr))
                        throw TypeError(".caffe.V0LayerParameter.blobsLr: array expected");
                    message.blobsLr = [];
                    for (var i = 0; i < object.blobsLr.length; ++i)
                        message.blobsLr[i] = Number(object.blobsLr[i]);
                }
                if (object.weightDecay) {
                    if (!Array.isArray(object.weightDecay))
                        throw TypeError(".caffe.V0LayerParameter.weightDecay: array expected");
                    message.weightDecay = [];
                    for (var i = 0; i < object.weightDecay.length; ++i)
                        message.weightDecay[i] = Number(object.weightDecay[i]);
                }
                if (object.randSkip != null)
                    message.randSkip = object.randSkip >>> 0;
                if (object.detFgThreshold != null)
                    message.detFgThreshold = Number(object.detFgThreshold);
                if (object.detBgThreshold != null)
                    message.detBgThreshold = Number(object.detBgThreshold);
                if (object.detFgFraction != null)
                    message.detFgFraction = Number(object.detFgFraction);
                if (object.detContextPad != null)
                    message.detContextPad = object.detContextPad >>> 0;
                if (object.detCropMode != null)
                    message.detCropMode = String(object.detCropMode);
                if (object.newNum != null)
                    message.newNum = object.newNum | 0;
                if (object.newChannels != null)
                    message.newChannels = object.newChannels | 0;
                if (object.newHeight != null)
                    message.newHeight = object.newHeight | 0;
                if (object.newWidth != null)
                    message.newWidth = object.newWidth | 0;
                if (object.shuffleImages != null)
                    message.shuffleImages = Boolean(object.shuffleImages);
                if (object.concatDim != null)
                    message.concatDim = object.concatDim >>> 0;
                if (object.hdf5OutputParam != null) {
                    if (typeof object.hdf5OutputParam !== "object")
                        throw TypeError(".caffe.V0LayerParameter.hdf5OutputParam: object expected");
                    message.hdf5OutputParam = $root.caffe.HDF5OutputParameter.fromObject(object.hdf5OutputParam);
                }
                return message;
            };
    
            V0LayerParameter.toObject = function toObject(message, options) {
                if (!options)
                    options = {};
                var object = {};
                if (options.arrays || options.defaults) {
                    object.blobs = [];
                    object.blobsLr = [];
                    object.weightDecay = [];
                }
                if (options.defaults) {
                    object.name = "";
                    object.type = "";
                    object.numOutput = 0;
                    object.biasterm = true;
                    object.weightFiller = null;
                    object.biasFiller = null;
                    object.pad = 0;
                    object.kernelsize = 0;
                    object.group = 1;
                    object.stride = 1;
                    object.pool = options.enums === String ? "MAX" : 0;
                    object.dropoutRatio = 0.5;
                    object.localSize = 5;
                    object.alpha = 1;
                    object.beta = 0.75;
                    object.source = "";
                    object.scale = 1;
                    object.meanfile = "";
                    object.batchsize = 0;
                    object.cropsize = 0;
                    object.mirror = false;
                    object.k = 1;
                    object.randSkip = 0;
                    object.detFgThreshold = 0.5;
                    object.detBgThreshold = 0.5;
                    object.detFgFraction = 0.25;
                    object.detContextPad = 0;
                    object.detCropMode = "warp";
                    object.newNum = 0;
                    object.newChannels = 0;
                    object.newHeight = 0;
                    object.newWidth = 0;
                    object.shuffleImages = false;
                    object.concatDim = 1;
                    object.hdf5OutputParam = null;
                }
                if (message.name != null && message.hasOwnProperty("name"))
                    object.name = message.name;
                if (message.type != null && message.hasOwnProperty("type"))
                    object.type = message.type;
                if (message.numOutput != null && message.hasOwnProperty("numOutput"))
                    object.numOutput = message.numOutput;
                if (message.biasterm != null && message.hasOwnProperty("biasterm"))
                    object.biasterm = message.biasterm;
                if (message.weightFiller != null && message.hasOwnProperty("weightFiller"))
                    object.weightFiller = $root.caffe.FillerParameter.toObject(message.weightFiller, options);
                if (message.biasFiller != null && message.hasOwnProperty("biasFiller"))
                    object.biasFiller = $root.caffe.FillerParameter.toObject(message.biasFiller, options);
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
                if (message.dropoutRatio != null && message.hasOwnProperty("dropoutRatio"))
                    object.dropoutRatio = options.json && !isFinite(message.dropoutRatio) ? String(message.dropoutRatio) : message.dropoutRatio;
                if (message.localSize != null && message.hasOwnProperty("localSize"))
                    object.localSize = message.localSize;
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
                if (message.blobsLr && message.blobsLr.length) {
                    object.blobsLr = [];
                    for (var j = 0; j < message.blobsLr.length; ++j)
                        object.blobsLr[j] = options.json && !isFinite(message.blobsLr[j]) ? String(message.blobsLr[j]) : message.blobsLr[j];
                }
                if (message.weightDecay && message.weightDecay.length) {
                    object.weightDecay = [];
                    for (var j = 0; j < message.weightDecay.length; ++j)
                        object.weightDecay[j] = options.json && !isFinite(message.weightDecay[j]) ? String(message.weightDecay[j]) : message.weightDecay[j];
                }
                if (message.randSkip != null && message.hasOwnProperty("randSkip"))
                    object.randSkip = message.randSkip;
                if (message.detFgThreshold != null && message.hasOwnProperty("detFgThreshold"))
                    object.detFgThreshold = options.json && !isFinite(message.detFgThreshold) ? String(message.detFgThreshold) : message.detFgThreshold;
                if (message.detBgThreshold != null && message.hasOwnProperty("detBgThreshold"))
                    object.detBgThreshold = options.json && !isFinite(message.detBgThreshold) ? String(message.detBgThreshold) : message.detBgThreshold;
                if (message.detFgFraction != null && message.hasOwnProperty("detFgFraction"))
                    object.detFgFraction = options.json && !isFinite(message.detFgFraction) ? String(message.detFgFraction) : message.detFgFraction;
                if (message.detContextPad != null && message.hasOwnProperty("detContextPad"))
                    object.detContextPad = message.detContextPad;
                if (message.detCropMode != null && message.hasOwnProperty("detCropMode"))
                    object.detCropMode = message.detCropMode;
                if (message.newNum != null && message.hasOwnProperty("newNum"))
                    object.newNum = message.newNum;
                if (message.newChannels != null && message.hasOwnProperty("newChannels"))
                    object.newChannels = message.newChannels;
                if (message.newHeight != null && message.hasOwnProperty("newHeight"))
                    object.newHeight = message.newHeight;
                if (message.newWidth != null && message.hasOwnProperty("newWidth"))
                    object.newWidth = message.newWidth;
                if (message.shuffleImages != null && message.hasOwnProperty("shuffleImages"))
                    object.shuffleImages = message.shuffleImages;
                if (message.concatDim != null && message.hasOwnProperty("concatDim"))
                    object.concatDim = message.concatDim;
                if (message.hdf5OutputParam != null && message.hasOwnProperty("hdf5OutputParam"))
                    object.hdf5OutputParam = $root.caffe.HDF5OutputParameter.toObject(message.hdf5OutputParam, options);
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
            PReLUParameter.prototype.channelShared = false;
    
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
                        message.channelShared = reader.bool();
                        break;
                    default:
                        reader.skipType(tag & 7);
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
                if (message.channelShared != null && message.hasOwnProperty("channelShared"))
                    if (typeof message.channelShared !== "boolean")
                        return "channelShared: boolean expected";
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
                if (object.channelShared != null)
                    message.channelShared = Boolean(object.channelShared);
                return message;
            };
    
            PReLUParameter.toObject = function toObject(message, options) {
                if (!options)
                    options = {};
                var object = {};
                if (options.defaults) {
                    object.filler = null;
                    object.channelShared = false;
                }
                if (message.filler != null && message.hasOwnProperty("filler"))
                    object.filler = $root.caffe.FillerParameter.toObject(message.filler, options);
                if (message.channelShared != null && message.hasOwnProperty("channelShared"))
                    object.channelShared = message.channelShared;
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
