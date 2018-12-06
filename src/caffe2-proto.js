/*eslint-disable block-scoped-var, id-length, no-control-regex, no-magic-numbers, no-prototype-builtins, no-redeclare, no-shadow, no-var, sort-vars*/
(function($protobuf) {
    "use strict";

    var $Reader = $protobuf.Reader, $TextReader = $protobuf.TextReader, $util = $protobuf.util;
    
    var $root = $protobuf.roots.caffe2 || ($protobuf.roots.caffe2 = {});
    
    $root.caffe2 = (function() {
    
        var caffe2 = {};
    
        caffe2.ExternalDataProto = (function() {
    
            function ExternalDataProto(properties) {
                this.strides = [];
                if (properties)
                    for (var keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                        if (properties[keys[i]] != null)
                            this[keys[i]] = properties[keys[i]];
            }
    
            ExternalDataProto.prototype.source_type = 0;
            ExternalDataProto.prototype.record_id = "";
            ExternalDataProto.prototype.record_size = $util.Long ? $util.Long.fromBits(0,0,true) : 0;
            ExternalDataProto.prototype.offset = $util.Long ? $util.Long.fromBits(0,0,false) : 0;
            ExternalDataProto.prototype.strides = $util.emptyArray;
    
            ExternalDataProto.create = function create(properties) {
                return new ExternalDataProto(properties);
            };
    
            ExternalDataProto.decode = function decode(reader, length) {
                if (!(reader instanceof $Reader))
                    reader = $Reader.create(reader);
                var end = length === undefined ? reader.len : reader.pos + length, message = new $root.caffe2.ExternalDataProto();
                while (reader.pos < end) {
                    var tag = reader.uint32();
                    switch (tag >>> 3) {
                    case 1:
                        message.source_type = reader.int32();
                        break;
                    case 2:
                        message.record_id = reader.string();
                        break;
                    case 5:
                        message.record_size = reader.uint64();
                        break;
                    case 3:
                        message.offset = reader.int64();
                        break;
                    case 4:
                        if (!(message.strides && message.strides.length))
                            message.strides = [];
                        if ((tag & 7) === 2) {
                            var end2 = reader.uint32() + reader.pos;
                            while (reader.pos < end2)
                                message.strides.push(reader.int64());
                        } else
                            message.strides.push(reader.int64());
                        break;
                    default:
                        reader.skipType(tag & 7);
                        break;
                    }
                }
                return message;
            };
    
            ExternalDataProto.decodeText = function decodeText(reader, block) {
                if (!(reader instanceof $TextReader))
                    reader = $TextReader.create(reader);
                var message = new $root.caffe2.ExternalDataProto();
                reader.start(block);
                while (!reader.end(block)) {
                    var tag = reader.tag();
                    switch (tag) {
                    case "source_type":
                        message.source_type = reader.enum($root.caffe2.ExternalDataProto.SourceType);
                        break;
                    case "record_id":
                        message.record_id = reader.string();
                        break;
                    case "record_size":
                        message.record_size = reader.uint64();
                        break;
                    case "offset":
                        message.offset = reader.int64();
                        break;
                    case "strides":
                        if (!(message.strides && message.strides.length))
                            message.strides = [];
                        message.strides.push(reader.int64());
                        break;
                    default:
                        reader.handle(tag, message);
                        break;
                    }
                }
                return message;
            };
    
            ExternalDataProto.verify = function verify(message) {
                if (typeof message !== "object" || message === null)
                    return "object expected";
                if (message.source_type != null && message.hasOwnProperty("source_type"))
                    switch (message.source_type) {
                    default:
                        return "source_type: enum value expected";
                    case 0:
                    case 1:
                        break;
                    }
                if (message.record_id != null && message.hasOwnProperty("record_id"))
                    if (!$util.isString(message.record_id))
                        return "record_id: string expected";
                if (message.record_size != null && message.hasOwnProperty("record_size"))
                    if (!$util.isInteger(message.record_size) && !(message.record_size && $util.isInteger(message.record_size.low) && $util.isInteger(message.record_size.high)))
                        return "record_size: integer|Long expected";
                if (message.offset != null && message.hasOwnProperty("offset"))
                    if (!$util.isInteger(message.offset) && !(message.offset && $util.isInteger(message.offset.low) && $util.isInteger(message.offset.high)))
                        return "offset: integer|Long expected";
                if (message.strides != null && message.hasOwnProperty("strides")) {
                    if (!Array.isArray(message.strides))
                        return "strides: array expected";
                    for (var i = 0; i < message.strides.length; ++i)
                        if (!$util.isInteger(message.strides[i]) && !(message.strides[i] && $util.isInteger(message.strides[i].low) && $util.isInteger(message.strides[i].high)))
                            return "strides: integer|Long[] expected";
                }
                return null;
            };
    
            ExternalDataProto.fromObject = function fromObject(object) {
                if (object instanceof $root.caffe2.ExternalDataProto)
                    return object;
                var message = new $root.caffe2.ExternalDataProto();
                switch (object.source_type) {
                case "INLINE_CONTAINER":
                case 0:
                    message.source_type = 0;
                    break;
                case "SIMPLE_FILE":
                case 1:
                    message.source_type = 1;
                    break;
                }
                if (object.record_id != null)
                    message.record_id = String(object.record_id);
                if (object.record_size != null)
                    if ($util.Long)
                        (message.record_size = $util.Long.fromValue(object.record_size)).unsigned = true;
                    else if (typeof object.record_size === "string")
                        message.record_size = parseInt(object.record_size, 10);
                    else if (typeof object.record_size === "number")
                        message.record_size = object.record_size;
                    else if (typeof object.record_size === "object")
                        message.record_size = new $util.LongBits(object.record_size.low >>> 0, object.record_size.high >>> 0).toNumber(true);
                if (object.offset != null)
                    if ($util.Long)
                        (message.offset = $util.Long.fromValue(object.offset)).unsigned = false;
                    else if (typeof object.offset === "string")
                        message.offset = parseInt(object.offset, 10);
                    else if (typeof object.offset === "number")
                        message.offset = object.offset;
                    else if (typeof object.offset === "object")
                        message.offset = new $util.LongBits(object.offset.low >>> 0, object.offset.high >>> 0).toNumber();
                if (object.strides) {
                    if (!Array.isArray(object.strides))
                        throw TypeError(".caffe2.ExternalDataProto.strides: array expected");
                    message.strides = [];
                    for (var i = 0; i < object.strides.length; ++i)
                        if ($util.Long)
                            (message.strides[i] = $util.Long.fromValue(object.strides[i])).unsigned = false;
                        else if (typeof object.strides[i] === "string")
                            message.strides[i] = parseInt(object.strides[i], 10);
                        else if (typeof object.strides[i] === "number")
                            message.strides[i] = object.strides[i];
                        else if (typeof object.strides[i] === "object")
                            message.strides[i] = new $util.LongBits(object.strides[i].low >>> 0, object.strides[i].high >>> 0).toNumber();
                }
                return message;
            };
    
            ExternalDataProto.toObject = function toObject(message, options) {
                if (!options)
                    options = {};
                var object = {};
                if (options.arrays || options.defaults)
                    object.strides = [];
                if (options.defaults) {
                    object.source_type = options.enums === String ? "INLINE_CONTAINER" : 0;
                    object.record_id = "";
                    if ($util.Long) {
                        var long = new $util.Long(0, 0, false);
                        object.offset = options.longs === String ? long.toString() : options.longs === Number ? long.toNumber() : long;
                    } else
                        object.offset = options.longs === String ? "0" : 0;
                    if ($util.Long) {
                        var long = new $util.Long(0, 0, true);
                        object.record_size = options.longs === String ? long.toString() : options.longs === Number ? long.toNumber() : long;
                    } else
                        object.record_size = options.longs === String ? "0" : 0;
                }
                if (message.source_type != null && message.hasOwnProperty("source_type"))
                    object.source_type = options.enums === String ? $root.caffe2.ExternalDataProto.SourceType[message.source_type] : message.source_type;
                if (message.record_id != null && message.hasOwnProperty("record_id"))
                    object.record_id = message.record_id;
                if (message.offset != null && message.hasOwnProperty("offset"))
                    if (typeof message.offset === "number")
                        object.offset = options.longs === String ? String(message.offset) : message.offset;
                    else
                        object.offset = options.longs === String ? $util.Long.prototype.toString.call(message.offset) : options.longs === Number ? new $util.LongBits(message.offset.low >>> 0, message.offset.high >>> 0).toNumber() : message.offset;
                if (message.strides && message.strides.length) {
                    object.strides = [];
                    for (var j = 0; j < message.strides.length; ++j)
                        if (typeof message.strides[j] === "number")
                            object.strides[j] = options.longs === String ? String(message.strides[j]) : message.strides[j];
                        else
                            object.strides[j] = options.longs === String ? $util.Long.prototype.toString.call(message.strides[j]) : options.longs === Number ? new $util.LongBits(message.strides[j].low >>> 0, message.strides[j].high >>> 0).toNumber() : message.strides[j];
                }
                if (message.record_size != null && message.hasOwnProperty("record_size"))
                    if (typeof message.record_size === "number")
                        object.record_size = options.longs === String ? String(message.record_size) : message.record_size;
                    else
                        object.record_size = options.longs === String ? $util.Long.prototype.toString.call(message.record_size) : options.longs === Number ? new $util.LongBits(message.record_size.low >>> 0, message.record_size.high >>> 0).toNumber(true) : message.record_size;
                return object;
            };
    
            ExternalDataProto.prototype.toJSON = function toJSON() {
                return this.constructor.toObject(this, $protobuf.util.toJSONOptions);
            };
    
            ExternalDataProto.SourceType = (function() {
                var valuesById = {}, values = Object.create(valuesById);
                values[valuesById[0] = "INLINE_CONTAINER"] = 0;
                values[valuesById[1] = "SIMPLE_FILE"] = 1;
                return values;
            })();
    
            return ExternalDataProto;
        })();
    
        caffe2.TensorProto = (function() {
    
            function TensorProto(properties) {
                this.dims = [];
                this.float_data = [];
                this.int32_data = [];
                this.string_data = [];
                this.double_data = [];
                this.int64_data = [];
                if (properties)
                    for (var keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                        if (properties[keys[i]] != null)
                            this[keys[i]] = properties[keys[i]];
            }
    
            TensorProto.prototype.dims = $util.emptyArray;
            TensorProto.prototype.data_type = 1;
            TensorProto.prototype.storage_type = 1;
            TensorProto.prototype.float_data = $util.emptyArray;
            TensorProto.prototype.int32_data = $util.emptyArray;
            TensorProto.prototype.byte_data = $util.newBuffer([]);
            TensorProto.prototype.string_data = $util.emptyArray;
            TensorProto.prototype.double_data = $util.emptyArray;
            TensorProto.prototype.int64_data = $util.emptyArray;
            TensorProto.prototype.raw_data = $util.newBuffer([]);
            TensorProto.prototype.external_data = null;
            TensorProto.prototype.name = "";
            TensorProto.prototype.device_detail = null;
            TensorProto.prototype.segment = null;
    
            TensorProto.create = function create(properties) {
                return new TensorProto(properties);
            };
    
            TensorProto.decode = function decode(reader, length) {
                if (!(reader instanceof $Reader))
                    reader = $Reader.create(reader);
                var end = length === undefined ? reader.len : reader.pos + length, message = new $root.caffe2.TensorProto();
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
                    case 12:
                        message.storage_type = reader.int32();
                        break;
                    case 3:
                        if (!(message.float_data && message.float_data.length))
                            message.float_data = [];
                        if ((tag & 7) === 2) {
                            var end2 = reader.uint32() + reader.pos;
                            while (reader.pos < end2)
                                message.float_data.push(reader.float());
                        } else
                            message.float_data.push(reader.float());
                        break;
                    case 4:
                        if (!(message.int32_data && message.int32_data.length))
                            message.int32_data = [];
                        if ((tag & 7) === 2) {
                            var end2 = reader.uint32() + reader.pos;
                            while (reader.pos < end2)
                                message.int32_data.push(reader.int32());
                        } else
                            message.int32_data.push(reader.int32());
                        break;
                    case 5:
                        message.byte_data = reader.bytes();
                        break;
                    case 6:
                        if (!(message.string_data && message.string_data.length))
                            message.string_data = [];
                        message.string_data.push(reader.bytes());
                        break;
                    case 9:
                        if (!(message.double_data && message.double_data.length))
                            message.double_data = [];
                        if ((tag & 7) === 2) {
                            var end2 = reader.uint32() + reader.pos;
                            while (reader.pos < end2)
                                message.double_data.push(reader.double());
                        } else
                            message.double_data.push(reader.double());
                        break;
                    case 10:
                        if (!(message.int64_data && message.int64_data.length))
                            message.int64_data = [];
                        if ((tag & 7) === 2) {
                            var end2 = reader.uint32() + reader.pos;
                            while (reader.pos < end2)
                                message.int64_data.push(reader.int64());
                        } else
                            message.int64_data.push(reader.int64());
                        break;
                    case 13:
                        message.raw_data = reader.bytes();
                        break;
                    case 14:
                        message.external_data = $root.caffe2.ExternalDataProto.decode(reader, reader.uint32());
                        break;
                    case 7:
                        message.name = reader.string();
                        break;
                    case 8:
                        message.device_detail = $root.caffe2.DeviceOption.decode(reader, reader.uint32());
                        break;
                    case 11:
                        message.segment = $root.caffe2.TensorProto.Segment.decode(reader, reader.uint32());
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
                var message = new $root.caffe2.TensorProto();
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
                        message.data_type = reader.enum($root.caffe2.TensorProto.DataType);
                        break;
                    case "storage_type":
                        message.storage_type = reader.enum($root.caffe2.TensorProto.StorageType);
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
                    case "byte_data":
                        message.byte_data = reader.bytes();
                        break;
                    case "string_data":
                        if (!(message.string_data && message.string_data.length))
                            message.string_data = [];
                        message.string_data.push(reader.bytes());
                        break;
                    case "double_data":
                        if (!(message.double_data && message.double_data.length))
                            message.double_data = [];
                        message.double_data.push(reader.double());
                        break;
                    case "int64_data":
                        if (!(message.int64_data && message.int64_data.length))
                            message.int64_data = [];
                        message.int64_data.push(reader.int64());
                        break;
                    case "raw_data":
                        message.raw_data = reader.bytes();
                        break;
                    case "external_data":
                        message.external_data = $root.caffe2.ExternalDataProto.decodeText(reader, true);
                        break;
                    case "name":
                        message.name = reader.string();
                        break;
                    case "device_detail":
                        message.device_detail = $root.caffe2.DeviceOption.decodeText(reader, true);
                        break;
                    case "segment":
                        message.segment = $root.caffe2.TensorProto.Segment.decodeText(reader, true);
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
                    case 7:
                    case 8:
                    case 9:
                    case 10:
                    case 12:
                    case 13:
                        break;
                    }
                if (message.storage_type != null && message.hasOwnProperty("storage_type"))
                    switch (message.storage_type) {
                    default:
                        return "storage_type: enum value expected";
                    case 1:
                    case 2:
                    case 3:
                    case 4:
                        break;
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
                if (message.byte_data != null && message.hasOwnProperty("byte_data"))
                    if (!(message.byte_data && typeof message.byte_data.length === "number" || $util.isString(message.byte_data)))
                        return "byte_data: buffer expected";
                if (message.string_data != null && message.hasOwnProperty("string_data")) {
                    if (!Array.isArray(message.string_data))
                        return "string_data: array expected";
                    for (var i = 0; i < message.string_data.length; ++i)
                        if (!(message.string_data[i] && typeof message.string_data[i].length === "number" || $util.isString(message.string_data[i])))
                            return "string_data: buffer[] expected";
                }
                if (message.double_data != null && message.hasOwnProperty("double_data")) {
                    if (!Array.isArray(message.double_data))
                        return "double_data: array expected";
                    for (var i = 0; i < message.double_data.length; ++i)
                        if (typeof message.double_data[i] !== "number")
                            return "double_data: number[] expected";
                }
                if (message.int64_data != null && message.hasOwnProperty("int64_data")) {
                    if (!Array.isArray(message.int64_data))
                        return "int64_data: array expected";
                    for (var i = 0; i < message.int64_data.length; ++i)
                        if (!$util.isInteger(message.int64_data[i]) && !(message.int64_data[i] && $util.isInteger(message.int64_data[i].low) && $util.isInteger(message.int64_data[i].high)))
                            return "int64_data: integer|Long[] expected";
                }
                if (message.raw_data != null && message.hasOwnProperty("raw_data"))
                    if (!(message.raw_data && typeof message.raw_data.length === "number" || $util.isString(message.raw_data)))
                        return "raw_data: buffer expected";
                if (message.external_data != null && message.hasOwnProperty("external_data")) {
                    var error = $root.caffe2.ExternalDataProto.verify(message.external_data);
                    if (error)
                        return "external_data." + error;
                }
                if (message.name != null && message.hasOwnProperty("name"))
                    if (!$util.isString(message.name))
                        return "name: string expected";
                if (message.device_detail != null && message.hasOwnProperty("device_detail")) {
                    var error = $root.caffe2.DeviceOption.verify(message.device_detail);
                    if (error)
                        return "device_detail." + error;
                }
                if (message.segment != null && message.hasOwnProperty("segment")) {
                    var error = $root.caffe2.TensorProto.Segment.verify(message.segment);
                    if (error)
                        return "segment." + error;
                }
                return null;
            };
    
            TensorProto.fromObject = function fromObject(object) {
                if (object instanceof $root.caffe2.TensorProto)
                    return object;
                var message = new $root.caffe2.TensorProto();
                if (object.dims) {
                    if (!Array.isArray(object.dims))
                        throw TypeError(".caffe2.TensorProto.dims: array expected");
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
                switch (object.data_type) {
                case "UNDEFINED":
                case 0:
                    message.data_type = 0;
                    break;
                case "FLOAT":
                case 1:
                    message.data_type = 1;
                    break;
                case "INT32":
                case 2:
                    message.data_type = 2;
                    break;
                case "BYTE":
                case 3:
                    message.data_type = 3;
                    break;
                case "STRING":
                case 4:
                    message.data_type = 4;
                    break;
                case "BOOL":
                case 5:
                    message.data_type = 5;
                    break;
                case "UINT8":
                case 6:
                    message.data_type = 6;
                    break;
                case "INT8":
                case 7:
                    message.data_type = 7;
                    break;
                case "UINT16":
                case 8:
                    message.data_type = 8;
                    break;
                case "INT16":
                case 9:
                    message.data_type = 9;
                    break;
                case "INT64":
                case 10:
                    message.data_type = 10;
                    break;
                case "FLOAT16":
                case 12:
                    message.data_type = 12;
                    break;
                case "DOUBLE":
                case 13:
                    message.data_type = 13;
                    break;
                }
                switch (object.storage_type) {
                case "TYPED":
                case 1:
                    message.storage_type = 1;
                    break;
                case "RAW":
                case 2:
                    message.storage_type = 2;
                    break;
                case "EXTERNAL":
                case 3:
                    message.storage_type = 3;
                    break;
                case "NO_CONTENT":
                case 4:
                    message.storage_type = 4;
                    break;
                }
                if (object.float_data) {
                    if (!Array.isArray(object.float_data))
                        throw TypeError(".caffe2.TensorProto.float_data: array expected");
                    message.float_data = [];
                    for (var i = 0; i < object.float_data.length; ++i)
                        message.float_data[i] = Number(object.float_data[i]);
                }
                if (object.int32_data) {
                    if (!Array.isArray(object.int32_data))
                        throw TypeError(".caffe2.TensorProto.int32_data: array expected");
                    message.int32_data = [];
                    for (var i = 0; i < object.int32_data.length; ++i)
                        message.int32_data[i] = object.int32_data[i] | 0;
                }
                if (object.byte_data != null)
                    if (typeof object.byte_data === "string")
                        $util.base64.decode(object.byte_data, message.byte_data = $util.newBuffer($util.base64.length(object.byte_data)), 0);
                    else if (object.byte_data.length)
                        message.byte_data = object.byte_data;
                if (object.string_data) {
                    if (!Array.isArray(object.string_data))
                        throw TypeError(".caffe2.TensorProto.string_data: array expected");
                    message.string_data = [];
                    for (var i = 0; i < object.string_data.length; ++i)
                        if (typeof object.string_data[i] === "string")
                            $util.base64.decode(object.string_data[i], message.string_data[i] = $util.newBuffer($util.base64.length(object.string_data[i])), 0);
                        else if (object.string_data[i].length)
                            message.string_data[i] = object.string_data[i];
                }
                if (object.double_data) {
                    if (!Array.isArray(object.double_data))
                        throw TypeError(".caffe2.TensorProto.double_data: array expected");
                    message.double_data = [];
                    for (var i = 0; i < object.double_data.length; ++i)
                        message.double_data[i] = Number(object.double_data[i]);
                }
                if (object.int64_data) {
                    if (!Array.isArray(object.int64_data))
                        throw TypeError(".caffe2.TensorProto.int64_data: array expected");
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
                if (object.raw_data != null)
                    if (typeof object.raw_data === "string")
                        $util.base64.decode(object.raw_data, message.raw_data = $util.newBuffer($util.base64.length(object.raw_data)), 0);
                    else if (object.raw_data.length)
                        message.raw_data = object.raw_data;
                if (object.external_data != null) {
                    if (typeof object.external_data !== "object")
                        throw TypeError(".caffe2.TensorProto.external_data: object expected");
                    message.external_data = $root.caffe2.ExternalDataProto.fromObject(object.external_data);
                }
                if (object.name != null)
                    message.name = String(object.name);
                if (object.device_detail != null) {
                    if (typeof object.device_detail !== "object")
                        throw TypeError(".caffe2.TensorProto.device_detail: object expected");
                    message.device_detail = $root.caffe2.DeviceOption.fromObject(object.device_detail);
                }
                if (object.segment != null) {
                    if (typeof object.segment !== "object")
                        throw TypeError(".caffe2.TensorProto.segment: object expected");
                    message.segment = $root.caffe2.TensorProto.Segment.fromObject(object.segment);
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
                    object.double_data = [];
                    object.int64_data = [];
                }
                if (options.defaults) {
                    object.data_type = options.enums === String ? "FLOAT" : 1;
                    if (options.bytes === String)
                        object.byte_data = "";
                    else {
                        object.byte_data = [];
                        if (options.bytes !== Array)
                            object.byte_data = $util.newBuffer(object.byte_data);
                    }
                    object.name = "";
                    object.device_detail = null;
                    object.segment = null;
                    object.storage_type = options.enums === String ? "TYPED" : 1;
                    if (options.bytes === String)
                        object.raw_data = "";
                    else {
                        object.raw_data = [];
                        if (options.bytes !== Array)
                            object.raw_data = $util.newBuffer(object.raw_data);
                    }
                    object.external_data = null;
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
                    object.data_type = options.enums === String ? $root.caffe2.TensorProto.DataType[message.data_type] : message.data_type;
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
                if (message.byte_data != null && message.hasOwnProperty("byte_data"))
                    object.byte_data = options.bytes === String ? $util.base64.encode(message.byte_data, 0, message.byte_data.length) : options.bytes === Array ? Array.prototype.slice.call(message.byte_data) : message.byte_data;
                if (message.string_data && message.string_data.length) {
                    object.string_data = [];
                    for (var j = 0; j < message.string_data.length; ++j)
                        object.string_data[j] = options.bytes === String ? $util.base64.encode(message.string_data[j], 0, message.string_data[j].length) : options.bytes === Array ? Array.prototype.slice.call(message.string_data[j]) : message.string_data[j];
                }
                if (message.name != null && message.hasOwnProperty("name"))
                    object.name = message.name;
                if (message.device_detail != null && message.hasOwnProperty("device_detail"))
                    object.device_detail = $root.caffe2.DeviceOption.toObject(message.device_detail, options);
                if (message.double_data && message.double_data.length) {
                    object.double_data = [];
                    for (var j = 0; j < message.double_data.length; ++j)
                        object.double_data[j] = options.json && !isFinite(message.double_data[j]) ? String(message.double_data[j]) : message.double_data[j];
                }
                if (message.int64_data && message.int64_data.length) {
                    object.int64_data = [];
                    for (var j = 0; j < message.int64_data.length; ++j)
                        if (typeof message.int64_data[j] === "number")
                            object.int64_data[j] = options.longs === String ? String(message.int64_data[j]) : message.int64_data[j];
                        else
                            object.int64_data[j] = options.longs === String ? $util.Long.prototype.toString.call(message.int64_data[j]) : options.longs === Number ? new $util.LongBits(message.int64_data[j].low >>> 0, message.int64_data[j].high >>> 0).toNumber() : message.int64_data[j];
                }
                if (message.segment != null && message.hasOwnProperty("segment"))
                    object.segment = $root.caffe2.TensorProto.Segment.toObject(message.segment, options);
                if (message.storage_type != null && message.hasOwnProperty("storage_type"))
                    object.storage_type = options.enums === String ? $root.caffe2.TensorProto.StorageType[message.storage_type] : message.storage_type;
                if (message.raw_data != null && message.hasOwnProperty("raw_data"))
                    object.raw_data = options.bytes === String ? $util.base64.encode(message.raw_data, 0, message.raw_data.length) : options.bytes === Array ? Array.prototype.slice.call(message.raw_data) : message.raw_data;
                if (message.external_data != null && message.hasOwnProperty("external_data"))
                    object.external_data = $root.caffe2.ExternalDataProto.toObject(message.external_data, options);
                return object;
            };
    
            TensorProto.prototype.toJSON = function toJSON() {
                return this.constructor.toObject(this, $protobuf.util.toJSONOptions);
            };
    
            TensorProto.DataType = (function() {
                var valuesById = {}, values = Object.create(valuesById);
                values[valuesById[0] = "UNDEFINED"] = 0;
                values[valuesById[1] = "FLOAT"] = 1;
                values[valuesById[2] = "INT32"] = 2;
                values[valuesById[3] = "BYTE"] = 3;
                values[valuesById[4] = "STRING"] = 4;
                values[valuesById[5] = "BOOL"] = 5;
                values[valuesById[6] = "UINT8"] = 6;
                values[valuesById[7] = "INT8"] = 7;
                values[valuesById[8] = "UINT16"] = 8;
                values[valuesById[9] = "INT16"] = 9;
                values[valuesById[10] = "INT64"] = 10;
                values[valuesById[12] = "FLOAT16"] = 12;
                values[valuesById[13] = "DOUBLE"] = 13;
                return values;
            })();
    
            TensorProto.StorageType = (function() {
                var valuesById = {}, values = Object.create(valuesById);
                values[valuesById[1] = "TYPED"] = 1;
                values[valuesById[2] = "RAW"] = 2;
                values[valuesById[3] = "EXTERNAL"] = 3;
                values[valuesById[4] = "NO_CONTENT"] = 4;
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
                    var end = length === undefined ? reader.len : reader.pos + length, message = new $root.caffe2.TensorProto.Segment();
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
                    if (!message.hasOwnProperty("begin"))
                        throw $util.ProtocolError("missing required 'begin'", { instance: message });
                    if (!message.hasOwnProperty("end"))
                        throw $util.ProtocolError("missing required 'end'", { instance: message });
                    return message;
                };
    
                Segment.decodeText = function decodeText(reader, block) {
                    if (!(reader instanceof $TextReader))
                        reader = $TextReader.create(reader);
                    var message = new $root.caffe2.TensorProto.Segment();
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
                    if (!message.hasOwnProperty("begin"))
                        throw $util.ProtocolError("missing required 'begin'", { instance: message });
                    if (!message.hasOwnProperty("end"))
                        throw $util.ProtocolError("missing required 'end'", { instance: message });
                    return message;
                };
    
                Segment.verify = function verify(message) {
                    if (typeof message !== "object" || message === null)
                        return "object expected";
                    if (!$util.isInteger(message.begin) && !(message.begin && $util.isInteger(message.begin.low) && $util.isInteger(message.begin.high)))
                        return "begin: integer|Long expected";
                    if (!$util.isInteger(message.end) && !(message.end && $util.isInteger(message.end.low) && $util.isInteger(message.end.high)))
                        return "end: integer|Long expected";
                    return null;
                };
    
                Segment.fromObject = function fromObject(object) {
                    if (object instanceof $root.caffe2.TensorProto.Segment)
                        return object;
                    var message = new $root.caffe2.TensorProto.Segment();
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
    
        caffe2.QTensorProto = (function() {
    
            function QTensorProto(properties) {
                this.dims = [];
                this.data = [];
                if (properties)
                    for (var keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                        if (properties[keys[i]] != null)
                            this[keys[i]] = properties[keys[i]];
            }
    
            QTensorProto.prototype.dims = $util.emptyArray;
            QTensorProto.prototype.precision = 0;
            QTensorProto.prototype.scale = 0;
            QTensorProto.prototype.bias = 0;
            QTensorProto.prototype.is_signed = false;
            QTensorProto.prototype.data = $util.emptyArray;
            QTensorProto.prototype.name = "";
            QTensorProto.prototype.data_type = 2;
    
            QTensorProto.create = function create(properties) {
                return new QTensorProto(properties);
            };
    
            QTensorProto.decode = function decode(reader, length) {
                if (!(reader instanceof $Reader))
                    reader = $Reader.create(reader);
                var end = length === undefined ? reader.len : reader.pos + length, message = new $root.caffe2.QTensorProto();
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
                        message.precision = reader.int32();
                        break;
                    case 3:
                        message.scale = reader.double();
                        break;
                    case 4:
                        message.bias = reader.double();
                        break;
                    case 5:
                        message.is_signed = reader.bool();
                        break;
                    case 6:
                        if (!(message.data && message.data.length))
                            message.data = [];
                        if ((tag & 7) === 2) {
                            var end2 = reader.uint32() + reader.pos;
                            while (reader.pos < end2)
                                message.data.push(reader.int32());
                        } else
                            message.data.push(reader.int32());
                        break;
                    case 7:
                        message.name = reader.string();
                        break;
                    case 8:
                        message.data_type = reader.int32();
                        break;
                    default:
                        reader.skipType(tag & 7);
                        break;
                    }
                }
                if (!message.hasOwnProperty("precision"))
                    throw $util.ProtocolError("missing required 'precision'", { instance: message });
                if (!message.hasOwnProperty("scale"))
                    throw $util.ProtocolError("missing required 'scale'", { instance: message });
                if (!message.hasOwnProperty("bias"))
                    throw $util.ProtocolError("missing required 'bias'", { instance: message });
                if (!message.hasOwnProperty("is_signed"))
                    throw $util.ProtocolError("missing required 'is_signed'", { instance: message });
                return message;
            };
    
            QTensorProto.decodeText = function decodeText(reader, block) {
                if (!(reader instanceof $TextReader))
                    reader = $TextReader.create(reader);
                var message = new $root.caffe2.QTensorProto();
                reader.start(block);
                while (!reader.end(block)) {
                    var tag = reader.tag();
                    switch (tag) {
                    case "dims":
                        if (!(message.dims && message.dims.length))
                            message.dims = [];
                        message.dims.push(reader.int64());
                        break;
                    case "precision":
                        message.precision = reader.int32();
                        break;
                    case "scale":
                        message.scale = reader.double();
                        break;
                    case "bias":
                        message.bias = reader.double();
                        break;
                    case "is_signed":
                        message.is_signed = reader.bool();
                        break;
                    case "data":
                        if (!(message.data && message.data.length))
                            message.data = [];
                        message.data.push(reader.int32());
                        break;
                    case "name":
                        message.name = reader.string();
                        break;
                    case "data_type":
                        message.data_type = reader.enum($root.caffe2.TensorProto.DataType);
                        break;
                    default:
                        reader.handle(tag, message);
                        break;
                    }
                }
                if (!message.hasOwnProperty("precision"))
                    throw $util.ProtocolError("missing required 'precision'", { instance: message });
                if (!message.hasOwnProperty("scale"))
                    throw $util.ProtocolError("missing required 'scale'", { instance: message });
                if (!message.hasOwnProperty("bias"))
                    throw $util.ProtocolError("missing required 'bias'", { instance: message });
                if (!message.hasOwnProperty("is_signed"))
                    throw $util.ProtocolError("missing required 'is_signed'", { instance: message });
                return message;
            };
    
            QTensorProto.verify = function verify(message) {
                if (typeof message !== "object" || message === null)
                    return "object expected";
                if (message.dims != null && message.hasOwnProperty("dims")) {
                    if (!Array.isArray(message.dims))
                        return "dims: array expected";
                    for (var i = 0; i < message.dims.length; ++i)
                        if (!$util.isInteger(message.dims[i]) && !(message.dims[i] && $util.isInteger(message.dims[i].low) && $util.isInteger(message.dims[i].high)))
                            return "dims: integer|Long[] expected";
                }
                if (!$util.isInteger(message.precision))
                    return "precision: integer expected";
                if (typeof message.scale !== "number")
                    return "scale: number expected";
                if (typeof message.bias !== "number")
                    return "bias: number expected";
                if (typeof message.is_signed !== "boolean")
                    return "is_signed: boolean expected";
                if (message.data != null && message.hasOwnProperty("data")) {
                    if (!Array.isArray(message.data))
                        return "data: array expected";
                    for (var i = 0; i < message.data.length; ++i)
                        if (!$util.isInteger(message.data[i]))
                            return "data: integer[] expected";
                }
                if (message.name != null && message.hasOwnProperty("name"))
                    if (!$util.isString(message.name))
                        return "name: string expected";
                if (message.data_type != null && message.hasOwnProperty("data_type"))
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
                    case 7:
                    case 8:
                    case 9:
                    case 10:
                    case 12:
                    case 13:
                        break;
                    }
                return null;
            };
    
            QTensorProto.fromObject = function fromObject(object) {
                if (object instanceof $root.caffe2.QTensorProto)
                    return object;
                var message = new $root.caffe2.QTensorProto();
                if (object.dims) {
                    if (!Array.isArray(object.dims))
                        throw TypeError(".caffe2.QTensorProto.dims: array expected");
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
                if (object.precision != null)
                    message.precision = object.precision | 0;
                if (object.scale != null)
                    message.scale = Number(object.scale);
                if (object.bias != null)
                    message.bias = Number(object.bias);
                if (object.is_signed != null)
                    message.is_signed = Boolean(object.is_signed);
                if (object.data) {
                    if (!Array.isArray(object.data))
                        throw TypeError(".caffe2.QTensorProto.data: array expected");
                    message.data = [];
                    for (var i = 0; i < object.data.length; ++i)
                        message.data[i] = object.data[i] | 0;
                }
                if (object.name != null)
                    message.name = String(object.name);
                switch (object.data_type) {
                case "UNDEFINED":
                case 0:
                    message.data_type = 0;
                    break;
                case "FLOAT":
                case 1:
                    message.data_type = 1;
                    break;
                case "INT32":
                case 2:
                    message.data_type = 2;
                    break;
                case "BYTE":
                case 3:
                    message.data_type = 3;
                    break;
                case "STRING":
                case 4:
                    message.data_type = 4;
                    break;
                case "BOOL":
                case 5:
                    message.data_type = 5;
                    break;
                case "UINT8":
                case 6:
                    message.data_type = 6;
                    break;
                case "INT8":
                case 7:
                    message.data_type = 7;
                    break;
                case "UINT16":
                case 8:
                    message.data_type = 8;
                    break;
                case "INT16":
                case 9:
                    message.data_type = 9;
                    break;
                case "INT64":
                case 10:
                    message.data_type = 10;
                    break;
                case "FLOAT16":
                case 12:
                    message.data_type = 12;
                    break;
                case "DOUBLE":
                case 13:
                    message.data_type = 13;
                    break;
                }
                return message;
            };
    
            QTensorProto.toObject = function toObject(message, options) {
                if (!options)
                    options = {};
                var object = {};
                if (options.arrays || options.defaults) {
                    object.dims = [];
                    object.data = [];
                }
                if (options.defaults) {
                    object.precision = 0;
                    object.scale = 0;
                    object.bias = 0;
                    object.is_signed = false;
                    object.name = "";
                    object.data_type = options.enums === String ? "INT32" : 2;
                }
                if (message.dims && message.dims.length) {
                    object.dims = [];
                    for (var j = 0; j < message.dims.length; ++j)
                        if (typeof message.dims[j] === "number")
                            object.dims[j] = options.longs === String ? String(message.dims[j]) : message.dims[j];
                        else
                            object.dims[j] = options.longs === String ? $util.Long.prototype.toString.call(message.dims[j]) : options.longs === Number ? new $util.LongBits(message.dims[j].low >>> 0, message.dims[j].high >>> 0).toNumber() : message.dims[j];
                }
                if (message.precision != null && message.hasOwnProperty("precision"))
                    object.precision = message.precision;
                if (message.scale != null && message.hasOwnProperty("scale"))
                    object.scale = options.json && !isFinite(message.scale) ? String(message.scale) : message.scale;
                if (message.bias != null && message.hasOwnProperty("bias"))
                    object.bias = options.json && !isFinite(message.bias) ? String(message.bias) : message.bias;
                if (message.is_signed != null && message.hasOwnProperty("is_signed"))
                    object.is_signed = message.is_signed;
                if (message.data && message.data.length) {
                    object.data = [];
                    for (var j = 0; j < message.data.length; ++j)
                        object.data[j] = message.data[j];
                }
                if (message.name != null && message.hasOwnProperty("name"))
                    object.name = message.name;
                if (message.data_type != null && message.hasOwnProperty("data_type"))
                    object.data_type = options.enums === String ? $root.caffe2.TensorProto.DataType[message.data_type] : message.data_type;
                return object;
            };
    
            QTensorProto.prototype.toJSON = function toJSON() {
                return this.constructor.toObject(this, $protobuf.util.toJSONOptions);
            };
    
            return QTensorProto;
        })();
    
        caffe2.TensorProtos = (function() {
    
            function TensorProtos(properties) {
                this.protos = [];
                if (properties)
                    for (var keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                        if (properties[keys[i]] != null)
                            this[keys[i]] = properties[keys[i]];
            }
    
            TensorProtos.prototype.protos = $util.emptyArray;
    
            TensorProtos.create = function create(properties) {
                return new TensorProtos(properties);
            };
    
            TensorProtos.decode = function decode(reader, length) {
                if (!(reader instanceof $Reader))
                    reader = $Reader.create(reader);
                var end = length === undefined ? reader.len : reader.pos + length, message = new $root.caffe2.TensorProtos();
                while (reader.pos < end) {
                    var tag = reader.uint32();
                    switch (tag >>> 3) {
                    case 1:
                        if (!(message.protos && message.protos.length))
                            message.protos = [];
                        message.protos.push($root.caffe2.TensorProto.decode(reader, reader.uint32()));
                        break;
                    default:
                        reader.skipType(tag & 7);
                        break;
                    }
                }
                return message;
            };
    
            TensorProtos.decodeText = function decodeText(reader, block) {
                if (!(reader instanceof $TextReader))
                    reader = $TextReader.create(reader);
                var message = new $root.caffe2.TensorProtos();
                reader.start(block);
                while (!reader.end(block)) {
                    var tag = reader.tag();
                    switch (tag) {
                    case "protos":
                        if (!(message.protos && message.protos.length))
                            message.protos = [];
                        message.protos.push($root.caffe2.TensorProto.decodeText(reader, true));
                        break;
                    default:
                        reader.handle(tag, message);
                        break;
                    }
                }
                return message;
            };
    
            TensorProtos.verify = function verify(message) {
                if (typeof message !== "object" || message === null)
                    return "object expected";
                if (message.protos != null && message.hasOwnProperty("protos")) {
                    if (!Array.isArray(message.protos))
                        return "protos: array expected";
                    for (var i = 0; i < message.protos.length; ++i) {
                        var error = $root.caffe2.TensorProto.verify(message.protos[i]);
                        if (error)
                            return "protos." + error;
                    }
                }
                return null;
            };
    
            TensorProtos.fromObject = function fromObject(object) {
                if (object instanceof $root.caffe2.TensorProtos)
                    return object;
                var message = new $root.caffe2.TensorProtos();
                if (object.protos) {
                    if (!Array.isArray(object.protos))
                        throw TypeError(".caffe2.TensorProtos.protos: array expected");
                    message.protos = [];
                    for (var i = 0; i < object.protos.length; ++i) {
                        if (typeof object.protos[i] !== "object")
                            throw TypeError(".caffe2.TensorProtos.protos: object expected");
                        message.protos[i] = $root.caffe2.TensorProto.fromObject(object.protos[i]);
                    }
                }
                return message;
            };
    
            TensorProtos.toObject = function toObject(message, options) {
                if (!options)
                    options = {};
                var object = {};
                if (options.arrays || options.defaults)
                    object.protos = [];
                if (message.protos && message.protos.length) {
                    object.protos = [];
                    for (var j = 0; j < message.protos.length; ++j)
                        object.protos[j] = $root.caffe2.TensorProto.toObject(message.protos[j], options);
                }
                return object;
            };
    
            TensorProtos.prototype.toJSON = function toJSON() {
                return this.constructor.toObject(this, $protobuf.util.toJSONOptions);
            };
    
            return TensorProtos;
        })();
    
        caffe2.TensorShape = (function() {
    
            function TensorShape(properties) {
                this.dims = [];
                this.unknown_dims = [];
                if (properties)
                    for (var keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                        if (properties[keys[i]] != null)
                            this[keys[i]] = properties[keys[i]];
            }
    
            TensorShape.prototype.dims = $util.emptyArray;
            TensorShape.prototype.data_type = 1;
            TensorShape.prototype.unknown_dims = $util.emptyArray;
            TensorShape.prototype.unknown_shape = false;
            TensorShape.prototype.name = "";
    
            TensorShape.create = function create(properties) {
                return new TensorShape(properties);
            };
    
            TensorShape.decode = function decode(reader, length) {
                if (!(reader instanceof $Reader))
                    reader = $Reader.create(reader);
                var end = length === undefined ? reader.len : reader.pos + length, message = new $root.caffe2.TensorShape();
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
                        if (!(message.unknown_dims && message.unknown_dims.length))
                            message.unknown_dims = [];
                        if ((tag & 7) === 2) {
                            var end2 = reader.uint32() + reader.pos;
                            while (reader.pos < end2)
                                message.unknown_dims.push(reader.int32());
                        } else
                            message.unknown_dims.push(reader.int32());
                        break;
                    case 4:
                        message.unknown_shape = reader.bool();
                        break;
                    case 5:
                        message.name = reader.string();
                        break;
                    default:
                        reader.skipType(tag & 7);
                        break;
                    }
                }
                return message;
            };
    
            TensorShape.decodeText = function decodeText(reader, block) {
                if (!(reader instanceof $TextReader))
                    reader = $TextReader.create(reader);
                var message = new $root.caffe2.TensorShape();
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
                        message.data_type = reader.enum($root.caffe2.TensorProto.DataType);
                        break;
                    case "unknown_dims":
                        if (!(message.unknown_dims && message.unknown_dims.length))
                            message.unknown_dims = [];
                        message.unknown_dims.push(reader.int32());
                        break;
                    case "unknown_shape":
                        message.unknown_shape = reader.bool();
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
    
            TensorShape.verify = function verify(message) {
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
                    case 7:
                    case 8:
                    case 9:
                    case 10:
                    case 12:
                    case 13:
                        break;
                    }
                if (message.unknown_dims != null && message.hasOwnProperty("unknown_dims")) {
                    if (!Array.isArray(message.unknown_dims))
                        return "unknown_dims: array expected";
                    for (var i = 0; i < message.unknown_dims.length; ++i)
                        if (!$util.isInteger(message.unknown_dims[i]))
                            return "unknown_dims: integer[] expected";
                }
                if (message.unknown_shape != null && message.hasOwnProperty("unknown_shape"))
                    if (typeof message.unknown_shape !== "boolean")
                        return "unknown_shape: boolean expected";
                if (message.name != null && message.hasOwnProperty("name"))
                    if (!$util.isString(message.name))
                        return "name: string expected";
                return null;
            };
    
            TensorShape.fromObject = function fromObject(object) {
                if (object instanceof $root.caffe2.TensorShape)
                    return object;
                var message = new $root.caffe2.TensorShape();
                if (object.dims) {
                    if (!Array.isArray(object.dims))
                        throw TypeError(".caffe2.TensorShape.dims: array expected");
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
                switch (object.data_type) {
                case "UNDEFINED":
                case 0:
                    message.data_type = 0;
                    break;
                case "FLOAT":
                case 1:
                    message.data_type = 1;
                    break;
                case "INT32":
                case 2:
                    message.data_type = 2;
                    break;
                case "BYTE":
                case 3:
                    message.data_type = 3;
                    break;
                case "STRING":
                case 4:
                    message.data_type = 4;
                    break;
                case "BOOL":
                case 5:
                    message.data_type = 5;
                    break;
                case "UINT8":
                case 6:
                    message.data_type = 6;
                    break;
                case "INT8":
                case 7:
                    message.data_type = 7;
                    break;
                case "UINT16":
                case 8:
                    message.data_type = 8;
                    break;
                case "INT16":
                case 9:
                    message.data_type = 9;
                    break;
                case "INT64":
                case 10:
                    message.data_type = 10;
                    break;
                case "FLOAT16":
                case 12:
                    message.data_type = 12;
                    break;
                case "DOUBLE":
                case 13:
                    message.data_type = 13;
                    break;
                }
                if (object.unknown_dims) {
                    if (!Array.isArray(object.unknown_dims))
                        throw TypeError(".caffe2.TensorShape.unknown_dims: array expected");
                    message.unknown_dims = [];
                    for (var i = 0; i < object.unknown_dims.length; ++i)
                        message.unknown_dims[i] = object.unknown_dims[i] | 0;
                }
                if (object.unknown_shape != null)
                    message.unknown_shape = Boolean(object.unknown_shape);
                if (object.name != null)
                    message.name = String(object.name);
                return message;
            };
    
            TensorShape.toObject = function toObject(message, options) {
                if (!options)
                    options = {};
                var object = {};
                if (options.arrays || options.defaults) {
                    object.dims = [];
                    object.unknown_dims = [];
                }
                if (options.defaults) {
                    object.data_type = options.enums === String ? "FLOAT" : 1;
                    object.unknown_shape = false;
                    object.name = "";
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
                    object.data_type = options.enums === String ? $root.caffe2.TensorProto.DataType[message.data_type] : message.data_type;
                if (message.unknown_dims && message.unknown_dims.length) {
                    object.unknown_dims = [];
                    for (var j = 0; j < message.unknown_dims.length; ++j)
                        object.unknown_dims[j] = message.unknown_dims[j];
                }
                if (message.unknown_shape != null && message.hasOwnProperty("unknown_shape"))
                    object.unknown_shape = message.unknown_shape;
                if (message.name != null && message.hasOwnProperty("name"))
                    object.name = message.name;
                return object;
            };
    
            TensorShape.prototype.toJSON = function toJSON() {
                return this.constructor.toObject(this, $protobuf.util.toJSONOptions);
            };
    
            return TensorShape;
        })();
    
        caffe2.TensorShapes = (function() {
    
            function TensorShapes(properties) {
                this.shapes = [];
                if (properties)
                    for (var keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                        if (properties[keys[i]] != null)
                            this[keys[i]] = properties[keys[i]];
            }
    
            TensorShapes.prototype.shapes = $util.emptyArray;
    
            TensorShapes.create = function create(properties) {
                return new TensorShapes(properties);
            };
    
            TensorShapes.decode = function decode(reader, length) {
                if (!(reader instanceof $Reader))
                    reader = $Reader.create(reader);
                var end = length === undefined ? reader.len : reader.pos + length, message = new $root.caffe2.TensorShapes();
                while (reader.pos < end) {
                    var tag = reader.uint32();
                    switch (tag >>> 3) {
                    case 1:
                        if (!(message.shapes && message.shapes.length))
                            message.shapes = [];
                        message.shapes.push($root.caffe2.TensorShape.decode(reader, reader.uint32()));
                        break;
                    default:
                        reader.skipType(tag & 7);
                        break;
                    }
                }
                return message;
            };
    
            TensorShapes.decodeText = function decodeText(reader, block) {
                if (!(reader instanceof $TextReader))
                    reader = $TextReader.create(reader);
                var message = new $root.caffe2.TensorShapes();
                reader.start(block);
                while (!reader.end(block)) {
                    var tag = reader.tag();
                    switch (tag) {
                    case "shapes":
                        if (!(message.shapes && message.shapes.length))
                            message.shapes = [];
                        message.shapes.push($root.caffe2.TensorShape.decodeText(reader, true));
                        break;
                    default:
                        reader.handle(tag, message);
                        break;
                    }
                }
                return message;
            };
    
            TensorShapes.verify = function verify(message) {
                if (typeof message !== "object" || message === null)
                    return "object expected";
                if (message.shapes != null && message.hasOwnProperty("shapes")) {
                    if (!Array.isArray(message.shapes))
                        return "shapes: array expected";
                    for (var i = 0; i < message.shapes.length; ++i) {
                        var error = $root.caffe2.TensorShape.verify(message.shapes[i]);
                        if (error)
                            return "shapes." + error;
                    }
                }
                return null;
            };
    
            TensorShapes.fromObject = function fromObject(object) {
                if (object instanceof $root.caffe2.TensorShapes)
                    return object;
                var message = new $root.caffe2.TensorShapes();
                if (object.shapes) {
                    if (!Array.isArray(object.shapes))
                        throw TypeError(".caffe2.TensorShapes.shapes: array expected");
                    message.shapes = [];
                    for (var i = 0; i < object.shapes.length; ++i) {
                        if (typeof object.shapes[i] !== "object")
                            throw TypeError(".caffe2.TensorShapes.shapes: object expected");
                        message.shapes[i] = $root.caffe2.TensorShape.fromObject(object.shapes[i]);
                    }
                }
                return message;
            };
    
            TensorShapes.toObject = function toObject(message, options) {
                if (!options)
                    options = {};
                var object = {};
                if (options.arrays || options.defaults)
                    object.shapes = [];
                if (message.shapes && message.shapes.length) {
                    object.shapes = [];
                    for (var j = 0; j < message.shapes.length; ++j)
                        object.shapes[j] = $root.caffe2.TensorShape.toObject(message.shapes[j], options);
                }
                return object;
            };
    
            TensorShapes.prototype.toJSON = function toJSON() {
                return this.constructor.toObject(this, $protobuf.util.toJSONOptions);
            };
    
            return TensorShapes;
        })();
    
        caffe2.Argument = (function() {
    
            function Argument(properties) {
                this.floats = [];
                this.ints = [];
                this.strings = [];
                this.tensors = [];
                this.nets = [];
                if (properties)
                    for (var keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                        if (properties[keys[i]] != null)
                            this[keys[i]] = properties[keys[i]];
            }
    
            Argument.prototype.name = "";
            Argument.prototype.f = 0;
            Argument.prototype.i = $util.Long ? $util.Long.fromBits(0,0,false) : 0;
            Argument.prototype.s = $util.newBuffer([]);
            Argument.prototype.t = null;
            Argument.prototype.n = null;
            Argument.prototype.floats = $util.emptyArray;
            Argument.prototype.ints = $util.emptyArray;
            Argument.prototype.strings = $util.emptyArray;
            Argument.prototype.tensors = $util.emptyArray;
            Argument.prototype.nets = $util.emptyArray;
    
            Argument.create = function create(properties) {
                return new Argument(properties);
            };
    
            Argument.decode = function decode(reader, length) {
                if (!(reader instanceof $Reader))
                    reader = $Reader.create(reader);
                var end = length === undefined ? reader.len : reader.pos + length, message = new $root.caffe2.Argument();
                while (reader.pos < end) {
                    var tag = reader.uint32();
                    switch (tag >>> 3) {
                    case 1:
                        message.name = reader.string();
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
                    case 10:
                        message.t = $root.caffe2.TensorProto.decode(reader, reader.uint32());
                        break;
                    case 8:
                        message.n = $root.caffe2.NetDef.decode(reader, reader.uint32());
                        break;
                    case 5:
                        if (!(message.floats && message.floats.length)) {
                            if (message.floats != -1) {
                                message.floats = [];
                                message.floatsCount = 0;
                            }
                        }
                        if (message.floatsCount < 1000000) {
                            if ((tag & 7) === 2) {
                                var end2 = reader.uint32() + reader.pos;
                                while (reader.pos < end2) {
                                    message.floats.push(reader.float());
                                    message.floatsCount++;
                                }
                            }
                            else {
                                message.floats.push(reader.float());
                                message.floatsCount++;
                            }
                        }
                        else {
                            message.floats = -1;
                            if ((tag & 7) === 2) {
                                var endx = reader.uint32() + reader.pos;
                                while (reader.pos < endx)
                                    reader.float();
                            }
                            else {
                                reader.float();
                            }
                        }
                        break;
                    case 6:
                        if (!(message.ints && message.ints.length))
                            message.ints = [];
                        if ((tag & 7) === 2) {
                            var end2 = reader.uint32() + reader.pos;
                            while (reader.pos < end2)
                                message.ints.push(reader.int64());
                        } else
                            message.ints.push(reader.int64());
                        break;
                    case 7:
                        if (!(message.strings && message.strings.length))
                            message.strings = [];
                        message.strings.push(reader.bytes());
                        break;
                    case 11:
                        if (!(message.tensors && message.tensors.length))
                            message.tensors = [];
                        message.tensors.push($root.caffe2.TensorProto.decode(reader, reader.uint32()));
                        break;
                    case 9:
                        if (!(message.nets && message.nets.length))
                            message.nets = [];
                        message.nets.push($root.caffe2.NetDef.decode(reader, reader.uint32()));
                        break;
                    default:
                        reader.skipType(tag & 7);
                        break;
                    }
                }
                return message;
            };
    
            Argument.decodeText = function decodeText(reader, block) {
                if (!(reader instanceof $TextReader))
                    reader = $TextReader.create(reader);
                var message = new $root.caffe2.Argument();
                reader.start(block);
                while (!reader.end(block)) {
                    var tag = reader.tag();
                    switch (tag) {
                    case "name":
                        message.name = reader.string();
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
                        message.t = $root.caffe2.TensorProto.decodeText(reader, true);
                        break;
                    case "n":
                        message.n = $root.caffe2.NetDef.decodeText(reader, true);
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
                        message.tensors.push($root.caffe2.TensorProto.decodeText(reader, true));
                        break;
                    case "nets":
                        if (!(message.nets && message.nets.length))
                            message.nets = [];
                        message.nets.push($root.caffe2.NetDef.decodeText(reader, true));
                        break;
                    default:
                        reader.handle(tag, message);
                        break;
                    }
                }
                return message;
            };
    
            Argument.verify = function verify(message) {
                if (typeof message !== "object" || message === null)
                    return "object expected";
                if (message.name != null && message.hasOwnProperty("name"))
                    if (!$util.isString(message.name))
                        return "name: string expected";
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
                    var error = $root.caffe2.TensorProto.verify(message.t);
                    if (error)
                        return "t." + error;
                }
                if (message.n != null && message.hasOwnProperty("n")) {
                    var error = $root.caffe2.NetDef.verify(message.n);
                    if (error)
                        return "n." + error;
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
                        var error = $root.caffe2.TensorProto.verify(message.tensors[i]);
                        if (error)
                            return "tensors." + error;
                    }
                }
                if (message.nets != null && message.hasOwnProperty("nets")) {
                    if (!Array.isArray(message.nets))
                        return "nets: array expected";
                    for (var i = 0; i < message.nets.length; ++i) {
                        var error = $root.caffe2.NetDef.verify(message.nets[i]);
                        if (error)
                            return "nets." + error;
                    }
                }
                return null;
            };
    
            Argument.fromObject = function fromObject(object) {
                if (object instanceof $root.caffe2.Argument)
                    return object;
                var message = new $root.caffe2.Argument();
                if (object.name != null)
                    message.name = String(object.name);
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
                        throw TypeError(".caffe2.Argument.t: object expected");
                    message.t = $root.caffe2.TensorProto.fromObject(object.t);
                }
                if (object.n != null) {
                    if (typeof object.n !== "object")
                        throw TypeError(".caffe2.Argument.n: object expected");
                    message.n = $root.caffe2.NetDef.fromObject(object.n);
                }
                if (object.floats) {
                    if (!Array.isArray(object.floats))
                        throw TypeError(".caffe2.Argument.floats: array expected");
                    message.floats = [];
                    for (var i = 0; i < object.floats.length; ++i)
                        message.floats[i] = Number(object.floats[i]);
                }
                if (object.ints) {
                    if (!Array.isArray(object.ints))
                        throw TypeError(".caffe2.Argument.ints: array expected");
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
                        throw TypeError(".caffe2.Argument.strings: array expected");
                    message.strings = [];
                    for (var i = 0; i < object.strings.length; ++i)
                        if (typeof object.strings[i] === "string")
                            $util.base64.decode(object.strings[i], message.strings[i] = $util.newBuffer($util.base64.length(object.strings[i])), 0);
                        else if (object.strings[i].length)
                            message.strings[i] = object.strings[i];
                }
                if (object.tensors) {
                    if (!Array.isArray(object.tensors))
                        throw TypeError(".caffe2.Argument.tensors: array expected");
                    message.tensors = [];
                    for (var i = 0; i < object.tensors.length; ++i) {
                        if (typeof object.tensors[i] !== "object")
                            throw TypeError(".caffe2.Argument.tensors: object expected");
                        message.tensors[i] = $root.caffe2.TensorProto.fromObject(object.tensors[i]);
                    }
                }
                if (object.nets) {
                    if (!Array.isArray(object.nets))
                        throw TypeError(".caffe2.Argument.nets: array expected");
                    message.nets = [];
                    for (var i = 0; i < object.nets.length; ++i) {
                        if (typeof object.nets[i] !== "object")
                            throw TypeError(".caffe2.Argument.nets: object expected");
                        message.nets[i] = $root.caffe2.NetDef.fromObject(object.nets[i]);
                    }
                }
                return message;
            };
    
            Argument.toObject = function toObject(message, options) {
                if (!options)
                    options = {};
                var object = {};
                if (options.arrays || options.defaults) {
                    object.floats = [];
                    object.ints = [];
                    object.strings = [];
                    object.nets = [];
                    object.tensors = [];
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
                    object.n = null;
                    object.t = null;
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
                if (message.n != null && message.hasOwnProperty("n"))
                    object.n = $root.caffe2.NetDef.toObject(message.n, options);
                if (message.nets && message.nets.length) {
                    object.nets = [];
                    for (var j = 0; j < message.nets.length; ++j)
                        object.nets[j] = $root.caffe2.NetDef.toObject(message.nets[j], options);
                }
                if (message.t != null && message.hasOwnProperty("t"))
                    object.t = $root.caffe2.TensorProto.toObject(message.t, options);
                if (message.tensors && message.tensors.length) {
                    object.tensors = [];
                    for (var j = 0; j < message.tensors.length; ++j)
                        object.tensors[j] = $root.caffe2.TensorProto.toObject(message.tensors[j], options);
                }
                return object;
            };
    
            Argument.prototype.toJSON = function toJSON() {
                return this.constructor.toObject(this, $protobuf.util.toJSONOptions);
            };
    
            return Argument;
        })();
    
        caffe2.DeviceTypeProto = (function() {
            var valuesById = {}, values = Object.create(valuesById);
            values[valuesById[0] = "PROTO_CPU"] = 0;
            values[valuesById[1] = "PROTO_CUDA"] = 1;
            values[valuesById[2] = "PROTO_MKLDNN"] = 2;
            values[valuesById[3] = "PROTO_OPENGL"] = 3;
            values[valuesById[4] = "PROTO_OPENCL"] = 4;
            values[valuesById[5] = "PROTO_IDEEP"] = 5;
            values[valuesById[6] = "PROTO_HIP"] = 6;
            values[valuesById[7] = "PROTO_FPGA"] = 7;
            values[valuesById[8] = "PROTO_COMPILE_TIME_MAX_DEVICE_TYPES"] = 8;
            values[valuesById[20901] = "PROTO_ONLY_FOR_TEST"] = 20901;
            return values;
        })();
    
        caffe2.DeviceOption = (function() {
    
            function DeviceOption(properties) {
                this.extra_info = [];
                if (properties)
                    for (var keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                        if (properties[keys[i]] != null)
                            this[keys[i]] = properties[keys[i]];
            }
    
            DeviceOption.prototype.device_type = 0;
            DeviceOption.prototype.device_id = 0;
            DeviceOption.prototype.random_seed = 0;
            DeviceOption.prototype.node_name = "";
            DeviceOption.prototype.numa_node_id = 0;
            DeviceOption.prototype.extra_info = $util.emptyArray;
    
            DeviceOption.create = function create(properties) {
                return new DeviceOption(properties);
            };
    
            DeviceOption.decode = function decode(reader, length) {
                if (!(reader instanceof $Reader))
                    reader = $Reader.create(reader);
                var end = length === undefined ? reader.len : reader.pos + length, message = new $root.caffe2.DeviceOption();
                while (reader.pos < end) {
                    var tag = reader.uint32();
                    switch (tag >>> 3) {
                    case 1:
                        message.device_type = reader.int32();
                        break;
                    case 2:
                        message.device_id = reader.int32();
                        break;
                    case 3:
                        message.random_seed = reader.uint32();
                        break;
                    case 4:
                        message.node_name = reader.string();
                        break;
                    case 5:
                        message.numa_node_id = reader.int32();
                        break;
                    case 6:
                        if (!(message.extra_info && message.extra_info.length))
                            message.extra_info = [];
                        message.extra_info.push(reader.string());
                        break;
                    default:
                        reader.skipType(tag & 7);
                        break;
                    }
                }
                return message;
            };
    
            DeviceOption.decodeText = function decodeText(reader, block) {
                if (!(reader instanceof $TextReader))
                    reader = $TextReader.create(reader);
                var message = new $root.caffe2.DeviceOption();
                reader.start(block);
                while (!reader.end(block)) {
                    var tag = reader.tag();
                    switch (tag) {
                    case "device_type":
                        message.device_type = reader.int32();
                        break;
                    case "device_id":
                        message.device_id = reader.int32();
                        break;
                    case "random_seed":
                        message.random_seed = reader.uint32();
                        break;
                    case "node_name":
                        message.node_name = reader.string();
                        break;
                    case "numa_node_id":
                        message.numa_node_id = reader.int32();
                        break;
                    case "extra_info":
                        if (!(message.extra_info && message.extra_info.length))
                            message.extra_info = [];
                        message.extra_info.push(reader.string());
                        break;
                    default:
                        reader.handle(tag, message);
                        break;
                    }
                }
                return message;
            };
    
            DeviceOption.verify = function verify(message) {
                if (typeof message !== "object" || message === null)
                    return "object expected";
                if (message.device_type != null && message.hasOwnProperty("device_type"))
                    if (!$util.isInteger(message.device_type))
                        return "device_type: integer expected";
                if (message.device_id != null && message.hasOwnProperty("device_id"))
                    if (!$util.isInteger(message.device_id))
                        return "device_id: integer expected";
                if (message.random_seed != null && message.hasOwnProperty("random_seed"))
                    if (!$util.isInteger(message.random_seed))
                        return "random_seed: integer expected";
                if (message.node_name != null && message.hasOwnProperty("node_name"))
                    if (!$util.isString(message.node_name))
                        return "node_name: string expected";
                if (message.numa_node_id != null && message.hasOwnProperty("numa_node_id"))
                    if (!$util.isInteger(message.numa_node_id))
                        return "numa_node_id: integer expected";
                if (message.extra_info != null && message.hasOwnProperty("extra_info")) {
                    if (!Array.isArray(message.extra_info))
                        return "extra_info: array expected";
                    for (var i = 0; i < message.extra_info.length; ++i)
                        if (!$util.isString(message.extra_info[i]))
                            return "extra_info: string[] expected";
                }
                return null;
            };
    
            DeviceOption.fromObject = function fromObject(object) {
                if (object instanceof $root.caffe2.DeviceOption)
                    return object;
                var message = new $root.caffe2.DeviceOption();
                if (object.device_type != null)
                    message.device_type = object.device_type | 0;
                if (object.device_id != null)
                    message.device_id = object.device_id | 0;
                if (object.random_seed != null)
                    message.random_seed = object.random_seed >>> 0;
                if (object.node_name != null)
                    message.node_name = String(object.node_name);
                if (object.numa_node_id != null)
                    message.numa_node_id = object.numa_node_id | 0;
                if (object.extra_info) {
                    if (!Array.isArray(object.extra_info))
                        throw TypeError(".caffe2.DeviceOption.extra_info: array expected");
                    message.extra_info = [];
                    for (var i = 0; i < object.extra_info.length; ++i)
                        message.extra_info[i] = String(object.extra_info[i]);
                }
                return message;
            };
    
            DeviceOption.toObject = function toObject(message, options) {
                if (!options)
                    options = {};
                var object = {};
                if (options.arrays || options.defaults)
                    object.extra_info = [];
                if (options.defaults) {
                    object.device_type = 0;
                    object.device_id = 0;
                    object.random_seed = 0;
                    object.node_name = "";
                    object.numa_node_id = 0;
                }
                if (message.device_type != null && message.hasOwnProperty("device_type"))
                    object.device_type = message.device_type;
                if (message.device_id != null && message.hasOwnProperty("device_id"))
                    object.device_id = message.device_id;
                if (message.random_seed != null && message.hasOwnProperty("random_seed"))
                    object.random_seed = message.random_seed;
                if (message.node_name != null && message.hasOwnProperty("node_name"))
                    object.node_name = message.node_name;
                if (message.numa_node_id != null && message.hasOwnProperty("numa_node_id"))
                    object.numa_node_id = message.numa_node_id;
                if (message.extra_info && message.extra_info.length) {
                    object.extra_info = [];
                    for (var j = 0; j < message.extra_info.length; ++j)
                        object.extra_info[j] = message.extra_info[j];
                }
                return object;
            };
    
            DeviceOption.prototype.toJSON = function toJSON() {
                return this.constructor.toObject(this, $protobuf.util.toJSONOptions);
            };
    
            return DeviceOption;
        })();
    
        caffe2.OperatorDef = (function() {
    
            function OperatorDef(properties) {
                this.input = [];
                this.output = [];
                this.arg = [];
                this.control_input = [];
                if (properties)
                    for (var keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                        if (properties[keys[i]] != null)
                            this[keys[i]] = properties[keys[i]];
            }
    
            OperatorDef.prototype.input = $util.emptyArray;
            OperatorDef.prototype.output = $util.emptyArray;
            OperatorDef.prototype.name = "";
            OperatorDef.prototype.type = "";
            OperatorDef.prototype.arg = $util.emptyArray;
            OperatorDef.prototype.device_option = null;
            OperatorDef.prototype.engine = "";
            OperatorDef.prototype.control_input = $util.emptyArray;
            OperatorDef.prototype.is_gradient_op = false;
            OperatorDef.prototype.debug_info = "";
            OperatorDef.prototype.domain = "";
            OperatorDef.prototype.op_version = $util.Long ? $util.Long.fromBits(0,0,false) : 0;
    
            OperatorDef.create = function create(properties) {
                return new OperatorDef(properties);
            };
    
            OperatorDef.decode = function decode(reader, length) {
                if (!(reader instanceof $Reader))
                    reader = $Reader.create(reader);
                var end = length === undefined ? reader.len : reader.pos + length, message = new $root.caffe2.OperatorDef();
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
                        message.type = reader.string();
                        break;
                    case 5:
                        if (!(message.arg && message.arg.length))
                            message.arg = [];
                        message.arg.push($root.caffe2.Argument.decode(reader, reader.uint32()));
                        break;
                    case 6:
                        message.device_option = $root.caffe2.DeviceOption.decode(reader, reader.uint32());
                        break;
                    case 7:
                        message.engine = reader.string();
                        break;
                    case 8:
                        if (!(message.control_input && message.control_input.length))
                            message.control_input = [];
                        message.control_input.push(reader.string());
                        break;
                    case 9:
                        message.is_gradient_op = reader.bool();
                        break;
                    case 10:
                        message.debug_info = reader.string();
                        break;
                    case 11:
                        message.domain = reader.string();
                        break;
                    case 12:
                        message.op_version = reader.int64();
                        break;
                    default:
                        reader.skipType(tag & 7);
                        break;
                    }
                }
                return message;
            };
    
            OperatorDef.decodeText = function decodeText(reader, block) {
                if (!(reader instanceof $TextReader))
                    reader = $TextReader.create(reader);
                var message = new $root.caffe2.OperatorDef();
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
                    case "type":
                        message.type = reader.string();
                        break;
                    case "arg":
                        if (!(message.arg && message.arg.length))
                            message.arg = [];
                        message.arg.push($root.caffe2.Argument.decodeText(reader, true));
                        break;
                    case "device_option":
                        message.device_option = $root.caffe2.DeviceOption.decodeText(reader, true);
                        break;
                    case "engine":
                        message.engine = reader.string();
                        break;
                    case "control_input":
                        if (!(message.control_input && message.control_input.length))
                            message.control_input = [];
                        message.control_input.push(reader.string());
                        break;
                    case "is_gradient_op":
                        message.is_gradient_op = reader.bool();
                        break;
                    case "debug_info":
                        message.debug_info = reader.string();
                        break;
                    case "domain":
                        message.domain = reader.string();
                        break;
                    case "op_version":
                        message.op_version = reader.int64();
                        break;
                    default:
                        reader.handle(tag, message);
                        break;
                    }
                }
                return message;
            };
    
            OperatorDef.verify = function verify(message) {
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
                if (message.type != null && message.hasOwnProperty("type"))
                    if (!$util.isString(message.type))
                        return "type: string expected";
                if (message.arg != null && message.hasOwnProperty("arg")) {
                    if (!Array.isArray(message.arg))
                        return "arg: array expected";
                    for (var i = 0; i < message.arg.length; ++i) {
                        var error = $root.caffe2.Argument.verify(message.arg[i]);
                        if (error)
                            return "arg." + error;
                    }
                }
                if (message.device_option != null && message.hasOwnProperty("device_option")) {
                    var error = $root.caffe2.DeviceOption.verify(message.device_option);
                    if (error)
                        return "device_option." + error;
                }
                if (message.engine != null && message.hasOwnProperty("engine"))
                    if (!$util.isString(message.engine))
                        return "engine: string expected";
                if (message.control_input != null && message.hasOwnProperty("control_input")) {
                    if (!Array.isArray(message.control_input))
                        return "control_input: array expected";
                    for (var i = 0; i < message.control_input.length; ++i)
                        if (!$util.isString(message.control_input[i]))
                            return "control_input: string[] expected";
                }
                if (message.is_gradient_op != null && message.hasOwnProperty("is_gradient_op"))
                    if (typeof message.is_gradient_op !== "boolean")
                        return "is_gradient_op: boolean expected";
                if (message.debug_info != null && message.hasOwnProperty("debug_info"))
                    if (!$util.isString(message.debug_info))
                        return "debug_info: string expected";
                if (message.domain != null && message.hasOwnProperty("domain"))
                    if (!$util.isString(message.domain))
                        return "domain: string expected";
                if (message.op_version != null && message.hasOwnProperty("op_version"))
                    if (!$util.isInteger(message.op_version) && !(message.op_version && $util.isInteger(message.op_version.low) && $util.isInteger(message.op_version.high)))
                        return "op_version: integer|Long expected";
                return null;
            };
    
            OperatorDef.fromObject = function fromObject(object) {
                if (object instanceof $root.caffe2.OperatorDef)
                    return object;
                var message = new $root.caffe2.OperatorDef();
                if (object.input) {
                    if (!Array.isArray(object.input))
                        throw TypeError(".caffe2.OperatorDef.input: array expected");
                    message.input = [];
                    for (var i = 0; i < object.input.length; ++i)
                        message.input[i] = String(object.input[i]);
                }
                if (object.output) {
                    if (!Array.isArray(object.output))
                        throw TypeError(".caffe2.OperatorDef.output: array expected");
                    message.output = [];
                    for (var i = 0; i < object.output.length; ++i)
                        message.output[i] = String(object.output[i]);
                }
                if (object.name != null)
                    message.name = String(object.name);
                if (object.type != null)
                    message.type = String(object.type);
                if (object.arg) {
                    if (!Array.isArray(object.arg))
                        throw TypeError(".caffe2.OperatorDef.arg: array expected");
                    message.arg = [];
                    for (var i = 0; i < object.arg.length; ++i) {
                        if (typeof object.arg[i] !== "object")
                            throw TypeError(".caffe2.OperatorDef.arg: object expected");
                        message.arg[i] = $root.caffe2.Argument.fromObject(object.arg[i]);
                    }
                }
                if (object.device_option != null) {
                    if (typeof object.device_option !== "object")
                        throw TypeError(".caffe2.OperatorDef.device_option: object expected");
                    message.device_option = $root.caffe2.DeviceOption.fromObject(object.device_option);
                }
                if (object.engine != null)
                    message.engine = String(object.engine);
                if (object.control_input) {
                    if (!Array.isArray(object.control_input))
                        throw TypeError(".caffe2.OperatorDef.control_input: array expected");
                    message.control_input = [];
                    for (var i = 0; i < object.control_input.length; ++i)
                        message.control_input[i] = String(object.control_input[i]);
                }
                if (object.is_gradient_op != null)
                    message.is_gradient_op = Boolean(object.is_gradient_op);
                if (object.debug_info != null)
                    message.debug_info = String(object.debug_info);
                if (object.domain != null)
                    message.domain = String(object.domain);
                if (object.op_version != null)
                    if ($util.Long)
                        (message.op_version = $util.Long.fromValue(object.op_version)).unsigned = false;
                    else if (typeof object.op_version === "string")
                        message.op_version = parseInt(object.op_version, 10);
                    else if (typeof object.op_version === "number")
                        message.op_version = object.op_version;
                    else if (typeof object.op_version === "object")
                        message.op_version = new $util.LongBits(object.op_version.low >>> 0, object.op_version.high >>> 0).toNumber();
                return message;
            };
    
            OperatorDef.toObject = function toObject(message, options) {
                if (!options)
                    options = {};
                var object = {};
                if (options.arrays || options.defaults) {
                    object.input = [];
                    object.output = [];
                    object.arg = [];
                    object.control_input = [];
                }
                if (options.defaults) {
                    object.name = "";
                    object.type = "";
                    object.device_option = null;
                    object.engine = "";
                    object.is_gradient_op = false;
                    object.debug_info = "";
                    object.domain = "";
                    if ($util.Long) {
                        var long = new $util.Long(0, 0, false);
                        object.op_version = options.longs === String ? long.toString() : options.longs === Number ? long.toNumber() : long;
                    } else
                        object.op_version = options.longs === String ? "0" : 0;
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
                if (message.type != null && message.hasOwnProperty("type"))
                    object.type = message.type;
                if (message.arg && message.arg.length) {
                    object.arg = [];
                    for (var j = 0; j < message.arg.length; ++j)
                        object.arg[j] = $root.caffe2.Argument.toObject(message.arg[j], options);
                }
                if (message.device_option != null && message.hasOwnProperty("device_option"))
                    object.device_option = $root.caffe2.DeviceOption.toObject(message.device_option, options);
                if (message.engine != null && message.hasOwnProperty("engine"))
                    object.engine = message.engine;
                if (message.control_input && message.control_input.length) {
                    object.control_input = [];
                    for (var j = 0; j < message.control_input.length; ++j)
                        object.control_input[j] = message.control_input[j];
                }
                if (message.is_gradient_op != null && message.hasOwnProperty("is_gradient_op"))
                    object.is_gradient_op = message.is_gradient_op;
                if (message.debug_info != null && message.hasOwnProperty("debug_info"))
                    object.debug_info = message.debug_info;
                if (message.domain != null && message.hasOwnProperty("domain"))
                    object.domain = message.domain;
                if (message.op_version != null && message.hasOwnProperty("op_version"))
                    if (typeof message.op_version === "number")
                        object.op_version = options.longs === String ? String(message.op_version) : message.op_version;
                    else
                        object.op_version = options.longs === String ? $util.Long.prototype.toString.call(message.op_version) : options.longs === Number ? new $util.LongBits(message.op_version.low >>> 0, message.op_version.high >>> 0).toNumber() : message.op_version;
                return object;
            };
    
            OperatorDef.prototype.toJSON = function toJSON() {
                return this.constructor.toObject(this, $protobuf.util.toJSONOptions);
            };
    
            return OperatorDef;
        })();
    
        caffe2.NetDef = (function() {
    
            function NetDef(properties) {
                this.op = [];
                this.arg = [];
                this.external_input = [];
                this.external_output = [];
                if (properties)
                    for (var keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                        if (properties[keys[i]] != null)
                            this[keys[i]] = properties[keys[i]];
            }
    
            NetDef.prototype.name = "";
            NetDef.prototype.op = $util.emptyArray;
            NetDef.prototype.type = "";
            NetDef.prototype.num_workers = 0;
            NetDef.prototype.device_option = null;
            NetDef.prototype.arg = $util.emptyArray;
            NetDef.prototype.external_input = $util.emptyArray;
            NetDef.prototype.external_output = $util.emptyArray;
    
            NetDef.create = function create(properties) {
                return new NetDef(properties);
            };
    
            NetDef.decode = function decode(reader, length) {
                if (!(reader instanceof $Reader))
                    reader = $Reader.create(reader);
                var end = length === undefined ? reader.len : reader.pos + length, message = new $root.caffe2.NetDef();
                while (reader.pos < end) {
                    var tag = reader.uint32();
                    switch (tag >>> 3) {
                    case 1:
                        message.name = reader.string();
                        break;
                    case 2:
                        if (!(message.op && message.op.length))
                            message.op = [];
                        message.op.push($root.caffe2.OperatorDef.decode(reader, reader.uint32()));
                        break;
                    case 3:
                        message.type = reader.string();
                        break;
                    case 4:
                        message.num_workers = reader.int32();
                        break;
                    case 5:
                        message.device_option = $root.caffe2.DeviceOption.decode(reader, reader.uint32());
                        break;
                    case 6:
                        if (!(message.arg && message.arg.length))
                            message.arg = [];
                        message.arg.push($root.caffe2.Argument.decode(reader, reader.uint32()));
                        break;
                    case 7:
                        if (!(message.external_input && message.external_input.length))
                            message.external_input = [];
                        message.external_input.push(reader.string());
                        break;
                    case 8:
                        if (!(message.external_output && message.external_output.length))
                            message.external_output = [];
                        message.external_output.push(reader.string());
                        break;
                    default:
                        reader.skipType(tag & 7);
                        break;
                    }
                }
                return message;
            };
    
            NetDef.decodeText = function decodeText(reader, block) {
                if (!(reader instanceof $TextReader))
                    reader = $TextReader.create(reader);
                var message = new $root.caffe2.NetDef();
                reader.start(block);
                while (!reader.end(block)) {
                    var tag = reader.tag();
                    switch (tag) {
                    case "name":
                        message.name = reader.string();
                        break;
                    case "op":
                        if (!(message.op && message.op.length))
                            message.op = [];
                        message.op.push($root.caffe2.OperatorDef.decodeText(reader, true));
                        break;
                    case "type":
                        message.type = reader.string();
                        break;
                    case "num_workers":
                        message.num_workers = reader.int32();
                        break;
                    case "device_option":
                        message.device_option = $root.caffe2.DeviceOption.decodeText(reader, true);
                        break;
                    case "arg":
                        if (!(message.arg && message.arg.length))
                            message.arg = [];
                        message.arg.push($root.caffe2.Argument.decodeText(reader, true));
                        break;
                    case "external_input":
                        if (!(message.external_input && message.external_input.length))
                            message.external_input = [];
                        message.external_input.push(reader.string());
                        break;
                    case "external_output":
                        if (!(message.external_output && message.external_output.length))
                            message.external_output = [];
                        message.external_output.push(reader.string());
                        break;
                    default:
                        reader.handle(tag, message);
                        break;
                    }
                }
                return message;
            };
    
            NetDef.verify = function verify(message) {
                if (typeof message !== "object" || message === null)
                    return "object expected";
                if (message.name != null && message.hasOwnProperty("name"))
                    if (!$util.isString(message.name))
                        return "name: string expected";
                if (message.op != null && message.hasOwnProperty("op")) {
                    if (!Array.isArray(message.op))
                        return "op: array expected";
                    for (var i = 0; i < message.op.length; ++i) {
                        var error = $root.caffe2.OperatorDef.verify(message.op[i]);
                        if (error)
                            return "op." + error;
                    }
                }
                if (message.type != null && message.hasOwnProperty("type"))
                    if (!$util.isString(message.type))
                        return "type: string expected";
                if (message.num_workers != null && message.hasOwnProperty("num_workers"))
                    if (!$util.isInteger(message.num_workers))
                        return "num_workers: integer expected";
                if (message.device_option != null && message.hasOwnProperty("device_option")) {
                    var error = $root.caffe2.DeviceOption.verify(message.device_option);
                    if (error)
                        return "device_option." + error;
                }
                if (message.arg != null && message.hasOwnProperty("arg")) {
                    if (!Array.isArray(message.arg))
                        return "arg: array expected";
                    for (var i = 0; i < message.arg.length; ++i) {
                        var error = $root.caffe2.Argument.verify(message.arg[i]);
                        if (error)
                            return "arg." + error;
                    }
                }
                if (message.external_input != null && message.hasOwnProperty("external_input")) {
                    if (!Array.isArray(message.external_input))
                        return "external_input: array expected";
                    for (var i = 0; i < message.external_input.length; ++i)
                        if (!$util.isString(message.external_input[i]))
                            return "external_input: string[] expected";
                }
                if (message.external_output != null && message.hasOwnProperty("external_output")) {
                    if (!Array.isArray(message.external_output))
                        return "external_output: array expected";
                    for (var i = 0; i < message.external_output.length; ++i)
                        if (!$util.isString(message.external_output[i]))
                            return "external_output: string[] expected";
                }
                return null;
            };
    
            NetDef.fromObject = function fromObject(object) {
                if (object instanceof $root.caffe2.NetDef)
                    return object;
                var message = new $root.caffe2.NetDef();
                if (object.name != null)
                    message.name = String(object.name);
                if (object.op) {
                    if (!Array.isArray(object.op))
                        throw TypeError(".caffe2.NetDef.op: array expected");
                    message.op = [];
                    for (var i = 0; i < object.op.length; ++i) {
                        if (typeof object.op[i] !== "object")
                            throw TypeError(".caffe2.NetDef.op: object expected");
                        message.op[i] = $root.caffe2.OperatorDef.fromObject(object.op[i]);
                    }
                }
                if (object.type != null)
                    message.type = String(object.type);
                if (object.num_workers != null)
                    message.num_workers = object.num_workers | 0;
                if (object.device_option != null) {
                    if (typeof object.device_option !== "object")
                        throw TypeError(".caffe2.NetDef.device_option: object expected");
                    message.device_option = $root.caffe2.DeviceOption.fromObject(object.device_option);
                }
                if (object.arg) {
                    if (!Array.isArray(object.arg))
                        throw TypeError(".caffe2.NetDef.arg: array expected");
                    message.arg = [];
                    for (var i = 0; i < object.arg.length; ++i) {
                        if (typeof object.arg[i] !== "object")
                            throw TypeError(".caffe2.NetDef.arg: object expected");
                        message.arg[i] = $root.caffe2.Argument.fromObject(object.arg[i]);
                    }
                }
                if (object.external_input) {
                    if (!Array.isArray(object.external_input))
                        throw TypeError(".caffe2.NetDef.external_input: array expected");
                    message.external_input = [];
                    for (var i = 0; i < object.external_input.length; ++i)
                        message.external_input[i] = String(object.external_input[i]);
                }
                if (object.external_output) {
                    if (!Array.isArray(object.external_output))
                        throw TypeError(".caffe2.NetDef.external_output: array expected");
                    message.external_output = [];
                    for (var i = 0; i < object.external_output.length; ++i)
                        message.external_output[i] = String(object.external_output[i]);
                }
                return message;
            };
    
            NetDef.toObject = function toObject(message, options) {
                if (!options)
                    options = {};
                var object = {};
                if (options.arrays || options.defaults) {
                    object.op = [];
                    object.arg = [];
                    object.external_input = [];
                    object.external_output = [];
                }
                if (options.defaults) {
                    object.name = "";
                    object.type = "";
                    object.num_workers = 0;
                    object.device_option = null;
                }
                if (message.name != null && message.hasOwnProperty("name"))
                    object.name = message.name;
                if (message.op && message.op.length) {
                    object.op = [];
                    for (var j = 0; j < message.op.length; ++j)
                        object.op[j] = $root.caffe2.OperatorDef.toObject(message.op[j], options);
                }
                if (message.type != null && message.hasOwnProperty("type"))
                    object.type = message.type;
                if (message.num_workers != null && message.hasOwnProperty("num_workers"))
                    object.num_workers = message.num_workers;
                if (message.device_option != null && message.hasOwnProperty("device_option"))
                    object.device_option = $root.caffe2.DeviceOption.toObject(message.device_option, options);
                if (message.arg && message.arg.length) {
                    object.arg = [];
                    for (var j = 0; j < message.arg.length; ++j)
                        object.arg[j] = $root.caffe2.Argument.toObject(message.arg[j], options);
                }
                if (message.external_input && message.external_input.length) {
                    object.external_input = [];
                    for (var j = 0; j < message.external_input.length; ++j)
                        object.external_input[j] = message.external_input[j];
                }
                if (message.external_output && message.external_output.length) {
                    object.external_output = [];
                    for (var j = 0; j < message.external_output.length; ++j)
                        object.external_output[j] = message.external_output[j];
                }
                return object;
            };
    
            NetDef.prototype.toJSON = function toJSON() {
                return this.constructor.toObject(this, $protobuf.util.toJSONOptions);
            };
    
            return NetDef;
        })();
    
        caffe2.ExecutionStep = (function() {
    
            function ExecutionStep(properties) {
                this.substep = [];
                this.network = [];
                if (properties)
                    for (var keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                        if (properties[keys[i]] != null)
                            this[keys[i]] = properties[keys[i]];
            }
    
            ExecutionStep.prototype.name = "";
            ExecutionStep.prototype.substep = $util.emptyArray;
            ExecutionStep.prototype.network = $util.emptyArray;
            ExecutionStep.prototype.num_iter = $util.Long ? $util.Long.fromBits(0,0,false) : 0;
            ExecutionStep.prototype.criteria_network = "";
            ExecutionStep.prototype.report_net = "";
            ExecutionStep.prototype.report_interval = 0;
            ExecutionStep.prototype.run_every_ms = $util.Long ? $util.Long.fromBits(0,0,false) : 0;
            ExecutionStep.prototype.concurrent_substeps = false;
            ExecutionStep.prototype.should_stop_blob = "";
            ExecutionStep.prototype.only_once = false;
            ExecutionStep.prototype.create_workspace = false;
            ExecutionStep.prototype.num_concurrent_instances = 0;
    
            ExecutionStep.create = function create(properties) {
                return new ExecutionStep(properties);
            };
    
            ExecutionStep.decode = function decode(reader, length) {
                if (!(reader instanceof $Reader))
                    reader = $Reader.create(reader);
                var end = length === undefined ? reader.len : reader.pos + length, message = new $root.caffe2.ExecutionStep();
                while (reader.pos < end) {
                    var tag = reader.uint32();
                    switch (tag >>> 3) {
                    case 1:
                        message.name = reader.string();
                        break;
                    case 2:
                        if (!(message.substep && message.substep.length))
                            message.substep = [];
                        message.substep.push($root.caffe2.ExecutionStep.decode(reader, reader.uint32()));
                        break;
                    case 3:
                        if (!(message.network && message.network.length))
                            message.network = [];
                        message.network.push(reader.string());
                        break;
                    case 4:
                        message.num_iter = reader.int64();
                        break;
                    case 5:
                        message.criteria_network = reader.string();
                        break;
                    case 7:
                        message.report_net = reader.string();
                        break;
                    case 8:
                        message.report_interval = reader.int32();
                        break;
                    case 11:
                        message.run_every_ms = reader.int64();
                        break;
                    case 6:
                        message.concurrent_substeps = reader.bool();
                        break;
                    case 9:
                        message.should_stop_blob = reader.string();
                        break;
                    case 10:
                        message.only_once = reader.bool();
                        break;
                    case 12:
                        message.create_workspace = reader.bool();
                        break;
                    case 13:
                        message.num_concurrent_instances = reader.int32();
                        break;
                    default:
                        reader.skipType(tag & 7);
                        break;
                    }
                }
                return message;
            };
    
            ExecutionStep.decodeText = function decodeText(reader, block) {
                if (!(reader instanceof $TextReader))
                    reader = $TextReader.create(reader);
                var message = new $root.caffe2.ExecutionStep();
                reader.start(block);
                while (!reader.end(block)) {
                    var tag = reader.tag();
                    switch (tag) {
                    case "name":
                        message.name = reader.string();
                        break;
                    case "substep":
                        if (!(message.substep && message.substep.length))
                            message.substep = [];
                        message.substep.push($root.caffe2.ExecutionStep.decodeText(reader, true));
                        break;
                    case "network":
                        if (!(message.network && message.network.length))
                            message.network = [];
                        message.network.push(reader.string());
                        break;
                    case "num_iter":
                        message.num_iter = reader.int64();
                        break;
                    case "criteria_network":
                        message.criteria_network = reader.string();
                        break;
                    case "report_net":
                        message.report_net = reader.string();
                        break;
                    case "report_interval":
                        message.report_interval = reader.int32();
                        break;
                    case "run_every_ms":
                        message.run_every_ms = reader.int64();
                        break;
                    case "concurrent_substeps":
                        message.concurrent_substeps = reader.bool();
                        break;
                    case "should_stop_blob":
                        message.should_stop_blob = reader.string();
                        break;
                    case "only_once":
                        message.only_once = reader.bool();
                        break;
                    case "create_workspace":
                        message.create_workspace = reader.bool();
                        break;
                    case "num_concurrent_instances":
                        message.num_concurrent_instances = reader.int32();
                        break;
                    default:
                        reader.handle(tag, message);
                        break;
                    }
                }
                return message;
            };
    
            ExecutionStep.verify = function verify(message) {
                if (typeof message !== "object" || message === null)
                    return "object expected";
                if (message.name != null && message.hasOwnProperty("name"))
                    if (!$util.isString(message.name))
                        return "name: string expected";
                if (message.substep != null && message.hasOwnProperty("substep")) {
                    if (!Array.isArray(message.substep))
                        return "substep: array expected";
                    for (var i = 0; i < message.substep.length; ++i) {
                        var error = $root.caffe2.ExecutionStep.verify(message.substep[i]);
                        if (error)
                            return "substep." + error;
                    }
                }
                if (message.network != null && message.hasOwnProperty("network")) {
                    if (!Array.isArray(message.network))
                        return "network: array expected";
                    for (var i = 0; i < message.network.length; ++i)
                        if (!$util.isString(message.network[i]))
                            return "network: string[] expected";
                }
                if (message.num_iter != null && message.hasOwnProperty("num_iter"))
                    if (!$util.isInteger(message.num_iter) && !(message.num_iter && $util.isInteger(message.num_iter.low) && $util.isInteger(message.num_iter.high)))
                        return "num_iter: integer|Long expected";
                if (message.criteria_network != null && message.hasOwnProperty("criteria_network"))
                    if (!$util.isString(message.criteria_network))
                        return "criteria_network: string expected";
                if (message.report_net != null && message.hasOwnProperty("report_net"))
                    if (!$util.isString(message.report_net))
                        return "report_net: string expected";
                if (message.report_interval != null && message.hasOwnProperty("report_interval"))
                    if (!$util.isInteger(message.report_interval))
                        return "report_interval: integer expected";
                if (message.run_every_ms != null && message.hasOwnProperty("run_every_ms"))
                    if (!$util.isInteger(message.run_every_ms) && !(message.run_every_ms && $util.isInteger(message.run_every_ms.low) && $util.isInteger(message.run_every_ms.high)))
                        return "run_every_ms: integer|Long expected";
                if (message.concurrent_substeps != null && message.hasOwnProperty("concurrent_substeps"))
                    if (typeof message.concurrent_substeps !== "boolean")
                        return "concurrent_substeps: boolean expected";
                if (message.should_stop_blob != null && message.hasOwnProperty("should_stop_blob"))
                    if (!$util.isString(message.should_stop_blob))
                        return "should_stop_blob: string expected";
                if (message.only_once != null && message.hasOwnProperty("only_once"))
                    if (typeof message.only_once !== "boolean")
                        return "only_once: boolean expected";
                if (message.create_workspace != null && message.hasOwnProperty("create_workspace"))
                    if (typeof message.create_workspace !== "boolean")
                        return "create_workspace: boolean expected";
                if (message.num_concurrent_instances != null && message.hasOwnProperty("num_concurrent_instances"))
                    if (!$util.isInteger(message.num_concurrent_instances))
                        return "num_concurrent_instances: integer expected";
                return null;
            };
    
            ExecutionStep.fromObject = function fromObject(object) {
                if (object instanceof $root.caffe2.ExecutionStep)
                    return object;
                var message = new $root.caffe2.ExecutionStep();
                if (object.name != null)
                    message.name = String(object.name);
                if (object.substep) {
                    if (!Array.isArray(object.substep))
                        throw TypeError(".caffe2.ExecutionStep.substep: array expected");
                    message.substep = [];
                    for (var i = 0; i < object.substep.length; ++i) {
                        if (typeof object.substep[i] !== "object")
                            throw TypeError(".caffe2.ExecutionStep.substep: object expected");
                        message.substep[i] = $root.caffe2.ExecutionStep.fromObject(object.substep[i]);
                    }
                }
                if (object.network) {
                    if (!Array.isArray(object.network))
                        throw TypeError(".caffe2.ExecutionStep.network: array expected");
                    message.network = [];
                    for (var i = 0; i < object.network.length; ++i)
                        message.network[i] = String(object.network[i]);
                }
                if (object.num_iter != null)
                    if ($util.Long)
                        (message.num_iter = $util.Long.fromValue(object.num_iter)).unsigned = false;
                    else if (typeof object.num_iter === "string")
                        message.num_iter = parseInt(object.num_iter, 10);
                    else if (typeof object.num_iter === "number")
                        message.num_iter = object.num_iter;
                    else if (typeof object.num_iter === "object")
                        message.num_iter = new $util.LongBits(object.num_iter.low >>> 0, object.num_iter.high >>> 0).toNumber();
                if (object.criteria_network != null)
                    message.criteria_network = String(object.criteria_network);
                if (object.report_net != null)
                    message.report_net = String(object.report_net);
                if (object.report_interval != null)
                    message.report_interval = object.report_interval | 0;
                if (object.run_every_ms != null)
                    if ($util.Long)
                        (message.run_every_ms = $util.Long.fromValue(object.run_every_ms)).unsigned = false;
                    else if (typeof object.run_every_ms === "string")
                        message.run_every_ms = parseInt(object.run_every_ms, 10);
                    else if (typeof object.run_every_ms === "number")
                        message.run_every_ms = object.run_every_ms;
                    else if (typeof object.run_every_ms === "object")
                        message.run_every_ms = new $util.LongBits(object.run_every_ms.low >>> 0, object.run_every_ms.high >>> 0).toNumber();
                if (object.concurrent_substeps != null)
                    message.concurrent_substeps = Boolean(object.concurrent_substeps);
                if (object.should_stop_blob != null)
                    message.should_stop_blob = String(object.should_stop_blob);
                if (object.only_once != null)
                    message.only_once = Boolean(object.only_once);
                if (object.create_workspace != null)
                    message.create_workspace = Boolean(object.create_workspace);
                if (object.num_concurrent_instances != null)
                    message.num_concurrent_instances = object.num_concurrent_instances | 0;
                return message;
            };
    
            ExecutionStep.toObject = function toObject(message, options) {
                if (!options)
                    options = {};
                var object = {};
                if (options.arrays || options.defaults) {
                    object.substep = [];
                    object.network = [];
                }
                if (options.defaults) {
                    object.name = "";
                    if ($util.Long) {
                        var long = new $util.Long(0, 0, false);
                        object.num_iter = options.longs === String ? long.toString() : options.longs === Number ? long.toNumber() : long;
                    } else
                        object.num_iter = options.longs === String ? "0" : 0;
                    object.criteria_network = "";
                    object.concurrent_substeps = false;
                    object.report_net = "";
                    object.report_interval = 0;
                    object.should_stop_blob = "";
                    object.only_once = false;
                    if ($util.Long) {
                        var long = new $util.Long(0, 0, false);
                        object.run_every_ms = options.longs === String ? long.toString() : options.longs === Number ? long.toNumber() : long;
                    } else
                        object.run_every_ms = options.longs === String ? "0" : 0;
                    object.create_workspace = false;
                    object.num_concurrent_instances = 0;
                }
                if (message.name != null && message.hasOwnProperty("name"))
                    object.name = message.name;
                if (message.substep && message.substep.length) {
                    object.substep = [];
                    for (var j = 0; j < message.substep.length; ++j)
                        object.substep[j] = $root.caffe2.ExecutionStep.toObject(message.substep[j], options);
                }
                if (message.network && message.network.length) {
                    object.network = [];
                    for (var j = 0; j < message.network.length; ++j)
                        object.network[j] = message.network[j];
                }
                if (message.num_iter != null && message.hasOwnProperty("num_iter"))
                    if (typeof message.num_iter === "number")
                        object.num_iter = options.longs === String ? String(message.num_iter) : message.num_iter;
                    else
                        object.num_iter = options.longs === String ? $util.Long.prototype.toString.call(message.num_iter) : options.longs === Number ? new $util.LongBits(message.num_iter.low >>> 0, message.num_iter.high >>> 0).toNumber() : message.num_iter;
                if (message.criteria_network != null && message.hasOwnProperty("criteria_network"))
                    object.criteria_network = message.criteria_network;
                if (message.concurrent_substeps != null && message.hasOwnProperty("concurrent_substeps"))
                    object.concurrent_substeps = message.concurrent_substeps;
                if (message.report_net != null && message.hasOwnProperty("report_net"))
                    object.report_net = message.report_net;
                if (message.report_interval != null && message.hasOwnProperty("report_interval"))
                    object.report_interval = message.report_interval;
                if (message.should_stop_blob != null && message.hasOwnProperty("should_stop_blob"))
                    object.should_stop_blob = message.should_stop_blob;
                if (message.only_once != null && message.hasOwnProperty("only_once"))
                    object.only_once = message.only_once;
                if (message.run_every_ms != null && message.hasOwnProperty("run_every_ms"))
                    if (typeof message.run_every_ms === "number")
                        object.run_every_ms = options.longs === String ? String(message.run_every_ms) : message.run_every_ms;
                    else
                        object.run_every_ms = options.longs === String ? $util.Long.prototype.toString.call(message.run_every_ms) : options.longs === Number ? new $util.LongBits(message.run_every_ms.low >>> 0, message.run_every_ms.high >>> 0).toNumber() : message.run_every_ms;
                if (message.create_workspace != null && message.hasOwnProperty("create_workspace"))
                    object.create_workspace = message.create_workspace;
                if (message.num_concurrent_instances != null && message.hasOwnProperty("num_concurrent_instances"))
                    object.num_concurrent_instances = message.num_concurrent_instances;
                return object;
            };
    
            ExecutionStep.prototype.toJSON = function toJSON() {
                return this.constructor.toObject(this, $protobuf.util.toJSONOptions);
            };
    
            return ExecutionStep;
        })();
    
        caffe2.PlanDef = (function() {
    
            function PlanDef(properties) {
                this.network = [];
                this.execution_step = [];
                if (properties)
                    for (var keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                        if (properties[keys[i]] != null)
                            this[keys[i]] = properties[keys[i]];
            }
    
            PlanDef.prototype.name = "";
            PlanDef.prototype.network = $util.emptyArray;
            PlanDef.prototype.execution_step = $util.emptyArray;
    
            PlanDef.create = function create(properties) {
                return new PlanDef(properties);
            };
    
            PlanDef.decode = function decode(reader, length) {
                if (!(reader instanceof $Reader))
                    reader = $Reader.create(reader);
                var end = length === undefined ? reader.len : reader.pos + length, message = new $root.caffe2.PlanDef();
                while (reader.pos < end) {
                    var tag = reader.uint32();
                    switch (tag >>> 3) {
                    case 1:
                        message.name = reader.string();
                        break;
                    case 2:
                        if (!(message.network && message.network.length))
                            message.network = [];
                        message.network.push($root.caffe2.NetDef.decode(reader, reader.uint32()));
                        break;
                    case 3:
                        if (!(message.execution_step && message.execution_step.length))
                            message.execution_step = [];
                        message.execution_step.push($root.caffe2.ExecutionStep.decode(reader, reader.uint32()));
                        break;
                    default:
                        reader.skipType(tag & 7);
                        break;
                    }
                }
                return message;
            };
    
            PlanDef.decodeText = function decodeText(reader, block) {
                if (!(reader instanceof $TextReader))
                    reader = $TextReader.create(reader);
                var message = new $root.caffe2.PlanDef();
                reader.start(block);
                while (!reader.end(block)) {
                    var tag = reader.tag();
                    switch (tag) {
                    case "name":
                        message.name = reader.string();
                        break;
                    case "network":
                        if (!(message.network && message.network.length))
                            message.network = [];
                        message.network.push($root.caffe2.NetDef.decodeText(reader, true));
                        break;
                    case "execution_step":
                        if (!(message.execution_step && message.execution_step.length))
                            message.execution_step = [];
                        message.execution_step.push($root.caffe2.ExecutionStep.decodeText(reader, true));
                        break;
                    default:
                        reader.handle(tag, message);
                        break;
                    }
                }
                return message;
            };
    
            PlanDef.verify = function verify(message) {
                if (typeof message !== "object" || message === null)
                    return "object expected";
                if (message.name != null && message.hasOwnProperty("name"))
                    if (!$util.isString(message.name))
                        return "name: string expected";
                if (message.network != null && message.hasOwnProperty("network")) {
                    if (!Array.isArray(message.network))
                        return "network: array expected";
                    for (var i = 0; i < message.network.length; ++i) {
                        var error = $root.caffe2.NetDef.verify(message.network[i]);
                        if (error)
                            return "network." + error;
                    }
                }
                if (message.execution_step != null && message.hasOwnProperty("execution_step")) {
                    if (!Array.isArray(message.execution_step))
                        return "execution_step: array expected";
                    for (var i = 0; i < message.execution_step.length; ++i) {
                        var error = $root.caffe2.ExecutionStep.verify(message.execution_step[i]);
                        if (error)
                            return "execution_step." + error;
                    }
                }
                return null;
            };
    
            PlanDef.fromObject = function fromObject(object) {
                if (object instanceof $root.caffe2.PlanDef)
                    return object;
                var message = new $root.caffe2.PlanDef();
                if (object.name != null)
                    message.name = String(object.name);
                if (object.network) {
                    if (!Array.isArray(object.network))
                        throw TypeError(".caffe2.PlanDef.network: array expected");
                    message.network = [];
                    for (var i = 0; i < object.network.length; ++i) {
                        if (typeof object.network[i] !== "object")
                            throw TypeError(".caffe2.PlanDef.network: object expected");
                        message.network[i] = $root.caffe2.NetDef.fromObject(object.network[i]);
                    }
                }
                if (object.execution_step) {
                    if (!Array.isArray(object.execution_step))
                        throw TypeError(".caffe2.PlanDef.execution_step: array expected");
                    message.execution_step = [];
                    for (var i = 0; i < object.execution_step.length; ++i) {
                        if (typeof object.execution_step[i] !== "object")
                            throw TypeError(".caffe2.PlanDef.execution_step: object expected");
                        message.execution_step[i] = $root.caffe2.ExecutionStep.fromObject(object.execution_step[i]);
                    }
                }
                return message;
            };
    
            PlanDef.toObject = function toObject(message, options) {
                if (!options)
                    options = {};
                var object = {};
                if (options.arrays || options.defaults) {
                    object.network = [];
                    object.execution_step = [];
                }
                if (options.defaults)
                    object.name = "";
                if (message.name != null && message.hasOwnProperty("name"))
                    object.name = message.name;
                if (message.network && message.network.length) {
                    object.network = [];
                    for (var j = 0; j < message.network.length; ++j)
                        object.network[j] = $root.caffe2.NetDef.toObject(message.network[j], options);
                }
                if (message.execution_step && message.execution_step.length) {
                    object.execution_step = [];
                    for (var j = 0; j < message.execution_step.length; ++j)
                        object.execution_step[j] = $root.caffe2.ExecutionStep.toObject(message.execution_step[j], options);
                }
                return object;
            };
    
            PlanDef.prototype.toJSON = function toJSON() {
                return this.constructor.toObject(this, $protobuf.util.toJSONOptions);
            };
    
            return PlanDef;
        })();
    
        caffe2.BlobProto = (function() {
    
            function BlobProto(properties) {
                if (properties)
                    for (var keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                        if (properties[keys[i]] != null)
                            this[keys[i]] = properties[keys[i]];
            }
    
            BlobProto.prototype.name = "";
            BlobProto.prototype.type = "";
            BlobProto.prototype.tensor = null;
            BlobProto.prototype.content = $util.newBuffer([]);
            BlobProto.prototype.qtensor = null;
            BlobProto.prototype.content_num_chunks = 0;
            BlobProto.prototype.content_chunk_id = 0;
    
            BlobProto.create = function create(properties) {
                return new BlobProto(properties);
            };
    
            BlobProto.decode = function decode(reader, length) {
                if (!(reader instanceof $Reader))
                    reader = $Reader.create(reader);
                var end = length === undefined ? reader.len : reader.pos + length, message = new $root.caffe2.BlobProto();
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
                        message.tensor = $root.caffe2.TensorProto.decode(reader, reader.uint32());
                        break;
                    case 4:
                        message.content = reader.bytes();
                        break;
                    case 5:
                        message.qtensor = $root.caffe2.QTensorProto.decode(reader, reader.uint32());
                        break;
                    case 6:
                        message.content_num_chunks = reader.int32();
                        break;
                    case 7:
                        message.content_chunk_id = reader.int32();
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
                var message = new $root.caffe2.BlobProto();
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
                    case "tensor":
                        message.tensor = $root.caffe2.TensorProto.decodeText(reader, true);
                        break;
                    case "content":
                        message.content = reader.bytes();
                        break;
                    case "qtensor":
                        message.qtensor = $root.caffe2.QTensorProto.decodeText(reader, true);
                        break;
                    case "content_num_chunks":
                        message.content_num_chunks = reader.int32();
                        break;
                    case "content_chunk_id":
                        message.content_chunk_id = reader.int32();
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
                if (message.name != null && message.hasOwnProperty("name"))
                    if (!$util.isString(message.name))
                        return "name: string expected";
                if (message.type != null && message.hasOwnProperty("type"))
                    if (!$util.isString(message.type))
                        return "type: string expected";
                if (message.tensor != null && message.hasOwnProperty("tensor")) {
                    var error = $root.caffe2.TensorProto.verify(message.tensor);
                    if (error)
                        return "tensor." + error;
                }
                if (message.content != null && message.hasOwnProperty("content"))
                    if (!(message.content && typeof message.content.length === "number" || $util.isString(message.content)))
                        return "content: buffer expected";
                if (message.qtensor != null && message.hasOwnProperty("qtensor")) {
                    var error = $root.caffe2.QTensorProto.verify(message.qtensor);
                    if (error)
                        return "qtensor." + error;
                }
                if (message.content_num_chunks != null && message.hasOwnProperty("content_num_chunks"))
                    if (!$util.isInteger(message.content_num_chunks))
                        return "content_num_chunks: integer expected";
                if (message.content_chunk_id != null && message.hasOwnProperty("content_chunk_id"))
                    if (!$util.isInteger(message.content_chunk_id))
                        return "content_chunk_id: integer expected";
                return null;
            };
    
            BlobProto.fromObject = function fromObject(object) {
                if (object instanceof $root.caffe2.BlobProto)
                    return object;
                var message = new $root.caffe2.BlobProto();
                if (object.name != null)
                    message.name = String(object.name);
                if (object.type != null)
                    message.type = String(object.type);
                if (object.tensor != null) {
                    if (typeof object.tensor !== "object")
                        throw TypeError(".caffe2.BlobProto.tensor: object expected");
                    message.tensor = $root.caffe2.TensorProto.fromObject(object.tensor);
                }
                if (object.content != null)
                    if (typeof object.content === "string")
                        $util.base64.decode(object.content, message.content = $util.newBuffer($util.base64.length(object.content)), 0);
                    else if (object.content.length)
                        message.content = object.content;
                if (object.qtensor != null) {
                    if (typeof object.qtensor !== "object")
                        throw TypeError(".caffe2.BlobProto.qtensor: object expected");
                    message.qtensor = $root.caffe2.QTensorProto.fromObject(object.qtensor);
                }
                if (object.content_num_chunks != null)
                    message.content_num_chunks = object.content_num_chunks | 0;
                if (object.content_chunk_id != null)
                    message.content_chunk_id = object.content_chunk_id | 0;
                return message;
            };
    
            BlobProto.toObject = function toObject(message, options) {
                if (!options)
                    options = {};
                var object = {};
                if (options.defaults) {
                    object.name = "";
                    object.type = "";
                    object.tensor = null;
                    if (options.bytes === String)
                        object.content = "";
                    else {
                        object.content = [];
                        if (options.bytes !== Array)
                            object.content = $util.newBuffer(object.content);
                    }
                    object.qtensor = null;
                    object.content_num_chunks = 0;
                    object.content_chunk_id = 0;
                }
                if (message.name != null && message.hasOwnProperty("name"))
                    object.name = message.name;
                if (message.type != null && message.hasOwnProperty("type"))
                    object.type = message.type;
                if (message.tensor != null && message.hasOwnProperty("tensor"))
                    object.tensor = $root.caffe2.TensorProto.toObject(message.tensor, options);
                if (message.content != null && message.hasOwnProperty("content"))
                    object.content = options.bytes === String ? $util.base64.encode(message.content, 0, message.content.length) : options.bytes === Array ? Array.prototype.slice.call(message.content) : message.content;
                if (message.qtensor != null && message.hasOwnProperty("qtensor"))
                    object.qtensor = $root.caffe2.QTensorProto.toObject(message.qtensor, options);
                if (message.content_num_chunks != null && message.hasOwnProperty("content_num_chunks"))
                    object.content_num_chunks = message.content_num_chunks;
                if (message.content_chunk_id != null && message.hasOwnProperty("content_chunk_id"))
                    object.content_chunk_id = message.content_chunk_id;
                return object;
            };
    
            BlobProto.prototype.toJSON = function toJSON() {
                return this.constructor.toObject(this, $protobuf.util.toJSONOptions);
            };
    
            return BlobProto;
        })();
    
        caffe2.DBReaderProto = (function() {
    
            function DBReaderProto(properties) {
                if (properties)
                    for (var keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                        if (properties[keys[i]] != null)
                            this[keys[i]] = properties[keys[i]];
            }
    
            DBReaderProto.prototype.name = "";
            DBReaderProto.prototype.source = "";
            DBReaderProto.prototype.db_type = "";
            DBReaderProto.prototype.key = "";
    
            DBReaderProto.create = function create(properties) {
                return new DBReaderProto(properties);
            };
    
            DBReaderProto.decode = function decode(reader, length) {
                if (!(reader instanceof $Reader))
                    reader = $Reader.create(reader);
                var end = length === undefined ? reader.len : reader.pos + length, message = new $root.caffe2.DBReaderProto();
                while (reader.pos < end) {
                    var tag = reader.uint32();
                    switch (tag >>> 3) {
                    case 1:
                        message.name = reader.string();
                        break;
                    case 2:
                        message.source = reader.string();
                        break;
                    case 3:
                        message.db_type = reader.string();
                        break;
                    case 4:
                        message.key = reader.string();
                        break;
                    default:
                        reader.skipType(tag & 7);
                        break;
                    }
                }
                return message;
            };
    
            DBReaderProto.decodeText = function decodeText(reader, block) {
                if (!(reader instanceof $TextReader))
                    reader = $TextReader.create(reader);
                var message = new $root.caffe2.DBReaderProto();
                reader.start(block);
                while (!reader.end(block)) {
                    var tag = reader.tag();
                    switch (tag) {
                    case "name":
                        message.name = reader.string();
                        break;
                    case "source":
                        message.source = reader.string();
                        break;
                    case "db_type":
                        message.db_type = reader.string();
                        break;
                    case "key":
                        message.key = reader.string();
                        break;
                    default:
                        reader.handle(tag, message);
                        break;
                    }
                }
                return message;
            };
    
            DBReaderProto.verify = function verify(message) {
                if (typeof message !== "object" || message === null)
                    return "object expected";
                if (message.name != null && message.hasOwnProperty("name"))
                    if (!$util.isString(message.name))
                        return "name: string expected";
                if (message.source != null && message.hasOwnProperty("source"))
                    if (!$util.isString(message.source))
                        return "source: string expected";
                if (message.db_type != null && message.hasOwnProperty("db_type"))
                    if (!$util.isString(message.db_type))
                        return "db_type: string expected";
                if (message.key != null && message.hasOwnProperty("key"))
                    if (!$util.isString(message.key))
                        return "key: string expected";
                return null;
            };
    
            DBReaderProto.fromObject = function fromObject(object) {
                if (object instanceof $root.caffe2.DBReaderProto)
                    return object;
                var message = new $root.caffe2.DBReaderProto();
                if (object.name != null)
                    message.name = String(object.name);
                if (object.source != null)
                    message.source = String(object.source);
                if (object.db_type != null)
                    message.db_type = String(object.db_type);
                if (object.key != null)
                    message.key = String(object.key);
                return message;
            };
    
            DBReaderProto.toObject = function toObject(message, options) {
                if (!options)
                    options = {};
                var object = {};
                if (options.defaults) {
                    object.name = "";
                    object.source = "";
                    object.db_type = "";
                    object.key = "";
                }
                if (message.name != null && message.hasOwnProperty("name"))
                    object.name = message.name;
                if (message.source != null && message.hasOwnProperty("source"))
                    object.source = message.source;
                if (message.db_type != null && message.hasOwnProperty("db_type"))
                    object.db_type = message.db_type;
                if (message.key != null && message.hasOwnProperty("key"))
                    object.key = message.key;
                return object;
            };
    
            DBReaderProto.prototype.toJSON = function toJSON() {
                return this.constructor.toObject(this, $protobuf.util.toJSONOptions);
            };
    
            return DBReaderProto;
        })();
    
        return caffe2;
    })();

    return $root;
})(protobuf);
