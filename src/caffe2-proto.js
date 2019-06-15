/*eslint-disable block-scoped-var, id-length, no-control-regex, no-magic-numbers, no-prototype-builtins, no-redeclare, no-shadow, no-var, sort-vars*/
(function($protobuf) {
    "use strict";

    var $Reader = $protobuf.Reader, $util = $protobuf.util;
    
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
    
            ExternalDataProto.decodeText = function decodeText(reader) {
                var message = new $root.caffe2.ExternalDataProto();
                reader.start();
                while (!reader.end()) {
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
                        if (reader.first())
                            while (!reader.last()) {
                                message.strides.push(reader.int64());
                                reader.next();
                            }
                        else
                            message.strides.push(reader.int64());
                        break;
                    default:
                        reader.field(tag, message);
                        break;
                    }
                }
                return message;
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
    
            TensorProto.decodeText = function decodeText(reader) {
                var message = new $root.caffe2.TensorProto();
                reader.start();
                while (!reader.end()) {
                    var tag = reader.tag();
                    switch (tag) {
                    case "dims":
                        if (!(message.dims && message.dims.length))
                            message.dims = [];
                        if (reader.first())
                            while (!reader.last()) {
                                message.dims.push(reader.int64());
                                reader.next();
                            }
                        else
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
                        if (reader.first())
                            while (!reader.last()) {
                                message.float_data.push(reader.float());
                                reader.next();
                            }
                        else
                            message.float_data.push(reader.float());
                        break;
                    case "int32_data":
                        if (!(message.int32_data && message.int32_data.length))
                            message.int32_data = [];
                        if (reader.first())
                            while (!reader.last()) {
                                message.int32_data.push(reader.int32());
                                reader.next();
                            }
                        else
                            message.int32_data.push(reader.int32());
                        break;
                    case "byte_data":
                        message.byte_data = reader.bytes();
                        break;
                    case "string_data":
                        if (!(message.string_data && message.string_data.length))
                            message.string_data = [];
                        if (reader.first())
                            while (!reader.last()) {
                                message.string_data.push(reader.bytes());
                                reader.next();
                            }
                        else
                            message.string_data.push(reader.bytes());
                        break;
                    case "double_data":
                        if (!(message.double_data && message.double_data.length))
                            message.double_data = [];
                        if (reader.first())
                            while (!reader.last()) {
                                message.double_data.push(reader.double());
                                reader.next();
                            }
                        else
                            message.double_data.push(reader.double());
                        break;
                    case "int64_data":
                        if (!(message.int64_data && message.int64_data.length))
                            message.int64_data = [];
                        if (reader.first())
                            while (!reader.last()) {
                                message.int64_data.push(reader.int64());
                                reader.next();
                            }
                        else
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
                        reader.field(tag, message);
                        break;
                    }
                }
                return message;
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
    
                Segment.decodeText = function decodeText(reader) {
                    var message = new $root.caffe2.TensorProto.Segment();
                    reader.start();
                    while (!reader.end()) {
                        var tag = reader.tag();
                        switch (tag) {
                        case "begin":
                            message.begin = reader.int64();
                            break;
                        case "end":
                            message.end = reader.int64();
                            break;
                        default:
                            reader.field(tag, message);
                            break;
                        }
                    }
                    if (!message.hasOwnProperty("begin"))
                        throw $util.ProtocolError("missing required 'begin'", { instance: message });
                    if (!message.hasOwnProperty("end"))
                        throw $util.ProtocolError("missing required 'end'", { instance: message });
                    return message;
                };
    
                return Segment;
            })();
    
            return TensorProto;
        })();
    
        caffe2.QTensorProto = (function() {
    
            function QTensorProto(properties) {
                this.dims = [];
                this.data = [];
                this.scales = [];
                this.biases = [];
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
            QTensorProto.prototype.scales = $util.emptyArray;
            QTensorProto.prototype.biases = $util.emptyArray;
            QTensorProto.prototype.axis = 0;
            QTensorProto.prototype.is_multiparam = false;
    
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
                    case 9:
                        if (!(message.scales && message.scales.length))
                            message.scales = [];
                        if ((tag & 7) === 2) {
                            var end2 = reader.uint32() + reader.pos;
                            while (reader.pos < end2)
                                message.scales.push(reader.double());
                        } else
                            message.scales.push(reader.double());
                        break;
                    case 10:
                        if (!(message.biases && message.biases.length))
                            message.biases = [];
                        if ((tag & 7) === 2) {
                            var end2 = reader.uint32() + reader.pos;
                            while (reader.pos < end2)
                                message.biases.push(reader.double());
                        } else
                            message.biases.push(reader.double());
                        break;
                    case 11:
                        message.axis = reader.int32();
                        break;
                    case 12:
                        message.is_multiparam = reader.bool();
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
    
            QTensorProto.decodeText = function decodeText(reader) {
                var message = new $root.caffe2.QTensorProto();
                reader.start();
                while (!reader.end()) {
                    var tag = reader.tag();
                    switch (tag) {
                    case "dims":
                        if (!(message.dims && message.dims.length))
                            message.dims = [];
                        if (reader.first())
                            while (!reader.last()) {
                                message.dims.push(reader.int64());
                                reader.next();
                            }
                        else
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
                        if (reader.first())
                            while (!reader.last()) {
                                message.data.push(reader.int32());
                                reader.next();
                            }
                        else
                            message.data.push(reader.int32());
                        break;
                    case "name":
                        message.name = reader.string();
                        break;
                    case "data_type":
                        message.data_type = reader.enum($root.caffe2.TensorProto.DataType);
                        break;
                    case "scales":
                        if (!(message.scales && message.scales.length))
                            message.scales = [];
                        if (reader.first())
                            while (!reader.last()) {
                                message.scales.push(reader.double());
                                reader.next();
                            }
                        else
                            message.scales.push(reader.double());
                        break;
                    case "biases":
                        if (!(message.biases && message.biases.length))
                            message.biases = [];
                        if (reader.first())
                            while (!reader.last()) {
                                message.biases.push(reader.double());
                                reader.next();
                            }
                        else
                            message.biases.push(reader.double());
                        break;
                    case "axis":
                        message.axis = reader.int32();
                        break;
                    case "is_multiparam":
                        message.is_multiparam = reader.bool();
                        break;
                    default:
                        reader.field(tag, message);
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
    
            TensorProtos.decodeText = function decodeText(reader) {
                var message = new $root.caffe2.TensorProtos();
                reader.start();
                while (!reader.end()) {
                    var tag = reader.tag();
                    switch (tag) {
                    case "protos":
                        if (!(message.protos && message.protos.length))
                            message.protos = [];
                        message.protos.push($root.caffe2.TensorProto.decodeText(reader, true));
                        break;
                    default:
                        reader.field(tag, message);
                        break;
                    }
                }
                return message;
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
    
            TensorShape.decodeText = function decodeText(reader) {
                var message = new $root.caffe2.TensorShape();
                reader.start();
                while (!reader.end()) {
                    var tag = reader.tag();
                    switch (tag) {
                    case "dims":
                        if (!(message.dims && message.dims.length))
                            message.dims = [];
                        if (reader.first())
                            while (!reader.last()) {
                                message.dims.push(reader.int64());
                                reader.next();
                            }
                        else
                            message.dims.push(reader.int64());
                        break;
                    case "data_type":
                        message.data_type = reader.enum($root.caffe2.TensorProto.DataType);
                        break;
                    case "unknown_dims":
                        if (!(message.unknown_dims && message.unknown_dims.length))
                            message.unknown_dims = [];
                        if (reader.first())
                            while (!reader.last()) {
                                message.unknown_dims.push(reader.int32());
                                reader.next();
                            }
                        else
                            message.unknown_dims.push(reader.int32());
                        break;
                    case "unknown_shape":
                        message.unknown_shape = reader.bool();
                        break;
                    case "name":
                        message.name = reader.string();
                        break;
                    default:
                        reader.field(tag, message);
                        break;
                    }
                }
                return message;
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
    
            TensorShapes.decodeText = function decodeText(reader) {
                var message = new $root.caffe2.TensorShapes();
                reader.start();
                while (!reader.end()) {
                    var tag = reader.tag();
                    switch (tag) {
                    case "shapes":
                        if (!(message.shapes && message.shapes.length))
                            message.shapes = [];
                        message.shapes.push($root.caffe2.TensorShape.decodeText(reader, true));
                        break;
                    default:
                        reader.field(tag, message);
                        break;
                    }
                }
                return message;
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
                this.qtensors = [];
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
            Argument.prototype.qtensors = $util.emptyArray;
    
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
                    case 12:
                        if (!(message.qtensors && message.qtensors.length))
                            message.qtensors = [];
                        message.qtensors.push($root.caffe2.QTensorProto.decode(reader, reader.uint32()));
                        break;
                    default:
                        reader.skipType(tag & 7);
                        break;
                    }
                }
                return message;
            };
    
            Argument.decodeText = function decodeText(reader) {
                var message = new $root.caffe2.Argument();
                reader.start();
                while (!reader.end()) {
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
                        if (reader.first())
                            while (!reader.last()) {
                                message.floats.push(reader.float());
                                reader.next();
                            }
                        else
                            message.floats.push(reader.float());
                        break;
                    case "ints":
                        if (!(message.ints && message.ints.length))
                            message.ints = [];
                        if (reader.first())
                            while (!reader.last()) {
                                message.ints.push(reader.int64());
                                reader.next();
                            }
                        else
                            message.ints.push(reader.int64());
                        break;
                    case "strings":
                        if (!(message.strings && message.strings.length))
                            message.strings = [];
                        if (reader.first())
                            while (!reader.last()) {
                                message.strings.push(reader.bytes());
                                reader.next();
                            }
                        else
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
                    case "qtensors":
                        if (!(message.qtensors && message.qtensors.length))
                            message.qtensors = [];
                        message.qtensors.push($root.caffe2.QTensorProto.decodeText(reader, true));
                        break;
                    default:
                        reader.field(tag, message);
                        break;
                    }
                }
                return message;
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
            values[valuesById[8] = "PROTO_MSNPU"] = 8;
            values[valuesById[9] = "PROTO_XLA"] = 9;
            values[valuesById[10] = "PROTO_COMPILE_TIME_MAX_DEVICE_TYPES"] = 10;
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
    
            DeviceOption.decodeText = function decodeText(reader) {
                var message = new $root.caffe2.DeviceOption();
                reader.start();
                while (!reader.end()) {
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
                        if (reader.first())
                            while (!reader.last()) {
                                message.extra_info.push(reader.string());
                                reader.next();
                            }
                        else
                            message.extra_info.push(reader.string());
                        break;
                    default:
                        reader.field(tag, message);
                        break;
                    }
                }
                return message;
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
    
            OperatorDef.decodeText = function decodeText(reader) {
                var message = new $root.caffe2.OperatorDef();
                reader.start();
                while (!reader.end()) {
                    var tag = reader.tag();
                    switch (tag) {
                    case "input":
                        if (!(message.input && message.input.length))
                            message.input = [];
                        if (reader.first())
                            while (!reader.last()) {
                                message.input.push(reader.string());
                                reader.next();
                            }
                        else
                            message.input.push(reader.string());
                        break;
                    case "output":
                        if (!(message.output && message.output.length))
                            message.output = [];
                        if (reader.first())
                            while (!reader.last()) {
                                message.output.push(reader.string());
                                reader.next();
                            }
                        else
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
                        if (reader.first())
                            while (!reader.last()) {
                                message.control_input.push(reader.string());
                                reader.next();
                            }
                        else
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
                        reader.field(tag, message);
                        break;
                    }
                }
                return message;
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
    
            NetDef.decodeText = function decodeText(reader) {
                var message = new $root.caffe2.NetDef();
                reader.start();
                while (!reader.end()) {
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
                        if (reader.first())
                            while (!reader.last()) {
                                message.external_input.push(reader.string());
                                reader.next();
                            }
                        else
                            message.external_input.push(reader.string());
                        break;
                    case "external_output":
                        if (!(message.external_output && message.external_output.length))
                            message.external_output = [];
                        if (reader.first())
                            while (!reader.last()) {
                                message.external_output.push(reader.string());
                                reader.next();
                            }
                        else
                            message.external_output.push(reader.string());
                        break;
                    default:
                        reader.field(tag, message);
                        break;
                    }
                }
                return message;
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
    
            ExecutionStep.decodeText = function decodeText(reader) {
                var message = new $root.caffe2.ExecutionStep();
                reader.start();
                while (!reader.end()) {
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
                        if (reader.first())
                            while (!reader.last()) {
                                message.network.push(reader.string());
                                reader.next();
                            }
                        else
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
                        reader.field(tag, message);
                        break;
                    }
                }
                return message;
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
    
            PlanDef.decodeText = function decodeText(reader) {
                var message = new $root.caffe2.PlanDef();
                reader.start();
                while (!reader.end()) {
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
                        reader.field(tag, message);
                        break;
                    }
                }
                return message;
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
    
            BlobProto.decodeText = function decodeText(reader) {
                var message = new $root.caffe2.BlobProto();
                reader.start();
                while (!reader.end()) {
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
                        reader.field(tag, message);
                        break;
                    }
                }
                return message;
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
    
            DBReaderProto.decodeText = function decodeText(reader) {
                var message = new $root.caffe2.DBReaderProto();
                reader.start();
                while (!reader.end()) {
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
                        reader.field(tag, message);
                        break;
                    }
                }
                return message;
            };
    
            return DBReaderProto;
        })();
    
        return caffe2;
    })();

    return $root;
})(protobuf);
