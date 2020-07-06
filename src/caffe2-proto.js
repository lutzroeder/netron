(function($protobuf) {
    "use strict";

    const $root = $protobuf.get('caffe2');

    $root.caffe2 = (function() {

        const caffe2 = {};

        caffe2.ExternalDataProto = (function() {

            function ExternalDataProto() {
                this.strides = [];
            }

            ExternalDataProto.prototype.source_type = 0;
            ExternalDataProto.prototype.record_id = "";
            ExternalDataProto.prototype.record_size = $protobuf.Long ? $protobuf.Long.fromBits(0, 0, true) : 0;
            ExternalDataProto.prototype.offset = $protobuf.Long ? $protobuf.Long.fromBits(0, 0, false) : 0;
            ExternalDataProto.prototype.strides = [];

            ExternalDataProto.decode = function (reader, length) {
                const message = new $root.caffe2.ExternalDataProto();
                const end = reader.next(length);
                while (reader.end(end)) {
                    const tag = reader.uint32();
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
                            message.strides = reader.array(message.strides, () => reader.int64(), tag);
                            break;
                        default:
                            reader.skipType(tag & 7);
                            break;
                    }
                }
                return message;
            };

            ExternalDataProto.decodeText = function (reader) {
                const message = new $root.caffe2.ExternalDataProto();
                reader.start();
                while (!reader.end()) {
                    const tag = reader.tag();
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
                            reader.array(message.strides, () => reader.int64());
                            break;
                        default:
                            reader.field(tag, message);
                            break;
                    }
                }
                return message;
            };

            ExternalDataProto.SourceType = (function() {
                const values = {};
                values["INLINE_CONTAINER"] = 0;
                values["SIMPLE_FILE"] = 1;
                return values;
            })();

            return ExternalDataProto;
        })();

        caffe2.TensorProto = (function() {

            function TensorProto() {
                this.dims = [];
                this.float_data = [];
                this.int32_data = [];
                this.string_data = [];
                this.double_data = [];
                this.int64_data = [];
            }

            TensorProto.prototype.dims = [];
            TensorProto.prototype.data_type = 1;
            TensorProto.prototype.storage_type = 1;
            TensorProto.prototype.float_data = [];
            TensorProto.prototype.int32_data = [];
            TensorProto.prototype.byte_data = new Uint8Array([]);
            TensorProto.prototype.string_data = [];
            TensorProto.prototype.double_data = [];
            TensorProto.prototype.int64_data = [];
            TensorProto.prototype.raw_data = new Uint8Array([]);
            TensorProto.prototype.external_data = null;
            TensorProto.prototype.name = "";
            TensorProto.prototype.device_detail = null;
            TensorProto.prototype.segment = null;

            TensorProto.decode = function (reader, length) {
                const message = new $root.caffe2.TensorProto();
                const end = reader.next(length);
                while (reader.end(end)) {
                    const tag = reader.uint32();
                    switch (tag >>> 3) {
                        case 1:
                            message.dims = reader.array(message.dims, () => reader.int64(), tag);
                            break;
                        case 2:
                            message.data_type = reader.int32();
                            break;
                        case 12:
                            message.storage_type = reader.int32();
                            break;
                        case 3:
                            message.float_data = reader.floats(message.float_data, tag);
                            break;
                        case 4:
                            message.int32_data = reader.array(message.int32_data, () => reader.int32(), tag);
                            break;
                        case 5:
                            message.byte_data = reader.bytes();
                            break;
                        case 6:
                            message.string_data.push(reader.bytes());
                            break;
                        case 9:
                            message.double_data = reader.doubles(message.double_data, tag);
                            break;
                        case 10:
                            message.int64_data = reader.array(message.int64_data, () => reader.int64(), tag);
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

            TensorProto.decodeText = function (reader) {
                const message = new $root.caffe2.TensorProto();
                reader.start();
                while (!reader.end()) {
                    const tag = reader.tag();
                    switch (tag) {
                        case "dims":
                            reader.array(message.dims, () => reader.int64());
                            break;
                        case "data_type":
                            message.data_type = reader.enum($root.caffe2.TensorProto.DataType);
                            break;
                        case "storage_type":
                            message.storage_type = reader.enum($root.caffe2.TensorProto.StorageType);
                            break;
                        case "float_data":
                            reader.array(message.float_data, () => reader.float());
                            break;
                        case "int32_data":
                            reader.array(message.int32_data, () => reader.int32());
                            break;
                        case "byte_data":
                            message.byte_data = reader.bytes();
                            break;
                        case "string_data":
                            reader.array(message.string_data, () => reader.bytes());
                            break;
                        case "double_data":
                            reader.array(message.double_data, () => reader.double());
                            break;
                        case "int64_data":
                            reader.array(message.int64_data, () => reader.int64());
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
                const values = {};
                values["UNDEFINED"] = 0;
                values["FLOAT"] = 1;
                values["INT32"] = 2;
                values["BYTE"] = 3;
                values["STRING"] = 4;
                values["BOOL"] = 5;
                values["UINT8"] = 6;
                values["INT8"] = 7;
                values["UINT16"] = 8;
                values["INT16"] = 9;
                values["INT64"] = 10;
                values["FLOAT16"] = 12;
                values["DOUBLE"] = 13;
                values["ZERO_COLLISION_HASH"] = 14;
                return values;
            })();

            TensorProto.StorageType = (function() {
                const values = {};
                values["TYPED"] = 1;
                values["RAW"] = 2;
                values["EXTERNAL"] = 3;
                values["NO_CONTENT"] = 4;
                return values;
            })();

            TensorProto.Segment = (function() {

                function Segment() {
                }

                Segment.prototype.begin = $protobuf.Long ? $protobuf.Long.fromBits(0, 0, false) : 0;
                Segment.prototype.end = $protobuf.Long ? $protobuf.Long.fromBits(0, 0, false) : 0;

                Segment.decode = function (reader, length) {
                    const message = new $root.caffe2.TensorProto.Segment();
                    const end = reader.next(length);
                    while (reader.end(end)) {
                        const tag = reader.uint32();
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
                    if (!Object.prototype.hasOwnProperty.call(message, 'begin')) {
                        throw $protobuf.Error("Excepted 'begin'.");
                    }
                    if (!Object.prototype.hasOwnProperty.call(message, 'end')) {
                        throw $protobuf.Error("Excepted 'end'.");
                    }
                    return message;
                };

                Segment.decodeText = function (reader) {
                    const message = new $root.caffe2.TensorProto.Segment();
                    reader.start();
                    while (!reader.end()) {
                        const tag = reader.tag();
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
                    if (!Object.prototype.hasOwnProperty.call(message, "begin"))
                        throw $protobuf.Error("Excepted 'begin'.");
                    if (!Object.prototype.hasOwnProperty.call(message, "end"))
                        throw $protobuf.Error("Excepted 'end'.");
                    return message;
                };

                return Segment;
            })();

            return TensorProto;
        })();

        caffe2.QTensorProto = (function() {

            function QTensorProto() {
                this.dims = [];
                this.data = [];
                this.scales = [];
                this.biases = [];
            }

            QTensorProto.prototype.dims = [];
            QTensorProto.prototype.precision = 0;
            QTensorProto.prototype.scale = 0;
            QTensorProto.prototype.bias = 0;
            QTensorProto.prototype.is_signed = false;
            QTensorProto.prototype.data = [];
            QTensorProto.prototype.name = "";
            QTensorProto.prototype.data_type = 2;
            QTensorProto.prototype.scales = [];
            QTensorProto.prototype.biases = [];
            QTensorProto.prototype.axis = 0;
            QTensorProto.prototype.is_multiparam = false;

            QTensorProto.decode = function (reader, length) {
                const message = new $root.caffe2.QTensorProto();
                const end = reader.next(length);
                while (reader.end(end)) {
                    const tag = reader.uint32();
                    switch (tag >>> 3) {
                        case 1:
                            message.dims = reader.array(message.dims, () => reader.int64(), tag);
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
                            message.data = reader.array(message.data, () => reader.int32(), tag);
                            break;
                        case 7:
                            message.name = reader.string();
                            break;
                        case 8:
                            message.data_type = reader.int32();
                            break;
                        case 9:
                            message.scales = reader.doubles(message.scales, tag);
                            break;
                        case 10:
                            message.biases = reader.doubles(message.biases, tag);
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
                if (!Object.prototype.hasOwnProperty.call(message, 'precision')) {
                    throw $protobuf.Error("Excepted 'precision'.");
                }
                if (!Object.prototype.hasOwnProperty.call(message, 'scale')) {
                    throw $protobuf.Error("Excepted 'scale'.");
                }
                if (!Object.prototype.hasOwnProperty.call(message, 'bias')) {
                    throw $protobuf.Error("Excepted 'bias'.");
                }
                if (!Object.prototype.hasOwnProperty.call(message, 'is_signed')) {
                    throw $protobuf.Error("Excepted 'is_signed'.");
                }
                return message;
            };

            QTensorProto.decodeText = function (reader) {
                const message = new $root.caffe2.QTensorProto();
                reader.start();
                while (!reader.end()) {
                    const tag = reader.tag();
                    switch (tag) {
                        case "dims":
                            reader.array(message.dims, () => reader.int64());
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
                            reader.array(message.data, () => reader.int32());
                            break;
                        case "name":
                            message.name = reader.string();
                            break;
                        case "data_type":
                            message.data_type = reader.enum($root.caffe2.TensorProto.DataType);
                            break;
                        case "scales":
                            reader.array(message.scales, () => reader.double());
                            break;
                        case "biases":
                            reader.array(message.biases, () => reader.double());
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
                if (!Object.prototype.hasOwnProperty.call(message, "precision"))
                    throw $protobuf.Error("Excepted 'precision'.");
                if (!Object.prototype.hasOwnProperty.call(message, "scale"))
                    throw $protobuf.Error("Excepted 'scale'.");
                if (!Object.prototype.hasOwnProperty.call(message, "bias"))
                    throw $protobuf.Error("Excepted 'bias'.");
                if (!Object.prototype.hasOwnProperty.call(message, "is_signed"))
                    throw $protobuf.Error("Excepted 'is_signed'.");
                return message;
            };

            return QTensorProto;
        })();

        caffe2.TensorProtos = (function() {

            function TensorProtos() {
                this.protos = [];
            }

            TensorProtos.prototype.protos = [];

            TensorProtos.decode = function (reader, length) {
                const message = new $root.caffe2.TensorProtos();
                const end = reader.next(length);
                while (reader.end(end)) {
                    const tag = reader.uint32();
                    switch (tag >>> 3) {
                        case 1:
                            message.protos.push($root.caffe2.TensorProto.decode(reader, reader.uint32()));
                            break;
                        default:
                            reader.skipType(tag & 7);
                            break;
                    }
                }
                return message;
            };

            TensorProtos.decodeText = function (reader) {
                const message = new $root.caffe2.TensorProtos();
                reader.start();
                while (!reader.end()) {
                    const tag = reader.tag();
                    switch (tag) {
                        case "protos":
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

            function TensorShape() {
                this.dims = [];
                this.unknown_dims = [];
            }

            TensorShape.prototype.dims = [];
            TensorShape.prototype.data_type = 1;
            TensorShape.prototype.unknown_dims = [];
            TensorShape.prototype.unknown_shape = false;
            TensorShape.prototype.name = "";

            TensorShape.decode = function (reader, length) {
                const message = new $root.caffe2.TensorShape();
                const end = reader.next(length);
                while (reader.end(end)) {
                    const tag = reader.uint32();
                    switch (tag >>> 3) {
                        case 1:
                            message.dims = reader.array(message.dims, () => reader.int64(), tag);
                            break;
                        case 2:
                            message.data_type = reader.int32();
                            break;
                        case 3:
                            message.unknown_dims = reader.array(message.unknown_dims, () => reader.int32(), tag);
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

            TensorShape.decodeText = function (reader) {
                const message = new $root.caffe2.TensorShape();
                reader.start();
                while (!reader.end()) {
                    const tag = reader.tag();
                    switch (tag) {
                        case "dims":
                            reader.array(message.dims, () => reader.int64());
                            break;
                        case "data_type":
                            message.data_type = reader.enum($root.caffe2.TensorProto.DataType);
                            break;
                        case "unknown_dims":
                            reader.array(message.unknown_dims, () => reader.int32());
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

            function TensorShapes() {
                this.shapes = [];
            }

            TensorShapes.prototype.shapes = [];

            TensorShapes.decode = function (reader, length) {
                const message = new $root.caffe2.TensorShapes();
                const end = reader.next(length);
                while (reader.end(end)) {
                    const tag = reader.uint32();
                    switch (tag >>> 3) {
                        case 1:
                            message.shapes.push($root.caffe2.TensorShape.decode(reader, reader.uint32()));
                            break;
                        default:
                            reader.skipType(tag & 7);
                            break;
                    }
                }
                return message;
            };

            TensorShapes.decodeText = function (reader) {
                const message = new $root.caffe2.TensorShapes();
                reader.start();
                while (!reader.end()) {
                    const tag = reader.tag();
                    switch (tag) {
                        case "shapes":
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

        caffe2.TensorBoundShape = (function() {

            function TensorBoundShape() {
                this.dim_type = [];
            }

            TensorBoundShape.prototype.shape = null;
            TensorBoundShape.prototype.dim_type = [];
            TensorBoundShape.prototype.name = "";

            TensorBoundShape.decode = function (reader, length) {
                const message = new $root.caffe2.TensorBoundShape();
                const end = reader.next(length);
                while (reader.end(end)) {
                    const tag = reader.uint32();
                    switch (tag >>> 3) {
                        case 1:
                            message.shape = $root.caffe2.TensorShape.decode(reader, reader.uint32());
                            break;
                        case 2:
                            message.dim_type = reader.array(message.dim_type, () => reader.int32(), tag);
                            break;
                        case 3:
                            message.name = reader.string();
                            break;
                        default:
                            reader.skipType(tag & 7);
                            break;
                    }
                }
                return message;
            };

            TensorBoundShape.decodeText = function (reader) {
                const message = new $root.caffe2.TensorBoundShape();
                reader.start();
                while (!reader.end()) {
                    const tag = reader.tag();
                    switch (tag) {
                        case "shape":
                            message.shape = $root.caffe2.TensorShape.decodeText(reader, true);
                            break;
                        case "dim_type":
                            reader.array(message.dim_type, () => reader.enum($root.caffe2.TensorBoundShape.DimType));
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

            TensorBoundShape.DimType = (function() {
                const values = {};
                values["UNKNOWN"] = 0;
                values["CONSTANT"] = 1;
                values["BATCH"] = 2;
                values["BATCH_OF_FEATURE_MAX"] = 3;
                values["BATCH_OF_FEATURE_MAX_DEFAULT"] = 4;
                values["FEATURE_MAX"] = 5;
                values["FEATURE_MAX_DEFAULT"] = 6;
                return values;
            })();

            return TensorBoundShape;
        })();

        caffe2.TensorBoundShapes = (function() {

            function TensorBoundShapes() {
                this.shapes = [];
            }

            TensorBoundShapes.prototype.shapes = [];
            TensorBoundShapes.prototype.max_batch_size = $protobuf.Long ? $protobuf.Long.fromBits(0, 0, false) : 0;
            TensorBoundShapes.prototype.max_feature_len = $protobuf.Long ? $protobuf.Long.fromBits(0, 0, false) : 0;

            TensorBoundShapes.decode = function (reader, length) {
                const message = new $root.caffe2.TensorBoundShapes();
                const end = reader.next(length);
                while (reader.end(end)) {
                    const tag = reader.uint32();
                    switch (tag >>> 3) {
                        case 1:
                            message.shapes.push($root.caffe2.TensorBoundShape.decode(reader, reader.uint32()));
                            break;
                        case 2:
                            message.max_batch_size = reader.int64();
                            break;
                        case 3:
                            message.max_feature_len = reader.int64();
                            break;
                        default:
                            reader.skipType(tag & 7);
                            break;
                    }
                }
                return message;
            };

            TensorBoundShapes.decodeText = function (reader) {
                const message = new $root.caffe2.TensorBoundShapes();
                reader.start();
                while (!reader.end()) {
                    const tag = reader.tag();
                    switch (tag) {
                        case "shapes":
                            message.shapes.push($root.caffe2.TensorBoundShape.decodeText(reader, true));
                            break;
                        case "max_batch_size":
                            message.max_batch_size = reader.int64();
                            break;
                        case "max_feature_len":
                            message.max_feature_len = reader.int64();
                            break;
                        default:
                            reader.field(tag, message);
                            break;
                    }
                }
                return message;
            };

            return TensorBoundShapes;
        })();

        caffe2.Argument = (function() {

            function Argument() {
                this.floats = [];
                this.ints = [];
                this.strings = [];
                this.tensors = [];
                this.nets = [];
                this.qtensors = [];
            }

            Argument.prototype.name = "";
            Argument.prototype.f = 0;
            Argument.prototype.i = $protobuf.Long ? $protobuf.Long.fromBits(0, 0, false) : 0;
            Argument.prototype.s = new Uint8Array([]);
            Argument.prototype.t = null;
            Argument.prototype.n = null;
            Argument.prototype.floats = [];
            Argument.prototype.ints = [];
            Argument.prototype.strings = [];
            Argument.prototype.tensors = [];
            Argument.prototype.nets = [];
            Argument.prototype.qtensors = [];

            Argument.decode = function (reader, length) {
                const message = new $root.caffe2.Argument();
                const end = reader.next(length);
                while (reader.end(end)) {
                    const tag = reader.uint32();
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
                            message.floats = reader.floats(message.floats, tag);
                            break;
                        case 6:
                            message.ints = reader.array(message.ints, () => reader.int64(), tag);
                            break;
                        case 7:
                            message.strings.push(reader.bytes());
                            break;
                        case 11:
                            message.tensors.push($root.caffe2.TensorProto.decode(reader, reader.uint32()));
                            break;
                        case 9:
                            message.nets.push($root.caffe2.NetDef.decode(reader, reader.uint32()));
                            break;
                        case 12:
                            message.qtensors.push($root.caffe2.QTensorProto.decode(reader, reader.uint32()));
                            break;
                        default:
                            reader.skipType(tag & 7);
                            break;
                    }
                }
                return message;
            };

            Argument.decodeText = function (reader) {
                const message = new $root.caffe2.Argument();
                reader.start();
                while (!reader.end()) {
                    const tag = reader.tag();
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
                            reader.array(message.floats, () => reader.float());
                            break;
                        case "ints":
                            reader.array(message.ints, () => reader.int64());
                            break;
                        case "strings":
                            reader.array(message.strings, () => reader.bytes());
                            break;
                        case "tensors":
                            message.tensors.push($root.caffe2.TensorProto.decodeText(reader, true));
                            break;
                        case "nets":
                            message.nets.push($root.caffe2.NetDef.decodeText(reader, true));
                            break;
                        case "qtensors":
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
            const values = {};
            values["PROTO_CPU"] = 0;
            values["PROTO_CUDA"] = 1;
            values["PROTO_MKLDNN"] = 2;
            values["PROTO_OPENGL"] = 3;
            values["PROTO_OPENCL"] = 4;
            values["PROTO_IDEEP"] = 5;
            values["PROTO_HIP"] = 6;
            values["PROTO_FPGA"] = 7;
            values["PROTO_MSNPU"] = 8;
            values["PROTO_XLA"] = 9;
            values["PROTO_COMPILE_TIME_MAX_DEVICE_TYPES"] = 10;
            values["PROTO_ONLY_FOR_TEST"] = 20901;
            return values;
        })();

        caffe2.DeviceOption = (function() {

            function DeviceOption() {
                this.extra_info = [];
            }

            DeviceOption.prototype.device_type = 0;
            DeviceOption.prototype.device_id = 0;
            DeviceOption.prototype.random_seed = 0;
            DeviceOption.prototype.node_name = "";
            DeviceOption.prototype.numa_node_id = 0;
            DeviceOption.prototype.extra_info = [];

            DeviceOption.decode = function (reader, length) {
                const message = new $root.caffe2.DeviceOption();
                const end = reader.next(length);
                while (reader.end(end)) {
                    const tag = reader.uint32();
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
                            message.extra_info.push(reader.string());
                            break;
                        default:
                            reader.skipType(tag & 7);
                            break;
                    }
                }
                return message;
            };

            DeviceOption.decodeText = function (reader) {
                const message = new $root.caffe2.DeviceOption();
                reader.start();
                while (!reader.end()) {
                    const tag = reader.tag();
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
                            reader.array(message.extra_info, () => reader.string());
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

            function OperatorDef() {
                this.input = [];
                this.output = [];
                this.arg = [];
                this.control_input = [];
            }

            OperatorDef.prototype.input = [];
            OperatorDef.prototype.output = [];
            OperatorDef.prototype.name = "";
            OperatorDef.prototype.type = "";
            OperatorDef.prototype.arg = [];
            OperatorDef.prototype.device_option = null;
            OperatorDef.prototype.engine = "";
            OperatorDef.prototype.control_input = [];
            OperatorDef.prototype.is_gradient_op = false;
            OperatorDef.prototype.debug_info = "";
            OperatorDef.prototype.domain = "";
            OperatorDef.prototype.op_version = $protobuf.Long ? $protobuf.Long.fromBits(0, 0, false) : 0;

            OperatorDef.decode = function (reader, length) {
                const message = new $root.caffe2.OperatorDef();
                const end = reader.next(length);
                while (reader.end(end)) {
                    const tag = reader.uint32();
                    switch (tag >>> 3) {
                        case 1:
                            message.input.push(reader.string());
                            break;
                        case 2:
                            message.output.push(reader.string());
                            break;
                        case 3:
                            message.name = reader.string();
                            break;
                        case 4:
                            message.type = reader.string();
                            break;
                        case 5:
                            message.arg.push($root.caffe2.Argument.decode(reader, reader.uint32()));
                            break;
                        case 6:
                            message.device_option = $root.caffe2.DeviceOption.decode(reader, reader.uint32());
                            break;
                        case 7:
                            message.engine = reader.string();
                            break;
                        case 8:
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

            OperatorDef.decodeText = function (reader) {
                const message = new $root.caffe2.OperatorDef();
                reader.start();
                while (!reader.end()) {
                    const tag = reader.tag();
                    switch (tag) {
                        case "input":
                            reader.array(message.input, () => reader.string());
                            break;
                        case "output":
                            reader.array(message.output, () => reader.string());
                            break;
                        case "name":
                            message.name = reader.string();
                            break;
                        case "type":
                            message.type = reader.string();
                            break;
                        case "arg":
                            message.arg.push($root.caffe2.Argument.decodeText(reader, true));
                            break;
                        case "device_option":
                            message.device_option = $root.caffe2.DeviceOption.decodeText(reader, true);
                            break;
                        case "engine":
                            message.engine = reader.string();
                            break;
                        case "control_input":
                            reader.array(message.control_input, () => reader.string());
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

        caffe2.MapFieldEntry = (function() {

            function MapFieldEntry() {
            }

            MapFieldEntry.prototype.key = "";
            MapFieldEntry.prototype.val = "";

            MapFieldEntry.decode = function (reader, length) {
                const message = new $root.caffe2.MapFieldEntry();
                const end = reader.next(length);
                while (reader.end(end)) {
                    const tag = reader.uint32();
                    switch (tag >>> 3) {
                        case 1:
                            message.key = reader.string();
                            break;
                        case 2:
                            message.val = reader.string();
                            break;
                        default:
                            reader.skipType(tag & 7);
                            break;
                    }
                }
                if (!Object.prototype.hasOwnProperty.call(message, 'key')) {
                    throw $protobuf.Error("Excepted 'key'.");
                }
                if (!Object.prototype.hasOwnProperty.call(message, 'val')) {
                    throw $protobuf.Error("Excepted 'val'.");
                }
                return message;
            };

            MapFieldEntry.decodeText = function (reader) {
                const message = new $root.caffe2.MapFieldEntry();
                reader.start();
                while (!reader.end()) {
                    const tag = reader.tag();
                    switch (tag) {
                        case "key":
                            message.key = reader.string();
                            break;
                        case "val":
                            message.val = reader.string();
                            break;
                        default:
                            reader.field(tag, message);
                            break;
                    }
                }
                if (!Object.prototype.hasOwnProperty.call(message, "key"))
                    throw $protobuf.Error("Excepted 'key'.");
                if (!Object.prototype.hasOwnProperty.call(message, "val"))
                    throw $protobuf.Error("Excepted 'val'.");
                return message;
            };

            return MapFieldEntry;
        })();

        caffe2.BackendOptions = (function() {

            function BackendOptions() {
                this.option = [];
            }

            BackendOptions.prototype.backend_name = "";
            BackendOptions.prototype.option = [];

            BackendOptions.decode = function (reader, length) {
                const message = new $root.caffe2.BackendOptions();
                const end = reader.next(length);
                while (reader.end(end)) {
                    const tag = reader.uint32();
                    switch (tag >>> 3) {
                        case 1:
                            message.backend_name = reader.string();
                            break;
                        case 2:
                            message.option.push($root.caffe2.MapFieldEntry.decode(reader, reader.uint32()));
                            break;
                        default:
                            reader.skipType(tag & 7);
                            break;
                    }
                }
                if (!Object.prototype.hasOwnProperty.call(message, 'backend_name')) {
                    throw $protobuf.Error("Excepted 'backend_name'.");
                }
                return message;
            };

            BackendOptions.decodeText = function (reader) {
                const message = new $root.caffe2.BackendOptions();
                reader.start();
                while (!reader.end()) {
                    const tag = reader.tag();
                    switch (tag) {
                        case "backend_name":
                            message.backend_name = reader.string();
                            break;
                        case "option":
                            message.option.push($root.caffe2.MapFieldEntry.decodeText(reader, true));
                            break;
                        default:
                            reader.field(tag, message);
                            break;
                    }
                }
                if (!Object.prototype.hasOwnProperty.call(message, "backend_name"))
                    throw $protobuf.Error("Excepted 'backend_name'.");
                return message;
            };

            return BackendOptions;
        })();

        caffe2.PartitionInfo = (function() {

            function PartitionInfo() {
                this.device_id = [];
                this.backend_options = [];
            }

            PartitionInfo.prototype.name = "";
            PartitionInfo.prototype.device_id = [];
            PartitionInfo.prototype.extra_info = "";
            PartitionInfo.prototype.backend_options = [];

            PartitionInfo.decode = function (reader, length) {
                const message = new $root.caffe2.PartitionInfo();
                const end = reader.next(length);
                while (reader.end(end)) {
                    const tag = reader.uint32();
                    switch (tag >>> 3) {
                        case 1:
                            message.name = reader.string();
                            break;
                        case 2:
                            message.device_id = reader.array(message.device_id, () => reader.int32(), tag);
                            break;
                        case 3:
                            message.extra_info = reader.string();
                            break;
                        case 4:
                            message.backend_options.push($root.caffe2.BackendOptions.decode(reader, reader.uint32()));
                            break;
                        default:
                            reader.skipType(tag & 7);
                            break;
                    }
                }
                if (!Object.prototype.hasOwnProperty.call(message, 'name')) {
                    throw $protobuf.Error("Excepted 'name'.");
                }
                return message;
            };

            PartitionInfo.decodeText = function (reader) {
                const message = new $root.caffe2.PartitionInfo();
                reader.start();
                while (!reader.end()) {
                    const tag = reader.tag();
                    switch (tag) {
                        case "name":
                            message.name = reader.string();
                            break;
                        case "device_id":
                            reader.array(message.device_id, () => reader.int32());
                            break;
                        case "extra_info":
                            message.extra_info = reader.string();
                            break;
                        case "backend_options":
                            message.backend_options.push($root.caffe2.BackendOptions.decodeText(reader, true));
                            break;
                        default:
                            reader.field(tag, message);
                            break;
                    }
                }
                if (!Object.prototype.hasOwnProperty.call(message, "name"))
                    throw $protobuf.Error("Excepted 'name'.");
                return message;
            };

            return PartitionInfo;
        })();

        caffe2.NetDef = (function() {

            function NetDef() {
                this.op = [];
                this.arg = [];
                this.external_input = [];
                this.external_output = [];
                this.partition_info = [];
            }

            NetDef.prototype.name = "";
            NetDef.prototype.op = [];
            NetDef.prototype.type = "";
            NetDef.prototype.num_workers = 0;
            NetDef.prototype.device_option = null;
            NetDef.prototype.arg = [];
            NetDef.prototype.external_input = [];
            NetDef.prototype.external_output = [];
            NetDef.prototype.partition_info = [];

            NetDef.decode = function (reader, length) {
                const message = new $root.caffe2.NetDef();
                const end = reader.next(length);
                while (reader.end(end)) {
                    const tag = reader.uint32();
                    switch (tag >>> 3) {
                        case 1:
                            message.name = reader.string();
                            break;
                        case 2:
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
                            message.arg.push($root.caffe2.Argument.decode(reader, reader.uint32()));
                            break;
                        case 7:
                            message.external_input.push(reader.string());
                            break;
                        case 8:
                            message.external_output.push(reader.string());
                            break;
                        case 9:
                            message.partition_info.push($root.caffe2.PartitionInfo.decode(reader, reader.uint32()));
                            break;
                        default:
                            reader.skipType(tag & 7);
                            break;
                    }
                }
                return message;
            };

            NetDef.decodeText = function (reader) {
                const message = new $root.caffe2.NetDef();
                reader.start();
                while (!reader.end()) {
                    const tag = reader.tag();
                    switch (tag) {
                        case "name":
                            message.name = reader.string();
                            break;
                        case "op":
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
                            message.arg.push($root.caffe2.Argument.decodeText(reader, true));
                            break;
                        case "external_input":
                            reader.array(message.external_input, () => reader.string());
                            break;
                        case "external_output":
                            reader.array(message.external_output, () => reader.string());
                            break;
                        case "partition_info":
                            message.partition_info.push($root.caffe2.PartitionInfo.decodeText(reader, true));
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

            function ExecutionStep() {
                this.substep = [];
                this.network = [];
            }

            ExecutionStep.prototype.name = "";
            ExecutionStep.prototype.substep = [];
            ExecutionStep.prototype.network = [];
            ExecutionStep.prototype.num_iter = $protobuf.Long ? $protobuf.Long.fromBits(0, 0, false) : 0;
            ExecutionStep.prototype.criteria_network = "";
            ExecutionStep.prototype.report_net = "";
            ExecutionStep.prototype.report_interval = 0;
            ExecutionStep.prototype.run_every_ms = $protobuf.Long ? $protobuf.Long.fromBits(0, 0, false) : 0;
            ExecutionStep.prototype.concurrent_substeps = false;
            ExecutionStep.prototype.should_stop_blob = "";
            ExecutionStep.prototype.only_once = false;
            ExecutionStep.prototype.create_workspace = false;
            ExecutionStep.prototype.num_concurrent_instances = 0;

            ExecutionStep.decode = function (reader, length) {
                const message = new $root.caffe2.ExecutionStep();
                const end = reader.next(length);
                while (reader.end(end)) {
                    const tag = reader.uint32();
                    switch (tag >>> 3) {
                        case 1:
                            message.name = reader.string();
                            break;
                        case 2:
                            message.substep.push($root.caffe2.ExecutionStep.decode(reader, reader.uint32()));
                            break;
                        case 3:
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

            ExecutionStep.decodeText = function (reader) {
                const message = new $root.caffe2.ExecutionStep();
                reader.start();
                while (!reader.end()) {
                    const tag = reader.tag();
                    switch (tag) {
                        case "name":
                            message.name = reader.string();
                            break;
                        case "substep":
                            message.substep.push($root.caffe2.ExecutionStep.decodeText(reader, true));
                            break;
                        case "network":
                            reader.array(message.network, () => reader.string());
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

            function PlanDef() {
                this.network = [];
                this.execution_step = [];
            }

            PlanDef.prototype.name = "";
            PlanDef.prototype.network = [];
            PlanDef.prototype.execution_step = [];

            PlanDef.decode = function (reader, length) {
                const message = new $root.caffe2.PlanDef();
                const end = reader.next(length);
                while (reader.end(end)) {
                    const tag = reader.uint32();
                    switch (tag >>> 3) {
                        case 1:
                            message.name = reader.string();
                            break;
                        case 2:
                            message.network.push($root.caffe2.NetDef.decode(reader, reader.uint32()));
                            break;
                        case 3:
                            message.execution_step.push($root.caffe2.ExecutionStep.decode(reader, reader.uint32()));
                            break;
                        default:
                            reader.skipType(tag & 7);
                            break;
                    }
                }
                return message;
            };

            PlanDef.decodeText = function (reader) {
                const message = new $root.caffe2.PlanDef();
                reader.start();
                while (!reader.end()) {
                    const tag = reader.tag();
                    switch (tag) {
                        case "name":
                            message.name = reader.string();
                            break;
                        case "network":
                            message.network.push($root.caffe2.NetDef.decodeText(reader, true));
                            break;
                        case "execution_step":
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

            function BlobProto() {
            }

            BlobProto.prototype.name = "";
            BlobProto.prototype.type = "";
            BlobProto.prototype.tensor = null;
            BlobProto.prototype.content = new Uint8Array([]);
            BlobProto.prototype.qtensor = null;
            BlobProto.prototype.content_num_chunks = 0;
            BlobProto.prototype.content_chunk_id = 0;

            BlobProto.decode = function (reader, length) {
                const message = new $root.caffe2.BlobProto();
                const end = reader.next(length);
                while (reader.end(end)) {
                    const tag = reader.uint32();
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

            BlobProto.decodeText = function (reader) {
                const message = new $root.caffe2.BlobProto();
                reader.start();
                while (!reader.end()) {
                    const tag = reader.tag();
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

            function DBReaderProto() {
            }

            DBReaderProto.prototype.name = "";
            DBReaderProto.prototype.source = "";
            DBReaderProto.prototype.db_type = "";
            DBReaderProto.prototype.key = "";

            DBReaderProto.decode = function (reader, length) {
                const message = new $root.caffe2.DBReaderProto();
                const end = reader.next(length);
                while (reader.end(end)) {
                    const tag = reader.uint32();
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

            DBReaderProto.decodeText = function (reader) {
                const message = new $root.caffe2.DBReaderProto();
                reader.start();
                while (!reader.end()) {
                    const tag = reader.tag();
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
