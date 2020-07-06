(function($protobuf) {
    "use strict";

    const $root = $protobuf.get('bigdl');

    $root.com = (function() {

        const com = {};

        com.intel = (function() {

            const intel = {};

            intel.analytics = (function() {

                const analytics = {};

                analytics.bigdl = (function() {

                    const bigdl = {};

                    bigdl.serialization = (function() {

                        const serialization = {};

                        serialization.BigDLModule = (function() {

                            function BigDLModule() {
                                this.subModules = [];
                                this.preModules = [];
                                this.nextModules = [];
                                this.attr = {};
                                this.parameters = [];
                                this.inputScales = [];
                                this.outputScales = [];
                                this.weightScales = [];
                            }

                            BigDLModule.prototype.name = "";
                            BigDLModule.prototype.subModules = [];
                            BigDLModule.prototype.weight = null;
                            BigDLModule.prototype.bias = null;
                            BigDLModule.prototype.preModules = [];
                            BigDLModule.prototype.nextModules = [];
                            BigDLModule.prototype.moduleType = "";
                            BigDLModule.prototype.attr = {};
                            BigDLModule.prototype.version = "";
                            BigDLModule.prototype.train = false;
                            BigDLModule.prototype.namePostfix = "";
                            BigDLModule.prototype.id = 0;
                            BigDLModule.prototype.inputShape = null;
                            BigDLModule.prototype.outputShape = null;
                            BigDLModule.prototype.hasParameters = false;
                            BigDLModule.prototype.parameters = [];
                            BigDLModule.prototype.isMklInt8Enabled = false;
                            BigDLModule.prototype.inputDimMasks = 0;
                            BigDLModule.prototype.inputScales = [];
                            BigDLModule.prototype.outputDimMasks = 0;
                            BigDLModule.prototype.outputScales = [];
                            BigDLModule.prototype.weightDimMasks = 0;
                            BigDLModule.prototype.weightScales = [];

                            BigDLModule.decode = function (reader, length) {
                                const message = new $root.com.intel.analytics.bigdl.serialization.BigDLModule();
                                const end = reader.next(length);
                                while (reader.end(end)) {
                                    const tag = reader.uint32();
                                    switch (tag >>> 3) {
                                        case 1:
                                            message.name = reader.string();
                                            break;
                                        case 2:
                                            message.subModules.push($root.com.intel.analytics.bigdl.serialization.BigDLModule.decode(reader, reader.uint32()));
                                            break;
                                        case 3:
                                            message.weight = $root.com.intel.analytics.bigdl.serialization.BigDLTensor.decode(reader, reader.uint32());
                                            break;
                                        case 4:
                                            message.bias = $root.com.intel.analytics.bigdl.serialization.BigDLTensor.decode(reader, reader.uint32());
                                            break;
                                        case 5:
                                            message.preModules.push(reader.string());
                                            break;
                                        case 6:
                                            message.nextModules.push(reader.string());
                                            break;
                                        case 7:
                                            message.moduleType = reader.string();
                                            break;
                                        case 8:
                                            reader.pair(message.attr, () => reader.string(), () => $root.com.intel.analytics.bigdl.serialization.AttrValue.decode(reader, reader.uint32()));
                                            break;
                                        case 9:
                                            message.version = reader.string();
                                            break;
                                        case 10:
                                            message.train = reader.bool();
                                            break;
                                        case 11:
                                            message.namePostfix = reader.string();
                                            break;
                                        case 12:
                                            message.id = reader.int32();
                                            break;
                                        case 13:
                                            message.inputShape = $root.com.intel.analytics.bigdl.serialization.Shape.decode(reader, reader.uint32());
                                            break;
                                        case 14:
                                            message.outputShape = $root.com.intel.analytics.bigdl.serialization.Shape.decode(reader, reader.uint32());
                                            break;
                                        case 15:
                                            message.hasParameters = reader.bool();
                                            break;
                                        case 16:
                                            message.parameters.push($root.com.intel.analytics.bigdl.serialization.BigDLTensor.decode(reader, reader.uint32()));
                                            break;
                                        case 17:
                                            message.isMklInt8Enabled = reader.bool();
                                            break;
                                        case 18:
                                            message.inputDimMasks = reader.int32();
                                            break;
                                        case 19:
                                            message.inputScales.push($root.com.intel.analytics.bigdl.serialization.AttrValue.decode(reader, reader.uint32()));
                                            break;
                                        case 20:
                                            message.outputDimMasks = reader.int32();
                                            break;
                                        case 21:
                                            message.outputScales.push($root.com.intel.analytics.bigdl.serialization.AttrValue.decode(reader, reader.uint32()));
                                            break;
                                        case 22:
                                            message.weightDimMasks = reader.int32();
                                            break;
                                        case 23:
                                            message.weightScales.push($root.com.intel.analytics.bigdl.serialization.AttrValue.decode(reader, reader.uint32()));
                                            break;
                                        default:
                                            reader.skipType(tag & 7);
                                            break;
                                    }
                                }
                                return message;
                            };

                            return BigDLModule;
                        })();

                        serialization.VarFormat = (function() {
                            const values = {};
                            values["EMPTY_FORMAT"] = 0;
                            values["DEFAULT"] = 1;
                            values["ONE_D"] = 2;
                            values["IN_OUT"] = 3;
                            values["OUT_IN"] = 4;
                            values["IN_OUT_KW_KH"] = 5;
                            values["OUT_IN_KW_KH"] = 6;
                            values["GP_OUT_IN_KW_KH"] = 7;
                            values["GP_IN_OUT_KW_KH"] = 8;
                            values["OUT_IN_KT_KH_KW"] = 9;
                            return values;
                        })();

                        serialization.InitMethodType = (function() {
                            const values = {};
                            values["EMPTY_INITIALIZATION"] = 0;
                            values["RANDOM_UNIFORM"] = 1;
                            values["RANDOM_UNIFORM_PARAM"] = 2;
                            values["RANDOM_NORMAL"] = 3;
                            values["ZEROS"] = 4;
                            values["ONES"] = 5;
                            values["CONST"] = 6;
                            values["XAVIER"] = 7;
                            values["BILINEARFILLER"] = 8;
                            return values;
                        })();

                        serialization.RegularizerType = (function() {
                            const values = {};
                            values["L1L2Regularizer"] = 0;
                            values["L1Regularizer"] = 1;
                            values["L2Regularizer"] = 2;
                            return values;
                        })();

                        serialization.InputDataFormat = (function() {
                            const values = {};
                            values["NCHW"] = 0;
                            values["NHWC"] = 1;
                            return values;
                        })();

                        serialization.TensorType = (function() {
                            const values = {};
                            values["DENSE"] = 0;
                            values["QUANT"] = 1;
                            return values;
                        })();

                        serialization.InitMethod = (function() {

                            function InitMethod() {
                                this.data = [];
                            }

                            InitMethod.prototype.methodType = 0;
                            InitMethod.prototype.data = [];

                            InitMethod.decode = function (reader, length) {
                                const message = new $root.com.intel.analytics.bigdl.serialization.InitMethod();
                                const end = reader.next(length);
                                while (reader.end(end)) {
                                    const tag = reader.uint32();
                                    switch (tag >>> 3) {
                                        case 1:
                                            message.methodType = reader.int32();
                                            break;
                                        case 2:
                                            message.data = reader.doubles(message.data, tag);
                                            break;
                                        default:
                                            reader.skipType(tag & 7);
                                            break;
                                    }
                                }
                                return message;
                            };

                            return InitMethod;
                        })();

                        serialization.BigDLTensor = (function() {

                            function BigDLTensor() {
                                this.size = [];
                                this.stride = [];
                            }

                            BigDLTensor.prototype.datatype = 0;
                            BigDLTensor.prototype.size = [];
                            BigDLTensor.prototype.stride = [];
                            BigDLTensor.prototype.offset = 0;
                            BigDLTensor.prototype.dimension = 0;
                            BigDLTensor.prototype.nElements = 0;
                            BigDLTensor.prototype.isScalar = false;
                            BigDLTensor.prototype.storage = null;
                            BigDLTensor.prototype.id = 0;
                            BigDLTensor.prototype.tensorType = 0;

                            BigDLTensor.decode = function (reader, length) {
                                const message = new $root.com.intel.analytics.bigdl.serialization.BigDLTensor();
                                const end = reader.next(length);
                                while (reader.end(end)) {
                                    const tag = reader.uint32();
                                    switch (tag >>> 3) {
                                        case 1:
                                            message.datatype = reader.int32();
                                            break;
                                        case 2:
                                            message.size = reader.array(message.size, () => reader.int32(), tag);
                                            break;
                                        case 3:
                                            message.stride = reader.array(message.stride, () => reader.int32(), tag);
                                            break;
                                        case 4:
                                            message.offset = reader.int32();
                                            break;
                                        case 5:
                                            message.dimension = reader.int32();
                                            break;
                                        case 6:
                                            message.nElements = reader.int32();
                                            break;
                                        case 7:
                                            message.isScalar = reader.bool();
                                            break;
                                        case 8:
                                            message.storage = $root.com.intel.analytics.bigdl.serialization.TensorStorage.decode(reader, reader.uint32());
                                            break;
                                        case 9:
                                            message.id = reader.int32();
                                            break;
                                        case 10:
                                            message.tensorType = reader.int32();
                                            break;
                                        default:
                                            reader.skipType(tag & 7);
                                            break;
                                    }
                                }
                                return message;
                            };

                            return BigDLTensor;
                        })();

                        serialization.TensorStorage = (function() {

                            function TensorStorage() {
                                this.float_data = [];
                                this.double_data = [];
                                this.bool_data = [];
                                this.string_data = [];
                                this.int_data = [];
                                this.long_data = [];
                                this.bytes_data = [];
                            }

                            TensorStorage.prototype.datatype = 0;
                            TensorStorage.prototype.float_data = [];
                            TensorStorage.prototype.double_data = [];
                            TensorStorage.prototype.bool_data = [];
                            TensorStorage.prototype.string_data = [];
                            TensorStorage.prototype.int_data = [];
                            TensorStorage.prototype.long_data = [];
                            TensorStorage.prototype.bytes_data = [];
                            TensorStorage.prototype.id = 0;

                            TensorStorage.decode = function (reader, length) {
                                const message = new $root.com.intel.analytics.bigdl.serialization.TensorStorage();
                                const end = reader.next(length);
                                while (reader.end(end)) {
                                    const tag = reader.uint32();
                                    switch (tag >>> 3) {
                                        case 1:
                                            message.datatype = reader.int32();
                                            break;
                                        case 2:
                                            message.float_data = reader.floats(message.float_data, tag);
                                            break;
                                        case 3:
                                            message.double_data = reader.doubles(message.double_data, tag);
                                            break;
                                        case 4:
                                            message.bool_data = reader.array(message.bool_data, () => reader.bool(), tag);
                                            break;
                                        case 5:
                                            message.string_data.push(reader.string());
                                            break;
                                        case 6:
                                            message.int_data = reader.array(message.int_data, () => reader.int32(), tag);
                                            break;
                                        case 7:
                                            message.long_data = reader.array(message.long_data, () => reader.int64(), tag);
                                            break;
                                        case 8:
                                            message.bytes_data.push(reader.bytes());
                                            break;
                                        case 9:
                                            message.id = reader.int32();
                                            break;
                                        default:
                                            reader.skipType(tag & 7);
                                            break;
                                    }
                                }
                                return message;
                            };

                            return TensorStorage;
                        })();

                        serialization.Regularizer = (function() {

                            function Regularizer() {
                                this.regularData = [];
                            }

                            Regularizer.prototype.regularizerType = 0;
                            Regularizer.prototype.regularData = [];

                            Regularizer.decode = function (reader, length) {
                                const message = new $root.com.intel.analytics.bigdl.serialization.Regularizer();
                                const end = reader.next(length);
                                while (reader.end(end)) {
                                    const tag = reader.uint32();
                                    switch (tag >>> 3) {
                                        case 1:
                                            message.regularizerType = reader.int32();
                                            break;
                                        case 2:
                                            message.regularData = reader.doubles(message.regularData, tag);
                                            break;
                                        default:
                                            reader.skipType(tag & 7);
                                            break;
                                    }
                                }
                                return message;
                            };

                            return Regularizer;
                        })();

                        serialization.DataType = (function() {
                            const values = {};
                            values["INT32"] = 0;
                            values["INT64"] = 1;
                            values["FLOAT"] = 2;
                            values["DOUBLE"] = 3;
                            values["STRING"] = 4;
                            values["BOOL"] = 5;
                            values["CHAR"] = 6;
                            values["SHORT"] = 7;
                            values["BYTES"] = 8;
                            values["REGULARIZER"] = 9;
                            values["TENSOR"] = 10;
                            values["VARIABLE_FORMAT"] = 11;
                            values["INITMETHOD"] = 12;
                            values["MODULE"] = 13;
                            values["NAME_ATTR_LIST"] = 14;
                            values["ARRAY_VALUE"] = 15;
                            values["DATA_FORMAT"] = 16;
                            values["CUSTOM"] = 17;
                            values["SHAPE"] = 18;
                            return values;
                        })();

                        serialization.AttrValue = (function() {

                            function AttrValue() {
                            }

                            AttrValue.prototype.dataType = 0;
                            AttrValue.prototype.subType = "";
                            AttrValue.prototype.int32Value = 0;
                            AttrValue.prototype.int64Value = $protobuf.Long ? $protobuf.Long.fromBits(0, 0, false) : 0;
                            AttrValue.prototype.floatValue = 0;
                            AttrValue.prototype.doubleValue = 0;
                            AttrValue.prototype.stringValue = "";
                            AttrValue.prototype.boolValue = false;
                            AttrValue.prototype.regularizerValue = null;
                            AttrValue.prototype.tensorValue = null;
                            AttrValue.prototype.variableFormatValue = 0;
                            AttrValue.prototype.initMethodValue = null;
                            AttrValue.prototype.bigDLModuleValue = null;
                            AttrValue.prototype.nameAttrListValue = null;
                            AttrValue.prototype.arrayValue = null;
                            AttrValue.prototype.dataFormatValue = 0;
                            AttrValue.prototype.customValue = null;
                            AttrValue.prototype.shape = null;

                            const valueSet = new Set([ "int32Value", "int64Value", "floatValue", "doubleValue", "stringValue", "boolValue", "regularizerValue", "tensorValue", "variableFormatValue", "initMethodValue", "bigDLModuleValue", "nameAttrListValue", "arrayValue", "dataFormatValue", "customValue", "shape"]);
                            Object.defineProperty(AttrValue.prototype, "value", {
                                get: function() { return Object.keys(this).find((key) => valueSet.has(key) && this[key] != null); }
                            });

                            AttrValue.decode = function (reader, length) {
                                const message = new $root.com.intel.analytics.bigdl.serialization.AttrValue();
                                const end = reader.next(length);
                                while (reader.end(end)) {
                                    const tag = reader.uint32();
                                    switch (tag >>> 3) {
                                        case 1:
                                            message.dataType = reader.int32();
                                            break;
                                        case 2:
                                            message.subType = reader.string();
                                            break;
                                        case 3:
                                            message.int32Value = reader.int32();
                                            break;
                                        case 4:
                                            message.int64Value = reader.int64();
                                            break;
                                        case 5:
                                            message.floatValue = reader.float();
                                            break;
                                        case 6:
                                            message.doubleValue = reader.double();
                                            break;
                                        case 7:
                                            message.stringValue = reader.string();
                                            break;
                                        case 8:
                                            message.boolValue = reader.bool();
                                            break;
                                        case 9:
                                            message.regularizerValue = $root.com.intel.analytics.bigdl.serialization.Regularizer.decode(reader, reader.uint32());
                                            break;
                                        case 10:
                                            message.tensorValue = $root.com.intel.analytics.bigdl.serialization.BigDLTensor.decode(reader, reader.uint32());
                                            break;
                                        case 11:
                                            message.variableFormatValue = reader.int32();
                                            break;
                                        case 12:
                                            message.initMethodValue = $root.com.intel.analytics.bigdl.serialization.InitMethod.decode(reader, reader.uint32());
                                            break;
                                        case 13:
                                            message.bigDLModuleValue = $root.com.intel.analytics.bigdl.serialization.BigDLModule.decode(reader, reader.uint32());
                                            break;
                                        case 14:
                                            message.nameAttrListValue = $root.com.intel.analytics.bigdl.serialization.NameAttrList.decode(reader, reader.uint32());
                                            break;
                                        case 15:
                                            message.arrayValue = $root.com.intel.analytics.bigdl.serialization.AttrValue.ArrayValue.decode(reader, reader.uint32());
                                            break;
                                        case 16:
                                            message.dataFormatValue = reader.int32();
                                            break;
                                        case 17:
                                            message.customValue = $root.google.protobuf.Any.decode(reader, reader.uint32());
                                            break;
                                        case 18:
                                            message.shape = $root.com.intel.analytics.bigdl.serialization.Shape.decode(reader, reader.uint32());
                                            break;
                                        default:
                                            reader.skipType(tag & 7);
                                            break;
                                    }
                                }
                                return message;
                            };

                            AttrValue.ArrayValue = (function() {

                                function ArrayValue() {
                                    this.i32 = [];
                                    this.i64 = [];
                                    this.flt = [];
                                    this.dbl = [];
                                    this.str = [];
                                    this.boolean = [];
                                    this.Regularizer = [];
                                    this.tensor = [];
                                    this.variableFormat = [];
                                    this.initMethod = [];
                                    this.bigDLModule = [];
                                    this.nameAttrList = [];
                                    this.dataFormat = [];
                                    this.custom = [];
                                    this.shape = [];
                                }

                                ArrayValue.prototype.size = 0;
                                ArrayValue.prototype.datatype = 0;
                                ArrayValue.prototype.i32 = [];
                                ArrayValue.prototype.i64 = [];
                                ArrayValue.prototype.flt = [];
                                ArrayValue.prototype.dbl = [];
                                ArrayValue.prototype.str = [];
                                ArrayValue.prototype.boolean = [];
                                ArrayValue.prototype.Regularizer = [];
                                ArrayValue.prototype.tensor = [];
                                ArrayValue.prototype.variableFormat = [];
                                ArrayValue.prototype.initMethod = [];
                                ArrayValue.prototype.bigDLModule = [];
                                ArrayValue.prototype.nameAttrList = [];
                                ArrayValue.prototype.dataFormat = [];
                                ArrayValue.prototype.custom = [];
                                ArrayValue.prototype.shape = [];

                                ArrayValue.decode = function (reader, length) {
                                    const message = new $root.com.intel.analytics.bigdl.serialization.AttrValue.ArrayValue();
                                    const end = reader.next(length);
                                    while (reader.end(end)) {
                                        const tag = reader.uint32();
                                        switch (tag >>> 3) {
                                            case 1:
                                                message.size = reader.int32();
                                                break;
                                            case 2:
                                                message.datatype = reader.int32();
                                                break;
                                            case 3:
                                                message.i32 = reader.array(message.i32, () => reader.int32(), tag);
                                                break;
                                            case 4:
                                                message.i64 = reader.array(message.i64, () => reader.int64(), tag);
                                                break;
                                            case 5:
                                                message.flt = reader.floats(message.flt, tag);
                                                break;
                                            case 6:
                                                message.dbl = reader.doubles(message.dbl, tag);
                                                break;
                                            case 7:
                                                message.str.push(reader.string());
                                                break;
                                            case 8:
                                                message.boolean = reader.array(message.boolean, () => reader.bool(), tag);
                                                break;
                                            case 9:
                                                message.Regularizer.push($root.com.intel.analytics.bigdl.serialization.Regularizer.decode(reader, reader.uint32()));
                                                break;
                                            case 10:
                                                message.tensor.push($root.com.intel.analytics.bigdl.serialization.BigDLTensor.decode(reader, reader.uint32()));
                                                break;
                                            case 11:
                                                message.variableFormat = reader.array(message.variableFormat, () => reader.int32(), tag);
                                                break;
                                            case 12:
                                                message.initMethod.push($root.com.intel.analytics.bigdl.serialization.InitMethod.decode(reader, reader.uint32()));
                                                break;
                                            case 13:
                                                message.bigDLModule.push($root.com.intel.analytics.bigdl.serialization.BigDLModule.decode(reader, reader.uint32()));
                                                break;
                                            case 14:
                                                message.nameAttrList.push($root.com.intel.analytics.bigdl.serialization.NameAttrList.decode(reader, reader.uint32()));
                                                break;
                                            case 15:
                                                message.dataFormat = reader.array(message.dataFormat, () => reader.int32(), tag);
                                                break;
                                            case 16:
                                                message.custom.push($root.google.protobuf.Any.decode(reader, reader.uint32()));
                                                break;
                                            case 17:
                                                message.shape.push($root.com.intel.analytics.bigdl.serialization.Shape.decode(reader, reader.uint32()));
                                                break;
                                            default:
                                                reader.skipType(tag & 7);
                                                break;
                                        }
                                    }
                                    return message;
                                };

                                return ArrayValue;
                            })();

                            return AttrValue;
                        })();

                        serialization.NameAttrList = (function() {

                            function NameAttrList() {
                                this.attr = {};
                            }

                            NameAttrList.prototype.name = "";
                            NameAttrList.prototype.attr = {};

                            NameAttrList.decode = function (reader, length) {
                                const message = new $root.com.intel.analytics.bigdl.serialization.NameAttrList();
                                const end = reader.next(length);
                                while (reader.end(end)) {
                                    const tag = reader.uint32();
                                    switch (tag >>> 3) {
                                        case 1:
                                            message.name = reader.string();
                                            break;
                                        case 2:
                                            reader.pair(message.attr, () => reader.string(), () => $root.com.intel.analytics.bigdl.serialization.AttrValue.decode(reader, reader.uint32()));
                                            break;
                                        default:
                                            reader.skipType(tag & 7);
                                            break;
                                    }
                                }
                                return message;
                            };

                            return NameAttrList;
                        })();

                        serialization.Shape = (function() {

                            function Shape() {
                                this.shapeValue = [];
                                this.shape = [];
                            }

                            Shape.prototype.shapeType = 0;
                            Shape.prototype.ssize = 0;
                            Shape.prototype.shapeValue = [];
                            Shape.prototype.shape = [];

                            Shape.decode = function (reader, length) {
                                const message = new $root.com.intel.analytics.bigdl.serialization.Shape();
                                const end = reader.next(length);
                                while (reader.end(end)) {
                                    const tag = reader.uint32();
                                    switch (tag >>> 3) {
                                        case 1:
                                            message.shapeType = reader.int32();
                                            break;
                                        case 2:
                                            message.ssize = reader.int32();
                                            break;
                                        case 3:
                                            message.shapeValue = reader.array(message.shapeValue, () => reader.int32(), tag);
                                            break;
                                        case 4:
                                            message.shape.push($root.com.intel.analytics.bigdl.serialization.Shape.decode(reader, reader.uint32()));
                                            break;
                                        default:
                                            reader.skipType(tag & 7);
                                            break;
                                    }
                                }
                                return message;
                            };

                            Shape.ShapeType = (function() {
                                const values = {};
                                values["SINGLE"] = 0;
                                values["MULTI"] = 1;
                                return values;
                            })();

                            return Shape;
                        })();

                        return serialization;
                    })();

                    return bigdl;
                })();

                return analytics;
            })();

            return intel;
        })();

        return com;
    })();

    $root.google = (function() {

        const google = {};

        google.protobuf = (function() {

            const protobuf = {};

            protobuf.Any = (function() {

                function Any() {
                }

                Any.prototype.type_url = "";
                Any.prototype.value = new Uint8Array([]);

                Any.decode = function (reader, length) {
                    const message = new $root.google.protobuf.Any();
                    const end = reader.next(length);
                    while (reader.end(end)) {
                        const tag = reader.uint32();
                        switch (tag >>> 3) {
                            case 1:
                                message.type_url = reader.string();
                                break;
                            case 2:
                                message.value = reader.bytes();
                                break;
                            default:
                                reader.skipType(tag & 7);
                                break;
                        }
                    }
                    return message;
                };

                return Any;
            })();

            return protobuf;
        })();

        return google;
    })();
    return $root;
})(protobuf);
