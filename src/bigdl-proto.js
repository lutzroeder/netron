/*eslint-disable block-scoped-var, id-length, no-control-regex, no-magic-numbers, no-prototype-builtins, no-redeclare, no-shadow, no-var, sort-vars*/
(function($protobuf) {
    "use strict";

    var $Reader = $protobuf.Reader, $util = $protobuf.util;
    
    var $root = $protobuf.roots.bigdl || ($protobuf.roots.bigdl = {});
    
    $root.com = (function() {
    
        var com = {};
    
        com.intel = (function() {
    
            var intel = {};
    
            intel.analytics = (function() {
    
                var analytics = {};
    
                analytics.bigdl = (function() {
    
                    var bigdl = {};
    
                    bigdl.serialization = (function() {
    
                        var serialization = {};
    
                        serialization.BigDLModule = (function() {
    
                            function BigDLModule(properties) {
                                this.subModules = [];
                                this.preModules = [];
                                this.nextModules = [];
                                this.attr = {};
                                this.parameters = [];
                                this.inputScales = [];
                                this.outputScales = [];
                                this.weightScales = [];
                                if (properties)
                                    for (var keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                                        if (properties[keys[i]] != null)
                                            this[keys[i]] = properties[keys[i]];
                            }
    
                            BigDLModule.prototype.name = "";
                            BigDLModule.prototype.subModules = $util.emptyArray;
                            BigDLModule.prototype.weight = null;
                            BigDLModule.prototype.bias = null;
                            BigDLModule.prototype.preModules = $util.emptyArray;
                            BigDLModule.prototype.nextModules = $util.emptyArray;
                            BigDLModule.prototype.moduleType = "";
                            BigDLModule.prototype.attr = $util.emptyObject;
                            BigDLModule.prototype.version = "";
                            BigDLModule.prototype.train = false;
                            BigDLModule.prototype.namePostfix = "";
                            BigDLModule.prototype.id = 0;
                            BigDLModule.prototype.inputShape = null;
                            BigDLModule.prototype.outputShape = null;
                            BigDLModule.prototype.hasParameters = false;
                            BigDLModule.prototype.parameters = $util.emptyArray;
                            BigDLModule.prototype.isMklInt8Enabled = false;
                            BigDLModule.prototype.inputDimMasks = 0;
                            BigDLModule.prototype.inputScales = $util.emptyArray;
                            BigDLModule.prototype.outputDimMasks = 0;
                            BigDLModule.prototype.outputScales = $util.emptyArray;
                            BigDLModule.prototype.weightDimMasks = 0;
                            BigDLModule.prototype.weightScales = $util.emptyArray;
    
                            BigDLModule.decode = function decode(reader, length) {
                                if (!(reader instanceof $Reader))
                                    reader = $Reader.create(reader);
                                var end = length === undefined ? reader.len : reader.pos + length, message = new $root.com.intel.analytics.bigdl.serialization.BigDLModule(), key;
                                while (reader.pos < end) {
                                    var tag = reader.uint32();
                                    switch (tag >>> 3) {
                                    case 1:
                                        message.name = reader.string();
                                        break;
                                    case 2:
                                        if (!(message.subModules && message.subModules.length))
                                            message.subModules = [];
                                        message.subModules.push($root.com.intel.analytics.bigdl.serialization.BigDLModule.decode(reader, reader.uint32()));
                                        break;
                                    case 3:
                                        message.weight = $root.com.intel.analytics.bigdl.serialization.BigDLTensor.decode(reader, reader.uint32());
                                        break;
                                    case 4:
                                        message.bias = $root.com.intel.analytics.bigdl.serialization.BigDLTensor.decode(reader, reader.uint32());
                                        break;
                                    case 5:
                                        if (!(message.preModules && message.preModules.length))
                                            message.preModules = [];
                                        message.preModules.push(reader.string());
                                        break;
                                    case 6:
                                        if (!(message.nextModules && message.nextModules.length))
                                            message.nextModules = [];
                                        message.nextModules.push(reader.string());
                                        break;
                                    case 7:
                                        message.moduleType = reader.string();
                                        break;
                                    case 8:
                                        reader.skip().pos++;
                                        if (message.attr === $util.emptyObject)
                                            message.attr = {};
                                        key = reader.string();
                                        reader.pos++;
                                        message.attr[key] = $root.com.intel.analytics.bigdl.serialization.AttrValue.decode(reader, reader.uint32());
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
                                        if (!(message.parameters && message.parameters.length))
                                            message.parameters = [];
                                        message.parameters.push($root.com.intel.analytics.bigdl.serialization.BigDLTensor.decode(reader, reader.uint32()));
                                        break;
                                    case 17:
                                        message.isMklInt8Enabled = reader.bool();
                                        break;
                                    case 18:
                                        message.inputDimMasks = reader.int32();
                                        break;
                                    case 19:
                                        if (!(message.inputScales && message.inputScales.length))
                                            message.inputScales = [];
                                        message.inputScales.push($root.com.intel.analytics.bigdl.serialization.AttrValue.decode(reader, reader.uint32()));
                                        break;
                                    case 20:
                                        message.outputDimMasks = reader.int32();
                                        break;
                                    case 21:
                                        if (!(message.outputScales && message.outputScales.length))
                                            message.outputScales = [];
                                        message.outputScales.push($root.com.intel.analytics.bigdl.serialization.AttrValue.decode(reader, reader.uint32()));
                                        break;
                                    case 22:
                                        message.weightDimMasks = reader.int32();
                                        break;
                                    case 23:
                                        if (!(message.weightScales && message.weightScales.length))
                                            message.weightScales = [];
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
                            var valuesById = {}, values = Object.create(valuesById);
                            values[valuesById[0] = "EMPTY_FORMAT"] = 0;
                            values[valuesById[1] = "DEFAULT"] = 1;
                            values[valuesById[2] = "ONE_D"] = 2;
                            values[valuesById[3] = "IN_OUT"] = 3;
                            values[valuesById[4] = "OUT_IN"] = 4;
                            values[valuesById[5] = "IN_OUT_KW_KH"] = 5;
                            values[valuesById[6] = "OUT_IN_KW_KH"] = 6;
                            values[valuesById[7] = "GP_OUT_IN_KW_KH"] = 7;
                            values[valuesById[8] = "GP_IN_OUT_KW_KH"] = 8;
                            values[valuesById[9] = "OUT_IN_KT_KH_KW"] = 9;
                            return values;
                        })();
    
                        serialization.InitMethodType = (function() {
                            var valuesById = {}, values = Object.create(valuesById);
                            values[valuesById[0] = "EMPTY_INITIALIZATION"] = 0;
                            values[valuesById[1] = "RANDOM_UNIFORM"] = 1;
                            values[valuesById[2] = "RANDOM_UNIFORM_PARAM"] = 2;
                            values[valuesById[3] = "RANDOM_NORMAL"] = 3;
                            values[valuesById[4] = "ZEROS"] = 4;
                            values[valuesById[5] = "ONES"] = 5;
                            values[valuesById[6] = "CONST"] = 6;
                            values[valuesById[7] = "XAVIER"] = 7;
                            values[valuesById[8] = "BILINEARFILLER"] = 8;
                            return values;
                        })();
    
                        serialization.RegularizerType = (function() {
                            var valuesById = {}, values = Object.create(valuesById);
                            values[valuesById[0] = "L1L2Regularizer"] = 0;
                            values[valuesById[1] = "L1Regularizer"] = 1;
                            values[valuesById[2] = "L2Regularizer"] = 2;
                            return values;
                        })();
    
                        serialization.InputDataFormat = (function() {
                            var valuesById = {}, values = Object.create(valuesById);
                            values[valuesById[0] = "NCHW"] = 0;
                            values[valuesById[1] = "NHWC"] = 1;
                            return values;
                        })();
    
                        serialization.TensorType = (function() {
                            var valuesById = {}, values = Object.create(valuesById);
                            values[valuesById[0] = "DENSE"] = 0;
                            values[valuesById[1] = "QUANT"] = 1;
                            return values;
                        })();
    
                        serialization.InitMethod = (function() {
    
                            function InitMethod(properties) {
                                this.data = [];
                                if (properties)
                                    for (var keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                                        if (properties[keys[i]] != null)
                                            this[keys[i]] = properties[keys[i]];
                            }
    
                            InitMethod.prototype.methodType = 0;
                            InitMethod.prototype.data = $util.emptyArray;
    
                            InitMethod.decode = function decode(reader, length) {
                                if (!(reader instanceof $Reader))
                                    reader = $Reader.create(reader);
                                var end = length === undefined ? reader.len : reader.pos + length, message = new $root.com.intel.analytics.bigdl.serialization.InitMethod();
                                while (reader.pos < end) {
                                    var tag = reader.uint32();
                                    switch (tag >>> 3) {
                                    case 1:
                                        message.methodType = reader.int32();
                                        break;
                                    case 2:
                                        if (!(message.data && message.data.length))
                                            message.data = [];
                                        if ((tag & 7) === 2) {
                                            var end2 = reader.uint32() + reader.pos;
                                            while (reader.pos < end2)
                                                message.data.push(reader.double());
                                        } else
                                            message.data.push(reader.double());
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
    
                            function BigDLTensor(properties) {
                                this.size = [];
                                this.stride = [];
                                if (properties)
                                    for (var keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                                        if (properties[keys[i]] != null)
                                            this[keys[i]] = properties[keys[i]];
                            }
    
                            BigDLTensor.prototype.datatype = 0;
                            BigDLTensor.prototype.size = $util.emptyArray;
                            BigDLTensor.prototype.stride = $util.emptyArray;
                            BigDLTensor.prototype.offset = 0;
                            BigDLTensor.prototype.dimension = 0;
                            BigDLTensor.prototype.nElements = 0;
                            BigDLTensor.prototype.isScalar = false;
                            BigDLTensor.prototype.storage = null;
                            BigDLTensor.prototype.id = 0;
                            BigDLTensor.prototype.tensorType = 0;
    
                            BigDLTensor.decode = function decode(reader, length) {
                                if (!(reader instanceof $Reader))
                                    reader = $Reader.create(reader);
                                var end = length === undefined ? reader.len : reader.pos + length, message = new $root.com.intel.analytics.bigdl.serialization.BigDLTensor();
                                while (reader.pos < end) {
                                    var tag = reader.uint32();
                                    switch (tag >>> 3) {
                                    case 1:
                                        message.datatype = reader.int32();
                                        break;
                                    case 2:
                                        if (!(message.size && message.size.length))
                                            message.size = [];
                                        if ((tag & 7) === 2) {
                                            var end2 = reader.uint32() + reader.pos;
                                            while (reader.pos < end2)
                                                message.size.push(reader.int32());
                                        } else
                                            message.size.push(reader.int32());
                                        break;
                                    case 3:
                                        if (!(message.stride && message.stride.length))
                                            message.stride = [];
                                        if ((tag & 7) === 2) {
                                            var end2 = reader.uint32() + reader.pos;
                                            while (reader.pos < end2)
                                                message.stride.push(reader.int32());
                                        } else
                                            message.stride.push(reader.int32());
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
    
                            function TensorStorage(properties) {
                                this.float_data = [];
                                this.double_data = [];
                                this.bool_data = [];
                                this.string_data = [];
                                this.int_data = [];
                                this.long_data = [];
                                this.bytes_data = [];
                                if (properties)
                                    for (var keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                                        if (properties[keys[i]] != null)
                                            this[keys[i]] = properties[keys[i]];
                            }
    
                            TensorStorage.prototype.datatype = 0;
                            TensorStorage.prototype.float_data = $util.emptyArray;
                            TensorStorage.prototype.double_data = $util.emptyArray;
                            TensorStorage.prototype.bool_data = $util.emptyArray;
                            TensorStorage.prototype.string_data = $util.emptyArray;
                            TensorStorage.prototype.int_data = $util.emptyArray;
                            TensorStorage.prototype.long_data = $util.emptyArray;
                            TensorStorage.prototype.bytes_data = $util.emptyArray;
                            TensorStorage.prototype.id = 0;
    
                            TensorStorage.decode = function decode(reader, length) {
                                if (!(reader instanceof $Reader))
                                    reader = $Reader.create(reader);
                                var end = length === undefined ? reader.len : reader.pos + length, message = new $root.com.intel.analytics.bigdl.serialization.TensorStorage();
                                while (reader.pos < end) {
                                    var tag = reader.uint32();
                                    switch (tag >>> 3) {
                                    case 1:
                                        message.datatype = reader.int32();
                                        break;
                                    case 2:
                                        if (!(message.float_data && message.float_data.length))
                                            message.float_data = [];
                                        if ((tag & 7) === 2) {
                                            var end2 = reader.uint32() + reader.pos;
                                            while (reader.pos < end2)
                                                message.float_data.push(reader.float());
                                        } else
                                            message.float_data.push(reader.float());
                                        break;
                                    case 3:
                                        if (!(message.double_data && message.double_data.length))
                                            message.double_data = [];
                                        if ((tag & 7) === 2) {
                                            var end2 = reader.uint32() + reader.pos;
                                            while (reader.pos < end2)
                                                message.double_data.push(reader.double());
                                        } else
                                            message.double_data.push(reader.double());
                                        break;
                                    case 4:
                                        if (!(message.bool_data && message.bool_data.length))
                                            message.bool_data = [];
                                        if ((tag & 7) === 2) {
                                            var end2 = reader.uint32() + reader.pos;
                                            while (reader.pos < end2)
                                                message.bool_data.push(reader.bool());
                                        } else
                                            message.bool_data.push(reader.bool());
                                        break;
                                    case 5:
                                        if (!(message.string_data && message.string_data.length))
                                            message.string_data = [];
                                        message.string_data.push(reader.string());
                                        break;
                                    case 6:
                                        if (!(message.int_data && message.int_data.length))
                                            message.int_data = [];
                                        if ((tag & 7) === 2) {
                                            var end2 = reader.uint32() + reader.pos;
                                            while (reader.pos < end2)
                                                message.int_data.push(reader.int32());
                                        } else
                                            message.int_data.push(reader.int32());
                                        break;
                                    case 7:
                                        if (!(message.long_data && message.long_data.length))
                                            message.long_data = [];
                                        if ((tag & 7) === 2) {
                                            var end2 = reader.uint32() + reader.pos;
                                            while (reader.pos < end2)
                                                message.long_data.push(reader.int64());
                                        } else
                                            message.long_data.push(reader.int64());
                                        break;
                                    case 8:
                                        if (!(message.bytes_data && message.bytes_data.length))
                                            message.bytes_data = [];
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
    
                            function Regularizer(properties) {
                                this.regularData = [];
                                if (properties)
                                    for (var keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                                        if (properties[keys[i]] != null)
                                            this[keys[i]] = properties[keys[i]];
                            }
    
                            Regularizer.prototype.regularizerType = 0;
                            Regularizer.prototype.regularData = $util.emptyArray;
    
                            Regularizer.decode = function decode(reader, length) {
                                if (!(reader instanceof $Reader))
                                    reader = $Reader.create(reader);
                                var end = length === undefined ? reader.len : reader.pos + length, message = new $root.com.intel.analytics.bigdl.serialization.Regularizer();
                                while (reader.pos < end) {
                                    var tag = reader.uint32();
                                    switch (tag >>> 3) {
                                    case 1:
                                        message.regularizerType = reader.int32();
                                        break;
                                    case 2:
                                        if (!(message.regularData && message.regularData.length))
                                            message.regularData = [];
                                        if ((tag & 7) === 2) {
                                            var end2 = reader.uint32() + reader.pos;
                                            while (reader.pos < end2)
                                                message.regularData.push(reader.double());
                                        } else
                                            message.regularData.push(reader.double());
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
                            var valuesById = {}, values = Object.create(valuesById);
                            values[valuesById[0] = "INT32"] = 0;
                            values[valuesById[1] = "INT64"] = 1;
                            values[valuesById[2] = "FLOAT"] = 2;
                            values[valuesById[3] = "DOUBLE"] = 3;
                            values[valuesById[4] = "STRING"] = 4;
                            values[valuesById[5] = "BOOL"] = 5;
                            values[valuesById[6] = "CHAR"] = 6;
                            values[valuesById[7] = "SHORT"] = 7;
                            values[valuesById[8] = "BYTES"] = 8;
                            values[valuesById[9] = "REGULARIZER"] = 9;
                            values[valuesById[10] = "TENSOR"] = 10;
                            values[valuesById[11] = "VARIABLE_FORMAT"] = 11;
                            values[valuesById[12] = "INITMETHOD"] = 12;
                            values[valuesById[13] = "MODULE"] = 13;
                            values[valuesById[14] = "NAME_ATTR_LIST"] = 14;
                            values[valuesById[15] = "ARRAY_VALUE"] = 15;
                            values[valuesById[16] = "DATA_FORMAT"] = 16;
                            values[valuesById[17] = "CUSTOM"] = 17;
                            values[valuesById[18] = "SHAPE"] = 18;
                            return values;
                        })();
    
                        serialization.AttrValue = (function() {
    
                            function AttrValue(properties) {
                                if (properties)
                                    for (var keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                                        if (properties[keys[i]] != null)
                                            this[keys[i]] = properties[keys[i]];
                            }
    
                            AttrValue.prototype.dataType = 0;
                            AttrValue.prototype.subType = "";
                            AttrValue.prototype.int32Value = 0;
                            AttrValue.prototype.int64Value = $util.Long ? $util.Long.fromBits(0,0,false) : 0;
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
    
                            var $oneOfFields;
    
                            Object.defineProperty(AttrValue.prototype, "value", {
                                get: $util.oneOfGetter($oneOfFields = ["int32Value", "int64Value", "floatValue", "doubleValue", "stringValue", "boolValue", "regularizerValue", "tensorValue", "variableFormatValue", "initMethodValue", "bigDLModuleValue", "nameAttrListValue", "arrayValue", "dataFormatValue", "customValue", "shape"]),
                                set: $util.oneOfSetter($oneOfFields)
                            });
    
                            AttrValue.decode = function decode(reader, length) {
                                if (!(reader instanceof $Reader))
                                    reader = $Reader.create(reader);
                                var end = length === undefined ? reader.len : reader.pos + length, message = new $root.com.intel.analytics.bigdl.serialization.AttrValue();
                                while (reader.pos < end) {
                                    var tag = reader.uint32();
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
    
                                function ArrayValue(properties) {
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
                                    if (properties)
                                        for (var keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                                            if (properties[keys[i]] != null)
                                                this[keys[i]] = properties[keys[i]];
                                }
    
                                ArrayValue.prototype.size = 0;
                                ArrayValue.prototype.datatype = 0;
                                ArrayValue.prototype.i32 = $util.emptyArray;
                                ArrayValue.prototype.i64 = $util.emptyArray;
                                ArrayValue.prototype.flt = $util.emptyArray;
                                ArrayValue.prototype.dbl = $util.emptyArray;
                                ArrayValue.prototype.str = $util.emptyArray;
                                ArrayValue.prototype.boolean = $util.emptyArray;
                                ArrayValue.prototype.Regularizer = $util.emptyArray;
                                ArrayValue.prototype.tensor = $util.emptyArray;
                                ArrayValue.prototype.variableFormat = $util.emptyArray;
                                ArrayValue.prototype.initMethod = $util.emptyArray;
                                ArrayValue.prototype.bigDLModule = $util.emptyArray;
                                ArrayValue.prototype.nameAttrList = $util.emptyArray;
                                ArrayValue.prototype.dataFormat = $util.emptyArray;
                                ArrayValue.prototype.custom = $util.emptyArray;
                                ArrayValue.prototype.shape = $util.emptyArray;
    
                                ArrayValue.decode = function decode(reader, length) {
                                    if (!(reader instanceof $Reader))
                                        reader = $Reader.create(reader);
                                    var end = length === undefined ? reader.len : reader.pos + length, message = new $root.com.intel.analytics.bigdl.serialization.AttrValue.ArrayValue();
                                    while (reader.pos < end) {
                                        var tag = reader.uint32();
                                        switch (tag >>> 3) {
                                        case 1:
                                            message.size = reader.int32();
                                            break;
                                        case 2:
                                            message.datatype = reader.int32();
                                            break;
                                        case 3:
                                            if (!(message.i32 && message.i32.length))
                                                message.i32 = [];
                                            if ((tag & 7) === 2) {
                                                var end2 = reader.uint32() + reader.pos;
                                                while (reader.pos < end2)
                                                    message.i32.push(reader.int32());
                                            } else
                                                message.i32.push(reader.int32());
                                            break;
                                        case 4:
                                            if (!(message.i64 && message.i64.length))
                                                message.i64 = [];
                                            if ((tag & 7) === 2) {
                                                var end2 = reader.uint32() + reader.pos;
                                                while (reader.pos < end2)
                                                    message.i64.push(reader.int64());
                                            } else
                                                message.i64.push(reader.int64());
                                            break;
                                        case 5:
                                            if (!(message.flt && message.flt.length))
                                                message.flt = [];
                                            if ((tag & 7) === 2) {
                                                var end2 = reader.uint32() + reader.pos;
                                                while (reader.pos < end2)
                                                    message.flt.push(reader.float());
                                            } else
                                                message.flt.push(reader.float());
                                            break;
                                        case 6:
                                            if (!(message.dbl && message.dbl.length))
                                                message.dbl = [];
                                            if ((tag & 7) === 2) {
                                                var end2 = reader.uint32() + reader.pos;
                                                while (reader.pos < end2)
                                                    message.dbl.push(reader.double());
                                            } else
                                                message.dbl.push(reader.double());
                                            break;
                                        case 7:
                                            if (!(message.str && message.str.length))
                                                message.str = [];
                                            message.str.push(reader.string());
                                            break;
                                        case 8:
                                            if (!(message.boolean && message.boolean.length))
                                                message.boolean = [];
                                            if ((tag & 7) === 2) {
                                                var end2 = reader.uint32() + reader.pos;
                                                while (reader.pos < end2)
                                                    message.boolean.push(reader.bool());
                                            } else
                                                message.boolean.push(reader.bool());
                                            break;
                                        case 9:
                                            if (!(message.Regularizer && message.Regularizer.length))
                                                message.Regularizer = [];
                                            message.Regularizer.push($root.com.intel.analytics.bigdl.serialization.Regularizer.decode(reader, reader.uint32()));
                                            break;
                                        case 10:
                                            if (!(message.tensor && message.tensor.length))
                                                message.tensor = [];
                                            message.tensor.push($root.com.intel.analytics.bigdl.serialization.BigDLTensor.decode(reader, reader.uint32()));
                                            break;
                                        case 11:
                                            if (!(message.variableFormat && message.variableFormat.length))
                                                message.variableFormat = [];
                                            if ((tag & 7) === 2) {
                                                var end2 = reader.uint32() + reader.pos;
                                                while (reader.pos < end2)
                                                    message.variableFormat.push(reader.int32());
                                            } else
                                                message.variableFormat.push(reader.int32());
                                            break;
                                        case 12:
                                            if (!(message.initMethod && message.initMethod.length))
                                                message.initMethod = [];
                                            message.initMethod.push($root.com.intel.analytics.bigdl.serialization.InitMethod.decode(reader, reader.uint32()));
                                            break;
                                        case 13:
                                            if (!(message.bigDLModule && message.bigDLModule.length))
                                                message.bigDLModule = [];
                                            message.bigDLModule.push($root.com.intel.analytics.bigdl.serialization.BigDLModule.decode(reader, reader.uint32()));
                                            break;
                                        case 14:
                                            if (!(message.nameAttrList && message.nameAttrList.length))
                                                message.nameAttrList = [];
                                            message.nameAttrList.push($root.com.intel.analytics.bigdl.serialization.NameAttrList.decode(reader, reader.uint32()));
                                            break;
                                        case 15:
                                            if (!(message.dataFormat && message.dataFormat.length))
                                                message.dataFormat = [];
                                            if ((tag & 7) === 2) {
                                                var end2 = reader.uint32() + reader.pos;
                                                while (reader.pos < end2)
                                                    message.dataFormat.push(reader.int32());
                                            } else
                                                message.dataFormat.push(reader.int32());
                                            break;
                                        case 16:
                                            if (!(message.custom && message.custom.length))
                                                message.custom = [];
                                            message.custom.push($root.google.protobuf.Any.decode(reader, reader.uint32()));
                                            break;
                                        case 17:
                                            if (!(message.shape && message.shape.length))
                                                message.shape = [];
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
    
                            function NameAttrList(properties) {
                                this.attr = {};
                                if (properties)
                                    for (var keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                                        if (properties[keys[i]] != null)
                                            this[keys[i]] = properties[keys[i]];
                            }
    
                            NameAttrList.prototype.name = "";
                            NameAttrList.prototype.attr = $util.emptyObject;
    
                            NameAttrList.decode = function decode(reader, length) {
                                if (!(reader instanceof $Reader))
                                    reader = $Reader.create(reader);
                                var end = length === undefined ? reader.len : reader.pos + length, message = new $root.com.intel.analytics.bigdl.serialization.NameAttrList(), key;
                                while (reader.pos < end) {
                                    var tag = reader.uint32();
                                    switch (tag >>> 3) {
                                    case 1:
                                        message.name = reader.string();
                                        break;
                                    case 2:
                                        reader.skip().pos++;
                                        if (message.attr === $util.emptyObject)
                                            message.attr = {};
                                        key = reader.string();
                                        reader.pos++;
                                        message.attr[key] = $root.com.intel.analytics.bigdl.serialization.AttrValue.decode(reader, reader.uint32());
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
    
                            function Shape(properties) {
                                this.shapeValue = [];
                                this.shape = [];
                                if (properties)
                                    for (var keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                                        if (properties[keys[i]] != null)
                                            this[keys[i]] = properties[keys[i]];
                            }
    
                            Shape.prototype.shapeType = 0;
                            Shape.prototype.ssize = 0;
                            Shape.prototype.shapeValue = $util.emptyArray;
                            Shape.prototype.shape = $util.emptyArray;
    
                            Shape.decode = function decode(reader, length) {
                                if (!(reader instanceof $Reader))
                                    reader = $Reader.create(reader);
                                var end = length === undefined ? reader.len : reader.pos + length, message = new $root.com.intel.analytics.bigdl.serialization.Shape();
                                while (reader.pos < end) {
                                    var tag = reader.uint32();
                                    switch (tag >>> 3) {
                                    case 1:
                                        message.shapeType = reader.int32();
                                        break;
                                    case 2:
                                        message.ssize = reader.int32();
                                        break;
                                    case 3:
                                        if (!(message.shapeValue && message.shapeValue.length))
                                            message.shapeValue = [];
                                        if ((tag & 7) === 2) {
                                            var end2 = reader.uint32() + reader.pos;
                                            while (reader.pos < end2)
                                                message.shapeValue.push(reader.int32());
                                        } else
                                            message.shapeValue.push(reader.int32());
                                        break;
                                    case 4:
                                        if (!(message.shape && message.shape.length))
                                            message.shape = [];
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
                                var valuesById = {}, values = Object.create(valuesById);
                                values[valuesById[0] = "SINGLE"] = 0;
                                values[valuesById[1] = "MULTI"] = 1;
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
    
        var google = {};
    
        google.protobuf = (function() {
    
            var protobuf = {};
    
            protobuf.Any = (function() {
    
                function Any(properties) {
                    if (properties)
                        for (var keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                            if (properties[keys[i]] != null)
                                this[keys[i]] = properties[keys[i]];
                }
    
                Any.prototype.type_url = "";
                Any.prototype.value = $util.newBuffer([]);
    
                Any.decode = function decode(reader, length) {
                    if (!(reader instanceof $Reader))
                        reader = $Reader.create(reader);
                    var end = length === undefined ? reader.len : reader.pos + length, message = new $root.google.protobuf.Any();
                    while (reader.pos < end) {
                        var tag = reader.uint32();
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
