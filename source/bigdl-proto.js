var $root = protobuf.get('bigdl');

$root.com = {};

$root.com.intel = {};

$root.com.intel.analytics = {};

$root.com.intel.analytics.bigdl = {};

$root.com.intel.analytics.bigdl.serialization = {};

$root.com.intel.analytics.bigdl.serialization.BigDLModule = class BigDLModule {

    constructor() {
        this.subModules = [];
        this.preModules = [];
        this.nextModules = [];
        this.attr = {};
        this.parameters = [];
        this.inputScales = [];
        this.outputScales = [];
        this.weightScales = [];
    }

    static decode(reader, length) {
        const message = new $root.com.intel.analytics.bigdl.serialization.BigDLModule();
        const end = length !== undefined ? reader.position + length : reader.length;
        while (reader.position < end) {
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
                    reader.entry(message.attr, () => reader.string(), () => $root.com.intel.analytics.bigdl.serialization.AttrValue.decode(reader, reader.uint32()));
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
    }
};

$root.com.intel.analytics.bigdl.serialization.BigDLModule.prototype.name = "";
$root.com.intel.analytics.bigdl.serialization.BigDLModule.prototype.weight = null;
$root.com.intel.analytics.bigdl.serialization.BigDLModule.prototype.bias = null;
$root.com.intel.analytics.bigdl.serialization.BigDLModule.prototype.moduleType = "";
$root.com.intel.analytics.bigdl.serialization.BigDLModule.prototype.version = "";
$root.com.intel.analytics.bigdl.serialization.BigDLModule.prototype.train = false;
$root.com.intel.analytics.bigdl.serialization.BigDLModule.prototype.namePostfix = "";
$root.com.intel.analytics.bigdl.serialization.BigDLModule.prototype.id = 0;
$root.com.intel.analytics.bigdl.serialization.BigDLModule.prototype.inputShape = null;
$root.com.intel.analytics.bigdl.serialization.BigDLModule.prototype.outputShape = null;
$root.com.intel.analytics.bigdl.serialization.BigDLModule.prototype.hasParameters = false;
$root.com.intel.analytics.bigdl.serialization.BigDLModule.prototype.isMklInt8Enabled = false;
$root.com.intel.analytics.bigdl.serialization.BigDLModule.prototype.inputDimMasks = 0;
$root.com.intel.analytics.bigdl.serialization.BigDLModule.prototype.outputDimMasks = 0;
$root.com.intel.analytics.bigdl.serialization.BigDLModule.prototype.weightDimMasks = 0;

$root.com.intel.analytics.bigdl.serialization.VarFormat = {
    "EMPTY_FORMAT": 0,
    "DEFAULT": 1,
    "ONE_D": 2,
    "IN_OUT": 3,
    "OUT_IN": 4,
    "IN_OUT_KW_KH": 5,
    "OUT_IN_KW_KH": 6,
    "GP_OUT_IN_KW_KH": 7,
    "GP_IN_OUT_KW_KH": 8,
    "OUT_IN_KT_KH_KW": 9
};

$root.com.intel.analytics.bigdl.serialization.InitMethodType = {
    "EMPTY_INITIALIZATION": 0,
    "RANDOM_UNIFORM": 1,
    "RANDOM_UNIFORM_PARAM": 2,
    "RANDOM_NORMAL": 3,
    "ZEROS": 4,
    "ONES": 5,
    "CONST": 6,
    "XAVIER": 7,
    "BILINEARFILLER": 8
};

$root.com.intel.analytics.bigdl.serialization.RegularizerType = {
    "L1L2Regularizer": 0,
    "L1Regularizer": 1,
    "L2Regularizer": 2
};

$root.com.intel.analytics.bigdl.serialization.InputDataFormat = {
    "NCHW": 0,
    "NHWC": 1
};

$root.com.intel.analytics.bigdl.serialization.TensorType = {
    "DENSE": 0,
    "QUANT": 1
};

$root.com.intel.analytics.bigdl.serialization.InitMethod = class InitMethod {

    constructor() {
        this.data = [];
    }

    static decode(reader, length) {
        const message = new $root.com.intel.analytics.bigdl.serialization.InitMethod();
        const end = length !== undefined ? reader.position + length : reader.length;
        while (reader.position < end) {
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
    }
};

$root.com.intel.analytics.bigdl.serialization.InitMethod.prototype.methodType = 0;

$root.com.intel.analytics.bigdl.serialization.BigDLTensor = class BigDLTensor {

    constructor() {
        this.size = [];
        this.stride = [];
    }

    static decode(reader, length) {
        const message = new $root.com.intel.analytics.bigdl.serialization.BigDLTensor();
        const end = length !== undefined ? reader.position + length : reader.length;
        while (reader.position < end) {
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
    }
};

$root.com.intel.analytics.bigdl.serialization.BigDLTensor.prototype.datatype = 0;
$root.com.intel.analytics.bigdl.serialization.BigDLTensor.prototype.offset = 0;
$root.com.intel.analytics.bigdl.serialization.BigDLTensor.prototype.dimension = 0;
$root.com.intel.analytics.bigdl.serialization.BigDLTensor.prototype.nElements = 0;
$root.com.intel.analytics.bigdl.serialization.BigDLTensor.prototype.isScalar = false;
$root.com.intel.analytics.bigdl.serialization.BigDLTensor.prototype.storage = null;
$root.com.intel.analytics.bigdl.serialization.BigDLTensor.prototype.id = 0;
$root.com.intel.analytics.bigdl.serialization.BigDLTensor.prototype.tensorType = 0;

$root.com.intel.analytics.bigdl.serialization.TensorStorage = class TensorStorage {

    constructor() {
        this.float_data = [];
        this.double_data = [];
        this.bool_data = [];
        this.string_data = [];
        this.int_data = [];
        this.long_data = [];
        this.bytes_data = [];
    }

    static decode(reader, length) {
        const message = new $root.com.intel.analytics.bigdl.serialization.TensorStorage();
        const end = length !== undefined ? reader.position + length : reader.length;
        while (reader.position < end) {
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
    }
};

$root.com.intel.analytics.bigdl.serialization.TensorStorage.prototype.datatype = 0;
$root.com.intel.analytics.bigdl.serialization.TensorStorage.prototype.id = 0;

$root.com.intel.analytics.bigdl.serialization.Regularizer = class Regularizer {

    constructor() {
        this.regularData = [];
    }

    static decode(reader, length) {
        const message = new $root.com.intel.analytics.bigdl.serialization.Regularizer();
        const end = length !== undefined ? reader.position + length : reader.length;
        while (reader.position < end) {
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
    }
};

$root.com.intel.analytics.bigdl.serialization.Regularizer.prototype.regularizerType = 0;

$root.com.intel.analytics.bigdl.serialization.DataType = {
    "INT32": 0,
    "INT64": 1,
    "FLOAT": 2,
    "DOUBLE": 3,
    "STRING": 4,
    "BOOL": 5,
    "CHAR": 6,
    "SHORT": 7,
    "BYTES": 8,
    "REGULARIZER": 9,
    "TENSOR": 10,
    "VARIABLE_FORMAT": 11,
    "INITMETHOD": 12,
    "MODULE": 13,
    "NAME_ATTR_LIST": 14,
    "ARRAY_VALUE": 15,
    "DATA_FORMAT": 16,
    "CUSTOM": 17,
    "SHAPE": 18
};

$root.com.intel.analytics.bigdl.serialization.AttrValue = class AttrValue {

    constructor() {
    }

    get value() {
        $root.com.intel.analytics.bigdl.serialization.AttrValue.valueSet = $root.com.intel.analytics.bigdl.serialization.AttrValue.valueSet || new Set([ "int32Value", "int64Value", "floatValue", "doubleValue", "stringValue", "boolValue", "regularizerValue", "tensorValue", "variableFormatValue", "initMethodValue", "bigDLModuleValue", "nameAttrListValue", "arrayValue", "dataFormatValue", "customValue", "shape"]);
        return Object.keys(this).find((key) => $root.com.intel.analytics.bigdl.serialization.AttrValue.valueSet.has(key) && this[key] != null);
    }

    static decode(reader, length) {
        const message = new $root.com.intel.analytics.bigdl.serialization.AttrValue();
        const end = length !== undefined ? reader.position + length : reader.length;
        while (reader.position < end) {
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
    }
};

$root.com.intel.analytics.bigdl.serialization.AttrValue.prototype.dataType = 0;
$root.com.intel.analytics.bigdl.serialization.AttrValue.prototype.subType = "";

$root.com.intel.analytics.bigdl.serialization.AttrValue.ArrayValue = class ArrayValue {

    constructor() {
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

    static decode(reader, length) {
        const message = new $root.com.intel.analytics.bigdl.serialization.AttrValue.ArrayValue();
        const end = length !== undefined ? reader.position + length : reader.length;
        while (reader.position < end) {
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
    }
};

$root.com.intel.analytics.bigdl.serialization.AttrValue.ArrayValue.prototype.size = 0;
$root.com.intel.analytics.bigdl.serialization.AttrValue.ArrayValue.prototype.datatype = 0;

$root.com.intel.analytics.bigdl.serialization.NameAttrList = class NameAttrList {

    constructor() {
        this.attr = {};
    }

    static decode(reader, length) {
        const message = new $root.com.intel.analytics.bigdl.serialization.NameAttrList();
        const end = length !== undefined ? reader.position + length : reader.length;
        while (reader.position < end) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.name = reader.string();
                    break;
                case 2:
                    reader.entry(message.attr, () => reader.string(), () => $root.com.intel.analytics.bigdl.serialization.AttrValue.decode(reader, reader.uint32()));
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }
};

$root.com.intel.analytics.bigdl.serialization.NameAttrList.prototype.name = "";

$root.com.intel.analytics.bigdl.serialization.Shape = class Shape {

    constructor() {
        this.shapeValue = [];
        this.shape = [];
    }

    static decode(reader, length) {
        const message = new $root.com.intel.analytics.bigdl.serialization.Shape();
        const end = length !== undefined ? reader.position + length : reader.length;
        while (reader.position < end) {
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
    }
};

$root.com.intel.analytics.bigdl.serialization.Shape.prototype.shapeType = 0;
$root.com.intel.analytics.bigdl.serialization.Shape.prototype.ssize = 0;

$root.com.intel.analytics.bigdl.serialization.Shape.ShapeType = {
    "SINGLE": 0,
    "MULTI": 1
};

$root.google = {};

$root.google.protobuf = {};

$root.google.protobuf.Any = class Any {

    constructor() {
    }

    static decode(reader, length) {
        const message = new $root.google.protobuf.Any();
        const end = length !== undefined ? reader.position + length : reader.length;
        while (reader.position < end) {
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
    }
};

$root.google.protobuf.Any.prototype.type_url = "";
$root.google.protobuf.Any.prototype.value = new Uint8Array([]);
