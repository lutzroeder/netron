var $root = protobuf.get('cntk');

$root.CNTK = {};

$root.CNTK.proto = {};

$root.CNTK.proto.NDShape = class NDShape {

    constructor() {
        this.shape_dim = [];
    }

    static decode(reader, length) {
        const message = new $root.CNTK.proto.NDShape();
        const end = length !== undefined ? reader.position + length : reader.length;
        while (reader.position < end) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.shape_dim = reader.array(message.shape_dim, () => reader.uint64(), tag);
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }
};

$root.CNTK.proto.Axis = class Axis {

    constructor() {
    }

    static decode(reader, length) {
        const message = new $root.CNTK.proto.Axis();
        const end = length !== undefined ? reader.position + length : reader.length;
        while (reader.position < end) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.static_axis_idx = reader.int32();
                    break;
                case 2:
                    message.name = reader.string();
                    break;
                case 3:
                    message.is_ordered_dynamic_axis = reader.bool();
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }
};

$root.CNTK.proto.Axis.prototype.static_axis_idx = 0;
$root.CNTK.proto.Axis.prototype.name = "";
$root.CNTK.proto.Axis.prototype.is_ordered_dynamic_axis = false;

$root.CNTK.proto.NDArrayView = class NDArrayView {

    constructor() {
    }

    get values() {
        $root.CNTK.proto.NDArrayView.valuesSet = $root.CNTK.proto.NDArrayView.valuesSet || new Set([ "float_values", "double_values", "bytes_value", "sint32_values"]);
        return Object.keys(this).find((key) => $root.CNTK.proto.NDArrayView.valuesSet.has(key) && this[key] != null);
    }

    static decode(reader, length) {
        const message = new $root.CNTK.proto.NDArrayView();
        const end = length !== undefined ? reader.position + length : reader.length;
        while (reader.position < end) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.data_type = reader.int32();
                    break;
                case 2:
                    message.storage_format = reader.int32();
                    break;
                case 3:
                    message.shape = $root.CNTK.proto.NDShape.decode(reader, reader.uint32());
                    break;
                case 4:
                    message.float_values = $root.CNTK.proto.NDArrayView.FloatValues.decode(reader, reader.uint32());
                    break;
                case 5:
                    message.double_values = $root.CNTK.proto.NDArrayView.DoubleValues.decode(reader, reader.uint32());
                    break;
                case 6:
                    message.bytes_value = $root.CNTK.proto.NDArrayView.BytesValue.decode(reader, reader.uint32());
                    break;
                case 7:
                    message.sint32_values = $root.CNTK.proto.NDArrayView.IntValues.decode(reader, reader.uint32());
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }
};

$root.CNTK.proto.NDArrayView.prototype.data_type = 0;
$root.CNTK.proto.NDArrayView.prototype.storage_format = 0;
$root.CNTK.proto.NDArrayView.prototype.shape = null;

$root.CNTK.proto.NDArrayView.DataType = {
    "Unknown": 0,
    "Float": 1,
    "Double": 2,
    "Float16": 4,
    "Int8": 5,
    "Int16": 6
};

$root.CNTK.proto.NDArrayView.StorageFormat = {
    "Dense": 0,
    "SparseCSC": 1,
    "SparseBlockCol": 2
};

$root.CNTK.proto.NDArrayView.FloatValues = class FloatValues {

    constructor() {
        this.value = [];
    }

    static decode(reader, length) {
        const message = new $root.CNTK.proto.NDArrayView.FloatValues();
        const end = length !== undefined ? reader.position + length : reader.length;
        while (reader.position < end) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.value = reader.floats(message.value, tag);
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }
};

$root.CNTK.proto.NDArrayView.DoubleValues = class DoubleValues {

    constructor() {
        this.value = [];
    }

    static decode(reader, length) {
        const message = new $root.CNTK.proto.NDArrayView.DoubleValues();
        const end = length !== undefined ? reader.position + length : reader.length;
        while (reader.position < end) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.value = reader.doubles(message.value, tag);
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }
};

$root.CNTK.proto.NDArrayView.BytesValue = class BytesValue {

    constructor() {
    }

    static decode(reader, length) {
        const message = new $root.CNTK.proto.NDArrayView.BytesValue();
        const end = length !== undefined ? reader.position + length : reader.length;
        while (reader.position < end) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
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

$root.CNTK.proto.NDArrayView.BytesValue.prototype.value = new Uint8Array([]);

$root.CNTK.proto.NDArrayView.IntValues = class IntValues {

    constructor() {
        this.value = [];
    }

    static decode(reader, length) {
        const message = new $root.CNTK.proto.NDArrayView.IntValues();
        const end = length !== undefined ? reader.position + length : reader.length;
        while (reader.position < end) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.value = reader.array(message.value, () => reader.sint32(), tag);
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }
};

$root.CNTK.proto.Vector = class Vector {

    constructor() {
        this.value = [];
    }

    static decode(reader, length) {
        const message = new $root.CNTK.proto.Vector();
        const end = length !== undefined ? reader.position + length : reader.length;
        while (reader.position < end) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.value.push($root.CNTK.proto.DictionaryValue.decode(reader, reader.uint32()));
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }
};

$root.CNTK.proto.Dictionary = class Dictionary {

    constructor() {
        this.data = {};
    }

    static decode(reader, length) {
        const message = new $root.CNTK.proto.Dictionary();
        const end = length !== undefined ? reader.position + length : reader.length;
        while (reader.position < end) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.version = reader.uint64();
                    break;
                case 2:
                    reader.entry(message.data, () => reader.string(), () => $root.CNTK.proto.DictionaryValue.decode(reader, reader.uint32()));
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }
};

$root.CNTK.proto.Dictionary.prototype.version = protobuf.Uint64.create(0);

$root.CNTK.proto.DictionaryValue = class DictionaryValue {

    constructor() {
    }

    get value() {
        $root.CNTK.proto.DictionaryValue.valueSet = $root.CNTK.proto.DictionaryValue.valueSet || new Set([ "bool_value", "int_value", "size_t_value", "float_value", "double_value", "string_value", "nd_shape_value", "axis_value", "vector_value", "dictionary_value", "nd_array_view_value"]);
        return Object.keys(this).find((key) => $root.CNTK.proto.DictionaryValue.valueSet.has(key) && this[key] != null);
    }

    static decode(reader, length) {
        const message = new $root.CNTK.proto.DictionaryValue();
        const end = length !== undefined ? reader.position + length : reader.length;
        while (reader.position < end) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.version = reader.uint64();
                    break;
                case 2:
                    message.value_type = reader.int32();
                    break;
                case 3:
                    message.bool_value = reader.bool();
                    break;
                case 4:
                    message.int_value = reader.int32();
                    break;
                case 5:
                    message.size_t_value = reader.uint64();
                    break;
                case 6:
                    message.float_value = reader.float();
                    break;
                case 7:
                    message.double_value = reader.double();
                    break;
                case 8:
                    message.string_value = reader.string();
                    break;
                case 9:
                    message.nd_shape_value = $root.CNTK.proto.NDShape.decode(reader, reader.uint32());
                    break;
                case 10:
                    message.axis_value = $root.CNTK.proto.Axis.decode(reader, reader.uint32());
                    break;
                case 11:
                    message.vector_value = $root.CNTK.proto.Vector.decode(reader, reader.uint32());
                    break;
                case 12:
                    message.dictionary_value = $root.CNTK.proto.Dictionary.decode(reader, reader.uint32());
                    break;
                case 13:
                    message.nd_array_view_value = $root.CNTK.proto.NDArrayView.decode(reader, reader.uint32());
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }
};

$root.CNTK.proto.DictionaryValue.prototype.version = protobuf.Uint64.create(0);
$root.CNTK.proto.DictionaryValue.prototype.value_type = 0;

$root.CNTK.proto.DictionaryValue.Type = {
    "None": 0,
    "Bool": 1,
    "Int": 2,
    "SizeT": 3,
    "Float": 4,
    "Double": 5,
    "String": 6,
    "NDShape": 7,
    "Axis": 8,
    "Vector": 9,
    "Dictionary": 10,
    "NDArrayView": 11
};
