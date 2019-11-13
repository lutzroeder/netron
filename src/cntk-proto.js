/*eslint-disable block-scoped-var, id-length, no-control-regex, no-magic-numbers, no-prototype-builtins, no-redeclare, no-shadow, no-var, sort-vars*/
(function($protobuf) {
    "use strict";

    var $Reader = $protobuf.Reader, $util = $protobuf.util;
    
    var $root = $protobuf.roots.cntk || ($protobuf.roots.cntk = {});
    
    $root.CNTK = (function() {
    
        var CNTK = {};
    
        CNTK.proto = (function() {
    
            var proto = {};
    
            proto.NDShape = (function() {
    
                function NDShape(properties) {
                    this.shape_dim = [];
                    if (properties)
                        for (var keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                            if (properties[keys[i]] != null)
                                this[keys[i]] = properties[keys[i]];
                }
    
                NDShape.prototype.shape_dim = $util.emptyArray;
    
                NDShape.decode = function decode(reader, length) {
                    if (!(reader instanceof $Reader))
                        reader = $Reader.create(reader);
                    var end = length === undefined ? reader.len : reader.pos + length, message = new $root.CNTK.proto.NDShape();
                    while (reader.pos < end) {
                        var tag = reader.uint32();
                        switch (tag >>> 3) {
                        case 1:
                            if (!(message.shape_dim && message.shape_dim.length))
                                message.shape_dim = [];
                            if ((tag & 7) === 2) {
                                var end2 = reader.uint32() + reader.pos;
                                while (reader.pos < end2)
                                    message.shape_dim.push(reader.uint64());
                            } else
                                message.shape_dim.push(reader.uint64());
                            break;
                        default:
                            reader.skipType(tag & 7);
                            break;
                        }
                    }
                    return message;
                };
    
                return NDShape;
            })();
    
            proto.Axis = (function() {
    
                function Axis(properties) {
                    if (properties)
                        for (var keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                            if (properties[keys[i]] != null)
                                this[keys[i]] = properties[keys[i]];
                }
    
                Axis.prototype.static_axis_idx = 0;
                Axis.prototype.name = "";
                Axis.prototype.is_ordered_dynamic_axis = false;
    
                Axis.decode = function decode(reader, length) {
                    if (!(reader instanceof $Reader))
                        reader = $Reader.create(reader);
                    var end = length === undefined ? reader.len : reader.pos + length, message = new $root.CNTK.proto.Axis();
                    while (reader.pos < end) {
                        var tag = reader.uint32();
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
                };
    
                return Axis;
            })();
    
            proto.NDArrayView = (function() {
    
                function NDArrayView(properties) {
                    if (properties)
                        for (var keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                            if (properties[keys[i]] != null)
                                this[keys[i]] = properties[keys[i]];
                }
    
                NDArrayView.prototype.data_type = 0;
                NDArrayView.prototype.storage_format = 0;
                NDArrayView.prototype.shape = null;
                NDArrayView.prototype.float_values = null;
                NDArrayView.prototype.double_values = null;
                NDArrayView.prototype.bytes_value = null;
                NDArrayView.prototype.sint32_values = null;
    
                var $oneOfFields;
    
                Object.defineProperty(NDArrayView.prototype, "values", {
                    get: $util.oneOfGetter($oneOfFields = ["float_values", "double_values", "bytes_value", "sint32_values"]),
                    set: $util.oneOfSetter($oneOfFields)
                });
    
                NDArrayView.decode = function decode(reader, length) {
                    if (!(reader instanceof $Reader))
                        reader = $Reader.create(reader);
                    var end = length === undefined ? reader.len : reader.pos + length, message = new $root.CNTK.proto.NDArrayView();
                    while (reader.pos < end) {
                        var tag = reader.uint32();
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
                };
    
                NDArrayView.DataType = (function() {
                    var valuesById = {}, values = Object.create(valuesById);
                    values[valuesById[0] = "Unknown"] = 0;
                    values[valuesById[1] = "Float"] = 1;
                    values[valuesById[2] = "Double"] = 2;
                    values[valuesById[4] = "Float16"] = 4;
                    values[valuesById[5] = "Int8"] = 5;
                    values[valuesById[6] = "Int16"] = 6;
                    return values;
                })();
    
                NDArrayView.StorageFormat = (function() {
                    var valuesById = {}, values = Object.create(valuesById);
                    values[valuesById[0] = "Dense"] = 0;
                    values[valuesById[1] = "SparseCSC"] = 1;
                    values[valuesById[2] = "SparseBlockCol"] = 2;
                    return values;
                })();
    
                NDArrayView.FloatValues = (function() {
    
                    function FloatValues(properties) {
                        this.value = [];
                        if (properties)
                            for (var keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                                if (properties[keys[i]] != null)
                                    this[keys[i]] = properties[keys[i]];
                    }
    
                    FloatValues.prototype.value = $util.emptyArray;
    
                    FloatValues.decode = function decode(reader, length) {
                        if (!(reader instanceof $Reader))
                            reader = $Reader.create(reader);
                        var end = length === undefined ? reader.len : reader.pos + length, message = new $root.CNTK.proto.NDArrayView.FloatValues();
                        while (reader.pos < end) {
                            var tag = reader.uint32();
                            switch (tag >>> 3) {
                            case 1:
                                if (!(message.value && message.value.length))
                                    message.value = [];
                                if ((tag & 7) === 2) {
                                    var end2 = reader.uint32() + reader.pos;
                                    if (message.value.length == 0 && (end2 - reader.pos) > 1048576) {
                                        var valueLength = end2 - reader.pos;
                                        var valueView = new DataView(reader.buf.buffer, reader.buf.byteOffset + reader.pos, valueLength);
                                        valueLength = valueLength >>> 2;
                                        var value = new Float32Array(valueLength);
                                        for (var i = 0; i < valueLength; i++) {
                                            value[i] = valueView.getFloat32(i << 2, true);
                                        }
                                        message.value = value;
                                        reader.pos = end2;
                                    }
                                    else {
                                        while (reader.pos < end2)
                                            message.value.push(reader.float());
                                    }
                                } else
                                    message.value.push(reader.float());
                                break;
                            default:
                                reader.skipType(tag & 7);
                                break;
                            }
                        }
                        return message;
                    };
    
                    return FloatValues;
                })();
    
                NDArrayView.DoubleValues = (function() {
    
                    function DoubleValues(properties) {
                        this.value = [];
                        if (properties)
                            for (var keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                                if (properties[keys[i]] != null)
                                    this[keys[i]] = properties[keys[i]];
                    }
    
                    DoubleValues.prototype.value = $util.emptyArray;
    
                    DoubleValues.decode = function decode(reader, length) {
                        if (!(reader instanceof $Reader))
                            reader = $Reader.create(reader);
                        var end = length === undefined ? reader.len : reader.pos + length, message = new $root.CNTK.proto.NDArrayView.DoubleValues();
                        while (reader.pos < end) {
                            var tag = reader.uint32();
                            switch (tag >>> 3) {
                            case 1:
                                if (!(message.value && message.value.length))
                                    message.value = [];
                                if ((tag & 7) === 2) {
                                    var end2 = reader.uint32() + reader.pos;
                                    while (reader.pos < end2)
                                        message.value.push(reader.double());
                                } else
                                    message.value.push(reader.double());
                                break;
                            default:
                                reader.skipType(tag & 7);
                                break;
                            }
                        }
                        return message;
                    };
    
                    return DoubleValues;
                })();
    
                NDArrayView.BytesValue = (function() {
    
                    function BytesValue(properties) {
                        if (properties)
                            for (var keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                                if (properties[keys[i]] != null)
                                    this[keys[i]] = properties[keys[i]];
                    }
    
                    BytesValue.prototype.value = $util.newBuffer([]);
    
                    BytesValue.decode = function decode(reader, length) {
                        if (!(reader instanceof $Reader))
                            reader = $Reader.create(reader);
                        var end = length === undefined ? reader.len : reader.pos + length, message = new $root.CNTK.proto.NDArrayView.BytesValue();
                        while (reader.pos < end) {
                            var tag = reader.uint32();
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
                    };
    
                    return BytesValue;
                })();
    
                NDArrayView.IntValues = (function() {
    
                    function IntValues(properties) {
                        this.value = [];
                        if (properties)
                            for (var keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                                if (properties[keys[i]] != null)
                                    this[keys[i]] = properties[keys[i]];
                    }
    
                    IntValues.prototype.value = $util.emptyArray;
    
                    IntValues.decode = function decode(reader, length) {
                        if (!(reader instanceof $Reader))
                            reader = $Reader.create(reader);
                        var end = length === undefined ? reader.len : reader.pos + length, message = new $root.CNTK.proto.NDArrayView.IntValues();
                        while (reader.pos < end) {
                            var tag = reader.uint32();
                            switch (tag >>> 3) {
                            case 1:
                                if (!(message.value && message.value.length))
                                    message.value = [];
                                if ((tag & 7) === 2) {
                                    var end2 = reader.uint32() + reader.pos;
                                    while (reader.pos < end2)
                                        message.value.push(reader.sint32());
                                } else
                                    message.value.push(reader.sint32());
                                break;
                            default:
                                reader.skipType(tag & 7);
                                break;
                            }
                        }
                        return message;
                    };
    
                    return IntValues;
                })();
    
                return NDArrayView;
            })();
    
            proto.Vector = (function() {
    
                function Vector(properties) {
                    this.value = [];
                    if (properties)
                        for (var keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                            if (properties[keys[i]] != null)
                                this[keys[i]] = properties[keys[i]];
                }
    
                Vector.prototype.value = $util.emptyArray;
    
                Vector.decode = function decode(reader, length) {
                    if (!(reader instanceof $Reader))
                        reader = $Reader.create(reader);
                    var end = length === undefined ? reader.len : reader.pos + length, message = new $root.CNTK.proto.Vector();
                    while (reader.pos < end) {
                        var tag = reader.uint32();
                        switch (tag >>> 3) {
                        case 1:
                            if (!(message.value && message.value.length))
                                message.value = [];
                            message.value.push($root.CNTK.proto.DictionaryValue.decode(reader, reader.uint32()));
                            break;
                        default:
                            reader.skipType(tag & 7);
                            break;
                        }
                    }
                    return message;
                };
    
                return Vector;
            })();
    
            proto.Dictionary = (function() {
    
                function Dictionary(properties) {
                    this.data = {};
                    if (properties)
                        for (var keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                            if (properties[keys[i]] != null)
                                this[keys[i]] = properties[keys[i]];
                }
    
                Dictionary.prototype.version = $util.Long ? $util.Long.fromBits(0,0,true) : 0;
                Dictionary.prototype.data = $util.emptyObject;
    
                Dictionary.decode = function decode(reader, length) {
                    if (!(reader instanceof $Reader))
                        reader = $Reader.create(reader);
                    var end = length === undefined ? reader.len : reader.pos + length, message = new $root.CNTK.proto.Dictionary(), key;
                    while (reader.pos < end) {
                        var tag = reader.uint32();
                        switch (tag >>> 3) {
                        case 1:
                            message.version = reader.uint64();
                            break;
                        case 2:
                            reader.skip().pos++;
                            if (message.data === $util.emptyObject)
                                message.data = {};
                            key = reader.string();
                            reader.pos++;
                            message.data[key] = $root.CNTK.proto.DictionaryValue.decode(reader, reader.uint32());
                            break;
                        default:
                            reader.skipType(tag & 7);
                            break;
                        }
                    }
                    return message;
                };
    
                return Dictionary;
            })();
    
            proto.DictionaryValue = (function() {
    
                function DictionaryValue(properties) {
                    if (properties)
                        for (var keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                            if (properties[keys[i]] != null)
                                this[keys[i]] = properties[keys[i]];
                }
    
                DictionaryValue.prototype.version = $util.Long ? $util.Long.fromBits(0,0,true) : 0;
                DictionaryValue.prototype.value_type = 0;
                DictionaryValue.prototype.bool_value = false;
                DictionaryValue.prototype.int_value = 0;
                DictionaryValue.prototype.size_t_value = $util.Long ? $util.Long.fromBits(0,0,true) : 0;
                DictionaryValue.prototype.float_value = 0;
                DictionaryValue.prototype.double_value = 0;
                DictionaryValue.prototype.string_value = "";
                DictionaryValue.prototype.nd_shape_value = null;
                DictionaryValue.prototype.axis_value = null;
                DictionaryValue.prototype.vector_value = null;
                DictionaryValue.prototype.dictionary_value = null;
                DictionaryValue.prototype.nd_array_view_value = null;
    
                var $oneOfFields;
    
                Object.defineProperty(DictionaryValue.prototype, "value", {
                    get: $util.oneOfGetter($oneOfFields = ["bool_value", "int_value", "size_t_value", "float_value", "double_value", "string_value", "nd_shape_value", "axis_value", "vector_value", "dictionary_value", "nd_array_view_value"]),
                    set: $util.oneOfSetter($oneOfFields)
                });
    
                DictionaryValue.decode = function decode(reader, length) {
                    if (!(reader instanceof $Reader))
                        reader = $Reader.create(reader);
                    var end = length === undefined ? reader.len : reader.pos + length, message = new $root.CNTK.proto.DictionaryValue();
                    while (reader.pos < end) {
                        var tag = reader.uint32();
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
                };
    
                DictionaryValue.Type = (function() {
                    var valuesById = {}, values = Object.create(valuesById);
                    values[valuesById[0] = "None"] = 0;
                    values[valuesById[1] = "Bool"] = 1;
                    values[valuesById[2] = "Int"] = 2;
                    values[valuesById[3] = "SizeT"] = 3;
                    values[valuesById[4] = "Float"] = 4;
                    values[valuesById[5] = "Double"] = 5;
                    values[valuesById[6] = "String"] = 6;
                    values[valuesById[7] = "NDShape"] = 7;
                    values[valuesById[8] = "Axis"] = 8;
                    values[valuesById[9] = "Vector"] = 9;
                    values[valuesById[10] = "Dictionary"] = 10;
                    values[valuesById[11] = "NDArrayView"] = 11;
                    return values;
                })();
    
                return DictionaryValue;
            })();
    
            return proto;
        })();
    
        return CNTK;
    })();

    return $root;
})(protobuf);
