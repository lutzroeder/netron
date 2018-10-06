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
    
                NDShape.create = function create(properties) {
                    return new NDShape(properties);
                };
    
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
    
                NDShape.verify = function verify(message) {
                    if (typeof message !== "object" || message === null)
                        return "object expected";
                    if (message.shape_dim != null && message.hasOwnProperty("shape_dim")) {
                        if (!Array.isArray(message.shape_dim))
                            return "shape_dim: array expected";
                        for (var i = 0; i < message.shape_dim.length; ++i)
                            if (!$util.isInteger(message.shape_dim[i]) && !(message.shape_dim[i] && $util.isInteger(message.shape_dim[i].low) && $util.isInteger(message.shape_dim[i].high)))
                                return "shape_dim: integer|Long[] expected";
                    }
                    return null;
                };
    
                NDShape.fromObject = function fromObject(object) {
                    if (object instanceof $root.CNTK.proto.NDShape)
                        return object;
                    var message = new $root.CNTK.proto.NDShape();
                    if (object.shape_dim) {
                        if (!Array.isArray(object.shape_dim))
                            throw TypeError(".CNTK.proto.NDShape.shape_dim: array expected");
                        message.shape_dim = [];
                        for (var i = 0; i < object.shape_dim.length; ++i)
                            if ($util.Long)
                                (message.shape_dim[i] = $util.Long.fromValue(object.shape_dim[i])).unsigned = true;
                            else if (typeof object.shape_dim[i] === "string")
                                message.shape_dim[i] = parseInt(object.shape_dim[i], 10);
                            else if (typeof object.shape_dim[i] === "number")
                                message.shape_dim[i] = object.shape_dim[i];
                            else if (typeof object.shape_dim[i] === "object")
                                message.shape_dim[i] = new $util.LongBits(object.shape_dim[i].low >>> 0, object.shape_dim[i].high >>> 0).toNumber(true);
                    }
                    return message;
                };
    
                NDShape.toObject = function toObject(message, options) {
                    if (!options)
                        options = {};
                    var object = {};
                    if (options.arrays || options.defaults)
                        object.shape_dim = [];
                    if (message.shape_dim && message.shape_dim.length) {
                        object.shape_dim = [];
                        for (var j = 0; j < message.shape_dim.length; ++j)
                            if (typeof message.shape_dim[j] === "number")
                                object.shape_dim[j] = options.longs === String ? String(message.shape_dim[j]) : message.shape_dim[j];
                            else
                                object.shape_dim[j] = options.longs === String ? $util.Long.prototype.toString.call(message.shape_dim[j]) : options.longs === Number ? new $util.LongBits(message.shape_dim[j].low >>> 0, message.shape_dim[j].high >>> 0).toNumber(true) : message.shape_dim[j];
                    }
                    return object;
                };
    
                NDShape.prototype.toJSON = function toJSON() {
                    return this.constructor.toObject(this, $protobuf.util.toJSONOptions);
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
    
                Axis.create = function create(properties) {
                    return new Axis(properties);
                };
    
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
    
                Axis.verify = function verify(message) {
                    if (typeof message !== "object" || message === null)
                        return "object expected";
                    if (message.static_axis_idx != null && message.hasOwnProperty("static_axis_idx"))
                        if (!$util.isInteger(message.static_axis_idx))
                            return "static_axis_idx: integer expected";
                    if (message.name != null && message.hasOwnProperty("name"))
                        if (!$util.isString(message.name))
                            return "name: string expected";
                    if (message.is_ordered_dynamic_axis != null && message.hasOwnProperty("is_ordered_dynamic_axis"))
                        if (typeof message.is_ordered_dynamic_axis !== "boolean")
                            return "is_ordered_dynamic_axis: boolean expected";
                    return null;
                };
    
                Axis.fromObject = function fromObject(object) {
                    if (object instanceof $root.CNTK.proto.Axis)
                        return object;
                    var message = new $root.CNTK.proto.Axis();
                    if (object.static_axis_idx != null)
                        message.static_axis_idx = object.static_axis_idx | 0;
                    if (object.name != null)
                        message.name = String(object.name);
                    if (object.is_ordered_dynamic_axis != null)
                        message.is_ordered_dynamic_axis = Boolean(object.is_ordered_dynamic_axis);
                    return message;
                };
    
                Axis.toObject = function toObject(message, options) {
                    if (!options)
                        options = {};
                    var object = {};
                    if (options.defaults) {
                        object.static_axis_idx = 0;
                        object.name = "";
                        object.is_ordered_dynamic_axis = false;
                    }
                    if (message.static_axis_idx != null && message.hasOwnProperty("static_axis_idx"))
                        object.static_axis_idx = message.static_axis_idx;
                    if (message.name != null && message.hasOwnProperty("name"))
                        object.name = message.name;
                    if (message.is_ordered_dynamic_axis != null && message.hasOwnProperty("is_ordered_dynamic_axis"))
                        object.is_ordered_dynamic_axis = message.is_ordered_dynamic_axis;
                    return object;
                };
    
                Axis.prototype.toJSON = function toJSON() {
                    return this.constructor.toObject(this, $protobuf.util.toJSONOptions);
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
    
                NDArrayView.create = function create(properties) {
                    return new NDArrayView(properties);
                };
    
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
    
                NDArrayView.verify = function verify(message) {
                    if (typeof message !== "object" || message === null)
                        return "object expected";
                    var properties = {};
                    if (message.data_type != null && message.hasOwnProperty("data_type"))
                        switch (message.data_type) {
                        default:
                            return "data_type: enum value expected";
                        case 0:
                        case 1:
                        case 2:
                        case 4:
                        case 5:
                        case 6:
                            break;
                        }
                    if (message.storage_format != null && message.hasOwnProperty("storage_format"))
                        switch (message.storage_format) {
                        default:
                            return "storage_format: enum value expected";
                        case 0:
                        case 1:
                        case 2:
                            break;
                        }
                    if (message.shape != null && message.hasOwnProperty("shape")) {
                        var error = $root.CNTK.proto.NDShape.verify(message.shape);
                        if (error)
                            return "shape." + error;
                    }
                    if (message.float_values != null && message.hasOwnProperty("float_values")) {
                        properties.values = 1;
                        {
                            var error = $root.CNTK.proto.NDArrayView.FloatValues.verify(message.float_values);
                            if (error)
                                return "float_values." + error;
                        }
                    }
                    if (message.double_values != null && message.hasOwnProperty("double_values")) {
                        if (properties.values === 1)
                            return "values: multiple values";
                        properties.values = 1;
                        {
                            var error = $root.CNTK.proto.NDArrayView.DoubleValues.verify(message.double_values);
                            if (error)
                                return "double_values." + error;
                        }
                    }
                    if (message.bytes_value != null && message.hasOwnProperty("bytes_value")) {
                        if (properties.values === 1)
                            return "values: multiple values";
                        properties.values = 1;
                        {
                            var error = $root.CNTK.proto.NDArrayView.BytesValue.verify(message.bytes_value);
                            if (error)
                                return "bytes_value." + error;
                        }
                    }
                    if (message.sint32_values != null && message.hasOwnProperty("sint32_values")) {
                        if (properties.values === 1)
                            return "values: multiple values";
                        properties.values = 1;
                        {
                            var error = $root.CNTK.proto.NDArrayView.IntValues.verify(message.sint32_values);
                            if (error)
                                return "sint32_values." + error;
                        }
                    }
                    return null;
                };
    
                NDArrayView.fromObject = function fromObject(object) {
                    if (object instanceof $root.CNTK.proto.NDArrayView)
                        return object;
                    var message = new $root.CNTK.proto.NDArrayView();
                    switch (object.data_type) {
                    case "Unknown":
                    case 0:
                        message.data_type = 0;
                        break;
                    case "Float":
                    case 1:
                        message.data_type = 1;
                        break;
                    case "Double":
                    case 2:
                        message.data_type = 2;
                        break;
                    case "Float16":
                    case 4:
                        message.data_type = 4;
                        break;
                    case "Int8":
                    case 5:
                        message.data_type = 5;
                        break;
                    case "Int16":
                    case 6:
                        message.data_type = 6;
                        break;
                    }
                    switch (object.storage_format) {
                    case "Dense":
                    case 0:
                        message.storage_format = 0;
                        break;
                    case "SparseCSC":
                    case 1:
                        message.storage_format = 1;
                        break;
                    case "SparseBlockCol":
                    case 2:
                        message.storage_format = 2;
                        break;
                    }
                    if (object.shape != null) {
                        if (typeof object.shape !== "object")
                            throw TypeError(".CNTK.proto.NDArrayView.shape: object expected");
                        message.shape = $root.CNTK.proto.NDShape.fromObject(object.shape);
                    }
                    if (object.float_values != null) {
                        if (typeof object.float_values !== "object")
                            throw TypeError(".CNTK.proto.NDArrayView.float_values: object expected");
                        message.float_values = $root.CNTK.proto.NDArrayView.FloatValues.fromObject(object.float_values);
                    }
                    if (object.double_values != null) {
                        if (typeof object.double_values !== "object")
                            throw TypeError(".CNTK.proto.NDArrayView.double_values: object expected");
                        message.double_values = $root.CNTK.proto.NDArrayView.DoubleValues.fromObject(object.double_values);
                    }
                    if (object.bytes_value != null) {
                        if (typeof object.bytes_value !== "object")
                            throw TypeError(".CNTK.proto.NDArrayView.bytes_value: object expected");
                        message.bytes_value = $root.CNTK.proto.NDArrayView.BytesValue.fromObject(object.bytes_value);
                    }
                    if (object.sint32_values != null) {
                        if (typeof object.sint32_values !== "object")
                            throw TypeError(".CNTK.proto.NDArrayView.sint32_values: object expected");
                        message.sint32_values = $root.CNTK.proto.NDArrayView.IntValues.fromObject(object.sint32_values);
                    }
                    return message;
                };
    
                NDArrayView.toObject = function toObject(message, options) {
                    if (!options)
                        options = {};
                    var object = {};
                    if (options.defaults) {
                        object.data_type = options.enums === String ? "Unknown" : 0;
                        object.storage_format = options.enums === String ? "Dense" : 0;
                        object.shape = null;
                    }
                    if (message.data_type != null && message.hasOwnProperty("data_type"))
                        object.data_type = options.enums === String ? $root.CNTK.proto.NDArrayView.DataType[message.data_type] : message.data_type;
                    if (message.storage_format != null && message.hasOwnProperty("storage_format"))
                        object.storage_format = options.enums === String ? $root.CNTK.proto.NDArrayView.StorageFormat[message.storage_format] : message.storage_format;
                    if (message.shape != null && message.hasOwnProperty("shape"))
                        object.shape = $root.CNTK.proto.NDShape.toObject(message.shape, options);
                    if (message.float_values != null && message.hasOwnProperty("float_values")) {
                        object.float_values = $root.CNTK.proto.NDArrayView.FloatValues.toObject(message.float_values, options);
                        if (options.oneofs)
                            object.values = "float_values";
                    }
                    if (message.double_values != null && message.hasOwnProperty("double_values")) {
                        object.double_values = $root.CNTK.proto.NDArrayView.DoubleValues.toObject(message.double_values, options);
                        if (options.oneofs)
                            object.values = "double_values";
                    }
                    if (message.bytes_value != null && message.hasOwnProperty("bytes_value")) {
                        object.bytes_value = $root.CNTK.proto.NDArrayView.BytesValue.toObject(message.bytes_value, options);
                        if (options.oneofs)
                            object.values = "bytes_value";
                    }
                    if (message.sint32_values != null && message.hasOwnProperty("sint32_values")) {
                        object.sint32_values = $root.CNTK.proto.NDArrayView.IntValues.toObject(message.sint32_values, options);
                        if (options.oneofs)
                            object.values = "sint32_values";
                    }
                    return object;
                };
    
                NDArrayView.prototype.toJSON = function toJSON() {
                    return this.constructor.toObject(this, $protobuf.util.toJSONOptions);
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
    
                    FloatValues.create = function create(properties) {
                        return new FloatValues(properties);
                    };
    
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
    
                    FloatValues.verify = function verify(message) {
                        if (typeof message !== "object" || message === null)
                            return "object expected";
                        if (message.value != null && message.hasOwnProperty("value")) {
                            if (!Array.isArray(message.value))
                                return "value: array expected";
                            for (var i = 0; i < message.value.length; ++i)
                                if (typeof message.value[i] !== "number")
                                    return "value: number[] expected";
                        }
                        return null;
                    };
    
                    FloatValues.fromObject = function fromObject(object) {
                        if (object instanceof $root.CNTK.proto.NDArrayView.FloatValues)
                            return object;
                        var message = new $root.CNTK.proto.NDArrayView.FloatValues();
                        if (object.value) {
                            if (!Array.isArray(object.value))
                                throw TypeError(".CNTK.proto.NDArrayView.FloatValues.value: array expected");
                            message.value = [];
                            for (var i = 0; i < object.value.length; ++i)
                                message.value[i] = Number(object.value[i]);
                        }
                        return message;
                    };
    
                    FloatValues.toObject = function toObject(message, options) {
                        if (!options)
                            options = {};
                        var object = {};
                        if (options.arrays || options.defaults)
                            object.value = [];
                        if (message.value && message.value.length) {
                            object.value = [];
                            for (var j = 0; j < message.value.length; ++j)
                                object.value[j] = options.json && !isFinite(message.value[j]) ? String(message.value[j]) : message.value[j];
                        }
                        return object;
                    };
    
                    FloatValues.prototype.toJSON = function toJSON() {
                        return this.constructor.toObject(this, $protobuf.util.toJSONOptions);
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
    
                    DoubleValues.create = function create(properties) {
                        return new DoubleValues(properties);
                    };
    
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
    
                    DoubleValues.verify = function verify(message) {
                        if (typeof message !== "object" || message === null)
                            return "object expected";
                        if (message.value != null && message.hasOwnProperty("value")) {
                            if (!Array.isArray(message.value))
                                return "value: array expected";
                            for (var i = 0; i < message.value.length; ++i)
                                if (typeof message.value[i] !== "number")
                                    return "value: number[] expected";
                        }
                        return null;
                    };
    
                    DoubleValues.fromObject = function fromObject(object) {
                        if (object instanceof $root.CNTK.proto.NDArrayView.DoubleValues)
                            return object;
                        var message = new $root.CNTK.proto.NDArrayView.DoubleValues();
                        if (object.value) {
                            if (!Array.isArray(object.value))
                                throw TypeError(".CNTK.proto.NDArrayView.DoubleValues.value: array expected");
                            message.value = [];
                            for (var i = 0; i < object.value.length; ++i)
                                message.value[i] = Number(object.value[i]);
                        }
                        return message;
                    };
    
                    DoubleValues.toObject = function toObject(message, options) {
                        if (!options)
                            options = {};
                        var object = {};
                        if (options.arrays || options.defaults)
                            object.value = [];
                        if (message.value && message.value.length) {
                            object.value = [];
                            for (var j = 0; j < message.value.length; ++j)
                                object.value[j] = options.json && !isFinite(message.value[j]) ? String(message.value[j]) : message.value[j];
                        }
                        return object;
                    };
    
                    DoubleValues.prototype.toJSON = function toJSON() {
                        return this.constructor.toObject(this, $protobuf.util.toJSONOptions);
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
    
                    BytesValue.create = function create(properties) {
                        return new BytesValue(properties);
                    };
    
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
    
                    BytesValue.verify = function verify(message) {
                        if (typeof message !== "object" || message === null)
                            return "object expected";
                        if (message.value != null && message.hasOwnProperty("value"))
                            if (!(message.value && typeof message.value.length === "number" || $util.isString(message.value)))
                                return "value: buffer expected";
                        return null;
                    };
    
                    BytesValue.fromObject = function fromObject(object) {
                        if (object instanceof $root.CNTK.proto.NDArrayView.BytesValue)
                            return object;
                        var message = new $root.CNTK.proto.NDArrayView.BytesValue();
                        if (object.value != null)
                            if (typeof object.value === "string")
                                $util.base64.decode(object.value, message.value = $util.newBuffer($util.base64.length(object.value)), 0);
                            else if (object.value.length)
                                message.value = object.value;
                        return message;
                    };
    
                    BytesValue.toObject = function toObject(message, options) {
                        if (!options)
                            options = {};
                        var object = {};
                        if (options.defaults)
                            if (options.bytes === String)
                                object.value = "";
                            else {
                                object.value = [];
                                if (options.bytes !== Array)
                                    object.value = $util.newBuffer(object.value);
                            }
                        if (message.value != null && message.hasOwnProperty("value"))
                            object.value = options.bytes === String ? $util.base64.encode(message.value, 0, message.value.length) : options.bytes === Array ? Array.prototype.slice.call(message.value) : message.value;
                        return object;
                    };
    
                    BytesValue.prototype.toJSON = function toJSON() {
                        return this.constructor.toObject(this, $protobuf.util.toJSONOptions);
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
    
                    IntValues.create = function create(properties) {
                        return new IntValues(properties);
                    };
    
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
    
                    IntValues.verify = function verify(message) {
                        if (typeof message !== "object" || message === null)
                            return "object expected";
                        if (message.value != null && message.hasOwnProperty("value")) {
                            if (!Array.isArray(message.value))
                                return "value: array expected";
                            for (var i = 0; i < message.value.length; ++i)
                                if (!$util.isInteger(message.value[i]))
                                    return "value: integer[] expected";
                        }
                        return null;
                    };
    
                    IntValues.fromObject = function fromObject(object) {
                        if (object instanceof $root.CNTK.proto.NDArrayView.IntValues)
                            return object;
                        var message = new $root.CNTK.proto.NDArrayView.IntValues();
                        if (object.value) {
                            if (!Array.isArray(object.value))
                                throw TypeError(".CNTK.proto.NDArrayView.IntValues.value: array expected");
                            message.value = [];
                            for (var i = 0; i < object.value.length; ++i)
                                message.value[i] = object.value[i] | 0;
                        }
                        return message;
                    };
    
                    IntValues.toObject = function toObject(message, options) {
                        if (!options)
                            options = {};
                        var object = {};
                        if (options.arrays || options.defaults)
                            object.value = [];
                        if (message.value && message.value.length) {
                            object.value = [];
                            for (var j = 0; j < message.value.length; ++j)
                                object.value[j] = message.value[j];
                        }
                        return object;
                    };
    
                    IntValues.prototype.toJSON = function toJSON() {
                        return this.constructor.toObject(this, $protobuf.util.toJSONOptions);
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
    
                Vector.create = function create(properties) {
                    return new Vector(properties);
                };
    
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
    
                Vector.verify = function verify(message) {
                    if (typeof message !== "object" || message === null)
                        return "object expected";
                    if (message.value != null && message.hasOwnProperty("value")) {
                        if (!Array.isArray(message.value))
                            return "value: array expected";
                        for (var i = 0; i < message.value.length; ++i) {
                            var error = $root.CNTK.proto.DictionaryValue.verify(message.value[i]);
                            if (error)
                                return "value." + error;
                        }
                    }
                    return null;
                };
    
                Vector.fromObject = function fromObject(object) {
                    if (object instanceof $root.CNTK.proto.Vector)
                        return object;
                    var message = new $root.CNTK.proto.Vector();
                    if (object.value) {
                        if (!Array.isArray(object.value))
                            throw TypeError(".CNTK.proto.Vector.value: array expected");
                        message.value = [];
                        for (var i = 0; i < object.value.length; ++i) {
                            if (typeof object.value[i] !== "object")
                                throw TypeError(".CNTK.proto.Vector.value: object expected");
                            message.value[i] = $root.CNTK.proto.DictionaryValue.fromObject(object.value[i]);
                        }
                    }
                    return message;
                };
    
                Vector.toObject = function toObject(message, options) {
                    if (!options)
                        options = {};
                    var object = {};
                    if (options.arrays || options.defaults)
                        object.value = [];
                    if (message.value && message.value.length) {
                        object.value = [];
                        for (var j = 0; j < message.value.length; ++j)
                            object.value[j] = $root.CNTK.proto.DictionaryValue.toObject(message.value[j], options);
                    }
                    return object;
                };
    
                Vector.prototype.toJSON = function toJSON() {
                    return this.constructor.toObject(this, $protobuf.util.toJSONOptions);
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
    
                Dictionary.create = function create(properties) {
                    return new Dictionary(properties);
                };
    
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
    
                Dictionary.verify = function verify(message) {
                    if (typeof message !== "object" || message === null)
                        return "object expected";
                    if (message.version != null && message.hasOwnProperty("version"))
                        if (!$util.isInteger(message.version) && !(message.version && $util.isInteger(message.version.low) && $util.isInteger(message.version.high)))
                            return "version: integer|Long expected";
                    if (message.data != null && message.hasOwnProperty("data")) {
                        if (!$util.isObject(message.data))
                            return "data: object expected";
                        var key = Object.keys(message.data);
                        for (var i = 0; i < key.length; ++i) {
                            var error = $root.CNTK.proto.DictionaryValue.verify(message.data[key[i]]);
                            if (error)
                                return "data." + error;
                        }
                    }
                    return null;
                };
    
                Dictionary.fromObject = function fromObject(object) {
                    if (object instanceof $root.CNTK.proto.Dictionary)
                        return object;
                    var message = new $root.CNTK.proto.Dictionary();
                    if (object.version != null)
                        if ($util.Long)
                            (message.version = $util.Long.fromValue(object.version)).unsigned = true;
                        else if (typeof object.version === "string")
                            message.version = parseInt(object.version, 10);
                        else if (typeof object.version === "number")
                            message.version = object.version;
                        else if (typeof object.version === "object")
                            message.version = new $util.LongBits(object.version.low >>> 0, object.version.high >>> 0).toNumber(true);
                    if (object.data) {
                        if (typeof object.data !== "object")
                            throw TypeError(".CNTK.proto.Dictionary.data: object expected");
                        message.data = {};
                        for (var keys = Object.keys(object.data), i = 0; i < keys.length; ++i) {
                            if (typeof object.data[keys[i]] !== "object")
                                throw TypeError(".CNTK.proto.Dictionary.data: object expected");
                            message.data[keys[i]] = $root.CNTK.proto.DictionaryValue.fromObject(object.data[keys[i]]);
                        }
                    }
                    return message;
                };
    
                Dictionary.toObject = function toObject(message, options) {
                    if (!options)
                        options = {};
                    var object = {};
                    if (options.objects || options.defaults)
                        object.data = {};
                    if (options.defaults)
                        if ($util.Long) {
                            var long = new $util.Long(0, 0, true);
                            object.version = options.longs === String ? long.toString() : options.longs === Number ? long.toNumber() : long;
                        } else
                            object.version = options.longs === String ? "0" : 0;
                    if (message.version != null && message.hasOwnProperty("version"))
                        if (typeof message.version === "number")
                            object.version = options.longs === String ? String(message.version) : message.version;
                        else
                            object.version = options.longs === String ? $util.Long.prototype.toString.call(message.version) : options.longs === Number ? new $util.LongBits(message.version.low >>> 0, message.version.high >>> 0).toNumber(true) : message.version;
                    var keys2;
                    if (message.data && (keys2 = Object.keys(message.data)).length) {
                        object.data = {};
                        for (var j = 0; j < keys2.length; ++j)
                            object.data[keys2[j]] = $root.CNTK.proto.DictionaryValue.toObject(message.data[keys2[j]], options);
                    }
                    return object;
                };
    
                Dictionary.prototype.toJSON = function toJSON() {
                    return this.constructor.toObject(this, $protobuf.util.toJSONOptions);
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
    
                DictionaryValue.create = function create(properties) {
                    return new DictionaryValue(properties);
                };
    
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
    
                DictionaryValue.verify = function verify(message) {
                    if (typeof message !== "object" || message === null)
                        return "object expected";
                    var properties = {};
                    if (message.version != null && message.hasOwnProperty("version"))
                        if (!$util.isInteger(message.version) && !(message.version && $util.isInteger(message.version.low) && $util.isInteger(message.version.high)))
                            return "version: integer|Long expected";
                    if (message.value_type != null && message.hasOwnProperty("value_type"))
                        switch (message.value_type) {
                        default:
                            return "value_type: enum value expected";
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
                        case 11:
                            break;
                        }
                    if (message.bool_value != null && message.hasOwnProperty("bool_value")) {
                        properties.value = 1;
                        if (typeof message.bool_value !== "boolean")
                            return "bool_value: boolean expected";
                    }
                    if (message.int_value != null && message.hasOwnProperty("int_value")) {
                        if (properties.value === 1)
                            return "value: multiple values";
                        properties.value = 1;
                        if (!$util.isInteger(message.int_value))
                            return "int_value: integer expected";
                    }
                    if (message.size_t_value != null && message.hasOwnProperty("size_t_value")) {
                        if (properties.value === 1)
                            return "value: multiple values";
                        properties.value = 1;
                        if (!$util.isInteger(message.size_t_value) && !(message.size_t_value && $util.isInteger(message.size_t_value.low) && $util.isInteger(message.size_t_value.high)))
                            return "size_t_value: integer|Long expected";
                    }
                    if (message.float_value != null && message.hasOwnProperty("float_value")) {
                        if (properties.value === 1)
                            return "value: multiple values";
                        properties.value = 1;
                        if (typeof message.float_value !== "number")
                            return "float_value: number expected";
                    }
                    if (message.double_value != null && message.hasOwnProperty("double_value")) {
                        if (properties.value === 1)
                            return "value: multiple values";
                        properties.value = 1;
                        if (typeof message.double_value !== "number")
                            return "double_value: number expected";
                    }
                    if (message.string_value != null && message.hasOwnProperty("string_value")) {
                        if (properties.value === 1)
                            return "value: multiple values";
                        properties.value = 1;
                        if (!$util.isString(message.string_value))
                            return "string_value: string expected";
                    }
                    if (message.nd_shape_value != null && message.hasOwnProperty("nd_shape_value")) {
                        if (properties.value === 1)
                            return "value: multiple values";
                        properties.value = 1;
                        {
                            var error = $root.CNTK.proto.NDShape.verify(message.nd_shape_value);
                            if (error)
                                return "nd_shape_value." + error;
                        }
                    }
                    if (message.axis_value != null && message.hasOwnProperty("axis_value")) {
                        if (properties.value === 1)
                            return "value: multiple values";
                        properties.value = 1;
                        {
                            var error = $root.CNTK.proto.Axis.verify(message.axis_value);
                            if (error)
                                return "axis_value." + error;
                        }
                    }
                    if (message.vector_value != null && message.hasOwnProperty("vector_value")) {
                        if (properties.value === 1)
                            return "value: multiple values";
                        properties.value = 1;
                        {
                            var error = $root.CNTK.proto.Vector.verify(message.vector_value);
                            if (error)
                                return "vector_value." + error;
                        }
                    }
                    if (message.dictionary_value != null && message.hasOwnProperty("dictionary_value")) {
                        if (properties.value === 1)
                            return "value: multiple values";
                        properties.value = 1;
                        {
                            var error = $root.CNTK.proto.Dictionary.verify(message.dictionary_value);
                            if (error)
                                return "dictionary_value." + error;
                        }
                    }
                    if (message.nd_array_view_value != null && message.hasOwnProperty("nd_array_view_value")) {
                        if (properties.value === 1)
                            return "value: multiple values";
                        properties.value = 1;
                        {
                            var error = $root.CNTK.proto.NDArrayView.verify(message.nd_array_view_value);
                            if (error)
                                return "nd_array_view_value." + error;
                        }
                    }
                    return null;
                };
    
                DictionaryValue.fromObject = function fromObject(object) {
                    if (object instanceof $root.CNTK.proto.DictionaryValue)
                        return object;
                    var message = new $root.CNTK.proto.DictionaryValue();
                    if (object.version != null)
                        if ($util.Long)
                            (message.version = $util.Long.fromValue(object.version)).unsigned = true;
                        else if (typeof object.version === "string")
                            message.version = parseInt(object.version, 10);
                        else if (typeof object.version === "number")
                            message.version = object.version;
                        else if (typeof object.version === "object")
                            message.version = new $util.LongBits(object.version.low >>> 0, object.version.high >>> 0).toNumber(true);
                    switch (object.value_type) {
                    case "None":
                    case 0:
                        message.value_type = 0;
                        break;
                    case "Bool":
                    case 1:
                        message.value_type = 1;
                        break;
                    case "Int":
                    case 2:
                        message.value_type = 2;
                        break;
                    case "SizeT":
                    case 3:
                        message.value_type = 3;
                        break;
                    case "Float":
                    case 4:
                        message.value_type = 4;
                        break;
                    case "Double":
                    case 5:
                        message.value_type = 5;
                        break;
                    case "String":
                    case 6:
                        message.value_type = 6;
                        break;
                    case "NDShape":
                    case 7:
                        message.value_type = 7;
                        break;
                    case "Axis":
                    case 8:
                        message.value_type = 8;
                        break;
                    case "Vector":
                    case 9:
                        message.value_type = 9;
                        break;
                    case "Dictionary":
                    case 10:
                        message.value_type = 10;
                        break;
                    case "NDArrayView":
                    case 11:
                        message.value_type = 11;
                        break;
                    }
                    if (object.bool_value != null)
                        message.bool_value = Boolean(object.bool_value);
                    if (object.int_value != null)
                        message.int_value = object.int_value | 0;
                    if (object.size_t_value != null)
                        if ($util.Long)
                            (message.size_t_value = $util.Long.fromValue(object.size_t_value)).unsigned = true;
                        else if (typeof object.size_t_value === "string")
                            message.size_t_value = parseInt(object.size_t_value, 10);
                        else if (typeof object.size_t_value === "number")
                            message.size_t_value = object.size_t_value;
                        else if (typeof object.size_t_value === "object")
                            message.size_t_value = new $util.LongBits(object.size_t_value.low >>> 0, object.size_t_value.high >>> 0).toNumber(true);
                    if (object.float_value != null)
                        message.float_value = Number(object.float_value);
                    if (object.double_value != null)
                        message.double_value = Number(object.double_value);
                    if (object.string_value != null)
                        message.string_value = String(object.string_value);
                    if (object.nd_shape_value != null) {
                        if (typeof object.nd_shape_value !== "object")
                            throw TypeError(".CNTK.proto.DictionaryValue.nd_shape_value: object expected");
                        message.nd_shape_value = $root.CNTK.proto.NDShape.fromObject(object.nd_shape_value);
                    }
                    if (object.axis_value != null) {
                        if (typeof object.axis_value !== "object")
                            throw TypeError(".CNTK.proto.DictionaryValue.axis_value: object expected");
                        message.axis_value = $root.CNTK.proto.Axis.fromObject(object.axis_value);
                    }
                    if (object.vector_value != null) {
                        if (typeof object.vector_value !== "object")
                            throw TypeError(".CNTK.proto.DictionaryValue.vector_value: object expected");
                        message.vector_value = $root.CNTK.proto.Vector.fromObject(object.vector_value);
                    }
                    if (object.dictionary_value != null) {
                        if (typeof object.dictionary_value !== "object")
                            throw TypeError(".CNTK.proto.DictionaryValue.dictionary_value: object expected");
                        message.dictionary_value = $root.CNTK.proto.Dictionary.fromObject(object.dictionary_value);
                    }
                    if (object.nd_array_view_value != null) {
                        if (typeof object.nd_array_view_value !== "object")
                            throw TypeError(".CNTK.proto.DictionaryValue.nd_array_view_value: object expected");
                        message.nd_array_view_value = $root.CNTK.proto.NDArrayView.fromObject(object.nd_array_view_value);
                    }
                    return message;
                };
    
                DictionaryValue.toObject = function toObject(message, options) {
                    if (!options)
                        options = {};
                    var object = {};
                    if (options.defaults) {
                        if ($util.Long) {
                            var long = new $util.Long(0, 0, true);
                            object.version = options.longs === String ? long.toString() : options.longs === Number ? long.toNumber() : long;
                        } else
                            object.version = options.longs === String ? "0" : 0;
                        object.value_type = options.enums === String ? "None" : 0;
                    }
                    if (message.version != null && message.hasOwnProperty("version"))
                        if (typeof message.version === "number")
                            object.version = options.longs === String ? String(message.version) : message.version;
                        else
                            object.version = options.longs === String ? $util.Long.prototype.toString.call(message.version) : options.longs === Number ? new $util.LongBits(message.version.low >>> 0, message.version.high >>> 0).toNumber(true) : message.version;
                    if (message.value_type != null && message.hasOwnProperty("value_type"))
                        object.value_type = options.enums === String ? $root.CNTK.proto.DictionaryValue.Type[message.value_type] : message.value_type;
                    if (message.bool_value != null && message.hasOwnProperty("bool_value")) {
                        object.bool_value = message.bool_value;
                        if (options.oneofs)
                            object.value = "bool_value";
                    }
                    if (message.int_value != null && message.hasOwnProperty("int_value")) {
                        object.int_value = message.int_value;
                        if (options.oneofs)
                            object.value = "int_value";
                    }
                    if (message.size_t_value != null && message.hasOwnProperty("size_t_value")) {
                        if (typeof message.size_t_value === "number")
                            object.size_t_value = options.longs === String ? String(message.size_t_value) : message.size_t_value;
                        else
                            object.size_t_value = options.longs === String ? $util.Long.prototype.toString.call(message.size_t_value) : options.longs === Number ? new $util.LongBits(message.size_t_value.low >>> 0, message.size_t_value.high >>> 0).toNumber(true) : message.size_t_value;
                        if (options.oneofs)
                            object.value = "size_t_value";
                    }
                    if (message.float_value != null && message.hasOwnProperty("float_value")) {
                        object.float_value = options.json && !isFinite(message.float_value) ? String(message.float_value) : message.float_value;
                        if (options.oneofs)
                            object.value = "float_value";
                    }
                    if (message.double_value != null && message.hasOwnProperty("double_value")) {
                        object.double_value = options.json && !isFinite(message.double_value) ? String(message.double_value) : message.double_value;
                        if (options.oneofs)
                            object.value = "double_value";
                    }
                    if (message.string_value != null && message.hasOwnProperty("string_value")) {
                        object.string_value = message.string_value;
                        if (options.oneofs)
                            object.value = "string_value";
                    }
                    if (message.nd_shape_value != null && message.hasOwnProperty("nd_shape_value")) {
                        object.nd_shape_value = $root.CNTK.proto.NDShape.toObject(message.nd_shape_value, options);
                        if (options.oneofs)
                            object.value = "nd_shape_value";
                    }
                    if (message.axis_value != null && message.hasOwnProperty("axis_value")) {
                        object.axis_value = $root.CNTK.proto.Axis.toObject(message.axis_value, options);
                        if (options.oneofs)
                            object.value = "axis_value";
                    }
                    if (message.vector_value != null && message.hasOwnProperty("vector_value")) {
                        object.vector_value = $root.CNTK.proto.Vector.toObject(message.vector_value, options);
                        if (options.oneofs)
                            object.value = "vector_value";
                    }
                    if (message.dictionary_value != null && message.hasOwnProperty("dictionary_value")) {
                        object.dictionary_value = $root.CNTK.proto.Dictionary.toObject(message.dictionary_value, options);
                        if (options.oneofs)
                            object.value = "dictionary_value";
                    }
                    if (message.nd_array_view_value != null && message.hasOwnProperty("nd_array_view_value")) {
                        object.nd_array_view_value = $root.CNTK.proto.NDArrayView.toObject(message.nd_array_view_value, options);
                        if (options.oneofs)
                            object.value = "nd_array_view_value";
                    }
                    return object;
                };
    
                DictionaryValue.prototype.toJSON = function toJSON() {
                    return this.constructor.toObject(this, $protobuf.util.toJSONOptions);
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
