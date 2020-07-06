(function($protobuf) {
    "use strict";

    const $root = $protobuf.get('cntk');

    $root.CNTK = (function() {

        const CNTK = {};

        CNTK.proto = (function() {

            const proto = {};

            proto.NDShape = (function() {

                function NDShape() {
                    this.shape_dim = [];
                }

                NDShape.prototype.shape_dim = [];

                NDShape.decode = function (reader, length) {
                    const message = new $root.CNTK.proto.NDShape();
                    const end = reader.next(length);
                    while (reader.end(end)) {
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
                };

                return NDShape;
            })();

            proto.Axis = (function() {

                function Axis() {
                }

                Axis.prototype.static_axis_idx = 0;
                Axis.prototype.name = "";
                Axis.prototype.is_ordered_dynamic_axis = false;

                Axis.decode = function (reader, length) {
                    const message = new $root.CNTK.proto.Axis();
                    const end = reader.next(length);
                    while (reader.end(end)) {
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
                };

                return Axis;
            })();

            proto.NDArrayView = (function() {

                function NDArrayView() {
                }

                NDArrayView.prototype.data_type = 0;
                NDArrayView.prototype.storage_format = 0;
                NDArrayView.prototype.shape = null;
                NDArrayView.prototype.float_values = null;
                NDArrayView.prototype.double_values = null;
                NDArrayView.prototype.bytes_value = null;
                NDArrayView.prototype.sint32_values = null;

                const valuesSet = new Set([ "float_values", "double_values", "bytes_value", "sint32_values"]);
                Object.defineProperty(NDArrayView.prototype, "values", {
                    get: function() { return Object.keys(this).find((key) => valuesSet.has(key) && this[key] != null); }
                });

                NDArrayView.decode = function (reader, length) {
                    const message = new $root.CNTK.proto.NDArrayView();
                    const end = reader.next(length);
                    while (reader.end(end)) {
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
                };

                NDArrayView.DataType = (function() {
                    const values = {};
                    values["Unknown"] = 0;
                    values["Float"] = 1;
                    values["Double"] = 2;
                    values["Float16"] = 4;
                    values["Int8"] = 5;
                    values["Int16"] = 6;
                    return values;
                })();

                NDArrayView.StorageFormat = (function() {
                    const values = {};
                    values["Dense"] = 0;
                    values["SparseCSC"] = 1;
                    values["SparseBlockCol"] = 2;
                    return values;
                })();

                NDArrayView.FloatValues = (function() {

                    function FloatValues() {
                        this.value = [];
                    }

                    FloatValues.prototype.value = [];

                    FloatValues.decode = function (reader, length) {
                        const message = new $root.CNTK.proto.NDArrayView.FloatValues();
                        const end = reader.next(length);
                        while (reader.end(end)) {
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
                    };

                    return FloatValues;
                })();

                NDArrayView.DoubleValues = (function() {

                    function DoubleValues() {
                        this.value = [];
                    }

                    DoubleValues.prototype.value = [];

                    DoubleValues.decode = function (reader, length) {
                        const message = new $root.CNTK.proto.NDArrayView.DoubleValues();
                        const end = reader.next(length);
                        while (reader.end(end)) {
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
                    };

                    return DoubleValues;
                })();

                NDArrayView.BytesValue = (function() {

                    function BytesValue() {
                    }

                    BytesValue.prototype.value = new Uint8Array([]);

                    BytesValue.decode = function (reader, length) {
                        const message = new $root.CNTK.proto.NDArrayView.BytesValue();
                        const end = reader.next(length);
                        while (reader.end(end)) {
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
                    };

                    return BytesValue;
                })();

                NDArrayView.IntValues = (function() {

                    function IntValues() {
                        this.value = [];
                    }

                    IntValues.prototype.value = [];

                    IntValues.decode = function (reader, length) {
                        const message = new $root.CNTK.proto.NDArrayView.IntValues();
                        const end = reader.next(length);
                        while (reader.end(end)) {
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
                    };

                    return IntValues;
                })();

                return NDArrayView;
            })();

            proto.Vector = (function() {

                function Vector() {
                    this.value = [];
                }

                Vector.prototype.value = [];

                Vector.decode = function (reader, length) {
                    const message = new $root.CNTK.proto.Vector();
                    const end = reader.next(length);
                    while (reader.end(end)) {
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
                };

                return Vector;
            })();

            proto.Dictionary = (function() {

                function Dictionary() {
                    this.data = {};
                }

                Dictionary.prototype.version = $protobuf.Long ? $protobuf.Long.fromBits(0, 0, true) : 0;
                Dictionary.prototype.data = {};

                Dictionary.decode = function (reader, length) {
                    const message = new $root.CNTK.proto.Dictionary();
                    const end = reader.next(length);
                    while (reader.end(end)) {
                        const tag = reader.uint32();
                        switch (tag >>> 3) {
                            case 1:
                                message.version = reader.uint64();
                                break;
                            case 2:
                                reader.pair(message.data, () => reader.string(), () => $root.CNTK.proto.DictionaryValue.decode(reader, reader.uint32()));
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

                function DictionaryValue() {
                }

                DictionaryValue.prototype.version = $protobuf.Long ? $protobuf.Long.fromBits(0, 0, true) : 0;
                DictionaryValue.prototype.value_type = 0;
                DictionaryValue.prototype.bool_value = false;
                DictionaryValue.prototype.int_value = 0;
                DictionaryValue.prototype.size_t_value = $protobuf.Long ? $protobuf.Long.fromBits(0, 0, true) : 0;
                DictionaryValue.prototype.float_value = 0;
                DictionaryValue.prototype.double_value = 0;
                DictionaryValue.prototype.string_value = "";
                DictionaryValue.prototype.nd_shape_value = null;
                DictionaryValue.prototype.axis_value = null;
                DictionaryValue.prototype.vector_value = null;
                DictionaryValue.prototype.dictionary_value = null;
                DictionaryValue.prototype.nd_array_view_value = null;

                const valueSet = new Set([ "bool_value", "int_value", "size_t_value", "float_value", "double_value", "string_value", "nd_shape_value", "axis_value", "vector_value", "dictionary_value", "nd_array_view_value"]);
                Object.defineProperty(DictionaryValue.prototype, "value", {
                    get: function() { return Object.keys(this).find((key) => valueSet.has(key) && this[key] != null); }
                });

                DictionaryValue.decode = function (reader, length) {
                    const message = new $root.CNTK.proto.DictionaryValue();
                    const end = reader.next(length);
                    while (reader.end(end)) {
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
                };

                DictionaryValue.Type = (function() {
                    const values = {};
                    values["None"] = 0;
                    values["Bool"] = 1;
                    values["Int"] = 2;
                    values["SizeT"] = 3;
                    values["Float"] = 4;
                    values["Double"] = 5;
                    values["String"] = 6;
                    values["NDShape"] = 7;
                    values["Axis"] = 8;
                    values["Vector"] = 9;
                    values["Dictionary"] = 10;
                    values["NDArrayView"] = 11;
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
