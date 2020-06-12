/*eslint-disable block-scoped-var, id-length, no-control-regex, no-magic-numbers, no-prototype-builtins, no-redeclare, no-shadow, no-var, sort-vars*/
(function($protobuf) {
    "use strict";

    var $Reader = $protobuf.Reader, $util = $protobuf.util;
    
    var $root = $protobuf.roots.uff || ($protobuf.roots.uff = {});
    
    $root.uff = (function() {
    
        var uff = {};
    
        uff.MetaGraph = (function() {
    
            function MetaGraph(properties) {
                this.descriptors = [];
                this.graphs = [];
                this.referenced_data = [];
                if (properties)
                    for (var keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                        if (properties[keys[i]] != null)
                            this[keys[i]] = properties[keys[i]];
            }
    
            MetaGraph.prototype.version = $util.Long ? $util.Long.fromBits(0,0,false) : 0;
            MetaGraph.prototype.descriptor_core_version = $util.Long ? $util.Long.fromBits(0,0,false) : 0;
            MetaGraph.prototype.descriptors = $util.emptyArray;
            MetaGraph.prototype.graphs = $util.emptyArray;
            MetaGraph.prototype.referenced_data = $util.emptyArray;
    
            MetaGraph.decode = function decode(reader, length) {
                if (!(reader instanceof $Reader))
                    reader = $Reader.create(reader);
                var end = length === undefined ? reader.len : reader.pos + length, message = new $root.uff.MetaGraph();
                while (reader.pos < end) {
                    var tag = reader.uint32();
                    switch (tag >>> 3) {
                    case 1:
                        message.version = reader.int64();
                        break;
                    case 2:
                        message.descriptor_core_version = reader.int64();
                        break;
                    case 3:
                        if (!(message.descriptors && message.descriptors.length))
                            message.descriptors = [];
                        message.descriptors.push($root.uff.Descriptor.decode(reader, reader.uint32()));
                        break;
                    case 4:
                        if (!(message.graphs && message.graphs.length))
                            message.graphs = [];
                        message.graphs.push($root.uff.Graph.decode(reader, reader.uint32()));
                        break;
                    case 5:
                        if (!(message.referenced_data && message.referenced_data.length))
                            message.referenced_data = [];
                        message.referenced_data.push($root.uff.KeyValuePair.decode(reader, reader.uint32()));
                        break;
                    default:
                        reader.skipType(tag & 7);
                        break;
                    }
                }
                return message;
            };
    
            MetaGraph.decodeText = function decodeText(reader) {
                var message = new $root.uff.MetaGraph();
                reader.start();
                while (!reader.end()) {
                    var tag = reader.tag();
                    switch (tag) {
                    case "version":
                        message.version = reader.int64();
                        break;
                    case "descriptor_core_version":
                        message.descriptor_core_version = reader.int64();
                        break;
                    case "descriptors":
                        if (!(message.descriptors && message.descriptors.length))
                            message.descriptors = [];
                        message.descriptors.push($root.uff.Descriptor.decodeText(reader, true));
                        break;
                    case "graphs":
                        if (!(message.graphs && message.graphs.length))
                            message.graphs = [];
                        message.graphs.push($root.uff.Graph.decodeText(reader, true));
                        break;
                    case "referenced_data":
                        if (!(message.referenced_data && message.referenced_data.length))
                            message.referenced_data = [];
                        message.referenced_data.push($root.uff.KeyValuePair.decodeText(reader, true));
                        break;
                    default:
                        reader.field(tag, message);
                        break;
                    }
                }
                return message;
            };
    
            return MetaGraph;
        })();
    
        uff.Descriptor = (function() {
    
            function Descriptor(properties) {
                if (properties)
                    for (var keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                        if (properties[keys[i]] != null)
                            this[keys[i]] = properties[keys[i]];
            }
    
            Descriptor.prototype.id = "";
            Descriptor.prototype.version = $util.Long ? $util.Long.fromBits(0,0,false) : 0;
    
            Descriptor.decode = function decode(reader, length) {
                if (!(reader instanceof $Reader))
                    reader = $Reader.create(reader);
                var end = length === undefined ? reader.len : reader.pos + length, message = new $root.uff.Descriptor();
                while (reader.pos < end) {
                    var tag = reader.uint32();
                    switch (tag >>> 3) {
                    case 1:
                        message.id = reader.string();
                        break;
                    case 2:
                        message.version = reader.int64();
                        break;
                    default:
                        reader.skipType(tag & 7);
                        break;
                    }
                }
                if (!message.hasOwnProperty("id"))
                    throw $util.ProtocolError("missing required 'id'", { instance: message });
                if (!message.hasOwnProperty("version"))
                    throw $util.ProtocolError("missing required 'version'", { instance: message });
                return message;
            };
    
            Descriptor.decodeText = function decodeText(reader) {
                var message = new $root.uff.Descriptor();
                reader.start();
                while (!reader.end()) {
                    var tag = reader.tag();
                    switch (tag) {
                    case "id":
                        message.id = reader.string();
                        break;
                    case "version":
                        message.version = reader.int64();
                        break;
                    default:
                        reader.field(tag, message);
                        break;
                    }
                }
                if (!message.hasOwnProperty("id"))
                    throw $util.ProtocolError("missing required 'id'", { instance: message });
                if (!message.hasOwnProperty("version"))
                    throw $util.ProtocolError("missing required 'version'", { instance: message });
                return message;
            };
    
            return Descriptor;
        })();
    
        uff.Graph = (function() {
    
            function Graph(properties) {
                this.nodes = [];
                if (properties)
                    for (var keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                        if (properties[keys[i]] != null)
                            this[keys[i]] = properties[keys[i]];
            }
    
            Graph.prototype.id = "";
            Graph.prototype.nodes = $util.emptyArray;
    
            Graph.decode = function decode(reader, length) {
                if (!(reader instanceof $Reader))
                    reader = $Reader.create(reader);
                var end = length === undefined ? reader.len : reader.pos + length, message = new $root.uff.Graph();
                while (reader.pos < end) {
                    var tag = reader.uint32();
                    switch (tag >>> 3) {
                    case 1:
                        message.id = reader.string();
                        break;
                    case 2:
                        if (!(message.nodes && message.nodes.length))
                            message.nodes = [];
                        message.nodes.push($root.uff.Node.decode(reader, reader.uint32()));
                        break;
                    default:
                        reader.skipType(tag & 7);
                        break;
                    }
                }
                return message;
            };
    
            Graph.decodeText = function decodeText(reader) {
                var message = new $root.uff.Graph();
                reader.start();
                while (!reader.end()) {
                    var tag = reader.tag();
                    switch (tag) {
                    case "id":
                        message.id = reader.string();
                        break;
                    case "nodes":
                        if (!(message.nodes && message.nodes.length))
                            message.nodes = [];
                        message.nodes.push($root.uff.Node.decodeText(reader, true));
                        break;
                    default:
                        reader.field(tag, message);
                        break;
                    }
                }
                return message;
            };
    
            return Graph;
        })();
    
        uff.Node = (function() {
    
            function Node(properties) {
                this.inputs = [];
                this.fields = [];
                this.extra_fields = [];
                if (properties)
                    for (var keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                        if (properties[keys[i]] != null)
                            this[keys[i]] = properties[keys[i]];
            }
    
            Node.prototype.id = "";
            Node.prototype.inputs = $util.emptyArray;
            Node.prototype.operation = "";
            Node.prototype.fields = $util.emptyArray;
            Node.prototype.extra_fields = $util.emptyArray;
    
            Node.decode = function decode(reader, length) {
                if (!(reader instanceof $Reader))
                    reader = $Reader.create(reader);
                var end = length === undefined ? reader.len : reader.pos + length, message = new $root.uff.Node();
                while (reader.pos < end) {
                    var tag = reader.uint32();
                    switch (tag >>> 3) {
                    case 1:
                        message.id = reader.string();
                        break;
                    case 2:
                        if (!(message.inputs && message.inputs.length))
                            message.inputs = [];
                        message.inputs.push(reader.string());
                        break;
                    case 3:
                        message.operation = reader.string();
                        break;
                    case 4:
                        if (!(message.fields && message.fields.length))
                            message.fields = [];
                        message.fields.push($root.uff.KeyValuePair.decode(reader, reader.uint32()));
                        break;
                    case 5:
                        if (!(message.extra_fields && message.extra_fields.length))
                            message.extra_fields = [];
                        message.extra_fields.push($root.uff.KeyValuePair.decode(reader, reader.uint32()));
                        break;
                    default:
                        reader.skipType(tag & 7);
                        break;
                    }
                }
                if (!message.hasOwnProperty("id"))
                    throw $util.ProtocolError("missing required 'id'", { instance: message });
                if (!message.hasOwnProperty("operation"))
                    throw $util.ProtocolError("missing required 'operation'", { instance: message });
                return message;
            };
    
            Node.decodeText = function decodeText(reader) {
                var message = new $root.uff.Node();
                reader.start();
                while (!reader.end()) {
                    var tag = reader.tag();
                    switch (tag) {
                    case "id":
                        message.id = reader.string();
                        break;
                    case "inputs":
                        if (!(message.inputs && message.inputs.length))
                            message.inputs = [];
                        if (reader.first())
                            while (!reader.last()) {
                                message.inputs.push(reader.string());
                                reader.next();
                            }
                        else
                            message.inputs.push(reader.string());
                        break;
                    case "operation":
                        message.operation = reader.string();
                        break;
                    case "fields":
                        if (!(message.fields && message.fields.length))
                            message.fields = [];
                        message.fields.push($root.uff.KeyValuePair.decodeText(reader, true));
                        break;
                    case "extra_fields":
                        if (!(message.extra_fields && message.extra_fields.length))
                            message.extra_fields = [];
                        message.extra_fields.push($root.uff.KeyValuePair.decodeText(reader, true));
                        break;
                    default:
                        reader.field(tag, message);
                        break;
                    }
                }
                if (!message.hasOwnProperty("id"))
                    throw $util.ProtocolError("missing required 'id'", { instance: message });
                if (!message.hasOwnProperty("operation"))
                    throw $util.ProtocolError("missing required 'operation'", { instance: message });
                return message;
            };
    
            return Node;
        })();
    
        uff.KeyValuePair = (function() {
    
            function KeyValuePair(properties) {
                if (properties)
                    for (var keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                        if (properties[keys[i]] != null)
                            this[keys[i]] = properties[keys[i]];
            }
    
            KeyValuePair.prototype.key = "";
            KeyValuePair.prototype.value = null;
    
            KeyValuePair.decode = function decode(reader, length) {
                if (!(reader instanceof $Reader))
                    reader = $Reader.create(reader);
                var end = length === undefined ? reader.len : reader.pos + length, message = new $root.uff.KeyValuePair();
                while (reader.pos < end) {
                    var tag = reader.uint32();
                    switch (tag >>> 3) {
                    case 1:
                        message.key = reader.string();
                        break;
                    case 2:
                        message.value = $root.uff.Value.decode(reader, reader.uint32());
                        break;
                    default:
                        reader.skipType(tag & 7);
                        break;
                    }
                }
                if (!message.hasOwnProperty("key"))
                    throw $util.ProtocolError("missing required 'key'", { instance: message });
                if (!message.hasOwnProperty("value"))
                    throw $util.ProtocolError("missing required 'value'", { instance: message });
                return message;
            };
    
            KeyValuePair.decodeText = function decodeText(reader) {
                var message = new $root.uff.KeyValuePair();
                reader.start();
                while (!reader.end()) {
                    var tag = reader.tag();
                    switch (tag) {
                    case "key":
                        message.key = reader.string();
                        break;
                    case "value":
                        message.value = $root.uff.Value.decodeText(reader, true);
                        break;
                    default:
                        reader.field(tag, message);
                        break;
                    }
                }
                if (!message.hasOwnProperty("key"))
                    throw $util.ProtocolError("missing required 'key'", { instance: message });
                if (!message.hasOwnProperty("value"))
                    throw $util.ProtocolError("missing required 'value'", { instance: message });
                return message;
            };
    
            return KeyValuePair;
        })();
    
        uff.Value = (function() {
    
            function Value(properties) {
                if (properties)
                    for (var keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                        if (properties[keys[i]] != null)
                            this[keys[i]] = properties[keys[i]];
            }
    
            Value.prototype.s = "";
            Value.prototype.s_list = null;
            Value.prototype.d = 0;
            Value.prototype.d_list = null;
            Value.prototype.b = false;
            Value.prototype.b_list = false;
            Value.prototype.i = $util.Long ? $util.Long.fromBits(0,0,false) : 0;
            Value.prototype.i_list = null;
            Value.prototype.blob = $util.newBuffer([]);
            Value.prototype.ref = "";
            Value.prototype.dtype = 65544;
            Value.prototype.dtype_list = null;
            Value.prototype.dim_orders = null;
            Value.prototype.dim_orders_list = null;
    
            var $oneOfFields;
    
            Object.defineProperty(Value.prototype, "type", {
                get: $util.oneOfGetter($oneOfFields = ["s", "s_list", "d", "d_list", "b", "b_list", "i", "i_list", "blob", "ref", "dtype", "dtype_list", "dim_orders", "dim_orders_list"]),
                set: $util.oneOfSetter($oneOfFields)
            });
    
            Value.decode = function decode(reader, length) {
                if (!(reader instanceof $Reader))
                    reader = $Reader.create(reader);
                var end = length === undefined ? reader.len : reader.pos + length, message = new $root.uff.Value();
                while (reader.pos < end) {
                    var tag = reader.uint32();
                    switch (tag >>> 3) {
                    case 1:
                        message.s = reader.string();
                        break;
                    case 2:
                        message.s_list = $root.uff.ListString.decode(reader, reader.uint32());
                        break;
                    case 3:
                        message.d = reader.double();
                        break;
                    case 4:
                        message.d_list = $root.uff.ListDouble.decode(reader, reader.uint32());
                        break;
                    case 5:
                        message.b = reader.bool();
                        break;
                    case 6:
                        message.b_list = reader.bool();
                        break;
                    case 7:
                        message.i = reader.int64();
                        break;
                    case 8:
                        message.i_list = $root.uff.ListInt.decode(reader, reader.uint32());
                        break;
                    case 9:
                        message.blob = reader.bytes();
                        break;
                    case 100:
                        message.ref = reader.string();
                        break;
                    case 101:
                        message.dtype = reader.int32();
                        break;
                    case 102:
                        message.dtype_list = $root.uff.ListDataType.decode(reader, reader.uint32());
                        break;
                    case 103:
                        message.dim_orders = $root.uff.DimOrders.decode(reader, reader.uint32());
                        break;
                    case 104:
                        message.dim_orders_list = $root.uff.ListDimOrders.decode(reader, reader.uint32());
                        break;
                    default:
                        reader.skipType(tag & 7);
                        break;
                    }
                }
                return message;
            };
    
            Value.decodeText = function decodeText(reader) {
                var message = new $root.uff.Value();
                reader.start();
                while (!reader.end()) {
                    var tag = reader.tag();
                    switch (tag) {
                    case "s":
                        message.s = reader.string();
                        break;
                    case "s_list":
                        message.s_list = $root.uff.ListString.decodeText(reader, true);
                        break;
                    case "d":
                        message.d = reader.double();
                        break;
                    case "d_list":
                        message.d_list = $root.uff.ListDouble.decodeText(reader, true);
                        break;
                    case "b":
                        message.b = reader.bool();
                        break;
                    case "b_list":
                        message.b_list = reader.bool();
                        break;
                    case "i":
                        message.i = reader.int64();
                        break;
                    case "i_list":
                        message.i_list = $root.uff.ListInt.decodeText(reader, true);
                        break;
                    case "blob":
                        message.blob = reader.bytes();
                        break;
                    case "ref":
                        message.ref = reader.string();
                        break;
                    case "dtype":
                        message.dtype = reader.enum($root.uff.DataType);
                        break;
                    case "dtype_list":
                        message.dtype_list = $root.uff.ListDataType.decodeText(reader, true);
                        break;
                    case "dim_orders":
                        message.dim_orders = $root.uff.DimOrders.decodeText(reader, true);
                        break;
                    case "dim_orders_list":
                        message.dim_orders_list = $root.uff.ListDimOrders.decodeText(reader, true);
                        break;
                    default:
                        reader.field(tag, message);
                        break;
                    }
                }
                return message;
            };
    
            return Value;
        })();
    
        uff.DataType = (function() {
            var valuesById = {}, values = Object.create(valuesById);
            values[valuesById[65544] = "DT_INT8"] = 65544;
            values[valuesById[65552] = "DT_INT16"] = 65552;
            values[valuesById[65568] = "DT_INT32"] = 65568;
            values[valuesById[65600] = "DT_INT64"] = 65600;
            values[valuesById[131088] = "DT_FLOAT16"] = 131088;
            values[valuesById[131104] = "DT_FLOAT32"] = 131104;
            return values;
        })();
    
        uff.DimOrder = (function() {
    
            function DimOrder(properties) {
                if (properties)
                    for (var keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                        if (properties[keys[i]] != null)
                            this[keys[i]] = properties[keys[i]];
            }
    
            DimOrder.prototype.key = $util.Long ? $util.Long.fromBits(0,0,false) : 0;
            DimOrder.prototype.value = null;
    
            DimOrder.decode = function decode(reader, length) {
                if (!(reader instanceof $Reader))
                    reader = $Reader.create(reader);
                var end = length === undefined ? reader.len : reader.pos + length, message = new $root.uff.DimOrder();
                while (reader.pos < end) {
                    var tag = reader.uint32();
                    switch (tag >>> 3) {
                    case 1:
                        message.key = reader.int64();
                        break;
                    case 2:
                        message.value = $root.uff.ListInt.decode(reader, reader.uint32());
                        break;
                    default:
                        reader.skipType(tag & 7);
                        break;
                    }
                }
                if (!message.hasOwnProperty("key"))
                    throw $util.ProtocolError("missing required 'key'", { instance: message });
                if (!message.hasOwnProperty("value"))
                    throw $util.ProtocolError("missing required 'value'", { instance: message });
                return message;
            };
    
            DimOrder.decodeText = function decodeText(reader) {
                var message = new $root.uff.DimOrder();
                reader.start();
                while (!reader.end()) {
                    var tag = reader.tag();
                    switch (tag) {
                    case "key":
                        message.key = reader.int64();
                        break;
                    case "value":
                        message.value = $root.uff.ListInt.decodeText(reader, true);
                        break;
                    default:
                        reader.field(tag, message);
                        break;
                    }
                }
                if (!message.hasOwnProperty("key"))
                    throw $util.ProtocolError("missing required 'key'", { instance: message });
                if (!message.hasOwnProperty("value"))
                    throw $util.ProtocolError("missing required 'value'", { instance: message });
                return message;
            };
    
            return DimOrder;
        })();
    
        uff.DimOrders = (function() {
    
            function DimOrders(properties) {
                this.orders = [];
                if (properties)
                    for (var keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                        if (properties[keys[i]] != null)
                            this[keys[i]] = properties[keys[i]];
            }
    
            DimOrders.prototype.orders = $util.emptyArray;
    
            DimOrders.decode = function decode(reader, length) {
                if (!(reader instanceof $Reader))
                    reader = $Reader.create(reader);
                var end = length === undefined ? reader.len : reader.pos + length, message = new $root.uff.DimOrders();
                while (reader.pos < end) {
                    var tag = reader.uint32();
                    switch (tag >>> 3) {
                    case 1:
                        if (!(message.orders && message.orders.length))
                            message.orders = [];
                        message.orders.push($root.uff.DimOrder.decode(reader, reader.uint32()));
                        break;
                    default:
                        reader.skipType(tag & 7);
                        break;
                    }
                }
                return message;
            };
    
            DimOrders.decodeText = function decodeText(reader) {
                var message = new $root.uff.DimOrders();
                reader.start();
                while (!reader.end()) {
                    var tag = reader.tag();
                    switch (tag) {
                    case "orders":
                        if (!(message.orders && message.orders.length))
                            message.orders = [];
                        message.orders.push($root.uff.DimOrder.decodeText(reader, true));
                        break;
                    default:
                        reader.field(tag, message);
                        break;
                    }
                }
                return message;
            };
    
            return DimOrders;
        })();
    
        uff.ListString = (function() {
    
            function ListString(properties) {
                this.val = [];
                if (properties)
                    for (var keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                        if (properties[keys[i]] != null)
                            this[keys[i]] = properties[keys[i]];
            }
    
            ListString.prototype.val = $util.emptyArray;
    
            ListString.decode = function decode(reader, length) {
                if (!(reader instanceof $Reader))
                    reader = $Reader.create(reader);
                var end = length === undefined ? reader.len : reader.pos + length, message = new $root.uff.ListString();
                while (reader.pos < end) {
                    var tag = reader.uint32();
                    switch (tag >>> 3) {
                    case 1:
                        if (!(message.val && message.val.length))
                            message.val = [];
                        message.val.push(reader.string());
                        break;
                    default:
                        reader.skipType(tag & 7);
                        break;
                    }
                }
                return message;
            };
    
            ListString.decodeText = function decodeText(reader) {
                var message = new $root.uff.ListString();
                reader.start();
                while (!reader.end()) {
                    var tag = reader.tag();
                    switch (tag) {
                    case "val":
                        if (!(message.val && message.val.length))
                            message.val = [];
                        if (reader.first())
                            while (!reader.last()) {
                                message.val.push(reader.string());
                                reader.next();
                            }
                        else
                            message.val.push(reader.string());
                        break;
                    default:
                        reader.field(tag, message);
                        break;
                    }
                }
                return message;
            };
    
            return ListString;
        })();
    
        uff.ListInt = (function() {
    
            function ListInt(properties) {
                this.val = [];
                if (properties)
                    for (var keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                        if (properties[keys[i]] != null)
                            this[keys[i]] = properties[keys[i]];
            }
    
            ListInt.prototype.val = $util.emptyArray;
    
            ListInt.decode = function decode(reader, length) {
                if (!(reader instanceof $Reader))
                    reader = $Reader.create(reader);
                var end = length === undefined ? reader.len : reader.pos + length, message = new $root.uff.ListInt();
                while (reader.pos < end) {
                    var tag = reader.uint32();
                    switch (tag >>> 3) {
                    case 1:
                        if (!(message.val && message.val.length))
                            message.val = [];
                        if ((tag & 7) === 2) {
                            var end2 = reader.uint32() + reader.pos;
                            while (reader.pos < end2)
                                message.val.push(reader.int64());
                        } else
                            message.val.push(reader.int64());
                        break;
                    default:
                        reader.skipType(tag & 7);
                        break;
                    }
                }
                return message;
            };
    
            ListInt.decodeText = function decodeText(reader) {
                var message = new $root.uff.ListInt();
                reader.start();
                while (!reader.end()) {
                    var tag = reader.tag();
                    switch (tag) {
                    case "val":
                        if (!(message.val && message.val.length))
                            message.val = [];
                        if (reader.first())
                            while (!reader.last()) {
                                message.val.push(reader.int64());
                                reader.next();
                            }
                        else
                            message.val.push(reader.int64());
                        break;
                    default:
                        reader.field(tag, message);
                        break;
                    }
                }
                return message;
            };
    
            return ListInt;
        })();
    
        uff.ListDouble = (function() {
    
            function ListDouble(properties) {
                this.val = [];
                if (properties)
                    for (var keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                        if (properties[keys[i]] != null)
                            this[keys[i]] = properties[keys[i]];
            }
    
            ListDouble.prototype.val = $util.emptyArray;
    
            ListDouble.decode = function decode(reader, length) {
                if (!(reader instanceof $Reader))
                    reader = $Reader.create(reader);
                var end = length === undefined ? reader.len : reader.pos + length, message = new $root.uff.ListDouble();
                while (reader.pos < end) {
                    var tag = reader.uint32();
                    switch (tag >>> 3) {
                    case 1:
                        if (!(message.val && message.val.length))
                            message.val = [];
                        if ((tag & 7) === 2) {
                            var end2 = reader.uint32() + reader.pos;
                            while (reader.pos < end2)
                                message.val.push(reader.double());
                        } else
                            message.val.push(reader.double());
                        break;
                    default:
                        reader.skipType(tag & 7);
                        break;
                    }
                }
                return message;
            };
    
            ListDouble.decodeText = function decodeText(reader) {
                var message = new $root.uff.ListDouble();
                reader.start();
                while (!reader.end()) {
                    var tag = reader.tag();
                    switch (tag) {
                    case "val":
                        if (!(message.val && message.val.length))
                            message.val = [];
                        if (reader.first())
                            while (!reader.last()) {
                                message.val.push(reader.double());
                                reader.next();
                            }
                        else
                            message.val.push(reader.double());
                        break;
                    default:
                        reader.field(tag, message);
                        break;
                    }
                }
                return message;
            };
    
            return ListDouble;
        })();
    
        uff.ListDataType = (function() {
    
            function ListDataType(properties) {
                this.val = [];
                if (properties)
                    for (var keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                        if (properties[keys[i]] != null)
                            this[keys[i]] = properties[keys[i]];
            }
    
            ListDataType.prototype.val = $util.emptyArray;
    
            ListDataType.decode = function decode(reader, length) {
                if (!(reader instanceof $Reader))
                    reader = $Reader.create(reader);
                var end = length === undefined ? reader.len : reader.pos + length, message = new $root.uff.ListDataType();
                while (reader.pos < end) {
                    var tag = reader.uint32();
                    switch (tag >>> 3) {
                    case 1:
                        if (!(message.val && message.val.length))
                            message.val = [];
                        if ((tag & 7) === 2) {
                            var end2 = reader.uint32() + reader.pos;
                            while (reader.pos < end2)
                                message.val.push(reader.int32());
                        } else
                            message.val.push(reader.int32());
                        break;
                    default:
                        reader.skipType(tag & 7);
                        break;
                    }
                }
                return message;
            };
    
            ListDataType.decodeText = function decodeText(reader) {
                var message = new $root.uff.ListDataType();
                reader.start();
                while (!reader.end()) {
                    var tag = reader.tag();
                    switch (tag) {
                    case "val":
                        if (!(message.val && message.val.length))
                            message.val = [];
                        if (reader.first())
                            while (!reader.last()) {
                                message.val.push(reader.enum($root.uff.DataType));
                                reader.next();
                            }
                        else
                            message.val.push(reader.enum($root.uff.DataType));
                        break;
                    default:
                        reader.field(tag, message);
                        break;
                    }
                }
                return message;
            };
    
            return ListDataType;
        })();
    
        uff.ListDimOrders = (function() {
    
            function ListDimOrders(properties) {
                this.val = [];
                if (properties)
                    for (var keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                        if (properties[keys[i]] != null)
                            this[keys[i]] = properties[keys[i]];
            }
    
            ListDimOrders.prototype.val = $util.emptyArray;
    
            ListDimOrders.decode = function decode(reader, length) {
                if (!(reader instanceof $Reader))
                    reader = $Reader.create(reader);
                var end = length === undefined ? reader.len : reader.pos + length, message = new $root.uff.ListDimOrders();
                while (reader.pos < end) {
                    var tag = reader.uint32();
                    switch (tag >>> 3) {
                    case 1:
                        if (!(message.val && message.val.length))
                            message.val = [];
                        message.val.push($root.uff.DimOrders.decode(reader, reader.uint32()));
                        break;
                    default:
                        reader.skipType(tag & 7);
                        break;
                    }
                }
                return message;
            };
    
            ListDimOrders.decodeText = function decodeText(reader) {
                var message = new $root.uff.ListDimOrders();
                reader.start();
                while (!reader.end()) {
                    var tag = reader.tag();
                    switch (tag) {
                    case "val":
                        if (!(message.val && message.val.length))
                            message.val = [];
                        message.val.push($root.uff.DimOrders.decodeText(reader, true));
                        break;
                    default:
                        reader.field(tag, message);
                        break;
                    }
                }
                return message;
            };
    
            return ListDimOrders;
        })();
    
        return uff;
    })();

    return $root;
})(protobuf);
