(function($protobuf) {
    "use strict";

    const $root = $protobuf.get('uff');

    $root.uff = (function() {

        const uff = {};

        uff.MetaGraph = (function() {

            function MetaGraph() {
                this.descriptors = [];
                this.graphs = [];
                this.referenced_data = [];
            }

            MetaGraph.prototype.version = $protobuf.Long ? $protobuf.Long.fromBits(0, 0, false) : 0;
            MetaGraph.prototype.descriptor_core_version = $protobuf.Long ? $protobuf.Long.fromBits(0, 0, false) : 0;
            MetaGraph.prototype.descriptors = [];
            MetaGraph.prototype.graphs = [];
            MetaGraph.prototype.referenced_data = [];

            MetaGraph.decode = function (reader, length) {
                const message = new $root.uff.MetaGraph();
                const end = reader.next(length);
                while (reader.end(end)) {
                    const tag = reader.uint32();
                    switch (tag >>> 3) {
                        case 1:
                            message.version = reader.int64();
                            break;
                        case 2:
                            message.descriptor_core_version = reader.int64();
                            break;
                        case 3:
                            message.descriptors.push($root.uff.Descriptor.decode(reader, reader.uint32()));
                            break;
                        case 4:
                            message.graphs.push($root.uff.Graph.decode(reader, reader.uint32()));
                            break;
                        case 5:
                            message.referenced_data.push($root.uff.KeyValuePair.decode(reader, reader.uint32()));
                            break;
                        default:
                            reader.skipType(tag & 7);
                            break;
                    }
                }
                return message;
            };

            MetaGraph.decodeText = function (reader) {
                const message = new $root.uff.MetaGraph();
                reader.start();
                while (!reader.end()) {
                    const tag = reader.tag();
                    switch (tag) {
                        case "version":
                            message.version = reader.int64();
                            break;
                        case "descriptor_core_version":
                            message.descriptor_core_version = reader.int64();
                            break;
                        case "descriptors":
                            message.descriptors.push($root.uff.Descriptor.decodeText(reader, true));
                            break;
                        case "graphs":
                            message.graphs.push($root.uff.Graph.decodeText(reader, true));
                            break;
                        case "referenced_data":
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

            function Descriptor() {
            }

            Descriptor.prototype.id = "";
            Descriptor.prototype.version = $protobuf.Long ? $protobuf.Long.fromBits(0, 0, false) : 0;

            Descriptor.decode = function (reader, length) {
                const message = new $root.uff.Descriptor();
                const end = reader.next(length);
                while (reader.end(end)) {
                    const tag = reader.uint32();
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
                if (!Object.prototype.hasOwnProperty.call(message, 'id')) {
                    throw $protobuf.Error("Excepted 'id'.");
                }
                if (!Object.prototype.hasOwnProperty.call(message, 'version')) {
                    throw $protobuf.Error("Excepted 'version'.");
                }
                return message;
            };

            Descriptor.decodeText = function (reader) {
                const message = new $root.uff.Descriptor();
                reader.start();
                while (!reader.end()) {
                    const tag = reader.tag();
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
                if (!Object.prototype.hasOwnProperty.call(message, "id"))
                    throw $protobuf.Error("Excepted 'id'.");
                if (!Object.prototype.hasOwnProperty.call(message, "version"))
                    throw $protobuf.Error("Excepted 'version'.");
                return message;
            };

            return Descriptor;
        })();

        uff.Graph = (function() {

            function Graph() {
                this.nodes = [];
            }

            Graph.prototype.id = "";
            Graph.prototype.nodes = [];

            Graph.decode = function (reader, length) {
                const message = new $root.uff.Graph();
                const end = reader.next(length);
                while (reader.end(end)) {
                    const tag = reader.uint32();
                    switch (tag >>> 3) {
                        case 1:
                            message.id = reader.string();
                            break;
                        case 2:
                            message.nodes.push($root.uff.Node.decode(reader, reader.uint32()));
                            break;
                        default:
                            reader.skipType(tag & 7);
                            break;
                    }
                }
                return message;
            };

            Graph.decodeText = function (reader) {
                const message = new $root.uff.Graph();
                reader.start();
                while (!reader.end()) {
                    const tag = reader.tag();
                    switch (tag) {
                        case "id":
                            message.id = reader.string();
                            break;
                        case "nodes":
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

            function Node() {
                this.inputs = [];
                this.fields = [];
                this.extra_fields = [];
            }

            Node.prototype.id = "";
            Node.prototype.inputs = [];
            Node.prototype.operation = "";
            Node.prototype.fields = [];
            Node.prototype.extra_fields = [];

            Node.decode = function (reader, length) {
                const message = new $root.uff.Node();
                const end = reader.next(length);
                while (reader.end(end)) {
                    const tag = reader.uint32();
                    switch (tag >>> 3) {
                        case 1:
                            message.id = reader.string();
                            break;
                        case 2:
                            message.inputs.push(reader.string());
                            break;
                        case 3:
                            message.operation = reader.string();
                            break;
                        case 4:
                            message.fields.push($root.uff.KeyValuePair.decode(reader, reader.uint32()));
                            break;
                        case 5:
                            message.extra_fields.push($root.uff.KeyValuePair.decode(reader, reader.uint32()));
                            break;
                        default:
                            reader.skipType(tag & 7);
                            break;
                    }
                }
                if (!Object.prototype.hasOwnProperty.call(message, 'id')) {
                    throw $protobuf.Error("Excepted 'id'.");
                }
                if (!Object.prototype.hasOwnProperty.call(message, 'operation')) {
                    throw $protobuf.Error("Excepted 'operation'.");
                }
                return message;
            };

            Node.decodeText = function (reader) {
                const message = new $root.uff.Node();
                reader.start();
                while (!reader.end()) {
                    const tag = reader.tag();
                    switch (tag) {
                        case "id":
                            message.id = reader.string();
                            break;
                        case "inputs":
                            reader.array(message.inputs, () => reader.string());
                            break;
                        case "operation":
                            message.operation = reader.string();
                            break;
                        case "fields":
                            message.fields.push($root.uff.KeyValuePair.decodeText(reader, true));
                            break;
                        case "extra_fields":
                            message.extra_fields.push($root.uff.KeyValuePair.decodeText(reader, true));
                            break;
                        default:
                            reader.field(tag, message);
                            break;
                    }
                }
                if (!Object.prototype.hasOwnProperty.call(message, "id"))
                    throw $protobuf.Error("Excepted 'id'.");
                if (!Object.prototype.hasOwnProperty.call(message, "operation"))
                    throw $protobuf.Error("Excepted 'operation'.");
                return message;
            };

            return Node;
        })();

        uff.KeyValuePair = (function() {

            function KeyValuePair() {
            }

            KeyValuePair.prototype.key = "";
            KeyValuePair.prototype.value = null;

            KeyValuePair.decode = function (reader, length) {
                const message = new $root.uff.KeyValuePair();
                const end = reader.next(length);
                while (reader.end(end)) {
                    const tag = reader.uint32();
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
                if (!Object.prototype.hasOwnProperty.call(message, 'key')) {
                    throw $protobuf.Error("Excepted 'key'.");
                }
                if (!Object.prototype.hasOwnProperty.call(message, 'value')) {
                    throw $protobuf.Error("Excepted 'value'.");
                }
                return message;
            };

            KeyValuePair.decodeText = function (reader) {
                const message = new $root.uff.KeyValuePair();
                reader.start();
                while (!reader.end()) {
                    const tag = reader.tag();
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
                if (!Object.prototype.hasOwnProperty.call(message, "key"))
                    throw $protobuf.Error("Excepted 'key'.");
                if (!Object.prototype.hasOwnProperty.call(message, "value"))
                    throw $protobuf.Error("Excepted 'value'.");
                return message;
            };

            return KeyValuePair;
        })();

        uff.Value = (function() {

            function Value() {
            }

            Value.prototype.s = "";
            Value.prototype.s_list = null;
            Value.prototype.d = 0;
            Value.prototype.d_list = null;
            Value.prototype.b = false;
            Value.prototype.b_list = false;
            Value.prototype.i = $protobuf.Long ? $protobuf.Long.fromBits(0, 0, false) : 0;
            Value.prototype.i_list = null;
            Value.prototype.blob = new Uint8Array([]);
            Value.prototype.ref = "";
            Value.prototype.dtype = 65544;
            Value.prototype.dtype_list = null;
            Value.prototype.dim_orders = null;
            Value.prototype.dim_orders_list = null;

            const typeSet = new Set([ "s", "s_list", "d", "d_list", "b", "b_list", "i", "i_list", "blob", "ref", "dtype", "dtype_list", "dim_orders", "dim_orders_list"]);
            Object.defineProperty(Value.prototype, "type", {
                get: function() { return Object.keys(this).find((key) => typeSet.has(key) && this[key] != null); }
            });

            Value.decode = function (reader, length) {
                const message = new $root.uff.Value();
                const end = reader.next(length);
                while (reader.end(end)) {
                    const tag = reader.uint32();
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

            Value.decodeText = function (reader) {
                const message = new $root.uff.Value();
                reader.start();
                while (!reader.end()) {
                    const tag = reader.tag();
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
            const values = {};
            values["DT_INT8"] = 65544;
            values["DT_INT16"] = 65552;
            values["DT_INT32"] = 65568;
            values["DT_INT64"] = 65600;
            values["DT_FLOAT16"] = 131088;
            values["DT_FLOAT32"] = 131104;
            return values;
        })();

        uff.DimOrder = (function() {

            function DimOrder() {
            }

            DimOrder.prototype.key = $protobuf.Long ? $protobuf.Long.fromBits(0, 0, false) : 0;
            DimOrder.prototype.value = null;

            DimOrder.decode = function (reader, length) {
                const message = new $root.uff.DimOrder();
                const end = reader.next(length);
                while (reader.end(end)) {
                    const tag = reader.uint32();
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
                if (!Object.prototype.hasOwnProperty.call(message, 'key')) {
                    throw $protobuf.Error("Excepted 'key'.");
                }
                if (!Object.prototype.hasOwnProperty.call(message, 'value')) {
                    throw $protobuf.Error("Excepted 'value'.");
                }
                return message;
            };

            DimOrder.decodeText = function (reader) {
                const message = new $root.uff.DimOrder();
                reader.start();
                while (!reader.end()) {
                    const tag = reader.tag();
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
                if (!Object.prototype.hasOwnProperty.call(message, "key"))
                    throw $protobuf.Error("Excepted 'key'.");
                if (!Object.prototype.hasOwnProperty.call(message, "value"))
                    throw $protobuf.Error("Excepted 'value'.");
                return message;
            };

            return DimOrder;
        })();

        uff.DimOrders = (function() {

            function DimOrders() {
                this.orders = [];
            }

            DimOrders.prototype.orders = [];

            DimOrders.decode = function (reader, length) {
                const message = new $root.uff.DimOrders();
                const end = reader.next(length);
                while (reader.end(end)) {
                    const tag = reader.uint32();
                    switch (tag >>> 3) {
                        case 1:
                            message.orders.push($root.uff.DimOrder.decode(reader, reader.uint32()));
                            break;
                        default:
                            reader.skipType(tag & 7);
                            break;
                    }
                }
                return message;
            };

            DimOrders.decodeText = function (reader) {
                const message = new $root.uff.DimOrders();
                reader.start();
                while (!reader.end()) {
                    const tag = reader.tag();
                    switch (tag) {
                        case "orders":
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

            function ListString() {
                this.val = [];
            }

            ListString.prototype.val = [];

            ListString.decode = function (reader, length) {
                const message = new $root.uff.ListString();
                const end = reader.next(length);
                while (reader.end(end)) {
                    const tag = reader.uint32();
                    switch (tag >>> 3) {
                        case 1:
                            message.val.push(reader.string());
                            break;
                        default:
                            reader.skipType(tag & 7);
                            break;
                    }
                }
                return message;
            };

            ListString.decodeText = function (reader) {
                const message = new $root.uff.ListString();
                reader.start();
                while (!reader.end()) {
                    const tag = reader.tag();
                    switch (tag) {
                        case "val":
                            reader.array(message.val, () => reader.string());
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

            function ListInt() {
                this.val = [];
            }

            ListInt.prototype.val = [];

            ListInt.decode = function (reader, length) {
                const message = new $root.uff.ListInt();
                const end = reader.next(length);
                while (reader.end(end)) {
                    const tag = reader.uint32();
                    switch (tag >>> 3) {
                        case 1:
                            message.val = reader.array(message.val, () => reader.int64(), tag);
                            break;
                        default:
                            reader.skipType(tag & 7);
                            break;
                    }
                }
                return message;
            };

            ListInt.decodeText = function (reader) {
                const message = new $root.uff.ListInt();
                reader.start();
                while (!reader.end()) {
                    const tag = reader.tag();
                    switch (tag) {
                        case "val":
                            reader.array(message.val, () => reader.int64());
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

            function ListDouble() {
                this.val = [];
            }

            ListDouble.prototype.val = [];

            ListDouble.decode = function (reader, length) {
                const message = new $root.uff.ListDouble();
                const end = reader.next(length);
                while (reader.end(end)) {
                    const tag = reader.uint32();
                    switch (tag >>> 3) {
                        case 1:
                            message.val = reader.doubles(message.val, tag);
                            break;
                        default:
                            reader.skipType(tag & 7);
                            break;
                    }
                }
                return message;
            };

            ListDouble.decodeText = function (reader) {
                const message = new $root.uff.ListDouble();
                reader.start();
                while (!reader.end()) {
                    const tag = reader.tag();
                    switch (tag) {
                        case "val":
                            reader.array(message.val, () => reader.double());
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

            function ListDataType() {
                this.val = [];
            }

            ListDataType.prototype.val = [];

            ListDataType.decode = function (reader, length) {
                const message = new $root.uff.ListDataType();
                const end = reader.next(length);
                while (reader.end(end)) {
                    const tag = reader.uint32();
                    switch (tag >>> 3) {
                        case 1:
                            message.val = reader.array(message.val, () => reader.int32(), tag);
                            break;
                        default:
                            reader.skipType(tag & 7);
                            break;
                    }
                }
                return message;
            };

            ListDataType.decodeText = function (reader) {
                const message = new $root.uff.ListDataType();
                reader.start();
                while (!reader.end()) {
                    const tag = reader.tag();
                    switch (tag) {
                        case "val":
                            reader.array(message.val, () => reader.enum($root.uff.DataType));
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

            function ListDimOrders() {
                this.val = [];
            }

            ListDimOrders.prototype.val = [];

            ListDimOrders.decode = function (reader, length) {
                const message = new $root.uff.ListDimOrders();
                const end = reader.next(length);
                while (reader.end(end)) {
                    const tag = reader.uint32();
                    switch (tag >>> 3) {
                        case 1:
                            message.val.push($root.uff.DimOrders.decode(reader, reader.uint32()));
                            break;
                        default:
                            reader.skipType(tag & 7);
                            break;
                    }
                }
                return message;
            };

            ListDimOrders.decodeText = function (reader) {
                const message = new $root.uff.ListDimOrders();
                reader.start();
                while (!reader.end()) {
                    const tag = reader.tag();
                    switch (tag) {
                        case "val":
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
