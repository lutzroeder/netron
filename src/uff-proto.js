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
                this.extra_fields = [];
            }

            MetaGraph.prototype.version = $protobuf.Long ? $protobuf.Long.fromBits(0, 0, false) : 0;
            MetaGraph.prototype.descriptor_core_version = $protobuf.Long ? $protobuf.Long.fromBits(0, 0, false) : 0;
            MetaGraph.prototype.descriptors = [];
            MetaGraph.prototype.graphs = [];
            MetaGraph.prototype.referenced_data = [];
            MetaGraph.prototype.extra_fields = [];

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
                            message.referenced_data.push($root.uff.MetaGraph.ReferencedDataEntry.decode(reader, reader.uint32()));
                            break;
                        case 100:
                            message.extra_fields.push($root.uff.MetaGraph.ExtraFieldsEntry.decode(reader, reader.uint32()));
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
                            message.referenced_data.push($root.uff.MetaGraph.ReferencedDataEntry.decodeText(reader, true));
                            break;
                        case "extra_fields":
                            message.extra_fields.push($root.uff.MetaGraph.ExtraFieldsEntry.decodeText(reader, true));
                            break;
                        default:
                            reader.field(tag, message);
                            break;
                    }
                }
                return message;
            };

            MetaGraph.ReferencedDataEntry = (function() {

                function ReferencedDataEntry() {
                }

                ReferencedDataEntry.prototype.key = "";
                ReferencedDataEntry.prototype.value = null;

                ReferencedDataEntry.decode = function (reader, length) {
                    const message = new $root.uff.MetaGraph.ReferencedDataEntry();
                    const end = reader.next(length);
                    while (reader.end(end)) {
                        const tag = reader.uint32();
                        switch (tag >>> 3) {
                            case 1:
                                message.key = reader.string();
                                break;
                            case 2:
                                message.value = $root.uff.Data.decode(reader, reader.uint32());
                                break;
                            default:
                                reader.skipType(tag & 7);
                                break;
                        }
                    }
                    return message;
                };

                ReferencedDataEntry.decodeText = function (reader) {
                    const message = new $root.uff.MetaGraph.ReferencedDataEntry();
                    reader.start();
                    while (!reader.end()) {
                        const tag = reader.tag();
                        switch (tag) {
                            case "key":
                                message.key = reader.string();
                                break;
                            case "value":
                                message.value = $root.uff.Data.decodeText(reader, true);
                                break;
                            default:
                                reader.field(tag, message);
                                break;
                        }
                    }
                    return message;
                };

                return ReferencedDataEntry;
            })();

            MetaGraph.ExtraFieldsEntry = (function() {

                function ExtraFieldsEntry() {
                }

                ExtraFieldsEntry.prototype.key = "";
                ExtraFieldsEntry.prototype.value = null;

                ExtraFieldsEntry.decode = function (reader, length) {
                    const message = new $root.uff.MetaGraph.ExtraFieldsEntry();
                    const end = reader.next(length);
                    while (reader.end(end)) {
                        const tag = reader.uint32();
                        switch (tag >>> 3) {
                            case 1:
                                message.key = reader.string();
                                break;
                            case 2:
                                message.value = $root.uff.Data.decode(reader, reader.uint32());
                                break;
                            default:
                                reader.skipType(tag & 7);
                                break;
                        }
                    }
                    return message;
                };

                ExtraFieldsEntry.decodeText = function (reader) {
                    const message = new $root.uff.MetaGraph.ExtraFieldsEntry();
                    reader.start();
                    while (!reader.end()) {
                        const tag = reader.tag();
                        switch (tag) {
                            case "key":
                                message.key = reader.string();
                                break;
                            case "value":
                                message.value = $root.uff.Data.decodeText(reader, true);
                                break;
                            default:
                                reader.field(tag, message);
                                break;
                        }
                    }
                    return message;
                };

                return ExtraFieldsEntry;
            })();

            return MetaGraph;
        })();

        uff.Descriptor = (function() {

            function Descriptor() {
            }

            Descriptor.prototype.id = "";
            Descriptor.prototype.version = $protobuf.Long ? $protobuf.Long.fromBits(0, 0, false) : 0;
            Descriptor.prototype.optional = false;

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
                        case 3:
                            message.optional = reader.bool();
                            break;
                        default:
                            reader.skipType(tag & 7);
                            break;
                    }
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
                        case "optional":
                            message.optional = reader.bool();
                            break;
                        default:
                            reader.field(tag, message);
                            break;
                    }
                }
                return message;
            };

            return Descriptor;
        })();

        uff.Graph = (function() {

            function Graph() {
                this.nodes = [];
                this.extra_fields = [];
            }

            Graph.prototype.id = "";
            Graph.prototype.nodes = [];
            Graph.prototype.extra_fields = [];

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
                        case 100:
                            message.extra_fields.push($root.uff.Graph.ExtraFieldsEntry.decode(reader, reader.uint32()));
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
                        case "extra_fields":
                            message.extra_fields.push($root.uff.Graph.ExtraFieldsEntry.decodeText(reader, true));
                            break;
                        default:
                            reader.field(tag, message);
                            break;
                    }
                }
                return message;
            };

            Graph.ExtraFieldsEntry = (function() {

                function ExtraFieldsEntry() {
                }

                ExtraFieldsEntry.prototype.key = "";
                ExtraFieldsEntry.prototype.value = null;

                ExtraFieldsEntry.decode = function (reader, length) {
                    const message = new $root.uff.Graph.ExtraFieldsEntry();
                    const end = reader.next(length);
                    while (reader.end(end)) {
                        const tag = reader.uint32();
                        switch (tag >>> 3) {
                            case 1:
                                message.key = reader.string();
                                break;
                            case 2:
                                message.value = $root.uff.Data.decode(reader, reader.uint32());
                                break;
                            default:
                                reader.skipType(tag & 7);
                                break;
                        }
                    }
                    return message;
                };

                ExtraFieldsEntry.decodeText = function (reader) {
                    const message = new $root.uff.Graph.ExtraFieldsEntry();
                    reader.start();
                    while (!reader.end()) {
                        const tag = reader.tag();
                        switch (tag) {
                            case "key":
                                message.key = reader.string();
                                break;
                            case "value":
                                message.value = $root.uff.Data.decodeText(reader, true);
                                break;
                            default:
                                reader.field(tag, message);
                                break;
                        }
                    }
                    return message;
                };

                return ExtraFieldsEntry;
            })();

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
                            message.fields.push($root.uff.Node.FieldsEntry.decode(reader, reader.uint32()));
                            break;
                        case 100:
                            message.extra_fields.push($root.uff.Node.ExtraFieldsEntry.decode(reader, reader.uint32()));
                            break;
                        default:
                            reader.skipType(tag & 7);
                            break;
                    }
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
                            message.fields.push($root.uff.Node.FieldsEntry.decodeText(reader, true));
                            break;
                        case "extra_fields":
                            message.extra_fields.push($root.uff.Node.ExtraFieldsEntry.decodeText(reader, true));
                            break;
                        default:
                            reader.field(tag, message);
                            break;
                    }
                }
                return message;
            };

            Node.FieldsEntry = (function() {

                function FieldsEntry() {
                }

                FieldsEntry.prototype.key = "";
                FieldsEntry.prototype.value = null;

                FieldsEntry.decode = function (reader, length) {
                    const message = new $root.uff.Node.FieldsEntry();
                    const end = reader.next(length);
                    while (reader.end(end)) {
                        const tag = reader.uint32();
                        switch (tag >>> 3) {
                            case 1:
                                message.key = reader.string();
                                break;
                            case 2:
                                message.value = $root.uff.Data.decode(reader, reader.uint32());
                                break;
                            default:
                                reader.skipType(tag & 7);
                                break;
                        }
                    }
                    return message;
                };

                FieldsEntry.decodeText = function (reader) {
                    const message = new $root.uff.Node.FieldsEntry();
                    reader.start();
                    while (!reader.end()) {
                        const tag = reader.tag();
                        switch (tag) {
                            case "key":
                                message.key = reader.string();
                                break;
                            case "value":
                                message.value = $root.uff.Data.decodeText(reader, true);
                                break;
                            default:
                                reader.field(tag, message);
                                break;
                        }
                    }
                    return message;
                };

                return FieldsEntry;
            })();

            Node.ExtraFieldsEntry = (function() {

                function ExtraFieldsEntry() {
                }

                ExtraFieldsEntry.prototype.key = "";
                ExtraFieldsEntry.prototype.value = null;

                ExtraFieldsEntry.decode = function (reader, length) {
                    const message = new $root.uff.Node.ExtraFieldsEntry();
                    const end = reader.next(length);
                    while (reader.end(end)) {
                        const tag = reader.uint32();
                        switch (tag >>> 3) {
                            case 1:
                                message.key = reader.string();
                                break;
                            case 2:
                                message.value = $root.uff.Data.decode(reader, reader.uint32());
                                break;
                            default:
                                reader.skipType(tag & 7);
                                break;
                        }
                    }
                    return message;
                };

                ExtraFieldsEntry.decodeText = function (reader) {
                    const message = new $root.uff.Node.ExtraFieldsEntry();
                    reader.start();
                    while (!reader.end()) {
                        const tag = reader.tag();
                        switch (tag) {
                            case "key":
                                message.key = reader.string();
                                break;
                            case "value":
                                message.value = $root.uff.Data.decodeText(reader, true);
                                break;
                            default:
                                reader.field(tag, message);
                                break;
                        }
                    }
                    return message;
                };

                return ExtraFieldsEntry;
            })();

            return Node;
        })();

        uff.Data = (function() {

            function Data() {
            }

            Data.prototype.s = "";
            Data.prototype.s_list = null;
            Data.prototype.d = 0;
            Data.prototype.d_list = null;
            Data.prototype.b = false;
            Data.prototype.b_list = null;
            Data.prototype.i = $protobuf.Long ? $protobuf.Long.fromBits(0, 0, false) : 0;
            Data.prototype.i_list = null;
            Data.prototype.blob = new Uint8Array([]);
            Data.prototype.ref = "";
            Data.prototype.dtype = 0;
            Data.prototype.dtype_list = null;
            Data.prototype.dim_orders = null;
            Data.prototype.dim_orders_list = null;

            const typeSet = new Set([ "s", "s_list", "d", "d_list", "b", "b_list", "i", "i_list", "blob", "ref", "dtype", "dtype_list", "dim_orders", "dim_orders_list"]);
            Object.defineProperty(Data.prototype, "type", {
                get: function() { return Object.keys(this).find((key) => typeSet.has(key) && this[key] != null); }
            });

            Data.decode = function (reader, length) {
                const message = new $root.uff.Data();
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
                            message.b_list = $root.uff.ListBool.decode(reader, reader.uint32());
                            break;
                        case 7:
                            message.i = reader.int64();
                            break;
                        case 8:
                            message.i_list = $root.uff.ListInt64.decode(reader, reader.uint32());
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
                            message.dim_orders = $root.uff.DimensionOrders.decode(reader, reader.uint32());
                            break;
                        case 104:
                            message.dim_orders_list = $root.uff.ListDimensionOrders.decode(reader, reader.uint32());
                            break;
                        default:
                            reader.skipType(tag & 7);
                            break;
                    }
                }
                return message;
            };

            Data.decodeText = function (reader) {
                const message = new $root.uff.Data();
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
                            message.b_list = $root.uff.ListBool.decodeText(reader, true);
                            break;
                        case "i":
                            message.i = reader.int64();
                            break;
                        case "i_list":
                            message.i_list = $root.uff.ListInt64.decodeText(reader, true);
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
                            message.dim_orders = $root.uff.DimensionOrders.decodeText(reader, true);
                            break;
                        case "dim_orders_list":
                            message.dim_orders_list = $root.uff.ListDimensionOrders.decodeText(reader, true);
                            break;
                        default:
                            reader.field(tag, message);
                            break;
                    }
                }
                return message;
            };

            return Data;
        })();

        uff.DataType = (function() {
            const values = {};
            values["DT_INVALID"] = 0;
            values["DT_INT8"] = 65544;
            values["DT_INT16"] = 65552;
            values["DT_INT32"] = 65568;
            values["DT_INT64"] = 65600;
            values["DT_FLOAT16"] = 131088;
            values["DT_FLOAT32"] = 131104;
            return values;
        })();

        uff.OrderEnum = (function() {
            const values = {};
            values["OE_ZERO"] = 0;
            values["OE_SPECIAL"] = -1;
            values["OE_INCREMENT"] = 2147483647;
            values["OE_DECREMENT"] = -2147483648;
            return values;
        })();

        uff.DimensionOrders = (function() {

            function DimensionOrders() {
                this.orders = [];
            }

            DimensionOrders.prototype.orders = [];

            DimensionOrders.decode = function (reader, length) {
                const message = new $root.uff.DimensionOrders();
                const end = reader.next(length);
                while (reader.end(end)) {
                    const tag = reader.uint32();
                    switch (tag >>> 3) {
                        case 1:
                            message.orders.push($root.uff.DimensionOrders.OrdersEntry.decode(reader, reader.uint32()));
                            break;
                        default:
                            reader.skipType(tag & 7);
                            break;
                    }
                }
                return message;
            };

            DimensionOrders.decodeText = function (reader) {
                const message = new $root.uff.DimensionOrders();
                reader.start();
                while (!reader.end()) {
                    const tag = reader.tag();
                    switch (tag) {
                        case "orders":
                            message.orders.push($root.uff.DimensionOrders.OrdersEntry.decodeText(reader, true));
                            break;
                        default:
                            reader.field(tag, message);
                            break;
                    }
                }
                return message;
            };

            DimensionOrders.OrdersEntry = (function() {

                function OrdersEntry() {
                }

                OrdersEntry.prototype.key = 0;
                OrdersEntry.prototype.value = null;

                OrdersEntry.decode = function (reader, length) {
                    const message = new $root.uff.DimensionOrders.OrdersEntry();
                    const end = reader.next(length);
                    while (reader.end(end)) {
                        const tag = reader.uint32();
                        switch (tag >>> 3) {
                            case 1:
                                message.key = reader.int32();
                                break;
                            case 2:
                                message.value = $root.uff.ListInt64.decode(reader, reader.uint32());
                                break;
                            default:
                                reader.skipType(tag & 7);
                                break;
                        }
                    }
                    return message;
                };

                OrdersEntry.decodeText = function (reader) {
                    const message = new $root.uff.DimensionOrders.OrdersEntry();
                    reader.start();
                    while (!reader.end()) {
                        const tag = reader.tag();
                        switch (tag) {
                            case "key":
                                message.key = reader.enum($root.uff.OrderEnum);
                                break;
                            case "value":
                                message.value = $root.uff.ListInt64.decodeText(reader, true);
                                break;
                            default:
                                reader.field(tag, message);
                                break;
                        }
                    }
                    return message;
                };

                return OrdersEntry;
            })();

            return DimensionOrders;
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

        uff.ListBool = (function() {

            function ListBool() {
                this.val = [];
            }

            ListBool.prototype.val = [];

            ListBool.decode = function (reader, length) {
                const message = new $root.uff.ListBool();
                const end = reader.next(length);
                while (reader.end(end)) {
                    const tag = reader.uint32();
                    switch (tag >>> 3) {
                        case 1:
                            message.val = reader.array(message.val, () => reader.bool(), tag);
                            break;
                        default:
                            reader.skipType(tag & 7);
                            break;
                    }
                }
                return message;
            };

            ListBool.decodeText = function (reader) {
                const message = new $root.uff.ListBool();
                reader.start();
                while (!reader.end()) {
                    const tag = reader.tag();
                    switch (tag) {
                        case "val":
                            reader.array(message.val, () => reader.bool());
                            break;
                        default:
                            reader.field(tag, message);
                            break;
                    }
                }
                return message;
            };

            return ListBool;
        })();

        uff.ListInt64 = (function() {

            function ListInt64() {
                this.val = [];
            }

            ListInt64.prototype.val = [];

            ListInt64.decode = function (reader, length) {
                const message = new $root.uff.ListInt64();
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

            ListInt64.decodeText = function (reader) {
                const message = new $root.uff.ListInt64();
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

            return ListInt64;
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

        uff.ListDimensionOrders = (function() {

            function ListDimensionOrders() {
                this.val = [];
            }

            ListDimensionOrders.prototype.val = [];

            ListDimensionOrders.decode = function (reader, length) {
                const message = new $root.uff.ListDimensionOrders();
                const end = reader.next(length);
                while (reader.end(end)) {
                    const tag = reader.uint32();
                    switch (tag >>> 3) {
                        case 1:
                            message.val.push($root.uff.DimensionOrders.decode(reader, reader.uint32()));
                            break;
                        default:
                            reader.skipType(tag & 7);
                            break;
                    }
                }
                return message;
            };

            ListDimensionOrders.decodeText = function (reader) {
                const message = new $root.uff.ListDimensionOrders();
                reader.start();
                while (!reader.end()) {
                    const tag = reader.tag();
                    switch (tag) {
                        case "val":
                            message.val.push($root.uff.DimensionOrders.decodeText(reader, true));
                            break;
                        default:
                            reader.field(tag, message);
                            break;
                    }
                }
                return message;
            };

            return ListDimensionOrders;
        })();

        return uff;
    })();
    return $root;
})(protobuf);
