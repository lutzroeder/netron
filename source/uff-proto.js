var $root = protobuf.get('uff');

$root.uff = {};

$root.uff.MetaGraph = class MetaGraph {

    constructor() {
        this.descriptors = [];
        this.graphs = [];
        this.referenced_data = [];
        this.extra_fields = [];
    }

    static decode(reader, length) {
        const message = new $root.uff.MetaGraph();
        const end = length !== undefined ? reader.position + length : reader.length;
        while (reader.position < end) {
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
    }

    static decodeText(reader) {
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
                    message.descriptors.push($root.uff.Descriptor.decodeText(reader));
                    break;
                case "graphs":
                    message.graphs.push($root.uff.Graph.decodeText(reader));
                    break;
                case "referenced_data":
                    message.referenced_data.push($root.uff.MetaGraph.ReferencedDataEntry.decodeText(reader));
                    break;
                case "extra_fields":
                    message.extra_fields.push($root.uff.MetaGraph.ExtraFieldsEntry.decodeText(reader));
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

$root.uff.MetaGraph.prototype.version = protobuf.Int64.create(0);
$root.uff.MetaGraph.prototype.descriptor_core_version = protobuf.Int64.create(0);

$root.uff.MetaGraph.ReferencedDataEntry = class ReferencedDataEntry {

    constructor() {
    }

    static decode(reader, length) {
        const message = new $root.uff.MetaGraph.ReferencedDataEntry();
        const end = length !== undefined ? reader.position + length : reader.length;
        while (reader.position < end) {
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
    }

    static decodeText(reader) {
        const message = new $root.uff.MetaGraph.ReferencedDataEntry();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "key":
                    message.key = reader.string();
                    break;
                case "value":
                    message.value = $root.uff.Data.decodeText(reader);
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

$root.uff.MetaGraph.ReferencedDataEntry.prototype.key = "";
$root.uff.MetaGraph.ReferencedDataEntry.prototype.value = null;

$root.uff.MetaGraph.ExtraFieldsEntry = class ExtraFieldsEntry {

    constructor() {
    }

    static decode(reader, length) {
        const message = new $root.uff.MetaGraph.ExtraFieldsEntry();
        const end = length !== undefined ? reader.position + length : reader.length;
        while (reader.position < end) {
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
    }

    static decodeText(reader) {
        const message = new $root.uff.MetaGraph.ExtraFieldsEntry();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "key":
                    message.key = reader.string();
                    break;
                case "value":
                    message.value = $root.uff.Data.decodeText(reader);
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

$root.uff.MetaGraph.ExtraFieldsEntry.prototype.key = "";
$root.uff.MetaGraph.ExtraFieldsEntry.prototype.value = null;

$root.uff.Descriptor = class Descriptor {

    constructor() {
    }

    static decode(reader, length) {
        const message = new $root.uff.Descriptor();
        const end = length !== undefined ? reader.position + length : reader.length;
        while (reader.position < end) {
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
    }

    static decodeText(reader) {
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
    }
};

$root.uff.Descriptor.prototype.id = "";
$root.uff.Descriptor.prototype.version = protobuf.Int64.create(0);
$root.uff.Descriptor.prototype.optional = false;

$root.uff.Graph = class Graph {

    constructor() {
        this.nodes = [];
        this.extra_fields = [];
    }

    static decode(reader, length) {
        const message = new $root.uff.Graph();
        const end = length !== undefined ? reader.position + length : reader.length;
        while (reader.position < end) {
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
    }

    static decodeText(reader) {
        const message = new $root.uff.Graph();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "id":
                    message.id = reader.string();
                    break;
                case "nodes":
                    message.nodes.push($root.uff.Node.decodeText(reader));
                    break;
                case "extra_fields":
                    message.extra_fields.push($root.uff.Graph.ExtraFieldsEntry.decodeText(reader));
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

$root.uff.Graph.prototype.id = "";

$root.uff.Graph.ExtraFieldsEntry = class ExtraFieldsEntry {

    constructor() {
    }

    static decode(reader, length) {
        const message = new $root.uff.Graph.ExtraFieldsEntry();
        const end = length !== undefined ? reader.position + length : reader.length;
        while (reader.position < end) {
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
    }

    static decodeText(reader) {
        const message = new $root.uff.Graph.ExtraFieldsEntry();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "key":
                    message.key = reader.string();
                    break;
                case "value":
                    message.value = $root.uff.Data.decodeText(reader);
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

$root.uff.Graph.ExtraFieldsEntry.prototype.key = "";
$root.uff.Graph.ExtraFieldsEntry.prototype.value = null;

$root.uff.Node = class Node {

    constructor() {
        this.inputs = [];
        this.fields = [];
        this.extra_fields = [];
    }

    static decode(reader, length) {
        const message = new $root.uff.Node();
        const end = length !== undefined ? reader.position + length : reader.length;
        while (reader.position < end) {
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
    }

    static decodeText(reader) {
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
                    message.fields.push($root.uff.Node.FieldsEntry.decodeText(reader));
                    break;
                case "extra_fields":
                    message.extra_fields.push($root.uff.Node.ExtraFieldsEntry.decodeText(reader));
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

$root.uff.Node.prototype.id = "";
$root.uff.Node.prototype.operation = "";

$root.uff.Node.FieldsEntry = class FieldsEntry {

    constructor() {
    }

    static decode(reader, length) {
        const message = new $root.uff.Node.FieldsEntry();
        const end = length !== undefined ? reader.position + length : reader.length;
        while (reader.position < end) {
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
    }

    static decodeText(reader) {
        const message = new $root.uff.Node.FieldsEntry();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "key":
                    message.key = reader.string();
                    break;
                case "value":
                    message.value = $root.uff.Data.decodeText(reader);
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

$root.uff.Node.FieldsEntry.prototype.key = "";
$root.uff.Node.FieldsEntry.prototype.value = null;

$root.uff.Node.ExtraFieldsEntry = class ExtraFieldsEntry {

    constructor() {
    }

    static decode(reader, length) {
        const message = new $root.uff.Node.ExtraFieldsEntry();
        const end = length !== undefined ? reader.position + length : reader.length;
        while (reader.position < end) {
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
    }

    static decodeText(reader) {
        const message = new $root.uff.Node.ExtraFieldsEntry();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "key":
                    message.key = reader.string();
                    break;
                case "value":
                    message.value = $root.uff.Data.decodeText(reader);
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

$root.uff.Node.ExtraFieldsEntry.prototype.key = "";
$root.uff.Node.ExtraFieldsEntry.prototype.value = null;

$root.uff.Data = class Data {

    constructor() {
    }

    get type() {
        $root.uff.Data.typeSet = $root.uff.Data.typeSet || new Set([ "s", "s_list", "d", "d_list", "b", "b_list", "i", "i_list", "blob", "ref", "dtype", "dtype_list", "dim_orders", "dim_orders_list"]);
        return Object.keys(this).find((key) => $root.uff.Data.typeSet.has(key) && this[key] != null);
    }

    static decode(reader, length) {
        const message = new $root.uff.Data();
        const end = length !== undefined ? reader.position + length : reader.length;
        while (reader.position < end) {
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
    }

    static decodeText(reader) {
        const message = new $root.uff.Data();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "s":
                    message.s = reader.string();
                    break;
                case "s_list":
                    message.s_list = $root.uff.ListString.decodeText(reader);
                    break;
                case "d":
                    message.d = reader.double();
                    break;
                case "d_list":
                    message.d_list = $root.uff.ListDouble.decodeText(reader);
                    break;
                case "b":
                    message.b = reader.bool();
                    break;
                case "b_list":
                    message.b_list = $root.uff.ListBool.decodeText(reader);
                    break;
                case "i":
                    message.i = reader.int64();
                    break;
                case "i_list":
                    message.i_list = $root.uff.ListInt64.decodeText(reader);
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
                    message.dtype_list = $root.uff.ListDataType.decodeText(reader);
                    break;
                case "dim_orders":
                    message.dim_orders = $root.uff.DimensionOrders.decodeText(reader);
                    break;
                case "dim_orders_list":
                    message.dim_orders_list = $root.uff.ListDimensionOrders.decodeText(reader);
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

$root.uff.DataType = {
    "DT_INVALID": 0,
    "DT_INT8": 65544,
    "DT_INT16": 65552,
    "DT_INT32": 65568,
    "DT_INT64": 65600,
    "DT_FLOAT16": 131088,
    "DT_FLOAT32": 131104
};

$root.uff.OrderEnum = {
    "OE_ZERO": 0,
    "OE_SPECIAL": -1,
    "OE_INCREMENT": 2147483647,
    "OE_DECREMENT": -2147483648
};

$root.uff.DimensionOrders = class DimensionOrders {

    constructor() {
        this.orders = [];
    }

    static decode(reader, length) {
        const message = new $root.uff.DimensionOrders();
        const end = length !== undefined ? reader.position + length : reader.length;
        while (reader.position < end) {
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
    }

    static decodeText(reader) {
        const message = new $root.uff.DimensionOrders();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "orders":
                    message.orders.push($root.uff.DimensionOrders.OrdersEntry.decodeText(reader));
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

$root.uff.DimensionOrders.OrdersEntry = class OrdersEntry {

    constructor() {
    }

    static decode(reader, length) {
        const message = new $root.uff.DimensionOrders.OrdersEntry();
        const end = length !== undefined ? reader.position + length : reader.length;
        while (reader.position < end) {
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
    }

    static decodeText(reader) {
        const message = new $root.uff.DimensionOrders.OrdersEntry();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "key":
                    message.key = reader.enum($root.uff.OrderEnum);
                    break;
                case "value":
                    message.value = $root.uff.ListInt64.decodeText(reader);
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

$root.uff.DimensionOrders.OrdersEntry.prototype.key = 0;
$root.uff.DimensionOrders.OrdersEntry.prototype.value = null;

$root.uff.ListString = class ListString {

    constructor() {
        this.val = [];
    }

    static decode(reader, length) {
        const message = new $root.uff.ListString();
        const end = length !== undefined ? reader.position + length : reader.length;
        while (reader.position < end) {
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
    }

    static decodeText(reader) {
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
    }
};

$root.uff.ListDouble = class ListDouble {

    constructor() {
        this.val = [];
    }

    static decode(reader, length) {
        const message = new $root.uff.ListDouble();
        const end = length !== undefined ? reader.position + length : reader.length;
        while (reader.position < end) {
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
    }

    static decodeText(reader) {
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
    }
};

$root.uff.ListBool = class ListBool {

    constructor() {
        this.val = [];
    }

    static decode(reader, length) {
        const message = new $root.uff.ListBool();
        const end = length !== undefined ? reader.position + length : reader.length;
        while (reader.position < end) {
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
    }

    static decodeText(reader) {
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
    }
};

$root.uff.ListInt64 = class ListInt64 {

    constructor() {
        this.val = [];
    }

    static decode(reader, length) {
        const message = new $root.uff.ListInt64();
        const end = length !== undefined ? reader.position + length : reader.length;
        while (reader.position < end) {
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
    }

    static decodeText(reader) {
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
    }
};

$root.uff.ListDataType = class ListDataType {

    constructor() {
        this.val = [];
    }

    static decode(reader, length) {
        const message = new $root.uff.ListDataType();
        const end = length !== undefined ? reader.position + length : reader.length;
        while (reader.position < end) {
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
    }

    static decodeText(reader) {
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
    }
};

$root.uff.ListDimensionOrders = class ListDimensionOrders {

    constructor() {
        this.val = [];
    }

    static decode(reader, length) {
        const message = new $root.uff.ListDimensionOrders();
        const end = length !== undefined ? reader.position + length : reader.length;
        while (reader.position < end) {
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
    }

    static decodeText(reader) {
        const message = new $root.uff.ListDimensionOrders();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "val":
                    message.val.push($root.uff.DimensionOrders.decodeText(reader));
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};
