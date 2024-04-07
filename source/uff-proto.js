
export const uff = {};

uff.MetaGraph = class MetaGraph {

    constructor() {
        this.descriptors = [];
        this.graphs = [];
        this.referenced_data = [];
        this.extra_fields = [];
    }

    static decode(reader, length) {
        const message = new uff.MetaGraph();
        const end = length === undefined ? reader.length : reader.position + length;
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
                    message.descriptors.push(uff.Descriptor.decode(reader, reader.uint32()));
                    break;
                case 4:
                    message.graphs.push(uff.Graph.decode(reader, reader.uint32()));
                    break;
                case 5:
                    message.referenced_data.push(uff.MetaGraph.ReferencedDataEntry.decode(reader, reader.uint32()));
                    break;
                case 100:
                    message.extra_fields.push(uff.MetaGraph.ExtraFieldsEntry.decode(reader, reader.uint32()));
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }

    static decodeText(reader) {
        const message = new uff.MetaGraph();
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
                    message.descriptors.push(uff.Descriptor.decodeText(reader));
                    break;
                case "graphs":
                    message.graphs.push(uff.Graph.decodeText(reader));
                    break;
                case "referenced_data":
                    message.referenced_data.push(uff.MetaGraph.ReferencedDataEntry.decodeText(reader));
                    break;
                case "extra_fields":
                    message.extra_fields.push(uff.MetaGraph.ExtraFieldsEntry.decodeText(reader));
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

uff.MetaGraph.prototype.version = 0n;
uff.MetaGraph.prototype.descriptor_core_version = 0n;

uff.MetaGraph.ReferencedDataEntry = class ReferencedDataEntry {

    static decode(reader, length) {
        const message = new uff.MetaGraph.ReferencedDataEntry();
        const end = length === undefined ? reader.length : reader.position + length;
        while (reader.position < end) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.key = reader.string();
                    break;
                case 2:
                    message.value = uff.Data.decode(reader, reader.uint32());
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }

    static decodeText(reader) {
        const message = new uff.MetaGraph.ReferencedDataEntry();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "key":
                    message.key = reader.string();
                    break;
                case "value":
                    message.value = uff.Data.decodeText(reader);
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

uff.MetaGraph.ReferencedDataEntry.prototype.key = "";
uff.MetaGraph.ReferencedDataEntry.prototype.value = null;

uff.MetaGraph.ExtraFieldsEntry = class ExtraFieldsEntry {

    static decode(reader, length) {
        const message = new uff.MetaGraph.ExtraFieldsEntry();
        const end = length === undefined ? reader.length : reader.position + length;
        while (reader.position < end) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.key = reader.string();
                    break;
                case 2:
                    message.value = uff.Data.decode(reader, reader.uint32());
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }

    static decodeText(reader) {
        const message = new uff.MetaGraph.ExtraFieldsEntry();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "key":
                    message.key = reader.string();
                    break;
                case "value":
                    message.value = uff.Data.decodeText(reader);
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

uff.MetaGraph.ExtraFieldsEntry.prototype.key = "";
uff.MetaGraph.ExtraFieldsEntry.prototype.value = null;

uff.Descriptor = class Descriptor {

    static decode(reader, length) {
        const message = new uff.Descriptor();
        const end = length === undefined ? reader.length : reader.position + length;
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
        const message = new uff.Descriptor();
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

uff.Descriptor.prototype.id = "";
uff.Descriptor.prototype.version = 0n;
uff.Descriptor.prototype.optional = false;

uff.Graph = class Graph {

    constructor() {
        this.nodes = [];
        this.extra_fields = [];
    }

    static decode(reader, length) {
        const message = new uff.Graph();
        const end = length === undefined ? reader.length : reader.position + length;
        while (reader.position < end) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.id = reader.string();
                    break;
                case 2:
                    message.nodes.push(uff.Node.decode(reader, reader.uint32()));
                    break;
                case 100:
                    message.extra_fields.push(uff.Graph.ExtraFieldsEntry.decode(reader, reader.uint32()));
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }

    static decodeText(reader) {
        const message = new uff.Graph();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "id":
                    message.id = reader.string();
                    break;
                case "nodes":
                    message.nodes.push(uff.Node.decodeText(reader));
                    break;
                case "extra_fields":
                    message.extra_fields.push(uff.Graph.ExtraFieldsEntry.decodeText(reader));
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

uff.Graph.prototype.id = "";

uff.Graph.ExtraFieldsEntry = class ExtraFieldsEntry {

    static decode(reader, length) {
        const message = new uff.Graph.ExtraFieldsEntry();
        const end = length === undefined ? reader.length : reader.position + length;
        while (reader.position < end) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.key = reader.string();
                    break;
                case 2:
                    message.value = uff.Data.decode(reader, reader.uint32());
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }

    static decodeText(reader) {
        const message = new uff.Graph.ExtraFieldsEntry();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "key":
                    message.key = reader.string();
                    break;
                case "value":
                    message.value = uff.Data.decodeText(reader);
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

uff.Graph.ExtraFieldsEntry.prototype.key = "";
uff.Graph.ExtraFieldsEntry.prototype.value = null;

uff.Node = class Node {

    constructor() {
        this.inputs = [];
        this.fields = [];
        this.extra_fields = [];
    }

    static decode(reader, length) {
        const message = new uff.Node();
        const end = length === undefined ? reader.length : reader.position + length;
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
                    message.fields.push(uff.Node.FieldsEntry.decode(reader, reader.uint32()));
                    break;
                case 100:
                    message.extra_fields.push(uff.Node.ExtraFieldsEntry.decode(reader, reader.uint32()));
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }

    static decodeText(reader) {
        const message = new uff.Node();
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
                    message.fields.push(uff.Node.FieldsEntry.decodeText(reader));
                    break;
                case "extra_fields":
                    message.extra_fields.push(uff.Node.ExtraFieldsEntry.decodeText(reader));
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

uff.Node.prototype.id = "";
uff.Node.prototype.operation = "";

uff.Node.FieldsEntry = class FieldsEntry {

    static decode(reader, length) {
        const message = new uff.Node.FieldsEntry();
        const end = length === undefined ? reader.length : reader.position + length;
        while (reader.position < end) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.key = reader.string();
                    break;
                case 2:
                    message.value = uff.Data.decode(reader, reader.uint32());
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }

    static decodeText(reader) {
        const message = new uff.Node.FieldsEntry();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "key":
                    message.key = reader.string();
                    break;
                case "value":
                    message.value = uff.Data.decodeText(reader);
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

uff.Node.FieldsEntry.prototype.key = "";
uff.Node.FieldsEntry.prototype.value = null;

uff.Node.ExtraFieldsEntry = class ExtraFieldsEntry {

    static decode(reader, length) {
        const message = new uff.Node.ExtraFieldsEntry();
        const end = length === undefined ? reader.length : reader.position + length;
        while (reader.position < end) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.key = reader.string();
                    break;
                case 2:
                    message.value = uff.Data.decode(reader, reader.uint32());
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }

    static decodeText(reader) {
        const message = new uff.Node.ExtraFieldsEntry();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "key":
                    message.key = reader.string();
                    break;
                case "value":
                    message.value = uff.Data.decodeText(reader);
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

uff.Node.ExtraFieldsEntry.prototype.key = "";
uff.Node.ExtraFieldsEntry.prototype.value = null;

uff.Data = class Data {

    get type() {
        uff.Data.typeSet = uff.Data.typeSet || new Set(["s", "s_list", "d", "d_list", "b", "b_list", "i", "i_list", "blob", "ref", "dtype", "dtype_list", "dim_orders", "dim_orders_list"]);
        return Object.keys(this).find((key) => uff.Data.typeSet.has(key) && this[key] !== null);
    }

    static decode(reader, length) {
        const message = new uff.Data();
        const end = length === undefined ? reader.length : reader.position + length;
        while (reader.position < end) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.s = reader.string();
                    break;
                case 2:
                    message.s_list = uff.ListString.decode(reader, reader.uint32());
                    break;
                case 3:
                    message.d = reader.double();
                    break;
                case 4:
                    message.d_list = uff.ListDouble.decode(reader, reader.uint32());
                    break;
                case 5:
                    message.b = reader.bool();
                    break;
                case 6:
                    message.b_list = uff.ListBool.decode(reader, reader.uint32());
                    break;
                case 7:
                    message.i = reader.int64();
                    break;
                case 8:
                    message.i_list = uff.ListInt64.decode(reader, reader.uint32());
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
                    message.dtype_list = uff.ListDataType.decode(reader, reader.uint32());
                    break;
                case 103:
                    message.dim_orders = uff.DimensionOrders.decode(reader, reader.uint32());
                    break;
                case 104:
                    message.dim_orders_list = uff.ListDimensionOrders.decode(reader, reader.uint32());
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }

    static decodeText(reader) {
        const message = new uff.Data();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "s":
                    message.s = reader.string();
                    break;
                case "s_list":
                    message.s_list = uff.ListString.decodeText(reader);
                    break;
                case "d":
                    message.d = reader.double();
                    break;
                case "d_list":
                    message.d_list = uff.ListDouble.decodeText(reader);
                    break;
                case "b":
                    message.b = reader.bool();
                    break;
                case "b_list":
                    message.b_list = uff.ListBool.decodeText(reader);
                    break;
                case "i":
                    message.i = reader.int64();
                    break;
                case "i_list":
                    message.i_list = uff.ListInt64.decodeText(reader);
                    break;
                case "blob":
                    message.blob = reader.bytes();
                    break;
                case "ref":
                    message.ref = reader.string();
                    break;
                case "dtype":
                    message.dtype = reader.enum(uff.DataType);
                    break;
                case "dtype_list":
                    message.dtype_list = uff.ListDataType.decodeText(reader);
                    break;
                case "dim_orders":
                    message.dim_orders = uff.DimensionOrders.decodeText(reader);
                    break;
                case "dim_orders_list":
                    message.dim_orders_list = uff.ListDimensionOrders.decodeText(reader);
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

uff.DataType = {
    "DT_INVALID": 0,
    "DT_INT8": 65544,
    "DT_INT16": 65552,
    "DT_INT32": 65568,
    "DT_INT64": 65600,
    "DT_FLOAT16": 131088,
    "DT_FLOAT32": 131104
};

uff.OrderEnum = {
    "OE_ZERO": 0,
    "OE_SPECIAL": -1,
    "OE_INCREMENT": 2147483647,
    "OE_DECREMENT": -2147483648
};

uff.DimensionOrders = class DimensionOrders {

    constructor() {
        this.orders = [];
    }

    static decode(reader, length) {
        const message = new uff.DimensionOrders();
        const end = length === undefined ? reader.length : reader.position + length;
        while (reader.position < end) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.orders.push(uff.DimensionOrders.OrdersEntry.decode(reader, reader.uint32()));
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }

    static decodeText(reader) {
        const message = new uff.DimensionOrders();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "orders":
                    message.orders.push(uff.DimensionOrders.OrdersEntry.decodeText(reader));
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

uff.DimensionOrders.OrdersEntry = class OrdersEntry {

    static decode(reader, length) {
        const message = new uff.DimensionOrders.OrdersEntry();
        const end = length === undefined ? reader.length : reader.position + length;
        while (reader.position < end) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.key = reader.int32();
                    break;
                case 2:
                    message.value = uff.ListInt64.decode(reader, reader.uint32());
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }

    static decodeText(reader) {
        const message = new uff.DimensionOrders.OrdersEntry();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "key":
                    message.key = reader.enum(uff.OrderEnum);
                    break;
                case "value":
                    message.value = uff.ListInt64.decodeText(reader);
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

uff.DimensionOrders.OrdersEntry.prototype.key = 0;
uff.DimensionOrders.OrdersEntry.prototype.value = null;

uff.ListString = class ListString {

    constructor() {
        this.val = [];
    }

    static decode(reader, length) {
        const message = new uff.ListString();
        const end = length === undefined ? reader.length : reader.position + length;
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
        const message = new uff.ListString();
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

uff.ListDouble = class ListDouble {

    constructor() {
        this.val = [];
    }

    static decode(reader, length) {
        const message = new uff.ListDouble();
        const end = length === undefined ? reader.length : reader.position + length;
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
        const message = new uff.ListDouble();
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

uff.ListBool = class ListBool {

    constructor() {
        this.val = [];
    }

    static decode(reader, length) {
        const message = new uff.ListBool();
        const end = length === undefined ? reader.length : reader.position + length;
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
        const message = new uff.ListBool();
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

uff.ListInt64 = class ListInt64 {

    constructor() {
        this.val = [];
    }

    static decode(reader, length) {
        const message = new uff.ListInt64();
        const end = length === undefined ? reader.length : reader.position + length;
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
        const message = new uff.ListInt64();
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

uff.ListDataType = class ListDataType {

    constructor() {
        this.val = [];
    }

    static decode(reader, length) {
        const message = new uff.ListDataType();
        const end = length === undefined ? reader.length : reader.position + length;
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
        const message = new uff.ListDataType();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "val":
                    reader.array(message.val, () => reader.enum(uff.DataType));
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

uff.ListDimensionOrders = class ListDimensionOrders {

    constructor() {
        this.val = [];
    }

    static decode(reader, length) {
        const message = new uff.ListDimensionOrders();
        const end = length === undefined ? reader.length : reader.position + length;
        while (reader.position < end) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.val.push(uff.DimensionOrders.decode(reader, reader.uint32()));
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }

    static decodeText(reader) {
        const message = new uff.ListDimensionOrders();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "val":
                    message.val.push(uff.DimensionOrders.decodeText(reader));
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};
