
export const serial_v2 = {};

serial_v2.Graph = class Graph {

    constructor() {
        this.op_node = [];
        this.graph_attr = {};
        this.op_defs = [];
    }

    static decode(reader, length) {
        const message = new serial_v2.Graph();
        const end = length === undefined ? reader.length : reader.position + length;
        while (reader.position < end) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.graph_name = reader.string();
                    break;
                case 5:
                    message.op_node.push(serial_v2.OPNode.decode(reader, reader.uint32()));
                    break;
                case 10:
                    message.subg_root = serial_v2.SubGraph.decode(reader, reader.uint32());
                    break;
                case 11:
                    reader.entry(message.graph_attr, () => reader.string(), () => serial_v2.AttrValue.decode(reader, reader.uint32()));
                    break;
                case 101:
                    message.op_defs.push(serial_v2.OpDef.decode(reader, reader.uint32()));
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }

    static decodeText(reader) {
        const message = new serial_v2.Graph();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "graph_name":
                    message.graph_name = reader.string();
                    break;
                case "op_node":
                    message.op_node.push(serial_v2.OPNode.decodeText(reader));
                    break;
                case "subg_root":
                    message.subg_root = serial_v2.SubGraph.decodeText(reader);
                    break;
                case "graph_attr":
                    reader.entry(message.graph_attr, () => reader.string(), () => serial_v2.AttrValue.decodeText(reader));
                    break;
                case "op_defs":
                    message.op_defs.push(serial_v2.OpDef.decodeText(reader));
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

serial_v2.Graph.prototype.graph_name = "";
serial_v2.Graph.prototype.subg_root = null;

serial_v2.OPNode = class OPNode {

    constructor() {
        this.op_attr = {};
        this.args = [];
    }

    static decode(reader, length) {
        const message = new serial_v2.OPNode();
        const end = length === undefined ? reader.length : reader.position + length;
        while (reader.position < end) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.op_name = reader.string();
                    break;
                case 2:
                    message.op_type = reader.string();
                    break;
                case 3:
                    reader.entry(message.op_attr, () => reader.string(), () => serial_v2.AttrValue.decode(reader, reader.uint32()));
                    break;
                case 4:
                    message.args.push(serial_v2.OpArg.decode(reader, reader.uint32()));
                    break;
                case 5:
                    message.output_tensor = serial_v2.Tensor.decode(reader, reader.uint32());
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }

    static decodeText(reader) {
        const message = new serial_v2.OPNode();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "op_name":
                    message.op_name = reader.string();
                    break;
                case "op_type":
                    message.op_type = reader.string();
                    break;
                case "op_attr":
                    reader.entry(message.op_attr, () => reader.string(), () => serial_v2.AttrValue.decodeText(reader));
                    break;
                case "args":
                    message.args.push(serial_v2.OpArg.decodeText(reader));
                    break;
                case "output_tensor":
                    message.output_tensor = serial_v2.Tensor.decodeText(reader);
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

serial_v2.OPNode.prototype.op_name = "";
serial_v2.OPNode.prototype.op_type = "";
serial_v2.OPNode.prototype.output_tensor = null;

serial_v2.OpArg = class OpArg {

    constructor() {
        this.arg_ops = [];
    }

    static decode(reader, length) {
        const message = new serial_v2.OpArg();
        const end = length === undefined ? reader.length : reader.position + length;
        while (reader.position < end) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.arg_name = reader.string();
                    break;
                case 2:
                    message.arg_ops.push(reader.string());
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }

    static decodeText(reader) {
        const message = new serial_v2.OpArg();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "arg_name":
                    message.arg_name = reader.string();
                    break;
                case "arg_ops":
                    reader.array(message.arg_ops, () => reader.string());
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

serial_v2.OpArg.prototype.arg_name = "";

serial_v2.Tensor = class Tensor {

    constructor() {
        this.tensor_dim = [];
        this.tensor_attr = {};
    }

    static decode(reader, length) {
        const message = new serial_v2.Tensor();
        const end = length === undefined ? reader.length : reader.position + length;
        while (reader.position < end) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.tensor_name = reader.string();
                    break;
                case 2:
                    message.tensor_dim = reader.array(message.tensor_dim, () => reader.uint32(), tag);
                    break;
                case 5:
                    message.data_type = reader.int32();
                    break;
                case 6:
                    message.tensor_bit_width = reader.int32();
                    break;
                case 10:
                    reader.entry(message.tensor_attr, () => reader.string(), () => serial_v2.AttrValue.decode(reader, reader.uint32()));
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }

    static decodeText(reader) {
        const message = new serial_v2.Tensor();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "tensor_name":
                    message.tensor_name = reader.string();
                    break;
                case "tensor_dim":
                    reader.array(message.tensor_dim, () => reader.uint32());
                    break;
                case "data_type":
                    message.data_type = reader.int32();
                    break;
                case "tensor_bit_width":
                    message.tensor_bit_width = reader.int32();
                    break;
                case "tensor_attr":
                    reader.entry(message.tensor_attr, () => reader.string(), () => serial_v2.AttrValue.decodeText(reader));
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

serial_v2.Tensor.prototype.tensor_name = "";
serial_v2.Tensor.prototype.data_type = 0;
serial_v2.Tensor.prototype.tensor_bit_width = 0;

serial_v2.SubGraph = class SubGraph {

    constructor() {
        this.op_name = [];
        this.subg_attr = {};
        this.subg_child = [];
    }

    static decode(reader, length) {
        const message = new serial_v2.SubGraph();
        const end = length === undefined ? reader.length : reader.position + length;
        while (reader.position < end) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.subgraph_name = reader.string();
                    break;
                case 3:
                    message.op_name.push(reader.string());
                    break;
                case 5:
                    reader.entry(message.subg_attr, () => reader.string(), () => serial_v2.AttrValue.decode(reader, reader.uint32()));
                    break;
                case 10:
                    message.subg_child.push(serial_v2.SubGraph.decode(reader, reader.uint32()));
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }

    static decodeText(reader) {
        const message = new serial_v2.SubGraph();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "subgraph_name":
                    message.subgraph_name = reader.string();
                    break;
                case "op_name":
                    reader.array(message.op_name, () => reader.string());
                    break;
                case "subg_attr":
                    reader.entry(message.subg_attr, () => reader.string(), () => serial_v2.AttrValue.decodeText(reader));
                    break;
                case "subg_child":
                    message.subg_child.push(serial_v2.SubGraph.decodeText(reader));
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

serial_v2.SubGraph.prototype.subgraph_name = "";

serial_v2.OpDef = class OpDef {

    constructor() {
        this.input_args = [];
        this.attrs = [];
    }

    static decode(reader, length) {
        const message = new serial_v2.OpDef();
        const end = length === undefined ? reader.length : reader.position + length;
        while (reader.position < end) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.name = reader.string();
                    break;
                case 2:
                    message.input_args.push(serial_v2.OpArgDef.decode(reader, reader.uint32()));
                    break;
                case 3:
                    message.attrs.push(serial_v2.AttrDef.decode(reader, reader.uint32()));
                    break;
                case 4:
                    message.annotation = reader.string();
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }

    static decodeText(reader) {
        const message = new serial_v2.OpDef();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "name":
                    message.name = reader.string();
                    break;
                case "input_args":
                    message.input_args.push(serial_v2.OpArgDef.decodeText(reader));
                    break;
                case "attrs":
                    message.attrs.push(serial_v2.AttrDef.decodeText(reader));
                    break;
                case "annotation":
                    message.annotation = reader.string();
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

serial_v2.OpDef.prototype.name = "";
serial_v2.OpDef.prototype.annotation = "";

serial_v2.AttrDef = class AttrDef {

    static decode(reader, length) {
        const message = new serial_v2.AttrDef();
        const end = length === undefined ? reader.length : reader.position + length;
        while (reader.position < end) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.name = reader.string();
                    break;
                case 3:
                    message.occur_type = reader.int32();
                    break;
                case 4:
                    message.default_value = serial_v2.AttrValue.decode(reader, reader.uint32());
                    break;
                case 6:
                    message.list_length = reader.int32();
                    break;
                case 7:
                    message.annotation = reader.string();
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }

    static decodeText(reader) {
        const message = new serial_v2.AttrDef();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "name":
                    message.name = reader.string();
                    break;
                case "occur_type":
                    message.occur_type = reader.enum(serial_v2.AttrDef.OccurType);
                    break;
                case "default_value":
                    message.default_value = serial_v2.AttrValue.decodeText(reader);
                    break;
                case "list_length":
                    message.list_length = reader.int32();
                    break;
                case "annotation":
                    message.annotation = reader.string();
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

serial_v2.AttrDef.prototype.name = "";
serial_v2.AttrDef.prototype.occur_type = 0;
serial_v2.AttrDef.prototype.default_value = null;
serial_v2.AttrDef.prototype.list_length = 0;
serial_v2.AttrDef.prototype.annotation = "";

serial_v2.AttrDef.OccurType = {
    "REQUIRED": 0,
    "OPTIONAL": 1
};

serial_v2.OpArgDef = class OpArgDef {

    static decode(reader, length) {
        const message = new serial_v2.OpArgDef();
        const end = length === undefined ? reader.length : reader.position + length;
        while (reader.position < end) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.name = reader.string();
                    break;
                case 2:
                    message.occur_type = reader.int32();
                    break;
                case 3:
                    message.data_type = reader.int32();
                    break;
                case 4:
                    message.annotation = reader.string();
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }

    static decodeText(reader) {
        const message = new serial_v2.OpArgDef();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "name":
                    message.name = reader.string();
                    break;
                case "occur_type":
                    message.occur_type = reader.enum(serial_v2.OpArgDef.OccurType);
                    break;
                case "data_type":
                    message.data_type = reader.int32();
                    break;
                case "annotation":
                    message.annotation = reader.string();
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

serial_v2.OpArgDef.prototype.name = "";
serial_v2.OpArgDef.prototype.occur_type = 0;
serial_v2.OpArgDef.prototype.data_type = 0;
serial_v2.OpArgDef.prototype.annotation = "";

serial_v2.OpArgDef.OccurType = {
    "REQUIRED": 0,
    "OPTIONAL": 1,
    "REPEATED": 2,
    "REQUIRED_AND_REPEATED": 3
};

serial_v2.AttrValue = class AttrValue {

    get value() {
        serial_v2.AttrValue.valueSet = serial_v2.AttrValue.valueSet || new Set(["bool_value", "int32_value", "uint32_value", "int64_value", "uint64_value", "float_value", "double_value", "string_value", "bytes_value", "bool_vec_value", "int32_vec_value", "uint32_vec_value", "int64_vec_value", "uint64_vec_value", "float_vec_value", "double_vec_value", "string_vec_value", "bytes_vec_value", "map_string_2_int32_value", "map_string_2_uint32_value", "map_string_2_int64_value", "map_string_2_uint64_value", "map_string_2_string_value", "map_string_2_bytes_value", "map_string_2_int32_vec_value", "map_string_2_uint32_vec_value", "map_string_2_int64_vec_value", "map_string_2_uint64_vec_value", "map_string_2_string_vec_value"]);
        return Object.keys(this).find((key) => serial_v2.AttrValue.valueSet.has(key) && this[key] !== null);
    }

    static decode(reader, length) {
        const message = new serial_v2.AttrValue();
        const end = length === undefined ? reader.length : reader.position + length;
        while (reader.position < end) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 4:
                    message.bool_value = reader.bool();
                    break;
                case 5:
                    message.int32_value = reader.int32();
                    break;
                case 6:
                    message.uint32_value = reader.uint32();
                    break;
                case 7:
                    message.int64_value = reader.int64();
                    break;
                case 8:
                    message.uint64_value = reader.uint64();
                    break;
                case 9:
                    message.float_value = reader.float();
                    break;
                case 10:
                    message.double_value = reader.double();
                    break;
                case 11:
                    message.string_value = reader.string();
                    break;
                case 12:
                    message.bytes_value = serial_v2.Bytes.decode(reader, reader.uint32());
                    break;
                case 13:
                    message.bool_vec_value = serial_v2.BoolVec.decode(reader, reader.uint32());
                    break;
                case 14:
                    message.int32_vec_value = serial_v2.Int32Vec.decode(reader, reader.uint32());
                    break;
                case 15:
                    message.uint32_vec_value = serial_v2.Uint32Vec.decode(reader, reader.uint32());
                    break;
                case 16:
                    message.int64_vec_value = serial_v2.Int64Vec.decode(reader, reader.uint32());
                    break;
                case 17:
                    message.uint64_vec_value = serial_v2.Uint64Vec.decode(reader, reader.uint32());
                    break;
                case 18:
                    message.float_vec_value = serial_v2.FloatVec.decode(reader, reader.uint32());
                    break;
                case 19:
                    message.double_vec_value = serial_v2.DoubleVec.decode(reader, reader.uint32());
                    break;
                case 20:
                    message.string_vec_value = serial_v2.StringVec.decode(reader, reader.uint32());
                    break;
                case 21:
                    message.bytes_vec_value = serial_v2.BytesVec.decode(reader, reader.uint32());
                    break;
                case 22:
                    message.map_string_2_int32_value = serial_v2.MapString2Int32.decode(reader, reader.uint32());
                    break;
                case 23:
                    message.map_string_2_uint32_value = serial_v2.MapString2Uint32.decode(reader, reader.uint32());
                    break;
                case 24:
                    message.map_string_2_int64_value = serial_v2.MapString2Int64.decode(reader, reader.uint32());
                    break;
                case 25:
                    message.map_string_2_uint64_value = serial_v2.MapString2Uint64.decode(reader, reader.uint32());
                    break;
                case 26:
                    message.map_string_2_string_value = serial_v2.MapString2String.decode(reader, reader.uint32());
                    break;
                case 27:
                    message.map_string_2_bytes_value = serial_v2.MapString2Bytes.decode(reader, reader.uint32());
                    break;
                case 28:
                    message.map_string_2_int32_vec_value = serial_v2.MapString2Int32Vec.decode(reader, reader.uint32());
                    break;
                case 29:
                    message.map_string_2_uint32_vec_value = serial_v2.MapString2Uint32Vec.decode(reader, reader.uint32());
                    break;
                case 30:
                    message.map_string_2_int64_vec_value = serial_v2.MapString2Int64Vec.decode(reader, reader.uint32());
                    break;
                case 31:
                    message.map_string_2_uint64_vec_value = serial_v2.MapString2Uint64Vec.decode(reader, reader.uint32());
                    break;
                case 32:
                    message.map_string_2_string_vec_value = serial_v2.MapString2StringVec.decode(reader, reader.uint32());
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }

    static decodeText(reader) {
        const message = new serial_v2.AttrValue();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "bool_value":
                    message.bool_value = reader.bool();
                    break;
                case "int32_value":
                    message.int32_value = reader.int32();
                    break;
                case "uint32_value":
                    message.uint32_value = reader.uint32();
                    break;
                case "int64_value":
                    message.int64_value = reader.int64();
                    break;
                case "uint64_value":
                    message.uint64_value = reader.uint64();
                    break;
                case "float_value":
                    message.float_value = reader.float();
                    break;
                case "double_value":
                    message.double_value = reader.double();
                    break;
                case "string_value":
                    message.string_value = reader.string();
                    break;
                case "bytes_value":
                    message.bytes_value = serial_v2.Bytes.decodeText(reader);
                    break;
                case "bool_vec_value":
                    message.bool_vec_value = serial_v2.BoolVec.decodeText(reader);
                    break;
                case "int32_vec_value":
                    message.int32_vec_value = serial_v2.Int32Vec.decodeText(reader);
                    break;
                case "uint32_vec_value":
                    message.uint32_vec_value = serial_v2.Uint32Vec.decodeText(reader);
                    break;
                case "int64_vec_value":
                    message.int64_vec_value = serial_v2.Int64Vec.decodeText(reader);
                    break;
                case "uint64_vec_value":
                    message.uint64_vec_value = serial_v2.Uint64Vec.decodeText(reader);
                    break;
                case "float_vec_value":
                    message.float_vec_value = serial_v2.FloatVec.decodeText(reader);
                    break;
                case "double_vec_value":
                    message.double_vec_value = serial_v2.DoubleVec.decodeText(reader);
                    break;
                case "string_vec_value":
                    message.string_vec_value = serial_v2.StringVec.decodeText(reader);
                    break;
                case "bytes_vec_value":
                    message.bytes_vec_value = serial_v2.BytesVec.decodeText(reader);
                    break;
                case "map_string_2_int32_value":
                    message.map_string_2_int32_value = serial_v2.MapString2Int32.decodeText(reader);
                    break;
                case "map_string_2_uint32_value":
                    message.map_string_2_uint32_value = serial_v2.MapString2Uint32.decodeText(reader);
                    break;
                case "map_string_2_int64_value":
                    message.map_string_2_int64_value = serial_v2.MapString2Int64.decodeText(reader);
                    break;
                case "map_string_2_uint64_value":
                    message.map_string_2_uint64_value = serial_v2.MapString2Uint64.decodeText(reader);
                    break;
                case "map_string_2_string_value":
                    message.map_string_2_string_value = serial_v2.MapString2String.decodeText(reader);
                    break;
                case "map_string_2_bytes_value":
                    message.map_string_2_bytes_value = serial_v2.MapString2Bytes.decodeText(reader);
                    break;
                case "map_string_2_int32_vec_value":
                    message.map_string_2_int32_vec_value = serial_v2.MapString2Int32Vec.decodeText(reader);
                    break;
                case "map_string_2_uint32_vec_value":
                    message.map_string_2_uint32_vec_value = serial_v2.MapString2Uint32Vec.decodeText(reader);
                    break;
                case "map_string_2_int64_vec_value":
                    message.map_string_2_int64_vec_value = serial_v2.MapString2Int64Vec.decodeText(reader);
                    break;
                case "map_string_2_uint64_vec_value":
                    message.map_string_2_uint64_vec_value = serial_v2.MapString2Uint64Vec.decodeText(reader);
                    break;
                case "map_string_2_string_vec_value":
                    message.map_string_2_string_vec_value = serial_v2.MapString2StringVec.decodeText(reader);
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

serial_v2.Bytes = class Bytes {

    static decode(reader, length) {
        const message = new serial_v2.Bytes();
        const end = length === undefined ? reader.length : reader.position + length;
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

    static decodeText(reader) {
        const message = new serial_v2.Bytes();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "value":
                    message.value = reader.bytes();
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

serial_v2.Bytes.prototype.value = new Uint8Array([]);

serial_v2.BoolVec = class BoolVec {

    constructor() {
        this.value = [];
    }

    static decode(reader, length) {
        const message = new serial_v2.BoolVec();
        const end = length === undefined ? reader.length : reader.position + length;
        while (reader.position < end) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.value = reader.array(message.value, () => reader.bool(), tag);
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }

    static decodeText(reader) {
        const message = new serial_v2.BoolVec();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "value":
                    reader.array(message.value, () => reader.bool());
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

serial_v2.Int32Vec = class Int32Vec {

    constructor() {
        this.value = [];
    }

    static decode(reader, length) {
        const message = new serial_v2.Int32Vec();
        const end = length === undefined ? reader.length : reader.position + length;
        while (reader.position < end) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.value = reader.array(message.value, () => reader.int32(), tag);
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }

    static decodeText(reader) {
        const message = new serial_v2.Int32Vec();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "value":
                    reader.array(message.value, () => reader.int32());
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

serial_v2.Uint32Vec = class Uint32Vec {

    constructor() {
        this.value = [];
    }

    static decode(reader, length) {
        const message = new serial_v2.Uint32Vec();
        const end = length === undefined ? reader.length : reader.position + length;
        while (reader.position < end) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.value = reader.array(message.value, () => reader.uint32(), tag);
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }

    static decodeText(reader) {
        const message = new serial_v2.Uint32Vec();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "value":
                    reader.array(message.value, () => reader.uint32());
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

serial_v2.Int64Vec = class Int64Vec {

    constructor() {
        this.value = [];
    }

    static decode(reader, length) {
        const message = new serial_v2.Int64Vec();
        const end = length === undefined ? reader.length : reader.position + length;
        while (reader.position < end) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.value = reader.array(message.value, () => reader.int64(), tag);
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }

    static decodeText(reader) {
        const message = new serial_v2.Int64Vec();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "value":
                    reader.array(message.value, () => reader.int64());
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

serial_v2.Uint64Vec = class Uint64Vec {

    constructor() {
        this.value = [];
    }

    static decode(reader, length) {
        const message = new serial_v2.Uint64Vec();
        const end = length === undefined ? reader.length : reader.position + length;
        while (reader.position < end) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.value = reader.array(message.value, () => reader.uint64(), tag);
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }

    static decodeText(reader) {
        const message = new serial_v2.Uint64Vec();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "value":
                    reader.array(message.value, () => reader.uint64());
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

serial_v2.FloatVec = class FloatVec {

    constructor() {
        this.value = [];
    }

    static decode(reader, length) {
        const message = new serial_v2.FloatVec();
        const end = length === undefined ? reader.length : reader.position + length;
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

    static decodeText(reader) {
        const message = new serial_v2.FloatVec();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "value":
                    reader.array(message.value, () => reader.float());
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

serial_v2.DoubleVec = class DoubleVec {

    constructor() {
        this.value = [];
    }

    static decode(reader, length) {
        const message = new serial_v2.DoubleVec();
        const end = length === undefined ? reader.length : reader.position + length;
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

    static decodeText(reader) {
        const message = new serial_v2.DoubleVec();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "value":
                    reader.array(message.value, () => reader.double());
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

serial_v2.StringVec = class StringVec {

    constructor() {
        this.value = [];
    }

    static decode(reader, length) {
        const message = new serial_v2.StringVec();
        const end = length === undefined ? reader.length : reader.position + length;
        while (reader.position < end) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.value.push(reader.string());
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }

    static decodeText(reader) {
        const message = new serial_v2.StringVec();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "value":
                    reader.array(message.value, () => reader.string());
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

serial_v2.BytesVec = class BytesVec {

    constructor() {
        this.value = [];
    }

    static decode(reader, length) {
        const message = new serial_v2.BytesVec();
        const end = length === undefined ? reader.length : reader.position + length;
        while (reader.position < end) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.value.push(serial_v2.Bytes.decode(reader, reader.uint32()));
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }

    static decodeText(reader) {
        const message = new serial_v2.BytesVec();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "value":
                    message.value.push(serial_v2.Bytes.decodeText(reader));
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

serial_v2.MapString2Int32 = class MapString2Int32 {

    constructor() {
        this.value = {};
    }

    static decode(reader, length) {
        const message = new serial_v2.MapString2Int32();
        const end = length === undefined ? reader.length : reader.position + length;
        while (reader.position < end) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    reader.entry(message.value, () => reader.string(), () => reader.int32());
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }

    static decodeText(reader) {
        const message = new serial_v2.MapString2Int32();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "value":
                    reader.entry(message.value, () => reader.string(), () => reader.int32());
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

serial_v2.MapString2Uint32 = class MapString2Uint32 {

    constructor() {
        this.value = {};
    }

    static decode(reader, length) {
        const message = new serial_v2.MapString2Uint32();
        const end = length === undefined ? reader.length : reader.position + length;
        while (reader.position < end) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    reader.entry(message.value, () => reader.string(), () => reader.uint32());
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }

    static decodeText(reader) {
        const message = new serial_v2.MapString2Uint32();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "value":
                    reader.entry(message.value, () => reader.string(), () => reader.uint32());
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

serial_v2.MapString2Int64 = class MapString2Int64 {

    constructor() {
        this.value = {};
    }

    static decode(reader, length) {
        const message = new serial_v2.MapString2Int64();
        const end = length === undefined ? reader.length : reader.position + length;
        while (reader.position < end) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    reader.entry(message.value, () => reader.string(), () => reader.int64());
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }

    static decodeText(reader) {
        const message = new serial_v2.MapString2Int64();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "value":
                    reader.entry(message.value, () => reader.string(), () => reader.int64());
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

serial_v2.MapString2Uint64 = class MapString2Uint64 {

    constructor() {
        this.value = {};
    }

    static decode(reader, length) {
        const message = new serial_v2.MapString2Uint64();
        const end = length === undefined ? reader.length : reader.position + length;
        while (reader.position < end) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    reader.entry(message.value, () => reader.string(), () => reader.uint64());
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }

    static decodeText(reader) {
        const message = new serial_v2.MapString2Uint64();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "value":
                    reader.entry(message.value, () => reader.string(), () => reader.uint64());
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

serial_v2.MapString2Bytes = class MapString2Bytes {

    constructor() {
        this.value = {};
    }

    static decode(reader, length) {
        const message = new serial_v2.MapString2Bytes();
        const end = length === undefined ? reader.length : reader.position + length;
        while (reader.position < end) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    reader.entry(message.value, () => reader.string(), () => serial_v2.Bytes.decode(reader, reader.uint32()));
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }

    static decodeText(reader) {
        const message = new serial_v2.MapString2Bytes();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "value":
                    reader.entry(message.value, () => reader.string(), () => serial_v2.Bytes.decodeText(reader));
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

serial_v2.MapString2String = class MapString2String {

    constructor() {
        this.value = {};
    }

    static decode(reader, length) {
        const message = new serial_v2.MapString2String();
        const end = length === undefined ? reader.length : reader.position + length;
        while (reader.position < end) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    reader.entry(message.value, () => reader.string(), () => reader.string());
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }

    static decodeText(reader) {
        const message = new serial_v2.MapString2String();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "value":
                    reader.entry(message.value, () => reader.string(), () => reader.string());
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

serial_v2.MapString2Int32Vec = class MapString2Int32Vec {

    constructor() {
        this.value = {};
    }

    static decode(reader, length) {
        const message = new serial_v2.MapString2Int32Vec();
        const end = length === undefined ? reader.length : reader.position + length;
        while (reader.position < end) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    reader.entry(message.value, () => reader.string(), () => serial_v2.Int32Vec.decode(reader, reader.uint32()));
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }

    static decodeText(reader) {
        const message = new serial_v2.MapString2Int32Vec();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "value":
                    reader.entry(message.value, () => reader.string(), () => serial_v2.Int32Vec.decodeText(reader));
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

serial_v2.MapString2Uint32Vec = class MapString2Uint32Vec {

    constructor() {
        this.value = {};
    }

    static decode(reader, length) {
        const message = new serial_v2.MapString2Uint32Vec();
        const end = length === undefined ? reader.length : reader.position + length;
        while (reader.position < end) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    reader.entry(message.value, () => reader.string(), () => serial_v2.Uint32Vec.decode(reader, reader.uint32()));
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }

    static decodeText(reader) {
        const message = new serial_v2.MapString2Uint32Vec();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "value":
                    reader.entry(message.value, () => reader.string(), () => serial_v2.Uint32Vec.decodeText(reader));
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

serial_v2.MapString2Int64Vec = class MapString2Int64Vec {

    constructor() {
        this.value = {};
    }

    static decode(reader, length) {
        const message = new serial_v2.MapString2Int64Vec();
        const end = length === undefined ? reader.length : reader.position + length;
        while (reader.position < end) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    reader.entry(message.value, () => reader.string(), () => serial_v2.Int64Vec.decode(reader, reader.uint32()));
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }

    static decodeText(reader) {
        const message = new serial_v2.MapString2Int64Vec();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "value":
                    reader.entry(message.value, () => reader.string(), () => serial_v2.Int64Vec.decodeText(reader));
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

serial_v2.MapString2Uint64Vec = class MapString2Uint64Vec {

    constructor() {
        this.value = {};
    }

    static decode(reader, length) {
        const message = new serial_v2.MapString2Uint64Vec();
        const end = length === undefined ? reader.length : reader.position + length;
        while (reader.position < end) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    reader.entry(message.value, () => reader.string(), () => serial_v2.Uint64Vec.decode(reader, reader.uint32()));
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }

    static decodeText(reader) {
        const message = new serial_v2.MapString2Uint64Vec();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "value":
                    reader.entry(message.value, () => reader.string(), () => serial_v2.Uint64Vec.decodeText(reader));
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

serial_v2.MapString2BytesVec = class MapString2BytesVec {

    constructor() {
        this.value = {};
    }

    static decode(reader, length) {
        const message = new serial_v2.MapString2BytesVec();
        const end = length === undefined ? reader.length : reader.position + length;
        while (reader.position < end) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    reader.entry(message.value, () => reader.string(), () => serial_v2.BytesVec.decode(reader, reader.uint32()));
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }

    static decodeText(reader) {
        const message = new serial_v2.MapString2BytesVec();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "value":
                    reader.entry(message.value, () => reader.string(), () => serial_v2.BytesVec.decodeText(reader));
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

serial_v2.MapString2StringVec = class MapString2StringVec {

    constructor() {
        this.value = {};
    }

    static decode(reader, length) {
        const message = new serial_v2.MapString2StringVec();
        const end = length === undefined ? reader.length : reader.position + length;
        while (reader.position < end) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    reader.entry(message.value, () => reader.string(), () => serial_v2.StringVec.decode(reader, reader.uint32()));
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }

    static decodeText(reader) {
        const message = new serial_v2.MapString2StringVec();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "value":
                    reader.entry(message.value, () => reader.string(), () => serial_v2.StringVec.decodeText(reader));
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};
