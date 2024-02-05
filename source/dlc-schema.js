
export const dlc = {};

dlc.v3 = dlc.v3 || {};

dlc.v3.Model = class Model {

    static decode(reader, position) {
        const $ = new dlc.v3.Model();
        $.unk1 = reader.int32_(position, 4, 0);
        $.nodes = reader.tableArray(position, 6, dlc.v3.Node.decode);
        $.unk2 = reader.typedArray(position, 8, Int32Array);
        $.unk3 = reader.typedArray(position, 10, Int32Array);
        $.attributes = reader.tableArray(position, 12, dlc.v3.Attribute.decode);
        return $;
    }
};

dlc.v3.Node = class Node {

    static decode(reader, position) {
        const $ = new dlc.v3.Node();
        $.index = reader.int32_(position, 4, 0);
        $.name = reader.string_(position, 6, null);
        $.type = reader.string_(position, 8, null);
        $.inputs = reader.strings_(position, 10);
        $.outputs = reader.strings_(position, 12);
        $.attributes = reader.tableArray(position, 14, dlc.v3.Attribute.decode);
        return $;
    }
};

dlc.v3.Tensor = class Tensor {

    static decode(reader, position) {
        const $ = new dlc.v3.Tensor();
        $.name = reader.string_(position, 4, null);
        $.shape = reader.typedArray(position, 6, Int32Array);
        $.data = reader.table(position, 8, dlc.v3.TensorData.decode);
        $.attributes = reader.tableArray(position, 10, dlc.v3.Attribute.decode);
        return $;
    }
};

dlc.v3.TensorData = class TensorData {

    static decode(reader, position) {
        const $ = new dlc.v3.TensorData();
        $.dtype = reader.uint8_(position, 4, 0);
        $.bytes = reader.typedArray(position, 6, Uint8Array);
        $.floats = reader.typedArray(position, 8, Float32Array);
        return $;
    }
};

dlc.v3.Attribute = class Attribute {

    static decode(reader, position) {
        const $ = new dlc.v3.Attribute();
        $.name = reader.string_(position, 4, null);
        $.type = reader.uint8_(position, 6, 0);
        $.bool_value = reader.bool_(position, 8, false);
        $.int32_value = reader.int32_(position, 10, 0);
        $.uint32_value = reader.uint32_(position, 12, 0);
        $.float32_value = reader.float32_(position, 14, 0);
        $.string_value = reader.string_(position, 16, null);
        $.unk6 = reader.typedArray(position, 18, Int8Array);
        $.byte_list = reader.typedArray(position, 20, Int8Array);
        $.int32_list = reader.typedArray(position, 22, Int32Array);
        $.float32_list = reader.typedArray(position, 24, Float32Array);
        $.unk10 = reader.typedArray(position, 26, Int8Array);
        $.attributes = reader.tableArray(position, 28, dlc.v3.Attribute.decode);
        return $;
    }
};

dlc.v3.Activation = {
    ReLU: 1,
    Sigmoid: 3
};

dlc.v3.ModelParameters = class ModelParameters {

    static decode(reader, position) {
        const $ = new dlc.v3.ModelParameters();
        $.nodes = reader.tableArray(position, 4, dlc.v3.NodeParameters.decode);
        return $;
    }
};

dlc.v3.NodeParameters = class NodeParameters {

    static decode(reader, position) {
        const $ = new dlc.v3.NodeParameters();
        $.name = reader.string_(position, 4, null);
        $.weights = reader.tableArray(position, 6, dlc.v3.Tensor.decode);
        return $;
    }
};

dlc.v4 = dlc.v4 || {};

dlc.v4.Model = class Model {

    static decode(reader, position) {
        const $ = new dlc.v4.Model();
        $.graphs = reader.tableArray(position, 4, dlc.v4.Graph.decode);
        return $;
    }
};

dlc.v4.Graph = class Graph {

    static decode(reader, position) {
        const $ = new dlc.v4.Graph();
        $.name = reader.string_(position, 4, null);
        $.nodes = reader.tableArray(position, 6, dlc.v4.Node.decode);
        $.tensors = reader.tableArray(position, 8, dlc.v4.Tensor.decode);
        return $;
    }
};

dlc.v4.Node = class Node {

    static decode(reader, position) {
        const $ = new dlc.v4.Node();
        $.name = reader.string_(position, 4, null);
        $.type = reader.string_(position, 6, null);
        $.inputs = reader.strings_(position, 8);
        $.outputs = reader.strings_(position, 10);
        $.attributes = reader.tableArray(position, 12, dlc.v4.Attribute.decode);
        return $;
    }
};

dlc.v4.Attribute = class Attribute {

    static decode(reader, position) {
        const $ = new dlc.v4.Attribute();
        $.name = reader.string_(position, 4, null);
        $.kind = reader.int32_(position, 6, 0);
        $.flag = reader.uint8_(position, 8, 0);
        $.value = reader.table(position, 10, dlc.v4.Value.decode);
        $.tensor = reader.table(position, 12, dlc.v4.Tensor.decode);
        return $;
    }
};

dlc.v4.Value = class Value {

    static decode(reader, position) {
        const $ = new dlc.v4.Value();
        $.kind = reader.int32_(position, 4, 0);
        $.int32_value = reader.int32_(position, 6, 0);
        $.float32_value = reader.float32_(position, 8, 0);
        $.string_value = reader.string_(position, 10, null);
        return $;
    }
};

dlc.v4.Tensor = class Tensor {

    static decode(reader, position) {
        const $ = new dlc.v4.Tensor();
        $.unk1 = reader.uint32_(position, 4, 0);
        $.name = reader.string_(position, 6, null);
        $.location = reader.int32_(position, 8, 0);
        $.shape = reader.typedArray(position, 10, Int32Array);
        $.unk2 = reader.int32_(position, 12, 0);
        $.info = reader.table(position, 14, dlc.v4.TensorInfo.decode);
        $.dtype = reader.int32_(position, 16, 0);
        $.output_dtype = reader.int32_(position, 18, 0);
        $.unk6 = reader.uint8_(position, 20, 0);
        return $;
    }
};

dlc.v4.TensorInfo = class TensorInfo {

    static decode(reader, position) {
        const $ = new dlc.v4.TensorInfo();
        $.i1 = reader.int32_(position, 4, 0);
        $.b1 = reader.uint8_(position, 6, 0);
        $.a = reader.table(position, 8, dlc.v4.TensorInfo1.decode);
        $.b = reader.table(position, 10, dlc.v4.TensorInfo2.decode);
        return $;
    }
};

dlc.v4.TensorInfo1 = class TensorInfo1 {

    static decode(reader, position) {
        const $ = new dlc.v4.TensorInfo1();
        $.i1 = reader.int32_(position, 4, 0);
        $.f1 = reader.float32_(position, 6, 0);
        $.f2 = reader.float32_(position, 8, 0);
        $.f3 = reader.float32_(position, 10, 0);
        $.i2 = reader.int32_(position, 12, 0);
        return $;
    }
};

dlc.v4.TensorInfo2 = class TensorInfo2 {

    static decode(reader, position) {
        const $ = new dlc.v4.TensorInfo2();
        $.i1 = reader.int32_(position, 4, 0);
        $.l = reader.tableArray(position, 6, dlc.v4.TensorInfo3.decode);
        return $;
    }
};

dlc.v4.TensorInfo3 = class TensorInfo3 {

    static decode(reader, position) {
        const $ = new dlc.v4.TensorInfo3();
        $.i1 = reader.int32_(position, 4, 0);
        $.f1 = reader.float32_(position, 6, 0);
        $.f2 = reader.float32_(position, 8, 0);
        $.f3 = reader.float32_(position, 10, 0);
        $.i2 = reader.int32_(position, 12, 0);
        $.b1 = reader.uint8_(position, 14, 0);
        return $;
    }
};

dlc.v4.ModelParameters64 = class ModelParameters64 {

    static decode(reader, position) {
        const $ = new dlc.v4.ModelParameters64();
        $.buffers = reader.tableArray(position, 4, dlc.v4.Buffer.decode);
        $.params = reader.typedArray(position, 6, Uint8Array);
        return $;
    }
};

dlc.v4.ModelParameters = class ModelParameters {

    static decode(reader, position) {
        const $ = new dlc.v4.ModelParameters();
        $.graphs = reader.tableArray(position, 4, dlc.v4.GraphParameters.decode);
        return $;
    }
};

dlc.v4.GraphParameters = class GraphParameters {

    static decode(reader, position) {
        const $ = new dlc.v4.GraphParameters();
        $.name = reader.string_(position, 4, null);
        $.tensors = reader.tableArray(position, 6, dlc.v4.TensorData.decode);
        $.nodes = reader.tableArray(position, 8, dlc.v4.NodeParameters.decode);
        return $;
    }
};

dlc.v4.NodeParameters = class NodeParameters {

    static decode(reader, position) {
        const $ = new dlc.v4.NodeParameters();
        $.tensors = reader.tableArray(position, 4, dlc.v4.TensorData.decode);
        return $;
    }
};

dlc.v4.TensorData = class TensorData {

    static decode(reader, position) {
        const $ = new dlc.v4.TensorData();
        $.name = reader.string_(position, 4, null);
        $.bytes = reader.typedArray(position, 6, Uint8Array);
        return $;
    }
};

dlc.v4.Buffer = class Buffer {

    static decode(reader, position) {
        const $ = new dlc.v4.Buffer();
        $.bytes = reader.typedArray(position, 4, Uint8Array);
        return $;
    }
};
