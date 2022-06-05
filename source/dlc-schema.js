var $root = flatbuffers.get('dlc');

$root.dlc = $root.dlc || {};

$root.dlc.NetDef = class NetDef {

    static decode(reader, position) {
        const $ = new $root.dlc.NetDef();
        $.unk1 = reader.int32_(position, 4, 0);
        $.nodes = reader.tableArray(position, 6, $root.dlc.Node.decode);
        $.unk2 = reader.typedArray(position, 8, Int32Array);
        $.unk3 = reader.typedArray(position, 10, Int32Array);
        $.attributes = reader.tableArray(position, 12, $root.dlc.Attribute.decode);
        return $;
    }
};

$root.dlc.NetParam = class NetParam {

    static decode(reader, position) {
        const $ = new $root.dlc.NetParam();
        $.weights = reader.tableArray(position, 4, $root.dlc.Weights.decode);
        return $;
    }
};

$root.dlc.Node = class Node {

    static decode(reader, position) {
        const $ = new $root.dlc.Node();
        $.index = reader.int32_(position, 4, 0);
        $.name = reader.string_(position, 6, null);
        $.type = reader.string_(position, 8, null);
        $.inputs = reader.strings_(position, 10);
        $.outputs = reader.strings_(position, 12);
        $.attributes = reader.tableArray(position, 14, $root.dlc.Attribute.decode);
        return $;
    }
};

$root.dlc.Weights = class Weights {

    static decode(reader, position) {
        const $ = new $root.dlc.Weights();
        $.name = reader.string_(position, 4, null);
        $.tensors = reader.tableArray(position, 6, $root.dlc.Tensor.decode);
        return $;
    }
};

$root.dlc.Tensor = class Tensor {

    static decode(reader, position) {
        const $ = new $root.dlc.Tensor();
        $.name = reader.string_(position, 4, null);
        $.shape = reader.typedArray(position, 6, Int32Array);
        $.data = reader.table(position, 8, $root.dlc.TensorData.decode);
        $.attributes = reader.tableArray(position, 10, $root.dlc.Attribute.decode);
        return $;
    }
};

$root.dlc.TensorData = class TensorData {

    static decode(reader, position) {
        const $ = new $root.dlc.TensorData();
        $.data_type = reader.uint8_(position, 4, 0);
        $.bytes = reader.typedArray(position, 6, Uint8Array);
        $.floats = reader.typedArray(position, 8, Float32Array);
        return $;
    }
};

$root.dlc.Attribute = class Attribute {

    static decode(reader, position) {
        const $ = new $root.dlc.Attribute();
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
        $.attributes = reader.tableArray(position, 28, $root.dlc.Attribute.decode);
        return $;
    }
};

$root.dlc.Activation = {
    ReLU: 1,
    Sigmoid: 3
};
