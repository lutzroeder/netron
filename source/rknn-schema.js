var $root = flatbuffers.get('rknn');

$root.rknn = $root.rknn || {};

$root.rknn.Model = class Model {

    static identifier(reader) {
        return reader.identifier === 'RKNN';
    }

    static create(reader) {
        return $root.rknn.Model.decode(reader, reader.root);
    }

    static decode(reader, position) {
        const $ = new $root.rknn.Model();
        $.var1 = reader.int32_(position, 4, 0);
        $.format = reader.string_(position, 6, null);
        $.graphs = reader.tableArray(position, 8, $root.rknn.Graph.decode);
        $.generator = reader.string_(position, 10, null);
        $.var2 = reader.tableArray(position, 12, $root.rknn.Type1.decode);
        $.var3 = reader.int32_(position, 14, 0);
        $.var4 = reader.int32_(position, 16, 0);
        $.compiler = reader.string_(position, 18, null);
        $.runtime = reader.string_(position, 20, null);
        $.source = reader.string_(position, 22, null);
        $.var5 = reader.bool_(position, 24, false);
        $.var6 = reader.int32_(position, 26, 0);
        $.input_json = reader.string_(position, 28, null);
        $.output_json = reader.string_(position, 30, null);
        return $;
    }
};

$root.rknn.Graph = class Graph {

    static decode(reader, position) {
        const $ = new $root.rknn.Graph();
        $.tensors = reader.tableArray(position, 4, $root.rknn.Tensor.decode);
        $.nodes = reader.tableArray(position, 6, $root.rknn.Node.decode);
        $.inputs = reader.typedArray(position, 8, Int32Array);
        $.outputs = reader.typedArray(position, 10, Int32Array);
        $.var1 = reader.tableArray(position, 12, $root.rknn.Type2.decode);
        return $;
    }
};

$root.rknn.Node = class Node {

    static decode(reader, position) {
        const $ = new $root.rknn.Node();
        $.var1 = reader.int32_(position, 4, 0);
        $.type = reader.string_(position, 6, null);
        $.name = reader.string_(position, 8, null);
        $.var2 = reader.int8_(position, 10, 0);
        $.inputs = reader.typedArray(position, 12, Int32Array);
        $.outputs = reader.typedArray(position, 14, Int32Array);
        $.var3 = reader.tableArray(position, 16, $root.rknn.Type3.decode);
        $.var4 = reader.int8_(position, 18, 0);
        $.var5 = reader.int32_(position, 20, 0);
        $.var6 = reader.int32_(position, 22, 0);
        return $;
    }
};

$root.rknn.Tensor = class Tensor {

    static decode(reader, position) {
        const $ = new $root.rknn.Tensor();
        $.data_type = reader.int8_(position, 4, 0);
        $.var02 = reader.int8_(position, 6, 0);
        $.kind = reader.int8_(position, 8, 0);
        $.var04 = reader.typedArray(position, 10, Int32Array);
        $.shape = reader.typedArray(position, 12, Int32Array);
        $.name = reader.string_(position, 14, null);
        $.var06 = reader.typedArray(position, 16, Int8Array);
        $.var07 = reader.string_(position, 18, null);
        $.var08 = reader.typedArray(position, 20, Int8Array);
        $.var09 = reader.typedArray(position, 22, Int8Array);
        $.var10 = reader.typedArray(position, 24, Int8Array);
        $.var11 = reader.typedArray(position, 26, Int8Array);
        $.size = reader.int32_(position, 28, 0);
        $.var13 = reader.int32_(position, 30, 0);
        $.var14 = reader.int32_(position, 32, 0);
        $.var15 = reader.int32_(position, 34, 0);
        $.var16 = reader.int32_(position, 36, 0);
        $.var17 = reader.int32_(position, 38, 0);
        $.index = reader.int32_(position, 40, 0);
        return $;
    }
};

$root.rknn.Type1 = class Type1 {

    static decode(reader, position) {
        const $ = new $root.rknn.Type1();
        $.var1 = reader.int32_(position, 4, 0);
        return $;
    }
};

$root.rknn.Type2 = class Type2 {

    static decode(reader, position) {
        const $ = new $root.rknn.Type2();
        $.var1 = reader.typedArray(position, 4, Int32Array);
        $.var2 = reader.typedArray(position, 6, Int32Array);
        $.var3 = reader.typedArray(position, 8, Int32Array);
        return $;
    }
};

$root.rknn.Type3 = class Type3 {

    static decode(reader, position) {
        const $ = new $root.rknn.Type3();
        $.var1 = reader.int32_(position, 4, 0);
        return $;
    }
};
