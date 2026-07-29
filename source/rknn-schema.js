
export const rknn = {};

export const rknx = {};

rknn.Model = class Model {

    static identifier(reader) {
        return reader.identifier === 'RKNN';
    }

    static create(reader) {
        return rknn.Model.decode(reader, reader.root);
    }

    static decode(reader, position) {
        const $ = new rknn.Model();
        $.var1 = reader.int32_(position, 4, 0);
        $.format = reader.string_(position, 6, null);
        $.graphs = reader.tables(position, 8, rknn.Graph);
        $.generator = reader.string_(position, 10, null);
        $.var2 = reader.tables(position, 12, rknn.Type1);
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

rknn.Graph = class Graph {

    static decode(reader, position) {
        const $ = new rknn.Graph();
        $.tensors = reader.tables(position, 4, rknn.Tensor);
        $.nodes = reader.tables(position, 6, rknn.Node);
        $.inputs = reader.array(position, 8, Int32Array);
        $.outputs = reader.array(position, 10, Int32Array);
        $.var1 = reader.tables(position, 12, rknn.Type2);
        return $;
    }
};

rknn.Node = class Node {

    static decode(reader, position) {
        const $ = new rknn.Node();
        $.var1 = reader.int32_(position, 4, 0);
        $.type = reader.string_(position, 6, null);
        $.name = reader.string_(position, 8, null);
        $.var2 = reader.int8_(position, 10, 0);
        $.inputs = reader.array(position, 12, Int32Array);
        $.outputs = reader.array(position, 14, Int32Array);
        $.var3 = reader.tables(position, 16, rknn.Type3);
        $.var4 = reader.int8_(position, 18, 0);
        $.var5 = reader.int32_(position, 20, 0);
        $.var6 = reader.int32_(position, 22, 0);
        return $;
    }
};

rknn.Tensor = class Tensor {

    static decode(reader, position) {
        const $ = new rknn.Tensor();
        $.data_type = reader.int8_(position, 4, 0);
        $.var02 = reader.int8_(position, 6, 0);
        $.kind = reader.int8_(position, 8, 0);
        $.var04 = reader.array(position, 10, Int32Array);
        $.shape = reader.array(position, 12, Int32Array);
        $.name = reader.string_(position, 14, null);
        $.var06 = reader.array(position, 16, Int8Array);
        $.var07 = reader.string_(position, 18, null);
        $.var08 = reader.array(position, 20, Int8Array);
        $.var09 = reader.array(position, 22, Int8Array);
        $.var10 = reader.array(position, 24, Int8Array);
        $.var11 = reader.array(position, 26, Int8Array);
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

rknn.Type1 = class Type1 {

    static decode(reader, position) {
        const $ = new rknn.Type1();
        $.var1 = reader.int32_(position, 4, 0);
        return $;
    }
};

rknn.Type2 = class Type2 {

    static decode(reader, position) {
        const $ = new rknn.Type2();
        $.var1 = reader.array(position, 4, Int32Array);
        $.var2 = reader.array(position, 6, Int32Array);
        $.var3 = reader.array(position, 8, Int32Array);
        return $;
    }
};

rknn.Type3 = class Type3 {

    static decode(reader, position) {
        const $ = new rknn.Type3();
        $.var1 = reader.int32_(position, 4, 0);
        return $;
    }
};

rknx.Header = class Header {

    static decode(reader, position) {
        const $ = new rknx.Header();
        $.version = reader.int32_(position, 4, 0);
        $.description = reader.string_(position, 6, null);
        $.compiler_version = reader.string_(position, 8, null);
        $.toolkit_version = reader.string_(position, 10, null);
        $.var4 = reader.int32_(position, 12, 0);
        $.hardware = reader.table(position, 14, rknx.Hardware);
        $.var6 = reader.int32_(position, 16, 0);
        $.var7 = reader.int32_(position, 18, 0);
        $.var8 = reader.int32_(position, 20, 0);
        $.io_configs = reader.tables(position, 22, rknx.IoConfig);
        $.llm_config = reader.table(position, 24, rknx.LlmConfig);
        $.var11 = reader.int32_(position, 26, 0);
        return $;
    }
};

rknx.Hardware = class Hardware {

    static decode(reader, position) {
        const $ = new rknx.Hardware();
        $.name = reader.string_(position, 4, null);
        $.var1 = reader.int32_(position, 6, 0);
        $.var2 = reader.int32_(position, 8, 0);
        return $;
    }
};

rknx.IoConfig = class IoConfig {

    static decode(reader, position) {
        const $ = new rknx.IoConfig();
        $.var0 = reader.int32_(position, 4, 0);
        $.input = reader.string_(position, 6, null);
        $.output = reader.string_(position, 8, null);
        return $;
    }
};

rknx.LlmConfig = class LlmConfig {

    static decode(reader, position) {
        const $ = new rknx.LlmConfig();
        $.var00 = reader.array(position, 4, Int32Array);
        $.vocab_size = reader.int32_(position, 6, 0);
        $.hidden_size = reader.int32_(position, 8, 0);
        $.var03 = reader.array(position, 10, Int32Array);
        $.var04 = reader.int32_(position, 12, 0);
        $.var05 = reader.int32_(position, 14, 0);
        $.var06 = reader.int32_(position, 16, 0);
        $.var07 = reader.array(position, 18, Int32Array);
        $.var08 = reader.int32_(position, 20, 0);
        $.var09 = reader.string_(position, 22, null);
        $.var10 = reader.string_(position, 24, null);
        $.config = reader.string_(position, 26, null);
        return $;
    }
};

rknx.Model = class Model {

    static create(reader) {
        return rknx.Model.decode(reader, reader.root);
    }

    static decode(reader, position) {
        const $ = new rknx.Model();
        $.var0 = reader.int32_(position, 4, 0);
        $.version = reader.int32_(position, 6, 0);
        $.subgraphs = reader.tables(position, 8, rknx.ComputeGraph);
        $.var3 = reader.int32_(position, 10, 0);
        $.var4 = reader.int32_(position, 12, 0);
        $.var5 = reader.int32_(position, 14, 0);
        $.var6 = reader.int32_(position, 16, 0);
        $.var7 = reader.int32_(position, 18, 0);
        return $;
    }
};

rknx.ComputeGraph = class ComputeGraph {

    static decode(reader, position) {
        const $ = new rknx.ComputeGraph();
        $.var0 = reader.int32_(position, 4, 0);
        $.name = reader.string_(position, 6, null);
        $.tensors = reader.tables(position, 8, rknx.Tensor);
        $.inputs = reader.array(position, 10, Int32Array);
        $.outputs = reader.array(position, 12, Int32Array);
        $.tasks = reader.tables(position, 14, rknx.Task);
        return $;
    }
};

rknx.Task = class Task {

    static decode(reader, position) {
        const $ = new rknx.Task();
        $.var0 = reader.int32_(position, 4, 0);
        $.var1 = reader.int32_(position, 6, 0);
        $.var2 = reader.int32_(position, 8, 0);
        $.var3 = reader.int32_(position, 10, 0);
        $.var4 = reader.int32_(position, 12, 0);
        return $;
    }
};

rknx.Tensor = class Tensor {

    static decode(reader, position) {
        const $ = new rknx.Tensor();
        $.var00 = reader.int32_(position, 4, 0);
        $.var01 = reader.int32_(position, 6, 0);
        $.var02 = reader.int32_(position, 8, 0);
        $.var03 = reader.int32_(position, 10, 0);
        $.shape = reader.array(position, 12, Int32Array);
        $.var05 = reader.array(position, 14, Int32Array);
        $.name = reader.string_(position, 16, null);
        $.var07 = reader.int32_(position, 18, 0);
        $.data_type = reader.string_(position, 20, null);
        $.var09 = reader.int32_(position, 22, 0);
        $.var10 = reader.int32_(position, 24, 0);
        $.var11 = reader.int32_(position, 26, 0);
        $.var12 = reader.int32_(position, 28, 0);
        $.size = reader.int32_(position, 30, 0);
        $.var14 = reader.int32_(position, 32, 0);
        $.stride = reader.array(position, 34, Int32Array);
        $.var16 = reader.int32_(position, 36, 0);
        $.var17 = reader.int32_(position, 38, 0);
        $.var18 = reader.int32_(position, 40, 0);
        $.var19 = reader.int32_(position, 42, 0);
        return $;
    }
};
