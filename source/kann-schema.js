
export const kann = {};

kann.Model = class Model {

    static identifier(reader) {
        return reader.identifier === 'KaNN';
    }

    static create(reader) {
        return kann.Model.decode(reader, reader.root);
    }

    static decode(reader, position) {
        const $ = new kann.Model();
        $.graph = reader.tables(position, 4, kann.Graph);
        return $;
    }
};

kann.Graph = class Graph {

    static decode(reader, position) {
        const $ = new kann.Graph();
        $.arcs = reader.tables(position, 4, kann.Arc);
        $.nodes = reader.tables(position, 6, kann.Node);
        $.inputs = reader.strings_(position, 8);
        $.outputs = reader.strings_(position, 10);
        return $;
    }
};

kann.Arc = class Arc {

    static decode(reader, position) {
        const $ = new kann.Arc();
        $.name = reader.string_(position, 4, null);
        $.type = reader.string_(position, 6, null);
        $.attributes = reader.tables(position, 8, kann.Attribute);
        return $;
    }
};

kann.Node = class Node {

    static decode(reader, position) {
        const $ = new kann.Node();
        $.name = reader.string_(position, 4, null);
        $.type = reader.string_(position, 6, null);
        $.inputs = reader.strings_(position, 8);
        $.outputs = reader.strings_(position, 10);
        $.attributes = reader.tables(position, 12, kann.Attribute);
        $.tensor = reader.table(position, 14, kann.Param);
        $.relu = reader.bool_(position, 16, false);
        $.params = reader.tables(position, 18, kann.Param);
        return $;
    }
};

kann.Param = class Param {

    static decode(reader, position) {
        const $ = new kann.Param();
        $.name = reader.string_(position, 4, null);
        $.type = reader.string_(position, 6, null);
        $.shape = reader.array(position, 8, Int32Array);
        $.value = reader.table(position, 10, kann.Data);
        $.scale = reader.table(position, 12, kann.Data);
        $.zero_point = reader.table(position, 14, kann.Data);
        return $;
    }
};

kann.Attribute = class Attribute {

    static decode(reader, position) {
        const $ = new kann.Attribute();
        $.name = reader.string_(position, 4, null);
        $.type = reader.string_(position, 6, null);
        $.value = reader.table(position, 8, kann.Data);
        $.attributes = reader.tables(position, 10, kann.Attribute);
        return $;
    }
};

kann.Data = class Data {

    static decode(reader, position) {
        const $ = new kann.Data();
        $.type = reader.string_(position, 4, null);
        $.value_string = reader.string_(position, 6, null);
        $.value_float = reader.float64_(position, 8, 0);
        $.value_int = reader.int64_(position, 10, 0n);
        $.value_uint = reader.uint64_(position, 12, 0n);
        $.list_string = reader.strings_(position, 14);
        $.list_float = reader.array(position, 16, Float64Array);
        $.list_int = reader.int64s_(position, 18);
        $.list_uint = reader.uint64s_(position, 20);
        return $;
    }
};
