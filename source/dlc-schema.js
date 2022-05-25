var $root = flatbuffers.get('dlc');

$root.dlc = $root.dlc || {};

$root.dlc.NetDefinition = class NetDefinition {

    static decode(/* reader, position */) {
        const $ = new $root.dlc.NetDefinition();
        return $;
    }
};

$root.dlc.NetParameter = class NetParameter {

    static decode(reader, position) {
        const $ = new $root.dlc.NetParameter();
        $.params = reader.tableArray(position, 4, $root.dlc.Parameter.decode);
        return $;
    }
};

$root.dlc.Parameter = class Parameter {

    static decode(reader, position) {
        const $ = new $root.dlc.Parameter();
        $.name = reader.string_(position, 4, null);
        return $;
    }
};
