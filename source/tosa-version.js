// SPDX-FileCopyrightText: Copyright 2025 Arm Limited and/or its affiliates <open-source-office@arm.com>

export const tosa = {};

tosa.Version = class Version {

    static decode(reader, position) {
        const $ = new tosa.Version();
        $._major = reader.int32_(position, 4, -1);
        $._minor = reader.int32_(position, 6, -1);
        $._patch = reader.int32_(position, 8, -1);
        $._draft = reader.bool_(position, 10, true);
        return $;
    }

    static decodeText(reader, json) {
        const $ = new tosa.Version();
        $._major = reader.value(json._major, -1);
        $._minor = reader.value(json._minor, -1);
        $._patch = reader.value(json._patch, -1);
        $._draft = reader.value(json._draft, true);
        return $;
    }
};

tosa.TosaGraph = class TosaGraph {

    static identifier(reader) {
        return reader.identifier === 'TOSA';
    }

    static create(reader) {
        return tosa.TosaGraph.decode(reader, reader.root);
    }

    static createText(reader) {
        return tosa.TosaGraph.decodeText(reader, reader.root);
    }

    static decode(reader, position) {
        const $ = new tosa.TosaGraph();
        $.version = reader.table(position, 4, tosa.Version);
        return $;
    }

    static decodeText(reader, json) {
        const $ = new tosa.TosaGraph();
        $.version = reader.object(json.version, tosa.Version);
        return $;
    }
};
