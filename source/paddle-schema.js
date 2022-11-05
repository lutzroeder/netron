var $root = flatbuffers.get('paddlelite');

$root.paddle = $root.paddle || {};

$root.paddle.lite = $root.paddle.lite || {};

$root.paddle.lite.fbs = $root.paddle.lite.fbs || {};

$root.paddle.lite.fbs.proto = $root.paddle.lite.fbs.proto || {};

$root.paddle.lite.fbs.proto.AttrType = {
    INT: 0,
    FLOAT: 1,
    STRING: 2,
    INTS: 3,
    FLOATS: 4,
    STRINGS: 5,
    BOOLEAN: 6,
    BOOLEANS: 7,
    BLOCK: 8,
    LONG: 9,
    BLOCKS: 10,
    LONGS: 11,
    FLOAT64S: 12,
    VAR: 13,
    VARS: 14,
    FLOAT64: 15
};

$root.paddle.lite.fbs.proto.Version = class Version {

    static decode(reader, position) {
        const $ = new $root.paddle.lite.fbs.proto.Version();
        $.version = reader.int64_(position, 4, 0);
        return $;
    }

    static decodeText(reader, json) {
        const $ = new $root.paddle.lite.fbs.proto.Version();
        $.version = reader.value(json.version, 0);
        return $;
    }
};

$root.paddle.lite.fbs.proto.OpDesc = class OpDesc {

    static decode(reader, position) {
        const $ = new $root.paddle.lite.fbs.proto.OpDesc();
        $.type = reader.string_(position, 4, null);
        $.inputs = reader.tableArray(position, 6, $root.paddle.lite.fbs.proto.OpDesc_.Var.decode);
        $.outputs = reader.tableArray(position, 8, $root.paddle.lite.fbs.proto.OpDesc_.Var.decode);
        $.attrs = reader.tableArray(position, 10, $root.paddle.lite.fbs.proto.OpDesc_.Attr.decode);
        $.is_target = reader.bool_(position, 12, false);
        return $;
    }

    static decodeText(reader, json) {
        const $ = new $root.paddle.lite.fbs.proto.OpDesc();
        $.type = reader.value(json.type, null);
        $.inputs = reader.objectArray(json.inputs, $root.paddle.lite.fbs.proto.OpDesc_.Var.decodeText);
        $.outputs = reader.objectArray(json.outputs, $root.paddle.lite.fbs.proto.OpDesc_.Var.decodeText);
        $.attrs = reader.objectArray(json.attrs, $root.paddle.lite.fbs.proto.OpDesc_.Attr.decodeText);
        $.is_target = reader.value(json.is_target, false);
        return $;
    }
};

$root.paddle.lite.fbs.proto.VarType = class VarType {

    static decode(reader, position) {
        const $ = new $root.paddle.lite.fbs.proto.VarType();
        $.type = reader.int32_(position, 4, 0);
        $.selected_rows = reader.table(position, 6, $root.paddle.lite.fbs.proto.VarType_.TensorDesc.decode);
        $.lod_tensor = reader.table(position, 8, $root.paddle.lite.fbs.proto.VarType_.LoDTensorDesc.decode);
        $.tensor_array = reader.table(position, 10, $root.paddle.lite.fbs.proto.VarType_.LoDTensorArrayDesc.decode);
        $.reader = reader.table(position, 12, $root.paddle.lite.fbs.proto.VarType_.ReaderDesc.decode);
        $.tuple = reader.table(position, 14, $root.paddle.lite.fbs.proto.VarType_.Tuple.decode);
        return $;
    }

    static decodeText(reader, json) {
        const $ = new $root.paddle.lite.fbs.proto.VarType();
        $.type = $root.paddle.lite.fbs.proto.VarType_.Type[json.type];
        $.selected_rows = reader.object(json.selected_rows, $root.paddle.lite.fbs.proto.VarType_.TensorDesc.decodeText);
        $.lod_tensor = reader.object(json.lod_tensor, $root.paddle.lite.fbs.proto.VarType_.LoDTensorDesc.decodeText);
        $.tensor_array = reader.object(json.tensor_array, $root.paddle.lite.fbs.proto.VarType_.LoDTensorArrayDesc.decodeText);
        $.reader = reader.object(json.reader, $root.paddle.lite.fbs.proto.VarType_.ReaderDesc.decodeText);
        $.tuple = reader.object(json.tuple, $root.paddle.lite.fbs.proto.VarType_.Tuple.decodeText);
        return $;
    }
};

$root.paddle.lite.fbs.proto.VarDesc = class VarDesc {

    static decode(reader, position) {
        const $ = new $root.paddle.lite.fbs.proto.VarDesc();
        $.name = reader.string_(position, 4, null);
        $.type = reader.table(position, 6, $root.paddle.lite.fbs.proto.VarType.decode);
        $.persistable = reader.bool_(position, 8, false);
        $.need_check_feed = reader.bool_(position, 10, false);
        return $;
    }

    static decodeText(reader, json) {
        const $ = new $root.paddle.lite.fbs.proto.VarDesc();
        $.name = reader.value(json.name, null);
        $.type = reader.object(json.type, $root.paddle.lite.fbs.proto.VarType.decodeText);
        $.persistable = reader.value(json.persistable, false);
        $.need_check_feed = reader.value(json.need_check_feed, false);
        return $;
    }
};

$root.paddle.lite.fbs.proto.BlockDesc = class BlockDesc {

    static decode(reader, position) {
        const $ = new $root.paddle.lite.fbs.proto.BlockDesc();
        $.idx = reader.int32_(position, 4, 0);
        $.parent_idx = reader.int32_(position, 6, 0);
        $.vars = reader.tableArray(position, 8, $root.paddle.lite.fbs.proto.VarDesc.decode);
        $.ops = reader.tableArray(position, 10, $root.paddle.lite.fbs.proto.OpDesc.decode);
        $.forward_block_idx = reader.int32_(position, 12, -1);
        return $;
    }

    static decodeText(reader, json) {
        const $ = new $root.paddle.lite.fbs.proto.BlockDesc();
        $.idx = reader.value(json.idx, 0);
        $.parent_idx = reader.value(json.parent_idx, 0);
        $.vars = reader.objectArray(json.vars, $root.paddle.lite.fbs.proto.VarDesc.decodeText);
        $.ops = reader.objectArray(json.ops, $root.paddle.lite.fbs.proto.OpDesc.decodeText);
        $.forward_block_idx = reader.value(json.forward_block_idx, -1);
        return $;
    }
};

$root.paddle.lite.fbs.proto.OpVersion = class OpVersion {

    static decode(reader, position) {
        const $ = new $root.paddle.lite.fbs.proto.OpVersion();
        $.version = reader.int32_(position, 4, 0);
        return $;
    }

    static decodeText(reader, json) {
        const $ = new $root.paddle.lite.fbs.proto.OpVersion();
        $.version = reader.value(json.version, 0);
        return $;
    }
};

$root.paddle.lite.fbs.proto.OpVersionMap = class OpVersionMap {

    static decode(reader, position) {
        const $ = new $root.paddle.lite.fbs.proto.OpVersionMap();
        $.pair = reader.tableArray(position, 4, $root.paddle.lite.fbs.proto.OpVersionMap_.OpVersionPair.decode);
        return $;
    }

    static decodeText(reader, json) {
        const $ = new $root.paddle.lite.fbs.proto.OpVersionMap();
        $.pair = reader.objectArray(json.pair, $root.paddle.lite.fbs.proto.OpVersionMap_.OpVersionPair.decodeText);
        return $;
    }
};

$root.paddle.lite.fbs.proto.ProgramDesc = class ProgramDesc {

    static create(reader) {
        return $root.paddle.lite.fbs.proto.ProgramDesc.decode(reader, reader.root);
    }

    static createText(reader) {
        return $root.paddle.lite.fbs.proto.ProgramDesc.decodeText(reader, reader.root);
    }

    static decode(reader, position) {
        const $ = new $root.paddle.lite.fbs.proto.ProgramDesc();
        $.blocks = reader.tableArray(position, 4, $root.paddle.lite.fbs.proto.BlockDesc.decode);
        $.version = reader.table(position, 6, $root.paddle.lite.fbs.proto.Version.decode);
        $.op_version_map = reader.table(position, 8, $root.paddle.lite.fbs.proto.OpVersionMap.decode);
        return $;
    }

    static decodeText(reader, json) {
        const $ = new $root.paddle.lite.fbs.proto.ProgramDesc();
        $.blocks = reader.objectArray(json.blocks, $root.paddle.lite.fbs.proto.BlockDesc.decodeText);
        $.version = reader.object(json.version, $root.paddle.lite.fbs.proto.Version.decodeText);
        $.op_version_map = reader.object(json.op_version_map, $root.paddle.lite.fbs.proto.OpVersionMap.decodeText);
        return $;
    }
};

$root.paddle.lite.fbs.proto.CombinedParamsDesc = class CombinedParamsDesc {

    static create(reader) {
        return $root.paddle.lite.fbs.proto.CombinedParamsDesc.decode(reader, reader.root);
    }

    static createText(reader) {
        return $root.paddle.lite.fbs.proto.CombinedParamsDesc.decodeText(reader, reader.root);
    }

    static decode(reader, position) {
        const $ = new $root.paddle.lite.fbs.proto.CombinedParamsDesc();
        $.params = reader.tableArray(position, 4, $root.paddle.lite.fbs.proto.ParamDesc.decode);
        return $;
    }

    static decodeText(reader, json) {
        const $ = new $root.paddle.lite.fbs.proto.CombinedParamsDesc();
        $.params = reader.objectArray(json.params, $root.paddle.lite.fbs.proto.ParamDesc.decodeText);
        return $;
    }
};

$root.paddle.lite.fbs.proto.ParamDesc = class ParamDesc {

    static create(reader) {
        return $root.paddle.lite.fbs.proto.ParamDesc.decode(reader, reader.root);
    }

    static createText(reader) {
        return $root.paddle.lite.fbs.proto.ParamDesc.decodeText(reader, reader.root);
    }

    static decode(reader, position) {
        const $ = new $root.paddle.lite.fbs.proto.ParamDesc();
        $.version = reader.table(position, 4, $root.paddle.lite.fbs.proto.ParamDesc_.VersionDesc.decode);
        $.name = reader.string_(position, 6, null);
        $.variable = reader.union(position, 8, $root.paddle.lite.fbs.proto.ParamDesc_.VariableDesc.decode);
        return $;
    }

    static decodeText(reader, json) {
        const $ = new $root.paddle.lite.fbs.proto.ParamDesc();
        $.version = reader.object(json.version, $root.paddle.lite.fbs.proto.ParamDesc_.VersionDesc.decodeText);
        $.name = reader.value(json.name, null);
        $.variable = $root.paddle.lite.fbs.proto.ParamDesc_.VariableDesc.decodeText(reader, json.variable, json.variable_type);
        return $;
    }
};

$root.paddle = $root.paddle || {};

$root.paddle.lite = $root.paddle.lite || {};

$root.paddle.lite.fbs = $root.paddle.lite.fbs || {};

$root.paddle.lite.fbs.proto = $root.paddle.lite.fbs.proto || {};

$root.paddle.lite.fbs.proto.VarType_ = $root.paddle.lite.fbs.proto.VarType_ || {};

$root.paddle.lite.fbs.proto.VarType_.Type = {
    BOOL: 0,
    INT16: 1,
    INT32: 2,
    INT64: 3,
    FP16: 4,
    FP32: 5,
    FP64: 6,
    LOD_TENSOR: 7,
    SELECTED_ROWS: 8,
    FEED_MINIBATCH: 9,
    FETCH_LIST: 10,
    STEP_SCOPES: 11,
    LOD_RANK_TABLE: 12,
    LOD_TENSOR_ARRAY: 13,
    PLACE_LIST: 14,
    READER: 15,
    RAW: 17,
    TUPLE: 18,
    SIZE_T: 19,
    UINT8: 20,
    INT8: 21
};

$root.paddle.lite.fbs.proto.VarType_.TensorDesc = class TensorDesc {

    static decode(reader, position) {
        const $ = new $root.paddle.lite.fbs.proto.VarType_.TensorDesc();
        $.data_type = reader.int32_(position, 4, 0);
        $.dims = reader.int64s_(position, 6);
        return $;
    }

    static decodeText(reader, json) {
        const $ = new $root.paddle.lite.fbs.proto.VarType_.TensorDesc();
        $.data_type = $root.paddle.lite.fbs.proto.VarType_.Type[json.data_type];
        $.dims = reader.array(json.dims);
        return $;
    }
};

$root.paddle.lite.fbs.proto.VarType_.LoDTensorDesc = class LoDTensorDesc {

    static decode(reader, position) {
        const $ = new $root.paddle.lite.fbs.proto.VarType_.LoDTensorDesc();
        $.tensor = reader.table(position, 4, $root.paddle.lite.fbs.proto.VarType_.TensorDesc.decode);
        $.lod_level = reader.int32_(position, 6, 0);
        return $;
    }

    static decodeText(reader, json) {
        const $ = new $root.paddle.lite.fbs.proto.VarType_.LoDTensorDesc();
        $.tensor = reader.object(json.tensor, $root.paddle.lite.fbs.proto.VarType_.TensorDesc.decodeText);
        $.lod_level = reader.value(json.lod_level, 0);
        return $;
    }
};

$root.paddle.lite.fbs.proto.VarType_.LoDTensorArrayDesc = class LoDTensorArrayDesc {

    static decode(reader, position) {
        const $ = new $root.paddle.lite.fbs.proto.VarType_.LoDTensorArrayDesc();
        $.tensor = reader.table(position, 4, $root.paddle.lite.fbs.proto.VarType_.TensorDesc.decode);
        $.lod_level = reader.int32_(position, 6, 0);
        return $;
    }

    static decodeText(reader, json) {
        const $ = new $root.paddle.lite.fbs.proto.VarType_.LoDTensorArrayDesc();
        $.tensor = reader.object(json.tensor, $root.paddle.lite.fbs.proto.VarType_.TensorDesc.decodeText);
        $.lod_level = reader.value(json.lod_level, 0);
        return $;
    }
};

$root.paddle.lite.fbs.proto.VarType_.ReaderDesc = class ReaderDesc {

    static decode(reader, position) {
        const $ = new $root.paddle.lite.fbs.proto.VarType_.ReaderDesc();
        $.lod_tensor = reader.tableArray(position, 4, $root.paddle.lite.fbs.proto.VarType_.LoDTensorDesc.decode);
        return $;
    }

    static decodeText(reader, json) {
        const $ = new $root.paddle.lite.fbs.proto.VarType_.ReaderDesc();
        $.lod_tensor = reader.objectArray(json.lod_tensor, $root.paddle.lite.fbs.proto.VarType_.LoDTensorDesc.decodeText);
        return $;
    }
};

$root.paddle.lite.fbs.proto.VarType_.Tuple = class Tuple {

    static decode(reader, position) {
        const $ = new $root.paddle.lite.fbs.proto.VarType_.Tuple();
        $.element_type = reader.typedArray(position, 4, Int32Array);
        return $;
    }

    static decodeText(reader, json) {
        const $ = new $root.paddle.lite.fbs.proto.VarType_.Tuple();
        $.element_type = reader.objectArray(json.element_type, $root.paddle.lite.fbs.proto.VarType_.Type.decodeText);
        return $;
    }
};

$root.paddle = $root.paddle || {};

$root.paddle.lite = $root.paddle.lite || {};

$root.paddle.lite.fbs = $root.paddle.lite.fbs || {};

$root.paddle.lite.fbs.proto = $root.paddle.lite.fbs.proto || {};

$root.paddle.lite.fbs.proto.CompatibleInfo_ = $root.paddle.lite.fbs.proto.CompatibleInfo_ || {};

$root.paddle.lite.fbs.proto.CompatibleInfo_.Type = {
    COMPATIBLE: 0,
    DEFINITELY_NOT: 1,
    POSSIBLE: 2,
    BUG_FIX: 3,
    PRECISION_CHANGE: 4
};

$root.paddle = $root.paddle || {};

$root.paddle.lite = $root.paddle.lite || {};

$root.paddle.lite.fbs = $root.paddle.lite.fbs || {};

$root.paddle.lite.fbs.proto = $root.paddle.lite.fbs.proto || {};

$root.paddle.lite.fbs.proto.OpDesc_ = $root.paddle.lite.fbs.proto.OpDesc_ || {};

$root.paddle.lite.fbs.proto.OpDesc_.Attr = class Attr {

    static decode(reader, position) {
        const $ = new $root.paddle.lite.fbs.proto.OpDesc_.Attr();
        $.name = reader.string_(position, 4, null);
        $.type = reader.int32_(position, 6, 0);
        $.i = reader.int32_(position, 8, 0);
        $.f = reader.float32_(position, 10, 0);
        $.s = reader.string_(position, 12, null);
        $.ints = reader.typedArray(position, 14, Int32Array);
        $.floats = reader.typedArray(position, 16, Float32Array);
        $.strings = reader.strings_(position, 18);
        $.b = reader.bool_(position, 20, false);
        $.bools = reader.bools_(position, 22);
        $.block_idx = reader.int32_(position, 24, 0);
        $.l = reader.int64_(position, 26, 0);
        $.blocks_idx = reader.typedArray(position, 28, Int32Array);
        $.longs = reader.int64s_(position, 30);
        $.float64 = reader.float64_(position, 32, 0);
        $.float64s = reader.typedArray(position, 34, Float64Array);
        return $;
    }

    static decodeText(reader, json) {
        const $ = new $root.paddle.lite.fbs.proto.OpDesc_.Attr();
        $.name = reader.value(json.name, null);
        $.type = $root.paddle.lite.fbs.proto.AttrType[json.type];
        $.i = reader.value(json.i, 0);
        $.f = reader.value(json.f, 0);
        $.s = reader.value(json.s, null);
        $.ints = reader.typedArray(json.ints, Int32Array);
        $.floats = reader.typedArray(json.floats, Float32Array);
        $.strings = reader.array(json.strings);
        $.b = reader.value(json.b, false);
        $.bools = reader.array(json.bools);
        $.block_idx = reader.value(json.block_idx, 0);
        $.l = reader.value(json.l, 0);
        $.blocks_idx = reader.typedArray(json.blocks_idx, Int32Array);
        $.longs = reader.array(json.longs);
        $.float64 = reader.value(json.float64, 0);
        $.float64s = reader.typedArray(json.float64s, Float64Array);
        return $;
    }
};

$root.paddle.lite.fbs.proto.OpDesc_.Var = class Var {

    static decode(reader, position) {
        const $ = new $root.paddle.lite.fbs.proto.OpDesc_.Var();
        $.parameter = reader.string_(position, 4, null);
        $.arguments = reader.strings_(position, 6);
        return $;
    }

    static decodeText(reader, json) {
        const $ = new $root.paddle.lite.fbs.proto.OpDesc_.Var();
        $.parameter = reader.value(json.parameter, null);
        $.arguments = reader.array(json.arguments);
        return $;
    }
};

$root.paddle = $root.paddle || {};

$root.paddle.lite = $root.paddle.lite || {};

$root.paddle.lite.fbs = $root.paddle.lite.fbs || {};

$root.paddle.lite.fbs.proto = $root.paddle.lite.fbs.proto || {};

$root.paddle.lite.fbs.proto.OpVersionMap_ = $root.paddle.lite.fbs.proto.OpVersionMap_ || {};

$root.paddle.lite.fbs.proto.OpVersionMap_.OpVersionPair = class OpVersionPair {

    static decode(reader, position) {
        const $ = new $root.paddle.lite.fbs.proto.OpVersionMap_.OpVersionPair();
        $.op_name = reader.string_(position, 4, null);
        $.op_version = reader.table(position, 6, $root.paddle.lite.fbs.proto.OpVersion.decode);
        return $;
    }

    static decodeText(reader, json) {
        const $ = new $root.paddle.lite.fbs.proto.OpVersionMap_.OpVersionPair();
        $.op_name = reader.value(json.op_name, null);
        $.op_version = reader.object(json.op_version, $root.paddle.lite.fbs.proto.OpVersion.decodeText);
        return $;
    }
};

$root.paddle = $root.paddle || {};

$root.paddle.lite = $root.paddle.lite || {};

$root.paddle.lite.fbs = $root.paddle.lite.fbs || {};

$root.paddle.lite.fbs.proto = $root.paddle.lite.fbs.proto || {};

$root.paddle.lite.fbs.proto.ParamDesc_ = $root.paddle.lite.fbs.proto.ParamDesc_ || {};

$root.paddle.lite.fbs.proto.ParamDesc_.LoDTensorDesc = class LoDTensorDesc {

    static decode(reader, position) {
        const $ = new $root.paddle.lite.fbs.proto.ParamDesc_.LoDTensorDesc();
        $.lod_level = reader.int32_(position, 4, 0);
        $.lod = reader.int64s_(position, 6);
        $.dim = reader.int64s_(position, 8);
        $.data_type = reader.int32_(position, 10, 0);
        $.data = reader.typedArray(position, 12, Int8Array);
        return $;
    }

    static decodeText(reader, json) {
        const $ = new $root.paddle.lite.fbs.proto.ParamDesc_.LoDTensorDesc();
        $.lod_level = reader.value(json.lod_level, 0);
        $.lod = reader.array(json.lod);
        $.dim = reader.array(json.dim);
        $.data_type = $root.paddle.lite.fbs.proto.VarType_.Type[json.data_type];
        $.data = reader.typedArray(json.data, Int8Array);
        return $;
    }
};

$root.paddle.lite.fbs.proto.ParamDesc_.VersionDesc = class VersionDesc {

    static decode(reader, position) {
        const $ = new $root.paddle.lite.fbs.proto.ParamDesc_.VersionDesc();
        $.version = reader.int32_(position, 4, 0);
        $.model_version = reader.int32_(position, 6, 0);
        return $;
    }

    static decodeText(reader, json) {
        const $ = new $root.paddle.lite.fbs.proto.ParamDesc_.VersionDesc();
        $.version = reader.value(json.version, 0);
        $.model_version = reader.value(json.model_version, 0);
        return $;
    }
};

$root.paddle.lite.fbs.proto.ParamDesc_.VariableDesc = class {

    static decode(reader, position, type) {
        switch (type) {
            case 1: return $root.paddle.lite.fbs.proto.ParamDesc_.LoDTensorDesc.decode(reader, position);
            default: return undefined;
        }
    }

    static decodeText(reader, json, type) {
        switch (type) {
            case 'LoDTensorDesc': return $root.paddle.lite.fbs.proto.ParamDesc_.LoDTensorDesc.decodeText(reader, json);
            default: return undefined;
        }
    }
};
