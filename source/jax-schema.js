
export const jax_export = {};

jax_export.serialization = jax_export.serialization || {};

jax_export.serialization.PyTreeDefKind = {
    leaf: 0, '0': 'leaf',
    none: 1, '1': 'none',
    tuple: 2, '2': 'tuple',
    list: 3, '3': 'list',
    dict: 4, '4': 'dict',
    custom: 5, '5': 'custom'
};

jax_export.serialization.PyTreeDef = class PyTreeDef {

    static decode(reader, position) {
        const $ = new jax_export.serialization.PyTreeDef();
        $.kind = reader.int8_(position, 4, 0);
        $.children = reader.tables(position, 6, jax_export.serialization.PyTreeDef);
        $.children_names = reader.strings_(position, 8);
        $.custom_name = reader.string_(position, 10, null);
        $.custom_auxdata = reader.array(position, 12, Int8Array);
        return $;
    }
};

jax_export.serialization.AbstractValueKind = {
    shapedArray: 0, '0': 'shapedArray',
    abstractToken: 1, '1': 'abstractToken'
};

jax_export.serialization.DType = {
    bool: 0, '0': 'bool',
    i8: 1, '1': 'i8',
    i16: 2, '2': 'i16',
    i32: 3, '3': 'i32',
    i64: 4, '4': 'i64',
    ui8: 5, '5': 'ui8',
    ui16: 6, '6': 'ui16',
    ui32: 7, '7': 'ui32',
    ui64: 8, '8': 'ui64',
    f0: 22, '22': 'f0',
    f16: 9, '9': 'f16',
    f32: 10, '10': 'f32',
    f64: 11, '11': 'f64',
    c64: 12, '12': 'c64',
    c128: 13, '13': 'c128',
    bf16: 14, '14': 'bf16',
    i4: 15, '15': 'i4',
    ui4: 16, '16': 'ui4',
    f8_e3m4: 24, '24': 'f8_e3m4',
    f8_e4m3: 23, '23': 'f8_e4m3',
    f8_e4m3b11fnuz: 17, '17': 'f8_e4m3b11fnuz',
    f8_e4m3fn: 18, '18': 'f8_e4m3fn',
    f8_e4m3fnuz: 19, '19': 'f8_e4m3fnuz',
    f8_e5m2: 20, '20': 'f8_e5m2',
    f8_e5m2fnuz: 21, '21': 'f8_e5m2fnuz',
    f8_e8m0fnu: 25, '25': 'f8_e8m0fnu',
    f4_e2m1fn: 26, '26': 'f4_e2m1fn',
    key_fry: 27, '27': 'key_fry',
    key_rbg: 28, '28': 'key_rbg',
    key_unsafe_rbg: 29, '29': 'key_unsafe_rbg'
};

jax_export.serialization.MemorySpace = {
    Missing: 0, '0': 'Missing',
    Device: 1, '1': 'Device',
    Host: 2, '2': 'Host',
    Any: 3, '3': 'Any'
};

jax_export.serialization.AxisType = {
    Missing: 0, '0': 'Missing',
    Auto: 1, '1': 'Auto',
    Explicit: 2, '2': 'Explicit',
    Manual: 3, '3': 'Manual'
};

jax_export.serialization.AbstractDevice = class AbstractDevice {

    static decode(reader, position) {
        const $ = new jax_export.serialization.AbstractDevice();
        $.device_kind = reader.string_(position, 4, null);
        $.num_cores = reader.uint32_(position, 6, null);
        return $;
    }
};

jax_export.serialization.AbstractMesh = class AbstractMesh {

    static decode(reader, position) {
        const $ = new jax_export.serialization.AbstractMesh();
        $.axis_sizes = reader.array(position, 4, Uint32Array);
        $.axis_names = reader.strings_(position, 6);
        $.axis_types = reader.array(position, 8, Int8Array);
        $.abstract_device = reader.table(position, 10, jax_export.serialization.AbstractDevice);
        return $;
    }
};

jax_export.serialization.PartitionSpecOneAxis = class PartitionSpecOneAxis {

    static decode(reader, position) {
        const $ = new jax_export.serialization.PartitionSpecOneAxis();
        $.axes = reader.strings_(position, 4);
        return $;
    }
};

jax_export.serialization.PartitionSpec = class PartitionSpec {

    static decode(reader, position) {
        const $ = new jax_export.serialization.PartitionSpec();
        $.partitions = reader.tables(position, 4, jax_export.serialization.PartitionSpecOneAxis);
        $.reduced = reader.strings_(position, 6);
        $.unreduced = reader.strings_(position, 8);
        return $;
    }
};

jax_export.serialization.NamedSharding = class NamedSharding {

    static decode(reader, position) {
        const $ = new jax_export.serialization.NamedSharding();
        $.mesh = reader.table(position, 4, jax_export.serialization.AbstractMesh);
        $.spec = reader.table(position, 6, jax_export.serialization.PartitionSpec);
        $.memory_kind = reader.string_(position, 8, null);
        return $;
    }
};

jax_export.serialization.AbstractValue = class AbstractValue {

    static decode(reader, position) {
        const $ = new jax_export.serialization.AbstractValue();
        $.kind = reader.int8_(position, 4, 0);
        $.shape = reader.strings_(position, 6);
        $.dtype = reader.int8_(position, 8, 0);
        $.memory_space = reader.int8_(position, 10, 0);
        return $;
    }
};

jax_export.serialization.ShardingKind = {
    unspecified: 0, '0': 'unspecified',
    hlo_sharding: 1, '1': 'hlo_sharding',
    named_sharding: 2, '2': 'named_sharding'
};

jax_export.serialization.Sharding = class Sharding {

    static decode(reader, position) {
        const $ = new jax_export.serialization.Sharding();
        $.kind = reader.int8_(position, 4, 0);
        $.hlo_sharding_proto = reader.array(position, 6, Int8Array);
        $.named_sharding = reader.table(position, 8, jax_export.serialization.NamedSharding);
        return $;
    }
};

jax_export.serialization.Effect = class Effect {

    static decode(reader, position) {
        const $ = new jax_export.serialization.Effect();
        $.type_name = reader.string_(position, 4, null);
        return $;
    }
};

jax_export.serialization.DisabledSafetyCheckKind = {
    platform: 0, '0': 'platform',
    custom_call: 1, '1': 'custom_call',
    shape_assertions: 2, '2': 'shape_assertions'
};

jax_export.serialization.DisabledSafetyCheck = class DisabledSafetyCheck {

    static decode(reader, position) {
        const $ = new jax_export.serialization.DisabledSafetyCheck();
        $.kind = reader.int8_(position, 4, 0);
        $.custom_call_target = reader.string_(position, 6, null);
        return $;
    }
};

jax_export.serialization.Exported = class Exported {

    static create(reader) {
        return jax_export.serialization.Exported.decode(reader, reader.root);
    }

    static decode(reader, position) {
        const $ = new jax_export.serialization.Exported();
        $.serialization_version = reader.uint16_(position, 4, 0);
        $.function_name = reader.string_(position, 6, null);
        $.in_tree = reader.table(position, 8, jax_export.serialization.PyTreeDef);
        $.in_avals = reader.tables(position, 10, jax_export.serialization.AbstractValue);
        $.out_tree = reader.table(position, 12, jax_export.serialization.PyTreeDef);
        $.out_avals = reader.tables(position, 14, jax_export.serialization.AbstractValue);
        $.nr_devices_short = reader.int16_(position, 16, 0);
        $.in_shardings = reader.tables(position, 18, jax_export.serialization.Sharding);
        $.out_shardings = reader.tables(position, 20, jax_export.serialization.Sharding);
        $.platforms = reader.strings_(position, 22);
        $.ordered_effects = reader.tables(position, 24, jax_export.serialization.Effect);
        $.unordered_effects = reader.tables(position, 26, jax_export.serialization.Effect);
        $.disabled_checks = reader.tables(position, 28, jax_export.serialization.DisabledSafetyCheck);
        $.mlir_module_serialized = reader.array(position, 30, Int8Array);
        $.calling_convention_version = reader.uint16_(position, 32, 0);
        $.module_kept_var_idx = reader.array(position, 34, Uint16Array);
        $.uses_global_constants = reader.bool_(position, 36, false);
        $.vjp = reader.table(position, 38, jax_export.serialization.Exported);
        $.nr_devices = reader.uint32_(position, 40, 0);
        return $;
    }
};
