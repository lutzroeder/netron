
export const tosa = {};

tosa.DType = {
    UNKNOWN: 0,
    BOOL: 1,
    UINT8: 2,
    INT4: 3,
    INT8: 4,
    INT16: 5,
    INT32: 6,
    INT48: 7,
    FP32: 8,
    UINT16: 9,
    FP16: 10,
    BF16: 11,
    SHAPE: 12
};

tosa.ResizeMode = {
    UNKNOWN: 0,
    NEAREST: 1,
    BILINEAR: 2
};

tosa.Op = {
    UNKNOWN: 0,
    ARGMAX: 1,
    AVG_POOL2D: 2,
    CONV2D: 3,
    CONV3D: 4,
    DEPTHWISE_CONV2D: 5,
    FULLY_CONNECTED: 6,
    MATMUL: 7,
    MAX_POOL2D: 8,
    TRANSPOSE_CONV2D: 9,
    CLAMP: 10,
    RESERVED: 11,
    SIGMOID: 12,
    TANH: 13,
    ADD: 14,
    ARITHMETIC_RIGHT_SHIFT: 15,
    BITWISE_AND: 16,
    BITWISE_OR: 17,
    BITWISE_XOR: 18,
    INTDIV: 19,
    LOGICAL_AND: 20,
    LOGICAL_LEFT_SHIFT: 21,
    LOGICAL_RIGHT_SHIFT: 22,
    LOGICAL_OR: 23,
    LOGICAL_XOR: 24,
    MAXIMUM: 25,
    MINIMUM: 26,
    MUL: 27,
    POW: 28,
    SUB: 29,
    TABLE: 30,
    ABS: 31,
    BITWISE_NOT: 32,
    CEIL: 33,
    CLZ: 34,
    EXP: 35,
    FLOOR: 36,
    LOG: 37,
    LOGICAL_NOT: 38,
    NEGATE: 39,
    RECIPROCAL: 40,
    RSQRT: 41,
    SELECT: 42,
    EQUAL: 43,
    GREATER: 44,
    GREATER_EQUAL: 45,
    REDUCE_ANY: 46,
    REDUCE_ALL: 47,
    REDUCE_MAX: 48,
    REDUCE_MIN: 49,
    REDUCE_PRODUCT: 50,
    REDUCE_SUM: 51,
    CONCAT: 52,
    PAD: 53,
    RESHAPE: 54,
    REVERSE: 55,
    SLICE: 56,
    TILE: 57,
    TRANSPOSE: 58,
    GATHER: 59,
    SCATTER: 60,
    RESIZE: 61,
    CAST: 62,
    RESCALE: 63,
    CONST: 64,
    IDENTITY: 65,
    CUSTOM: 66,
    COND_IF: 67,
    WHILE_LOOP: 68,
    FFT2D: 69,
    RFFT2D: 70,
    ERF: 71,
    DIM: 72
};

tosa.Attribute = class {

    static decode(reader, position, type) {
        switch (type) {
            case 1: return tosa.PoolAttribute.decode(reader, position);
            case 2: return tosa.ConvAttribute.decode(reader, position);
            case 3: return tosa.TransposeConvAttribute.decode(reader, position);
            case 4: return tosa.PadAttribute.decode(reader, position);
            case 5: return tosa.AxisAttribute.decode(reader, position);
            case 6: return tosa.ReshapeAttribute.decode(reader, position);
            case 7: return tosa.SliceAttribute.decode(reader, position);
            case 8: return tosa.TileAttribute.decode(reader, position);
            case 9: return tosa.ResizeAttribute.decode(reader, position);
            case 10: return tosa.ClampAttribute.decode(reader, position);
            case 11: return tosa.RescaleAttribute.decode(reader, position);
            case 12: return tosa.MulAttribute.decode(reader, position);
            case 13: return tosa.ArithmeticRightShiftAttribute.decode(reader, position);
            case 14: return tosa.CondIfAttribute.decode(reader, position);
            case 15: return tosa.WhileLoopAttribute.decode(reader, position);
            case 16: return tosa.TransposeAttribute.decode(reader, position);
            case 17: return tosa.TableAttribute.decode(reader, position);
            case 18: return tosa.MatMulAttribute.decode(reader, position);
            case 19: return tosa.FullyConnectedAttribute.decode(reader, position);
            case 20: return tosa.NegateAttribute.decode(reader, position);
            case 21: return tosa.CustomAttribute.decode(reader, position);
            case 22: return tosa.FFTAttribute.decode(reader, position);
            case 23: return tosa.RFFTAttribute.decode(reader, position);
            default: return undefined;
        }
    }

    static decodeText(reader, json, type) {
        switch (type) {
            case 'PoolAttribute': return tosa.PoolAttribute.decodeText(reader, json);
            case 'ConvAttribute': return tosa.ConvAttribute.decodeText(reader, json);
            case 'TransposeConvAttribute': return tosa.TransposeConvAttribute.decodeText(reader, json);
            case 'PadAttribute': return tosa.PadAttribute.decodeText(reader, json);
            case 'AxisAttribute': return tosa.AxisAttribute.decodeText(reader, json);
            case 'ReshapeAttribute': return tosa.ReshapeAttribute.decodeText(reader, json);
            case 'SliceAttribute': return tosa.SliceAttribute.decodeText(reader, json);
            case 'TileAttribute': return tosa.TileAttribute.decodeText(reader, json);
            case 'ResizeAttribute': return tosa.ResizeAttribute.decodeText(reader, json);
            case 'ClampAttribute': return tosa.ClampAttribute.decodeText(reader, json);
            case 'RescaleAttribute': return tosa.RescaleAttribute.decodeText(reader, json);
            case 'MulAttribute': return tosa.MulAttribute.decodeText(reader, json);
            case 'ArithmeticRightShiftAttribute': return tosa.ArithmeticRightShiftAttribute.decodeText(reader, json);
            case 'CondIfAttribute': return tosa.CondIfAttribute.decodeText(reader, json);
            case 'WhileLoopAttribute': return tosa.WhileLoopAttribute.decodeText(reader, json);
            case 'TransposeAttribute': return tosa.TransposeAttribute.decodeText(reader, json);
            case 'TableAttribute': return tosa.TableAttribute.decodeText(reader, json);
            case 'MatMulAttribute': return tosa.MatMulAttribute.decodeText(reader, json);
            case 'FullyConnectedAttribute': return tosa.FullyConnectedAttribute.decodeText(reader, json);
            case 'NegateAttribute': return tosa.NegateAttribute.decodeText(reader, json);
            case 'CustomAttribute': return tosa.CustomAttribute.decodeText(reader, json);
            case 'FFTAttribute': return tosa.FFTAttribute.decodeText(reader, json);
            case 'RFFTAttribute': return tosa.RFFTAttribute.decodeText(reader, json);
            default: return undefined;
        }
    }
};

tosa.PoolAttribute = class PoolAttribute {

    static decode(reader, position) {
        const $ = new tosa.PoolAttribute();
        $.pad = reader.array(position, 4, Int32Array);
        $.kernel = reader.array(position, 6, Int32Array);
        $.stride = reader.array(position, 8, Int32Array);
        $.input_zp = reader.int32_(position, 10, 0);
        $.output_zp = reader.int32_(position, 12, 0);
        $.accum_dtype = reader.uint32_(position, 14, 0);
        return $;
    }

    static decodeText(reader, json) {
        const $ = new tosa.PoolAttribute();
        $.pad = reader.array(json.pad, Int32Array);
        $.kernel = reader.array(json.kernel, Int32Array);
        $.stride = reader.array(json.stride, Int32Array);
        $.input_zp = reader.value(json.input_zp, 0);
        $.output_zp = reader.value(json.output_zp, 0);
        $.accum_dtype = tosa.DType[json.accum_dtype];
        return $;
    }
};

tosa.ConvAttribute = class ConvAttribute {

    static decode(reader, position) {
        const $ = new tosa.ConvAttribute();
        $.pad = reader.array(position, 4, Int32Array);
        $.stride = reader.array(position, 6, Int32Array);
        $.dilation = reader.array(position, 8, Int32Array);
        $.input_zp = reader.int32_(position, 10, 0);
        $.weight_zp = reader.int32_(position, 12, 0);
        $.local_bound = reader.bool_(position, 14, false);
        return $;
    }

    static decodeText(reader, json) {
        const $ = new tosa.ConvAttribute();
        $.pad = reader.array(json.pad, Int32Array);
        $.stride = reader.array(json.stride, Int32Array);
        $.dilation = reader.array(json.dilation, Int32Array);
        $.input_zp = reader.value(json.input_zp, 0);
        $.weight_zp = reader.value(json.weight_zp, 0);
        $.local_bound = reader.value(json.local_bound, false);
        return $;
    }
};

tosa.TransposeConvAttribute = class TransposeConvAttribute {

    static decode(reader, position) {
        const $ = new tosa.TransposeConvAttribute();
        $.out_pad = reader.array(position, 4, Int32Array);
        $.stride = reader.array(position, 6, Int32Array);
        $.output_shape = reader.array(position, 8, Int32Array);
        $.input_zp = reader.int32_(position, 10, 0);
        $.weight_zp = reader.int32_(position, 12, 0);
        $.local_bound = reader.bool_(position, 14, false);
        return $;
    }

    static decodeText(reader, json) {
        const $ = new tosa.TransposeConvAttribute();
        $.out_pad = reader.array(json.out_pad, Int32Array);
        $.stride = reader.array(json.stride, Int32Array);
        $.output_shape = reader.array(json.output_shape, Int32Array);
        $.input_zp = reader.value(json.input_zp, 0);
        $.weight_zp = reader.value(json.weight_zp, 0);
        $.local_bound = reader.value(json.local_bound, false);
        return $;
    }
};

tosa.PadAttribute = class PadAttribute {

    static decode(reader, position) {
        const $ = new tosa.PadAttribute();
        $.padding = reader.array(position, 4, Int32Array);
        $.pad_const_int = reader.int32_(position, 6, 0);
        $.pad_const_fp = reader.array(position, 8, Uint8Array);
        return $;
    }

    static decodeText(reader, json) {
        const $ = new tosa.PadAttribute();
        $.padding = reader.array(json.padding, Int32Array);
        $.pad_const_int = reader.value(json.pad_const_int, 0);
        $.pad_const_fp = reader.array(json.pad_const_fp, Uint8Array);
        return $;
    }
};

tosa.AxisAttribute = class AxisAttribute {

    static decode(reader, position) {
        const $ = new tosa.AxisAttribute();
        $.axis = reader.int32_(position, 4, 0);
        return $;
    }

    static decodeText(reader, json) {
        const $ = new tosa.AxisAttribute();
        $.axis = reader.value(json.axis, 0);
        return $;
    }
};

tosa.ReshapeAttribute = class ReshapeAttribute {

    static decode(reader, position) {
        const $ = new tosa.ReshapeAttribute();
        $.new_shape = reader.array(position, 4, Int32Array);
        return $;
    }

    static decodeText(reader, json) {
        const $ = new tosa.ReshapeAttribute();
        $.new_shape = reader.array(json.new_shape, Int32Array);
        return $;
    }
};

tosa.SliceAttribute = class SliceAttribute {

    static decode(reader, position) {
        const $ = new tosa.SliceAttribute();
        $.start = reader.array(position, 4, Int32Array);
        $.size = reader.array(position, 6, Int32Array);
        return $;
    }

    static decodeText(reader, json) {
        const $ = new tosa.SliceAttribute();
        $.start = reader.array(json.start, Int32Array);
        $.size = reader.array(json.size, Int32Array);
        return $;
    }
};

tosa.TileAttribute = class TileAttribute {

    static decode(reader, position) {
        const $ = new tosa.TileAttribute();
        $.multiples = reader.array(position, 4, Int32Array);
        return $;
    }

    static decodeText(reader, json) {
        const $ = new tosa.TileAttribute();
        $.multiples = reader.array(json.multiples, Int32Array);
        return $;
    }
};

tosa.ResizeAttribute = class ResizeAttribute {

    static decode(reader, position) {
        const $ = new tosa.ResizeAttribute();
        $.scale = reader.array(position, 4, Int16Array);
        $.offset = reader.array(position, 6, Int16Array);
        $.border = reader.array(position, 8, Int16Array);
        $.mode = reader.uint32_(position, 10, 0);
        return $;
    }

    static decodeText(reader, json) {
        const $ = new tosa.ResizeAttribute();
        $.scale = reader.array(json.scale, Int16Array);
        $.offset = reader.array(json.offset, Int16Array);
        $.border = reader.array(json.border, Int16Array);
        $.mode = tosa.ResizeMode[json.mode];
        return $;
    }
};

tosa.ClampAttribute = class ClampAttribute {

    static decode(reader, position) {
        const $ = new tosa.ClampAttribute();
        $.min_int = reader.int32_(position, 4, 0);
        $.max_int = reader.int32_(position, 6, 0);
        $.min_fp = reader.array(position, 8, Uint8Array);
        $.max_fp = reader.array(position, 10, Uint8Array);
        return $;
    }

    static decodeText(reader, json) {
        const $ = new tosa.ClampAttribute();
        $.min_int = reader.value(json.min_int, 0);
        $.max_int = reader.value(json.max_int, 0);
        $.min_fp = reader.array(json.min_fp, Uint8Array);
        $.max_fp = reader.array(json.max_fp, Uint8Array);
        return $;
    }
};

tosa.RescaleAttribute = class RescaleAttribute {

    static decode(reader, position) {
        const $ = new tosa.RescaleAttribute();
        $.input_zp = reader.int32_(position, 4, 0);
        $.output_zp = reader.int32_(position, 6, 0);
        $.multiplier = reader.array(position, 8, Int32Array);
        $.shift = reader.array(position, 10, Int32Array);
        $.scale32 = reader.bool_(position, 12, false);
        $.double_round = reader.bool_(position, 14, false);
        $.per_channel = reader.bool_(position, 16, false);
        $.input_unsigned = reader.bool_(position, 18, false);
        $.output_unsigned = reader.bool_(position, 20, false);
        return $;
    }

    static decodeText(reader, json) {
        const $ = new tosa.RescaleAttribute();
        $.input_zp = reader.value(json.input_zp, 0);
        $.output_zp = reader.value(json.output_zp, 0);
        $.multiplier = reader.array(json.multiplier, Int32Array);
        $.shift = reader.array(json.shift, Int32Array);
        $.scale32 = reader.value(json.scale32, false);
        $.double_round = reader.value(json.double_round, false);
        $.per_channel = reader.value(json.per_channel, false);
        $.input_unsigned = reader.value(json.input_unsigned, false);
        $.output_unsigned = reader.value(json.output_unsigned, false);
        return $;
    }
};

tosa.MulAttribute = class MulAttribute {

    static decode(reader, position) {
        const $ = new tosa.MulAttribute();
        $.shift = reader.int32_(position, 4, 0);
        return $;
    }

    static decodeText(reader, json) {
        const $ = new tosa.MulAttribute();
        $.shift = reader.value(json.shift, 0);
        return $;
    }
};

tosa.ArithmeticRightShiftAttribute = class ArithmeticRightShiftAttribute {

    static decode(reader, position) {
        const $ = new tosa.ArithmeticRightShiftAttribute();
        $.round = reader.bool_(position, 4, false);
        return $;
    }

    static decodeText(reader, json) {
        const $ = new tosa.ArithmeticRightShiftAttribute();
        $.round = reader.value(json.round, false);
        return $;
    }
};

tosa.CondIfAttribute = class CondIfAttribute {

    static decode(reader, position) {
        const $ = new tosa.CondIfAttribute();
        $.then_branch = reader.string_(position, 4, null);
        $.else_branch = reader.string_(position, 6, null);
        return $;
    }

    static decodeText(reader, json) {
        const $ = new tosa.CondIfAttribute();
        $.then_branch = reader.value(json.then_branch, null);
        $.else_branch = reader.value(json.else_branch, null);
        return $;
    }
};

tosa.WhileLoopAttribute = class WhileLoopAttribute {

    static decode(reader, position) {
        const $ = new tosa.WhileLoopAttribute();
        $.cond_branch = reader.string_(position, 4, null);
        $.body_branch = reader.string_(position, 6, null);
        return $;
    }

    static decodeText(reader, json) {
        const $ = new tosa.WhileLoopAttribute();
        $.cond_branch = reader.value(json.cond_branch, null);
        $.body_branch = reader.value(json.body_branch, null);
        return $;
    }
};

tosa.TransposeAttribute = class TransposeAttribute {

    static decode(reader, position) {
        const $ = new tosa.TransposeAttribute();
        $.perms = reader.array(position, 4, Int32Array);
        return $;
    }

    static decodeText(reader, json) {
        const $ = new tosa.TransposeAttribute();
        $.perms = reader.array(json.perms, Int32Array);
        return $;
    }
};

tosa.TableAttribute = class TableAttribute {

    static decode(reader, position) {
        const $ = new tosa.TableAttribute();
        $.table = reader.array(position, 4, Int16Array);
        return $;
    }

    static decodeText(reader, json) {
        const $ = new tosa.TableAttribute();
        $.table = reader.array(json.table, Int16Array);
        return $;
    }
};

tosa.MatMulAttribute = class MatMulAttribute {

    static decode(reader, position) {
        const $ = new tosa.MatMulAttribute();
        $.a_zp = reader.int32_(position, 4, 0);
        $.b_zp = reader.int32_(position, 6, 0);
        return $;
    }

    static decodeText(reader, json) {
        const $ = new tosa.MatMulAttribute();
        $.a_zp = reader.value(json.a_zp, 0);
        $.b_zp = reader.value(json.b_zp, 0);
        return $;
    }
};

tosa.FullyConnectedAttribute = class FullyConnectedAttribute {

    static decode(reader, position) {
        const $ = new tosa.FullyConnectedAttribute();
        $.input_zp = reader.int32_(position, 4, 0);
        $.weight_zp = reader.int32_(position, 6, 0);
        return $;
    }

    static decodeText(reader, json) {
        const $ = new tosa.FullyConnectedAttribute();
        $.input_zp = reader.value(json.input_zp, 0);
        $.weight_zp = reader.value(json.weight_zp, 0);
        return $;
    }
};

tosa.NegateAttribute = class NegateAttribute {

    static decode(reader, position) {
        const $ = new tosa.NegateAttribute();
        $.input1_zp = reader.int32_(position, 4, 0);
        $.output_zp = reader.int32_(position, 6, 0);
        return $;
    }

    static decodeText(reader, json) {
        const $ = new tosa.NegateAttribute();
        $.input1_zp = reader.value(json.input1_zp, 0);
        $.output_zp = reader.value(json.output_zp, 0);
        return $;
    }
};

tosa.CustomAttribute = class CustomAttribute {

    static decode(reader, position) {
        const $ = new tosa.CustomAttribute();
        $.operator_name = reader.string_(position, 4, null);
        $.domain_name = reader.string_(position, 6, null);
        $.implementation_attrs = reader.array(position, 8, Uint8Array);
        return $;
    }

    static decodeText(reader, json) {
        const $ = new tosa.CustomAttribute();
        $.operator_name = reader.value(json.operator_name, null);
        $.domain_name = reader.value(json.domain_name, null);
        $.implementation_attrs = reader.array(json.implementation_attrs, Uint8Array);
        return $;
    }
};

tosa.FFTAttribute = class FFTAttribute {

    static decode(reader, position) {
        const $ = new tosa.FFTAttribute();
        $.inverse = reader.bool_(position, 4, false);
        $.local_bound = reader.bool_(position, 6, false);
        return $;
    }

    static decodeText(reader, json) {
        const $ = new tosa.FFTAttribute();
        $.inverse = reader.value(json.inverse, false);
        $.local_bound = reader.value(json.local_bound, false);
        return $;
    }
};

tosa.RFFTAttribute = class RFFTAttribute {

    static decode(reader, position) {
        const $ = new tosa.RFFTAttribute();
        $.local_bound = reader.bool_(position, 4, false);
        return $;
    }

    static decodeText(reader, json) {
        const $ = new tosa.RFFTAttribute();
        $.local_bound = reader.value(json.local_bound, false);
        return $;
    }
};

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

tosa.TosaTensor = class TosaTensor {

    static decode(reader, position) {
        const $ = new tosa.TosaTensor();
        $.name = reader.string_(position, 4, null);
        $.shape = reader.array(position, 6, Int32Array);
        $.type = reader.uint32_(position, 8, 0);
        $.data = reader.array(position, 10, Uint8Array);
        $.variable = reader.bool_(position, 12, false);
        $.is_unranked = reader.bool_(position, 14, false);
        $.variable_name = reader.string_(position, 16, null);
        return $;
    }

    static decodeText(reader, json) {
        const $ = new tosa.TosaTensor();
        $.name = reader.value(json.name, null);
        $.shape = reader.array(json.shape, Int32Array);
        $.type = tosa.DType[json.type];
        $.data = reader.array(json.data, Uint8Array);
        $.variable = reader.value(json.variable, false);
        $.is_unranked = reader.value(json.is_unranked, false);
        $.variable_name = reader.value(json.variable_name, null);
        return $;
    }
};

tosa.TosaOperator = class TosaOperator {

    static decode(reader, position) {
        const $ = new tosa.TosaOperator();
        $.op = reader.uint32_(position, 4, 0);
        $.attribute = reader.union(position, 6, tosa.Attribute);
        $.inputs = reader.strings_(position, 10);
        $.outputs = reader.strings_(position, 12);
        return $;
    }

    static decodeText(reader, json) {
        const $ = new tosa.TosaOperator();
        $.op = tosa.Op[json.op];
        $.attribute = tosa.Attribute.decodeText(reader, json.attribute, json.attribute_type);
        $.inputs = reader.array(json.inputs);
        $.outputs = reader.array(json.outputs);
        return $;
    }
};

tosa.TosaBasicBlock = class TosaBasicBlock {

    static decode(reader, position) {
        const $ = new tosa.TosaBasicBlock();
        $.name = reader.string_(position, 4, null);
        $.operators = reader.tables(position, 6, tosa.TosaOperator);
        $.tensors = reader.tables(position, 8, tosa.TosaTensor);
        $.inputs = reader.strings_(position, 10);
        $.outputs = reader.strings_(position, 12);
        return $;
    }

    static decodeText(reader, json) {
        const $ = new tosa.TosaBasicBlock();
        $.name = reader.value(json.name, null);
        $.operators = reader.objects(json.operators, tosa.TosaOperator);
        $.tensors = reader.objects(json.tensors, tosa.TosaTensor);
        $.inputs = reader.array(json.inputs);
        $.outputs = reader.array(json.outputs);
        return $;
    }
};

tosa.TosaRegion = class TosaRegion {

    static decode(reader, position) {
        const $ = new tosa.TosaRegion();
        $.name = reader.string_(position, 4, null);
        $.blocks = reader.tables(position, 6, tosa.TosaBasicBlock);
        return $;
    }

    static decodeText(reader, json) {
        const $ = new tosa.TosaRegion();
        $.name = reader.value(json.name, null);
        $.blocks = reader.objects(json.blocks, tosa.TosaBasicBlock);
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
        $.regions = reader.tables(position, 6, tosa.TosaRegion);
        return $;
    }

    static decodeText(reader, json) {
        const $ = new tosa.TosaGraph();
        $.version = reader.object(json.version, tosa.Version);
        $.regions = reader.objects(json.regions, tosa.TosaRegion);
        return $;
    }
};
