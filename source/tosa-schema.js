
export const tosa = {};

tosa.v0 = {};

tosa.v0.DType = {
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

tosa.v0.ResizeMode = {
    UNKNOWN: 0,
    NEAREST: 1,
    BILINEAR: 2
};

tosa.v0.Op = {
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

tosa.v0.Attribute = class {

    static decode(reader, position, type) {
        switch (type) {
            case 1: return tosa.v0.PoolAttribute.decode(reader, position);
            case 2: return tosa.v0.ConvAttribute.decode(reader, position);
            case 3: return tosa.v0.TransposeConvAttribute.decode(reader, position);
            case 4: return tosa.v0.PadAttribute.decode(reader, position);
            case 5: return tosa.v0.AxisAttribute.decode(reader, position);
            case 6: return tosa.v0.ReshapeAttribute.decode(reader, position);
            case 7: return tosa.v0.SliceAttribute.decode(reader, position);
            case 8: return tosa.v0.TileAttribute.decode(reader, position);
            case 9: return tosa.v0.ResizeAttribute.decode(reader, position);
            case 10: return tosa.v0.ClampAttribute.decode(reader, position);
            case 11: return tosa.v0.RescaleAttribute.decode(reader, position);
            case 12: return tosa.v0.MulAttribute.decode(reader, position);
            case 13: return tosa.v0.ArithmeticRightShiftAttribute.decode(reader, position);
            case 14: return tosa.v0.CondIfAttribute.decode(reader, position);
            case 15: return tosa.v0.WhileLoopAttribute.decode(reader, position);
            case 16: return tosa.v0.TransposeAttribute.decode(reader, position);
            case 17: return tosa.v0.TableAttribute.decode(reader, position);
            case 18: return tosa.v0.MatMulAttribute.decode(reader, position);
            case 19: return tosa.v0.FullyConnectedAttribute.decode(reader, position);
            case 20: return tosa.v0.NegateAttribute.decode(reader, position);
            case 21: return tosa.v0.CustomAttribute.decode(reader, position);
            case 22: return tosa.v0.FFTAttribute.decode(reader, position);
            case 23: return tosa.v0.RFFTAttribute.decode(reader, position);
            default: return undefined;
        }
    }

    static decodeText(reader, json, type) {
        switch (type) {
            case 'PoolAttribute': return tosa.v0.PoolAttribute.decodeText(reader, json);
            case 'ConvAttribute': return tosa.v0.ConvAttribute.decodeText(reader, json);
            case 'TransposeConvAttribute': return tosa.v0.TransposeConvAttribute.decodeText(reader, json);
            case 'PadAttribute': return tosa.v0.PadAttribute.decodeText(reader, json);
            case 'AxisAttribute': return tosa.v0.AxisAttribute.decodeText(reader, json);
            case 'ReshapeAttribute': return tosa.v0.ReshapeAttribute.decodeText(reader, json);
            case 'SliceAttribute': return tosa.v0.SliceAttribute.decodeText(reader, json);
            case 'TileAttribute': return tosa.v0.TileAttribute.decodeText(reader, json);
            case 'ResizeAttribute': return tosa.v0.ResizeAttribute.decodeText(reader, json);
            case 'ClampAttribute': return tosa.v0.ClampAttribute.decodeText(reader, json);
            case 'RescaleAttribute': return tosa.v0.RescaleAttribute.decodeText(reader, json);
            case 'MulAttribute': return tosa.v0.MulAttribute.decodeText(reader, json);
            case 'ArithmeticRightShiftAttribute': return tosa.v0.ArithmeticRightShiftAttribute.decodeText(reader, json);
            case 'CondIfAttribute': return tosa.v0.CondIfAttribute.decodeText(reader, json);
            case 'WhileLoopAttribute': return tosa.v0.WhileLoopAttribute.decodeText(reader, json);
            case 'TransposeAttribute': return tosa.v0.TransposeAttribute.decodeText(reader, json);
            case 'TableAttribute': return tosa.v0.TableAttribute.decodeText(reader, json);
            case 'MatMulAttribute': return tosa.v0.MatMulAttribute.decodeText(reader, json);
            case 'FullyConnectedAttribute': return tosa.v0.FullyConnectedAttribute.decodeText(reader, json);
            case 'NegateAttribute': return tosa.v0.NegateAttribute.decodeText(reader, json);
            case 'CustomAttribute': return tosa.v0.CustomAttribute.decodeText(reader, json);
            case 'FFTAttribute': return tosa.v0.FFTAttribute.decodeText(reader, json);
            case 'RFFTAttribute': return tosa.v0.RFFTAttribute.decodeText(reader, json);
            default: return undefined;
        }
    }
};

tosa.v0.PoolAttribute = class PoolAttribute {

    static decode(reader, position) {
        const $ = new tosa.v0.PoolAttribute();
        $.pad = reader.array(position, 4, Int32Array);
        $.kernel = reader.array(position, 6, Int32Array);
        $.stride = reader.array(position, 8, Int32Array);
        $.input_zp = reader.int32_(position, 10, 0);
        $.output_zp = reader.int32_(position, 12, 0);
        $.accum_dtype = reader.uint32_(position, 14, 0);
        return $;
    }

    static decodeText(reader, json) {
        const $ = new tosa.v0.PoolAttribute();
        $.pad = reader.array(json.pad, Int32Array);
        $.kernel = reader.array(json.kernel, Int32Array);
        $.stride = reader.array(json.stride, Int32Array);
        $.input_zp = reader.value(json.input_zp, 0);
        $.output_zp = reader.value(json.output_zp, 0);
        $.accum_dtype = tosa.v0.DType[json.accum_dtype];
        return $;
    }
};

tosa.v0.ConvAttribute = class ConvAttribute {

    static decode(reader, position) {
        const $ = new tosa.v0.ConvAttribute();
        $.pad = reader.array(position, 4, Int32Array);
        $.stride = reader.array(position, 6, Int32Array);
        $.dilation = reader.array(position, 8, Int32Array);
        $.input_zp = reader.int32_(position, 10, 0);
        $.weight_zp = reader.int32_(position, 12, 0);
        $.local_bound = reader.bool_(position, 14, false);
        return $;
    }

    static decodeText(reader, json) {
        const $ = new tosa.v0.ConvAttribute();
        $.pad = reader.array(json.pad, Int32Array);
        $.stride = reader.array(json.stride, Int32Array);
        $.dilation = reader.array(json.dilation, Int32Array);
        $.input_zp = reader.value(json.input_zp, 0);
        $.weight_zp = reader.value(json.weight_zp, 0);
        $.local_bound = reader.value(json.local_bound, false);
        return $;
    }
};

tosa.v0.TransposeConvAttribute = class TransposeConvAttribute {

    static decode(reader, position) {
        const $ = new tosa.v0.TransposeConvAttribute();
        $.out_pad = reader.array(position, 4, Int32Array);
        $.stride = reader.array(position, 6, Int32Array);
        $.output_shape = reader.array(position, 8, Int32Array);
        $.input_zp = reader.int32_(position, 10, 0);
        $.weight_zp = reader.int32_(position, 12, 0);
        $.local_bound = reader.bool_(position, 14, false);
        return $;
    }

    static decodeText(reader, json) {
        const $ = new tosa.v0.TransposeConvAttribute();
        $.out_pad = reader.array(json.out_pad, Int32Array);
        $.stride = reader.array(json.stride, Int32Array);
        $.output_shape = reader.array(json.output_shape, Int32Array);
        $.input_zp = reader.value(json.input_zp, 0);
        $.weight_zp = reader.value(json.weight_zp, 0);
        $.local_bound = reader.value(json.local_bound, false);
        return $;
    }
};

tosa.v0.PadAttribute = class PadAttribute {

    static decode(reader, position) {
        const $ = new tosa.v0.PadAttribute();
        $.padding = reader.array(position, 4, Int32Array);
        $.pad_const_int = reader.int32_(position, 6, 0);
        $.pad_const_fp = reader.array(position, 8, Uint8Array);
        return $;
    }

    static decodeText(reader, json) {
        const $ = new tosa.v0.PadAttribute();
        $.padding = reader.array(json.padding, Int32Array);
        $.pad_const_int = reader.value(json.pad_const_int, 0);
        $.pad_const_fp = reader.array(json.pad_const_fp, Uint8Array);
        return $;
    }
};

tosa.v0.AxisAttribute = class AxisAttribute {

    static decode(reader, position) {
        const $ = new tosa.v0.AxisAttribute();
        $.axis = reader.int32_(position, 4, 0);
        return $;
    }

    static decodeText(reader, json) {
        const $ = new tosa.v0.AxisAttribute();
        $.axis = reader.value(json.axis, 0);
        return $;
    }
};

tosa.v0.ReshapeAttribute = class ReshapeAttribute {

    static decode(reader, position) {
        const $ = new tosa.v0.ReshapeAttribute();
        $.new_shape = reader.array(position, 4, Int32Array);
        return $;
    }

    static decodeText(reader, json) {
        const $ = new tosa.v0.ReshapeAttribute();
        $.new_shape = reader.array(json.new_shape, Int32Array);
        return $;
    }
};

tosa.v0.SliceAttribute = class SliceAttribute {

    static decode(reader, position) {
        const $ = new tosa.v0.SliceAttribute();
        $.start = reader.array(position, 4, Int32Array);
        $.size = reader.array(position, 6, Int32Array);
        return $;
    }

    static decodeText(reader, json) {
        const $ = new tosa.v0.SliceAttribute();
        $.start = reader.array(json.start, Int32Array);
        $.size = reader.array(json.size, Int32Array);
        return $;
    }
};

tosa.v0.TileAttribute = class TileAttribute {

    static decode(reader, position) {
        const $ = new tosa.v0.TileAttribute();
        $.multiples = reader.array(position, 4, Int32Array);
        return $;
    }

    static decodeText(reader, json) {
        const $ = new tosa.v0.TileAttribute();
        $.multiples = reader.array(json.multiples, Int32Array);
        return $;
    }
};

tosa.v0.ResizeAttribute = class ResizeAttribute {

    static decode(reader, position) {
        const $ = new tosa.v0.ResizeAttribute();
        $.scale = reader.array(position, 4, Int16Array);
        $.offset = reader.array(position, 6, Int16Array);
        $.border = reader.array(position, 8, Int16Array);
        $.mode = reader.uint32_(position, 10, 0);
        return $;
    }

    static decodeText(reader, json) {
        const $ = new tosa.v0.ResizeAttribute();
        $.scale = reader.array(json.scale, Int16Array);
        $.offset = reader.array(json.offset, Int16Array);
        $.border = reader.array(json.border, Int16Array);
        $.mode = tosa.v0.ResizeMode[json.mode];
        return $;
    }
};

tosa.v0.ClampAttribute = class ClampAttribute {

    static decode(reader, position) {
        const $ = new tosa.v0.ClampAttribute();
        $.min_int = reader.int32_(position, 4, 0);
        $.max_int = reader.int32_(position, 6, 0);
        $.min_fp = reader.array(position, 8, Uint8Array);
        $.max_fp = reader.array(position, 10, Uint8Array);
        return $;
    }

    static decodeText(reader, json) {
        const $ = new tosa.v0.ClampAttribute();
        $.min_int = reader.value(json.min_int, 0);
        $.max_int = reader.value(json.max_int, 0);
        $.min_fp = reader.array(json.min_fp, Uint8Array);
        $.max_fp = reader.array(json.max_fp, Uint8Array);
        return $;
    }
};

tosa.v0.RescaleAttribute = class RescaleAttribute {

    static decode(reader, position) {
        const $ = new tosa.v0.RescaleAttribute();
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
        const $ = new tosa.v0.RescaleAttribute();
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

tosa.v0.MulAttribute = class MulAttribute {

    static decode(reader, position) {
        const $ = new tosa.v0.MulAttribute();
        $.shift = reader.int32_(position, 4, 0);
        return $;
    }

    static decodeText(reader, json) {
        const $ = new tosa.v0.MulAttribute();
        $.shift = reader.value(json.shift, 0);
        return $;
    }
};

tosa.v0.ArithmeticRightShiftAttribute = class ArithmeticRightShiftAttribute {

    static decode(reader, position) {
        const $ = new tosa.v0.ArithmeticRightShiftAttribute();
        $.round = reader.bool_(position, 4, false);
        return $;
    }

    static decodeText(reader, json) {
        const $ = new tosa.v0.ArithmeticRightShiftAttribute();
        $.round = reader.value(json.round, false);
        return $;
    }
};

tosa.v0.CondIfAttribute = class CondIfAttribute {

    static decode(reader, position) {
        const $ = new tosa.v0.CondIfAttribute();
        $.then_branch = reader.string_(position, 4, null);
        $.else_branch = reader.string_(position, 6, null);
        return $;
    }

    static decodeText(reader, json) {
        const $ = new tosa.v0.CondIfAttribute();
        $.then_branch = reader.value(json.then_branch, null);
        $.else_branch = reader.value(json.else_branch, null);
        return $;
    }
};

tosa.v0.WhileLoopAttribute = class WhileLoopAttribute {

    static decode(reader, position) {
        const $ = new tosa.v0.WhileLoopAttribute();
        $.cond_branch = reader.string_(position, 4, null);
        $.body_branch = reader.string_(position, 6, null);
        return $;
    }

    static decodeText(reader, json) {
        const $ = new tosa.v0.WhileLoopAttribute();
        $.cond_branch = reader.value(json.cond_branch, null);
        $.body_branch = reader.value(json.body_branch, null);
        return $;
    }
};

tosa.v0.TransposeAttribute = class TransposeAttribute {

    static decode(reader, position) {
        const $ = new tosa.v0.TransposeAttribute();
        $.perms = reader.array(position, 4, Int32Array);
        return $;
    }

    static decodeText(reader, json) {
        const $ = new tosa.v0.TransposeAttribute();
        $.perms = reader.array(json.perms, Int32Array);
        return $;
    }
};

tosa.v0.TableAttribute = class TableAttribute {

    static decode(reader, position) {
        const $ = new tosa.v0.TableAttribute();
        $.table = reader.array(position, 4, Int16Array);
        return $;
    }

    static decodeText(reader, json) {
        const $ = new tosa.v0.TableAttribute();
        $.table = reader.array(json.table, Int16Array);
        return $;
    }
};

tosa.v0.MatMulAttribute = class MatMulAttribute {

    static decode(reader, position) {
        const $ = new tosa.v0.MatMulAttribute();
        $.a_zp = reader.int32_(position, 4, 0);
        $.b_zp = reader.int32_(position, 6, 0);
        return $;
    }

    static decodeText(reader, json) {
        const $ = new tosa.v0.MatMulAttribute();
        $.a_zp = reader.value(json.a_zp, 0);
        $.b_zp = reader.value(json.b_zp, 0);
        return $;
    }
};

tosa.v0.FullyConnectedAttribute = class FullyConnectedAttribute {

    static decode(reader, position) {
        const $ = new tosa.v0.FullyConnectedAttribute();
        $.input_zp = reader.int32_(position, 4, 0);
        $.weight_zp = reader.int32_(position, 6, 0);
        return $;
    }

    static decodeText(reader, json) {
        const $ = new tosa.v0.FullyConnectedAttribute();
        $.input_zp = reader.value(json.input_zp, 0);
        $.weight_zp = reader.value(json.weight_zp, 0);
        return $;
    }
};

tosa.v0.NegateAttribute = class NegateAttribute {

    static decode(reader, position) {
        const $ = new tosa.v0.NegateAttribute();
        $.input1_zp = reader.int32_(position, 4, 0);
        $.output_zp = reader.int32_(position, 6, 0);
        return $;
    }

    static decodeText(reader, json) {
        const $ = new tosa.v0.NegateAttribute();
        $.input1_zp = reader.value(json.input1_zp, 0);
        $.output_zp = reader.value(json.output_zp, 0);
        return $;
    }
};

tosa.v0.CustomAttribute = class CustomAttribute {

    static decode(reader, position) {
        const $ = new tosa.v0.CustomAttribute();
        $.operator_name = reader.string_(position, 4, null);
        $.domain_name = reader.string_(position, 6, null);
        $.implementation_attrs = reader.array(position, 8, Uint8Array);
        return $;
    }

    static decodeText(reader, json) {
        const $ = new tosa.v0.CustomAttribute();
        $.operator_name = reader.value(json.operator_name, null);
        $.domain_name = reader.value(json.domain_name, null);
        $.implementation_attrs = reader.array(json.implementation_attrs, Uint8Array);
        return $;
    }
};

tosa.v0.FFTAttribute = class FFTAttribute {

    static decode(reader, position) {
        const $ = new tosa.v0.FFTAttribute();
        $.inverse = reader.bool_(position, 4, false);
        $.local_bound = reader.bool_(position, 6, false);
        return $;
    }

    static decodeText(reader, json) {
        const $ = new tosa.v0.FFTAttribute();
        $.inverse = reader.value(json.inverse, false);
        $.local_bound = reader.value(json.local_bound, false);
        return $;
    }
};

tosa.v0.RFFTAttribute = class RFFTAttribute {

    static decode(reader, position) {
        const $ = new tosa.v0.RFFTAttribute();
        $.local_bound = reader.bool_(position, 4, false);
        return $;
    }

    static decodeText(reader, json) {
        const $ = new tosa.v0.RFFTAttribute();
        $.local_bound = reader.value(json.local_bound, false);
        return $;
    }
};

tosa.v0.Version = class Version {

    static decode(reader, position) {
        const $ = new tosa.v0.Version();
        $._major = reader.int32_(position, 4, -1);
        $._minor = reader.int32_(position, 6, -1);
        $._patch = reader.int32_(position, 8, -1);
        $._draft = reader.bool_(position, 10, true);
        return $;
    }

    static decodeText(reader, json) {
        const $ = new tosa.v0.Version();
        $._major = reader.value(json._major, -1);
        $._minor = reader.value(json._minor, -1);
        $._patch = reader.value(json._patch, -1);
        $._draft = reader.value(json._draft, true);
        return $;
    }
};

tosa.v0.TosaTensor = class TosaTensor {

    static decode(reader, position) {
        const $ = new tosa.v0.TosaTensor();
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
        const $ = new tosa.v0.TosaTensor();
        $.name = reader.value(json.name, null);
        $.shape = reader.array(json.shape, Int32Array);
        $.type = tosa.v0.DType[json.type];
        $.data = reader.array(json.data, Uint8Array);
        $.variable = reader.value(json.variable, false);
        $.is_unranked = reader.value(json.is_unranked, false);
        $.variable_name = reader.value(json.variable_name, null);
        return $;
    }
};

tosa.v0.TosaOperator = class TosaOperator {

    static decode(reader, position) {
        const $ = new tosa.v0.TosaOperator();
        $.op = reader.uint32_(position, 4, 0);
        $.attribute = reader.union(position, 6, tosa.v0.Attribute);
        $.inputs = reader.strings_(position, 10);
        $.outputs = reader.strings_(position, 12);
        return $;
    }

    static decodeText(reader, json) {
        const $ = new tosa.v0.TosaOperator();
        $.op = tosa.v0.Op[json.op];
        $.attribute = tosa.v0.Attribute.decodeText(reader, json.attribute, json.attribute_type);
        $.inputs = reader.array(json.inputs);
        $.outputs = reader.array(json.outputs);
        return $;
    }
};

tosa.v0.TosaBasicBlock = class TosaBasicBlock {

    static decode(reader, position) {
        const $ = new tosa.v0.TosaBasicBlock();
        $.name = reader.string_(position, 4, null);
        $.operators = reader.tables(position, 6, tosa.v0.TosaOperator);
        $.tensors = reader.tables(position, 8, tosa.v0.TosaTensor);
        $.inputs = reader.strings_(position, 10);
        $.outputs = reader.strings_(position, 12);
        return $;
    }

    static decodeText(reader, json) {
        const $ = new tosa.v0.TosaBasicBlock();
        $.name = reader.value(json.name, null);
        $.operators = reader.objects(json.operators, tosa.v0.TosaOperator);
        $.tensors = reader.objects(json.tensors, tosa.v0.TosaTensor);
        $.inputs = reader.array(json.inputs);
        $.outputs = reader.array(json.outputs);
        return $;
    }
};

tosa.v0.TosaRegion = class TosaRegion {

    static decode(reader, position) {
        const $ = new tosa.v0.TosaRegion();
        $.name = reader.string_(position, 4, null);
        $.blocks = reader.tables(position, 6, tosa.v0.TosaBasicBlock);
        return $;
    }

    static decodeText(reader, json) {
        const $ = new tosa.v0.TosaRegion();
        $.name = reader.value(json.name, null);
        $.blocks = reader.objects(json.blocks, tosa.v0.TosaBasicBlock);
        return $;
    }
};

tosa.v0.TosaGraph = class TosaGraph {

    static identifier(reader) {
        return reader.identifier === 'TOSA';
    }

    static create(reader) {
        return tosa.v0.TosaGraph.decode(reader, reader.root);
    }

    static createText(reader) {
        return tosa.v0.TosaGraph.decodeText(reader, reader.root);
    }

    static decode(reader, position) {
        const $ = new tosa.v0.TosaGraph();
        $.version = reader.table(position, 4, tosa.v0.Version);
        $.regions = reader.tables(position, 6, tosa.v0.TosaRegion);
        return $;
    }

    static decodeText(reader, json) {
        const $ = new tosa.v0.TosaGraph();
        $.version = reader.object(json.version, tosa.v0.Version);
        $.regions = reader.objects(json.regions, tosa.v0.TosaRegion);
        return $;
    }
};

tosa.v1 = {};

tosa.v1.DType = {
    UNKNOWN: 0,
    BOOL: 1,
    INT4: 2,
    INT8: 3,
    INT16: 4,
    INT32: 5,
    INT48: 6,
    FP32: 7,
    FP16: 8,
    BF16: 9,
    SHAPE: 10,
    FP8E4M3: 11,
    FP8E5M2: 12
};

tosa.v1.ResizeMode = {
    UNKNOWN: 0,
    NEAREST: 1,
    BILINEAR: 2
};

tosa.v1.NanPropagationMode = {
    UNKNOWN: 0,
    PROPAGATE: 1,
    IGNORE: 2
};

tosa.v1.RoundingMode = {
    UNKNOWN: 0,
    SINGLE_ROUND: 1,
    INEXACT_ROUND: 2,
    DOUBLE_ROUND: 3
};

tosa.v1.Op = {
    UNKNOWN: 0,
    ARGMAX: 1,
    AVG_POOL2D: 2,
    CONV2D: 3,
    CONV3D: 4,
    DEPTHWISE_CONV2D: 5,
    FFT2D: 6,
    MATMUL: 7,
    MAX_POOL2D: 8,
    RFFT2D: 9,
    TRANSPOSE_CONV2D: 10,
    CLAMP: 11,
    ERF: 12,
    SIGMOID: 13,
    TANH: 14,
    ADD: 15,
    ARITHMETIC_RIGHT_SHIFT: 16,
    BITWISE_AND: 17,
    BITWISE_OR: 18,
    BITWISE_XOR: 19,
    INTDIV: 20,
    LOGICAL_AND: 21,
    LOGICAL_LEFT_SHIFT: 22,
    LOGICAL_RIGHT_SHIFT: 23,
    LOGICAL_OR: 24,
    LOGICAL_XOR: 25,
    MAXIMUM: 26,
    MINIMUM: 27,
    MUL: 28,
    POW: 29,
    SUB: 30,
    TABLE: 31,
    ABS: 32,
    BITWISE_NOT: 33,
    CEIL: 34,
    CLZ: 35,
    COS: 36,
    EXP: 37,
    FLOOR: 38,
    LOG: 39,
    LOGICAL_NOT: 40,
    NEGATE: 41,
    RECIPROCAL: 42,
    RSQRT: 43,
    SIN: 44,
    SELECT: 45,
    EQUAL: 46,
    GREATER: 47,
    GREATER_EQUAL: 48,
    REDUCE_ALL: 49,
    REDUCE_ANY: 50,
    REDUCE_MAX: 51,
    REDUCE_MIN: 52,
    REDUCE_PRODUCT: 53,
    REDUCE_SUM: 54,
    CONCAT: 55,
    PAD: 56,
    RESHAPE: 57,
    REVERSE: 58,
    SLICE: 59,
    TILE: 60,
    TRANSPOSE: 61,
    GATHER: 62,
    SCATTER: 63,
    RESIZE: 64,
    CAST: 65,
    RESCALE: 66,
    CONST: 67,
    IDENTITY: 68,
    CUSTOM: 69,
    COND_IF: 70,
    WHILE_LOOP: 71,
    VARIABLE: 72,
    VARIABLE_WRITE: 73,
    VARIABLE_READ: 74,
    CONST_SHAPE: 75
};

tosa.v1.Attribute = class {

    static decode(reader, position, type) {
        switch (type) {
            case 1: return tosa.v1.ArgMaxAttribute.decode(reader, position);
            case 2: return tosa.v1.AvgPool2dAttribute.decode(reader, position);
            case 3: return tosa.v1.Conv2dAttribute.decode(reader, position);
            case 4: return tosa.v1.Conv3dAttribute.decode(reader, position);
            case 5: return tosa.v1.DepthwiseConv2dAttribute.decode(reader, position);
            case 6: return tosa.v1.FFT2dAttribute.decode(reader, position);
            case 7: return tosa.v1.MatMulAttribute.decode(reader, position);
            case 8: return tosa.v1.MaxPool2dAttribute.decode(reader, position);
            case 9: return tosa.v1.RFFT2dAttribute.decode(reader, position);
            case 10: return tosa.v1.TransposeConv2dAttribute.decode(reader, position);
            case 11: return tosa.v1.ClampAttribute.decode(reader, position);
            case 12: return tosa.v1.ErfAttribute.decode(reader, position);
            case 13: return tosa.v1.SigmoidAttribute.decode(reader, position);
            case 14: return tosa.v1.TanhAttribute.decode(reader, position);
            case 15: return tosa.v1.AddAttribute.decode(reader, position);
            case 16: return tosa.v1.ArithmeticRightShiftAttribute.decode(reader, position);
            case 17: return tosa.v1.BitwiseAndAttribute.decode(reader, position);
            case 18: return tosa.v1.BitwiseOrAttribute.decode(reader, position);
            case 19: return tosa.v1.BitwiseXorAttribute.decode(reader, position);
            case 20: return tosa.v1.IntDivAttribute.decode(reader, position);
            case 21: return tosa.v1.LogicalAndAttribute.decode(reader, position);
            case 22: return tosa.v1.LogicalLeftShiftAttribute.decode(reader, position);
            case 23: return tosa.v1.LogicalRightShiftAttribute.decode(reader, position);
            case 24: return tosa.v1.LogicalOrAttribute.decode(reader, position);
            case 25: return tosa.v1.LogicalXorAttribute.decode(reader, position);
            case 26: return tosa.v1.MaximumAttribute.decode(reader, position);
            case 27: return tosa.v1.MinimumAttribute.decode(reader, position);
            case 28: return tosa.v1.MulAttribute.decode(reader, position);
            case 29: return tosa.v1.PowAttribute.decode(reader, position);
            case 30: return tosa.v1.SubAttribute.decode(reader, position);
            case 31: return tosa.v1.TableAttribute.decode(reader, position);
            case 32: return tosa.v1.AbsAttribute.decode(reader, position);
            case 33: return tosa.v1.BitwiseNotAttribute.decode(reader, position);
            case 34: return tosa.v1.CeilAttribute.decode(reader, position);
            case 35: return tosa.v1.ClzAttribute.decode(reader, position);
            case 36: return tosa.v1.CosAttribute.decode(reader, position);
            case 37: return tosa.v1.ExpAttribute.decode(reader, position);
            case 38: return tosa.v1.FloorAttribute.decode(reader, position);
            case 39: return tosa.v1.LogAttribute.decode(reader, position);
            case 40: return tosa.v1.LogicalNotAttribute.decode(reader, position);
            case 41: return tosa.v1.NegateAttribute.decode(reader, position);
            case 42: return tosa.v1.ReciprocalAttribute.decode(reader, position);
            case 43: return tosa.v1.RsqrtAttribute.decode(reader, position);
            case 44: return tosa.v1.SinAttribute.decode(reader, position);
            case 45: return tosa.v1.SelectAttribute.decode(reader, position);
            case 46: return tosa.v1.EqualAttribute.decode(reader, position);
            case 47: return tosa.v1.GreaterAttribute.decode(reader, position);
            case 48: return tosa.v1.GreaterEqualAttribute.decode(reader, position);
            case 49: return tosa.v1.ReduceAllAttribute.decode(reader, position);
            case 50: return tosa.v1.ReduceAnyAttribute.decode(reader, position);
            case 51: return tosa.v1.ReduceMaxAttribute.decode(reader, position);
            case 52: return tosa.v1.ReduceMinAttribute.decode(reader, position);
            case 53: return tosa.v1.ReduceProductAttribute.decode(reader, position);
            case 54: return tosa.v1.ReduceSumAttribute.decode(reader, position);
            case 55: return tosa.v1.ConcatAttribute.decode(reader, position);
            case 56: return tosa.v1.PadAttribute.decode(reader, position);
            case 57: return tosa.v1.ReshapeAttribute.decode(reader, position);
            case 58: return tosa.v1.ReverseAttribute.decode(reader, position);
            case 59: return tosa.v1.SliceAttribute.decode(reader, position);
            case 60: return tosa.v1.TileAttribute.decode(reader, position);
            case 61: return tosa.v1.TransposeAttribute.decode(reader, position);
            case 62: return tosa.v1.GatherAttribute.decode(reader, position);
            case 63: return tosa.v1.ScatterAttribute.decode(reader, position);
            case 64: return tosa.v1.ResizeAttribute.decode(reader, position);
            case 65: return tosa.v1.CastAttribute.decode(reader, position);
            case 66: return tosa.v1.RescaleAttribute.decode(reader, position);
            case 67: return tosa.v1.ConstAttribute.decode(reader, position);
            case 68: return tosa.v1.IdentityAttribute.decode(reader, position);
            case 69: return tosa.v1.CustomAttribute.decode(reader, position);
            case 70: return tosa.v1.CondIfAttribute.decode(reader, position);
            case 71: return tosa.v1.WhileLoopAttribute.decode(reader, position);
            case 72: return tosa.v1.VariableAttribute.decode(reader, position);
            case 73: return tosa.v1.VariableWriteAttribute.decode(reader, position);
            case 74: return tosa.v1.VariableReadAttribute.decode(reader, position);
            case 75: return tosa.v1.ConstShapeAttribute.decode(reader, position);
            default: return undefined;
        }
    }

    static decodeText(reader, json, type) {
        switch (type) {
            case 'ArgMaxAttribute': return tosa.v1.ArgMaxAttribute.decodeText(reader, json);
            case 'AvgPool2dAttribute': return tosa.v1.AvgPool2dAttribute.decodeText(reader, json);
            case 'Conv2dAttribute': return tosa.v1.Conv2dAttribute.decodeText(reader, json);
            case 'Conv3dAttribute': return tosa.v1.Conv3dAttribute.decodeText(reader, json);
            case 'DepthwiseConv2dAttribute': return tosa.v1.DepthwiseConv2dAttribute.decodeText(reader, json);
            case 'FFT2dAttribute': return tosa.v1.FFT2dAttribute.decodeText(reader, json);
            case 'MatMulAttribute': return tosa.v1.MatMulAttribute.decodeText(reader, json);
            case 'MaxPool2dAttribute': return tosa.v1.MaxPool2dAttribute.decodeText(reader, json);
            case 'RFFT2dAttribute': return tosa.v1.RFFT2dAttribute.decodeText(reader, json);
            case 'TransposeConv2dAttribute': return tosa.v1.TransposeConv2dAttribute.decodeText(reader, json);
            case 'ClampAttribute': return tosa.v1.ClampAttribute.decodeText(reader, json);
            case 'ErfAttribute': return tosa.v1.ErfAttribute.decodeText(reader, json);
            case 'SigmoidAttribute': return tosa.v1.SigmoidAttribute.decodeText(reader, json);
            case 'TanhAttribute': return tosa.v1.TanhAttribute.decodeText(reader, json);
            case 'AddAttribute': return tosa.v1.AddAttribute.decodeText(reader, json);
            case 'ArithmeticRightShiftAttribute': return tosa.v1.ArithmeticRightShiftAttribute.decodeText(reader, json);
            case 'BitwiseAndAttribute': return tosa.v1.BitwiseAndAttribute.decodeText(reader, json);
            case 'BitwiseOrAttribute': return tosa.v1.BitwiseOrAttribute.decodeText(reader, json);
            case 'BitwiseXorAttribute': return tosa.v1.BitwiseXorAttribute.decodeText(reader, json);
            case 'IntDivAttribute': return tosa.v1.IntDivAttribute.decodeText(reader, json);
            case 'LogicalAndAttribute': return tosa.v1.LogicalAndAttribute.decodeText(reader, json);
            case 'LogicalLeftShiftAttribute': return tosa.v1.LogicalLeftShiftAttribute.decodeText(reader, json);
            case 'LogicalRightShiftAttribute': return tosa.v1.LogicalRightShiftAttribute.decodeText(reader, json);
            case 'LogicalOrAttribute': return tosa.v1.LogicalOrAttribute.decodeText(reader, json);
            case 'LogicalXorAttribute': return tosa.v1.LogicalXorAttribute.decodeText(reader, json);
            case 'MaximumAttribute': return tosa.v1.MaximumAttribute.decodeText(reader, json);
            case 'MinimumAttribute': return tosa.v1.MinimumAttribute.decodeText(reader, json);
            case 'MulAttribute': return tosa.v1.MulAttribute.decodeText(reader, json);
            case 'PowAttribute': return tosa.v1.PowAttribute.decodeText(reader, json);
            case 'SubAttribute': return tosa.v1.SubAttribute.decodeText(reader, json);
            case 'TableAttribute': return tosa.v1.TableAttribute.decodeText(reader, json);
            case 'AbsAttribute': return tosa.v1.AbsAttribute.decodeText(reader, json);
            case 'BitwiseNotAttribute': return tosa.v1.BitwiseNotAttribute.decodeText(reader, json);
            case 'CeilAttribute': return tosa.v1.CeilAttribute.decodeText(reader, json);
            case 'ClzAttribute': return tosa.v1.ClzAttribute.decodeText(reader, json);
            case 'CosAttribute': return tosa.v1.CosAttribute.decodeText(reader, json);
            case 'ExpAttribute': return tosa.v1.ExpAttribute.decodeText(reader, json);
            case 'FloorAttribute': return tosa.v1.FloorAttribute.decodeText(reader, json);
            case 'LogAttribute': return tosa.v1.LogAttribute.decodeText(reader, json);
            case 'LogicalNotAttribute': return tosa.v1.LogicalNotAttribute.decodeText(reader, json);
            case 'NegateAttribute': return tosa.v1.NegateAttribute.decodeText(reader, json);
            case 'ReciprocalAttribute': return tosa.v1.ReciprocalAttribute.decodeText(reader, json);
            case 'RsqrtAttribute': return tosa.v1.RsqrtAttribute.decodeText(reader, json);
            case 'SinAttribute': return tosa.v1.SinAttribute.decodeText(reader, json);
            case 'SelectAttribute': return tosa.v1.SelectAttribute.decodeText(reader, json);
            case 'EqualAttribute': return tosa.v1.EqualAttribute.decodeText(reader, json);
            case 'GreaterAttribute': return tosa.v1.GreaterAttribute.decodeText(reader, json);
            case 'GreaterEqualAttribute': return tosa.v1.GreaterEqualAttribute.decodeText(reader, json);
            case 'ReduceAllAttribute': return tosa.v1.ReduceAllAttribute.decodeText(reader, json);
            case 'ReduceAnyAttribute': return tosa.v1.ReduceAnyAttribute.decodeText(reader, json);
            case 'ReduceMaxAttribute': return tosa.v1.ReduceMaxAttribute.decodeText(reader, json);
            case 'ReduceMinAttribute': return tosa.v1.ReduceMinAttribute.decodeText(reader, json);
            case 'ReduceProductAttribute': return tosa.v1.ReduceProductAttribute.decodeText(reader, json);
            case 'ReduceSumAttribute': return tosa.v1.ReduceSumAttribute.decodeText(reader, json);
            case 'ConcatAttribute': return tosa.v1.ConcatAttribute.decodeText(reader, json);
            case 'PadAttribute': return tosa.v1.PadAttribute.decodeText(reader, json);
            case 'ReshapeAttribute': return tosa.v1.ReshapeAttribute.decodeText(reader, json);
            case 'ReverseAttribute': return tosa.v1.ReverseAttribute.decodeText(reader, json);
            case 'SliceAttribute': return tosa.v1.SliceAttribute.decodeText(reader, json);
            case 'TileAttribute': return tosa.v1.TileAttribute.decodeText(reader, json);
            case 'TransposeAttribute': return tosa.v1.TransposeAttribute.decodeText(reader, json);
            case 'GatherAttribute': return tosa.v1.GatherAttribute.decodeText(reader, json);
            case 'ScatterAttribute': return tosa.v1.ScatterAttribute.decodeText(reader, json);
            case 'ResizeAttribute': return tosa.v1.ResizeAttribute.decodeText(reader, json);
            case 'CastAttribute': return tosa.v1.CastAttribute.decodeText(reader, json);
            case 'RescaleAttribute': return tosa.v1.RescaleAttribute.decodeText(reader, json);
            case 'ConstAttribute': return tosa.v1.ConstAttribute.decodeText(reader, json);
            case 'IdentityAttribute': return tosa.v1.IdentityAttribute.decodeText(reader, json);
            case 'CustomAttribute': return tosa.v1.CustomAttribute.decodeText(reader, json);
            case 'CondIfAttribute': return tosa.v1.CondIfAttribute.decodeText(reader, json);
            case 'WhileLoopAttribute': return tosa.v1.WhileLoopAttribute.decodeText(reader, json);
            case 'VariableAttribute': return tosa.v1.VariableAttribute.decodeText(reader, json);
            case 'VariableWriteAttribute': return tosa.v1.VariableWriteAttribute.decodeText(reader, json);
            case 'VariableReadAttribute': return tosa.v1.VariableReadAttribute.decodeText(reader, json);
            case 'ConstShapeAttribute': return tosa.v1.ConstShapeAttribute.decodeText(reader, json);
            default: return undefined;
        }
    }
};

tosa.v1.ArgMaxAttribute = class ArgMaxAttribute {

    static decode(reader, position) {
        const $ = new tosa.v1.ArgMaxAttribute();
        $.axis = reader.int32_(position, 4, 0);
        $.nan_mode = reader.uint32_(position, 6, 0);
        return $;
    }

    static decodeText(reader, json) {
        const $ = new tosa.v1.ArgMaxAttribute();
        $.axis = reader.value(json.axis, 0);
        $.nan_mode = tosa.v1.NanPropagationMode[json.nan_mode];
        return $;
    }
};

tosa.v1.AvgPool2dAttribute = class AvgPool2dAttribute {

    static decode(reader, position) {
        const $ = new tosa.v1.AvgPool2dAttribute();
        $.kernel = reader.array(position, 4, Int32Array);
        $.stride = reader.array(position, 6, Int32Array);
        $.pad = reader.array(position, 8, Int32Array);
        $.acc_type = reader.uint32_(position, 10, 0);
        return $;
    }

    static decodeText(reader, json) {
        const $ = new tosa.v1.AvgPool2dAttribute();
        $.kernel = reader.array(json.kernel, Int32Array);
        $.stride = reader.array(json.stride, Int32Array);
        $.pad = reader.array(json.pad, Int32Array);
        $.acc_type = tosa.v1.DType[json.acc_type];
        return $;
    }
};

tosa.v1.Conv2dAttribute = class Conv2dAttribute {

    static decode(reader, position) {
        const $ = new tosa.v1.Conv2dAttribute();
        $.pad = reader.array(position, 4, Int32Array);
        $.stride = reader.array(position, 6, Int32Array);
        $.dilation = reader.array(position, 8, Int32Array);
        $.local_bound = reader.bool_(position, 10, false);
        $.acc_type = reader.uint32_(position, 12, 0);
        return $;
    }

    static decodeText(reader, json) {
        const $ = new tosa.v1.Conv2dAttribute();
        $.pad = reader.array(json.pad, Int32Array);
        $.stride = reader.array(json.stride, Int32Array);
        $.dilation = reader.array(json.dilation, Int32Array);
        $.local_bound = reader.value(json.local_bound, false);
        $.acc_type = tosa.v1.DType[json.acc_type];
        return $;
    }
};

tosa.v1.Conv3dAttribute = class Conv3dAttribute {

    static decode(reader, position) {
        const $ = new tosa.v1.Conv3dAttribute();
        $.pad = reader.array(position, 4, Int32Array);
        $.stride = reader.array(position, 6, Int32Array);
        $.dilation = reader.array(position, 8, Int32Array);
        $.local_bound = reader.bool_(position, 10, false);
        $.acc_type = reader.uint32_(position, 12, 0);
        return $;
    }

    static decodeText(reader, json) {
        const $ = new tosa.v1.Conv3dAttribute();
        $.pad = reader.array(json.pad, Int32Array);
        $.stride = reader.array(json.stride, Int32Array);
        $.dilation = reader.array(json.dilation, Int32Array);
        $.local_bound = reader.value(json.local_bound, false);
        $.acc_type = tosa.v1.DType[json.acc_type];
        return $;
    }
};

tosa.v1.DepthwiseConv2dAttribute = class DepthwiseConv2dAttribute {

    static decode(reader, position) {
        const $ = new tosa.v1.DepthwiseConv2dAttribute();
        $.pad = reader.array(position, 4, Int32Array);
        $.stride = reader.array(position, 6, Int32Array);
        $.dilation = reader.array(position, 8, Int32Array);
        $.local_bound = reader.bool_(position, 10, false);
        $.acc_type = reader.uint32_(position, 12, 0);
        return $;
    }

    static decodeText(reader, json) {
        const $ = new tosa.v1.DepthwiseConv2dAttribute();
        $.pad = reader.array(json.pad, Int32Array);
        $.stride = reader.array(json.stride, Int32Array);
        $.dilation = reader.array(json.dilation, Int32Array);
        $.local_bound = reader.value(json.local_bound, false);
        $.acc_type = tosa.v1.DType[json.acc_type];
        return $;
    }
};

tosa.v1.FFT2dAttribute = class FFT2dAttribute {

    static decode(reader, position) {
        const $ = new tosa.v1.FFT2dAttribute();
        $.inverse = reader.bool_(position, 4, false);
        $.local_bound = reader.bool_(position, 6, false);
        return $;
    }

    static decodeText(reader, json) {
        const $ = new tosa.v1.FFT2dAttribute();
        $.inverse = reader.value(json.inverse, false);
        $.local_bound = reader.value(json.local_bound, false);
        return $;
    }
};

tosa.v1.MatMulAttribute = class MatMulAttribute {

    static decode(/* reader, position */) {
        const $ = new tosa.v1.MatMulAttribute();
        return $;
    }

    static decodeText(/* reader, json */) {
        const $ = new tosa.v1.MatMulAttribute();
        return $;
    }
};

tosa.v1.MaxPool2dAttribute = class MaxPool2dAttribute {

    static decode(reader, position) {
        const $ = new tosa.v1.MaxPool2dAttribute();
        $.kernel = reader.array(position, 4, Int32Array);
        $.stride = reader.array(position, 6, Int32Array);
        $.pad = reader.array(position, 8, Int32Array);
        $.nan_mode = reader.uint32_(position, 10, 0);
        return $;
    }

    static decodeText(reader, json) {
        const $ = new tosa.v1.MaxPool2dAttribute();
        $.kernel = reader.array(json.kernel, Int32Array);
        $.stride = reader.array(json.stride, Int32Array);
        $.pad = reader.array(json.pad, Int32Array);
        $.nan_mode = tosa.v1.NanPropagationMode[json.nan_mode];
        return $;
    }
};

tosa.v1.RFFT2dAttribute = class RFFT2dAttribute {

    static decode(reader, position) {
        const $ = new tosa.v1.RFFT2dAttribute();
        $.local_bound = reader.bool_(position, 4, false);
        return $;
    }

    static decodeText(reader, json) {
        const $ = new tosa.v1.RFFT2dAttribute();
        $.local_bound = reader.value(json.local_bound, false);
        return $;
    }
};

tosa.v1.TransposeConv2dAttribute = class TransposeConv2dAttribute {

    static decode(reader, position) {
        const $ = new tosa.v1.TransposeConv2dAttribute();
        $.out_pad = reader.array(position, 4, Int32Array);
        $.stride = reader.array(position, 6, Int32Array);
        $.local_bound = reader.bool_(position, 8, false);
        $.acc_type = reader.uint32_(position, 10, 0);
        return $;
    }

    static decodeText(reader, json) {
        const $ = new tosa.v1.TransposeConv2dAttribute();
        $.out_pad = reader.array(json.out_pad, Int32Array);
        $.stride = reader.array(json.stride, Int32Array);
        $.local_bound = reader.value(json.local_bound, false);
        $.acc_type = tosa.v1.DType[json.acc_type];
        return $;
    }
};

tosa.v1.ClampAttribute = class ClampAttribute {

    static decode(reader, position) {
        const $ = new tosa.v1.ClampAttribute();
        $.min_val = reader.array(position, 4, Uint8Array);
        $.max_val = reader.array(position, 6, Uint8Array);
        $.nan_mode = reader.uint32_(position, 8, 0);
        return $;
    }

    static decodeText(reader, json) {
        const $ = new tosa.v1.ClampAttribute();
        $.min_val = reader.array(json.min_val, Uint8Array);
        $.max_val = reader.array(json.max_val, Uint8Array);
        $.nan_mode = tosa.v1.NanPropagationMode[json.nan_mode];
        return $;
    }
};

tosa.v1.ErfAttribute = class ErfAttribute {

    static decode(/* reader, position */) {
        const $ = new tosa.v1.ErfAttribute();
        return $;
    }

    static decodeText(/* reader, json */) {
        const $ = new tosa.v1.ErfAttribute();
        return $;
    }
};

tosa.v1.SigmoidAttribute = class SigmoidAttribute {

    static decode(/* reader, position */) {
        const $ = new tosa.v1.SigmoidAttribute();
        return $;
    }

    static decodeText(/* reader, json */) {
        const $ = new tosa.v1.SigmoidAttribute();
        return $;
    }
};

tosa.v1.TanhAttribute = class TanhAttribute {

    static decode(/* reader, position */) {
        const $ = new tosa.v1.TanhAttribute();
        return $;
    }

    static decodeText(/* reader, json */) {
        const $ = new tosa.v1.TanhAttribute();
        return $;
    }
};

tosa.v1.AddAttribute = class AddAttribute {

    static decode(/* reader, position */) {
        const $ = new tosa.v1.AddAttribute();
        return $;
    }

    static decodeText(/* reader, json */) {
        const $ = new tosa.v1.AddAttribute();
        return $;
    }
};

tosa.v1.ArithmeticRightShiftAttribute = class ArithmeticRightShiftAttribute {

    static decode(reader, position) {
        const $ = new tosa.v1.ArithmeticRightShiftAttribute();
        $.round = reader.bool_(position, 4, false);
        return $;
    }

    static decodeText(reader, json) {
        const $ = new tosa.v1.ArithmeticRightShiftAttribute();
        $.round = reader.value(json.round, false);
        return $;
    }
};

tosa.v1.BitwiseAndAttribute = class BitwiseAndAttribute {

    static decode(/* reader, position */) {
        const $ = new tosa.v1.BitwiseAndAttribute();
        return $;
    }

    static decodeText(/* reader, json */) {
        const $ = new tosa.v1.BitwiseAndAttribute();
        return $;
    }
};

tosa.v1.BitwiseOrAttribute = class BitwiseOrAttribute {

    static decode(/* reader, position */) {
        const $ = new tosa.v1.BitwiseOrAttribute();
        return $;
    }

    static decodeText(/* reader, json */) {
        const $ = new tosa.v1.BitwiseOrAttribute();
        return $;
    }
};

tosa.v1.BitwiseXorAttribute = class BitwiseXorAttribute {

    static decode(/* reader, position */) {
        const $ = new tosa.v1.BitwiseXorAttribute();
        return $;
    }

    static decodeText(/* reader, json */) {
        const $ = new tosa.v1.BitwiseXorAttribute();
        return $;
    }
};

tosa.v1.IntDivAttribute = class IntDivAttribute {

    static decode(/* reader, position */) {
        const $ = new tosa.v1.IntDivAttribute();
        return $;
    }

    static decodeText(/* reader, json */) {
        const $ = new tosa.v1.IntDivAttribute();
        return $;
    }
};

tosa.v1.LogicalAndAttribute = class LogicalAndAttribute {

    static decode(/* reader, position */) {
        const $ = new tosa.v1.LogicalAndAttribute();
        return $;
    }

    static decodeText(/* reader, json */) {
        const $ = new tosa.v1.LogicalAndAttribute();
        return $;
    }
};

tosa.v1.LogicalLeftShiftAttribute = class LogicalLeftShiftAttribute {

    static decode(/* reader, position */) {
        const $ = new tosa.v1.LogicalLeftShiftAttribute();
        return $;
    }

    static decodeText(/* reader, json */) {
        const $ = new tosa.v1.LogicalLeftShiftAttribute();
        return $;
    }
};

tosa.v1.LogicalRightShiftAttribute = class LogicalRightShiftAttribute {

    static decode(/* reader, position */) {
        const $ = new tosa.v1.LogicalRightShiftAttribute();
        return $;
    }

    static decodeText(/* reader, json */) {
        const $ = new tosa.v1.LogicalRightShiftAttribute();
        return $;
    }
};

tosa.v1.LogicalOrAttribute = class LogicalOrAttribute {

    static decode(/* reader, position */) {
        const $ = new tosa.v1.LogicalOrAttribute();
        return $;
    }

    static decodeText(/* reader, json */) {
        const $ = new tosa.v1.LogicalOrAttribute();
        return $;
    }
};

tosa.v1.LogicalXorAttribute = class LogicalXorAttribute {

    static decode(/* reader, position */) {
        const $ = new tosa.v1.LogicalXorAttribute();
        return $;
    }

    static decodeText(/* reader, json */) {
        const $ = new tosa.v1.LogicalXorAttribute();
        return $;
    }
};

tosa.v1.MaximumAttribute = class MaximumAttribute {

    static decode(reader, position) {
        const $ = new tosa.v1.MaximumAttribute();
        $.nan_mode = reader.uint32_(position, 4, 0);
        return $;
    }

    static decodeText(reader, json) {
        const $ = new tosa.v1.MaximumAttribute();
        $.nan_mode = tosa.v1.NanPropagationMode[json.nan_mode];
        return $;
    }
};

tosa.v1.MinimumAttribute = class MinimumAttribute {

    static decode(reader, position) {
        const $ = new tosa.v1.MinimumAttribute();
        $.nan_mode = reader.uint32_(position, 4, 0);
        return $;
    }

    static decodeText(reader, json) {
        const $ = new tosa.v1.MinimumAttribute();
        $.nan_mode = tosa.v1.NanPropagationMode[json.nan_mode];
        return $;
    }
};

tosa.v1.MulAttribute = class MulAttribute {

    static decode(/* reader, position */) {
        const $ = new tosa.v1.MulAttribute();
        return $;
    }

    static decodeText(/* reader, json */) {
        const $ = new tosa.v1.MulAttribute();
        return $;
    }
};

tosa.v1.PowAttribute = class PowAttribute {

    static decode(/* reader, position */) {
        const $ = new tosa.v1.PowAttribute();
        return $;
    }

    static decodeText(/* reader, json */) {
        const $ = new tosa.v1.PowAttribute();
        return $;
    }
};

tosa.v1.SubAttribute = class SubAttribute {

    static decode(/* reader, position */) {
        const $ = new tosa.v1.SubAttribute();
        return $;
    }

    static decodeText(/* reader, json */) {
        const $ = new tosa.v1.SubAttribute();
        return $;
    }
};

tosa.v1.TableAttribute = class TableAttribute {

    static decode(/* reader, position */) {
        const $ = new tosa.v1.TableAttribute();
        return $;
    }

    static decodeText(/* reader, json */) {
        const $ = new tosa.v1.TableAttribute();
        return $;
    }
};

tosa.v1.AbsAttribute = class AbsAttribute {

    static decode(/* reader, position */) {
        const $ = new tosa.v1.AbsAttribute();
        return $;
    }

    static decodeText(/* reader, json */) {
        const $ = new tosa.v1.AbsAttribute();
        return $;
    }
};

tosa.v1.BitwiseNotAttribute = class BitwiseNotAttribute {

    static decode(/* reader, position */) {
        const $ = new tosa.v1.BitwiseNotAttribute();
        return $;
    }

    static decodeText(/* reader, json */) {
        const $ = new tosa.v1.BitwiseNotAttribute();
        return $;
    }
};

tosa.v1.CeilAttribute = class CeilAttribute {

    static decode(/* reader, position */) {
        const $ = new tosa.v1.CeilAttribute();
        return $;
    }

    static decodeText(/* reader, json */) {
        const $ = new tosa.v1.CeilAttribute();
        return $;
    }
};

tosa.v1.ClzAttribute = class ClzAttribute {

    static decode(/* reader, position */) {
        const $ = new tosa.v1.ClzAttribute();
        return $;
    }

    static decodeText(/* reader, json */) {
        const $ = new tosa.v1.ClzAttribute();
        return $;
    }
};

tosa.v1.CosAttribute = class CosAttribute {

    static decode(/* reader, position */) {
        const $ = new tosa.v1.CosAttribute();
        return $;
    }

    static decodeText(/* reader, json */) {
        const $ = new tosa.v1.CosAttribute();
        return $;
    }
};

tosa.v1.ExpAttribute = class ExpAttribute {

    static decode(/* reader, position */) {
        const $ = new tosa.v1.ExpAttribute();
        return $;
    }

    static decodeText(/* reader, json */) {
        const $ = new tosa.v1.ExpAttribute();
        return $;
    }
};

tosa.v1.FloorAttribute = class FloorAttribute {

    static decode(/* reader, position */) {
        const $ = new tosa.v1.FloorAttribute();
        return $;
    }

    static decodeText(/* reader, json */) {
        const $ = new tosa.v1.FloorAttribute();
        return $;
    }
};

tosa.v1.LogAttribute = class LogAttribute {

    static decode(/* reader, position */) {
        const $ = new tosa.v1.LogAttribute();
        return $;
    }

    static decodeText(/* reader, json */) {
        const $ = new tosa.v1.LogAttribute();
        return $;
    }
};

tosa.v1.LogicalNotAttribute = class LogicalNotAttribute {

    static decode(/* reader, position */) {
        const $ = new tosa.v1.LogicalNotAttribute();
        return $;
    }

    static decodeText(/* reader, json */) {
        const $ = new tosa.v1.LogicalNotAttribute();
        return $;
    }
};

tosa.v1.NegateAttribute = class NegateAttribute {

    static decode(/* reader, position */) {
        const $ = new tosa.v1.NegateAttribute();
        return $;
    }

    static decodeText(/* reader, json */) {
        const $ = new tosa.v1.NegateAttribute();
        return $;
    }
};

tosa.v1.ReciprocalAttribute = class ReciprocalAttribute {

    static decode(/* reader, position */) {
        const $ = new tosa.v1.ReciprocalAttribute();
        return $;
    }

    static decodeText(/* reader, json */) {
        const $ = new tosa.v1.ReciprocalAttribute();
        return $;
    }
};

tosa.v1.RsqrtAttribute = class RsqrtAttribute {

    static decode(/* reader, position */) {
        const $ = new tosa.v1.RsqrtAttribute();
        return $;
    }

    static decodeText(/* reader, json */) {
        const $ = new tosa.v1.RsqrtAttribute();
        return $;
    }
};

tosa.v1.SinAttribute = class SinAttribute {

    static decode(/* reader, position */) {
        const $ = new tosa.v1.SinAttribute();
        return $;
    }

    static decodeText(/* reader, json */) {
        const $ = new tosa.v1.SinAttribute();
        return $;
    }
};

tosa.v1.SelectAttribute = class SelectAttribute {

    static decode(/* reader, position */) {
        const $ = new tosa.v1.SelectAttribute();
        return $;
    }

    static decodeText(/* reader, json */) {
        const $ = new tosa.v1.SelectAttribute();
        return $;
    }
};

tosa.v1.EqualAttribute = class EqualAttribute {

    static decode(/* reader, position */) {
        const $ = new tosa.v1.EqualAttribute();
        return $;
    }

    static decodeText(/* reader, json */) {
        const $ = new tosa.v1.EqualAttribute();
        return $;
    }
};

tosa.v1.GreaterAttribute = class GreaterAttribute {

    static decode(/* reader, position */) {
        const $ = new tosa.v1.GreaterAttribute();
        return $;
    }

    static decodeText(/* reader, json */) {
        const $ = new tosa.v1.GreaterAttribute();
        return $;
    }
};

tosa.v1.GreaterEqualAttribute = class GreaterEqualAttribute {

    static decode(/* reader, position */) {
        const $ = new tosa.v1.GreaterEqualAttribute();
        return $;
    }

    static decodeText(/* reader, json */) {
        const $ = new tosa.v1.GreaterEqualAttribute();
        return $;
    }
};

tosa.v1.ReduceAllAttribute = class ReduceAllAttribute {

    static decode(reader, position) {
        const $ = new tosa.v1.ReduceAllAttribute();
        $.axis = reader.int32_(position, 4, 0);
        return $;
    }

    static decodeText(reader, json) {
        const $ = new tosa.v1.ReduceAllAttribute();
        $.axis = reader.value(json.axis, 0);
        return $;
    }
};

tosa.v1.ReduceAnyAttribute = class ReduceAnyAttribute {

    static decode(reader, position) {
        const $ = new tosa.v1.ReduceAnyAttribute();
        $.axis = reader.int32_(position, 4, 0);
        return $;
    }

    static decodeText(reader, json) {
        const $ = new tosa.v1.ReduceAnyAttribute();
        $.axis = reader.value(json.axis, 0);
        return $;
    }
};

tosa.v1.ReduceMaxAttribute = class ReduceMaxAttribute {

    static decode(reader, position) {
        const $ = new tosa.v1.ReduceMaxAttribute();
        $.axis = reader.int32_(position, 4, 0);
        $.nan_mode = reader.uint32_(position, 6, 0);
        return $;
    }

    static decodeText(reader, json) {
        const $ = new tosa.v1.ReduceMaxAttribute();
        $.axis = reader.value(json.axis, 0);
        $.nan_mode = tosa.v1.NanPropagationMode[json.nan_mode];
        return $;
    }
};

tosa.v1.ReduceMinAttribute = class ReduceMinAttribute {

    static decode(reader, position) {
        const $ = new tosa.v1.ReduceMinAttribute();
        $.axis = reader.int32_(position, 4, 0);
        $.nan_mode = reader.uint32_(position, 6, 0);
        return $;
    }

    static decodeText(reader, json) {
        const $ = new tosa.v1.ReduceMinAttribute();
        $.axis = reader.value(json.axis, 0);
        $.nan_mode = tosa.v1.NanPropagationMode[json.nan_mode];
        return $;
    }
};

tosa.v1.ReduceProductAttribute = class ReduceProductAttribute {

    static decode(reader, position) {
        const $ = new tosa.v1.ReduceProductAttribute();
        $.axis = reader.int32_(position, 4, 0);
        return $;
    }

    static decodeText(reader, json) {
        const $ = new tosa.v1.ReduceProductAttribute();
        $.axis = reader.value(json.axis, 0);
        return $;
    }
};

tosa.v1.ReduceSumAttribute = class ReduceSumAttribute {

    static decode(reader, position) {
        const $ = new tosa.v1.ReduceSumAttribute();
        $.axis = reader.int32_(position, 4, 0);
        return $;
    }

    static decodeText(reader, json) {
        const $ = new tosa.v1.ReduceSumAttribute();
        $.axis = reader.value(json.axis, 0);
        return $;
    }
};

tosa.v1.ConcatAttribute = class ConcatAttribute {

    static decode(reader, position) {
        const $ = new tosa.v1.ConcatAttribute();
        $.axis = reader.int32_(position, 4, 0);
        return $;
    }

    static decodeText(reader, json) {
        const $ = new tosa.v1.ConcatAttribute();
        $.axis = reader.value(json.axis, 0);
        return $;
    }
};

tosa.v1.PadAttribute = class PadAttribute {

    static decode(/* reader, position */) {
        const $ = new tosa.v1.PadAttribute();
        return $;
    }

    static decodeText(/* reader, json */) {
        const $ = new tosa.v1.PadAttribute();
        return $;
    }
};

tosa.v1.ReshapeAttribute = class ReshapeAttribute {

    static decode(/* reader, position */) {
        const $ = new tosa.v1.ReshapeAttribute();
        return $;
    }

    static decodeText(/* reader, json */) {
        const $ = new tosa.v1.ReshapeAttribute();
        return $;
    }
};

tosa.v1.ReverseAttribute = class ReverseAttribute {

    static decode(reader, position) {
        const $ = new tosa.v1.ReverseAttribute();
        $.axis = reader.int32_(position, 4, 0);
        return $;
    }

    static decodeText(reader, json) {
        const $ = new tosa.v1.ReverseAttribute();
        $.axis = reader.value(json.axis, 0);
        return $;
    }
};

tosa.v1.SliceAttribute = class SliceAttribute {

    static decode(/* reader, position */) {
        const $ = new tosa.v1.SliceAttribute();
        return $;
    }

    static decodeText(/* reader, json */) {
        const $ = new tosa.v1.SliceAttribute();
        return $;
    }
};

tosa.v1.TileAttribute = class TileAttribute {

    static decode(/* reader, position */) {
        const $ = new tosa.v1.TileAttribute();
        return $;
    }

    static decodeText(/* reader, json */) {
        const $ = new tosa.v1.TileAttribute();
        return $;
    }
};

tosa.v1.TransposeAttribute = class TransposeAttribute {

    static decode(reader, position) {
        const $ = new tosa.v1.TransposeAttribute();
        $.perms = reader.array(position, 4, Int32Array);
        return $;
    }

    static decodeText(reader, json) {
        const $ = new tosa.v1.TransposeAttribute();
        $.perms = reader.array(json.perms, Int32Array);
        return $;
    }
};

tosa.v1.GatherAttribute = class GatherAttribute {

    static decode(/* reader, position */) {
        const $ = new tosa.v1.GatherAttribute();
        return $;
    }

    static decodeText(/* reader, json */) {
        const $ = new tosa.v1.GatherAttribute();
        return $;
    }
};

tosa.v1.ScatterAttribute = class ScatterAttribute {

    static decode(/* reader, position */) {
        const $ = new tosa.v1.ScatterAttribute();
        return $;
    }

    static decodeText(/* reader, json */) {
        const $ = new tosa.v1.ScatterAttribute();
        return $;
    }
};

tosa.v1.ResizeAttribute = class ResizeAttribute {

    static decode(reader, position) {
        const $ = new tosa.v1.ResizeAttribute();
        $.mode = reader.uint32_(position, 4, 0);
        return $;
    }

    static decodeText(reader, json) {
        const $ = new tosa.v1.ResizeAttribute();
        $.mode = tosa.v1.ResizeMode[json.mode];
        return $;
    }
};

tosa.v1.CastAttribute = class CastAttribute {

    static decode(/* reader, position */) {
        const $ = new tosa.v1.CastAttribute();
        return $;
    }

    static decodeText(/* reader, json */) {
        const $ = new tosa.v1.CastAttribute();
        return $;
    }
};

tosa.v1.RescaleAttribute = class RescaleAttribute {

    static decode(reader, position) {
        const $ = new tosa.v1.RescaleAttribute();
        $.scale32 = reader.bool_(position, 4, false);
        $.rounding_mode = reader.uint32_(position, 6, 0);
        $.per_channel = reader.bool_(position, 8, false);
        $.input_unsigned = reader.bool_(position, 10, false);
        $.output_unsigned = reader.bool_(position, 12, false);
        return $;
    }

    static decodeText(reader, json) {
        const $ = new tosa.v1.RescaleAttribute();
        $.scale32 = reader.value(json.scale32, false);
        $.rounding_mode = tosa.v1.RoundingMode[json.rounding_mode];
        $.per_channel = reader.value(json.per_channel, false);
        $.input_unsigned = reader.value(json.input_unsigned, false);
        $.output_unsigned = reader.value(json.output_unsigned, false);
        return $;
    }
};

tosa.v1.ConstAttribute = class ConstAttribute {

    static decode(/* reader, position */) {
        const $ = new tosa.v1.ConstAttribute();
        return $;
    }

    static decodeText(/* reader, json */) {
        const $ = new tosa.v1.ConstAttribute();
        return $;
    }
};

tosa.v1.IdentityAttribute = class IdentityAttribute {

    static decode(/* reader, position */) {
        const $ = new tosa.v1.IdentityAttribute();
        return $;
    }

    static decodeText(/* reader, json */) {
        const $ = new tosa.v1.IdentityAttribute();
        return $;
    }
};

tosa.v1.CustomAttribute = class CustomAttribute {

    static decode(reader, position) {
        const $ = new tosa.v1.CustomAttribute();
        $.operator_name = reader.string_(position, 4, null);
        $.domain_name = reader.string_(position, 6, null);
        $.implementation_attrs = reader.array(position, 8, Uint8Array);
        return $;
    }

    static decodeText(reader, json) {
        const $ = new tosa.v1.CustomAttribute();
        $.operator_name = reader.value(json.operator_name, null);
        $.domain_name = reader.value(json.domain_name, null);
        $.implementation_attrs = reader.array(json.implementation_attrs, Uint8Array);
        return $;
    }
};

tosa.v1.CondIfAttribute = class CondIfAttribute {

    static decode(reader, position) {
        const $ = new tosa.v1.CondIfAttribute();
        $.then_graph = reader.string_(position, 4, null);
        $.else_graph = reader.string_(position, 6, null);
        return $;
    }

    static decodeText(reader, json) {
        const $ = new tosa.v1.CondIfAttribute();
        $.then_graph = reader.value(json.then_graph, null);
        $.else_graph = reader.value(json.else_graph, null);
        return $;
    }
};

tosa.v1.WhileLoopAttribute = class WhileLoopAttribute {

    static decode(reader, position) {
        const $ = new tosa.v1.WhileLoopAttribute();
        $.cond_graph = reader.string_(position, 4, null);
        $.body_graph = reader.string_(position, 6, null);
        return $;
    }

    static decodeText(reader, json) {
        const $ = new tosa.v1.WhileLoopAttribute();
        $.cond_graph = reader.value(json.cond_graph, null);
        $.body_graph = reader.value(json.body_graph, null);
        return $;
    }
};

tosa.v1.VariableAttribute = class VariableAttribute {

    static decode(/* reader, position */) {
        const $ = new tosa.v1.VariableAttribute();
        return $;
    }

    static decodeText(/* reader, json */) {
        const $ = new tosa.v1.VariableAttribute();
        return $;
    }
};

tosa.v1.VariableWriteAttribute = class VariableWriteAttribute {

    static decode(/* reader, position */) {
        const $ = new tosa.v1.VariableWriteAttribute();
        return $;
    }

    static decodeText(/* reader, json */) {
        const $ = new tosa.v1.VariableWriteAttribute();
        return $;
    }
};

tosa.v1.VariableReadAttribute = class VariableReadAttribute {

    static decode(/* reader, position */) {
        const $ = new tosa.v1.VariableReadAttribute();
        return $;
    }

    static decodeText(/* reader, json */) {
        const $ = new tosa.v1.VariableReadAttribute();
        return $;
    }
};

tosa.v1.ConstShapeAttribute = class ConstShapeAttribute {

    static decode(/* reader, position */) {
        const $ = new tosa.v1.ConstShapeAttribute();
        return $;
    }

    static decodeText(/* reader, json */) {
        const $ = new tosa.v1.ConstShapeAttribute();
        return $;
    }
};

tosa.v1.Version = class Version {

    static decode(reader, position) {
        const $ = new tosa.v1.Version();
        $._major = reader.int32_(position, 4, -1);
        $._minor = reader.int32_(position, 6, -1);
        $._patch = reader.int32_(position, 8, -1);
        $._draft = reader.bool_(position, 10, true);
        return $;
    }

    static decodeText(reader, json) {
        const $ = new tosa.v1.Version();
        $._major = reader.value(json._major, -1);
        $._minor = reader.value(json._minor, -1);
        $._patch = reader.value(json._patch, -1);
        $._draft = reader.value(json._draft, true);
        return $;
    }
};

tosa.v1.TosaTensor = class TosaTensor {

    static decode(reader, position) {
        const $ = new tosa.v1.TosaTensor();
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
        const $ = new tosa.v1.TosaTensor();
        $.name = reader.value(json.name, null);
        $.shape = reader.array(json.shape, Int32Array);
        $.type = tosa.v1.DType[json.type];
        $.data = reader.array(json.data, Uint8Array);
        $.variable = reader.value(json.variable, false);
        $.is_unranked = reader.value(json.is_unranked, false);
        $.variable_name = reader.value(json.variable_name, null);
        return $;
    }
};

tosa.v1.TosaShape = class TosaShape {

    static decode(reader, position) {
        const $ = new tosa.v1.TosaShape();
        $.name = reader.string_(position, 4, null);
        $.rank = reader.uint32_(position, 6, 0);
        $.data = reader.array(position, 8, Uint8Array);
        return $;
    }

    static decodeText(reader, json) {
        const $ = new tosa.v1.TosaShape();
        $.name = reader.value(json.name, null);
        $.rank = reader.value(json.rank, 0);
        $.data = reader.array(json.data, Uint8Array);
        return $;
    }
};

tosa.v1.OpLocation = class OpLocation {

    static decode(reader, position) {
        const $ = new tosa.v1.OpLocation();
        $.text = reader.string_(position, 4, null);
        return $;
    }

    static decodeText(reader, json) {
        const $ = new tosa.v1.OpLocation();
        $.text = reader.value(json.text, null);
        return $;
    }
};

tosa.v1.TosaOperator = class TosaOperator {

    static decode(reader, position) {
        const $ = new tosa.v1.TosaOperator();
        $.op = reader.uint32_(position, 4, 0);
        $.attribute = reader.union(position, 6, tosa.v1.Attribute);
        $.inputs = reader.strings_(position, 10);
        $.outputs = reader.strings_(position, 12);
        $.location = reader.table(position, 14, tosa.v1.OpLocation);
        return $;
    }

    static decodeText(reader, json) {
        const $ = new tosa.v1.TosaOperator();
        $.op = tosa.v1.Op[json.op];
        $.attribute = tosa.v1.Attribute.decodeText(reader, json.attribute, json.attribute_type);
        $.inputs = reader.array(json.inputs);
        $.outputs = reader.array(json.outputs);
        $.location = reader.object(json.location, tosa.v1.OpLocation);
        return $;
    }
};

tosa.v1.TosaBasicBlock = class TosaBasicBlock {

    static decode(reader, position) {
        const $ = new tosa.v1.TosaBasicBlock();
        $.name = reader.string_(position, 4, null);
        $.operators = reader.tables(position, 6, tosa.v1.TosaOperator);
        $.tensors = reader.tables(position, 8, tosa.v1.TosaTensor);
        $.inputs = reader.strings_(position, 10);
        $.outputs = reader.strings_(position, 12);
        $.shapes = reader.tables(position, 14, tosa.v1.TosaShape);
        return $;
    }

    static decodeText(reader, json) {
        const $ = new tosa.v1.TosaBasicBlock();
        $.name = reader.value(json.name, null);
        $.operators = reader.objects(json.operators, tosa.v1.TosaOperator);
        $.tensors = reader.objects(json.tensors, tosa.v1.TosaTensor);
        $.inputs = reader.array(json.inputs);
        $.outputs = reader.array(json.outputs);
        $.shapes = reader.objects(json.shapes, tosa.v1.TosaShape);
        return $;
    }
};

tosa.v1.TosaRegion = class TosaRegion {

    static decode(reader, position) {
        const $ = new tosa.v1.TosaRegion();
        $.name = reader.string_(position, 4, null);
        $.blocks = reader.tables(position, 6, tosa.v1.TosaBasicBlock);
        return $;
    }

    static decodeText(reader, json) {
        const $ = new tosa.v1.TosaRegion();
        $.name = reader.value(json.name, null);
        $.blocks = reader.objects(json.blocks, tosa.v1.TosaBasicBlock);
        return $;
    }
};

tosa.v1.TosaGraph = class TosaGraph {

    static identifier(reader) {
        return reader.identifier === 'TOSA';
    }

    static create(reader) {
        return tosa.v1.TosaGraph.decode(reader, reader.root);
    }

    static createText(reader) {
        return tosa.v1.TosaGraph.decodeText(reader, reader.root);
    }

    static decode(reader, position) {
        const $ = new tosa.v1.TosaGraph();
        $.version = reader.table(position, 4, tosa.v1.Version);
        $.regions = reader.tables(position, 6, tosa.v1.TosaRegion);
        return $;
    }

    static decodeText(reader, json) {
        const $ = new tosa.v1.TosaGraph();
        $.version = reader.object(json.version, tosa.v1.Version);
        $.regions = reader.objects(json.regions, tosa.v1.TosaRegion);
        return $;
    }
};
